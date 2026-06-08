"""
Wav2Vec2 Fine-tuning - Mental Health Audio Classification
NORMAL vs DEPRESI | 5-Fold StratifiedKFold CV

📝 DEEP LEARNING PIPELINE NOTE: WAV2VEC2 TRAINING & PERFORMANCE
- Role: Fine-tunes a pretrained self-supervised speech model for depression classification.
- Architecture:
  1. Backbone: facebook/wav2vec2-base.
  2. Freeze Strategy: CNN feature extractor layers are FROZEN to preserve speech representations.
  3. Fine-tuning: Transformer Encoder layers are trainable to learn temporal patterns.
  4. Pooling: Mean pooling over timesteps.
  5. Classifier Head: Dropout(0.5) → Linear(768→384) → BN → GELU → Dropout(0.3) → Linear(384→2).
- Anti-Leakage Split: Patient-level 5-Fold Stratified CV.
- Imbalance Handled: WeightedRandomSampler + class weights in CrossEntropyLoss.

- Architecture Changes (v3 — stabilized retrain):
  - LR_HEAD 1e-4 → 2e-5: cegah collapse di epoch 1 (root cause v2 instability)
  - Warmup 3 → 8 epoch: fine-tuning model 94M param butuh warmup lebih panjang
  - EPOCHS 25 → 40: LR lebih kecil butuh lebih banyak iterasi
  - EARLY_STOP 10 → 15: sinkron dengan LR lebih kecil
  - label_smoothing 0.0 → 0.05: tambah regularisasi ringan
  - CosineAnnealing T_max disesuaikan ke EPOCHS-warmup_epochs

- Final Test Performance (v2 — tidak stabil):
  - Test Set Accuracy: 42%
  - Test Set Macro F1: 0.3942
  - CV Mean Macro F1: 0.3517 ± 0.1460 (fold 1/2/4 collapse ke semua DEPRESI)

Strategi:
- Baca raw waveform dari data/cleaned/{pid}.wav
- Freeze CNN feature extractor Wav2Vec2 (lapisan bawah)
- Fine-tune Transformer layers + tambah classifier head
- 5-Fold CV → Mean ± Std Macro F1
- Best fold dievaluasi di test set
- Handle imbalance: WeightedRandomSampler + class weights
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import sys

# ── MLflow Tracking ───────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))
try:
    from experiments.mlflow_tracking import MLflowTracker, print_ui_instructions
except ImportError:
    class MLflowTracker:  # type: ignore
        def __init__(self, *a, **kw): pass
        def start_run(self, *a, **kw): pass
        def log_epoch(self, *a, **kw): pass
        def log_fold_summary(self, *a, **kw): pass
        def log_cv_summary(self, *a, **kw): pass
        def log_artifacts(self, *a, **kw): pass
        def end_run(self): pass
    def print_ui_instructions(): pass

from dataloader_wav2vec import (
    collect_all_files, get_cv_splits, make_loader,
    Wav2VecDataset, CLASS_NAMES,
)

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
MODEL_DIR    = PROJECT_ROOT / "models" / "dl" / "wav2vec"

# Output V3 ke subfolder tersendiri agar tidak overwrite V2
METRICS_DIR  = RESULTS_DIR / "metrics" / "WAV2VEC" / "WAV2VEC_v3"
PLOTS_DIR    = RESULTS_DIR / "plots" / "WAV2VEC" / "WAV2VEC_v3"
CM_DIR       = RESULTS_DIR / "confusion_matrix" / "WAV2VEC" / "WAV2VEC_v3"

for d in [METRICS_DIR, PLOTS_DIR, CM_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "train_wav2vec.log"),
        logging.StreamHandler(),
    ],
)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
PRETRAINED    = "facebook/wav2vec2-base"
N_FOLDS       = int(os.environ.get("N_FOLDS", 5))
EPOCHS        = int(os.environ.get("EPOCHS", 40))    # naik 25→40: LR lebih kecil butuh lebih banyak iterasi
BATCH_SIZE    = 4                                     # kecil karena model besar
LR_BACKBONE   = 5e-6                                  # SANGAT KECIL untuk pretrained layers (pertahankan)
LR_HEAD       = 2e-5                                  # turun 1e-4→2e-5: cegah overshoot/collapse di epoch 1
WEIGHT_DECAY  = 1e-4
MAX_GRAD_NORM = 1.0
SAMPLE_RATE   = 16000
MAX_DURATION  = 30.0                                  # detik
TEST_SIZE     = 0.10
RANDOM_STATE  = 42
EARLY_STOP    = 15                                    # naik 10→15: sinkron dengan LR lebih kecil
FREEZE_LAYERS = 6                                     # freeze first 6 transformer layers
WARMUP_EPOCHS = 8                                     # naik 3→8: model 94M param butuh warmup lebih panjang
LABEL_SMOOTH  = 0.05                                  # tambah regularisasi ringan


# ── Model ─────────────────────────────────────────────────────────────────────
class Wav2Vec2Classifier(nn.Module):
    """
    Wav2Vec2 + mean pooling + classifier head.

    Strategi freeze OPTIMAL (tuned):
    - CNN feature extractor: FROZEN — preserve speech representations
    - Transformer layers 0-5: FROZEN — preserve low-level acoustic patterns
    - Transformer layers 6-11: TRAINABLE — fine-tune high-level representations
    - Classifier head: TRAINABLE dengan dropout lebih tinggi
    """

    def __init__(self, pretrained_name: str, num_classes: int = 2, dropout: float = 0.5, freeze_layers: int = 6, **kwargs):
        super().__init__()
        attn_impl = kwargs.get("attn_implementation", "eager")
        # Coba load dengan attn_implementation, fallback ke tanpa, lalu fallback ke local_files_only
        def _load(name, **kw):
            try:
                return Wav2Vec2Model.from_pretrained(name, attn_implementation=attn_impl, **kw)
            except TypeError:
                return Wav2Vec2Model.from_pretrained(name, **kw)

        try:
            self.wav2vec2 = _load(pretrained_name)
        except OSError:
            # Tidak ada koneksi internet atau model tidak tersedia online → pakai cache lokal
            import logging as _log
            _log.getLogger(__name__).warning(
                f"Tidak bisa download '{pretrained_name}' dari HuggingFace. "
                "Menggunakan cache lokal (local_files_only=True)."
            )
            self.wav2vec2 = _load(pretrained_name, local_files_only=True)
        
        hidden_size     = self.wav2vec2.config.hidden_size
        
        # Classifier head lebih robust: dropout → BatchNorm → Linear → Dropout → Linear
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),  # 0.3 jika dropout=0.5
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Freeze CNN feature extractor
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Freeze first N transformer layers untuk stabilitas
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        n_layers = len(self.wav2vec2.encoder.layers)
        logging.info(
            f"Wav2Vec2 freeze strategy: "
            f"feature_extractor=FROZEN | "
            f"transformer[0:{freeze_layers}]=FROZEN | "
            f"transformer[{freeze_layers}:{n_layers}]=TRAINABLE | "
            f"classifier=TRAINABLE (dropout={dropout})"
        )

    def forward(self, input_values, attention_mask=None):
        out          = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden       = out.last_hidden_state   # (B, T, H)
        pooled       = hidden.mean(dim=1)      # mean pooling → (B, H)
        return self.classifier(pooled)         # (B, num_classes)


# ── Collate function ──────────────────────────────────────────────────────────
def collate_fn(batch):
    """Pad waveforms ke panjang yang sama dalam satu batch."""
    waveforms = [b["waveform"] for b in batch]
    labels    = torch.stack([b["label"] for b in batch])

    # Pad ke panjang terpanjang dalam batch
    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.zeros(len(waveforms), max_len)
    mask    = torch.zeros(len(waveforms), max_len, dtype=torch.long)

    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
        mask[i, :w.shape[0]]   = 1

    return {"input_values": padded, "attention_mask": mask, "labels": labels}


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_fold_history(history, fold_num):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(ep, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="tomato")
    axes[0].set_title(f"Loss — Fold {fold_num}"); axes[0].legend(); axes[0].grid(alpha=0.4)

    axes[1].plot(ep, history["train_f1"], label="Train F1", color="steelblue")
    axes[1].plot(ep, history["val_f1"],   label="Val F1",   color="tomato")
    axes[1].set_title(f"Macro F1 — Fold {fold_num}")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(alpha=0.4)

    axes[2].plot(ep, history["val_recall_normal"],  label="Recall NORMAL",  color="steelblue", marker="o", markersize=3)
    axes[2].plot(ep, history["val_recall_depresi"], label="Recall DEPRESI", color="tomato",    marker="s", markersize=3)
    axes[2].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_title(f"Recall per Kelas — Fold {fold_num}")
    axes[2].set_ylim(0, 1.05); axes[2].legend(); axes[2].grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"wav2vec_fold_{fold_num}_curves.png", dpi=150)
    plt.close()


def plot_cv_summary(scores):
    mean_f1, std_f1 = np.mean(scores), np.std(scores)
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, scores, color="steelblue", alpha=0.8, edgecolor="black")
    plt.axhline(mean_f1, color="tomato", linestyle="--", lw=2,
                label=f"Mean={mean_f1:.4f} ± {std_f1:.4f}")
    plt.fill_between(range(len(folds)), mean_f1-std_f1, mean_f1+std_f1,
                     alpha=0.15, color="tomato")
    for bar, val in zip(bars, scores):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{val:.4f}", ha="center", fontsize=10)
    plt.title("Wav2Vec2 — 5-Fold CV Macro F1")
    plt.ylabel("Macro F1"); plt.ylim(0, 1.05)
    plt.legend(); plt.grid(alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wav2vec_cv_summary.png", dpi=150)
    plt.close()


def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CM_DIR / filename, dpi=150)
    plt.close()


# ── Single fold training ──────────────────────────────────────────────────
def train_one_fold(train_samples, val_samples, fold_num, device,
                   tracker: "MLflowTracker" = None):
    train_loader = make_loader(
        train_samples, BATCH_SIZE, shuffle=True, weighted=True,
        sample_rate=SAMPLE_RATE, max_duration_sec=MAX_DURATION, num_workers=0,
    )
    val_loader = make_loader(
        val_samples, BATCH_SIZE, shuffle=False,
        sample_rate=SAMPLE_RATE, max_duration_sec=MAX_DURATION, num_workers=0,
    )

    # Override collate_fn untuk padding
    train_loader = DataLoader(
        train_loader.dataset, batch_size=BATCH_SIZE,
        sampler=train_loader.sampler if hasattr(train_loader, 'sampler') else None,
        shuffle=False if hasattr(train_loader, 'sampler') else True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_loader.dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = Wav2Vec2Classifier(PRETRAINED, num_classes=2, dropout=0.5, freeze_layers=FREEZE_LAYERS).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    logging.info(
        f"Fold {fold_num} | Total: {total_params:,} | "
        f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%) | "
        f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)"
    )

    # Class weights
    labels_t = torch.tensor([s[1] for s in train_samples], dtype=torch.long)
    counts   = torch.bincount(labels_t, minlength=2).float().clamp(min=1)
    cw       = (1.0 / counts); cw = (cw / cw.sum() * 2).to(device)
    logging.info(f"Class weights: NORMAL={cw[0]:.3f} | DEPRESI={cw[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH)
    
    # Differential learning rate: backbone vs head (TUNED!)
    backbone_params = [p for n, p in model.named_parameters() 
                      if 'wav2vec2' in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() 
                      if 'classifier' in n and p.requires_grad]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': WEIGHT_DECAY},
        {'params': head_params,     'lr': LR_HEAD,     'weight_decay': WEIGHT_DECAY * 2},  # head diregularisasi lebih
    ])
    
    logging.info(
        f"Optimizer: backbone_lr={LR_BACKBONE:.2e} ({len(backbone_params)} groups) | "
        f"head_lr={LR_HEAD:.2e} ({len(head_params)} groups)"
    )
    
    # Cosine annealing with warmup — warmup lebih panjang untuk stabilitas
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=1e-7
    )

    history = {k: [] for k in [
        "train_loss", "val_loss", "train_f1", "val_f1",
        "val_recall_normal", "val_recall_depresi",
    ]}

    best_f1, best_epoch, best_state = -1.0, 0, None
    best_preds, best_lbls, no_improve, last_lr = [], [], 0, LR_BACKBONE

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        t_loss, all_p, all_l = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Fold{fold_num} Ep{epoch+1:2d} Train", leave=False):
            iv   = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            optimizer.zero_grad()
            out  = model(iv, mask)
            loss = criterion(out, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            t_loss += loss.item()
            _, pred = torch.max(out, 1)
            all_p.extend(pred.cpu().numpy()); all_l.extend(lbls.cpu().numpy())

        e_tl = t_loss / max(1, len(train_loader))
        e_tf = f1_score(all_l, all_p, average="macro", zero_division=0)

        # ── Val ──
        model.eval()
        v_loss, all_p, all_l = 0.0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold{fold_num} Ep{epoch+1:2d} Val  ", leave=False):
                iv   = batch["input_values"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["labels"].to(device)
                out  = model(iv, mask)
                loss = criterion(out, lbls)
                v_loss += loss.item()
                _, pred = torch.max(out, 1)
                all_p.extend(pred.cpu().numpy()); all_l.extend(lbls.cpu().numpy())

        e_vl  = v_loss / max(1, len(val_loader))
        e_vf  = f1_score(all_l, all_p, average="macro", zero_division=0)
        _, rec, _, _ = precision_recall_fscore_support(all_l, all_p, labels=[0,1], zero_division=0)

        history["train_loss"].append(e_tl)
        history["val_loss"].append(e_vl)
        history["train_f1"].append(e_tf)
        history["val_f1"].append(e_vf)
        history["val_recall_normal"].append(rec[0])
        history["val_recall_depresi"].append(rec[1])

        pred_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter(all_p).items())}
        logging.info(
            f"[Fold{fold_num}] Ep{epoch+1:2d} | "
            f"TL:{e_tl:.4f} TF1:{e_tf:.4f} | "
            f"VL:{e_vl:.4f} VF1:{e_vf:.4f} | "
            f"Rec N:{rec[0]:.3f} D:{rec[1]:.3f} | Dist:{pred_dist}"
        )

        # ── MLflow: log metrics per epoch ─────────────────────────────
        if tracker:
            tracker.log_epoch(epoch + 1, {
                "train/loss"         : e_tl,
                "train/macro_f1"     : e_tf,
                "val/loss"           : e_vl,
                "val/macro_f1"       : e_vf,
                "val/recall_normal"  : float(rec[0]),
                "val/recall_depresi" : float(rec[1]),
                "lr"                 : optimizer.param_groups[0]["lr"],
            })

        if e_vf > best_f1:
            best_f1 = e_vf; best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = list(all_p); best_lbls = list(all_l)
            no_improve = 0
            logging.info(f"  ✅ Best F1: {best_f1:.4f} @ epoch {best_epoch}")
        else:
            no_improve += 1

        # Scheduler step
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        new_lr_backbone = optimizer.param_groups[0]["lr"]
        new_lr_head     = optimizer.param_groups[1]["lr"]
        if epoch == 0 or (epoch < WARMUP_EPOCHS) or (new_lr_backbone != last_lr):
            logging.info(f"  📊 LR: backbone={new_lr_backbone:.2e} | head={new_lr_head:.2e}")
        last_lr = new_lr_backbone

        if no_improve >= EARLY_STOP:
            logging.info(f"  Early stopping @ epoch {epoch+1}")
            break

    return best_f1, best_epoch, best_state, best_preds, best_lbls, history


# ── Main: 5-Fold CV ───────────────────────────────────────────────────────────
def run_cv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Pretrained model: {PRETRAINED}")

    # ── MLflow Tracker ────────────────────────────────────────────────────
    tracker = MLflowTracker(experiment_name="Wav2Vec2-FineTuned")
    print_ui_instructions()

    HPARAMS = {
        "model"         : "Wav2Vec2-FineTuned-v3",
        "pretrained"    : PRETRAINED,
        "n_folds"       : N_FOLDS,
        "epochs"        : EPOCHS,
        "batch_size"    : BATCH_SIZE,
        "lr_backbone"   : LR_BACKBONE,
        "lr_head"       : LR_HEAD,
        "weight_decay"  : WEIGHT_DECAY,
        "freeze_layers" : FREEZE_LAYERS,
        "warmup_epochs" : WARMUP_EPOCHS,
        "label_smooth"  : LABEL_SMOOTH,
        "dropout"       : 0.5,
        "max_duration"  : MAX_DURATION,
        "sample_rate"   : SAMPLE_RATE,
        "early_stop"    : EARLY_STOP,
        "random_state"  : RANDOM_STATE,
        "device"        : str(device),
        "scheduler"     : "cosine_annealing_warmup",
    }

    patients_list, patients_labels, all_files = collect_all_files()
    test_samples, fold_splits = get_cv_splits(
        patients_list, patients_labels, all_files,
        n_folds=N_FOLDS, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    fold_f1_scores, fold_results = [], []
    fold_metrics = []   # dict metrik lengkap per fold untuk tabel perbandingan

    for fold, (train_s, val_s) in enumerate(fold_splits, start=1):
        logging.info(f"\n{'='*60}")
        logging.info(f"FOLD {fold}/{N_FOLDS}")
        logging.info(f"{'='*60}")

        tr_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter([s[1] for s in train_s]).items())}
        vl_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter([s[1] for s in val_s]).items())}
        logging.info(f"Train: {tr_dist} | Val: {vl_dist}")

        # ── MLflow: start run untuk fold ini ─────────────────────────────
        tracker.start_run(
            run_name=f"fold_{fold}",
            params={**HPARAMS,
                    "fold"        : fold,
                    "train_files" : len(train_s),
                    "val_files"   : len(val_s)},
        )

        result = train_one_fold(train_s, val_s, fold, device, tracker=tracker)
        best_f1, best_epoch, best_state, best_preds, best_lbls, history = result

        fold_f1_scores.append(best_f1)
        fold_results.append(result)
        logging.info(f"Fold {fold} selesai — Best F1: {best_f1:.4f} @ epoch {best_epoch}")

        plot_fold_history(history, fold)
        cm_path     = CM_DIR / f"wav2vec_cm_fold_{fold}.png"
        curves_path = PLOTS_DIR / f"wav2vec_fold_{fold}_curves.png"
        plot_cm(best_lbls, best_preds,
                title=f"Confusion Matrix — Fold {fold} (F1={best_f1:.4f})",
                filename=f"wav2vec_cm_fold_{fold}.png")

        report_path = METRICS_DIR / f"wav2vec_report_fold_{fold}.txt"
        report = classification_report(best_lbls, best_preds,
                                       target_names=CLASS_NAMES, labels=[0,1], zero_division=0)
        with open(report_path, "w") as f:
            f.write(f"=== Fold {fold} | Best Epoch {best_epoch} | Macro F1: {best_f1:.4f} ===\n\n")
            f.write(report)
        print(report)

        # Kumpulkan metrik lengkap per fold untuk tabel perbandingan
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            best_lbls, best_preds, labels=[0, 1], zero_division=0
        )
        fold_acc = sum(p == l for p, l in zip(best_preds, best_lbls)) / max(1, len(best_lbls))
        fold_metrics.append({
            "Fold"           : fold,
            "Best Epoch"     : best_epoch,
            "Val Accuracy"   : round(fold_acc, 4),
            "Macro F1"       : round(best_f1, 4),
            "Prec NORMAL"    : round(float(prec_arr[0]), 4),
            "Prec DEPRESI"   : round(float(prec_arr[1]), 4),
            "Recall NORMAL"  : round(float(rec_arr[0]), 4),
            "Recall DEPRESI" : round(float(rec_arr[1]), 4),
            "F1 NORMAL"      : round(float(f1_arr[0]), 4),
            "F1 DEPRESI"     : round(float(f1_arr[1]), 4),
        })

        # ── MLflow: log fold summary + artifacts ────────────────────────
        tracker.log_fold_summary(fold=fold, best_f1=best_f1, best_epoch=best_epoch)
        tracker.log_artifacts([cm_path, curves_path, report_path])
        tracker.end_run()

    # ── Ringkasan CV ──────────────────────────────────────────────────────
    mean_f1, std_f1 = np.mean(fold_f1_scores), np.std(fold_f1_scores)
    logging.info(f"\n{'='*60}")
    logging.info(f"5-FOLD CV SELESAI")
    logging.info(f"F1 per fold : {[f'{v:.4f}' for v in fold_f1_scores]}")
    logging.info(f"Mean F1     : {mean_f1:.4f} ± {std_f1:.4f}")
    logging.info(f"{'='*60}")

    # ── Tabel perbandingan per fold ───────────────────────────────────────
    df_fold_compare = pd.DataFrame(fold_metrics)
    mean_row = {"Fold": "Mean", "Best Epoch": "-"}
    std_row  = {"Fold": "Std",  "Best Epoch": "-"}
    for col in df_fold_compare.columns[2:]:
        mean_row[col] = round(float(df_fold_compare[col].mean()), 4)
        std_row[col]  = round(float(df_fold_compare[col].std()),  4)
    df_fold_compare = pd.concat(
        [df_fold_compare, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )

    logging.info(f"\n{'='*80}")
    logging.info("PERBANDINGAN METRIK PER FOLD — Wav2Vec2")
    logging.info(f"{'='*80}")
    logging.info("\n" + df_fold_compare.to_string(index=False))
    logging.info(f"{'='*80}\n")

    fold_compare_path = METRICS_DIR / "wav2vec_fold_comparison.csv"
    df_fold_compare.to_csv(fold_compare_path, index=False)
    logging.info(f"Tabel perbandingan per fold disimpan → {fold_compare_path}")

    plot_cv_summary(fold_f1_scores)
    cv_summary_path = PLOTS_DIR / "wav2vec_cv_summary.png"

    with open(METRICS_DIR / "wav2vec_cv_summary.txt", "w") as f:
        f.write("=== Wav2Vec2 5-Fold CV Summary ===\n\n")
        for i, v in enumerate(fold_f1_scores, 1):
            f.write(f"Fold {i}: Macro F1 = {v:.4f}\n")
        f.write(f"\nMean : {mean_f1:.4f}\nStd  : {std_f1:.4f}\n")
        f.write(f"\n{'='*60}\n")
        f.write("Detail per Fold:\n\n")
        f.write(df_fold_compare.to_string(index=False))
        f.write("\n")

    # ── Evaluasi best fold di test set ───────────────────────────────────
    best_fold_idx = int(np.argmax(fold_f1_scores))
    best_fold_num = best_fold_idx + 1
    _, _, best_state, _, _, _ = fold_results[best_fold_idx]

    logging.info(f"\nEvaluasi best fold ({best_fold_num}) di test set ({len(test_samples)} samples)...")
    model = Wav2Vec2Classifier(PRETRAINED, num_classes=2, dropout=0.5, freeze_layers=FREEZE_LAYERS).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    test_loader = DataLoader(
        Wav2VecDataset(test_samples, sample_rate=SAMPLE_RATE, max_duration_sec=MAX_DURATION),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    test_preds, test_lbls = [], []
    with torch.no_grad():
        for batch in test_loader:
            iv   = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            _, pred = torch.max(model(iv, mask), 1)
            test_preds.extend(pred.cpu().numpy())
            test_lbls.extend(batch["labels"].numpy())

    test_f1 = f1_score(test_lbls, test_preds, average="macro", zero_division=0)
    logging.info(f"Test Set Macro F1: {test_f1:.4f}")

    plot_cm(test_lbls, test_preds,
            title=f"Confusion Matrix — Test Set (F1={test_f1:.4f})",
            filename="wav2vec_cm_test_set.png")

    test_report = classification_report(test_lbls, test_preds,
                                        target_names=CLASS_NAMES, labels=[0,1], zero_division=0)
    print("\n=== TEST SET REPORT ===")
    print(test_report)

    test_report_path = METRICS_DIR / "wav2vec_report_test_set.txt"
    with open(test_report_path, "w") as f:
        f.write(f"=== Wav2Vec2 Test Set (Best Fold: {best_fold_num}) ===\n\n")
        f.write(f"CV Mean F1 : {mean_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"Test F1    : {test_f1:.4f}\n\n")
        f.write(test_report)

    torch.save(best_state, MODEL_DIR / "best_model_v3.pt")
    logging.info(f"Model tersimpan → {MODEL_DIR / 'best_model_v3.pt'}")

    # ── MLflow: log CV summary ke run terpisah ───────────────────────────────
    tracker.start_run(
        run_name="cv_summary",
        params={**HPARAMS, "best_fold": best_fold_num},
    )
    tracker.log_cv_summary(
        mean_f1=mean_f1,
        std_f1=std_f1,
        test_f1=test_f1,
        fold_f1_scores=fold_f1_scores,
    )
    cm_test_path = CM_DIR / "wav2vec_cm_test_set.png"
    tracker.log_artifacts([cv_summary_path, cm_test_path, test_report_path,
                           MODEL_DIR / "best_model_v3.pt"])
    tracker.end_run()

    logging.info("✅ Semua hasil tersimpan di results/")
    print_ui_instructions()
    return mean_f1, std_f1, test_f1


if __name__ == "__main__":
    run_cv()
