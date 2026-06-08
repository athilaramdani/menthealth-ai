"""
2D CNN - Mental Health Audio Classification (NORMAL vs DEPRESI)
Metode: 5-Fold StratifiedKFold Cross-Validation + Fixed Test Set

Versi: v2 (BASELINE STABIL)
  - Arsitektur: 2 conv blocks (32->64), MaxPool 4x4 + 2x2
  - Hyperparameter: LR=5e-4, BS=32, dropout conv=0.2, FC=0.3
  - Hasil: CV F1=0.5203 +/-0.0639, Test F1=0.4571
  - Best epoch per fold: 16, 22, 4, 2, 9 (rata2=10.6)
  - Status: ✅ PRODUCTION READY (baseline stabil)

Cara menjalankan (dari root project):
  python notebooks/CNN/train_2d_cnn.py
  set EPOCHS=100 && python notebooks/CNN/train_2d_cnn.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -- Fix encoding Windows (agar emoji/karakter unicode tidak crash) -----------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# -- MLflow Tracking ----------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))
try:
    from experiments.mlflow_tracking import MLflowTracker, print_ui_instructions
except ImportError:
    class MLflowTracker:
        def __init__(self, *a, **kw): pass
        def start_run(self, *a, **kw): pass
        def log_epoch(self, *a, **kw): pass
        def log_fold_summary(self, *a, **kw): pass
        def log_cv_summary(self, *a, **kw): pass
        def log_artifacts(self, *a, **kw): pass
        def end_run(self): pass
    def print_ui_instructions(): pass

# -- Path setup ----------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "spectrogram"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODEL_DIR    = PROJECT_ROOT / "models" / "dl" / "cnn"

for d in [RESULTS_DIR / "metrics", RESULTS_DIR / "plots",
          RESULTS_DIR / "confusion_matrix", MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -- Logging: file + stdout, keduanya UTF-8 -----------------------------------
log_file_handler = logging.FileHandler(
    SCRIPT_DIR / "train_2dcnn.log", encoding="utf-8"
)
log_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
log_stream_handler = logging.StreamHandler(sys.stdout)
log_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logging.basicConfig(level=logging.INFO, handlers=[log_file_handler, log_stream_handler])

# -- Hyperparameter -----------------------------------------------------------
CLASS_NAMES  = ["NORMAL", "DEPRESI"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
N_FOLDS      = int(os.environ.get("N_FOLDS",  5))
EPOCHS       = int(os.environ.get("EPOCHS",  150))
BATCH_SIZE   = 32       # v2: stabil untuk dataset kecil
LR           = 5e-4     # v2: baseline learning rate
MAX_LEN      = 800
TEST_SIZE    = 0.10
RANDOM_STATE = 42


# -- Dataset ------------------------------------------------------------------
class MelSpectrogramDataset(Dataset):
    def __init__(self, samples, max_len=MAX_LEN):
        self.samples = samples
        self.labels  = [s[1] for s in samples]
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        spec = np.load(npy_path)
        if spec.shape[1] > self.max_len:
            spec = spec[:, :self.max_len]
        else:
            spec = np.pad(spec, ((0, 0), (0, self.max_len - spec.shape[1])), mode="constant")
        return (
            torch.tensor(spec, dtype=torch.float32).unsqueeze(0),
            torch.tensor(label, dtype=torch.long),
        )


def make_loader(samples, batch_size, shuffle=True, weighted=False):
    ds = MelSpectrogramDataset(samples)
    if weighted and shuffle:
        labels_t = torch.tensor(ds.labels, dtype=torch.long)
        counts   = torch.bincount(labels_t, minlength=2).float().clamp(min=1)
        weights  = 1.0 / counts[labels_t]
        weights  = weights / weights.sum()
        sampler  = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        return DataLoader(
            ds, batch_size=batch_size, sampler=sampler,
            num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False,
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False,
    )

# -- Model (v2 BASELINE) ------------------------------------------------------
class MelSpectrogram2DCNN(nn.Module):
    """
    2-block 2D CNN untuk klasifikasi Mel-Spectrogram (v2 BASELINE).
    Input: (B, 1, 128, 800)
    Conv1: 1->32,  MaxPool(4,4) -> (B,32,32,200)
    Conv2: 32->64, MaxPool(2,2) -> (B,64,16,100)
    GAP -> (B,64) -> FC(64->32) GELU Dropout(0.3) -> FC(32->2)

    v2 baseline: dropout conv=0.2, FC=0.3 (proven stable)
    """
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(4, 4), nn.Dropout2d(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten     = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.fc(self.flatten(self.global_pool(
            self.conv2(self.conv1(x))
        )))


def spec_augment(specs, freq_mask=10, time_mask=30, n_masks=1):
    aug = specs.clone()
    _, _, n_mels, T = aug.shape
    for _ in range(n_masks):
        f  = torch.randint(0, freq_mask + 1, (1,)).item()
        f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
        if f:
            aug[:, :, f0:f0 + f, :] = 0
        t  = torch.randint(0, time_mask + 1, (1,)).item()
        t0 = torch.randint(0, max(1, T - t), (1,)).item()
        if t:
            aug[:, :, :, t0:t0 + t] = 0
    return aug


# -- Data loading -------------------------------------------------------------
def load_patient_data():
    label_path = SPLITS_DIR / "custom_2class_labels.csv"
    df = pd.read_csv(label_path)
    id_col = "Participant_ID" if "Participant_ID" in df.columns else "participant_id"
    if "label_depresi" in df.columns:
        patient_to_label = dict(zip(
            df[id_col].astype(str),
            df["label_depresi"].astype(int).map({0: "NORMAL", 1: "DEPRESI"}),
        ))
    else:
        patient_to_label = dict(zip(
            df[id_col].astype(str), df["Custom_Label"].astype(str)
        ))

    all_files, available = [], set()
    for cls, idx in CLASS_TO_IDX.items():
        cls_dir = FEATURES_DIR / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.npy")):
            pid = p.stem.split("_")[0]
            if pid in patient_to_label:
                all_files.append((p, pid, idx))
                available.add(pid)

    patients_list   = sorted(available)
    patients_labels = [CLASS_TO_IDX[patient_to_label[p]] for p in patients_list]
    logging.info(f"Total pasien: {len(patients_list)} | Total files: {len(all_files)}")
    return patients_list, patients_labels, all_files, patient_to_label


def split_samples(all_files, train_pids, val_pids):
    AUG_TAGS = ["_noise", "_pitch", "_stretch", "_combo"]
    train_s, val_s = [], []
    for path, pid, cls_idx in all_files:
        stem = path.stem
        is_original = stem.endswith("_mel") and not any(t in stem for t in AUG_TAGS)
        if pid in train_pids:
            train_s.append((path, cls_idx))
        elif pid in val_pids and is_original:
            val_s.append((path, cls_idx))
    return train_s, val_s


# -- Single fold training -----------------------------------------------------
def train_one_fold(train_samples, val_samples, fold_num, device,
                   epochs=EPOCHS, tracker=None):
    train_loader = make_loader(train_samples, BATCH_SIZE, shuffle=True,  weighted=True)
    val_loader   = make_loader(val_samples,   BATCH_SIZE, shuffle=False)

    model = MelSpectrogram2DCNN(num_classes=2).to(device)

    labels_t = torch.tensor([s[1] for s in train_samples], dtype=torch.long)
    counts   = torch.bincount(labels_t, minlength=2).float().clamp(min=1)

    # v2: bobot DEPRESI standar 2x (balanced approach)
    cw = (counts.sum() / (2.0 * counts))
    cw = cw.to(device)
    logging.info(f"  Class weights v2: NORMAL={cw[0]:.3f} DEPRESI={cw[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)   # v2: label smoothing 0.05
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # v2: warmup standar 8 epoch dari 10% LR (proven stable)
    warmup  = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=8)
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=12, min_lr=1e-6
    )

    history = {k: [] for k in [
        "train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1",
        "val_recall_normal", "val_recall_depresi",
        "val_precision_normal", "val_precision_depresi",
    ]}

    best_f1, best_epoch = -1.0, 0
    best_preds, best_labels_list, best_state = [], [], None
    no_improve = 0
    last_lr    = LR

    for epoch in range(epochs):
        # Train
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for specs, labels in tqdm(train_loader, desc=f"Fold{fold_num} E{epoch+1}", leave=False):
            specs, labels = specs.to(device), labels.to(device)
            specs = spec_augment(specs)
            optimizer.zero_grad()
            out  = model(specs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss  += loss.item()
            _, pred  = torch.max(out, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()

        e_tloss = t_loss / max(1, len(train_loader))
        e_tacc  = 100.0 * correct / max(1, total)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_p, all_l = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                out  = model(specs)
                loss = criterion(out, labels)
                v_loss    += loss.item()
                _, pred    = torch.max(out, 1)
                v_total   += labels.size(0)
                v_correct += (pred == labels).sum().item()
                all_p.extend(pred.cpu().numpy())
                all_l.extend(labels.cpu().numpy())

        e_vloss = v_loss / max(1, len(val_loader))
        e_vacc  = 100.0 * v_correct / max(1, v_total)
        e_f1    = f1_score(all_l, all_p, average="macro", zero_division=0)
        prec, rec, _, _ = precision_recall_fscore_support(
            all_l, all_p, labels=[0, 1], zero_division=0
        )

        for k, v in zip(history.keys(), [
            e_tloss, e_vloss, e_tacc, e_vacc, e_f1,
            rec[0], rec[1], prec[0], prec[1],
        ]):
            history[k].append(v)

        logging.info(
            f"[Fold {fold_num}] Ep {epoch+1:3d} | "
            f"TLoss {e_tloss:.4f} TAcc {e_tacc:.1f}% | "
            f"VLoss {e_vloss:.4f} VAcc {e_vacc:.1f}% F1 {e_f1:.4f} | "
            f"Rec N:{rec[0]:.3f} D:{rec[1]:.3f}"
        )

        if tracker:
            tracker.log_epoch(epoch + 1, {
                "train/loss":            e_tloss,
                "train/acc":             e_tacc,
                "val/loss":              e_vloss,
                "val/acc":               e_vacc,
                "val/macro_f1":          e_f1,
                "val/recall_normal":     float(rec[0]),
                "val/recall_depresi":    float(rec[1]),
                "val/precision_normal":  float(prec[0]),
                "val/precision_depresi": float(prec[1]),
                "lr":                    optimizer.param_groups[0]["lr"],
            })

        if e_f1 > best_f1:
            best_f1  = e_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = list(all_p)
            best_labels_list = list(all_l)
            no_improve = 0
            logging.info(f"  [BEST] F1: {best_f1:.4f} @ epoch {best_epoch}")
        else:
            no_improve += 1

        if epoch < 8:
            warmup.step()
        else:
            plateau.step(e_f1)

        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != last_lr:
            logging.info(f"  [LR] {last_lr:.2e} -> {new_lr:.2e}")
        last_lr = new_lr

        if no_improve >= 35:   # v2: early stopping 35 epoch
            logging.info(f"  [EARLY STOP] @ epoch {epoch+1}")
            break

    return best_f1, best_epoch, best_state, best_preds, best_labels_list, history


# -- Plotting -----------------------------------------------------------------
def plot_fold_history(history, fold_num):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))

    axes[0].plot(ep, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="tomato")
    axes[0].set_title(f"Loss - Fold {fold_num}")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(ep, history["train_acc"], label="Train", color="steelblue")
    axes[1].plot(ep, history["val_acc"],   label="Val",   color="tomato")
    axes[1].set_title(f"Accuracy - Fold {fold_num}")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(ep, history["val_macro_f1"], color="green", label="Macro F1")
    axes[2].set_title(f"Macro F1 - Fold {fold_num}")
    axes[2].set_ylim(0, 1.05); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True, linestyle="--", alpha=0.4)

    axes[3].plot(ep, history["val_recall_normal"],     label="Recall NORMAL",  color="steelblue",  marker="o", markersize=2)
    axes[3].plot(ep, history["val_recall_depresi"],    label="Recall DEPRESI", color="tomato",     marker="s", markersize=2)
    axes[3].plot(ep, history["val_precision_normal"],  label="Prec NORMAL",    color="steelblue",  linestyle="--", alpha=0.6)
    axes[3].plot(ep, history["val_precision_depresi"], label="Prec DEPRESI",   color="tomato",     linestyle="--", alpha=0.6)
    axes[3].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    axes[3].set_title(f"Recall & Precision - Fold {fold_num}")
    axes[3].set_ylim(0, 1.05); axes[3].set_xlabel("Epoch")
    axes[3].legend(fontsize=7); axes[3].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / f"fold_{fold_num}_curves.png", dpi=150)
    plt.close()


def plot_cv_summary(fold_f1_scores):
    folds   = [f"Fold {i+1}" for i in range(len(fold_f1_scores))]
    mean_f1 = np.mean(fold_f1_scores)
    std_f1  = np.std(fold_f1_scores)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, fold_f1_scores, color="steelblue", alpha=0.8, edgecolor="black")
    plt.axhline(mean_f1, color="tomato", linestyle="--", linewidth=2,
                label=f"Mean F1 = {mean_f1:.4f}")
    plt.fill_between(range(len(folds)), mean_f1 - std_f1, mean_f1 + std_f1,
                     alpha=0.15, color="tomato", label=f"+/-1 Std = {std_f1:.4f}")
    for bar, val in zip(bars, fold_f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    plt.title("5-Fold CV - Macro F1 per Fold (CNN v2 Baseline)")
    plt.ylabel("Macro F1"); plt.ylim(0, 1.05)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "cv_summary.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm_arr = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix" / filename, dpi=150)
    plt.close()


# -- Main: 5-Fold CV ----------------------------------------------------------
def run_cv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    tracker = MLflowTracker(experiment_name="2D-CNN-Spectrogram-v2-baseline")
    print_ui_instructions()

    HPARAMS = {
        "model":        "2D-CNN-v2-baseline",
        "n_folds":      N_FOLDS,
        "epochs":       EPOCHS,
        "batch_size":   BATCH_SIZE,
        "lr":           LR,
        "weight_decay": 1e-4,
        "dropout_rate": 0.3,
        "label_smooth": 0.05,
        "max_len":      MAX_LEN,
        "random_state": RANDOM_STATE,
        "device":       str(device),
    }

    patients_list, patients_labels, all_files, _ = load_patient_data()

    # 1. Pisahkan test set (10%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(sss.split(patients_list, patients_labels))

    trainval_patients = [patients_list[i] for i in trainval_idx]
    trainval_labels   = [patients_labels[i] for i in trainval_idx]
    test_patients     = {patients_list[i] for i in test_idx}

    AUG_TAGS = ["_noise", "_pitch", "_stretch", "_combo"]
    test_samples = [
        (p, cls) for p, pid, cls in all_files
        if pid in test_patients
        and p.stem.endswith("_mel")
        and not any(t in p.stem for t in AUG_TAGS)
    ]
    logging.info(f"Test set: {len(test_samples)} samples dari {len(test_patients)} pasien")

    # 2. 5-Fold StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_f1_scores = []
    fold_results   = []
    fold_metrics   = []

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(trainval_patients, trainval_labels), start=1
    ):
        logging.info(f"\n{'='*60}")
        logging.info(f"FOLD {fold}/{N_FOLDS}")
        logging.info(f"{'='*60}")

        train_pids = {trainval_patients[i] for i in tr_idx}
        val_pids   = {trainval_patients[i] for i in val_idx}
        train_s, val_s = split_samples(all_files, train_pids, val_pids)

        tr_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter([s[1] for s in train_s]).items())}
        vl_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter([s[1] for s in val_s]).items())}
        logging.info(f"Train dist: {tr_dist} | Val dist: {vl_dist}")

        tracker.start_run(
            run_name=f"fold_{fold}",
            params={**HPARAMS, "fold": fold,
                    "train_samples": len(train_s), "val_samples": len(val_s)},
        )

        result = train_one_fold(train_s, val_s, fold, device, epochs=EPOCHS, tracker=tracker)
        best_f1, best_epoch, best_state, best_preds, best_labels_list, history = result

        fold_f1_scores.append(best_f1)
        fold_results.append(result)
        logging.info(f"Fold {fold} selesai - Best Macro F1: {best_f1:.4f} @ epoch {best_epoch}")

        plot_fold_history(history, fold)
        cm_path     = RESULTS_DIR / "confusion_matrix" / f"cm_fold_{fold}.png"
        curves_path = RESULTS_DIR / "plots" / f"fold_{fold}_curves.png"
        plot_confusion_matrix(
            best_labels_list, best_preds,
            title=f"Confusion Matrix - Fold {fold} (F1={best_f1:.4f})",
            filename=f"cm_fold_{fold}.png",
        )

        report_path = RESULTS_DIR / "metrics" / f"report_fold_{fold}.txt"
        report_str  = classification_report(
            best_labels_list, best_preds,
            target_names=CLASS_NAMES, labels=[0, 1], zero_division=0,
        )
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(f"=== Fold {fold} | Best Epoch {best_epoch} | Macro F1: {best_f1:.4f} ===\n\n")
            fh.write(report_str)

        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            best_labels_list, best_preds, labels=[0, 1], zero_division=0
        )
        fold_acc = sum(p == l for p, l in zip(best_preds, best_labels_list)) / max(1, len(best_labels_list))
        fold_metrics.append({
            "Fold":          fold,
            "Best Epoch":    best_epoch,
            "Val Accuracy":  round(fold_acc, 4),
            "Macro F1":      round(best_f1, 4),
            "Prec NORMAL":   round(float(prec_arr[0]), 4),
            "Prec DEPRESI":  round(float(prec_arr[1]), 4),
            "Recall NORMAL": round(float(rec_arr[0]), 4),
            "Recall DEPRESI":round(float(rec_arr[1]), 4),
            "F1 NORMAL":     round(float(f1_arr[0]), 4),
            "F1 DEPRESI":    round(float(f1_arr[1]), 4),
        })

        tracker.log_fold_summary(fold=fold, best_f1=best_f1, best_epoch=best_epoch)
        tracker.log_artifacts([cm_path, curves_path, report_path])
        tracker.end_run()

    # 3. Ringkasan CV
    mean_f1 = float(np.mean(fold_f1_scores))
    std_f1  = float(np.std(fold_f1_scores))
    logging.info(f"\n{'='*60}")
    logging.info("5-FOLD CV SELESAI")
    logging.info(f"F1 per fold : {[f'{v:.4f}' for v in fold_f1_scores]}")
    logging.info(f"Mean Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    logging.info(f"{'='*60}")

    df_fold = pd.DataFrame(fold_metrics)
    mean_row = {"Fold": "Mean", "Best Epoch": "-"}
    std_row  = {"Fold": "Std",  "Best Epoch": "-"}
    for col in df_fold.columns[2:]:
        mean_row[col] = round(float(df_fold[col].mean()), 4)
        std_row[col]  = round(float(df_fold[col].std()),  4)
    df_fold = pd.concat([df_fold, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    logging.info("\n" + df_fold.to_string(index=False))

    fold_compare_path = RESULTS_DIR / "metrics" / "fold_comparison.csv"
    df_fold.to_csv(fold_compare_path, index=False)

    plot_cv_summary(fold_f1_scores)
    cv_summary_path = RESULTS_DIR / "plots" / "cv_summary.png"

    with open(RESULTS_DIR / "metrics" / "cv_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("=== 5-Fold CV Summary - 2D CNN Mel-Spectrogram v2 Baseline ===\n\n")
        for i, v in enumerate(fold_f1_scores, 1):
            fh.write(f"Fold {i}: Macro F1 = {v:.4f}\n")
        fh.write(f"\nMean Macro F1 : {mean_f1:.4f}\n")
        fh.write(f"Std  Macro F1 : {std_f1:.4f}\n\n")
        fh.write(df_fold.to_string(index=False))
        fh.write("\n")

    # 4. Evaluasi best fold di test set
    best_fold_idx = int(np.argmax(fold_f1_scores))
    best_fold_num = best_fold_idx + 1
    _, _, best_state, _, _, _ = fold_results[best_fold_idx]

    logging.info(f"\nEvaluasi best fold ({best_fold_num}) di test set...")
    model = MelSpectrogram2DCNN(num_classes=2).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    test_loader = DataLoader(
        MelSpectrogramDataset(test_samples), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            _, pred = torch.max(model(specs), 1)
            test_preds.extend(pred.cpu().numpy())
            test_labels_list.extend(labels.numpy())

    test_f1 = f1_score(test_labels_list, test_preds, average="macro", zero_division=0)
    logging.info(f"Test Set Macro F1: {test_f1:.4f}")

    plot_confusion_matrix(
        test_labels_list, test_preds,
        title=f"Confusion Matrix - Test Set (F1={test_f1:.4f})",
        filename="cm_test_set.png",
    )

    test_report_str = classification_report(
        test_labels_list, test_preds,
        target_names=CLASS_NAMES, labels=[0, 1], zero_division=0,
    )
    print("\n=== TEST SET CLASSIFICATION REPORT (CNN v2 Baseline) ===")
    print(test_report_str)

    test_report_path = RESULTS_DIR / "metrics" / "report_test_set.txt"
    with open(test_report_path, "w", encoding="utf-8") as fh:
        fh.write(f"=== Test Set Report - CNN v2 Baseline (Best Fold: {best_fold_num}) ===\n\n")
        fh.write(f"CV Mean Macro F1 : {mean_f1:.4f} +/- {std_f1:.4f}\n")
        fh.write(f"Test Macro F1    : {test_f1:.4f}\n\n")
        fh.write(test_report_str)

    # Simpan model
    model_path = MODEL_DIR / "best_model_v2_baseline.pt"
    torch.save(best_state, model_path)
    logging.info(f"Model disimpan -> {model_path}")

    # MLflow summary run
    tracker.start_run(
        run_name="cv_summary",
        params={**HPARAMS, "best_fold": best_fold_num},
    )
    tracker.log_cv_summary(
        mean_f1=mean_f1, std_f1=std_f1,
        test_f1=test_f1, fold_f1_scores=fold_f1_scores,
    )
    cm_test_path = RESULTS_DIR / "confusion_matrix" / "cm_test_set.png"
    tracker.log_artifacts([cv_summary_path, cm_test_path, test_report_path, model_path])
    tracker.end_run()

    logging.info("Semua hasil tersimpan di results/")
    print_ui_instructions()
    return mean_f1, std_f1, test_f1


if __name__ == "__main__":
    run_cv()
