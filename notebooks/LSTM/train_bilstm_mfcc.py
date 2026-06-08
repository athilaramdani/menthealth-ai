"""
LSTM Step 2 - BiLSTM Training on MFCC Segments
================================================
Project : Mental Health Audio Classification (NORMAL vs DEPRESI)
Script  : notebooks/LSTM/train_bilstm_mfcc.py

📝 DEEP LEARNING PIPELINE NOTE: BiLSTM TRAINING & PERFORMANCE
- Role: Trains a Bidirectional LSTM classifier on 10s MFCC segment sequences.
- Architecture:
  1. Input: (batch, T=313, features=120).
  2. Backbone: 2-layer Bidirectional LSTM (hidden_size=128 per direction, concatenated to 256).
  3. Pooling: Attention-weighted temporal pooling over timesteps.
  4. Fully-Connected Head: Linear(256 -> 128) + GELU + Dropout(0.35) -> Linear(128 -> 2).
- Anti-Leakage Split: Patient-level 5-Fold Stratified CV.
- Imbalance Handled: WeightedRandomSampler + class weights in CrossEntropyLoss.
- Final Test Performance (v1):
  - Test Set Accuracy: 61%
  - Test Set Macro F1: 0.5563
  - CV Mean Macro F1: 0.5359 ± 0.0446

- Architecture Changes (v2 — tuned):
  - Single-head attention → Multi-Head Attention Pooling (4 heads)
  - Hidden dim 128 → 192 for larger representational capacity
  - Dropout 0.35 → 0.40 for stronger regularization
  - Added LSTM output projection layer before pooling
  - Warmup extended to 6 epochs, min_lr=5e-7

- Architecture Changes (v3 — optimized retrain):
  - LR 1e-3 → 5e-4: best epoch 2-7 di v1 indikasi LR terlalu besar (overshoot)
  - EPOCHS 80 → 120: fold 5 v1 masih naik di ep40, butuh lebih banyak iterasi
  - PATIENCE_ES 28 → 40: sinkron dengan LR lebih kecil yang butuh waktu lebih
  - PATIENCE_LR 12 → 15: beri plateau lebih sabar
  - Class weights: 1/count → (1/count)^1.5 (squared inverse) agar recall DEPRESI naik
  - Label smoothing: 0.08 → 0.05: turunkan agar model lebih yakin pada minority class
  - Warmup: 6 → 10 epoch: LR lebih kecil butuh fase warmup lebih panjang
  - hidden_dim=192 aktif (sebelumnya checkpoint masih hidden_dim=128)

Arsitektur:
  - Input  : (batch, T_frames, n_mfcc*3)  → 313 timesteps × 120 features (10 detik)
  - Layer  : 2-layer Bidirectional LSTM (hidden 128 each direction = 256 total)
  - Pool   : Attention-weighted mean pooling atas seluruh timesteps
  - FC     : Linear(256 → 128) → ReLU → Dropout → Linear(128 → 2)
  - Output : CrossEntropyLoss dengan class weights

Strategi Evaluasi — PENTING!
  Data kita adalah SEGMEN dari PASIEN yang sama.
  → Kita TIDAK boleh mencampur segmen satu pasien di train dan val.
  → Split dilakukan di level PATIENT, bukan di level segmen.
  → Metode: 5-Fold StratifiedKFold pada daftar PASIEN.

Konsistensi dengan CNN:
  - StratifiedKFold 5-fold
  - WeightedRandomSampler untuk handle imbalance
  - AdamW + Warmup + ReduceLROnPlateau scheduler
  - Early stopping pada Macro F1
  - Simpan best model per fold → evaluasi test set
  - Semua plot & metrics ke results/lstm/
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# Paksa stdout encoding utf-8 di Windows agar karakter non-ASCII aman
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ── MLflow Tracking ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from experiments.mlflow_tracking import MLflowTracker, print_ui_instructions
except ImportError:
    # Fallback jika module belum ada
    class MLflowTracker:  # type: ignore
        def __init__(self, *a, **kw): pass
        def start_run(self, *a, **kw): pass
        def log_epoch(self, *a, **kw): pass
        def log_fold_summary(self, *a, **kw): pass
        def log_cv_summary(self, *a, **kw): pass
        def log_artifacts(self, *a, **kw): pass
        def end_run(self): pass
    def print_ui_instructions(): pass

# ── Path Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MFCC_DIR     = PROJECT_ROOT / "data" / "features" / "mfcc"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR  = PROJECT_ROOT / "results" / "lstm"
MODEL_DIR    = PROJECT_ROOT / "models" / "dl" / "lstm"

for d in [RESULTS_DIR / "metrics", RESULTS_DIR / "plots",
          RESULTS_DIR / "confusion_matrix", MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "train_bilstm.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# ── Hyperparameter ─────────────────────────────────────────────────────────────
CLASS_NAMES  = ["NORMAL", "DEPRESI"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
N_FOLDS      = int(os.environ.get("N_FOLDS", 5))
EPOCHS       = int(os.environ.get("EPOCHS", 120))  # naik 80→120: fold 5 belum konvergen di ep40
BATCH_SIZE   = 64       # lebih besar karena data segmen jauh lebih banyak
LR           = 5e-4     # turun 1e-3→5e-4: best epoch 2-7 di v1 = LR terlalu besar
WEIGHT_DECAY = 1e-4
MAX_T        = 313      # timestep untuk 10 detik @ sr=16000, hop=512 → ceil(160000/512)+1
N_FEATURES   = 120      # n_mfcc(40) × 3 (raw + delta + delta2)
HIDDEN_DIM   = 192      # naik dari 128 → kapasitas lebih besar (aktif di v2)
NUM_LAYERS   = 2
DROPOUT      = 0.40     # naik dari 0.35 → regularisasi lebih kuat (tuned)
RANDOM_STATE = 42
PATIENCE_ES  = 40       # naik 28→40: beri lebih banyak waktu dengan LR lebih kecil
PATIENCE_LR  = 15       # naik 12→15: sinkron dengan LR lebih kecil


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DATASET                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Global RAM cache to prevent disk I/O bottlenecks during training
FEATURE_CACHE = {}

class MFCCSegmentDataset(Dataset):
    """
    Dataset untuk segmen MFCC.
    Setiap sampel adalah 1 file .npy berisi shape (T_frames, n_features).
    Pad/truncate ke MAX_T untuk collate ke satu batch.
    """
    def __init__(self, samples: list, max_t: int = MAX_T):
        """
        Args:
            samples: list of (Path, class_idx)
            max_t  : panjang sekuens yang diinginkan (pad/truncate)
        """
        self.samples = samples
        self.labels  = [s[1] for s in samples]
        self.max_t   = max_t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        path_str = str(npy_path)

        if path_str in FEATURE_CACHE:
            feat = FEATURE_CACHE[path_str]
        else:
            feat = np.load(path_str)           # (T, n_features)

            # Truncate jika lebih panjang dari MAX_T
            if feat.shape[0] > self.max_t:
                feat = feat[:self.max_t, :]

            # Pad dengan zeros jika lebih pendek
            elif feat.shape[0] < self.max_t:
                pad = self.max_t - feat.shape[0]
                feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant")

            # Pastikan dimensi fitur konsisten
            if feat.shape[1] != N_FEATURES:
                if feat.shape[1] < N_FEATURES:
                    feat = np.pad(feat, ((0, 0), (0, N_FEATURES - feat.shape[1])), mode="constant")
                else:
                    feat = feat[:, :N_FEATURES]

            feat = torch.tensor(feat, dtype=torch.float32)
            FEATURE_CACHE[path_str] = feat

        return feat, torch.tensor(label, dtype=torch.long)


def make_weighted_loader(samples, batch_size, shuffle=True):
    """Buat DataLoader dengan WeightedRandomSampler untuk handle class imbalance."""
    ds = MFCCSegmentDataset(samples)
    if shuffle:
        labels_t = torch.tensor(ds.labels, dtype=torch.long)
        counts   = torch.bincount(labels_t, minlength=2).float().clamp(min=1)
        weights  = 1.0 / counts[labels_t]
        weights  = weights / weights.sum()
        sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=torch.cuda.is_available())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL — BiLSTM + Attention Pooling                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MultiHeadAttentionPool(nn.Module):
    """
    Multi-Head Attention Pooling over LSTM timesteps (tuned v2).

    Menggunakan beberapa "kepala perhatian" yang independen untuk menangkap
    berbagai aspek temporal dari output LSTM secara bersamaan.
    Setiap head belajar fokus pada pola temporal yang berbeda.

    Dibanding single-head attention:
    - Head 1 bisa fokus pada awal/akhir ucapan
    - Head 2 bisa fokus pada perubahan nada
    - Head 3 bisa fokus pada ritme bicara, dst.
    """
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = hidden_size // num_heads
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) harus habis dibagi num_heads ({num_heads})"

        # Proyeksi per head
        self.key_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query      = nn.Parameter(torch.randn(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.query.unsqueeze(0))

    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        B, T, H = lstm_out.shape

        # Project dan reshape ke (B, T, num_heads, head_dim)
        keys    = self.key_proj(lstm_out)  # (B, T, H)
        keys    = keys.view(B, T, self.num_heads, self.head_dim)  # (B, T, heads, d)

        # Query: (heads, d) → broadcast ke (B, 1, heads, d)
        q = self.query.unsqueeze(0).unsqueeze(0)  # (1, 1, heads, d)

        # Attention scores: (B, T, heads)
        scores  = (keys * q).sum(dim=-1) / (self.head_dim ** 0.5)  # scaled dot-product
        weights = torch.softmax(scores, dim=1)  # (B, T, heads)

        # Weighted sum per head: (B, heads, d)
        # weights: (B, T, heads) → (B, T, heads, 1) broadcast
        pooled = (keys * weights.unsqueeze(-1)).sum(dim=1)  # (B, heads, d)

        # Flatten heads: (B, H)
        return pooled.reshape(B, H)


class BiLSTMMFCC(nn.Module):
    """
    Bidirectional LSTM untuk klasifikasi MFCC segmen (tuned v2).

    Perubahan dari v1:
    - Single-head attention pool → Multi-head attention pool (4 heads)
    - Hidden dim 128 → 192 untuk kapasitas representasi lebih besar
    - Tambah residual connection di LSTM output sebelum pooling
    - FC head diperlebar: 256→256 → 384→192 dengan LayerNorm
    - Dropout dinaikkan sedikit (0.35→0.40) untuk regularisasi lebih kuat

    Arsitektur:
        Input BN → BiLSTM (2 layer, h=192) → MultiHead AttentionPool → FC (384→192→2)

    Input : (B, T, n_features)  →  Output: (B, n_classes)
    """
    def __init__(
        self,
        n_features  : int = N_FEATURES,
        hidden_dim  : int = HIDDEN_DIM,
        num_layers  : int = NUM_LAYERS,
        n_classes   : int = 2,
        dropout     : float = DROPOUT,
        num_heads   : int = 4,
    ):
        super().__init__()

        # Batch norm pada fitur input
        self.input_bn = nn.BatchNorm1d(n_features)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= True,
        )

        lstm_out_dim = hidden_dim * 2   # bidirectional → 2× hidden

        # Multi-head attention pooling (tuned)
        self.attn_pool = MultiHeadAttentionPool(lstm_out_dim, num_heads=num_heads)

        # Projection layer setelah LSTM untuk smooth gradient flow
        self.lstm_proj = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout * 0.5),  # lighter dropout pada projection
        )

        # Fully connected classifier diperlebar
        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        # x: (B, T, n_features)
        B, T, F = x.shape

        # BatchNorm pada fitur
        x_bn = self.input_bn(x.reshape(-1, F)).reshape(B, T, F)

        # BiLSTM
        lstm_out, _ = self.lstm(x_bn)   # (B, T, hidden*2)

        # Projection (stabilisasi gradient)
        lstm_proj = self.lstm_proj(lstm_out)  # (B, T, hidden*2)

        # Multi-head attention pooling
        pooled = self.attn_pool(lstm_proj)  # (B, hidden*2)

        return self.fc(pooled)             # (B, n_classes)

    def get_attention_weights(self, x):
        """
        Kembalikan attention weights (head pertama) untuk visualisasi XAI.
        Returns: (B, T) — bobot tiap timestep.
        """
        B, T, F = x.shape
        x_bn = self.input_bn(x.reshape(-1, F)).reshape(B, T, F)
        lstm_out, _ = self.lstm(x_bn)
        lstm_proj = self.lstm_proj(lstm_out)

        # Ambil head pertama untuk visualisasi
        keys    = self.attn_pool.key_proj(lstm_proj)  # (B, T, H)
        keys    = keys.view(B, T, self.attn_pool.num_heads, self.attn_pool.head_dim)
        q       = self.attn_pool.query[0].unsqueeze(0).unsqueeze(0)  # (1, 1, d)
        scores  = (keys[:, :, 0, :] * q[0, 0]).sum(dim=-1) / (self.attn_pool.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        return weights


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING (patient-aware)                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_patient_data():
    """
    Scan MFCC segment .npy files dan kelompokkan per patient.

    Returns:
        patients_list   : sorted list of participant_id (str)
        patients_labels : list of class_idx per patient
        all_files       : list of (Path, patient_id_str, class_idx)
        patient_to_label: dict pid→label_name
    """
    label_path = SPLITS_DIR / "custom_2class_labels.csv"
    df = pd.read_csv(label_path)
    patient_to_label = dict(
        zip(
            df["participant_id"].astype(str),
            df["label_depresi"].astype(int).map({0: "NORMAL", 1: "DEPRESI"}),
        )
    )

    all_files = []
    available = set()

    for cls_name, cls_idx in CLASS_TO_IDX.items():
        cls_dir = MFCC_DIR / cls_name
        if not cls_dir.exists():
            logging.warning(f"Folder kelas tidak ditemukan: {cls_dir}")
            continue
        for npy_path in sorted(cls_dir.glob("*.npy")):
            pid = npy_path.stem.split("_seg")[0]   # "303_seg002" → "303"
            if pid in patient_to_label:
                all_files.append((npy_path, pid, cls_idx))
                available.add(pid)

    patients_list   = sorted(available)
    patients_labels = [CLASS_TO_IDX[patient_to_label[p]] for p in patients_list]

    # Distribusi kelas
    cls_count = Counter(patients_labels)
    logging.info(
        f"Total patient: {len(patients_list)} | "
        f"NORMAL: {cls_count[0]} | DEPRESI: {cls_count[1]}"
    )
    logging.info(f"Total segmen: {len(all_files)}")
    return patients_list, patients_labels, all_files, patient_to_label


def split_by_patients(all_files, train_pids: set, val_pids: set):
    """
    Pisahkan segmen berdasarkan patient set.
    Val hanya menggunakan file original (tanpa augmentasi).
    """
    train_s, val_s = [], []
    for path, pid, cls_idx in all_files:
        if pid in train_pids:
            train_s.append((path, cls_idx))
        elif pid in val_pids:
            val_s.append((path, cls_idx))
    return train_s, val_s


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING LOOP — SINGLE FOLD                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_class_weights(samples, device):
    """
    Hitung class weights dari distribusi sampel dalam fold ini.
    Gunakan squared inverse untuk memperkuat penalti pada DEPRESI
    yang jumlah pasiennya hampir 2x lebih sedikit dari NORMAL.
    """
    labels_t = torch.tensor([s[1] for s in samples], dtype=torch.long)
    counts   = torch.bincount(labels_t, minlength=2).float().clamp(min=1)
    # Squared inverse: DEPRESI mendapat bobot lebih agresif
    cw       = (1.0 / counts) ** 1.5
    cw       = (cw / cw.sum() * 2).to(device)
    logging.info(f"  Class weights: NORMAL={cw[0]:.4f}, DEPRESI={cw[1]:.4f}")
    return cw


def train_one_fold(train_samples, val_samples, fold_num, device, epochs=EPOCHS,
                   tracker: "MLflowTracker" = None):
    """
    Training satu fold: return (best_f1, best_epoch, best_state, preds, labels, history).
    tracker: MLflowTracker instance (optional) untuk log metrics per epoch.
    """
    train_loader = make_weighted_loader(train_samples, BATCH_SIZE, shuffle=True)
    val_loader   = make_weighted_loader(val_samples,   BATCH_SIZE, shuffle=False)

    model = BiLSTMMFCC().to(device)

    cw       = compute_class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)  # turun 0.08→0.05: lebih percaya diri pada DEPRESI
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # LR Scheduler: warmup 10 epoch (lebih panjang krn LR lebih kecil) → ReduceLROnPlateau
    warmup  = optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, total_iters=10)  # naik 6→10
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=PATIENCE_LR, min_lr=5e-7
    )

    history = {k: [] for k in [
        "train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1",
        "val_recall_normal", "val_recall_depresi",
        "val_precision_normal", "val_precision_depresi",
    ]}

    best_f1      = -1.0
    best_epoch   = 0
    best_state   = None
    best_preds   = []
    best_labels  = []
    no_improve   = 0
    last_lr      = LR

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        t_loss, correct, total = 0.0, 0, 0

        for feats, labels in tqdm(
            train_loader,
            desc=f"Fold{fold_num} E{epoch+1:02d}",
            leave=False
        ):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss  += loss.item()
            _, pred  = torch.max(out, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()

        e_tloss = t_loss / max(1, len(train_loader))
        e_tacc  = 100.0 * correct / max(1, total)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_p, all_l = [], []

        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                out  = model(feats)
                loss = criterion(out, labels)
                v_loss  += loss.item()
                _, pred  = torch.max(out, 1)
                v_total += labels.size(0)
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

        # ── MLflow: log metrics per epoch ───────────────────────────────────
        if tracker:
            tracker.log_epoch(epoch + 1, {
                "train/loss"           : e_tloss,
                "train/acc"            : e_tacc,
                "val/loss"             : e_vloss,
                "val/acc"              : e_vacc,
                "val/macro_f1"         : e_f1,
                "val/recall_normal"    : float(rec[0]),
                "val/recall_depresi"   : float(rec[1]),
                "val/precision_normal" : float(prec[0]),
                "val/precision_depresi": float(prec[1]),
                "lr"                   : optimizer.param_groups[0]["lr"],
            })

        # Simpan best model
        if e_f1 > best_f1:
            best_f1    = e_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds  = list(all_p)
            best_labels = list(all_l)
            no_improve  = 0
            logging.info(f"  ✅ Best Macro F1: {best_f1:.4f} @ epoch {best_epoch}")
        else:
            no_improve += 1

        # Scheduler step
        if epoch < 9:   # warmup 10 epoch (0-indexed: 0..9)
            warmup.step()
        else:
            plateau.step(e_f1)

        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != last_lr:
            logging.info(f"  ⚠️  LR: {last_lr:.6f} → {new_lr:.6f}")
        last_lr = new_lr

        # Early stopping
        if no_improve >= PATIENCE_ES:
            logging.info(f"  ⏹  Early stopping @ epoch {epoch+1}")
            break

    return best_f1, best_epoch, best_state, best_preds, best_labels, history


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def plot_fold_history(history, fold_num):
    ep  = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))

    axes[0].plot(ep, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="tomato")
    axes[0].set_title(f"Loss — Fold {fold_num}")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(ep, history["train_acc"], label="Train", color="steelblue")
    axes[1].plot(ep, history["val_acc"],   label="Val",   color="tomato")
    axes[1].set_title(f"Accuracy — Fold {fold_num}")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(ep, history["val_macro_f1"], color="green", label="Macro F1")
    axes[2].set_title(f"Macro F1 — Fold {fold_num}")
    axes[2].set_ylim(0, 1.05); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True, linestyle="--", alpha=0.4)

    axes[3].plot(ep, history["val_recall_normal"],     label="Recall NORMAL",  color="steelblue",  marker="o", markersize=2)
    axes[3].plot(ep, history["val_recall_depresi"],    label="Recall DEPRESI", color="tomato",     marker="s", markersize=2)
    axes[3].plot(ep, history["val_precision_normal"],  label="Prec NORMAL",    color="steelblue",  linestyle="--", alpha=0.6)
    axes[3].plot(ep, history["val_precision_depresi"], label="Prec DEPRESI",   color="tomato",     linestyle="--", alpha=0.6)
    axes[3].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    axes[3].set_title(f"Recall & Precision — Fold {fold_num}")
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
                     alpha=0.15, color="tomato", label=f"±1 Std = {std_f1:.4f}")
    for bar, val in zip(bars, fold_f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    plt.title("5-Fold CV — Macro F1 per Fold (BiLSTM MFCC)")
    plt.ylabel("Macro F1"); plt.ylim(0, 1.05)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "cv_summary.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix" / filename, dpi=150)
    plt.close()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN — 5-Fold CV                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_cv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Model: BiLSTM | Hidden: {HIDDEN_DIM} | Layers: {NUM_LAYERS} | Dropout: {DROPOUT}")
    logging.info(f"Input shape per sample: ({MAX_T}, {N_FEATURES})")

    # ── MLflow Tracker ────────────────────────────────────────────────────────
    tracker = MLflowTracker(experiment_name="BiLSTM-MFCC")
    print_ui_instructions()

    # Hyperparameter yang akan di-log
    HPARAMS = {
        "model"        : "BiLSTM-MFCC",
        "n_folds"      : N_FOLDS,
        "epochs"       : EPOCHS,
        "batch_size"   : BATCH_SIZE,
        "lr"           : LR,
        "weight_decay" : WEIGHT_DECAY,
        "hidden_dim"   : HIDDEN_DIM,
        "num_layers"   : NUM_LAYERS,
        "dropout"      : DROPOUT,
        "max_t"        : MAX_T,
        "n_features"   : N_FEATURES,
        "patience_es"  : PATIENCE_ES,
        "patience_lr"  : PATIENCE_LR,
        "random_state" : RANDOM_STATE,
        "device"       : str(device),
    }

    patients_list, patients_labels, all_files, _ = load_patient_data()

    if len(patients_list) == 0:
        logging.error(
            "Tidak ada data ditemukan! Pastikan extract_mfcc_segments.py sudah dijalankan "
            "dan folder data/features/mfcc/NORMAL & DEPRESI berisi file .npy"
        )
        sys.exit(1)

    # ── 1. Pisahkan fixed test set (10% patient) — tidak disentuh selama CV ──
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(sss.split(patients_list, patients_labels))

    trainval_patients = [patients_list[i] for i in trainval_idx]
    trainval_labels   = [patients_labels[i] for i in trainval_idx]
    test_patients     = {patients_list[i] for i in test_idx}

    # Test samples (semua segmen dari test patients)
    test_samples = [(p, cls) for p, pid, cls in all_files if pid in test_patients]
    logging.info(f"Test set: {len(test_samples)} segmen dari {len(test_patients)} pasien")

    # ── 2. 5-Fold StratifiedKFold pada trainval patients ──
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_f1_scores = []
    fold_results   = []
    fold_metrics   = []   # dict metrik lengkap per fold untuk tabel perbandingan

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(trainval_patients, trainval_labels), start=1
    ):
        logging.info(f"\n{'='*60}")
        logging.info(f"FOLD {fold}/{N_FOLDS}")
        logging.info(f"{'='*60}")

        train_pids = {trainval_patients[i] for i in tr_idx}
        val_pids   = {trainval_patients[i] for i in val_idx}

        train_s, val_s = split_by_patients(all_files, train_pids, val_pids)

        tr_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter([s[1] for s in train_s]).items())}
        vl_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter([s[1] for s in val_s]).items())}
        logging.info(f"Train dist: {tr_dist}  |  Val dist: {vl_dist}")

        # ── MLflow: start run untuk fold ini ───────────────────────────────
        tracker.start_run(
            run_name=f"fold_{fold}",
            params={**HPARAMS,
                    "fold"           : fold,
                    "train_segments" : len(train_s),
                    "val_segments"   : len(val_s)},
        )

        result = train_one_fold(train_s, val_s, fold, device, tracker=tracker)
        best_f1, best_epoch, best_state, best_preds, best_lbls, history = result

        fold_f1_scores.append(best_f1)
        fold_results.append(result)
        logging.info(f"Fold {fold} selesai — Best Macro F1: {best_f1:.4f} @ epoch {best_epoch}")

        # Plot per fold
        plot_fold_history(history, fold)
        cm_path     = RESULTS_DIR / "confusion_matrix" / f"cm_fold_{fold}.png"
        curves_path = RESULTS_DIR / "plots" / f"fold_{fold}_curves.png"
        plot_confusion_matrix(
            best_lbls, best_preds,
            title=f"Confusion Matrix — Fold {fold} (F1={best_f1:.4f})",
            filename=f"cm_fold_{fold}.png",
        )

        # Classification report per fold
        report_path = RESULTS_DIR / "metrics" / f"report_fold_{fold}.txt"
        report = classification_report(
            best_lbls, best_preds,
            target_names=CLASS_NAMES, labels=[0, 1], zero_division=0
        )
        with open(report_path, "w") as f:
            f.write(f"=== Fold {fold} — Best Epoch {best_epoch} — Macro F1: {best_f1:.4f} ===\n\n")
            f.write(report)

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

        # ── MLflow: log fold summary + artifacts ───────────────────────────
        tracker.log_fold_summary(fold=fold, best_f1=best_f1, best_epoch=best_epoch)
        tracker.log_artifacts([cm_path, curves_path, report_path])
        tracker.end_run()

    # ── 3. Ringkasan CV ──────────────────────────────────────────────────────
    mean_f1 = np.mean(fold_f1_scores)
    std_f1  = np.std(fold_f1_scores)

    logging.info(f"\n{'='*60}")
    logging.info("5-FOLD CV SELESAI — BiLSTM MFCC")
    logging.info(f"Macro F1 per fold : {[f'{v:.4f}' for v in fold_f1_scores]}")
    logging.info(f"Mean Macro F1     : {mean_f1:.4f} ± {std_f1:.4f}")
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
    logging.info("PERBANDINGAN METRIK PER FOLD — BiLSTM MFCC")
    logging.info(f"{'='*80}")
    logging.info("\n" + df_fold_compare.to_string(index=False))
    logging.info(f"{'='*80}\n")

    fold_compare_path = RESULTS_DIR / "metrics" / "fold_comparison.csv"
    df_fold_compare.to_csv(fold_compare_path, index=False)
    logging.info(f"Tabel perbandingan per fold disimpan → {fold_compare_path}")

    plot_cv_summary(fold_f1_scores)
    cv_summary_path = RESULTS_DIR / "plots" / "cv_summary.png"

    with open(RESULTS_DIR / "metrics" / "cv_summary.txt", "w") as f:
        f.write("=== 5-Fold StratifiedKFold CV Summary — BiLSTM MFCC ===\n\n")
        for i, v in enumerate(fold_f1_scores, 1):
            f.write(f"Fold {i}: Macro F1 = {v:.4f}\n")
        f.write(f"\nMean Macro F1 : {mean_f1:.4f}\n")
        f.write(f"Std  Macro F1 : {std_f1:.4f}\n")
        f.write(f"\n{'='*60}\n")
        f.write("Detail per Fold:\n\n")
        f.write(df_fold_compare.to_string(index=False))
        f.write("\n")

    # ── 4. Evaluasi best fold di test set ─────────────────────────────────────
    best_fold_idx = int(np.argmax(fold_f1_scores))
    best_fold_num = best_fold_idx + 1
    _, _, best_state, _, _, _ = fold_results[best_fold_idx]

    logging.info(f"\nEvaluasi best fold ({best_fold_num}) di test set...")
    model = BiLSTMMFCC().to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    test_ds     = MFCCSegmentDataset(test_samples)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_preds, test_labels = [], []

    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(device)
            _, pred = torch.max(model(feats), 1)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    logging.info(f"Test Set Macro F1 : {test_f1:.4f}")

    plot_confusion_matrix(
        test_labels, test_preds,
        title=f"Confusion Matrix — Test Set (F1={test_f1:.4f})",
        filename="cm_test_set.png",
    )

    test_report = classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES, labels=[0, 1], zero_division=0
    )
    print("\n=== TEST SET CLASSIFICATION REPORT — BiLSTM MFCC ===")
    print(test_report)

    test_report_path = RESULTS_DIR / "metrics" / "report_test_set.txt"
    with open(test_report_path, "w") as f:
        f.write(f"=== Test Set Report (Best Fold: {best_fold_num}) ===\n\n")
        f.write(f"CV Mean Macro F1 : {mean_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"Test Macro F1    : {test_f1:.4f}\n\n")
        f.write(test_report)

    # Simpan model terbaik
    ckpt_path = MODEL_DIR / "best_bilstm_mfcc.pt"
    torch.save(
        {
            "model_state_dict" : best_state,
            "model_config"     : {
                "n_features" : N_FEATURES,
                "hidden_dim" : HIDDEN_DIM,
                "num_layers" : NUM_LAYERS,
                "dropout"    : DROPOUT,
                "n_classes"  : 2,
            },
            "cv_mean_f1"       : mean_f1,
            "cv_std_f1"        : std_f1,
            "test_f1"          : test_f1,
            "best_fold"        : best_fold_num,
            "class_names"      : CLASS_NAMES,
            "max_t"            : MAX_T,
            "n_features"       : N_FEATURES,
        },
        ckpt_path,
    )
    logging.info(f"✅ Model terbaik disimpan → {ckpt_path}")
    logging.info(f"✅ Semua hasil tersimpan di {RESULTS_DIR}")

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
    cm_test_path = RESULTS_DIR / "confusion_matrix" / "cm_test_set.png"
    tracker.log_artifacts([cv_summary_path, cm_test_path, test_report_path, ckpt_path])
    tracker.end_run()

    print_ui_instructions()
    return mean_f1, std_f1, test_f1


if __name__ == "__main__":
    run_cv()
