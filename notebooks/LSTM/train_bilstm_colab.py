"""
LSTM – Colab Training Script (All-in-one)
==========================================
Project : Mental Health Audio Classification (NORMAL vs DEPRESI)
Script  : notebooks/LSTM/train_bilstm_colab.py

Didesain untuk Google Colab:
  - Self-contained: tidak perlu import dari file lain.
  - Langsung install dependensi yang dibutuhkan.
  - Mount Google Drive dan load dataset dari Drive.
  - Simpan checkpoint dan hasil ke Drive.

Usage di Colab:
  1. Upload file ini ke Google Drive atau copy paste ke Colab cell.
  2. Pastikan data/cleaned/ sudah ada di Drive.
  3. Jalankan semua cell secara berurutan.
"""

# ============================================================
# CELL 1 — Install & Import
# ============================================================
# !pip install -q librosa soundfile tqdm seaborn scikit-learn

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
import librosa
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CELL 2 — Mount Google Drive (jika di Colab)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================
# CELL 3 — Path Configuration
# ============================================================
# Sesuaikan path ini dengan lokasi dataset di Drive Anda
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # Ubah ini sesuai struktur folder di Google Drive Anda
    PROJECT_ROOT = Path("/content/drive/MyDrive/menthealth-ai")
else:
    # Untuk local development
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLEANED_DIR  = PROJECT_ROOT / "data" / "cleaned"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
MFCC_OUT_DIR = PROJECT_ROOT / "data" / "features" / "mfcc"
RESULTS_DIR  = PROJECT_ROOT / "results" / "lstm"
MODEL_DIR    = PROJECT_ROOT / "models" / "dl" / "lstm"

for d in [MFCC_OUT_DIR / "NORMAL", MFCC_OUT_DIR / "DEPRESI",
          RESULTS_DIR / "metrics", RESULTS_DIR / "plots",
          RESULTS_DIR / "confusion_matrix", MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Project root : {PROJECT_ROOT}")
print(f"Cleaned dir  : {CLEANED_DIR} (exists: {CLEANED_DIR.exists()})")
print(f"MFCC out dir : {MFCC_OUT_DIR}")

# ============================================================
# CELL 4 — Konfigurasi Global
# ============================================================
TARGET_SR    = 16000
N_MFCC       = 40
HOP_LENGTH   = 512
N_FFT        = 1024
SEG_SEC      = 10.0
MIN_SEG_SEC  = 4.0
MAX_T        = 313      # frames untuk 10 detik @ sr=16000, hop=512
N_FEATURES   = N_MFCC * 3    # MFCC + delta + delta2

CLASS_NAMES  = ["NORMAL", "DEPRESI"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

N_FOLDS      = 5
EPOCHS       = 80
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT      = 0.35
RANDOM_STATE = 42
PATIENCE_ES  = 25
PATIENCE_LR  = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")

# ============================================================
# CELL 5 — MFCC Extraction (jalankan hanya sekali)
# ============================================================
def extract_mfcc_from_segment(y_seg, sr):
    mfcc   = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=N_MFCC,
                                   hop_length=HOP_LENGTH, n_fft=N_FFT)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta1, delta2], axis=0)
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True) + 1e-8
    features = (features - mean) / std
    return features.T.astype(np.float32)


def run_extraction(force=False):
    """Ekstrak MFCC jika belum ada. Set force=True untuk re-extract."""
    normal_count  = len(list((MFCC_OUT_DIR / "NORMAL").glob("*.npy")))
    depresi_count = len(list((MFCC_OUT_DIR / "DEPRESI").glob("*.npy")))

    if (normal_count + depresi_count) > 0 and not force:
        print(f"✅ MFCC sudah ada: NORMAL={normal_count}, DEPRESI={depresi_count}")
        print("   Gunakan force=True untuk re-extract.")
        return

    label_path = SPLITS_DIR / "custom_2class_labels.csv"
    df = pd.read_csv(label_path)
    label_map = dict(zip(df["participant_id"].astype(str), df["label_depresi"].astype(int)))

    wav_files = sorted(CLEANED_DIR.glob("*.wav"))
    print(f"Mulai ekstraksi MFCC dari {len(wav_files)} file WAV...")

    total_saved = 0
    for wav_path in tqdm(wav_files, desc="Extracting"):
        pid = wav_path.stem
        if pid not in label_map:
            continue

        label_int  = label_map[pid]
        label_name = "DEPRESI" if label_int == 1 else "NORMAL"
        out_dir    = MFCC_OUT_DIR / label_name

        try:
            y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
            total_sec  = len(y) / sr
            seg_samples = int(SEG_SEC * sr)
            min_samples  = int(MIN_SEG_SEC * sr)
            n_segs = int(total_sec // SEG_SEC)

            for seg_idx in range(n_segs + 1):
                start = seg_idx * seg_samples
                if start >= len(y): break
                y_seg = y[start:start + seg_samples]
                if len(y_seg) < min_samples: continue
                if len(y_seg) < seg_samples:
                    y_seg = np.pad(y_seg, (0, seg_samples - len(y_seg)), mode="constant")

                feat = extract_mfcc_from_segment(y_seg, sr)
                np.save(str(out_dir / f"{pid}_seg{seg_idx:03d}.npy"), feat)
                total_saved += 1
        except Exception as e:
            print(f"  ❌ [{pid}] Error: {e}")

    n = len(list((MFCC_OUT_DIR / "NORMAL").glob("*.npy")))
    d = len(list((MFCC_OUT_DIR / "DEPRESI").glob("*.npy")))
    print(f"\n✅ Ekstraksi selesai! NORMAL={n}, DEPRESI={d}, Total={n+d}")


# Jalankan ekstraksi (hanya jika belum ada .npy)
run_extraction(force=False)

# ============================================================
# CELL 6 — Dataset & DataLoader
# ============================================================
FEATURE_CACHE = {}

class MFCCSegmentDataset(Dataset):
    def __init__(self, samples, max_t=MAX_T):
        self.samples = samples
        self.labels  = [s[1] for s in samples]
        self.max_t   = max_t

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        path_str = str(npy_path)

        if path_str in FEATURE_CACHE:
            feat = FEATURE_CACHE[path_str]
        else:
            feat = np.load(path_str)
            if feat.shape[0] > self.max_t:
                feat = feat[:self.max_t, :]
            elif feat.shape[0] < self.max_t:
                feat = np.pad(feat, ((0, self.max_t - feat.shape[0]), (0, 0)), mode="constant")
            if feat.shape[1] < N_FEATURES:
                feat = np.pad(feat, ((0, 0), (0, N_FEATURES - feat.shape[1])), mode="constant")
            elif feat.shape[1] > N_FEATURES:
                feat = feat[:, :N_FEATURES]
            feat = torch.tensor(feat, dtype=torch.float32)
            FEATURE_CACHE[path_str] = feat

        return feat, torch.tensor(label, dtype=torch.long)


def make_weighted_loader(samples, batch_size, shuffle=True):
    ds = MFCCSegmentDataset(samples)
    if shuffle:
        labels_t = torch.tensor(ds.labels, dtype=torch.long)
        counts   = torch.bincount(labels_t, minlength=2).float().clamp(min=1)
        weights  = 1.0 / counts[labels_t]
        weights  = weights / weights.sum()
        sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=2,
                          pin_memory=torch.cuda.is_available())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                      pin_memory=torch.cuda.is_available())

# ============================================================
# CELL 7 — Model
# ============================================================
class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class BiLSTMMFCC(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, n_classes=2, dropout=DROPOUT):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.lstm     = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        lstm_out = hidden_dim * 2
        self.attn_pool = AttentionPool(lstm_out)
        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        B, T, F = x.shape
        x = self.input_bn(x.reshape(-1, F)).reshape(B, T, F)
        out, _ = self.lstm(x)
        return self.fc(self.attn_pool(out))

    def get_attention_weights(self, x):
        B, T, F = x.shape
        x = self.input_bn(x.reshape(-1, F)).reshape(B, T, F)
        out, _ = self.lstm(x)
        return torch.softmax(self.attn_pool.attn(out), dim=1).squeeze(-1)


n_params = sum(p.numel() for p in BiLSTMMFCC().parameters())
print(f"BiLSTM parameters: {n_params:,}")

# ============================================================
# CELL 8 — Data Loading
# ============================================================
def load_patient_data():
    df = pd.read_csv(SPLITS_DIR / "custom_2class_labels.csv")
    p2l = dict(zip(df["participant_id"].astype(str),
                   df["label_depresi"].astype(int).map({0: "NORMAL", 1: "DEPRESI"})))

    all_files, avail = [], set()
    for cls, idx in CLASS_TO_IDX.items():
        cls_dir = MFCC_DIR = MFCC_OUT_DIR / cls
        if not cls_dir.exists(): continue
        for p in sorted(cls_dir.glob("*.npy")):
            pid = p.stem.split("_seg")[0]
            if pid in p2l:
                all_files.append((p, pid, idx)); avail.add(pid)

    plist  = sorted(avail)
    plbls  = [CLASS_TO_IDX[p2l[p]] for p in plist]
    dist   = Counter(plbls)
    print(f"Patients: {len(plist)} | NORMAL={dist[0]} DEPRESI={dist[1]} | Segments={len(all_files)}")
    return plist, plbls, all_files


plist, plbls, all_files = load_patient_data()

# ============================================================
# CELL 9 — Training Functions
# ============================================================
def train_one_fold(train_s, val_s, fold_num, device, epochs=EPOCHS):
    train_loader = make_weighted_loader(train_s, BATCH_SIZE, shuffle=True)
    val_loader   = make_weighted_loader(val_s,   BATCH_SIZE, shuffle=False)

    model    = BiLSTMMFCC().to(device)
    lbl_t    = torch.tensor([s[1] for s in train_s], dtype=torch.long)
    counts   = torch.bincount(lbl_t, minlength=2).float().clamp(min=1)
    cw       = (1.0 / counts / (1.0 / counts).sum() * 2).to(device)
    crit     = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    opt      = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup   = optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=5)
    plateau  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=PATIENCE_LR, min_lr=1e-6)

    hist = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","val_f1",
                              "rec_normal","rec_depresi"]}
    best_f1, best_epoch, best_state = -1.0, 0, None
    best_preds, best_lbls, no_imp = [], [], 0

    for ep in range(epochs):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for feats, lbls in tqdm(train_loader, desc=f"Fold{fold_num} Ep{ep+1}", leave=False):
            feats, lbls = feats.to(device), lbls.to(device)
            opt.zero_grad()
            out  = model(feats); loss = crit(out, lbls)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); tc += (out.argmax(1) == lbls).sum().item(); tt += lbls.size(0)

        model.eval(); vl, vc, vt = 0.0, 0, 0; ap, al = [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats, lbls = feats.to(device), lbls.to(device)
                out = model(feats); vl += crit(out, lbls).item()
                p   = out.argmax(1); vc += (p == lbls).sum().item(); vt += lbls.size(0)
                ap.extend(p.cpu().numpy()); al.extend(lbls.cpu().numpy())

        f1  = f1_score(al, ap, average="macro", zero_division=0)
        _, rec, _, _ = precision_recall_fscore_support(al, ap, labels=[0,1], zero_division=0)

        for k, v in zip(hist.keys(), [tl/max(1,len(train_loader)), vl/max(1,len(val_loader)),
                                       100*tc/max(1,tt), 100*vc/max(1,vt), f1, rec[0], rec[1]]):
            hist[k].append(v)

        print(f"[F{fold_num}] Ep{ep+1:02d} | TLoss={tl/len(train_loader):.4f} "
              f"TAcc={100*tc/tt:.1f}% | VLoss={vl/len(val_loader):.4f} "
              f"VAcc={100*vc/vt:.1f}% F1={f1:.4f} | Rec N:{rec[0]:.3f} D:{rec[1]:.3f}")

        if f1 > best_f1:
            best_f1 = f1; best_epoch = ep+1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = list(ap); best_lbls = list(al); no_imp = 0
            print(f"  ✅ Best F1: {best_f1:.4f} @ epoch {best_epoch}")
        else:
            no_imp += 1

        if ep < 5: warmup.step()
        else: plateau.step(f1)
        if no_imp >= PATIENCE_ES:
            print(f"  ⏹ Early stop @ epoch {ep+1}"); break

    return best_f1, best_epoch, best_state, best_preds, best_lbls, hist

# ============================================================
# CELL 10 — Run 5-Fold CV
# ============================================================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_STATE)
tv_idx, test_idx = next(sss.split(plist, plbls))

tv_p    = [plist[i] for i in tv_idx]
tv_l    = [plbls[i] for i in tv_idx]
test_p  = {plist[i] for i in test_idx}
test_s  = [(p, c) for p, pid, c in all_files if pid in test_p]
print(f"Test set: {len(test_s)} segmen dari {len(test_p)} pasien")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_f1s, fold_res = [], []

for fold, (tr_i, val_i) in enumerate(skf.split(tv_p, tv_l), 1):
    print(f"\n{'='*50}\nFOLD {fold}/{N_FOLDS}\n{'='*50}")
    tr_p  = {tv_p[i] for i in tr_i}
    vl_p  = {tv_p[i] for i in val_i}
    tr_s  = [(p, c) for p, pid, c in all_files if pid in tr_p]
    vl_s  = [(p, c) for p, pid, c in all_files if pid in vl_p]
    print(f"Train: {len(tr_s)} segs | Val: {len(vl_s)} segs")

    res = train_one_fold(tr_s, vl_s, fold, DEVICE)
    f1, ep, state, preds, lbls, hist = res
    fold_f1s.append(f1); fold_res.append(res)
    print(f"Fold {fold} done — Best F1: {f1:.4f} @ epoch {ep}")

    # Simpan checkpoint per fold
    ckpt = MODEL_DIR / f"fold_{fold}_bilstm.pt"
    torch.save({"state": state, "f1": f1, "epoch": ep}, ckpt)

mean_f1 = np.mean(fold_f1s)
std_f1  = np.std(fold_f1s)
print(f"\n{'='*50}")
print(f"5-FOLD CV SELESAI")
print(f"F1 per fold: {[f'{v:.4f}' for v in fold_f1s]}")
print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

# ============================================================
# CELL 11 — Evaluasi Test Set
# ============================================================
best_idx = int(np.argmax(fold_f1s))
_, _, best_state, _, _, _ = fold_res[best_idx]

model_final = BiLSTMMFCC().to(DEVICE)
model_final.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
model_final.eval()

test_ds  = MFCCSegmentDataset(test_s)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
tp, tl_  = [], []

with torch.no_grad():
    for feats, lbls in test_dl:
        _, pred = torch.max(model_final(feats.to(DEVICE)), 1)
        tp.extend(pred.cpu().numpy()); tl_.extend(lbls.numpy())

test_f1 = f1_score(tl_, tp, average="macro", zero_division=0)
print(f"\nTest Macro F1: {test_f1:.4f}")
print(classification_report(tl_, tp, target_names=CLASS_NAMES, zero_division=0))

# Simpan model final
torch.save({
    "model_state_dict": best_state,
    "model_config": {"n_features": N_FEATURES, "hidden_dim": HIDDEN_DIM,
                     "num_layers": NUM_LAYERS, "dropout": DROPOUT, "n_classes": 2},
    "cv_mean_f1": mean_f1, "cv_std_f1": std_f1, "test_f1": test_f1,
    "best_fold": best_idx + 1, "class_names": CLASS_NAMES,
    "max_t": MAX_T, "n_features": N_FEATURES,
}, MODEL_DIR / "best_bilstm_mfcc.pt")
print(f"✅ Model saved → {MODEL_DIR / 'best_bilstm_mfcc.pt'}")

# ============================================================
# CELL 12 — Confusion Matrix & Plots
# ============================================================
cm = confusion_matrix(tl_, tp, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Confusion Matrix — Test Set (BiLSTM MFCC, F1={test_f1:.4f})")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix" / "cm_test_bilstm.png", dpi=150)
plt.show()

# CV Summary bar chart
folds = [f"Fold {i+1}" for i in range(N_FOLDS)]
plt.figure(figsize=(8, 5))
bars = plt.bar(folds, fold_f1s, color="steelblue", alpha=0.8, edgecolor="black")
plt.axhline(mean_f1, color="tomato", linestyle="--", linewidth=2, label=f"Mean={mean_f1:.4f}")
for bar, v in zip(bars, fold_f1s):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)
plt.title("5-Fold CV — Macro F1 (BiLSTM MFCC)"); plt.ylabel("Macro F1"); plt.ylim(0, 1.05)
plt.legend(); plt.tight_layout()
plt.savefig(RESULTS_DIR / "plots" / "cv_summary_bilstm.png", dpi=150)
plt.show()
print("✅ Semua plot disimpan!")
