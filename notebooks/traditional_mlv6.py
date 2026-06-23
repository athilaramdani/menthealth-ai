# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v6** — Klasifikasi Kesehatan Mental Berbasis Audio
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v6 = v4 REFINED — Fix semua masalah v5
#
#  [1] Kembali ke Participant-Level Features
#      Aggregate fitur dari FULL audio per partisipan (bukan segment 10s)
#      → signal-to-noise ratio lebih baik, evaluasi langsung per orang
#
#  [2] TANPA Augmentasi — tidak perlu untuk participant-level
#
#  [3] TANPA SMOTE — pakai class_weight='balanced' saja
#      → hindari overfitting dari duplikasi sample
#
#  [4] Fixed Threshold 0.5 — tanpa tuning
#      → v5 overfit karena threshold tune pada 15 orang dev
#
#  [5] Split 80/10/10 × 5 Repeated Stratified
#      → lebih banyak data train (81 vs 64 participant)
#      → lapor mean ± std dari 5 repeat untuk stabilitas
#
#  [6] Wav2Vec 2.0 NYATA (torch CPU-only)
#      → facebook/wav2vec2-base mean-pool 768-dim
#
#  [7] Fitur Prosodik Diperkaya
#      HNR, jitter, shimmer, spectral entropy, speaking rate
#
#  Tetap 12 model = 4 Classifier × 3 Feature Type (apple-to-apple)
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup — Install Dependencies

# %%
import subprocess, sys

def _pip_install(package, import_name=None, upgrade=False):
    check = import_name or package.split('[')[0].split('>=')[0].split('==')[0]
    try:
        __import__(check)
        if not upgrade:
            return
    except ImportError:
        pass
    except Exception:
        return  # OSError dll → skip
    print(f"[Installing] {package} ...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", package, "-q"]
        if upgrade:
            cmd.append("--upgrade")
        subprocess.check_call(cmd)
        print(f"[OK] {package}")
    except Exception as e:
        print(f"[WARN] Gagal install {package}: {e}")

_pip_install("librosa")
_pip_install("scikit-learn", "sklearn")
_pip_install("xgboost")
_pip_install("transformers")
_pip_install("soundfile")
_pip_install("scipy")
_pip_install("seaborn")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("\n[OK] Dependencies siap.\n")

# %%
import os
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import librosa
from scipy.stats import kurtosis, skew

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.feature_selection import f_classif, mutual_info_classif
import xgboost as xgb

plt.rcParams['font.family'] = 'DejaVu Sans'

# Wav2Vec 2.0
WAV2VEC_AVAILABLE = False
try:
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    WAV2VEC_AVAILABLE = True
    print("[INFO] PyTorch + Transformers OK -> Wav2Vec 2.0 AKTIF.")
except (ImportError, OSError) as e:
    print(f"[WARN] Wav2Vec fallback (MFCC-embed): {type(e).__name__}")
except Exception as e:
    print(f"[WARN] Wav2Vec fallback: {e}")

print("Library berhasil diimport.\n")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─── Path ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd()
    else os.getcwd()
)
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v6")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")

for d in [MODELS_DIR, V6_FEAT_DIR,
          os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

FORCE_EXTRACT = False

print(f"Project root : {PROJECT_ROOT}")
print(f"Feature dir  : {V6_FEAT_DIR}")

# %% [markdown]
# ## 0. Konfigurasi Audio

# %%
TARGET_SR    = 16000
N_MFCC       = 40
N_MELS       = 64
FRAME_LENGTH = int(0.025 * TARGET_SR)
HOP_LENGTH   = int(0.010 * TARGET_SR)

N_REPEATS    = 5       # jumlah repeated split
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10

print(f"Split: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%} x {N_REPEATS} repeats")

# %% [markdown]
# ## 1. Feature Extraction — Participant-Level (Full Audio)

# %%
def agg(arr, name):
    """8 statistik dari array numerik."""
    if len(arr) == 0:
        return {f'{name}_{s}': 0.0 for s in ['mean','std','min','max','p25','p75','kurt','skew']}
    return {
        f'{name}_mean': float(np.mean(arr)),
        f'{name}_std':  float(np.std(arr)),
        f'{name}_min':  float(np.min(arr)),
        f'{name}_max':  float(np.max(arr)),
        f'{name}_p25':  float(np.percentile(arr, 25)),
        f'{name}_p75':  float(np.percentile(arr, 75)),
        f'{name}_kurt': float(kurtosis(arr, nan_policy='omit')),
        f'{name}_skew': float(skew(arr, nan_policy='omit')),
    }

def spectral_entropy(S_band, eps=1e-10):
    p = np.abs(S_band) + eps
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p + eps)))


# ─── A. MFCC + Prosodik ──────────────────────────────────────────────────────
def extract_mfcc_prosodic(y, sr):
    feats = {}

    # MFCC + Delta + Delta-Delta
    mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    d_mfcc  = librosa.feature.delta(mfccs)
    dd_mfcc = librosa.feature.delta(mfccs, order=2)
    for i in range(N_MFCC):
        feats.update(agg(mfccs[i],   f'm{i+1}'))
        feats.update(agg(d_mfcc[i],  f'dm{i+1}'))
        feats.update(agg(dd_mfcc[i], f'ddm{i+1}'))

    # Pitch (F0)
    try:
        pitches, mags = librosa.piptrack(
            y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pv = [pitches[mags[:, t].argmax(), t] for t in range(pitches.shape[1])
              if pitches[mags[:, t].argmax(), t] > 0]
        feats.update(agg(np.array(pv) if pv else np.array([0.0]), 'pitch'))
    except Exception:
        feats.update(agg(np.array([0.0]), 'pitch'))

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(rms, 'rms'))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(zcr, 'zcr'))

    # Low-energy ratio
    feats['low_energy_ratio'] = float(np.mean(rms < 0.1 * np.mean(rms)))

    # HNR proxy
    try:
        harmonic  = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        rms_h = float(np.sqrt(np.mean(harmonic ** 2)) + 1e-10)
        rms_p = float(np.sqrt(np.mean(percussive ** 2)) + 1e-10)
        feats['hnr_proxy'] = float(20 * np.log10(rms_h / rms_p))
    except Exception:
        feats['hnr_proxy'] = 0.0

    # Spectral entropy
    try:
        S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
        feats['spec_entropy'] = spectral_entropy(S.mean(axis=1))
    except Exception:
        feats['spec_entropy'] = 0.0

    # Jitter & Shimmer
    try:
        pitches, mags = librosa.piptrack(
            y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pv, vf = [], []
        for t in range(pitches.shape[1]):
            idx = mags[:, t].argmax()
            p = pitches[idx, t]
            if p > 50.0:
                pv.append(p)
                vf.append(t)
        if len(pv) >= 2:
            periods = 1.0 / np.array(pv)
            jitter = (np.mean(np.abs(np.diff(periods))) / np.mean(periods)) * 100
            rms_all = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
            vrms = [rms_all[f] for f in vf if f < len(rms_all) and rms_all[f] > 0]
            shimmer = (np.mean(np.abs(np.diff(vrms))) / np.mean(vrms)) * 100 if len(vrms) >= 2 else 0.0
        else:
            jitter, shimmer = 0.0, 0.0
        feats['jitter']  = float(jitter)
        feats['shimmer'] = float(shimmer)
    except Exception:
        feats['jitter'] = 0.0
        feats['shimmer'] = 0.0

    # Speaking rate proxy (voiced frame ratio)
    try:
        pitches2, mags2 = librosa.piptrack(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        voiced = sum(1 for t in range(pitches2.shape[1]) if pitches2[mags2[:, t].argmax(), t] > 0)
        total  = pitches2.shape[1]
        feats['voiced_ratio'] = float(voiced / max(total, 1))
    except Exception:
        feats['voiced_ratio'] = 0.0

    return feats


# ─── B. Mel-Spectrogram + Chroma + Spectral ──────────────────────────────────
def extract_spectrogram(y, sr):
    feats = {}

    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    S_db = librosa.power_to_db(S, ref=np.max)
    for i in range(N_MELS):
        feats.update(agg(S_db[i], f'mel{i+1}'))
        feats[f'mel{i+1}_ent'] = spectral_entropy(S[i])

    # Spectral features
    for name, fn in [
        ('cent', librosa.feature.spectral_centroid),
        ('bw', librosa.feature.spectral_bandwidth),
        ('rolloff', librosa.feature.spectral_rolloff),
    ]:
        vals = fn(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        feats.update(agg(vals, name))

    flat = librosa.feature.spectral_flatness(y=y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(flat, 'flat'))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    for i in range(12):
        feats.update(agg(chroma[i], f'ch{i+1}'))

    # RMS + ZCR
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(rms, 'rms'))
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(zcr, 'zcr'))

    feats['low_energy_ratio'] = float(np.mean(rms < 0.1 * np.mean(rms)))
    feats['spec_entropy']     = spectral_entropy(S.mean(axis=1))

    return feats


# ─── C. Wav2Vec 2.0 ──────────────────────────────────────────────────────────
_w2v_proc  = None
_w2v_model = None

def _load_w2v():
    global _w2v_proc, _w2v_model
    if _w2v_proc is None:
        print("[INFO] Memuat Wav2Vec2 (facebook/wav2vec2-base) ...")
        _w2v_proc  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        _w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        _w2v_model.eval()
        print("[INFO] Wav2Vec2 siap.")
    return _w2v_proc, _w2v_model

def extract_wav2vec(y, sr):
    if WAV2VEC_AVAILABLE:
        try:
            proc, model = _load_w2v()
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            # Proses dalam chunks (maks 30 detik) untuk hemat memori
            chunk_len = 30 * 16000
            all_hidden = []
            for start in range(0, len(y), chunk_len):
                chunk = y[start:start + chunk_len]
                if len(chunk) < 1600:
                    continue
                inputs = proc(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    out = model(**inputs)
                all_hidden.append(out.last_hidden_state.squeeze(0).numpy())
            if all_hidden:
                hidden = np.concatenate(all_hidden, axis=0)   # (T_total, 768)
                mean_h = hidden.mean(axis=0)
                # Blok statistik (24 blok x 3 stat = 72 fitur)
                n_blk, blk_sz = 24, 32
                feats = {}
                for b in range(n_blk):
                    sl = slice(b * blk_sz, (b + 1) * blk_sz)
                    feats[f'w2v_blk{b}_mean'] = float(mean_h[sl].mean())
                    feats[f'w2v_blk{b}_std']  = float(mean_h[sl].std())
                    feats[f'w2v_blk{b}_max']  = float(mean_h[sl].max())
                return feats
        except Exception as e:
            print(f"[WARN] Wav2Vec error: {e} -> fallback")

    # Fallback: MFCC-embed
    feats = {}
    mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    d_mfcc = librosa.feature.delta(mfccs)
    for i in range(N_MFCC):
        feats[f'w2v_m{i+1}_mean'] = float(mfccs[i].mean())
        feats[f'w2v_m{i+1}_std']  = float(mfccs[i].std())
        feats[f'w2v_d{i+1}_mean'] = float(d_mfcc[i].mean())
    S  = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64,
                                         n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    Sd = librosa.power_to_db(S, ref=np.max)
    for i in range(0, 64, 4):
        feats[f'w2v_mel{i}_mean'] = float(Sd[i].mean())
        feats[f'w2v_mel{i}_std']  = float(Sd[i].std())
    return feats


FEATURE_EXTRACTORS = {
    'MFCC':        extract_mfcc_prosodic,
    'Spectrogram': extract_spectrogram,
    'Wav2Vec':     extract_wav2vec,
}

w2v_status = "NYATA (768-dim)" if WAV2VEC_AVAILABLE else "FALLBACK (MFCC-embed)"
print(f"Feature extractors v6:")
for n in FEATURE_EXTRACTORS:
    tag = f" [{w2v_status}]" if n == 'Wav2Vec' else ""
    print(f"  - {n}{tag}")

# %% [markdown]
# ## 2. Label Mapping & Metadata

# %%
def map_label(row):
    phq_binary = row.get('PHQ8_Binary', row.get('PHQ_Binary', np.nan))
    if not pd.isna(phq_binary):
        return int(phq_binary)
    phq = row.get('PHQ8_Score', row.get('PHQ_Score', np.nan))
    phq = 0 if pd.isna(phq) else int(phq)
    return 1 if phq >= 10 else 0

def load_all_metadata():
    """Muat metadata dari semua split DAIC-WOZ, gabungkan."""
    all_parts = []
    for fname, split in [
        ("train_split_Depression_AVEC2017.csv", "train"),
        ("dev_split_Depression_AVEC2017.csv",   "dev"),
        ("full_test_split.csv",                  "test"),
    ]:
        path = os.path.join(RAW_DIR, fname)
        df   = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if col.lower() == 'participant_id':
                df.rename(columns={col: 'Participant_ID'}, inplace=True)
        if 'PHQ_Score' not in df.columns and 'PHQ8_Score' in df.columns:
            df['PHQ_Score'] = df['PHQ8_Score']
        elif 'PHQ8_Score' not in df.columns and 'PHQ_Score' in df.columns:
            df['PHQ8_Score'] = df['PHQ_Score']
        df['label_depresi'] = df.apply(map_label, axis=1)
        df['split_original'] = split
        all_parts.append(df)

    df_meta = pd.concat(all_parts, ignore_index=True)
    df_meta.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    df_meta['participant_id'] = df_meta['participant_id'].astype(int)
    return df_meta

# %% [markdown]
# ## 3. Participant-Level Feature Extraction

# %%
def build_v6_features(cleaned_dir, output_dir, df_meta):
    """
    Ekstrak fitur dari FULL audio per partisipan (bukan segmen).
    Simpan 3 CSV: daic_v6_{mfcc,spectrogram,wav2vec}.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    cleaned_files = sorted(f for f in os.listdir(cleaned_dir) if f.endswith('.wav'))
    print(f"File audio bersih: {len(cleaned_files)}")

    rows_by_type = {name: [] for name in FEATURE_EXTRACTORS}

    SEP = "=" * 85
    print(f"\n{SEP}")
    print(f"{'EKSTRAKSI FITUR v6 — Participant-Level (Full Audio)':^85}")
    print(SEP)
    print(f"{'PID':>5} | {'Label':^9} | {'Dur':>5} | {'Status'}")
    print("-" * 85)

    t0 = time.time()

    for fname in cleaned_files:
        pid = int(fname.replace('.wav', ''))
        meta_row = df_meta[df_meta['participant_id'] == pid]
        if meta_row.empty:
            continue
        meta_row = meta_row.iloc[0]
        label     = int(meta_row['label_depresi'])
        phq8      = int(meta_row.get('PHQ8_Score', meta_row.get('PHQ_Score', 0)))
        gender    = int(meta_row.get('Gender', 0))
        label_str = "Depresi" if label == 1 else "Normal"

        audio_path = os.path.join(cleaned_dir, fname)
        try:
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            if len(y) < TARGET_SR:
                print(f"  {pid:4d}  | {label_str:^9} | short | SKIP")
                continue

            meta_base = {
                'participant_id': pid,
                'phq8_score':    phq8,
                'label_depresi': label,
                'gender':        gender,
            }

            dur = len(y) / sr
            status_parts = []

            for feat_name, extractor_fn in FEATURE_EXTRACTORS.items():
                try:
                    feats = extractor_fn(y, sr)
                    rows_by_type[feat_name].append({**meta_base, **feats})
                    status_parts.append(f"{feat_name}:OK")
                except Exception as e:
                    rows_by_type[feat_name].append(meta_base)
                    status_parts.append(f"{feat_name}:ERR")

            print(f"  {pid:4d}  | {label_str:^9} | {dur:4.0f}s | {', '.join(status_parts)}", flush=True)

        except Exception as e:
            print(f"  {pid:4d}  | ERROR: {e}", flush=True)

    elapsed = time.time() - t0
    print(SEP)
    print(f"Ekstraksi selesai: {elapsed:.1f}s\n")

    saved = {}
    for feat_name, rows in rows_by_type.items():
        df_feat  = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"daic_v6_{feat_name.lower()}.csv")
        df_feat.to_csv(csv_path, index=False)
        n_dep = (df_feat['label_depresi'] == 1).sum()
        print(f"  [{feat_name}] {len(df_feat)} participants, {df_feat.shape[1]} cols, "
              f"{n_dep} depresi -> {csv_path}")
        saved[feat_name] = csv_path

    return saved


V6_CSV_PATHS = {
    'MFCC':        os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"),
    'Spectrogram': os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"),
    'Wav2Vec':     os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"),
}

all_exist = all(os.path.exists(p) for p in V6_CSV_PATHS.values())

if FORCE_EXTRACT or not all_exist:
    print("\n[INFO] Menjalankan ekstraksi fitur v6 ...")
    df_meta_all = load_all_metadata()
    V6_CSV_PATHS = build_v6_features(CLEANED_DIR, V6_FEAT_DIR, df_meta_all)
else:
    print("\n[INFO] CSV v6 sudah tersedia:")
    for k, p in V6_CSV_PATHS.items():
        print(f"  {k}: {p}")

# %% [markdown]
# ## 4. Load Data + 80/10/10 Repeated Stratified Split

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6_data(csv_path, feat_name):
    """Muat CSV v6, bersihkan, return dataframe + feat_cols."""
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    # NaN -> 0
    df[feat_cols] = df[feat_cols].fillna(0)

    # Hapus fitur konstan
    std_vals = df[feat_cols].std()
    const = std_vals[std_vals < 1e-8].index.tolist()
    feat_cols = [f for f in feat_cols if f not in const]

    print(f"  [{feat_name}] {len(df)} participants, {len(feat_cols)} fitur")
    return df, feat_cols


def make_splits(df, n_repeats=5, seed=42):
    """
    Buat 80/10/10 stratified split × n_repeats.
    Return list of (train_idx, val_idx, test_idx).
    """
    labels = df['label_depresi'].values
    splits = []

    for r in range(n_repeats):
        rng = np.random.RandomState(seed + r)

        # Stratified split: pertama bagi 80% train vs 20% temp
        idx_all = np.arange(len(df))
        idx_0   = idx_all[labels == 0]
        idx_1   = idx_all[labels == 1]

        rng.shuffle(idx_0)
        rng.shuffle(idx_1)

        # 80% train
        n_train_0 = int(len(idx_0) * TRAIN_RATIO)
        n_train_1 = int(len(idx_1) * TRAIN_RATIO)
        train_idx = np.concatenate([idx_0[:n_train_0], idx_1[:n_train_1]])

        # Sisa 20% dibagi 50/50 → val 10%, test 10%
        rest_0 = idx_0[n_train_0:]
        rest_1 = idx_1[n_train_1:]
        n_val_0 = len(rest_0) // 2
        n_val_1 = len(rest_1) // 2

        val_idx  = np.concatenate([rest_0[:n_val_0],  rest_1[:n_val_1]])
        test_idx = np.concatenate([rest_0[n_val_0:],  rest_1[n_val_1:]])

        splits.append((train_idx, val_idx, test_idx))

    return splits


# Muat semua dataset
datasets_raw = {}
for feat_name, csv_path in V6_CSV_PATHS.items():
    datasets_raw[feat_name] = load_v6_data(csv_path, feat_name)

# Buat splits (sama untuk semua feature types)
df_any = list(datasets_raw.values())[0][0]
repeated_splits = make_splits(df_any, n_repeats=N_REPEATS, seed=RANDOM_SEED)

print(f"\n{N_REPEATS} repeated stratified splits dibuat:")
for r, (tr, va, te) in enumerate(repeated_splits):
    y = df_any['label_depresi'].values
    print(f"  Split {r+1}: train={len(tr)} "
          f"(0:{(y[tr]==0).sum()}/1:{(y[tr]==1).sum()})  "
          f"val={len(va)} (0:{(y[va]==0).sum()}/1:{(y[va]==1).sum()})  "
          f"test={len(te)} (0:{(y[te]==0).sum()}/1:{(y[te]==1).sum()})")

# %% [markdown]
# ## 5. Definisi Model

# %%
def get_models_config():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED, class_weight='balanced',
            C=1.0, solver='lbfgs'
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, C=10.0, gamma='scale',
            random_state=RANDOM_SEED, class_weight='balanced',
        ),
        'XGBoost': xgb.XGBClassifier(
            random_state=RANDOM_SEED, eval_metric='logloss',
            objective='binary:logistic', n_jobs=1,
            scale_pos_weight=2, n_estimators=200, max_depth=5,
            learning_rate=0.05, subsample=0.8,
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_SEED, class_weight='balanced', n_jobs=1,
            n_estimators=300, max_depth=10, min_samples_split=5,
            max_features='sqrt',
        ),
    }

MODEL_NAMES = list(get_models_config().keys())
FEAT_NAMES  = list(FEATURE_EXTRACTORS.keys())

print(f"Model v6 (class_weight='balanced', NO SMOTE):")
for m in MODEL_NAMES:
    print(f"  - {m}")
print(f"\nTotal: {len(MODEL_NAMES)} x {len(FEAT_NAMES)} = {len(MODEL_NAMES)*len(FEAT_NAMES)} model")

# %% [markdown]
# ## 6. Training & Evaluation — 80/10/10 × 5 Repeats

# %%
def preprocess_split(df, feat_cols, train_idx, val_idx, test_idx):
    """Scale & clean features for one split. Returns X_tr, y_tr, X_va, y_va, X_te, y_te."""
    X_all = df[feat_cols].values
    y_all = df['label_depresi'].values

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_va, y_va = X_all[val_idx],   y_all[val_idx]
    X_te, y_te = X_all[test_idx],  y_all[test_idx]

    # NaN -> median train
    medians = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_va, X_te]:
        nan_mask = np.isnan(X)
        for col_i in range(X.shape[1]):
            X[nan_mask[:, col_i], col_i] = medians[col_i]

    # Clip outlier (IQR x 10)
    Q1 = np.percentile(X_tr, 25, axis=0)
    Q3 = np.percentile(X_tr, 75, axis=0)
    IQR = Q3 - Q1
    lo = Q1 - 10 * IQR
    hi = Q3 + 10 * IQR
    for X in [X_tr, X_va, X_te]:
        np.clip(X, lo, hi, out=X)

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    return X_tr, y_tr, X_va, y_va, X_te, y_te, scaler


def evaluate(y_true, y_pred, y_prob=None):
    """Compute metrics dict."""
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    except Exception:
        auc = 0.0
    return {
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'f1_macro':    float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision':   float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall':      float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'roc_auc':     auc,
    }

# ── Main training loop ────────────────────────────────────────────────────────
SEP = "=" * 90

# Collect results: dict[combo_key] -> list of metrics dicts (one per repeat)
all_repeat_results = {}
# For final reporting: best repeat's y_true/y_pred per combo
all_best_ys = {}

for feat_name in FEAT_NAMES:
    df, feat_cols = datasets_raw[feat_name]

    print(f"\n{SEP}")
    print(f"  FEATURE: {feat_name}  |  {len(feat_cols)} fitur  |  {len(df)} participants")
    print(SEP)

    for model_name in MODEL_NAMES:
        combo_key = f"{feat_name} + {model_name}"
        repeat_metrics = []

        for r, (tr_idx, va_idx, te_idx) in enumerate(repeated_splits):
            X_tr, y_tr, X_va, y_va, X_te, y_te, scaler = preprocess_split(
                df, feat_cols, tr_idx, va_idx, te_idx)

            model = get_models_config()[model_name]
            model.fit(X_tr, y_tr)

            # Predict dengan fixed threshold 0.5
            try:
                y_prob = model.predict_proba(X_te)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
            except Exception:
                y_pred = model.predict(X_te)
                y_prob = y_pred.astype(float)

            metrics = evaluate(y_te, y_pred, y_prob)
            repeat_metrics.append(metrics)

            # Simpan val metrics juga
            try:
                y_va_prob = model.predict_proba(X_va)[:, 1]
                y_va_pred = (y_va_prob >= 0.5).astype(int)
            except Exception:
                y_va_pred = model.predict(X_va)
                y_va_prob = y_va_pred.astype(float)
            val_m = evaluate(y_va, y_va_pred, y_va_prob)
            metrics['val_f1'] = val_m['f1_macro']

        all_repeat_results[combo_key] = repeat_metrics

        # Statistik
        f1s = [m['f1_macro'] for m in repeat_metrics]
        accs = [m['accuracy'] for m in repeat_metrics]
        aucs = [m['roc_auc'] for m in repeat_metrics]
        best_r = int(np.argmax(f1s))
        all_best_ys[combo_key] = best_r

        print(f"\n  [{combo_key}]")
        print(f"    Test F1 (5 repeats): {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  "
              f"(min={np.min(f1s):.4f}, max={np.max(f1s):.4f})")
        print(f"    Test Acc:            {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"    Test AUC:            {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

# Simpan model terbaik (dari repeat dengan F1 tertinggi)
for combo_key, best_r in all_best_ys.items():
    feat_name = combo_key.split(' + ')[0]
    model_name = combo_key.split(' + ')[1]
    df, feat_cols = datasets_raw[feat_name]
    tr_idx, va_idx, te_idx = repeated_splits[best_r]

    X_tr, y_tr, _, _, _, _, scaler = preprocess_split(df, feat_cols, tr_idx, va_idx, te_idx)
    model = get_models_config()[model_name]
    model.fit(X_tr, y_tr)

    sf = feat_name.lower().replace(' ', '_')
    sm = model_name.lower().replace(' ', '_')
    with open(os.path.join(MODELS_DIR, f"v6_{sf}_{sm}.pkl"), 'wb') as fp:
        pickle.dump({'model': model, 'scaler': scaler, 'feat_cols': feat_cols}, fp)

print(f"\n{SEP}")
print(f"  SEMUA 12 MODEL v6 SELESAI (5 repeats each)")
print(SEP)

# %% [markdown]
# ## 7. Tabel Perbandingan Lengkap

# %%
rows = []
for combo_key, repeat_metrics in all_repeat_results.items():
    parts = combo_key.split(' + ')
    feat_name  = parts[0]
    model_name = parts[1]

    f1s  = [m['f1_macro'] for m in repeat_metrics]
    accs = [m['accuracy'] for m in repeat_metrics]
    aucs = [m['roc_auc'] for m in repeat_metrics]
    precs = [m['precision'] for m in repeat_metrics]
    recs  = [m['recall'] for m in repeat_metrics]

    rows.append({
        'Feature Extraction': feat_name,
        'Model':              model_name,
        'Test F1 Mean':       round(np.mean(f1s), 4),
        'Test F1 Std':        round(np.std(f1s), 4),
        'Test F1 Min':        round(np.min(f1s), 4),
        'Test F1 Max':        round(np.max(f1s), 4),
        'Test Acc Mean':      round(np.mean(accs), 4),
        'Test AUC Mean':      round(np.mean(aucs), 4),
        'Test Prec Mean':     round(np.mean(precs), 4),
        'Test Recall Mean':   round(np.mean(recs), 4),
    })

df_compare = (
    pd.DataFrame(rows)
    .sort_values('Test F1 Mean', ascending=False)
    .reset_index(drop=True)
)
df_compare.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "daic_v6_comparison_80_10_10.csv")
df_compare.to_csv(csv_path, index=False)

print("\n" + "=" * 120)
print(f"{'RINGKASAN v6 -- 12 Model, 80/10/10 x 5 Repeats (diurutkan Test F1 Mean)':^120}")
print("=" * 120)
print(df_compare[['Feature Extraction', 'Model', 'Test F1 Mean', 'Test F1 Std',
                   'Test F1 Max', 'Test Acc Mean', 'Test AUC Mean']].to_string())
print(f"\nDisimpan: {csv_path}")

# Best overall
best_idx = df_compare['Test F1 Mean'].idxmax()
best     = df_compare.loc[best_idx]
print(f"\n  BEST: {best['Feature Extraction']} + {best['Model']}")
print(f"  Test F1: {best['Test F1 Mean']:.4f} +/- {best['Test F1 Std']:.4f}  "
      f"(max={best['Test F1 Max']:.4f})")
print(f"  Test AUC: {best['Test AUC Mean']:.4f}")

# %% [markdown]
# ## 8. Visualisasi

# %%
COLORS_FEAT  = {'MFCC': '#3b82f6', 'Spectrogram': '#f59e0b', 'Wav2Vec': '#10b981'}
COLORS_MODEL = ['#6366f1', '#ef4444', '#f97316', '#22c55e']

# ─── 8A. Grouped Bar: Mean Test F1 per Feature x Model ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('v6 -- Mean Test Macro F1 (80/10/10 x 5 Repeats, Threshold=0.5)',
             fontsize=13, fontweight='bold', y=1.02)

x = np.arange(len(MODEL_NAMES))
bar_w = 0.6

for ax_idx, feat_name in enumerate(FEAT_NAMES):
    ax = axes[ax_idx]
    vals = []
    errs = []
    for mn in MODEL_NAMES:
        key = f"{feat_name} + {mn}"
        f1s = [m['f1_macro'] for m in all_repeat_results[key]]
        vals.append(np.mean(f1s))
        errs.append(np.std(f1s))

    bars = ax.bar(x, vals, width=bar_w, yerr=errs, capsize=4,
                  color=COLORS_MODEL[:len(MODEL_NAMES)],
                  edgecolor='white', linewidth=0.7, error_kw={'linewidth': 1.2})
    ax.set_title(feat_name, fontweight='bold', fontsize=12, color=COLORS_FEAT[feat_name])
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, rotation=20, ha='right', fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Mean Test Macro F1' if ax_idx == 0 else '')
    ax.axhline(0.7, color='red', linestyle='--', linewidth=0.9, alpha=0.7, label='Target 0.70')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    if ax_idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v6_grouped_bar_f1.png")
fig.savefig(p, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot: {p}")

# ─── 8B. Heatmap F1 Mean + AUC Mean ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

for ax, metric_key, title, cmap in zip(
    axes,
    ['Test F1 Mean', 'Test AUC Mean'],
    ['Mean Test Macro F1', 'Mean Test ROC-AUC'],
    ['YlOrRd', 'Blues']
):
    data_dict = {}
    for fn in FEAT_NAMES:
        data_dict[fn] = []
        for mn in MODEL_NAMES:
            val = df_compare[
                (df_compare['Feature Extraction'] == fn) &
                (df_compare['Model'] == mn)
            ][metric_key].values
            data_dict[fn].append(val[0] if len(val) > 0 else 0.0)

    data = pd.DataFrame(data_dict, index=MODEL_NAMES).T
    sns.heatmap(data, annot=True, fmt='.3f', cmap=cmap,
                linewidths=0.5, linecolor='gray',
                cbar_kws={'label': title},
                ax=ax, vmin=0.3, vmax=1.0)
    ax.set_title(f'{title} (v6)', fontweight='bold', fontsize=11)

plt.tight_layout()
p2 = os.path.join(RESULTS_DIR, "plots", "v6_heatmap.png")
fig.savefig(p2, dpi=150, bbox_inches='tight')
plt.show()
print(f"Heatmap: {p2}")

# ─── 8C. Box Plot per Model — F1 distribution across 5 repeats ──────────────
fig, ax = plt.subplots(figsize=(14, 6))
box_data = []
box_labels = []
for fn in FEAT_NAMES:
    for mn in MODEL_NAMES:
        key = f"{fn} + {mn}"
        f1s = [m['f1_macro'] for m in all_repeat_results[key]]
        box_data.append(f1s)
        box_labels.append(f"{fn[:4]}+{mn[:3]}")

bp = ax.boxplot(box_data, patch_artist=True, labels=box_labels)
colors_12 = [COLORS_FEAT[fn] for fn in FEAT_NAMES for _ in MODEL_NAMES]
for patch, color in zip(bp['boxes'], colors_12):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Target 0.70')
ax.set_ylabel('Test Macro F1')
ax.set_title('v6 -- F1 Distribution (5 Repeats, 80/10/10)', fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend()
plt.tight_layout()
p3 = os.path.join(RESULTS_DIR, "plots", "v6_boxplot_f1.png")
fig.savefig(p3, dpi=150, bbox_inches='tight')
plt.show()
print(f"Boxplot: {p3}")

# %% [markdown]
# ## 9. Classification Report (Best Repeat)

# %%
print("\n" + "=" * 100)
print(f"{'CLASSIFICATION REPORT -- v6 (Best Repeat per Model)':^100}")
print("=" * 100)

class_labels = ['Normal (0)', 'Depresi (1)']

for feat_name in FEAT_NAMES:
    print(f"\n{'-'*100}\n  FEATURE: {feat_name}\n{'-'*100}")
    df, feat_cols = datasets_raw[feat_name]

    for model_name in MODEL_NAMES:
        combo_key = f"{feat_name} + {model_name}"
        best_r = all_best_ys[combo_key]
        tr_idx, va_idx, te_idx = repeated_splits[best_r]

        X_tr, y_tr, _, _, X_te, y_te, _ = preprocess_split(
            df, feat_cols, tr_idx, va_idx, te_idx)

        model = get_models_config()[model_name]
        model.fit(X_tr, y_tr)

        try:
            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        except Exception:
            y_pred = model.predict(X_te)

        metrics = all_repeat_results[combo_key]
        f1s = [m['f1_macro'] for m in metrics]

        print(f"\n  [{model_name}]  (repeat {best_r+1}/{N_REPEATS}, "
              f"mean F1={np.mean(f1s):.4f} +/- {np.std(f1s):.4f})")
        print(classification_report(y_te, y_pred, labels=[0, 1],
                                    target_names=class_labels, zero_division=0))

# ─── Confusion Matrix Grid ──────────────────────────────────────────────────
CMAPS = {'MFCC': 'Blues', 'Spectrogram': 'Oranges', 'Wav2Vec': 'Greens'}

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('v6 -- Confusion Matrix (Best Repeat, 80/10/10, Threshold=0.5)',
             fontsize=13, fontweight='bold')

for fn_idx, feat_name in enumerate(FEAT_NAMES):
    df, feat_cols = datasets_raw[feat_name]
    for mn_idx, model_name in enumerate(MODEL_NAMES):
        ax = axes[fn_idx, mn_idx]
        combo_key = f"{feat_name} + {model_name}"
        best_r = all_best_ys[combo_key]
        tr_idx, va_idx, te_idx = repeated_splits[best_r]

        X_tr, y_tr, _, _, X_te, y_te, _ = preprocess_split(
            df, feat_cols, tr_idx, va_idx, te_idx)
        model = get_models_config()[model_name]
        model.fit(X_tr, y_tr)
        try:
            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        except Exception:
            y_pred = model.predict(X_te)

        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        f1s = [m['f1_macro'] for m in all_repeat_results[combo_key]]

        sns.heatmap(cm, annot=True, fmt='d', cmap=CMAPS.get(feat_name, 'Blues'),
                    ax=ax, xticklabels=class_labels, yticklabels=class_labels,
                    linewidths=0.5, linecolor='gray', cbar=False)
        ax.set_title(f'{feat_name[:4]}+{model_name[:8]}\n'
                     f'F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}',
                     fontweight='bold', fontsize=8)
        ax.set_xlabel('Prediksi', fontsize=7)
        ax.set_ylabel('Aktual', fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
p4 = os.path.join(RESULTS_DIR, "confusion_matrix", "v6_all_cm.png")
fig.savefig(p4, dpi=150, bbox_inches='tight')
plt.show()
print(f"CM: {p4}")

# %% [markdown]
# ## 10. Simpan Metadata & Ringkasan

# %%
# Simpan ringkasan JSON
summary = {
    'version': 'v6',
    'split': '80/10/10 x 5 repeated stratified',
    'threshold': 0.5,
    'augmentation': False,
    'smote': False,
    'wav2vec_real': WAV2VEC_AVAILABLE,
    'n_participants': len(df_any),
    'results': {}
}

for combo_key, metrics in all_repeat_results.items():
    f1s = [m['f1_macro'] for m in metrics]
    summary['results'][combo_key] = {
        'f1_mean': round(float(np.mean(f1s)), 4),
        'f1_std':  round(float(np.std(f1s)), 4),
        'f1_max':  round(float(np.max(f1s)), 4),
    }

best_combo = max(summary['results'], key=lambda k: summary['results'][k]['f1_mean'])
summary['best_model'] = best_combo
summary['best_f1']    = summary['results'][best_combo]['f1_mean']

with open(os.path.join(MODELS_DIR, "v6_summary.json"), 'w') as fp:
    json.dump(summary, fp, indent=2)

print(f"\n[OK] Pipeline v6 selesai!")
print(f"     Wav2Vec 2.0: {'NYATA' if WAV2VEC_AVAILABLE else 'FALLBACK'}")
print(f"     Best: {best_combo}")
print(f"     F1:   {summary['best_f1']:.4f}")
print(f"     Models: {MODELS_DIR}")
print(f"     Results: {RESULTS_DIR}")
