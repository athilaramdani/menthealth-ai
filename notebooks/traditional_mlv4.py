# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ) - Multi Feature Extraction Comparison
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# **Eksperimen v4 — Apple-to-Apple Comparison (12 Model)**:
# Setiap model diuji dengan feature engineering yang *identik* agar perbandingan
# antar model benar-benar adil (apple-to-apple). Pipeline ini menghasilkan total:
#   12 Model = 4 Classifier × 3 Feature Extractor
#
# **Feature Extraction**:
#   1. MFCC          — statistik aggregated MFCC (mean, std, min, max, p25, p75)
#   2. Spectrogram   — mel-spectrogram statistik aggregated per mel-band
#   3. Wav2Vec       — mean-pooled representation dari Wav2Vec 2.0 (facebook/wav2vec2-base)
#                      (fallback ke Word2Vec-style MFCC embedding jika torch/transformers tidak tersedia)
#
# **Classifier**:
#   1. Logistic Regression
#   2. SVM (RBF kernel)
#   3. XGBoost
#   4. Random Forest
#
# **Protocol**: GroupKFold (5-fold) — anti-leakage berbasis participant_id
#               GridSearchCV dengan scoring macro F1

# %%
import os
import sys
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython_fn = globals().get('get_ipython', None)
    if get_ipython_fn is None:
        import builtins
        get_ipython_fn = getattr(builtins, 'get_ipython', None)
    if get_ipython_fn is not None:
        cfg = get_ipython_fn().__class__.__name__
        if cfg != 'ZMQInteractiveShell':
            matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import librosa

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.feature_selection import f_classif, mutual_info_classif
import xgboost as xgb

plt.rcParams['font.family'] = 'DejaVu Sans'

# ── Wav2Vec / Transformers (opsional) ──────────────────────────────────────────
WAV2VEC_AVAILABLE = False
try:
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    WAV2VEC_AVAILABLE = True
    print("[INFO] PyTorch + Transformers tersedia. Akan menggunakan Wav2Vec 2.0.")
except ImportError:
    print("[WARN] PyTorch / Transformers tidak tersedia. "
          "Wav2Vec feature akan menggunakan fallback (statistical MFCC embedding).")

print("Library berhasil diimport.\n")

# %%
# ── Konfigurasi Path ─────────────────────────────────────────────────────────
PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd()
    else os.getcwd()
)

CLEANED_DIR  = os.path.join(PROJECT_ROOT, "data", "cleaned")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")      # reuse existing CSV jika ada
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models", "ml_v4")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "v4")

for d in [
    MODELS_DIR,
    os.path.join(RESULTS_DIR, "metrics"),
    os.path.join(RESULTS_DIR, "plots"),
    os.path.join(RESULTS_DIR, "confusion_matrix"),
]:
    os.makedirs(d, exist_ok=True)

# Kita pakai dataset yang sudah diekstrak sebelumnya (dari v2/v3) sebagai sumber audio row
FINAL_FEATURES_PATH = os.path.join(FEATURES_DIR, "daic_features_final.csv")
FEATURE_LIST_PATH   = os.path.join(FEATURES_DIR, "daic_feature_list.txt")

# Jika ingin paksa re-ekstraksi audio mentah (jarang diperlukan)
FORCE_EXTRACT = False

RANDOM_SEED = 42

print(f"Project root : {PROJECT_ROOT}")
print(f"Base features: {FINAL_FEATURES_PATH}")

# %% [markdown]
# ## 0. Audio Feature Extraction Helpers
# Tiga fungsi utama:
#   A. extract_mfcc_features(y, sr)        → MFCC statistical aggregation
#   B. extract_spectrogram_features(y, sr) → Mel-Spectrogram statistical aggregation
#   C. extract_wav2vec_features(y, sr)     → Wav2Vec 2.0 mean pooling (atau fallback)

# %%
TARGET_SR    = 16000
N_MFCC       = 40          # Lebih banyak dari v2 untuk representasi lebih kaya
N_MELS       = 64          # Jumlah mel bands
FRAME_LENGTH = int(0.025 * TARGET_SR)   # 25 ms
HOP_LENGTH   = int(0.010 * TARGET_SR)   # 10 ms

# ──────────────────────────────────────────────────────────────────────────────
# Helper: aggregasi statistik dari array 1-D
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_feature(feat_array, name):
    if len(feat_array) == 0:
        return {
            f'{name}_mean': 0.0, f'{name}_std': 0.0,
            f'{name}_min':  0.0, f'{name}_max': 0.0,
            f'{name}_p25':  0.0, f'{name}_p75': 0.0,
        }
    return {
        f'{name}_mean': float(np.mean(feat_array)),
        f'{name}_std':  float(np.std(feat_array)),
        f'{name}_min':  float(np.min(feat_array)),
        f'{name}_max':  float(np.max(feat_array)),
        f'{name}_p25':  float(np.percentile(feat_array, 25)),
        f'{name}_p75':  float(np.percentile(feat_array, 75)),
    }

# ──────────────────────────────────────────────────────────────────────────────
# A. MFCC Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_mfcc_features(y, sr):
    """
    Ekstrak N_MFCC koefisien MFCC + delta + delta-delta, lalu agregasikan
    (mean, std, min, max, p25, p75) → dimensi = N_MFCC * 3 * 6
    """
    features = {}
    mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    d_mfcc  = librosa.feature.delta(mfccs)
    dd_mfcc = librosa.feature.delta(mfccs, order=2)

    for i in range(N_MFCC):
        features.update(aggregate_feature(mfccs[i],  f'mfcc_{i+1}'))
        features.update(aggregate_feature(d_mfcc[i], f'dmfcc_{i+1}'))
        features.update(aggregate_feature(dd_mfcc[i],f'ddmfcc_{i+1}'))

    # Pitch (F0)
    try:
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        pitch_vals = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            p   = pitches[idx, t]
            if p > 0:
                pitch_vals.append(p)
        features.update(aggregate_feature(
            np.array(pitch_vals) if pitch_vals else np.array([0.0]), 'pitch'
        ))
    except Exception:
        features.update(aggregate_feature(np.array([0.0]), 'pitch'))

    # RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(rms, 'rms_energy'))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(zcr, 'zcr'))

    return features


# ──────────────────────────────────────────────────────────────────────────────
# B. Spectrogram (Mel-Spectrogram) Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_spectrogram_features(y, sr):
    """
    Ekstrak Mel-Spectrogram (N_MELS bands) dalam dB, lalu agregasikan
    setiap band → dimensi = N_MELS * 6 + spectral features
    """
    features = {}

    # Mel-Spectrogram (dB)
    S  = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    for i in range(N_MELS):
        features.update(aggregate_feature(S_db[i], f'mel_{i+1}'))

    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(cent, 'spec_centroid'))

    # Spectral Bandwidth
    bw = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(bw, 'spec_bandwidth'))

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(rolloff, 'spec_rolloff'))

    # Spectral Flatness
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(flatness, 'spec_flatness'))

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    for i in range(12):
        features.update(aggregate_feature(chroma[i], f'chroma_{i+1}'))

    # RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(rms, 'rms_energy'))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(zcr, 'zcr'))

    return features


# ──────────────────────────────────────────────────────────────────────────────
# C. Wav2Vec 2.0 Feature Extraction (dengan fallback)
# ──────────────────────────────────────────────────────────────────────────────
_wav2vec_processor = None
_wav2vec_model     = None

def _load_wav2vec_model():
    """Lazy-load Wav2Vec2 model & processor sekali saja."""
    global _wav2vec_processor, _wav2vec_model
    if _wav2vec_processor is None:
        print("[INFO] Memuat Wav2Vec2 model (facebook/wav2vec2-base) ...")
        _wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        _wav2vec_model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        _wav2vec_model.eval()
        print("[INFO] Wav2Vec2 model berhasil dimuat.")
    return _wav2vec_processor, _wav2vec_model


def extract_wav2vec_features(y, sr):
    """
    Ekstrak representasi Wav2Vec 2.0:
      - Mean-pool seluruh hidden state dari last_hidden_state → 768-dim vektor.
    Jika torch/transformers tidak tersedia, fallback ke:
      - MFCC + delta (N_MFCC=40) mean-pool + spectral features → vektor kompatibel.
    """
    if WAV2VEC_AVAILABLE:
        try:
            processor, model = _load_wav2vec_model()
            # Resampling ke 16kHz jika perlu
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)

            # Proses audio → tensor
            inputs = processor(
                y, sampling_rate=16000,
                return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)

            # Mean-pool over time axis → (768,)
            hidden = outputs.last_hidden_state.squeeze(0).numpy()   # (T, 768)
            feat_vec = hidden.mean(axis=0)                           # (768,)

            features = {f'w2v_{i}': float(feat_vec[i]) for i in range(len(feat_vec))}
            return features

        except Exception as e:
            print(f"[WARN] Wav2Vec extraction gagal ({e}). Menggunakan fallback.")
            # fall through ke fallback

    # ── Fallback: MFCC 40 + delta mean-pooled + spectral ──────────────────
    features = {}
    mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    d_mfcc = librosa.feature.delta(mfccs)

    # Mean per koefisien (bukan aggregasi penuh) → representasi embedding-style
    for i in range(N_MFCC):
        features[f'w2v_mfcc_{i+1}']  = float(mfccs[i].mean())
        features[f'w2v_dmfcc_{i+1}'] = float(d_mfcc[i].mean())

    # Spectral summary
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features['w2v_centroid_mean'] = float(cent.mean())
    features['w2v_centroid_std']  = float(cent.std())

    bw = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features['w2v_bw_mean'] = float(bw.mean())
    features['w2v_bw_std']  = float(bw.std())

    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features['w2v_rolloff_mean'] = float(rolloff.mean())
    features['w2v_rolloff_std']  = float(rolloff.std())

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features['w2v_rms_mean'] = float(rms.mean())
    features['w2v_rms_std']  = float(rms.std())

    return features


# Mapping nama feature extractor → fungsi
FEATURE_EXTRACTORS = {
    'MFCC':        extract_mfcc_features,
    'Spectrogram': extract_spectrogram_features,
    'Wav2Vec':     extract_wav2vec_features,
}

print("Feature extractor berhasil didefinisikan:")
for name in FEATURE_EXTRACTORS:
    print(f"  - {name}")

# %% [markdown]
# ## 1. Label Mapping

# %%
def map_label_strategi_v1(row):
    """
    Pelabelan biner berdasarkan PHQ-8 score.
    Kelas 0: Normal/Non-Depresi (PHQ < 10)
    Kelas 1: Depresi             (PHQ >= 10)
    """
    phq_binary = row.get('PHQ8_Binary', row.get('PHQ_Binary', np.nan))
    if not pd.isna(phq_binary):
        return int(phq_binary)
    phq_score = row.get('PHQ8_Score', row.get('PHQ_Score', np.nan))
    phq_score = 0 if pd.isna(phq_score) else int(phq_score)
    return 1 if phq_score >= 10 else 0

# %% [markdown]
# ## 2. Multi-Feature Extraction Pipeline
# Mengekstrak ketiga jenis fitur (MFCC, Spectrogram, Wav2Vec) dari setiap audio
# dalam satu pass agar efisien. Hasilnya disimpan dalam dict terpisah per feature type.

# %%
def build_multi_feature_dataset(cleaned_dir, output_dir):
    """
    Satu pass melalui semua audio bersih, mengekstrak tiga representasi sekaligus.
    Hasil disimpan di tiga CSV:
      - daic_v4_mfcc.csv
      - daic_v4_spectrogram.csv
      - daic_v4_wav2vec.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_dir         = os.path.join(os.path.dirname(cleaned_dir), "raw", "DAIC-WOZ")
    train_split_path = os.path.join(raw_dir, "train_split_Depression_AVEC2017.csv")
    dev_split_path   = os.path.join(raw_dir, "dev_split_Depression_AVEC2017.csv")
    test_split_path  = os.path.join(raw_dir, "full_test_split.csv")

    if not all(os.path.exists(p) for p in [train_split_path, dev_split_path, test_split_path]):
        raise FileNotFoundError(f"Metadata split DAIC-WOZ tidak ditemukan di: {raw_dir}")

    all_parts = []
    for path, split in [
        (train_split_path, 'train'),
        (dev_split_path,   'dev'),
        (test_split_path,  'test'),
    ]:
        df_part = pd.read_csv(path)
        df_part.columns = [c.strip() for c in df_part.columns]
        # Normalize Participant_ID column
        for col in df_part.columns:
            if col.lower() == 'participant_id':
                df_part.rename(columns={col: 'Participant_ID'}, inplace=True)
        if 'PHQ_Score' not in df_part.columns and 'PHQ8_Score' in df_part.columns:
            df_part['PHQ_Score'] = df_part['PHQ8_Score']
        elif 'PHQ8_Score' not in df_part.columns and 'PHQ_Score' in df_part.columns:
            df_part['PHQ8_Score'] = df_part['PHQ_Score']
        df_part['label_depresi'] = df_part.apply(map_label_strategi_v1, axis=1)
        df_part['split'] = split
        all_parts.append(df_part)

    df_meta = pd.concat(all_parts, ignore_index=True)
    df_meta.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    df_meta['participant_id'] = df_meta['participant_id'].astype(int)

    cleaned_files = sorted([f for f in os.listdir(cleaned_dir) if f.endswith('.wav')])
    print(f"Ditemukan {len(cleaned_files)} file audio bersih.")

    # Struktur akumulator: dict of lists per feature type
    rows_by_type = {name: [] for name in FEATURE_EXTRACTORS}
    META_KEYS = ['participant_id', 'phq8_score', 'label_depresi', 'split', 'gender']

    sep = "=" * 90
    print(f"\n{sep}")
    print(f"{'EKSTRAKSI MULTI-FITUR (MFCC | Spectrogram | Wav2Vec)':^90}")
    print(sep)
    print(f"{'PID':>6} | {'Label':^9} | {'Durasi':>7} | {'MFCC':^6} | {'Spec':^6} | {'W2V':^6} | Status")
    print("-" * 90)

    t_start = time.time()

    for file in cleaned_files:
        pid = int(file.replace('.wav', ''))
        meta_row = df_meta[df_meta['participant_id'] == pid]
        if meta_row.empty:
            print(f"  PID {pid:03d} — tidak ada metadata, dilewati.")
            continue

        audio_path = os.path.join(cleaned_dir, file)
        try:
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            if len(y) < TARGET_SR:
                print(f"  PID {pid:03d} — audio < 1 detik, dilewati.")
                continue

            dur = len(y) / sr
            label_str = "Depresi" if meta_row.iloc[0]['label_depresi'] == 1 else "Normal"

            # Metadata baris
            meta_vals = {
                'participant_id': pid,
                'phq8_score':     int(meta_row.iloc[0].get('PHQ8_Score', 0)),
                'label_depresi':  int(meta_row.iloc[0]['label_depresi']),
                'split':          meta_row.iloc[0]['split'],
                'gender':         int(meta_row.iloc[0].get('Gender', 0)),
            }

            feat_counts = {}
            ok_status   = []

            for feat_name, extractor_fn in FEATURE_EXTRACTORS.items():
                try:
                    feats = extractor_fn(y, sr)
                    row   = {**meta_vals, **feats}
                    rows_by_type[feat_name].append(row)
                    feat_counts[feat_name] = len(feats)
                    ok_status.append(feat_name[:3])
                except Exception as e_feat:
                    print(f"\n  [WARN] PID {pid:03d} {feat_name} gagal: {e_feat}")
                    rows_by_type[feat_name].append(meta_vals)   # placeholder
                    feat_counts[feat_name] = 0
                    ok_status.append('ERR')

            mfcc_n = feat_counts.get('MFCC', 0)
            spec_n = feat_counts.get('Spectrogram', 0)
            w2v_n  = feat_counts.get('Wav2Vec', 0)
            print(
                f"  {pid:4d}   | {label_str:^9} | {dur:5.1f}s   | "
                f"{mfcc_n:4d}   | {spec_n:4d}   | {w2v_n:4d}   | OK",
                flush=True
            )

        except Exception as e:
            print(f"  PID {pid:03d} — ERROR: {e}", flush=True)

    elapsed = time.time() - t_start
    print("=" * 90)
    print(f"\nEkstraksi selesai dalam {elapsed:.1f} detik.\n")

    # Simpan CSV per feature type
    saved_paths = {}
    for feat_name, rows in rows_by_type.items():
        df_feat = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"daic_v4_{feat_name.lower()}.csv")
        df_feat.to_csv(csv_path, index=False)
        print(f"Disimpan: {csv_path}  (shape: {df_feat.shape})")
        saved_paths[feat_name] = csv_path

    return saved_paths


# ──────────────────────────────────────────────────────────────────────────────
# Cek apakah semua CSV sudah ada; jika tidak, jalankan ekstraksi
# ──────────────────────────────────────────────────────────────────────────────
V4_FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v4")
os.makedirs(V4_FEATURES_DIR, exist_ok=True)

V4_CSV_PATHS = {
    'MFCC':        os.path.join(V4_FEATURES_DIR, "daic_v4_mfcc.csv"),
    'Spectrogram': os.path.join(V4_FEATURES_DIR, "daic_v4_spectrogram.csv"),
    'Wav2Vec':     os.path.join(V4_FEATURES_DIR, "daic_v4_wav2vec.csv"),
}

all_exist = all(os.path.exists(p) for p in V4_CSV_PATHS.values())

if FORCE_EXTRACT or not all_exist:
    print("\n[INFO] Menjalankan ekstraksi multi-fitur ...")
    V4_CSV_PATHS = build_multi_feature_dataset(CLEANED_DIR, V4_FEATURES_DIR)
else:
    print("\n[INFO] Semua CSV fitur v4 sudah tersedia:")
    for k, p in V4_CSV_PATHS.items():
        print(f"  {k}: {p}")

# %% [markdown]
# ## 3. Load & Prepare Dataset per Feature Type

# %%
def load_and_prepare(csv_path, feat_name):
    """
    Muat CSV, pisahkan train/dev/test, scale fitur (fit hanya pada train),
    return dict berisi X, y, scaler, feature names, dan dataframes.
    """
    META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'split', 'gender']

    df = pd.read_csv(csv_path)

    # Identifikasi kolom fitur
    feat_cols = [c for c in df.columns if c not in META_COLS]

    # Isi NaN dengan median train
    train_mask = df['split'] == 'train'
    medians = df.loc[train_mask, feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(medians)

    # Hapus fitur konstan
    std_vals   = df.loc[train_mask, feat_cols].std()
    const_feat = std_vals[std_vals < 1e-8].index.tolist()
    feat_cols  = [f for f in feat_cols if f not in const_feat]

    # Clip outlier IQR × 10
    Q1  = df.loc[train_mask, feat_cols].quantile(0.25)
    Q3  = df.loc[train_mask, feat_cols].quantile(0.75)
    IQR = Q3 - Q1
    for col in feat_cols:
        lo = Q1[col] - 10 * IQR[col]
        hi = Q3[col] + 10 * IQR[col]
        df[col] = df[col].clip(lower=lo, upper=hi)

    # Hapus fitur dengan korelasi > 0.95 (hitung dari train)
    corr_mat  = df.loc[train_mask, feat_cols].corr().abs()
    upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop   = [c for c in upper_tri.columns if any(upper_tri[c] > 0.95)]
    feat_cols = [f for f in feat_cols if f not in to_drop]

    # Feature selection dengan F-test + MI (pada train)
    X_tr_sel = df.loc[train_mask, feat_cols].values
    y_tr_sel = df.loc[train_mask, 'label_depresi'].values
    if len(feat_cols) > 0 and len(np.unique(y_tr_sel)) > 1:
        f_scores, p_vals = f_classif(X_tr_sel, y_tr_sel)
        mi_scores        = mutual_info_classif(X_tr_sel, y_tr_sel, random_state=RANDOM_SEED)
        df_sel = pd.DataFrame({
            'feature': feat_cols,
            'p_value': p_vals,
            'mi_score': mi_scores,
            'significant': p_vals < 0.05
        })
        sig_feats    = df_sel[df_sel['significant']]['feature'].tolist()
        top_mi_feats = df_sel.sort_values('mi_score', ascending=False).head(60)['feature'].tolist()
        final_feats  = list(set(sig_feats) | set(top_mi_feats))
        feat_cols    = [f for f in feat_cols if f in final_feats]

    if len(feat_cols) == 0:
        # Fallback jika semua fitur di-drop
        feat_cols = [c for c in df.columns if c not in META_COLS][:10]

    # Split
    df_tr  = df[df['split'] == 'train'].reset_index(drop=True)
    df_dev = df[df['split'] == 'dev'].reset_index(drop=True)
    df_te  = df[df['split'] == 'test'].reset_index(drop=True)

    X_train = df_tr[feat_cols].values
    y_train = df_tr['label_depresi'].values
    groups  = df_tr['participant_id'].values

    X_dev   = df_dev[feat_cols].values
    y_dev   = df_dev['label_depresi'].values

    X_test  = df_te[feat_cols].values
    y_test  = df_te['label_depresi'].values

    # Scale (fit hanya pada train)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_sc = scaler.transform(X_train)
    X_dev_sc   = scaler.transform(X_dev)
    X_test_sc  = scaler.transform(X_test)

    # Simpan scaler
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{feat_name.lower()}.pkl")
    with open(scaler_path, 'wb') as fp:
        pickle.dump(scaler, fp)

    print(f"\n[{feat_name}] Dataset siap:")
    print(f"  Jumlah fitur final : {len(feat_cols)}")
    print(f"  Train shape        : {X_train_sc.shape}")
    print(f"  Dev shape          : {X_dev_sc.shape}")
    print(f"  Test shape         : {X_test_sc.shape}")
    print(f"  Distribusi y_train : {dict(zip(*np.unique(y_train, return_counts=True)))}")

    return {
        'feat_name':    feat_name,
        'feat_cols':    feat_cols,
        'scaler':       scaler,
        'X_train_sc':   X_train_sc,
        'y_train':      y_train,
        'groups':       groups,
        'X_dev_sc':     X_dev_sc,
        'y_dev':        y_dev,
        'X_test_sc':    X_test_sc,
        'y_test':       y_test,
    }

# Load semua feature types
datasets = {}
for feat_name, csv_path in V4_CSV_PATHS.items():
    datasets[feat_name] = load_and_prepare(csv_path, feat_name)

# %% [markdown]
# ## 4. Definisi Model & Hyperparameter Grid (Identik untuk Semua Feature Types)
# Konfigurasi model **sama persis** untuk setiap feature type — inilah kunci
# perbandingan apple-to-apple.

# %%
def get_models_config():
    """
    Mengembalikan dict konfigurasi model yang IDENTIK untuk setiap feature type.
    Dipanggil ulang agar setiap feature type mendapatkan instance yang bersih.
    """
    return {
        'Logistic Regression': {
            'model': LogisticRegression(
                max_iter=2000, random_state=RANDOM_SEED, class_weight='balanced'
            ),
            'param_grid': {
                'C':      [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear'],
            }
        },
        'SVM': {
            'model': SVC(
                kernel='rbf', probability=True,
                random_state=RANDOM_SEED, class_weight='balanced',
                decision_function_shape='ovr'
            ),
            'param_grid': {
                'C':     [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto'],
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(
                random_state=RANDOM_SEED,
                eval_metric='logloss',
                objective='binary:logistic',
                n_jobs=-1
            ),
            'param_grid': {
                'n_estimators':  [50, 100],
                'max_depth':     [3, 5],
                'learning_rate': [0.05, 0.1],
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1
            ),
            'param_grid': {
                'n_estimators':    [50, 100, 200],
                'max_depth':       [None, 5, 10],
                'min_samples_split': [2, 5],
            }
        },
    }

print("Konfigurasi model (identik untuk semua feature type):")
for mname in get_models_config():
    print(f"  - {mname}")
print(f"\nTotal kombinasi: {len(get_models_config())} model × {len(FEATURE_EXTRACTORS)} feature type = "
      f"{len(get_models_config()) * len(FEATURE_EXTRACTORS)} model")

# %% [markdown]
# ## 5. Training Pipeline — Apple-to-Apple
# Setiap (feature_type, model) dilatih dengan protokol yang **identik**:
#   - GroupKFold(n_splits=5) berdasarkan participant_id
#   - GridSearchCV dengan scoring='f1_macro'
#   - Evaluasi pada Train / Dev / Test (tingkat partisipan langsung)

# %%
def evaluate_split(model, X, y, prefix=''):
    """Evaluasi performa model pada satu split."""
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc    = float(roc_auc_score(y, y_prob))
    except Exception:
        auc = 0.0
    return {
        f'{prefix}accuracy':         float(accuracy_score(y, y_pred)),
        f'{prefix}f1_macro':         float(f1_score(y, y_pred, average='macro', zero_division=0)),
        f'{prefix}f1_weighted':      float(f1_score(y, y_pred, average='weighted', zero_division=0)),
        f'{prefix}precision_macro':  float(precision_score(y, y_pred, average='macro', zero_division=0)),
        f'{prefix}recall_macro':     float(recall_score(y, y_pred, average='macro', zero_division=0)),
        f'{prefix}roc_auc':          auc,
    }

cv_splitter = GroupKFold(n_splits=5)

# Master results container: { (feat_name, model_name): metrics_dict }
all_results   = {}
all_models    = {}   # simpan trained model objects
all_y_test    = {}   # simpan y_true & y_pred per combo untuk confusion matrix

sep = "=" * 80

for feat_name, ds in datasets.items():
    print(f"\n{sep}")
    print(f"  FEATURE TYPE: {feat_name}  |  {ds['X_train_sc'].shape[1]} fitur")
    print(sep)

    models_cfg = get_models_config()

    for model_name, cfg in models_cfg.items():
        combo_key = f"{feat_name} + {model_name}"
        print(f"\n  [{combo_key}] Training ...")

        grid_search = GridSearchCV(
            estimator=cfg['model'],
            param_grid=cfg['param_grid'],
            cv=cv_splitter,
            scoring='f1_macro',
            n_jobs=-1,
            refit=True
        )
        grid_search.fit(
            ds['X_train_sc'], ds['y_train'],
            groups=ds['groups']
        )
        best_model = grid_search.best_estimator_

        train_metrics = evaluate_split(best_model, ds['X_train_sc'], ds['y_train'], 'train_')
        dev_metrics   = evaluate_split(best_model, ds['X_dev_sc'],   ds['y_dev'],   'val_')
        test_metrics  = evaluate_split(best_model, ds['X_test_sc'],  ds['y_test'],  'test_')

        y_pred_test = best_model.predict(ds['X_test_sc'])

        print(f"    Best params  : {grid_search.best_params_}")
        print(f"    CV Macro F1  : {grid_search.best_score_:.4f}")
        print(f"    Val  Macro F1: {dev_metrics['val_f1_macro']:.4f}  (Acc: {dev_metrics['val_accuracy']:.4f})")
        print(f"    Test Macro F1: {test_metrics['test_f1_macro']:.4f}  (Acc: {test_metrics['test_accuracy']:.4f})")

        all_results[combo_key] = {
            'feature_type': feat_name,
            'model_name':   model_name,
            'best_params':  grid_search.best_params_,
            'best_cv_f1':   float(grid_search.best_score_),
            **train_metrics,
            **dev_metrics,
            **test_metrics,
        }
        all_models[combo_key]  = best_model
        all_y_test[combo_key]  = (ds['y_test'], y_pred_test)

        # Simpan model individual
        safe_feat  = feat_name.lower().replace(' ', '_')
        safe_model = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(MODELS_DIR, f"{safe_feat}_{safe_model}.pkl")
        with open(model_path, 'wb') as fp:
            pickle.dump(best_model, fp)

print(f"\n{sep}")
print("  SEMUA 12 MODEL BERHASIL DILATIH")
print(sep)

# %% [markdown]
# ## 6. Tabel Perbandingan Apple-to-Apple (12 Model)

# %%
comparison_rows = []
for combo_key, res in all_results.items():
    comparison_rows.append({
        'Feature Extraction':  res['feature_type'],
        'Model':               res['model_name'],
        'CV Macro F1':         round(res['best_cv_f1'], 4),
        'Val Accuracy':        round(res['val_accuracy'], 4),
        'Val Macro F1':        round(res['val_f1_macro'], 4),
        'Test Accuracy':       round(res['test_accuracy'], 4),
        'Test Macro F1':       round(res['test_f1_macro'], 4),
        'Test Weighted F1':    round(res['test_f1_weighted'], 4),
        'Test Precision':      round(res['test_precision_macro'], 4),
        'Test Recall':         round(res['test_recall_macro'], 4),
        'Test ROC-AUC':        round(res['test_roc_auc'], 4),
    })

df_compare = pd.DataFrame(comparison_rows).sort_values(
    ['Feature Extraction', 'Test Macro F1'], ascending=[True, False]
).reset_index(drop=True)

comparison_csv = os.path.join(RESULTS_DIR, "metrics", "daic_v4_comparison_12models.csv")
df_compare.to_csv(comparison_csv, index=False)

print("\n" + "=" * 100)
print(f"{'RINGKASAN APPLE-TO-APPLE: 12 MODEL (4 Classifier × 3 Feature Extraction)':^100}")
print("=" * 100)
print(df_compare.to_string(index=False))
print(f"\nTabel disimpan di: {comparison_csv}")

# Best per feature type
print("\n--- BEST MODEL PER FEATURE TYPE ---")
for feat_name in FEATURE_EXTRACTORS:
    subset = {k: v for k, v in all_results.items() if v['feature_type'] == feat_name}
    best_k = max(subset, key=lambda k: subset[k]['test_f1_macro'])
    r = subset[best_k]
    print(f"  {feat_name:12s}: {r['model_name']:20s}  Test F1={r['test_f1_macro']:.4f}  AUC={r['test_roc_auc']:.4f}")

# Global best
global_best_k = max(all_results, key=lambda k: all_results[k]['test_f1_macro'])
global_best   = all_results[global_best_k]
print(f"\n  GLOBAL BEST: {global_best_k}")
print(f"  Test Macro F1 = {global_best['test_f1_macro']:.4f}  |  Test Acc = {global_best['test_accuracy']:.4f}")

# Save global best info
best_info = {
    'combo_key':          global_best_k,
    'feature_type':       global_best['feature_type'],
    'model_name':         global_best['model_name'],
    'best_params':        global_best['best_params'],
    'best_cv_f1':         global_best['best_cv_f1'],
    'test_f1_macro':      global_best['test_f1_macro'],
    'test_accuracy':      global_best['test_accuracy'],
    'test_roc_auc':       global_best['test_roc_auc'],
}
with open(os.path.join(MODELS_DIR, "best_model_v4_info.json"), 'w') as fp:
    json.dump(best_info, fp, indent=2)

# %% [markdown]
# ## 7. Visualisasi Perbandingan — Apple-to-Apple

# %%
COLORS_FEAT  = {'MFCC': '#3b82f6', 'Spectrogram': '#f59e0b', 'Wav2Vec': '#10b981'}
COLORS_MODEL = ['#6366f1', '#ef4444', '#f97316', '#22c55e']   # LR, SVM, XGB, RF
MODEL_NAMES  = list(get_models_config().keys())
FEAT_NAMES   = list(FEATURE_EXTRACTORS.keys())

# ─── 7A. Grouped Bar: Test Macro F1 per Feature Type × Model ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('Perbandingan Apple-to-Apple: Test Macro F1\n(4 Classifier × 3 Feature Extraction)',
             fontsize=13, fontweight='bold', y=1.02)

x = np.arange(len(MODEL_NAMES))
bar_width = 0.65

for ax_idx, feat_name in enumerate(FEAT_NAMES):
    ax = axes[ax_idx]
    values = [
        all_results[f"{feat_name} + {mn}"]['test_f1_macro']
        for mn in MODEL_NAMES
    ]
    bars = ax.bar(x, values, width=bar_width, color=COLORS_MODEL,
                  edgecolor='white', linewidth=0.8)
    ax.set_title(feat_name, fontweight='bold', fontsize=12,
                 color=COLORS_FEAT[feat_name])
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Test Macro F1' if ax_idx == 0 else '')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Legend model
patches = [mpatches.Patch(color=COLORS_MODEL[i], label=MODEL_NAMES[i])
           for i in range(len(MODEL_NAMES))]
fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "plots", "v4_apple_to_apple_f1.png")
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot Apple-to-Apple F1 disimpan di: {plot_path}")

# ─── 7B. Heatmap: Test Macro F1 (Feature Type × Model) ──────────────────────
heatmap_data = pd.DataFrame(
    index=FEAT_NAMES,
    columns=MODEL_NAMES,
    data=[
        [all_results[f"{fn} + {mn}"]['test_f1_macro'] for mn in MODEL_NAMES]
        for fn in FEAT_NAMES
    ]
).astype(float)

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(
    heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
    linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Test Macro F1'},
    ax=ax, vmin=0.3, vmax=1.0
)
ax.set_title('Heatmap Test Macro F1 — Apple-to-Apple (Feature × Model)',
             fontweight='bold', fontsize=12, pad=12)
ax.set_xlabel('Classifier', fontsize=10)
ax.set_ylabel('Feature Extraction', fontsize=10)
plt.tight_layout()
heatmap_path = os.path.join(RESULTS_DIR, "plots", "v4_heatmap_f1.png")
fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Heatmap disimpan di: {heatmap_path}")

# ─── 7C. Heatmap: Test ROC-AUC ───────────────────────────────────────────────
heatmap_auc = pd.DataFrame(
    index=FEAT_NAMES,
    columns=MODEL_NAMES,
    data=[
        [all_results[f"{fn} + {mn}"]['test_roc_auc'] for mn in MODEL_NAMES]
        for fn in FEAT_NAMES
    ]
).astype(float)

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(
    heatmap_auc, annot=True, fmt='.3f', cmap='Blues',
    linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Test ROC-AUC'},
    ax=ax, vmin=0.3, vmax=1.0
)
ax.set_title('Heatmap Test ROC-AUC — Apple-to-Apple (Feature × Model)',
             fontweight='bold', fontsize=12, pad=12)
ax.set_xlabel('Classifier', fontsize=10)
ax.set_ylabel('Feature Extraction', fontsize=10)
plt.tight_layout()
heatmap_auc_path = os.path.join(RESULTS_DIR, "plots", "v4_heatmap_auc.png")
fig.savefig(heatmap_auc_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Heatmap AUC disimpan di: {heatmap_auc_path}")

# ─── 7D. Multi-Metric Comparison: semua 12 model sekaligus ───────────────────
combo_labels = [f"{fn[:4]}\n{mn[:4]}" for fn in FEAT_NAMES for mn in MODEL_NAMES]
combo_f1     = [all_results[f"{fn} + {mn}"]['test_f1_macro']
                for fn in FEAT_NAMES for mn in MODEL_NAMES]
combo_acc    = [all_results[f"{fn} + {mn}"]['test_accuracy']
                for fn in FEAT_NAMES for mn in MODEL_NAMES]
combo_auc    = [all_results[f"{fn} + {mn}"]['test_roc_auc']
                for fn in FEAT_NAMES for mn in MODEL_NAMES]

combo_colors = []
for fn in FEAT_NAMES:
    combo_colors.extend([COLORS_FEAT[fn]] * len(MODEL_NAMES))

x12 = np.arange(len(combo_labels))
bw  = 0.28

fig, ax = plt.subplots(figsize=(20, 6))
b1 = ax.bar(x12 - bw, combo_f1,  width=bw, color=combo_colors, alpha=0.9,
            edgecolor='black', linewidth=0.5, label='Test Macro F1')
b2 = ax.bar(x12,      combo_acc, width=bw, color=combo_colors, alpha=0.55,
            edgecolor='black', linewidth=0.5, label='Test Accuracy')
b3 = ax.bar(x12 + bw, combo_auc, width=bw, color=combo_colors, alpha=0.3,
            edgecolor='black', linewidth=0.5, label='Test ROC-AUC')

# Annotate F1 values
for bar, val in zip(b1, combo_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xticks(x12)
ax.set_xticklabels(combo_labels, fontsize=8)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score')
ax.set_title('Perbandingan 12 Model — Test Macro F1 / Accuracy / ROC-AUC\n'
             '(warna menunjukkan Feature Extraction: Biru=MFCC, Kuning=Spectrogram, Hijau=Wav2Vec)',
             fontsize=11, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Group separator lines
for sep_x in [3.5, 7.5]:
    ax.axvline(x=sep_x, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)

# Legend
feat_patches = [mpatches.Patch(color=COLORS_FEAT[fn], label=fn) for fn in FEAT_NAMES]
metric_handles = [
    mpatches.Patch(color='gray', alpha=0.9, label='Test Macro F1'),
    mpatches.Patch(color='gray', alpha=0.55, label='Test Accuracy'),
    mpatches.Patch(color='gray', alpha=0.3,  label='Test ROC-AUC'),
]
ax.legend(handles=feat_patches + metric_handles, loc='upper right',
          fontsize=8, ncol=2, framealpha=0.85)

plt.tight_layout()
multi_path = os.path.join(RESULTS_DIR, "plots", "v4_12model_multi_metric.png")
fig.savefig(multi_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot 12-model multi-metric disimpan di: {multi_path}")

# %% [markdown]
# ## 8. Confusion Matrix — Semua 12 Model

# %%
class_labels = ['Normal (0)', 'Depresi (1)']

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle(
    'Confusion Matrix — 12 Model Apple-to-Apple\n(Baris: Feature Extraction | Kolom: Classifier)',
    fontsize=13, fontweight='bold'
)

cmaps = {'MFCC': 'Blues', 'Spectrogram': 'Oranges', 'Wav2Vec': 'Greens'}

for row_idx, feat_name in enumerate(FEAT_NAMES):
    for col_idx, model_name in enumerate(MODEL_NAMES):
        ax       = axes[row_idx, col_idx]
        combo_k  = f"{feat_name} + {model_name}"
        y_true, y_pred = all_y_test[combo_k]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        f1 = all_results[combo_k]['test_f1_macro']
        acc = all_results[combo_k]['test_accuracy']

        sns.heatmap(
            cm, annot=True, fmt='d',
            cmap=cmaps.get(feat_name, 'Blues'),
            ax=ax,
            xticklabels=class_labels,
            yticklabels=class_labels,
            linewidths=0.5, linecolor='gray',
            cbar=False
        )
        ax.set_title(
            f'{feat_name} + {model_name}\nF1={f1:.3f}  Acc={acc:.3f}',
            fontweight='bold', fontsize=8
        )
        ax.set_xlabel('Prediksi', fontsize=7)
        ax.set_ylabel('Aktual', fontsize=7)
        ax.tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix", "v4_12model_confusion_matrices.png")
fig.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot Confusion Matrix 12 model disimpan di: {cm_path}")

# %% [markdown]
# ## 9. Classification Report — Semua 12 Model

# %%
print("\n" + "=" * 100)
print(f"{'CLASSIFICATION REPORT — SEMUA 12 MODEL':^100}")
print("=" * 100)

for feat_name in FEAT_NAMES:
    print(f"\n{'─' * 100}")
    print(f"  FEATURE TYPE: {feat_name}")
    print(f"{'─' * 100}")
    for model_name in MODEL_NAMES:
        combo_k = f"{feat_name} + {model_name}"
        y_true, y_pred = all_y_test[combo_k]
        print(f"\n  Model: {model_name}")
        print(classification_report(y_true, y_pred, labels=[0, 1],
                                    target_names=class_labels, zero_division=0))

# %% [markdown]
# ## 10. Ekspor Model Terbaik & Ringkasan Akhir

# %%
# Copy best model ke folder root models_dir dengan nama yang jelas
best_combo_key = max(all_results, key=lambda k: all_results[k]['test_f1_macro'])
best_model_obj = all_models[best_combo_key]
best_res       = all_results[best_combo_key]

best_model_export_path = os.path.join(MODELS_DIR, "best_model_v4.pkl")
with open(best_model_export_path, 'wb') as fp:
    pickle.dump(best_model_obj, fp)

print("\n" + "=" * 80)
print(f"  ✓ MODEL TERBAIK (v4): {best_combo_key}")
print(f"  Test Macro F1 : {best_res['test_f1_macro']:.4f}")
print(f"  Test Accuracy : {best_res['test_accuracy']:.4f}")
print(f"  Test ROC-AUC  : {best_res['test_roc_auc']:.4f}")
print(f"  Best Params   : {best_res['best_params']}")
print(f"  Disimpan di   : {best_model_export_path}")
print("=" * 80)

# ─── Tabel ringkasan final ───────────────────────────────────────────────────
print("\n--- RINGKASAN FINAL (diurutkan berdasarkan Test Macro F1) ---")
df_final_sorted = df_compare.sort_values('Test Macro F1', ascending=False).reset_index(drop=True)
df_final_sorted.index += 1
print(df_final_sorted[['Feature Extraction', 'Model', 'Test Macro F1',
                         'Test Accuracy', 'Test ROC-AUC']].to_string())

print("\n[OK] Pipeline Traditional ML v4 (12 Model — Apple-to-Apple) Selesai!")
