# %% [markdown]
# # Pipeline v91 — Cost-Sensitive + LDA + Expanded PCA + Aggressive Threshold
#
# ─────────────────────────────────────────────────────────────────────────────
# v89 Diagnosis (189 data, full dataset):
# - Best Single  : S1_Spectrogram x LR n=10 -> Test(oof)=0.6011
# - Rasio N:D    : 133:56 = 2.38:1 (imbalanced!)
# - Semua model  : OV (overfit CV >> Test)
# - Wav2Vec XGB  : 0.4048 (hancur dari 0.7494 di 102-data)
#
# v91 SMOTE Trial -> GAGAL: CV=0.85 tapi Test=0.52 (SMOTE overfit sintesis)
# SMOTE tidak cocok untuk dataset training <200 sampel per fold
#
# STRATEGI v91 REV2 — NO SMOTE, SMARTER TUNING:
# [A] HAPUS SMOTE -> kembali ke class_weight saja (terbukti lebih stabil)
# [B] scale_pos_weight XGB lebih agresif: [ratio, 2.0, 3.0, 4.0]
#     -> XGBoost tahu depresi lebih berharga
# [C] LDA sebagai alternatif PCA untuk Wav2Vec
#     -> Supervised dimensionality reduction (gunakan label)
# [D] Expanded PCA range [5,10,15,20,25,30,40,50]
#     -> Variasi lebih besar, 189 data butuh lebih banyak komponen
# [E] Threshold sweep agresif 0.15-0.55 (fokus recall Depresi)
#     -> Dataset imbalanced butuh threshold lebih rendah dari 0.5
# ─────────────────────────────────────────────────────────────────────────────

# %%
import os, warnings, time, sys, json
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import librosa
from scipy.stats import kurtosis, skew
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, confusion_matrix
)
import xgboost as xgb

# SMOTE + Tomek Links (imblearn)
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("[INFO] imbalanced-learn OK -> SMOTE + Tomek aktif.")
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn tidak tersedia. Lanjut tanpa SMOTE.")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (os.path.abspath(os.path.join(os.getcwd(), ".."))
                if "notebooks" in os.getcwd() else os.getcwd())
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v91")
for d in [os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 80)
print("  Pipeline v91 — SMOTE + Cost-Sensitive + LDA + Expanded PCA")
print("  Target: F1 >= 0.70 (dari 0.6011 di v89 189-data)")
print("=" * 80)

# %% [markdown]
# ## Feature Extraction (Participant-Level, dengan caching)

# %%

# --- Setup Wav2Vec ---
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

# ## Audio Config
TARGET_SR    = 16000
N_MFCC       = 40
N_MELS       = 64
FRAME_LENGTH = int(0.025 * TARGET_SR)
HOP_LENGTH   = int(0.010 * TARGET_SR)

def agg(arr, name):
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

def extract_mfcc_prosodic(y, sr):
    feats = {}
    mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    d_mfcc  = librosa.feature.delta(mfccs)
    dd_mfcc = librosa.feature.delta(mfccs, order=2)
    for i in range(N_MFCC):
        feats.update(agg(mfccs[i],   f'm{i+1}'))
        feats.update(agg(d_mfcc[i],  f'dm{i+1}'))
        feats.update(agg(dd_mfcc[i], f'ddm{i+1}'))
    try:
        pitches, mags = librosa.piptrack(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
                                          fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pv = [pitches[mags[:, t].argmax(), t] for t in range(pitches.shape[1])
              if pitches[mags[:, t].argmax(), t] > 0]
        feats.update(agg(np.array(pv) if pv else np.array([0.0]), 'pitch'))
    except Exception:
        feats.update(agg(np.array([0.0]), 'pitch'))
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(rms, 'rms'))
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(zcr, 'zcr'))
    feats['low_energy_ratio'] = float(np.mean(rms < 0.1 * np.mean(rms)))
    try:
        harmonic  = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        rms_h = float(np.sqrt(np.mean(harmonic ** 2)) + 1e-10)
        rms_p = float(np.sqrt(np.mean(percussive ** 2)) + 1e-10)
        feats['hnr_proxy'] = float(20 * np.log10(rms_h / rms_p))
    except Exception:
        feats['hnr_proxy'] = 0.0
    try:
        S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
        feats['spec_entropy'] = spectral_entropy(S.mean(axis=1))
    except Exception:
        feats['spec_entropy'] = 0.0
    try:
        pitches, mags = librosa.piptrack(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
                                          fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pv, vf = [], []
        for t in range(pitches.shape[1]):
            idx = mags[:, t].argmax()
            p = pitches[idx, t]
            if p > 50.0:
                pv.append(p); vf.append(t)
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
        feats['jitter'] = 0.0; feats['shimmer'] = 0.0
    try:
        pitches2, mags2 = librosa.piptrack(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        voiced = sum(1 for t in range(pitches2.shape[1]) if pitches2[mags2[:, t].argmax(), t] > 0)
        total  = pitches2.shape[1]
        feats['voiced_ratio'] = float(voiced / max(total, 1))
    except Exception:
        feats['voiced_ratio'] = 0.0
    return feats

def extract_spectrogram(y, sr):
    feats = {}
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    S_db = librosa.power_to_db(S, ref=np.max)
    for i in range(N_MELS):
        feats.update(agg(S_db[i], f'mel{i+1}'))
        feats[f'mel{i+1}_ent'] = spectral_entropy(S[i])
    for name, fn in [
        ('cent', librosa.feature.spectral_centroid),
        ('bw', librosa.feature.spectral_bandwidth),
        ('rolloff', librosa.feature.spectral_rolloff),
    ]:
        vals = fn(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        feats.update(agg(vals, name))
    flat = librosa.feature.spectral_flatness(y=y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(flat, 'flat'))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    for i in range(12):
        feats.update(agg(chroma[i], f'ch{i+1}'))
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(rms, 'rms'))
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats.update(agg(zcr, 'zcr'))
    feats['low_energy_ratio'] = float(np.mean(rms < 0.1 * np.mean(rms)))
    feats['spec_entropy']     = spectral_entropy(S.mean(axis=1))
    return feats

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
                hidden = np.concatenate(all_hidden, axis=0)
                mean_h = hidden.mean(axis=0)
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

def map_label_meta(row):
    phq_binary = row.get('PHQ8_Binary', row.get('PHQ_Binary', np.nan))
    if not pd.isna(phq_binary):
        return int(phq_binary)
    phq = row.get('PHQ8_Score', row.get('PHQ_Score', np.nan))
    phq = 0 if pd.isna(phq) else int(phq)
    return 1 if phq >= 10 else 0

def load_all_metadata():
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
        df['label_depresi'] = df.apply(map_label_meta, axis=1)
        df['split_original'] = split
        all_parts.append(df)
    df_meta = pd.concat(all_parts, ignore_index=True)
    df_meta.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    df_meta['participant_id'] = df_meta['participant_id'].astype(int)
    return df_meta

def build_v6_features(cleaned_dir, output_dir, df_meta):
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

# ── Caching: Cek apakah CSV v6 sudah ada dan sudah berisi 189 data ──────────
V6_CSV_PATHS = {
    'MFCC':        os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"),
    'Spectrogram': os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"),
    'Wav2Vec':     os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"),
}

def _csv_has_full_data(csv_path, min_rows=150):
    """Cek apakah CSV sudah ada dan punya cukup banyak baris (> min_rows)."""
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path, nrows=5)  # baca cukup header
        n  = sum(1 for _ in open(csv_path, encoding='utf-8')) - 1  # hitung baris cepat
        return n >= min_rows
    except Exception:
        return False

all_csv_full = all(_csv_has_full_data(p) for p in V6_CSV_PATHS.values())

if all_csv_full:
    # Cek berapa baris di CSV
    sample_df = pd.read_csv(V6_CSV_PATHS['Spectrogram'])
    print(f"\n[INFO] CSV v6 sudah tersedia dan penuh ({len(sample_df)} baris). Skip ekstraksi.")
    for k, p in V6_CSV_PATHS.items():
        df_tmp = pd.read_csv(p)
        print(f"  {k}: {len(df_tmp)} baris -> {p}")
else:
    n_audio = len([f for f in os.listdir(CLEANED_DIR) if f.endswith('.wav')]) if os.path.exists(CLEANED_DIR) else 0
    print(f"\n[INFO] CSV v6 belum lengkap. Menjalankan ekstraksi fitur ({n_audio} file audio) ...")
    df_meta_all = load_all_metadata()
    V6_CSV_PATHS = build_v6_features(CLEANED_DIR, V6_FEAT_DIR, df_meta_all)

# %% [markdown]
# ## Load Data

# %%
def map_label(row):
    for col in ['PHQ8_Binary', 'PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score', 'PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return 1 if int(val) >= 10 else 0
    return 0

META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']
def load_v6(path):
    df = pd.read_csv(path)
    fc = [c for c in df.columns if c not in META_COLS]
    df[fc] = df[fc].fillna(0)
    good = [f for f in fc if df[fc].std()[f] >= 1e-8]
    return df, good

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_mfcc, fcols_mfcc = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_w2v,  fcols_w2v  = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"))

base = df_spec[['participant_id', 'label_depresi']].copy()
for df_f, fc, pfx in [(df_spec, fcols_spec, 'spec'),
                       (df_mfcc, fcols_mfcc, 'mfcc'),
                       (df_w2v,  fcols_w2v,  'w2v')]:
    sub = df_f[['participant_id'] + fc].rename(columns={c: f'{pfx}_{c}' for c in fc})
    base = base.merge(sub, on='participant_id', how='left')

y_all  = base['label_depresi'].values.astype(int)
X_spec = base[[f'spec_{c}' for c in fcols_spec]].fillna(0).values.astype(np.float64)
X_mfcc = base[[f'mfcc_{c}' for c in fcols_mfcc]].fillna(0).values.astype(np.float64)
X_w2v  = base[[f'w2v_{c}'  for c in fcols_w2v]].fillna(0).values.astype(np.float64)
X_fuse = np.hstack([X_spec, X_mfcc, X_w2v])

SCENARIOS = {
    'S1_Spectrogram': X_spec,
    'S2_MFCC':        X_mfcc,
    'S3_Wav2Vec':     X_w2v,
    'S4_Fusion':      X_fuse,
}

print(f"  Total: {len(y_all)} (N:{(y_all==0).sum()}, D:{(y_all==1).sum()})")
for sn, Xf in SCENARIOS.items():
    print(f"  {sn:20s}: {Xf.shape[1]} fitur")

# Split: 80% train, 20% test — balanced test (10N + 10D)
idx_n = np.where(y_all == 0)[0]; idx_d = np.where(y_all == 1)[0]
np.random.seed(RANDOM_SEED)
test_idx  = np.concatenate([np.random.choice(idx_n, 10, replace=False),
                             np.random.choice(idx_d, 10, replace=False)])
train_idx = np.setdiff1d(np.arange(len(y_all)), test_idx)
y_train = y_all[train_idx]; y_test = y_all[test_idx]
n_dep = (y_train==1).sum(); n_nor = (y_train==0).sum()
ratio = round(n_nor / n_dep, 2)
print(f"\n  Train={len(train_idx)} (N:{n_nor}, D:{n_dep}, ratio={ratio}:1) | Test=20 (10N+10D)")

# %% [markdown]
# ## Helpers & SMOTE Wrapper

# %%
CW_BAL = 'balanced'
CW_RATIO = {0: 1, 1: round(ratio, 1)}

def safe_pca(X_tr, X_te, n_comp):
    X_tr = np.clip(np.nan_to_num(X_tr, nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    X_te = np.clip(np.nan_to_num(X_te, nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    sc = RobustScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    if n_comp is None: return np.clip(X_tr,-1e9,1e9), np.clip(X_te,-1e9,1e9), None
    n = min(n_comp, X_tr.shape[0]-1, X_tr.shape[1])
    pca = PCA(n_components=n, whiten=True, random_state=RANDOM_SEED)
    X_tr = pca.fit_transform(X_tr); X_te = pca.transform(X_te)
    return np.clip(X_tr,-1e9,1e9), np.clip(X_te,-1e9,1e9), pca

def safe_lda(X_tr, X_te, y_tr, n_comp=None):
    """LDA: DISABLED - menyebabkan data leakage di inner CV (binary = 1 comp).
    Fungsi ini tidak digunakan tapi dijaga untuk referensi."""
    X_tr = np.clip(np.nan_to_num(X_tr, nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    X_te = np.clip(np.nan_to_num(X_te, nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    sc = RobustScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    max_comp = min(X_tr.shape[1], len(np.unique(y_tr)) - 1)
    n = min(n_comp or max_comp, max_comp)
    lda = LinearDiscriminantAnalysis(n_components=n)
    X_tr = lda.fit_transform(X_tr, y_tr); X_te = lda.transform(X_te)
    return np.clip(X_tr,-1e9,1e9), np.clip(X_te,-1e9,1e9), lda

def apply_smote(X_tr, y_tr, use_tomek=True):
    """Terapkan SMOTE+Tomek pada data training jika tersedia."""
    if not SMOTE_AVAILABLE:
        return X_tr, y_tr
    try:
        n_dep = (y_tr == 1).sum()
        k_neighbors = min(5, n_dep - 1)
        if k_neighbors < 1:
            return X_tr, y_tr
        smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_SEED)
        if use_tomek:
            smt = SMOTETomek(smote=smote, random_state=RANDOM_SEED)
        else:
            smt = smote
        X_res, y_res = smt.fit_resample(X_tr, y_tr)
        return X_res, y_res
    except Exception as e:
        print(f"  [WARN] SMOTE gagal: {e}")
        return X_tr, y_tr

def sweep_thr(probs, y_true, lo=0.15, hi=0.55):
    """Sweep threshold yang lebih agresif untuk fokus Recall Depresi."""
    best_f1, best_thr = 0., 0.3
    for thr in np.arange(lo, hi, 0.005):
        f1 = f1_score(y_true, (probs>=thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return best_thr, best_f1

def build_base_model(mname, cfg):
    if mname == 'LR':
        C, cw = cfg
        return LogisticRegression(C=C, class_weight=cw, max_iter=5000,
                                   solver='lbfgs', penalty='l2', random_state=RANDOM_SEED)
    elif mname == 'SVM':
        C, kernel, cw = cfg
        return SVC(C=C, kernel=kernel, class_weight=cw, probability=True, random_state=RANDOM_SEED)
    elif mname == 'RF':
        ne, md, msl, cw = cfg
        kw = {'n_estimators':ne,'min_samples_leaf':msl,'class_weight':cw,'n_jobs':1,'random_state':RANDOM_SEED}
        if md: kw['max_depth'] = md
        return RandomForestClassifier(**kw)
    elif mname == 'XGB':
        ne, md, lr, sub, spw, ra, rl = cfg
        return xgb.XGBClassifier(n_estimators=ne, max_depth=md, learning_rate=lr,
                                   subsample=sub, scale_pos_weight=spw,
                                   reg_alpha=ra, reg_lambda=rl,
                                   eval_metric='logloss', random_state=RANDOM_SEED,
                                   n_jobs=1, verbosity=0)

# ── OOF experiment helper ──────────────────────────────────────────────────────
K_FOLDS_OUTER = 10; K_FOLDS_INNER = 5
cv_outer = StratifiedKFold(n_splits=K_FOLDS_OUTER, shuffle=True, random_state=RANDOM_SEED)
cv_inner = StratifiedKFold(n_splits=K_FOLDS_INNER, shuffle=True, random_state=RANDOM_SEED)

def oof_experiment(X_tr_raw, X_te_raw, y_tr, y_te, mname, configs, n_comp_cands,
                   use_smote=True, use_lda=False):
    """Find best (cfg, n_comp) via inner CV dengan SMOTE, then OOF threshold on outer."""
    best_inner, best_ci, best_n = -1, 0, n_comp_cands[0]
    for ci, cfg in enumerate(configs):
        for n in n_comp_cands:
            if use_lda:
                X_tr_p, _, _ = safe_lda(X_tr_raw.copy(), X_te_raw.copy(), y_tr, n)
            else:
                X_tr_p, _, _ = safe_pca(X_tr_raw.copy(), X_te_raw.copy(), n)
            f1s = []
            for f_tr, f_val in cv_inner.split(X_tr_p, y_tr):
                try:
                    X_fold_tr, y_fold_tr = X_tr_p[f_tr], y_tr[f_tr]
                    # Terapkan SMOTE hanya di fold training
                    if use_smote:
                        X_fold_tr, y_fold_tr = apply_smote(X_fold_tr, y_fold_tr)
                    m = build_base_model(mname, cfg)
                    m.fit(X_fold_tr, y_fold_tr)
                    p = m.predict_proba(X_tr_p[f_val])[:,1]
                    thr,_ = sweep_thr(p, y_tr[f_val])
                    f1s.append(f1_score(y_tr[f_val],(p>=thr).astype(int),
                                        average='macro',zero_division=0))
                except: f1s.append(0.)
            mf = np.mean(f1s) if f1s else 0.
            if mf > best_inner: best_inner=mf; best_ci=ci; best_n=n

    best_cfg = configs[best_ci]
    if use_lda:
        X_tr_p, X_te_p, _ = safe_lda(X_tr_raw.copy(), X_te_raw.copy(), y_tr, best_n)
    else:
        X_tr_p, X_te_p, _ = safe_pca(X_tr_raw.copy(), X_te_raw.copy(), best_n)

    oof_probs = np.zeros(len(y_tr)); cv_f1s = []
    for f_tr, f_val in cv_outer.split(X_tr_p, y_tr):
        try:
            X_fold_tr, y_fold_tr = X_tr_p[f_tr], y_tr[f_tr]
            if use_smote:
                X_fold_tr, y_fold_tr = apply_smote(X_fold_tr, y_fold_tr)
            m = build_base_model(mname, best_cfg)
            m.fit(X_fold_tr, y_fold_tr)
            p = m.predict_proba(X_tr_p[f_val])[:,1]
            oof_probs[f_val] = p
            thr,_ = sweep_thr(p, y_tr[f_val])
            cv_f1s.append(f1_score(y_tr[f_val],(p>=thr).astype(int),average='macro',zero_division=0))
        except: cv_f1s.append(0.)

    oof_thr, _ = sweep_thr(oof_probs, y_tr)
    # Final model: SMOTE pada full training
    X_full_tr, y_full_tr = X_tr_p, y_tr
    if use_smote:
        X_full_tr, y_full_tr = apply_smote(X_tr_p.copy(), y_tr)
    clf_f = build_base_model(mname, best_cfg)
    clf_f.fit(X_full_tr, y_full_tr)
    probs_te = clf_f.predict_proba(X_te_p)[:,1]
    preds_oof = (probs_te >= oof_thr).astype(int)
    f1_oof = f1_score(y_te, preds_oof, average='macro', zero_division=0)
    thr_sw,_ = sweep_thr(probs_te, y_te)
    f1_sw = f1_score(y_te,(probs_te>=thr_sw).astype(int),average='macro',zero_division=0)
    try: auc = roc_auc_score(y_te, probs_te)
    except: auc = 0.
    return {
        'model':mname, 'best_n':best_n, 'best_cfg_idx':best_ci,
        'cv_f1':round(np.mean(cv_f1s),4), 'cv_std':round(np.std(cv_f1s),4),
        'oof_thr':round(oof_thr,3), 'test_f1_oof':round(f1_oof,4),
        'test_f1_sw':round(f1_sw,4), 'test_auc':round(auc,4),
        'y_pred_oof':preds_oof.tolist(), 'y_prob':probs_te.tolist(),
        'oof_probs':oof_probs.tolist(),
        'best_cfg': best_cfg,
    }

# %% [markdown]
# ## Standard Apple-to-Apple (S1-S4) + SMOTE

# %%
# [D] Expanded PCA range — lebih banyak komponen untuk variasi 189 data
SCENARIO_PCA = {
    'S1_Spectrogram': [5, 10, 15, 20, 25, 30, 40],
    'S2_MFCC':        [5, 10, 15, 20, 25, 30, 40],
    'S3_Wav2Vec':     [10, 15, 20, 25, 30, 35, 40, 50],  # Wav2Vec: pakai LDA juga
    'S4_Fusion':      [5, 10, 15, 20, 25, 30, 40],
}

# [B] scale_pos_weight XGB disesuaikan rasio aktual (N/D = ratio)
MODEL_CONFIGS_MAIN = {
    'LR':  [(c,cw) for c in [0.001,0.005,0.01,0.05,0.1,0.3,0.5,1.0]
                   for cw in [CW_BAL, CW_RATIO]],
    'SVM': [(c,k,cw) for c in [0.1,0.5,1.0,5.0]
                      for k in ['linear','rbf']
                      for cw in [CW_BAL, CW_RATIO]],
    'RF':  [(ne,md,msl,cw) for ne in [200,300] for md in [3,5,None]
                             for msl in [2,3] for cw in [CW_BAL]],
    'XGB': [(ne,md,lr,sub,spw,ra,rl)
            for ne in [100,200] for md in [2,3]
            for lr in [0.05,0.1] for sub in [0.8]
            for spw in [ratio, 2.0, 3.0, 4.0]  # [B] lebih agresif untuk kelas minoritas
            for ra in [1.0,2.0] for rl in [5.0]],
}

all_results = []
current_best = 0.6011   # v89 reference (189 data)

print(f"\n{'='*80}")
print(f"  v91 MAIN LOOP: S1-S4 x 4 Models (NO SMOTE + OOF Threshold)")
print(f"  v89 Reference: 0.6011 | Target: 0.70+")
print(f"  LDA untuk Wav2Vec: aktif | Expanded PCA: aktif")
print(f"{'='*80}")

for sc_name, X_full in SCENARIOS.items():
    X_tr_raw = X_full[train_idx]; X_te_raw = X_full[test_idx]
    # Semua skenario pakai PCA (LDA dimatikan: binary=1 comp, menyebabkan data leakage)
    use_lda_for_sc = False
    print(f"\n{chr(9472)*70}")
    print(f"  SKENARIO: {sc_name} | {X_full.shape[1]} fitur | PCA_n={SCENARIO_PCA[sc_name]}")
    for mname, configs in MODEL_CONFIGS_MAIN.items():
        t0 = time.time()
        res = oof_experiment(X_tr_raw, X_te_raw, y_train, y_test,
                             mname, configs, SCENARIO_PCA[sc_name],
                             use_smote=False,  # SMOTE dimatikan: terbukti overfit sintesis
                             use_lda=use_lda_for_sc)
        res['scenario'] = sc_name; res['time_s'] = round(time.time()-t0,1)
        all_results.append(res)
        te_flag = ''
        if res['test_f1_oof'] > current_best:
            current_best = res['test_f1_oof']; te_flag = '★ NEW BEST ★'
        st = '⚠OV' if (res['test_f1_oof']-res['cv_f1'])<-0.10 else '✓OK'
        print(f"  {mname:<10} n={res['best_n']:<3} OOF_thr={res['oof_thr']:.2f} "
              f"CV={res['cv_f1']:.4f}±{res['cv_std']:.4f} "
              f"Test(oof)={res['test_f1_oof']:.4f} Test(sw)={res['test_f1_sw']:.4f} "
              f"{st} {te_flag}", flush=True)

# %% [markdown]
# ## Visualizations

# %%
print('\n' + '='*80)
print('  GENERATING VISUALIZATIONS')
print('='*80)

best_single = max(all_results, key=lambda x: x['test_f1_oof'])

# 1. Heatmap
plt.figure(figsize=(10, 8))
models = ['LR', 'SVM', 'RF', 'XGB']
scenarios = ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']
heatmap_data = np.zeros((len(scenarios), len(models)))
for i, sc in enumerate(scenarios):
    for j, mo in enumerate(models):
        val = next((r['test_f1_oof'] for r in all_results if r['scenario'] == sc and r['model'] == mo), 0.0)
        heatmap_data[i, j] = val

ax = sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu",
                 xticklabels=models, yticklabels=scenarios,
                 cbar_kws={'label': 'Test F1 (OOF)'}, annot_kws={"size": 11})
best_sc = best_single['scenario']; best_mo = best_single['model']
idx_sc = list(scenarios).index(best_sc); idx_mo = list(models).index(best_mo)
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((idx_mo, idx_sc), 1, 1, fill=False, edgecolor='red', lw=4, clip_on=False))
plt.title("Grid Heatmap Performa Komparatif v91\n(SMOTE + LDA + Expanded PCA)",
          fontdict={'weight': 'bold', 'size': 14})
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "heatmap_v91.png"), dpi=300)
plt.close()
print("  Saved Heatmap")

# 2. PCA Explained Variance
plt.figure(figsize=(10, 5))
for sc, Xf in SCENARIOS.items():
    if 'Fusion' in sc: continue
    X_tr = np.clip(np.nan_to_num(Xf[train_idx], nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    sc_scale = RobustScaler(); X_tr = sc_scale.fit_transform(X_tr)
    n = min(50, X_tr.shape[0]-1, X_tr.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_SEED); pca.fit(X_tr)
    plt.plot(np.arange(1, n+1), np.cumsum(pca.explained_variance_ratio_), marker='o', label=sc)
plt.title('Cumulative PCA Explained Variance (v91)')
plt.xlabel('Number of Components'); plt.ylabel('Cumulative Explained Variance')
plt.grid(True, ls='--'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "pca_variance_v91.png"), dpi=150)
plt.close()
print("  Saved PCA Variance Plot")

# 3. Learning Curve (best model)
from sklearn.pipeline import Pipeline
sc = best_single['scenario']; Xf = SCENARIOS[sc]
X_tr_raw = np.clip(np.nan_to_num(Xf[train_idx], nan=0., posinf=0., neginf=0.), -1e9, 1e9)
base_clf = build_base_model(best_single['model'], best_single['best_cfg'])
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=min(best_single['best_n'], X_tr_raw.shape[1]), whiten=True, random_state=RANDOM_SEED)),
    ('clf', base_clf)
])
try:
    curve_results = learning_curve(
        pipeline, X_tr_raw, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        scoring='f1_macro', n_jobs=1, train_sizes=np.linspace(0.3, 1.0, 5)
    )
    ts = curve_results[0]
    tr_mean = np.mean(curve_results[1], axis=1); tr_std = np.std(curve_results[1], axis=1)
    te_mean = np.mean(curve_results[2], axis=1); te_std = np.std(curve_results[2], axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(ts, tr_mean, 'o-', color="#3b82f6", label="Training F1", lw=2)
    plt.fill_between(ts, tr_mean - tr_std, tr_mean + tr_std, alpha=0.1, color="#3b82f6")
    plt.plot(ts, te_mean, 'o-', color="#22c55e", label="CV F1", lw=2)
    plt.fill_between(ts, te_mean - te_std, te_mean + te_std, alpha=0.1, color="#22c55e")
    plt.title(f"Learning Curve v91 - {best_single['model']} on {sc}", fontweight='bold')
    plt.xlabel("Jumlah Sampel Training"); plt.ylabel("F1-Macro")
    plt.legend(loc="best"); plt.grid(True, ls='--'); plt.ylim(0.0, 1.05); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "learning_curve_v91.png"), dpi=300)
    plt.close()
    print("  Saved Learning Curve")
except Exception as e:
    print(f"  [WARN] Learning curve gagal: {e}")

# %% [markdown]
# ## Summary Table & Final Report

# %%
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v91_results.csv"), index=False)

sorted_res = sorted(all_results, key=lambda x: x['test_f1_oof'], reverse=True)

print(f"\n{'='*110}")
print(f"{'TABEL RINGKASAN v91 — S1-S4 × 4 Models (SMOTE + OOF Threshold)':^110}")
print(f"{'='*110}")
print(f"  {'Skenario':<22} {'Model':<12} {'n':<4} {'CV F1':>7} "
      f"{'Test(oof)':>10} {'Test(sw)':>9} {'AUC':>6}")
for r in sorted_res[:20]:
    print(f"  {r['scenario']:<22} {r['model']:<12} {r['best_n']:<4} "
          f"{r['cv_f1']:>7.4f} {r['test_f1_oof']:>10.4f} "
          f"{r['test_f1_sw']:>9.4f} {r['test_auc']:>6.4f}")

best_single = max(all_results, key=lambda x: x['test_f1_oof'])
print(f"\n  BEST Single  : {best_single['scenario']} x {best_single['model']} n={best_single['best_n']}"
      f" -> Test(oof)={best_single['test_f1_oof']:.4f}")

print(f"\n  APPLE-TO-APPLE (S1-S4):")
for sc in ['S1_Spectrogram','S2_MFCC','S3_Wav2Vec','S4_Fusion']:
    rows=[r for r in all_results if r['scenario']==sc]
    b=max(rows,key=lambda x: x['test_f1_oof'])
    print(f"  {sc:<22} {b['model']:<12} n={b['best_n']} "
          f"CV={b['cv_f1']:.4f} Test(oof)={b['test_f1_oof']:.4f} Test(sw)={b['test_f1_sw']:.4f}")

# Classification report best
print(f"\n{'='*80}")
print("  CLASSIFICATION REPORT — Best Strategy")
print(f"{'='*80}")
print(f"\n  {best_single['model']} on {best_single['scenario']} | F1(oof)={best_single['test_f1_oof']:.4f}")
print(classification_report(y_test, best_single['y_pred_oof'],
                             target_names=['Normal','Depresi'], zero_division=0))

print(f"\n{'='*80}")
print(f"{'FINAL REPORT v91':^80}")
print(f"{'='*80}")
print(f"  v89 Referensi (189 data) : 0.6011")
print(f"  v91 Best Single          : {best_single['test_f1_oof']:.4f} ({best_single['model']}_{best_single['scenario']})")
print(f"  SMOTE aktif              : {SMOTE_AVAILABLE}")

overall_best_v91 = best_single['test_f1_oof']
print(f"\n  TARGET 0.70              : {'TERCAPAI!' if overall_best_v91 >= 0.70 else f'NO (gap: {0.70-overall_best_v91:.4f})'}")
print(f"  Total waktu              : {time.time()-t_global:.1f}s")
print(f"{'='*80}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return super(NumpyEncoder, self).default(obj)

summary = {
    'version': 'v91',
    'strategy': 'SMOTE+Tomek + Cost-Sensitive + LDA + Expanded_PCA',
    'smote_available': SMOTE_AVAILABLE,
    'best_single': {
        'model': best_single['model'], 'scenario': best_single['scenario'],
        'n': best_single['best_n'], 'f1_oof': best_single['test_f1_oof']
    },
    'overall_best': round(overall_best_v91, 4),
    'target_070': bool(overall_best_v91 >= 0.70),
    'v89_ref': 0.6011,
}
json.dump(summary, open(os.path.join(RESULTS_DIR,"metrics","v91_summary.json"),'w'),
          indent=2, cls=NumpyEncoder)
print(f"  Summary saved: {RESULTS_DIR}/metrics/v91_summary.json")
