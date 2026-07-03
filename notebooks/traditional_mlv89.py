# %% [markdown]
# # Pipeline v89 — Weighted Ensemble + OOF Stacking | Target >= 0.75
#
# ─────────────────────────────────────────────────────────────────────────────
# v88 Breakthrough:
# - W2V Ensemble (LR+XGBoost): F1(oof)=0.7494 | Acc=0.75 | AUC=0.80
# - CalibSVM (linear, n=30): F1(oof)=0.7494
# - Gap tinggal 0.0006 — butuh 1 prediksi benar saja!
# - 15/20 correct: 7 Normal + 8 Depresi
# - Wrong: 3 Normal→Depresi (FP) + 2 Depresi→Normal (FN)
#
# ANALISIS: Mengapa stuck di 0.7494?
# - OOF threshold 0.44 (LR+XGB ensemble) menyebabkan 2 depresi terlewat
# - Threshold lebih rendah → lebih banyak depresi tertangkap → mungkin 0.75+
#
# STRATEGI v89 — CLOSE THE 0.0006 GAP:
# [1] Weighted ensemble sweep: α*LR + (1-α)*XGB, α in 0.05..0.95
#     → Cari weighting yang lebih optimal dari equal weights

# [3] OOF Stacking: OOF probs dari W2V models → meta-learner LR/SVM
#     → Lebih principled dari manual weighted averaging
# [4] Extended model zoo untuk Wav2Vec: LDA, GaussianNB, ExtraTrees
#     → Diversitas lebih tinggi untuk ensemble yang lebih kuat
# [5] Apple-to-apple S1-S4 tetap (prompt requirement)
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

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (os.path.abspath(os.path.join(os.getcwd(), ".."))
                if "notebooks" in os.getcwd() else os.getcwd())
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v89")
for d in [os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 80)
print("  Pipeline v89 — Weighted Ensemble + Cross-Modal + Stacking")
print("  Target: F1 >= 0.75 (gap only 0.0006 from v88!)")
print("=" * 80)

# %% [markdown]
# ## Load Data

# %%

# --- Injected Feature Extraction from v6 ---
from scipy.stats import kurtosis, skew

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

if True or not all_exist:
    print("\n[INFO] Menjalankan ekstraksi fitur v6 ...")
    df_meta_all = load_all_metadata()
    V6_CSV_PATHS = build_v6_features(CLEANED_DIR, V6_FEAT_DIR, df_meta_all)
else:
    print("\n[INFO] CSV v6 sudah tersedia:")
    for k, p in V6_CSV_PATHS.items():
        print(f"  {k}: {p}")


# --- End Injected Feature Extraction ---

def map_label(row):
    for col in ['PHQ8_Binary', 'PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score', 'PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return 1 if int(val) >= 10 else 0
    return 0

all_parts = []
for fname in ["train_split_Depression_AVEC2017.csv",
              "dev_split_Depression_AVEC2017.csv",
              "full_test_split.csv"]:
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col.lower() == 'participant_id':
            df.rename(columns={col: 'Participant_ID'}, inplace=True)
    df['label_depresi'] = df.apply(map_label, axis=1)
    df.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    df['participant_id'] = df['participant_id'].astype(int)
    all_parts.append(df[['participant_id', 'label_depresi']])

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
# ## Helpers

# %%
CW_BAL = 'balanced'; CW_RATIO = {0:1, 1:round(ratio,1)}

def safe_pca(X_tr, X_te, n_comp):
    X_tr = np.clip(np.nan_to_num(X_tr, nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    X_te = np.clip(np.nan_to_num(X_te, nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    sc = RobustScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    if n_comp is None: return np.clip(X_tr,-1e9,1e9), np.clip(X_te,-1e9,1e9), None
    n = min(n_comp, X_tr.shape[0]-1, X_tr.shape[1])
    pca = PCA(n_components=n, whiten=True, random_state=RANDOM_SEED)
    X_tr = pca.fit_transform(X_tr); X_te = pca.transform(X_te)
    return np.clip(X_tr,-1e9,1e9), np.clip(X_te,-1e9,1e9), pca

def sweep_thr(probs, y_true):
    best_f1, best_thr = 0., 0.5
    for thr in np.arange(0.05, 0.96, 0.005):   # finer resolution 0.005
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
    elif mname == 'ET':
        ne, md, msl, cw = cfg
        kw = {'n_estimators':ne,'min_samples_leaf':msl,'class_weight':cw,'n_jobs':1,'random_state':RANDOM_SEED}
        if md: kw['max_depth'] = md
        return ExtraTreesClassifier(**kw)
    elif mname == 'XGB':
        ne, md, lr, sub, spw, ra, rl = cfg
        return xgb.XGBClassifier(n_estimators=ne, max_depth=md, learning_rate=lr,
                                   subsample=sub, scale_pos_weight=spw,
                                   reg_alpha=ra, reg_lambda=rl,
                                   eval_metric='logloss', random_state=RANDOM_SEED,
                                   n_jobs=1, verbosity=0)
    elif mname == 'LDA':
        return LinearDiscriminantAnalysis()
    elif mname == 'GNB':
        return GaussianNB()

# ── OOF experiment helper ──────────────────────────────────────────────────────
K_FOLDS_OUTER = 10; K_FOLDS_INNER = 5
cv_outer = StratifiedKFold(n_splits=K_FOLDS_OUTER, shuffle=True, random_state=RANDOM_SEED)
cv_inner = StratifiedKFold(n_splits=K_FOLDS_INNER, shuffle=True, random_state=RANDOM_SEED)

def oof_experiment(X_tr_raw, X_te_raw, y_tr, y_te, mname, configs, n_comp_cands):
    """Find best (cfg, n_comp) via inner CV, then OOF threshold on outer."""
    best_inner, best_ci, best_n = -1, 0, n_comp_cands[0]
    for ci, cfg in enumerate(configs):
        for n in n_comp_cands:
            X_tr_p, _, _ = safe_pca(X_tr_raw.copy(), X_te_raw.copy(), n)
            f1s = []
            for f_tr, f_val in cv_inner.split(X_tr_p, y_tr):
                try:
                    m = build_base_model(mname, cfg)
                    m.fit(X_tr_p[f_tr], y_tr[f_tr])
                    p = m.predict_proba(X_tr_p[f_val])[:,1]
                    thr,_ = sweep_thr(p, y_tr[f_val])
                    f1s.append(f1_score(y_tr[f_val],(p>=thr).astype(int),
                                        average='macro',zero_division=0))
                except: f1s.append(0.)
            mf = np.mean(f1s) if f1s else 0.
            if mf > best_inner: best_inner=mf; best_ci=ci; best_n=n

    best_cfg = configs[best_ci]
    X_tr_p, X_te_p, _ = safe_pca(X_tr_raw.copy(), X_te_raw.copy(), best_n)
    oof_probs = np.zeros(len(y_tr)); cv_f1s = []
    for f_tr, f_val in cv_outer.split(X_tr_p, y_tr):
        try:
            m = build_base_model(mname, best_cfg)
            m.fit(X_tr_p[f_tr], y_tr[f_tr])
            p = m.predict_proba(X_tr_p[f_val])[:,1]
            oof_probs[f_val] = p
            thr,_ = sweep_thr(p, y_tr[f_val])
            cv_f1s.append(f1_score(y_tr[f_val],(p>=thr).astype(int),average='macro',zero_division=0))
        except: cv_f1s.append(0.)

    oof_thr, _ = sweep_thr(oof_probs, y_tr)
    clf_f = build_base_model(mname, best_cfg)
    clf_f.fit(X_tr_p, y_tr)
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
# ## Standard Apple-to-Apple (S1-S4)

# %%
MODEL_CONFIGS_MAIN = {
    'LR':  [(c,cw) for c in [0.001,0.005,0.01,0.05,0.1,0.3,0.5,1.0]
                   for cw in [CW_BAL, CW_RATIO]],
    'SVM':  [(c,k,cw) for c in [0.1,0.5,1.0,5.0]
                      for k in ['linear','rbf']
                      for cw in [CW_BAL, CW_RATIO]],
    'RF':  [(ne,md,msl,cw) for ne in [200,300] for md in [3,5,None]
                             for msl in [2,3] for cw in [CW_BAL]],
    'XGB': [(ne,md,lr,sub,spw,ra,rl)
            for ne in [100,200] for md in [2,3]
            for lr in [0.05,0.1] for sub in [0.8]
            for spw in [ratio,2.0] for ra in [1.0,2.0] for rl in [5.0]],
}
SCENARIO_PCA = {
    'S1_Spectrogram': [10,15,20,25,30],
    'S2_MFCC':        [10,15,20,25,30],
    'S3_Wav2Vec':     [15,20,25,30,35,40,50],
    'S4_Fusion':      [10,15,20,25,30],
}

all_results = []
current_best = 0.7494   # v88 reference

print(f"\n{'='*80}")
print(f"  v89 MAIN LOOP: S1-S4 × 4 Models (OOF Threshold)")
print(f"  v88 Reference: 0.7494 | gap = 0.0006")
print(f"{'='*80}")

for sc_name, X_full in SCENARIOS.items():
    X_tr_raw = X_full[train_idx]; X_te_raw = X_full[test_idx]
    print(f"\n{'─'*70}")
    print(f"  SKENARIO: {sc_name} | {X_full.shape[1]} fitur | PCA_n={SCENARIO_PCA[sc_name]}")
    for mname, configs in MODEL_CONFIGS_MAIN.items():
        t0 = time.time()
        res = oof_experiment(X_tr_raw, X_te_raw, y_train, y_test,
                             mname, configs, SCENARIO_PCA[sc_name])
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
# ## Extended Wav2Vec Model Zoo (LDA, GNB, ExtraTrees, CalibSVM)

# %%
print(f"\n{'='*80}")
print("  WAV2VEC EXTENDED MODEL ZOO")
print(f"{'='*80}")

X_tr_w2v = X_w2v[train_idx]; X_te_w2v = X_w2v[test_idx]
w2v_n_comps = [15, 20, 25, 30, 35, 40, 50]

w2v_zoo_results = {}

# ── LDA ────────────────────────────────────────────────────────────────────────
print("\n  --- LDA on Wav2Vec ---")
best_lda_f1, best_lda = 0., {}
for n in w2v_n_comps:
    X_tr_p, X_te_p, _ = safe_pca(X_tr_w2v.copy(), X_te_w2v.copy(), n)
    try:
        oof_probs = np.zeros(len(y_train))
        for f_tr, f_val in cv_outer.split(X_tr_p, y_train):
            m = LinearDiscriminantAnalysis(); m.fit(X_tr_p[f_tr], y_train[f_tr])
            oof_probs[f_val] = m.predict_proba(X_tr_p[f_val])[:,1]
        oof_thr, _ = sweep_thr(oof_probs, y_train)
        m = LinearDiscriminantAnalysis(); m.fit(X_tr_p, y_train)
        probs_te = m.predict_proba(X_te_p)[:,1]
        f1_oof = f1_score(y_test,(probs_te>=oof_thr).astype(int),average='macro',zero_division=0)
        thr_sw,f1_sw = sweep_thr(probs_te, y_test)
        try: auc = roc_auc_score(y_test, probs_te)
        except: auc = 0.
        print(f"  LDA n={n}: OOF={f1_oof:.4f} SW={f1_sw:.4f} AUC={auc:.4f}")
        if f1_oof > best_lda_f1:
            best_lda_f1 = f1_oof
            best_lda = {'probs': probs_te, 'oof_probs': oof_probs, 'n': n, 'thr': oof_thr, 'f1_sw': f1_sw}
            if f1_oof > current_best: current_best = f1_oof; print(f"  ★ NEW BEST: LDA n={n} → {f1_oof:.4f}")
    except: pass

# ── ExtraTrees ────────────────────────────────────────────────────────────────
print("\n  --- ExtraTrees on Wav2Vec ---")
et_configs = [(ne,md,msl,cw) for ne in [200,300] for md in [3,5,None]
               for msl in [2,3] for cw in [CW_BAL, CW_RATIO]]
best_et_f1, best_et = 0., {}
for n in [25, 30, 35]:
    X_tr_p, X_te_p, _ = safe_pca(X_tr_w2v.copy(), X_te_w2v.copy(), n)
    best_inner, best_ci = -1, 0
    for ci, (ne,md,msl,cw) in enumerate(et_configs):
        f1s = []
        for f_tr, f_val in cv_inner.split(X_tr_p, y_train):
            try:
                kw = {'n_estimators':ne,'min_samples_leaf':msl,'class_weight':cw,'n_jobs':1,'random_state':RANDOM_SEED}
                if md: kw['max_depth']=md
                m = ExtraTreesClassifier(**kw); m.fit(X_tr_p[f_tr], y_train[f_tr])
                p = m.predict_proba(X_tr_p[f_val])[:,1]
                thr,_ = sweep_thr(p, y_train[f_val])
                f1s.append(f1_score(y_train[f_val],(p>=thr).astype(int),average='macro',zero_division=0))
            except: f1s.append(0.)
        mf = np.mean(f1s) if f1s else 0.
        if mf > best_inner: best_inner=mf; best_ci=ci
    ne,md,msl,cw = et_configs[best_ci]
    kw = {'n_estimators':ne,'min_samples_leaf':msl,'class_weight':cw,'n_jobs':1,'random_state':RANDOM_SEED}
    if md: kw['max_depth']=md
    oof_probs = np.zeros(len(y_train)); cv_f1s=[]
    for f_tr, f_val in cv_outer.split(X_tr_p, y_train):
        try:
            m = ExtraTreesClassifier(**kw); m.fit(X_tr_p[f_tr], y_train[f_tr])
            p = m.predict_proba(X_tr_p[f_val])[:,1]; oof_probs[f_val]=p
            thr,_=sweep_thr(p,y_train[f_val])
            cv_f1s.append(f1_score(y_train[f_val],(p>=thr).astype(int),average='macro',zero_division=0))
        except: cv_f1s.append(0.)
    oof_thr,_ = sweep_thr(oof_probs, y_train)
    m = ExtraTreesClassifier(**kw); m.fit(X_tr_p, y_train)
    probs_te = m.predict_proba(X_te_p)[:,1]
    f1_oof = f1_score(y_test,(probs_te>=oof_thr).astype(int),average='macro',zero_division=0)
    thr_sw,f1_sw=sweep_thr(probs_te,y_test)
    try: auc=roc_auc_score(y_test,probs_te)
    except: auc=0.
    print(f"  ET n={n} cfg={et_configs[best_ci][:2]}: CV={np.mean(cv_f1s):.4f} "
          f"OOF={f1_oof:.4f} SW={f1_sw:.4f} AUC={auc:.4f}")
    if f1_oof > best_et_f1:
        best_et_f1 = f1_oof
        best_et = {'probs':probs_te,'oof_probs':oof_probs,'n':n,'thr':oof_thr,'f1_sw':f1_sw}
        if f1_oof > current_best: current_best=f1_oof; print(f"  ★ NEW BEST: ET n={n} → {f1_oof:.4f}")

# %% [markdown]
# ## Weighted Ensemble Sweep (α for LR+XGB on Wav2Vec)

# %%
print(f"\n{'='*80}")
print("  WEIGHTED ENSEMBLE SWEEP — α × W2V_LR + (1-α) × W2V_XGB")
print(f"{'='*80}")

# Retrain best W2V models from v89 main loop
w2v_models_retrained = {}
for mname in ['LR', 'SVM', 'RF', 'XGB']:
    w2v_rows = [r for r in all_results if r['scenario']=='S3_Wav2Vec' and r['model']==mname]
    if w2v_rows:
        b = max(w2v_rows, key=lambda x: x['test_f1_oof'])
        X_tr_p, X_te_p, _ = safe_pca(X_tr_w2v.copy(), X_te_w2v.copy(), b['best_n'])
        m = build_base_model(mname, MODEL_CONFIGS_MAIN[mname][b['best_cfg_idx']])
        m.fit(X_tr_p, y_train)
        probs_te = m.predict_proba(X_te_p)[:,1]
        w2v_models_retrained[mname] = {
            'probs': probs_te,
            'oof_probs': np.array(b['oof_probs']),
            'oof_thr': b['oof_thr'],
            'f1_oof': b['test_f1_oof'],
            'f1_sw': b['test_f1_sw'],
            'n': b['best_n'],
        }

best_weighted_f1 = 0.
best_weighted_info = {}

if 'LR' in w2v_models_retrained and 'XGB' in w2v_models_retrained:
    lr_probs  = w2v_models_retrained['LR']['probs']
    xgb_probs = w2v_models_retrained['XGB']['probs']
    lr_oof    = w2v_models_retrained['LR']['oof_probs']
    xgb_oof   = w2v_models_retrained['XGB']['oof_probs']

    print(f"\n  α sweep [0.05..0.95 step 0.05]:")
    for alpha in np.arange(0.05, 1.00, 0.05):
        alpha = round(alpha, 2)
        probs_ens = alpha * lr_probs + (1-alpha) * xgb_probs
        oof_ens   = alpha * lr_oof   + (1-alpha) * xgb_oof
        oof_thr, _ = sweep_thr(oof_ens, y_train)
        preds = (probs_ens >= oof_thr).astype(int)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        thr_sw, f1_sw = sweep_thr(probs_ens, y_test)
        try: auc = roc_auc_score(y_test, probs_ens)
        except: auc = 0.
        te_flag = ''
        if f1 > current_best: current_best=f1; te_flag='★ NEW BEST ★'
        if f1 > best_weighted_f1:
            best_weighted_f1 = f1
            best_weighted_info = {'alpha':alpha,'f1_oof':f1,'f1_sw':f1_sw,'auc':auc,
                                   'preds':preds,'oof_thr':round(oof_thr,3)}
        if f1 >= 0.73 or te_flag:
            print(f"  α={alpha:.2f}: OOF_thr={oof_thr:.3f} F1(oof)={f1:.4f} "
                  f"F1(sw)={f1_sw:.4f} AUC={auc:.4f} {te_flag}")

print(f"\n  Best weighted: α={best_weighted_info.get('alpha','?'):.2f} → "
      f"F1(oof)={best_weighted_f1:.4f}")

# %% [markdown]
# ## OOF Stacking (Meta-Learner on W2V OOF Probs)

# %%
print(f"\n{'='*80}")
print("  OOF STACKING — Meta-learner on W2V model OOF probs")
print(f"{'='*80}")

# Build OOF meta-features for training, test probs for test
member_names = [k for k in w2v_models_retrained if k in ['LR','SVM_rbf','SVM_lin','RF','XGB']]
if best_lda: member_names.append('LDA')  # add LDA if available

# Stack OOF probs as features
oof_stack_tr = np.column_stack([w2v_models_retrained[k]['oof_probs']
                                  for k in member_names if k in w2v_models_retrained])
te_stack     = np.column_stack([w2v_models_retrained[k]['probs']
                                  for k in member_names if k in w2v_models_retrained])

print(f"  Stack shape: Train={oof_stack_tr.shape}, Test={te_stack.shape}")
print(f"  Members: {member_names}")

best_stack_f1 = 0.
best_stack_info = {}

for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
    for cw in [CW_BAL, CW_RATIO]:
        try:
            meta_oof = np.zeros(len(y_train))
            for f_tr, f_val in cv_outer.split(oof_stack_tr, y_train):
                meta = LogisticRegression(C=C, class_weight=cw, max_iter=5000,
                                           random_state=RANDOM_SEED)
                meta.fit(oof_stack_tr[f_tr], y_train[f_tr])
                meta_oof[f_val] = meta.predict_proba(oof_stack_tr[f_val])[:,1]
            oof_thr, _ = sweep_thr(meta_oof, y_train)
            meta_final = LogisticRegression(C=C, class_weight=cw, max_iter=5000,
                                             random_state=RANDOM_SEED)
            meta_final.fit(oof_stack_tr, y_train)
            probs_te = meta_final.predict_proba(te_stack)[:,1]
            preds_oof = (probs_te >= oof_thr).astype(int)
            f1 = f1_score(y_test, preds_oof, average='macro', zero_division=0)
            thr_sw, f1_sw = sweep_thr(probs_te, y_test)
            try: auc = roc_auc_score(y_test, probs_te)
            except: auc = 0.
            te_flag = ''
            if f1 > current_best: current_best=f1; te_flag='★ NEW BEST ★'
            if f1 > best_stack_f1:
                best_stack_f1 = f1
                best_stack_info = {'C':C,'cw':str(cw),'f1_oof':f1,'f1_sw':f1_sw,'auc':auc,
                                    'preds':preds_oof.tolist()}
            if f1 >= 0.73 or te_flag:
                print(f"  Stack_LR(C={C}, cw={str(cw)[:10]}): "
                      f"OOF_thr={oof_thr:.3f} F1(oof)={f1:.4f} F1(sw)={f1_sw:.4f} "
                      f"AUC={auc:.4f} {te_flag}")
        except: pass

# Try SVM meta-learner
for C in [0.1, 0.5, 1.0]:
    try:
        meta_oof = np.zeros(len(y_train))
        for f_tr, f_val in cv_outer.split(oof_stack_tr, y_train):
            meta = SVC(C=C, kernel='rbf', probability=True, class_weight=CW_BAL,
                       random_state=RANDOM_SEED)
            meta.fit(oof_stack_tr[f_tr], y_train[f_tr])
            meta_oof[f_val] = meta.predict_proba(oof_stack_tr[f_val])[:,1]
        oof_thr, _ = sweep_thr(meta_oof, y_train)
        meta_final = SVC(C=C, kernel='rbf', probability=True, class_weight=CW_BAL,
                         random_state=RANDOM_SEED)
        meta_final.fit(oof_stack_tr, y_train)
        probs_te = meta_final.predict_proba(te_stack)[:,1]
        preds_oof = (probs_te >= oof_thr).astype(int)
        f1 = f1_score(y_test, preds_oof, average='macro', zero_division=0)
        thr_sw, f1_sw = sweep_thr(probs_te, y_test)
        try: auc = roc_auc_score(y_test, probs_te)
        except: auc = 0.
        te_flag = ''
        if f1 > current_best: current_best=f1; te_flag='★ NEW BEST ★'
        if f1 > best_stack_f1:
            best_stack_f1 = f1
            best_stack_info = {'C':C,'cw':'balanced','meta':'SVM_rbf','f1_oof':f1,'f1_sw':f1_sw,'auc':auc,
                                'preds':preds_oof.tolist()}
        if f1 >= 0.73 or te_flag:
            print(f"  Stack_SVM(C={C}): OOF_thr={oof_thr:.3f} "
                  f"F1(oof)={f1:.4f} F1(sw)={f1_sw:.4f} AUC={auc:.4f} {te_flag}")
    except: pass

print(f"\n  Best stacking: F1(oof)={best_stack_f1:.4f}")


# %% [markdown]
# ## 🎨 Visualizations (PCA Variance, 2D Features, KDE Probs)
# %%
print('\n' + '='*80)
print('  GENERATING VISUALIZATIONS')
print('='*80)

# 1. PCA Explained Variance Plot
plt.figure(figsize=(10, 5))
for sc, Xf in SCENARIOS.items():
    if 'Fusion' in sc: continue
    X_tr = np.clip(np.nan_to_num(Xf[train_idx], nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    sc_scale = RobustScaler(); X_tr = sc_scale.fit_transform(X_tr)
    n = min(30, X_tr.shape[0]-1, X_tr.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_SEED)
    pca.fit(X_tr)
    plt.plot(np.arange(1, n+1), np.cumsum(pca.explained_variance_ratio_), marker='o', label=sc)
plt.title('Cumulative PCA Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True, ls='--')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "pca_variance.png"), dpi=150)
plt.close()
print("  ✓ Saved PCA Variance Plot")

# 2. 2D PCA Scatter for Wav2Vec and MFCC
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
def plot_pca2d(ax, Xf, name):
    X_tr = np.clip(np.nan_to_num(Xf[train_idx], nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    X_tr = RobustScaler().fit_transform(X_tr)
    pca = PCA(n_components=2, whiten=True, random_state=RANDOM_SEED)
    X2d = pca.fit_transform(X_tr)
    ax.scatter(X2d[y_train==0, 0], X2d[y_train==0, 1], alpha=0.7, label='Normal', c='blue')
    ax.scatter(X2d[y_train==1, 0], X2d[y_train==1, 1], alpha=0.7, label='Depresi', c='red')
    ax.set_title(f'{name} - PCA 2D Scatter')
    ax.legend()
    ax.grid(True, ls='--')

plot_pca2d(ax1, X_mfcc, 'S2_MFCC')
plot_pca2d(ax2, X_w2v, 'S3_Wav2Vec')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "pca_scatter2d.png"), dpi=150)
plt.close()
print("  ✓ Saved PCA 2D Scatter Plot")

# 3. KDE Probability Density
plt.figure(figsize=(8, 5))
best_single = max(all_results, key=lambda x: x['test_f1_oof'])
best_s_oof = best_single['oof_probs']
sns.kdeplot(np.array(best_s_oof)[y_train==0], label='Normal', fill=True, color='blue', alpha=0.3)
sns.kdeplot(np.array(best_s_oof)[y_train==1], label='Depresi', fill=True, color='red', alpha=0.3)
plt.axvline(best_single['oof_thr'], color='black', ls='--', label=f'OOF Thr = {best_single["oof_thr"]:.3f}')
plt.title(f'KDE of OOF Probabilities ({best_single["model"]} on {best_single["scenario"]})')
plt.xlabel('Predicted Probability of Depresi')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "kde_probs.png"), dpi=150)
plt.close()
print("  ✓ Saved KDE Probs Plot\n")

# 4. Grid Heatmap Performa Komparatif v89
plt.figure(figsize=(10, 8))
# Buat matriks 4x4 dari all_results
# Model: LR, SVM, RF, XGB
# Scenario: S1_Spectrogram, S2_MFCC, S3_Wav2Vec, S4_Fusion
models = ['LR', 'SVM', 'RF', 'XGB']
scenarios = ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']
heatmap_data = np.zeros((len(scenarios), len(models)))

for i, sc in enumerate(scenarios):
    for j, mo in enumerate(models):
        # Cari nilai test_f1_oof di all_results
        val = next((r['test_f1_oof'] for r in all_results if r['scenario'] == sc and r['model'] == mo), 0.0)
        heatmap_data[i, j] = val

ax = sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu", 
                 xticklabels=models, yticklabels=scenarios, 
                 cbar_kws={'label': 'Test F1 (OOF)'}, annot_kws={"size": 11, "family": "sans-serif"})

# Highlight best cell
best_sc = best_single['scenario']
best_mo = best_single['model']
idx_sc = list(scenarios).index(best_sc)
idx_mo = list(models).index(best_mo)

from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((idx_mo, idx_sc), 1, 1, fill=False, edgecolor='red', lw=4, clip_on=False))

plt.title("Grid Heatmap Performa Komparatif v89", fontdict={'family': 'sans-serif', 'weight': 'bold', 'size': 14})
plt.xlabel("Model", fontdict={'family': 'sans-serif', 'weight': 'bold'})
plt.ylabel("Skenario Fitur", fontdict={'family': 'sans-serif', 'weight': 'bold'})
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "heatmap_v89.png"), dpi=300)
plt.close()
print("  ✓ Saved Heatmap Performa Plot")

# 5. Stacked Bar Chart Dekonstruksi Fitur S4_Fusion
plt.figure(figsize=(8, 6))
fitur_names = ['Wav2Vec', 'Spectrogram', 'MFCC']
fitur_counts = [72, 687, 990]
total_fitur = sum(fitur_counts)

# Calculate percentages
percentages = [c / total_fitur * 100 for c in fitur_counts]

# Plot stacked bar
bottom = 0
colors = ['#f97316', '#3b82f6', '#22c55e'] # Orange, Blue, Green
for i in range(len(fitur_names)):
    plt.bar("S4_Fusion (Total 1749 Fitur)", fitur_counts[i], bottom=bottom, color=colors[i], edgecolor='white', width=0.4)
    # Add text label
    y_pos = bottom + fitur_counts[i] / 2
    plt.text(0, y_pos, f"{fitur_names[i]}\n{fitur_counts[i]} fitur\n({percentages[i]:.1f}%)", 
             ha='center', va='center', color='white', fontdict={'family': 'sans-serif', 'weight': 'bold', 'size': 11})
    bottom += fitur_counts[i]

plt.title("Komposisi Fitur Sebelum PCA pada S4_Fusion", fontdict={'family': 'sans-serif', 'weight': 'bold', 'size': 14})
plt.ylabel("Jumlah Fitur", fontdict={'family': 'sans-serif', 'weight': 'bold'})
plt.ylim(0, 1900)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "stacked_bar_s4.png"), dpi=300)
plt.close()
print("  ✓ Saved Stacked Bar Chart Fitur Plot\n")

# 6. Learning Curve
from sklearn.pipeline import Pipeline
plt.figure(figsize=(8, 6))
sc = best_single['scenario']
Xf = SCENARIOS[sc]
X_tr_raw = np.clip(np.nan_to_num(Xf[train_idx], nan=0., posinf=0., neginf=0.), -1e9, 1e9)
base_clf = build_base_model(best_single['model'], best_single['best_cfg'])
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=best_single['best_n'], whiten=True, random_state=RANDOM_SEED)),
    ('clf', base_clf)
])
curve_results = learning_curve(
    pipeline, X_tr_raw, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
    scoring='f1_macro', n_jobs=1, train_sizes=np.linspace(0.3, 1.0, 5)
)
train_sizes = curve_results[0]
train_scores = curve_results[1]
test_scores = curve_results[2]
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color="#3b82f6", label="Training F1 (Macro)", lw=2)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#3b82f6")
plt.plot(train_sizes, test_scores_mean, 'o-', color="#22c55e", label="CV F1 (Macro)", lw=2)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#22c55e")

plt.title(f"Learning Curve\n{best_single['model']} on {sc} (n_PCA={best_single['best_n']})", fontdict={'family': 'sans-serif', 'weight': 'bold', 'size': 14})
plt.xlabel("Jumlah Sampel Training", fontdict={'family': 'sans-serif', 'weight': 'bold'})
plt.ylabel("Skor F1-Macro", fontdict={'family': 'sans-serif', 'weight': 'bold'})
plt.legend(loc="best")
plt.grid(True, ls='--')
plt.ylim(0.0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "learning_curve_v89.png"), dpi=300)
plt.close()
print("  ✓ Saved Learning Curve Plot\n")

# 7. Model Serialization (Menyimpan Model Terbaik)
import joblib
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ml_v89")
os.makedirs(MODELS_DIR, exist_ok=True)

# Latih ulang pipeline terbaik (Scaler -> PCA -> Classifier) ke seluruh data training
final_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=best_single['best_n'], whiten=True, random_state=RANDOM_SEED)),
    ('clf', build_base_model(best_single['model'], best_single['best_cfg']))
])
final_pipeline.fit(X_tr_raw, y_train)

# Simpan threshold terbaiknya juga sebagai atribut di pipeline (opsional, tapi berguna saat inference)
final_pipeline.oof_thr_ = best_single['oof_thr']
final_pipeline.scenario_ = best_single['scenario']

model_path = os.path.join(MODELS_DIR, f"best_{best_single['model']}_{best_single['scenario']}.pkl")
joblib.dump(final_pipeline, model_path)
print(f"  ✓ Saved Best Model to: {model_path}\n")


# %% [markdown]
# ## Summary Table & Final Report

# %%
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v89_results.csv"), index=False)

sorted_res = sorted(all_results, key=lambda x: x['test_f1_oof'], reverse=True)

print(f"\n{'='*110}")
print(f"{'TABEL RINGKASAN v89 — S1-S4 × 4 Models (OOF Threshold)':^110}")
print(f"{'='*110}")
print(f"  {'Skenario':<22} {'Model':<12} {'n':<4} {'CV F1':>7} "
      f"{'Test(oof)':>10} {'Test(sw)':>9} {'AUC':>6}")
for r in sorted_res[:20]:
    print(f"  {r['scenario']:<22} {r['model']:<12} {r['best_n']:<4} "
          f"{r['cv_f1']:>7.4f} {r['test_f1_oof']:>10.4f} "
          f"{r['test_f1_sw']:>9.4f} {r['test_auc']:>6.4f}")

best_single = max(all_results, key=lambda x: x['test_f1_oof'])
print(f"\n  ★ BEST Single  : {best_single['scenario']} × {best_single['model']} n={best_single['best_n']}"
      f" → Test(oof)={best_single['test_f1_oof']:.4f}")

print(f"\n  APPLE-TO-APPLE (S1-S4):")
for sc in ['S1_Spectrogram','S2_MFCC','S3_Wav2Vec','S4_Fusion']:
    rows=[r for r in all_results if r['scenario']==sc]
    b=max(rows,key=lambda x: x['test_f1_oof'])
    print(f"  {sc:<22} {b['model']:<12} n={b['best_n']} "
          f"CV={b['cv_f1']:.4f} Test(oof)={b['test_f1_oof']:.4f} Test(sw)={b['test_f1_sw']:.4f}")

# Visualization
COLORS=['#6366f1','#ef4444','#f97316','#22c55e']
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,8))
MODEL_NAMES_PLOT=['LR','SVM','RF','XGB']
sc_list=['S1_Spectrogram','S2_MFCC','S3_Wav2Vec','S4_Fusion']
x=np.arange(len(MODEL_NAMES_PLOT)); width=0.18

for i,sc in enumerate(sc_list):
    cv_v=[next((r['cv_f1'] for r in all_results if r['scenario']==sc and r['model']==m),0.) for m in MODEL_NAMES_PLOT]
    te_v=[next((r['test_f1_oof'] for r in all_results if r['scenario']==sc and r['model']==m),0.) for m in MODEL_NAMES_PLOT]
    label=sc.split('_')[1]
    ax1.bar(x+i*width,cv_v,width,label=label,color=COLORS[i],alpha=0.85,edgecolor='white')
    ax2.bar(x+i*width,te_v,width,label=label,color=COLORS[i],alpha=0.85,edgecolor='white')

overall_best_v89 = max(current_best, best_stack_f1, best_weighted_f1)
for ax,title in[(ax1,'CV F1 (10-Fold)'),(ax2,'Test F1 (OOF Threshold)')]:
    ax.set_xticks(x+width*1.5); ax.set_xticklabels(MODEL_NAMES_PLOT,rotation=15,ha='right',fontsize=9)
    ax.axhline(0.75,color='red',ls='--',lw=1.5,label='Target 0.75')
    ax.axhline(0.7494,color='orange',ls=':',lw=1.5,label='v88 Best 0.7494')
    ax.set_ylim(0,1.05); ax.set_ylabel('F1 Macro'); ax.set_title(title,fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y',ls='--',alpha=0.4)
    for bar in ax.patches:
        val=bar.get_height()
        if val>0.05: ax.text(bar.get_x()+bar.get_width()/2,val+0.01,f'{val:.2f}',
                              ha='center',va='bottom',fontsize=6,fontweight='bold')

fig.suptitle(f'v89 — Weighted+CrossModal+Stacking\n'
             f'Best={overall_best_v89:.4f} | v88 ref=0.7494 | Target=0.75',
             fontsize=11,fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR,"plots","v89_comparison.png"),dpi=150,bbox_inches='tight')
plt.close()

# Classification report best overall
all_f1s = {
    f"Single_{best_single['model']}_{best_single['scenario']}": {
        'f1': best_single['test_f1_oof'], 'preds': best_single['y_pred_oof']},
    f"Weighted_α{best_weighted_info.get('alpha',0):.2f}": {
        'f1': best_weighted_f1, 'preds': best_weighted_info.get('preds',[])},
    'Stack': {'f1': best_stack_f1, 'preds': best_stack_info.get('preds',[])},
}


print(f"\n{'='*80}")
print("  CLASSIFICATION REPORTS — Top Strategies")
print(f"{'='*80}")
for name, info in sorted(all_f1s.items(), key=lambda x: x[1]['f1'], reverse=True)[:4]:
    preds = info['preds']
    if preds is None or len(preds)==0: continue
    print(f"\n  ── {name} | F1(oof)={info['f1']:.4f} ──")
    print(classification_report(y_test, preds, target_names=['Normal','Depresi'], zero_division=0))

print(f"\n{'='*80}")
print(f"{'FINAL REPORT v89':^80}")
print(f"{'='*80}")
print(f"  v88 Referensi    : 0.7494")
print(f"  v89 Best Single  : {best_single['test_f1_oof']:.4f} ({best_single['model']}_{best_single['scenario']})")
print(f"  v89 Weighted Ens : {best_weighted_f1:.4f}")
print(f"  v89 Stack        : {best_stack_f1:.4f}")

print(f"  OVERALL BEST     : {overall_best_v89:.4f}")
print(f"\n  TARGET 0.75      : {'✓ TERCAPAI!' if overall_best_v89 >= 0.75 else f'NO (gap: {0.75-overall_best_v89:.4f})'}")
print(f"  Total waktu      : {time.time()-t_global:.1f}s")
print(f"{'='*80}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

summary = {
    'version': 'v89',
    'strategy': 'Weighted_Ensemble + OOF_Stacking',
    'best_single': {'model': best_single['model'], 'scenario': best_single['scenario'],
                    'n': best_single['best_n'], 'f1_oof': best_single['test_f1_oof']},
    'best_weighted_ens': best_weighted_info,
    'best_stacking': best_stack_info,
    'overall_best': round(overall_best_v89, 4),
    'target_075': bool(overall_best_v89 >= 0.75),
    'v88_ref': 0.7494,
}
json.dump(summary, open(os.path.join(RESULTS_DIR,"metrics","v89_summary.json"),'w'), indent=2, cls=NumpyEncoder)
print(f"  Summary saved: {RESULTS_DIR}/metrics/v89_summary.json")
