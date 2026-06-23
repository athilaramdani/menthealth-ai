# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v7** — Klasifikasi Kesehatan Mental Berbasis Audio
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v7 = v6 + Feature Fusion + Dual Split
#
#  [1] Reuse fitur v6 (participant-level, Wav2Vec NYATA)
#      → MFCC (990 fitur) + Spectrogram (687) + Wav2Vec (72)
#
#  [2] Feature Fusion BARU
#      Gabungkan MFCC + Spectrogram + Wav2Vec → 1 vector per participant
#      → PCA dimensionality reduction → compact representation
#
#  [3] Dual Split
#      Mode A: AVEC2017 official (64 train / 15 val / 23 test)
#      Mode B: 80/10/10 × 5 repeated stratified
#
#  [4] 16 model = 4 Feature Type × 4 Classifier
#      Feature: MFCC, Spectrogram, Wav2Vec, Fused
#      Model:   LR, SVM, XGBoost, Random Forest
#
#  [5] Tetap: class_weight='balanced', threshold 0.5, NO SMOTE
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

# %%
import subprocess, sys

def _pip_install(pkg, name=None, upgrade=False):
    check = name or pkg.split('[')[0].split('>=')[0].split('==')[0]
    try:
        __import__(check)
        if not upgrade: return
    except ImportError: pass
    except Exception: return
    print(f"[Installing] {pkg} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"] + (["--upgrade"] if upgrade else []))
        print(f"[OK] {pkg}")
    except Exception as e:
        print(f"[WARN] {e}")

_pip_install("librosa")
_pip_install("scikit-learn", "sklearn")
_pip_install("xgboost")
_pip_install("seaborn")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("[OK] Dependencies siap.\n")

# %%
import os, pickle, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

plt.rcParams['font.family'] = 'DejaVu Sans'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─── Path ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd()
    else os.getcwd()
)
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v7")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v7")

for d in [MODELS_DIR,
          os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

print(f"Project root : {PROJECT_ROOT}")
print(f"V6 features  : {V6_FEAT_DIR}")
print(f"Results      : {RESULTS_DIR}")

# %% [markdown]
# ## 1. Load Fitur v6 + Build Feature Fusion

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_clean(csv_path, feat_name):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    # Hapus fitur konstan
    std_vals = df[feat_cols].std()
    const = std_vals[std_vals < 1e-8].index.tolist()
    feat_cols = [f for f in feat_cols if f not in const]
    print(f"  [{feat_name}] {len(df)} participants, {len(feat_cols)} fitur")
    return df, feat_cols

# Load individual
df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"), "MFCC")
df_spec, cols_spec = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"), "Spectrogram")
df_w2v,  cols_w2v  = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"), "Wav2Vec")

# Build Fused: merge on participant_id
df_fused = df_mfcc[META_COLS + cols_mfcc].copy()
# Rename cols agar tidak tabrakan
spec_renamed = {c: f'spec_{c}' for c in cols_spec}
w2v_renamed  = {c: f'w2v_{c}'  for c in cols_w2v}

df_spec_r = df_spec[['participant_id'] + cols_spec].rename(columns=spec_renamed)
df_w2v_r  = df_w2v[['participant_id']  + cols_w2v].rename(columns=w2v_renamed)

df_fused = df_fused.merge(df_spec_r, on='participant_id', how='inner')
df_fused = df_fused.merge(df_w2v_r, on='participant_id', how='inner')
cols_fused = cols_mfcc + list(spec_renamed.values()) + list(w2v_renamed.values())

# Remove constant in fused
std_f = df_fused[cols_fused].std()
const_f = std_f[std_f < 1e-8].index.tolist()
cols_fused = [c for c in cols_fused if c not in const_f]

print(f"  [Fused] {len(df_fused)} participants, {len(cols_fused)} fitur (sebelum PCA)")

# Store all datasets
datasets = {
    'MFCC':        (df_mfcc, cols_mfcc),
    'Spectrogram': (df_spec, cols_spec),
    'Wav2Vec':     (df_w2v,  cols_w2v),
    'Fused':       (df_fused, cols_fused),
}

FEAT_NAMES  = list(datasets.keys())

# %% [markdown]
# ## 2. Model Config

# %%
def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED, class_weight='balanced',
            C=1.0, solver='lbfgs',
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

MODEL_NAMES = list(get_models().keys())
print(f"Model: {MODEL_NAMES}")
print(f"Feature: {FEAT_NAMES}")
print(f"Total: {len(MODEL_NAMES)} x {len(FEAT_NAMES)} = {len(MODEL_NAMES)*len(FEAT_NAMES)} model per split mode")

# %% [markdown]
# ## 3. Preprocessing & Evaluation Utils

# %%
def preprocess(X_tr, X_va, X_te, use_pca=False, pca_components=100):
    """NaN fill, clip outlier, scale, optional PCA."""
    # NaN -> median
    medians = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_va, X_te]:
        for col_i in range(X.shape[1]):
            mask = np.isnan(X[:, col_i])
            X[mask, col_i] = medians[col_i]

    # Clip IQR x 10
    Q1 = np.percentile(X_tr, 25, axis=0)
    Q3 = np.percentile(X_tr, 75, axis=0)
    IQR = Q3 - Q1
    lo, hi = Q1 - 10 * IQR, Q3 + 10 * IQR
    for X in [X_tr, X_va, X_te]:
        np.clip(X, lo, hi, out=X)

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    # PCA (hanya untuk Fused) — n_components dinamis agar tidak melebihi n_samples
    pca = None
    if use_pca and X_tr.shape[1] > pca_components:
        max_comp = min(pca_components, X_tr.shape[0] - 1, X_tr.shape[1])
        pca = PCA(n_components=max_comp, random_state=RANDOM_SEED)
        X_tr = pca.fit_transform(X_tr)
        X_va = pca.transform(X_va)
        X_te = pca.transform(X_te)

    return X_tr, X_va, X_te, scaler, pca


def calc_metrics(y_true, y_pred, y_prob=None):
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    except Exception:
        auc = 0.0
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1_macro':  float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'roc_auc':   auc,
    }


def train_eval(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    try:
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    except Exception:
        y_pred = model.predict(X_te)
        y_prob = y_pred.astype(float)
    return calc_metrics(y_te, y_pred, y_prob), y_pred, y_prob

# %% [markdown]
# ## 4. Split Definitions

# %%
# ─── A: AVEC2017 Official ─────────────────────────────────────────────────
def load_avec_pids():
    """Return dict of split -> list of participant_ids."""
    splits = {}
    for fname, split in [
        ("train_split_Depression_AVEC2017.csv", "train"),
        ("dev_split_Depression_AVEC2017.csv",   "dev"),
        ("full_test_split.csv",                  "test"),
    ]:
        df = pd.read_csv(os.path.join(RAW_DIR, fname))
        df.columns = [c.strip() for c in df.columns]
        pid_col = [c for c in df.columns if 'participant' in c.lower()][0]
        splits[split] = df[pid_col].astype(int).tolist()
    return splits

avec_pids = load_avec_pids()
print("AVEC2017 split:")
for s, pids in avec_pids.items():
    print(f"  {s}: {len(pids)} participants")


# ─── B: 80/10/10 Repeated Stratified ──────────────────────────────────────
N_REPEATS   = 5
TRAIN_RATIO = 0.80

def make_repeated_splits(df, n_repeats=5, seed=42):
    labels = df['label_depresi'].values
    splits = []
    for r in range(n_repeats):
        rng = np.random.RandomState(seed + r)
        idx_all = np.arange(len(df))
        idx_0, idx_1 = idx_all[labels == 0], idx_all[labels == 1]
        rng.shuffle(idx_0); rng.shuffle(idx_1)
        n0 = int(len(idx_0) * TRAIN_RATIO)
        n1 = int(len(idx_1) * TRAIN_RATIO)
        tr = np.concatenate([idx_0[:n0], idx_1[:n1]])
        rest_0, rest_1 = idx_0[n0:], idx_1[n1:]
        nv0, nv1 = len(rest_0) // 2, len(rest_1) // 2
        va = np.concatenate([rest_0[:nv0], rest_1[:nv1]])
        te = np.concatenate([rest_0[nv0:], rest_1[nv1:]])
        splits.append((tr, va, te))
    return splits

print(f"\n80/10/10 x {N_REPEATS} repeated stratified splits")

# %% [markdown]
# ## 5A. Training — AVEC2017 Official Split

# %%
SEP = "=" * 95
PCA_COMPONENTS = 100

print(f"\n{'#' * 95}")
print(f"{'MODE A: AVEC2017 OFFICIAL SPLIT (64/15/23)':^95}")
print(f"{'#' * 95}")

avec_results = {}
avec_ys = {}

for feat_name in FEAT_NAMES:
    df, feat_cols = datasets[feat_name]
    use_pca = (feat_name == 'Fused')

    # Map PIDs to indices
    pid_list = df['participant_id'].tolist()
    pid2idx = {pid: i for i, pid in enumerate(pid_list)}

    tr_idx = [pid2idx[p] for p in avec_pids['train'] if p in pid2idx]
    va_idx = [pid2idx[p] for p in avec_pids['dev']   if p in pid2idx]
    te_idx = [pid2idx[p] for p in avec_pids['test']  if p in pid2idx]

    X_all = df[feat_cols].values
    y_all = df['label_depresi'].values

    X_tr, X_va, X_te = X_all[tr_idx], X_all[va_idx], X_all[te_idx]
    y_tr, y_va, y_te = y_all[tr_idx], y_all[va_idx], y_all[te_idx]

    X_tr, X_va, X_te, scaler, pca = preprocess(X_tr, X_va, X_te, use_pca=use_pca, pca_components=PCA_COMPONENTS)

    pca_tag = f" (PCA {X_tr.shape[1]})" if pca else ""
    print(f"\n{SEP}")
    print(f"  FEATURE: {feat_name}  |  {X_tr.shape[1]} fitur{pca_tag}  "
          f"|  Train={len(tr_idx)} Val={len(va_idx)} Test={len(te_idx)}")
    print(SEP)

    for model_name in MODEL_NAMES:
        combo = f"{feat_name} + {model_name}"
        model = get_models()[model_name]
        metrics, y_pred, y_prob = train_eval(model, X_tr, y_tr, X_te, y_te)

        # Val metrics too
        _, y_va_pred, y_va_prob = train_eval(get_models()[model_name], X_tr, y_tr, X_va, y_va)
        val_m = calc_metrics(y_va, y_va_pred, y_va_prob)

        avec_results[combo] = {**metrics, 'val_f1': val_m['f1_macro']}
        avec_ys[combo] = (y_te, y_pred)

        print(f"\n  [{combo}]")
        print(f"    Val  F1 : {val_m['f1_macro']:.4f}")
        print(f"    Test F1 : {metrics['f1_macro']:.4f}  |  Acc: {metrics['accuracy']:.4f}  "
              f"|  AUC: {metrics['roc_auc']:.4f}")

        # Save model
        sf = feat_name.lower().replace(' ', '_')
        sm = model_name.lower().replace(' ', '_')
        retrain_model = get_models()[model_name]
        retrain_model.fit(X_tr, y_tr)
        with open(os.path.join(MODELS_DIR, f"v7_avec_{sf}_{sm}.pkl"), 'wb') as fp:
            pickle.dump({'model': retrain_model, 'scaler': scaler, 'pca': pca, 'feat_cols': feat_cols}, fp)

print(f"\n{SEP}")
print(f"  AVEC2017: 16 MODEL SELESAI")
print(SEP)

# %% [markdown]
# ## 5B. Training — 80/10/10 × 5 Repeats

# %%
print(f"\n{'#' * 95}")
print(f"{'MODE B: 80/10/10 x 5 REPEATED STRATIFIED':^95}")
print(f"{'#' * 95}")

repeat_results = {}  # combo -> list of metric dicts

for feat_name in FEAT_NAMES:
    df, feat_cols = datasets[feat_name]
    use_pca = (feat_name == 'Fused')

    splits = make_repeated_splits(df, n_repeats=N_REPEATS, seed=RANDOM_SEED)

    print(f"\n{SEP}")
    print(f"  FEATURE: {feat_name}  |  {len(feat_cols)} fitur  |  {len(df)} participants")
    print(SEP)

    for model_name in MODEL_NAMES:
        combo = f"{feat_name} + {model_name}"
        metrics_list = []

        for r, (tr_idx, va_idx, te_idx) in enumerate(splits):
            X_all = df[feat_cols].values
            y_all = df['label_depresi'].values
            X_tr, X_va, X_te = X_all[tr_idx], X_all[va_idx], X_all[te_idx]
            y_tr, y_va, y_te = y_all[tr_idx], y_all[va_idx], y_all[te_idx]

            X_tr, X_va, X_te, _, _ = preprocess(X_tr, X_va, X_te, use_pca=use_pca, pca_components=PCA_COMPONENTS)

            model = get_models()[model_name]
            m, _, _ = train_eval(model, X_tr, y_tr, X_te, y_te)
            metrics_list.append(m)

        repeat_results[combo] = metrics_list

        f1s = [m['f1_macro'] for m in metrics_list]
        accs = [m['accuracy'] for m in metrics_list]
        aucs = [m['roc_auc'] for m in metrics_list]
        print(f"\n  [{combo}]")
        print(f"    Test F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  "
              f"(min={np.min(f1s):.4f}, max={np.max(f1s):.4f})")
        print(f"    Acc:     {np.mean(accs):.4f}  |  AUC: {np.mean(aucs):.4f}")

print(f"\n{SEP}")
print(f"  80/10/10: 16 MODEL × 5 REPEATS SELESAI")
print(SEP)

# %% [markdown]
# ## 6. Tabel Perbandingan Lengkap

# %%
# ─── A: AVEC2017 ──────────────────────────────────────────────────────────
rows_a = []
for combo, m in avec_results.items():
    parts = combo.split(' + ')
    rows_a.append({
        'Feature': parts[0], 'Model': parts[1],
        'Test F1': round(m['f1_macro'], 4),
        'Test Acc': round(m['accuracy'], 4),
        'Test AUC': round(m['roc_auc'], 4),
        'Val F1': round(m['val_f1'], 4),
    })

df_avec = (pd.DataFrame(rows_a)
           .sort_values('Test F1', ascending=False)
           .reset_index(drop=True))
df_avec.index += 1

print("\n" + "=" * 110)
print(f"{'RINGKASAN v7 -- MODE A: AVEC2017 (64/15/23), 16 Model':^110}")
print("=" * 110)
print(df_avec.to_string())
csv_a = os.path.join(RESULTS_DIR, "metrics", "v7_avec2017_results.csv")
df_avec.to_csv(csv_a, index=False)
print(f"\nDisimpan: {csv_a}")

best_a = df_avec.iloc[0]
print(f"\n  BEST AVEC: {best_a['Feature']} + {best_a['Model']}  |  Test F1: {best_a['Test F1']}")

# ─── B: 80/10/10 ─────────────────────────────────────────────────────────
rows_b = []
for combo, mlist in repeat_results.items():
    parts = combo.split(' + ')
    f1s = [m['f1_macro'] for m in mlist]
    accs = [m['accuracy'] for m in mlist]
    aucs = [m['roc_auc'] for m in mlist]
    rows_b.append({
        'Feature': parts[0], 'Model': parts[1],
        'F1 Mean': round(np.mean(f1s), 4),
        'F1 Std': round(np.std(f1s), 4),
        'F1 Max': round(np.max(f1s), 4),
        'Acc Mean': round(np.mean(accs), 4),
        'AUC Mean': round(np.mean(aucs), 4),
    })

df_repeat = (pd.DataFrame(rows_b)
             .sort_values('F1 Mean', ascending=False)
             .reset_index(drop=True))
df_repeat.index += 1

print("\n" + "=" * 110)
print(f"{'RINGKASAN v7 -- MODE B: 80/10/10 x 5 Repeats, 16 Model':^110}")
print("=" * 110)
print(df_repeat.to_string())
csv_b = os.path.join(RESULTS_DIR, "metrics", "v7_80_10_10_results.csv")
df_repeat.to_csv(csv_b, index=False)
print(f"\nDisimpan: {csv_b}")

best_b = df_repeat.iloc[0]
print(f"\n  BEST 80/10/10: {best_b['Feature']} + {best_b['Model']}  "
      f"|  F1 Mean: {best_b['F1 Mean']} +/- {best_b['F1 Std']}  (max={best_b['F1 Max']})")

# ─── Side-by-side best ───────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"{'PERBANDINGAN BEST MODEL — AVEC vs 80/10/10':^110}")
print("=" * 110)
print(f"  AVEC2017   : {best_a['Feature']} + {best_a['Model']}  →  Test F1 = {best_a['Test F1']}")
print(f"  80/10/10   : {best_b['Feature']} + {best_b['Model']}  →  F1 Mean = {best_b['F1 Mean']}  (max={best_b['F1 Max']})")
print("=" * 110)

# %% [markdown]
# ## 7. Visualisasi

# %%
COLORS_FEAT  = {'MFCC': '#3b82f6', 'Spectrogram': '#f59e0b', 'Wav2Vec': '#10b981', 'Fused': '#8b5cf6'}
COLORS_MODEL = ['#6366f1', '#ef4444', '#f97316', '#22c55e']

# ─── 7A. Grouped Bar — AVEC2017 ──────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
fig.suptitle('v7 — AVEC2017 Test Macro F1 (16 Model, Threshold=0.5)',
             fontsize=13, fontweight='bold', y=1.02)

x = np.arange(len(MODEL_NAMES))
for ax_idx, fn in enumerate(FEAT_NAMES):
    ax = axes[ax_idx]
    vals = []
    for mn in MODEL_NAMES:
        combo = f"{fn} + {mn}"
        vals.append(avec_results.get(combo, {}).get('f1_macro', 0.0))
    bars = ax.bar(x, vals, width=0.6, color=COLORS_MODEL, edgecolor='white', linewidth=0.7)
    ax.set_title(fn, fontweight='bold', fontsize=12, color=COLORS_FEAT[fn])
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, rotation=20, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Test Macro F1' if ax_idx == 0 else '')
    ax.axhline(0.7, color='red', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
p1 = os.path.join(RESULTS_DIR, "plots", "v7_avec_bar.png")
fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.show()
print(f"Plot: {p1}")

# ─── 7B. Box Plot — 80/10/10 ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
box_data, box_labels = [], []
for fn in FEAT_NAMES:
    for mn in MODEL_NAMES:
        combo = f"{fn} + {mn}"
        f1s = [m['f1_macro'] for m in repeat_results.get(combo, [{'f1_macro': 0}])]
        box_data.append(f1s)
        box_labels.append(f"{fn[:4]}+{mn[:3]}")

bp = ax.boxplot(box_data, patch_artist=True, labels=box_labels)
colors_16 = [COLORS_FEAT[fn] for fn in FEAT_NAMES for _ in MODEL_NAMES]
for patch, c in zip(bp['boxes'], colors_16):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.axhline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Target 0.70')
ax.set_ylabel('Test Macro F1')
ax.set_title('v7 — F1 Distribution (80/10/10 x 5 Repeats, 16 Model)', fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend()
plt.tight_layout()
p2 = os.path.join(RESULTS_DIR, "plots", "v7_boxplot.png")
fig.savefig(p2, dpi=150, bbox_inches='tight'); plt.show()
print(f"Boxplot: {p2}")

# ─── 7C. Heatmap AVEC ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
for ax, metric, title, cmap in zip(axes, ['f1_macro', 'roc_auc'],
                                     ['Test Macro F1', 'Test ROC-AUC'], ['YlOrRd', 'Blues']):
    data = {}
    for fn in FEAT_NAMES:
        data[fn] = [avec_results.get(f"{fn} + {mn}", {}).get(metric, 0) for mn in MODEL_NAMES]
    hdf = pd.DataFrame(data, index=MODEL_NAMES).T
    sns.heatmap(hdf, annot=True, fmt='.3f', cmap=cmap, linewidths=0.5,
                cbar_kws={'label': title}, ax=ax, vmin=0.3, vmax=1.0)
    ax.set_title(f'{title} — AVEC2017 (v7)', fontweight='bold')
plt.tight_layout()
p3 = os.path.join(RESULTS_DIR, "plots", "v7_heatmap_avec.png")
fig.savefig(p3, dpi=150, bbox_inches='tight'); plt.show()
print(f"Heatmap: {p3}")

# ─── 7D. Heatmap 80/10/10 Mean ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
for ax, metric, title, cmap in zip(axes, ['f1_macro', 'roc_auc'],
                                     ['Mean Test F1', 'Mean Test AUC'], ['YlOrRd', 'Blues']):
    data = {}
    for fn in FEAT_NAMES:
        vals = []
        for mn in MODEL_NAMES:
            combo = f"{fn} + {mn}"
            ms = repeat_results.get(combo, [])
            vals.append(np.mean([m[metric] for m in ms]) if ms else 0)
        data[fn] = vals
    hdf = pd.DataFrame(data, index=MODEL_NAMES).T
    sns.heatmap(hdf, annot=True, fmt='.3f', cmap=cmap, linewidths=0.5,
                cbar_kws={'label': title}, ax=ax, vmin=0.3, vmax=1.0)
    ax.set_title(f'{title} — 80/10/10 (v7)', fontweight='bold')
plt.tight_layout()
p4 = os.path.join(RESULTS_DIR, "plots", "v7_heatmap_80_10_10.png")
fig.savefig(p4, dpi=150, bbox_inches='tight'); plt.show()
print(f"Heatmap: {p4}")

# %% [markdown]
# ## 8. Classification Report — AVEC2017

# %%
class_labels = ['Normal (0)', 'Depresi (1)']

print("\n" + "=" * 100)
print(f"{'CLASSIFICATION REPORT v7 — AVEC2017 (16 Model)':^100}")
print("=" * 100)

for feat_name in FEAT_NAMES:
    df, feat_cols = datasets[feat_name]
    use_pca = (feat_name == 'Fused')
    pid_list = df['participant_id'].tolist()
    pid2idx = {pid: i for i, pid in enumerate(pid_list)}

    tr_idx = [pid2idx[p] for p in avec_pids['train'] if p in pid2idx]
    va_idx = [pid2idx[p] for p in avec_pids['dev']   if p in pid2idx]
    te_idx = [pid2idx[p] for p in avec_pids['test']  if p in pid2idx]

    X_all, y_all = df[feat_cols].values, df['label_depresi'].values
    X_tr, X_va, X_te = X_all[tr_idx], X_all[va_idx], X_all[te_idx]
    y_tr, y_va, y_te = y_all[tr_idx], y_all[va_idx], y_all[te_idx]
    X_tr, X_va, X_te, _, _ = preprocess(X_tr, X_va, X_te, use_pca=use_pca, pca_components=PCA_COMPONENTS)

    print(f"\n{'-'*100}\n  FEATURE: {feat_name}\n{'-'*100}")
    for model_name in MODEL_NAMES:
        combo = f"{feat_name} + {model_name}"
        model = get_models()[model_name]
        model.fit(X_tr, y_tr)
        try:
            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        except Exception:
            y_pred = model.predict(X_te)
        m = avec_results[combo]
        print(f"\n  [{model_name}]  Test F1={m['f1_macro']:.4f}")
        print(classification_report(y_te, y_pred, labels=[0,1],
                                    target_names=class_labels, zero_division=0))

# ─── Confusion Matrix Grid — AVEC2017 ───────────────────────────────────
CMAPS = {'MFCC': 'Blues', 'Spectrogram': 'Oranges', 'Wav2Vec': 'Greens', 'Fused': 'Purples'}
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle('v7 — Confusion Matrix (AVEC2017, Threshold=0.5)', fontsize=13, fontweight='bold')

for fn_idx, feat_name in enumerate(FEAT_NAMES):
    df, feat_cols = datasets[feat_name]
    use_pca = (feat_name == 'Fused')
    pid_list = df['participant_id'].tolist()
    pid2idx = {pid: i for i, pid in enumerate(pid_list)}
    tr_idx = [pid2idx[p] for p in avec_pids['train'] if p in pid2idx]
    te_idx = [pid2idx[p] for p in avec_pids['test']  if p in pid2idx]
    va_idx = [pid2idx[p] for p in avec_pids['dev']   if p in pid2idx]
    X_all, y_all = df[feat_cols].values, df['label_depresi'].values
    X_tr, X_va, X_te = X_all[tr_idx], X_all[va_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    X_tr, X_va, X_te, _, _ = preprocess(X_tr, X_va, X_te, use_pca=use_pca, pca_components=PCA_COMPONENTS)

    for mn_idx, model_name in enumerate(MODEL_NAMES):
        ax = axes[fn_idx, mn_idx]
        model = get_models()[model_name]
        model.fit(X_tr, y_tr)
        try:
            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        except:
            y_pred = model.predict(X_te)
        cm = confusion_matrix(y_te, y_pred, labels=[0,1])
        combo = f"{feat_name} + {model_name}"
        f1 = avec_results[combo]['f1_macro']
        sns.heatmap(cm, annot=True, fmt='d', cmap=CMAPS[feat_name],
                    ax=ax, xticklabels=class_labels, yticklabels=class_labels,
                    linewidths=0.5, cbar=False)
        ax.set_title(f'{feat_name[:5]}+{model_name[:8]}\nF1={f1:.3f}', fontweight='bold', fontsize=8)
        ax.set_xlabel('Pred', fontsize=7); ax.set_ylabel('True', fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97])
p5 = os.path.join(RESULTS_DIR, "confusion_matrix", "v7_avec_cm.png")
fig.savefig(p5, dpi=150, bbox_inches='tight'); plt.show()
print(f"CM: {p5}")

# %% [markdown]
# ## 9. Ringkasan Final

# %%
summary = {
    'version': 'v7',
    'features': FEAT_NAMES,
    'models': MODEL_NAMES,
    'total_models': len(FEAT_NAMES) * len(MODEL_NAMES),
    'pca_components': PCA_COMPONENTS,
    'threshold': 0.5,
    'smote': False, 'augmentation': False,
    'avec2017': {
        combo: {k: round(v, 4) for k, v in m.items()} for combo, m in avec_results.items()
    },
    'repeated_80_10_10': {
        combo: {
            'f1_mean': round(float(np.mean([m['f1_macro'] for m in mlist])), 4),
            'f1_std':  round(float(np.std([m['f1_macro'] for m in mlist])), 4),
            'f1_max':  round(float(np.max([m['f1_macro'] for m in mlist])), 4),
        } for combo, mlist in repeat_results.items()
    },
}

# Best per mode
best_avec_combo = max(avec_results, key=lambda k: avec_results[k]['f1_macro'])
best_rep_combo  = max(repeat_results, key=lambda k: np.mean([m['f1_macro'] for m in repeat_results[k]]))

summary['best_avec'] = {
    'model': best_avec_combo,
    'f1': round(avec_results[best_avec_combo]['f1_macro'], 4)
}
summary['best_repeated'] = {
    'model': best_rep_combo,
    'f1_mean': round(float(np.mean([m['f1_macro'] for m in repeat_results[best_rep_combo]])), 4),
    'f1_max':  round(float(np.max([m['f1_macro'] for m in repeat_results[best_rep_combo]])), 4),
}

with open(os.path.join(MODELS_DIR, "v7_summary.json"), 'w') as fp:
    json.dump(summary, fp, indent=2)

print(f"\n{'=' * 95}")
print(f"{'PIPELINE v7 SELESAI':^95}")
print(f"{'=' * 95}")
print(f"\n  Mode A (AVEC2017):")
print(f"    Best: {best_avec_combo}")
print(f"    F1:   {summary['best_avec']['f1']}")
print(f"\n  Mode B (80/10/10 x 5):")
print(f"    Best: {best_rep_combo}")
print(f"    F1:   {summary['best_repeated']['f1_mean']} +/- "
      f"{round(float(np.std([m['f1_macro'] for m in repeat_results[best_rep_combo]])), 4)}  "
      f"(max={summary['best_repeated']['f1_max']})")
print(f"\n  Models : {MODELS_DIR}")
print(f"  Results: {RESULTS_DIR}")
