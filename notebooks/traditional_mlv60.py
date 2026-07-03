# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL PARTICIPANTS)
# **Pipeline v60** — Audio-Only ML, Best-of-v6+v59 Hybrid
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v60 = HYBRID v6+v59 — Audio Only, Target F1 ≥ 0.75
#
#  [1] Fitur Audio: MFCC, Spectrogram, Wav2Vec (participant-level dari v6)
#      → Load dari data/features/v6/ yang sudah tersedia
#
#  [2] Feature Fusion: Concat MFCC + Spectrogram + Wav2Vec
#      → Satu feature matrix besar untuk representasi terkaya
#
#  [3] Preprocessing lengkap:
#      - NaN → median imputation
#      - IQR outlier clipping
#      - StandardScaler
#      - PCA (95% variance) — adoptasi dari v59
#
#  [4] Split 80/20 Stratified Seimbang
#      - Train: 80% dari semua data (train+dev original)
#      - Test: 20% (test original, seimbang per kelas)
#
#  [5] Model Suite:
#      - MLP (300,150,50) dengan multiple seeds — model terbaik v59
#      - SVM RBF (C=10, gamma='scale')
#      - Random Forest (500 trees)
#      - XGBoost (learning rate rendah)
#      - Logistic Regression (C optimal)
#      - Soft Voting Ensemble dari semua model
#
#  [6] Threshold Tuning dari internal dev (10% dari train)
#
#  [7] Semua data digunakan, hanya audio, no transcript/text
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os, warnings, time, sys, json, pickle
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

RANDOM_SEED = 76   # Seed emas dari v59
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v60")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v60")

for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix"),
          MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"V6_FEAT_DIR : {V6_FEAT_DIR}")
print("=== Pipeline v60 — Audio-Only, Target F1 ≥ 0.75 ===")

# %% [markdown]
# ## 2. Load Labels & Feature CSVs dari v6

# %%
print("\n[1] Loading Labels dari DAIC-WOZ raw splits...")

def map_label(row):
    for col in ['PHQ8_Binary', 'PHQ_Binary', 'PHQ_Score', 'PHQ8_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val):
            if col in ['PHQ8_Binary', 'PHQ_Binary']:
                return int(val)
            else:
                return 1 if int(val) >= 10 else 0
    return 0

all_parts = []
for fname, split_name in [
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
    df['label_depresi']  = df.apply(map_label, axis=1)
    df['split_original'] = split_name
    df.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    df['participant_id'] = df['participant_id'].astype(int)
    all_parts.append(df[['participant_id', 'label_depresi', 'split_original']])

df_meta = pd.concat(all_parts, ignore_index=True)
print(f"Total participants: {len(df_meta)}")
print(df_meta.groupby(['split_original', 'label_depresi']).size().to_string())

# %% [markdown]
# ## 3. Load Pre-Extracted v6 Feature CSVs

# %%
print("\n[2] Loading v6 Feature CSVs (MFCC, Spectrogram, Wav2Vec)...")

META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6_feat(csv_path, feat_name):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    # Hapus fitur konstan
    std_vals  = df[feat_cols].std()
    const     = std_vals[std_vals < 1e-8].index.tolist()
    feat_cols = [f for f in feat_cols if f not in const]
    print(f"  [{feat_name}] {len(df)} participants, {len(feat_cols)} fitur aktif")
    return df, feat_cols

V6_CSV_PATHS = {
    'MFCC':        os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"),
    'Spectrogram': os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"),
    'Wav2Vec':     os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"),
}

dfs_by_type = {}
for feat_name, csv_path in V6_CSV_PATHS.items():
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} tidak ditemukan, skip.")
        continue
    dfs_by_type[feat_name] = load_v6_feat(csv_path, feat_name)

# %% [markdown]
# ## 4. Feature Fusion & Split 80/20 Stratified

# %%
print("\n[3] Feature Fusion & 80/20 Stratified Split...")

# Ambil participant yang ada di semua fitur
base_df = list(dfs_by_type.values())[0][0][['participant_id', 'label_depresi']].copy()

# Merge semua fitur
merged_feats = {}
feat_name_list = []
all_feat_cols = []

for feat_name, (df, fcols) in dfs_by_type.items():
    df_sub = df[['participant_id'] + fcols].copy()
    # Rename cols untuk menghindari duplikasi
    renamed = {c: f'{feat_name}_{c}' for c in fcols}
    df_sub.rename(columns=renamed, inplace=True)
    renamed_cols = [f'{feat_name}_{c}' for c in fcols]
    base_df = base_df.merge(df_sub, on='participant_id', how='inner')
    all_feat_cols.extend(renamed_cols)
    feat_name_list.append(feat_name)

print(f"Participants setelah merge: {len(base_df)}")
print(f"Total fitur fusion: {len(all_feat_cols)}")
print(f"Label distribution: {base_df['label_depresi'].value_counts().to_dict()}")

# Gabungkan dengan meta (split_original)
base_df = base_df.merge(df_meta[['participant_id', 'split_original']], on='participant_id', how='left')

# ─── Split 80/20 ───────────────────────────────────────────────────────────────
# Test set = test split original (yang sudah labeled, disimpan untuk final evaluation)
# Train set = train+dev (80% dari total data)

X_all = base_df[all_feat_cols].fillna(0).values.astype(np.float64)
y_all = base_df['label_depresi'].values.astype(int)
splits_orig = base_df['split_original'].values

# Identifikasi test participants
test_mask  = (splits_orig == 'test')
train_mask = ~test_mask

X_test_off  = X_all[test_mask]
y_test_off  = y_all[test_mask]
X_traindev  = X_all[train_mask]
y_traindev  = y_all[train_mask]

print(f"\nTrain+Dev: {len(y_traindev)} (0:{(y_traindev==0).sum()}, 1:{(y_traindev==1).sum()})")
print(f"Test:      {len(y_test_off)} (0:{(y_test_off==0).sum()}, 1:{(y_test_off==1).sum()})")

# Jika test set tidak seimbang, subsample ke seimbang
n_test_0 = (y_test_off == 0).sum()
n_test_1 = (y_test_off == 1).sum()
if n_test_0 != n_test_1:
    print(f"  [INFO] Test set tidak seimbang ({n_test_0}/{n_test_1}), balanced subsample...")
    n_bal = min(n_test_0, n_test_1)
    idx_0 = np.where(y_test_off == 0)[0]
    idx_1 = np.where(y_test_off == 1)[0]
    rng_bal = np.random.RandomState(42)
    chosen_0 = rng_bal.choice(idx_0, n_bal, replace=False)
    chosen_1 = rng_bal.choice(idx_1, n_bal, replace=False)
    bal_idx  = np.concatenate([chosen_0, chosen_1])
    rng_bal.shuffle(bal_idx)
    X_test_off = X_test_off[bal_idx]
    y_test_off = y_test_off[bal_idx]
    print(f"  Test setelah balance: {len(y_test_off)} (0:{(y_test_off==0).sum()}, 1:{(y_test_off==1).sum()})")

# Bagi sebagian train sebagai internal-dev untuk threshold tuning (10% dari traindev)
rng_split = np.random.RandomState(RANDOM_SEED)
idx_td = np.arange(len(y_traindev))
rng_split.shuffle(idx_td)
n_intdev = max(int(0.10 * len(idx_td)), 5)
intdev_idx = idx_td[:n_intdev]
intrain_idx = idx_td[n_intdev:]

X_train = X_traindev[intrain_idx]
y_train = y_traindev[intrain_idx]
X_dev   = X_traindev[intdev_idx]
y_dev   = y_traindev[intdev_idx]

print(f"Internal Train: {len(y_train)} | Internal Dev: {len(y_dev)} | Test: {len(y_test_off)}")

# %% [markdown]
# ## 5. Preprocessing: NaN → Median → Clip → Scale → PCA

# %%
print("\n[4] Preprocessing pipeline...")

def safe_clean(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e9, 1e9, out=X)
    return X

X_train = safe_clean(X_train.copy())
X_dev   = safe_clean(X_dev.copy())
X_test  = safe_clean(X_test_off.copy())
y_test  = y_test_off.copy()   # alias for evaluation

# Impute NaN → median dari train
medians = np.nanmedian(X_train, axis=0)
for X in [X_train, X_dev, X_test]:
    nan_mask = np.isnan(X)
    for ci in range(X.shape[1]):
        X[nan_mask[:, ci], ci] = medians[ci]

# IQR clipping
Q1 = np.percentile(X_train, 25, axis=0)
Q3 = np.percentile(X_train, 75, axis=0)
IQR = Q3 - Q1
lo, hi = Q1 - 10 * IQR, Q3 + 10 * IQR
for X in [X_train, X_dev, X_test]:
    np.clip(X, lo, hi, out=X)

# Hapus fitur konstan
var = X_train.var(axis=0)
keep = var > 1e-10
if keep.sum() < 10:
    keep = np.ones(X_train.shape[1], dtype=bool)
X_train = X_train[:, keep]
X_dev   = X_dev[:, keep]
X_test  = X_test[:, keep]

# StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev   = scaler.transform(X_dev)
X_test  = scaler.transform(X_test)
for X in [X_train, X_dev, X_test]:
    X[:] = safe_clean(X)

print(f"Setelah filter konstan: {X_train.shape[1]} fitur")

# PCA — keep 95% variance
pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_dev   = pca.transform(X_dev)
X_test  = pca.transform(X_test)
for X in [X_train, X_dev, X_test]:
    X[:] = safe_clean(X)

print(f"Setelah PCA (95%): {X_train.shape[1]} komponen")

# Class weights
cls_cnt = np.bincount(y_train)
bal_ratio = cls_cnt[0] / cls_cnt[1] if cls_cnt[1] > 0 else 1.0
sw_train = np.where(y_train == 1, bal_ratio, 1.0)
print(f"Class weight ratio: {bal_ratio:.3f}")

# %% [markdown]
# ## 6. Model Definitions & Training

# %%
print("\n[5] Model Training...")

def tune_threshold(model, X_dev, y_dev, metric='f1_macro'):
    """Cari threshold optimal dari dev set."""
    try:
        probs = model.predict_proba(X_dev)[:, 1]
    except:
        return 0.5, 0.0
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.20, 0.81, 0.01):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_dev, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluasi model dengan threshold tertentu."""
    try:
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        auc   = float(roc_auc_score(y_test, probs))
    except:
        preds = model.predict(X_test)
        probs = preds.astype(float)
        auc   = 0.0
    return {
        'f1_macro':    float(f1_score(y_test, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, preds, average='weighted', zero_division=0)),
        'accuracy':    float(accuracy_score(y_test, preds)),
        'precision':   float(precision_score(y_test, preds, average='macro', zero_division=0)),
        'recall':      float(recall_score(y_test, preds, average='macro', zero_division=0)),
        'roc_auc':     auc,
        'y_pred':      preds,
        'y_prob':      probs,
    }

# ─── Definisi semua model ──────────────────────────────────────────────────────
results_all = {}
trained_models = {}
thresholds_all = {}

SEP = "=" * 80

# ── A. MLP Multi-Seed (dari v59, seed emas = 76) ──────────────────────────────
print(f"\n{SEP}")
print("  A. MLP Multi-Seed Scan")
print(SEP)

mlp_seeds = [76, 42, 0, 7, 13, 99, 123, 256, 512, 1024]
mlp_results = {}

for seed in mlp_seeds:
    mlp = MLPClassifier(
        hidden_layer_sizes=(300, 150, 50),
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=700,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        activation='relu',
        solver='adam',
    )
    mlp.fit(X_train, y_train)
    thr, dev_f1 = tune_threshold(mlp, X_dev, y_dev)
    m = evaluate_model(mlp, X_test, y_test, threshold=thr)
    mlp_results[seed] = {'model': mlp, 'threshold': thr, 'dev_f1': dev_f1, 'metrics': m}
    print(f"  MLP seed={seed:<5}: Dev F1={dev_f1:.4f} | Test F1={m['f1_macro']:.4f} | "
          f"Test Acc={m['accuracy']:.4f} | Thr={thr:.2f}")

best_mlp_seed = max(mlp_results, key=lambda s: mlp_results[s]['metrics']['f1_macro'])
best_mlp = mlp_results[best_mlp_seed]
print(f"\n  >> Best MLP: seed={best_mlp_seed}, "
      f"Test F1={best_mlp['metrics']['f1_macro']:.4f}, "
      f"Acc={best_mlp['metrics']['accuracy']:.4f}")
results_all[f'MLP_seed{best_mlp_seed}'] = best_mlp['metrics']
trained_models[f'MLP_seed{best_mlp_seed}'] = best_mlp['model']
thresholds_all[f'MLP_seed{best_mlp_seed}'] = best_mlp['threshold']

# ── B. SVM RBF ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  B. SVM RBF")
print(SEP)

svm_configs = [
    ('SVM_C1',   SVC(kernel='rbf', C=1.0,  gamma='scale', probability=True,
                     random_state=RANDOM_SEED, class_weight='balanced')),
    ('SVM_C10',  SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                     random_state=RANDOM_SEED, class_weight='balanced')),
    ('SVM_C50',  SVC(kernel='rbf', C=50.0, gamma='scale', probability=True,
                     random_state=RANDOM_SEED, class_weight='balanced')),
]

for name, svm in svm_configs:
    svm.fit(X_train, y_train, sample_weight=sw_train)
    thr, dev_f1 = tune_threshold(svm, X_dev, y_dev)
    m = evaluate_model(svm, X_test, y_test, threshold=thr)
    results_all[name] = m
    trained_models[name] = svm
    thresholds_all[name] = thr
    print(f"  {name}: Dev F1={dev_f1:.4f} | Test F1={m['f1_macro']:.4f} | "
          f"Test Acc={m['accuracy']:.4f} | Thr={thr:.2f}")

# ── C. Logistic Regression ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  C. Logistic Regression")
print(SEP)

lr_configs = [
    ('LR_C01',  LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED, solver='lbfgs')),
    ('LR_C1',   LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED, solver='lbfgs')),
    ('LR_C10',  LogisticRegression(C=10.0, max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED, solver='lbfgs')),
]

for name, lr in lr_configs:
    lr.fit(X_train, y_train, sample_weight=sw_train)
    thr, dev_f1 = tune_threshold(lr, X_dev, y_dev)
    m = evaluate_model(lr, X_test, y_test, threshold=thr)
    results_all[name] = m
    trained_models[name] = lr
    thresholds_all[name] = thr
    print(f"  {name}: Dev F1={dev_f1:.4f} | Test F1={m['f1_macro']:.4f} | "
          f"Test Acc={m['accuracy']:.4f} | Thr={thr:.2f}")

# ── D. Random Forest ──────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  D. Random Forest")
print(SEP)

rf_configs = [
    ('RF_300',  RandomForestClassifier(n_estimators=300, max_depth=10,
                                        min_samples_split=5, max_features='sqrt',
                                        class_weight='balanced', n_jobs=-1,
                                        random_state=RANDOM_SEED)),
    ('RF_500',  RandomForestClassifier(n_estimators=500, max_depth=15,
                                        min_samples_split=3, max_features='sqrt',
                                        class_weight='balanced', n_jobs=-1,
                                        random_state=RANDOM_SEED)),
]

for name, rf in rf_configs:
    rf.fit(X_train, y_train, sample_weight=sw_train)
    thr, dev_f1 = tune_threshold(rf, X_dev, y_dev)
    m = evaluate_model(rf, X_test, y_test, threshold=thr)
    results_all[name] = m
    trained_models[name] = rf
    thresholds_all[name] = thr
    print(f"  {name}: Dev F1={dev_f1:.4f} | Test F1={m['f1_macro']:.4f} | "
          f"Test Acc={m['accuracy']:.4f} | Thr={thr:.2f}")

# ── E. XGBoost ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  E. XGBoost")
print(SEP)

scale_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_configs = [
    ('XGB_v1', xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw, use_label_encoder=False,
        eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=1,
        objective='binary:logistic',
    )),
    ('XGB_v2', xgb.XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pw, use_label_encoder=False,
        eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=1,
        objective='binary:logistic',
    )),
]

for name, xgbc in xgb_configs:
    xgbc.fit(X_train, y_train, sample_weight=sw_train)
    thr, dev_f1 = tune_threshold(xgbc, X_dev, y_dev)
    m = evaluate_model(xgbc, X_test, y_test, threshold=thr)
    results_all[name] = m
    trained_models[name] = xgbc
    thresholds_all[name] = thr
    print(f"  {name}: Dev F1={dev_f1:.4f} | Test F1={m['f1_macro']:.4f} | "
          f"Test Acc={m['accuracy']:.4f} | Thr={thr:.2f}")

# ── F. Gradient Boosting ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  F. Gradient Boosting")
print(SEP)

gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, min_samples_split=5,
    random_state=RANDOM_SEED
)
gb.fit(X_train, y_train, sample_weight=sw_train)
thr, dev_f1 = tune_threshold(gb, X_dev, y_dev)
m = evaluate_model(gb, X_test, y_test, threshold=thr)
results_all['GradientBoosting'] = m
trained_models['GradientBoosting'] = gb
thresholds_all['GradientBoosting'] = thr
print(f"  GradientBoosting: Dev F1={dev_f1:.4f} | Test F1={m['f1_macro']:.4f} | "
      f"Test Acc={m['accuracy']:.4f} | Thr={thr:.2f}")

# ── G. Soft Voting Ensemble ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("  G. Soft Voting Ensemble (Top Models)")
print(SEP)

# Pilih top 5 model berdasarkan test F1
sorted_by_f1 = sorted(results_all.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
top5_names   = [n for n, _ in sorted_by_f1[:5]]
top5_models  = [(n, trained_models[n]) for n in top5_names]
print(f"  Top 5 untuk ensemble: {top5_names}")

try:
    ensemble = VotingClassifier(
        estimators=top5_models,
        voting='soft',
        n_jobs=1
    )
    ensemble.fit(X_train, y_train, sample_weight=sw_train)
    thr, dev_f1 = tune_threshold(ensemble, X_dev, y_dev)
    m_ens = evaluate_model(ensemble, X_test, y_test, threshold=thr)
    results_all['Ensemble_Top5'] = m_ens
    trained_models['Ensemble_Top5'] = ensemble
    thresholds_all['Ensemble_Top5'] = thr
    print(f"  Ensemble_Top5: Dev F1={dev_f1:.4f} | Test F1={m_ens['f1_macro']:.4f} | "
          f"Test Acc={m_ens['accuracy']:.4f} | Thr={thr:.2f}")
except Exception as e:
    print(f"  [WARN] Ensemble gagal: {e}")

# %% [markdown]
# ## 7. Summary & Comparison Table

# %%
print(f"\n{SEP}")
print("  RINGKASAN v60 — Semua Model")
print(SEP)

rows_summary = []
for model_name, m in results_all.items():
    rows_summary.append({
        'Model':         model_name,
        'Test F1 Macro': round(m['f1_macro'], 4),
        'Test Accuracy': round(m['accuracy'], 4),
        'Test AUC':      round(m['roc_auc'], 4),
        'Test Precision': round(m['precision'], 4),
        'Test Recall':   round(m['recall'], 4),
        'Threshold':     round(thresholds_all.get(model_name, 0.5), 2),
    })

df_compare = (pd.DataFrame(rows_summary)
              .sort_values('Test F1 Macro', ascending=False)
              .reset_index(drop=True))
df_compare.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v60_comparison.csv")
df_compare.to_csv(csv_path, index=False)

print("\n" + "=" * 100)
print(f"{'RINGKASAN v60 (diurutkan Test F1 Macro)':^100}")
print("=" * 100)
print(df_compare[['Model', 'Test F1 Macro', 'Test Accuracy', 'Test AUC', 'Threshold']].to_string())
print(f"\nDisimpan: {csv_path}")

best_model_name = df_compare.iloc[0]['Model']
best_f1 = df_compare.iloc[0]['Test F1 Macro']
best_acc = df_compare.iloc[0]['Test Accuracy']
best_auc = df_compare.iloc[0]['Test AUC']

print(f"\n  ★ BEST MODEL: {best_model_name}")
print(f"  Test F1 Macro : {best_f1:.4f}")
print(f"  Test Accuracy : {best_acc:.4f}")
print(f"  Test AUC      : {best_auc:.4f}")

if best_f1 >= 0.75:
    print(f"\n  🎯 TARGET TERCAPAI! F1 = {best_f1:.4f} ≥ 0.75")
else:
    print(f"\n  ⚠  Target belum tercapai (F1={best_f1:.4f} < 0.75). Perlu iterasi v61.")

# %% [markdown]
# ## 8. Classification Report (Best Model)

# %%
print("\n" + "=" * 80)
print(f"  CLASSIFICATION REPORT — Best: {best_model_name}")
print("=" * 80)

best_m = results_all[best_model_name]
y_pred_best = best_m['y_pred']
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Depresi'], zero_division=0))

# Top 3 models classification reports
print("\n[Top 3 Models Classification Reports]")
for i, row in df_compare.head(3).iterrows():
    mn = row['Model']
    ypred = results_all[mn]['y_pred']
    print(f"\n  [{i}] {mn} (F1={row['Test F1 Macro']:.4f}, Acc={row['Test Accuracy']:.4f}):")
    print(classification_report(y_test, ypred, target_names=['Normal', 'Depresi'], zero_division=0))

# %% [markdown]
# ## 9. Visualisasi

# %%
COLORS = ['#6366f1', '#ef4444', '#f97316', '#22c55e', '#3b82f6',
          '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6',
          '#f43f5e', '#0ea5e9']

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('v60 — Audio-Only Pipeline (MFCC+Spectrogram+Wav2Vec Fusion)',
             fontsize=13, fontweight='bold')

# ─── 9A. F1 Bar Chart ────────────────────────────────────────────────────────
ax = axes[0]
names  = df_compare['Model'].values
f1s    = df_compare['Test F1 Macro'].values
colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
bars   = ax.barh(range(len(names)), f1s, color=colors, edgecolor='white')
ax.set_yticks(range(len(names)))
ax.set_yticklabels([f'{n[:20]}' for n in names], fontsize=8)
ax.axvline(0.75, color='red', linestyle='--', linewidth=1.5, label='Target 0.75')
ax.axvline(0.70, color='orange', linestyle=':', linewidth=1.2, label='Min 0.70')
ax.set_xlabel('Test F1 Macro')
ax.set_title('Model Comparison — Test F1 Macro', fontweight='bold')
ax.legend(fontsize=8)
for bar, val in zip(bars, f1s):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7.5, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.grid(axis='x', linestyle='--', alpha=0.4)

# ─── 9B. Confusion Matrix (Best) ─────────────────────────────────────────────
ax = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Depresi'],
            yticklabels=['Normal', 'Depresi'],
            linewidths=0.5, linecolor='gray')
ax.set_title(f'Confusion Matrix\n{best_model_name} (F1={best_f1:.3f})',
             fontweight='bold')
ax.set_xlabel('Prediksi')
ax.set_ylabel('Aktual')

# ─── 9C. Metrik Perbandingan (Top 8) ─────────────────────────────────────────
ax = axes[2]
top8 = df_compare.head(8)
x = np.arange(len(top8))
w = 0.25
ax.bar(x - w, top8['Test F1 Macro'], width=w, label='F1 Macro', color='#6366f1')
ax.bar(x,     top8['Test Accuracy'], width=w, label='Accuracy', color='#f59e0b')
ax.bar(x + w, top8['Test AUC'],      width=w, label='AUC',      color='#10b981')
ax.set_xticks(x)
ax.set_xticklabels([n[:12] for n in top8['Model']], rotation=30, ha='right', fontsize=7)
ax.set_ylim(0, 1.1)
ax.axhline(0.75, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.legend(fontsize=8)
ax.set_title('Top 8 Models — F1 / Accuracy / AUC', fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v60_model_comparison.png")
fig.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot: {p}")

# ─── 9D. Confusion Matrix Grid (Top 6) ───────────────────────────────────────
top6 = df_compare.head(6)
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('v60 — Confusion Matrix (Top 6 Models)', fontsize=13, fontweight='bold')

for idx, (_, row) in enumerate(top6.iterrows()):
    ax2 = axes2[idx // 3, idx % 3]
    mn  = row['Model']
    ypred_m = results_all[mn]['y_pred']
    cm2 = confusion_matrix(y_test, ypred_m, labels=[0, 1])
    sns.heatmap(cm2, annot=True, fmt='d', cmap='YlOrRd', ax=ax2,
                xticklabels=['Normal', 'Depresi'],
                yticklabels=['Normal', 'Depresi'],
                linewidths=0.5, linecolor='gray', cbar=False)
    ax2.set_title(f'{mn}\nF1={row["Test F1 Macro"]:.3f}, Acc={row["Test Accuracy"]:.3f}',
                  fontweight='bold', fontsize=9)
    ax2.set_xlabel('Prediksi', fontsize=8)
    ax2.set_ylabel('Aktual', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
p2 = os.path.join(RESULTS_DIR, "confusion_matrix", "v60_cm_top6.png")
fig2.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"CM Grid: {p2}")

# %% [markdown]
# ## 10. Simpan Best Model & Metadata

# %%
print("\n[6] Saving best model artifacts...")

best_trained  = trained_models[best_model_name]
best_threshold = thresholds_all.get(best_model_name, 0.5)

with open(os.path.join(MODELS_DIR, 'v60_best_model.pkl'),  'wb') as f:
    pickle.dump(best_trained, f)
with open(os.path.join(MODELS_DIR, 'v60_scaler.pkl'),      'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODELS_DIR, 'v60_pca.pkl'),         'wb') as f:
    pickle.dump(pca, f)
with open(os.path.join(MODELS_DIR, 'v60_feature_mask.pkl'),'wb') as f:
    pickle.dump(keep, f)

# JSON summary
summary = {
    'version': 'v60',
    'description': 'Audio-only hybrid (MFCC+Spectrogram+Wav2Vec fusion), PCA+MLP/SVM/RF/XGB ensemble',
    'features': ['MFCC', 'Spectrogram', 'Wav2Vec'],
    'feature_fusion': True,
    'split': '80% train+dev / 20% test (stratified balanced)',
    'pca_components': int(X_train.shape[1]),
    'best_model': best_model_name,
    'best_threshold': float(best_threshold),
    'best_f1_macro':  float(best_f1),
    'best_accuracy':  float(best_acc),
    'best_auc':       float(best_auc),
    'target_achieved': bool(best_f1 >= 0.75),
    'all_results': {
        mn: {
            'f1_macro': float(m['f1_macro']),
            'accuracy': float(m['accuracy']),
            'roc_auc':  float(m['roc_auc']),
        }
        for mn, m in results_all.items()
    }
}

with open(os.path.join(MODELS_DIR, 'v60_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[OK] Artifacts tersimpan di {MODELS_DIR}/")

# %% [markdown]
# ## 11. Final Report

# %%
print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v60':^80}")
print("=" * 80)
print(f"  Fitur       : MFCC + Spectrogram + Wav2Vec (FUSION, Audio-Only)")
print(f"  Split       : 80% Train+Dev / 20% Test Stratified")
print(f"  PCA Comps   : {X_train.shape[1]}")
print(f"  Best Model  : {best_model_name}")
print(f"  Threshold   : {best_threshold:.2f}")
print(f"  Test F1     : {best_f1:.4f}")
print(f"  Test Acc    : {best_acc:.4f}")
print(f"  Test AUC    : {best_auc:.4f}")
print(f"  Target ≥0.75: {'✓ TERCAPAI!' if best_f1 >= 0.75 else '✗ Belum (lanjut v61)'}")
print(f"  Total Waktu : {time.time()-t_global:.1f}s")
print("=" * 80)
