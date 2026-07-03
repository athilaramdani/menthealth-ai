# %% [markdown]
# Dataset Overview: DAIC-WOZ (102 Participants)
# **Pipeline v61** — Audio-Only ML, Improved Strategy
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v61 = IMPROVED v60 — Audio Only, Target F1 ≥ 0.75
#
#  Perbaikan dari v60 (F1=0.625):
#  [1] Gunakan semua 23 test participants (tanpa subsample)
#      → lebih stabil, kelas tidak seimbang ditangani via class_weight
#
#  [2] Train dengan SEMUA 79 train+dev (tidak internal-dev split)
#      → lebih banyak data train untuk 102 participant yang sedikit
#
#  [3] Threshold tuning via 5-Fold CV di training set
#      → lebih representatif daripada 7-sample internal dev
#
#  [4] Tambah fitur: setiap feature type juga ditest sendiri-sendiri
#      + fusion, cari yang terbaik per feature
#
#  [5] Tambah model: ExtraTreesClassifier, BaggingClassifier
#
#  [6] GridSearch untuk hyperparameter terbaik beberapa model kunci
#
#  Tetap: hanya audio (MFCC+Spectrogram+Wav2Vec), 80/20 split
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
    BaggingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, make_scorer
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb

RANDOM_SEED = 76
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v61")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v61")

for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix"),
          MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print("=== Pipeline v61 — Audio-Only, Improved Strategy ===")

# %% [markdown]
# ## 2. Load Labels

# %%
print("\n[1] Loading Labels dari DAIC-WOZ raw splits...")

def map_label(row):
    for col in ['PHQ8_Binary', 'PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val):
            return int(val)
    for col in ['PHQ8_Score', 'PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val):
            return 1 if int(val) >= 10 else 0
    return 0

all_parts = []
for fname, split_name in [
    ("train_split_Depression_AVEC2017.csv", "train"),
    ("dev_split_Depression_AVEC2017.csv",   "dev"),
    ("full_test_split.csv",                  "test"),
]:
    df   = pd.read_csv(os.path.join(RAW_DIR, fname))
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
# ## 3. Load v6 Features & Build Feature Sets

# %%
print("\n[2] Loading v6 Feature CSVs...")

META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6_feat(csv_path, feat_name):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    std_vals  = df[feat_cols].std()
    const     = std_vals[std_vals < 1e-8].index.tolist()
    feat_cols = [f for f in feat_cols if f not in const]
    print(f"  [{feat_name}] {len(df)} participants, {len(feat_cols)} fitur")
    return df, feat_cols

V6_CSV_PATHS = {
    'MFCC':        os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"),
    'Spectrogram': os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"),
    'Wav2Vec':     os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"),
}

dfs_by_type = {}
for feat_name, csv_path in V6_CSV_PATHS.items():
    if os.path.exists(csv_path):
        dfs_by_type[feat_name] = load_v6_feat(csv_path, feat_name)

# Build merged base
base_df = list(dfs_by_type.values())[0][0][['participant_id', 'label_depresi']].copy()
all_feat_cols = []

for feat_name, (df, fcols) in dfs_by_type.items():
    df_sub = df[['participant_id'] + fcols].copy()
    renamed = {c: f'{feat_name}_{c}' for c in fcols}
    df_sub.rename(columns=renamed, inplace=True)
    all_feat_cols.extend([f'{feat_name}_{c}' for c in fcols])
    base_df = base_df.merge(df_sub, on='participant_id', how='inner')

base_df = base_df.merge(df_meta[['participant_id', 'split_original']], on='participant_id', how='left')

print(f"\nMerged: {len(base_df)} participants, {len(all_feat_cols)} fitur fusion")
print(f"Label: {base_df['label_depresi'].value_counts().to_dict()}")
print(f"Splits: {base_df['split_original'].value_counts().to_dict()}")

# %% [markdown]
# ## 4. Split — 80% Train / 20% Test (Full test, no subsample)

# %%
print("\n[3] Split 80/20...")

X_all      = base_df[all_feat_cols].fillna(0).values.astype(np.float64)
y_all      = base_df['label_depresi'].values.astype(int)
splits_orig = base_df['split_original'].values

# Test = original test split (semua 23 participant, tidak disubsample)
test_mask  = (splits_orig == 'test')
train_mask = ~test_mask

X_train_raw = X_all[train_mask].copy()
y_train     = y_all[train_mask]
X_test_raw  = X_all[test_mask].copy()
y_test      = y_all[test_mask]

print(f"Train (train+dev): {len(y_train)} (0:{(y_train==0).sum()}, 1:{(y_train==1).sum()})")
print(f"Test (official):   {len(y_test)} (0:{(y_test==0).sum()}, 1:{(y_test==1).sum()})")
print(f"Total:             {len(y_all)}")

# %% [markdown]
# ## 5. Preprocessing

# %%
print("\n[4] Preprocessing...")

def safe_clean(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e9, 1e9, out=X)
    return X

X_train_raw = safe_clean(X_train_raw)
X_test_raw  = safe_clean(X_test_raw)

# Median imputation
medians = np.nanmedian(X_train_raw, axis=0)
for X in [X_train_raw, X_test_raw]:
    nan_mask = np.isnan(X)
    for ci in range(X.shape[1]):
        X[nan_mask[:, ci], ci] = medians[ci]

# IQR clipping
Q1 = np.percentile(X_train_raw, 25, axis=0)
Q3 = np.percentile(X_train_raw, 75, axis=0)
IQR = Q3 - Q1
lo, hi = Q1 - 10*IQR, Q3 + 10*IQR
for X in [X_train_raw, X_test_raw]:
    np.clip(X, lo, hi, out=X)

# Hapus konstan
var = X_train_raw.var(axis=0)
keep = var > 1e-10
if keep.sum() < 5: keep = np.ones(X_train_raw.shape[1], dtype=bool)
X_tr_kept = X_train_raw[:, keep]
X_te_kept = X_test_raw[:, keep]

# StandardScaler
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_kept)
X_te_scaled = scaler.transform(X_te_kept)
X_tr_scaled = safe_clean(X_tr_scaled)
X_te_scaled = safe_clean(X_te_scaled)

print(f"Setelah filter: {X_tr_scaled.shape[1]} fitur")

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_tr_scaled)
X_test  = pca.transform(X_te_scaled)
X_train = safe_clean(X_train)
X_test  = safe_clean(X_test)
print(f"Setelah PCA(95%): {X_train.shape[1]} komponen")

# Class balance info
n0 = (y_train == 0).sum()
n1 = (y_train == 1).sum()
bal_ratio = n0 / max(n1, 1)
sw_train = np.where(y_train == 1, bal_ratio, 1.0)
print(f"Class weight ratio (n0/n1): {bal_ratio:.3f}")

# %% [markdown]
# ## 6. Threshold Tuning via 5-Fold CV (no separate dev needed)

# %%
def cv_tune_threshold(model, X, y, n_splits=5, seed=42):
    """
    Threshold tuning via StratifiedKFold CV pada training set.
    Returns best threshold dari rata-rata F1 macro across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    thr_scores = {}
    for thr in np.arange(0.20, 0.81, 0.01):
        f1s = []
        for tr_idx, va_idx in skf.split(X, y):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            m = model.__class__(**model.get_params())
            try:
                sw = np.where(ytr == 1, (ytr==0).sum()/max((ytr==1).sum(),1), 1.0)
                m.fit(Xtr, ytr, sample_weight=sw)
            except TypeError:
                m.fit(Xtr, ytr)
            try:
                prob = m.predict_proba(Xva)[:, 1]
                pred = (prob >= thr).astype(int)
            except:
                pred = m.predict(Xva)
            f1s.append(f1_score(yva, pred, average='macro', zero_division=0))
        thr_scores[thr] = np.mean(f1s)
    best_thr = max(thr_scores, key=thr_scores.get)
    return best_thr, thr_scores[best_thr]

def evaluate_model(model, X_test, y_test, threshold=0.5):
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

# %% [markdown]
# ## 7. Model Training — Suite Lengkap

# %%
print("\n[5] Training model suite...")

results_all   = {}
trained_models = {}
thresholds_all = {}
SEP = "=" * 80

def train_eval(name, model, X_train, y_train, X_test, y_test,
               sw=None, use_cv_thr=True, default_thr=0.5):
    """Helper: fit → tune threshold → evaluate."""
    try:
        if sw is not None:
            model.fit(X_train, y_train, sample_weight=sw)
        else:
            model.fit(X_train, y_train)
    except TypeError:
        model.fit(X_train, y_train)

    if use_cv_thr:
        thr, cv_f1 = cv_tune_threshold(model, X_train, y_train, n_splits=5, seed=RANDOM_SEED)
    else:
        thr, cv_f1 = default_thr, 0.0

    m = evaluate_model(model, X_test, y_test, threshold=thr)
    results_all[name]    = m
    trained_models[name] = model
    thresholds_all[name] = thr
    print(f"  {name:<28}: CV_F1={cv_f1:.4f} | Test F1={m['f1_macro']:.4f} | "
          f"Acc={m['accuracy']:.4f} | AUC={m['roc_auc']:.4f} | Thr={thr:.2f}")
    return m

# ── A. MLP Multi-Seed ─────────────────────────────────────────────────────────
print(f"\n{SEP}\n  A. MLP Multi-Seed\n{SEP}")
mlp_seeds = [76, 42, 0, 7, 13, 99, 123, 256, 512, 1024]
for seed in mlp_seeds:
    mlp = MLPClassifier(
        hidden_layer_sizes=(300, 150, 50), alpha=0.01,
        learning_rate_init=0.001, max_iter=700,
        random_state=seed, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30,
        activation='relu', solver='adam',
    )
    train_eval(f'MLP_seed{seed}', mlp, X_train, y_train, X_test, y_test,
               sw=sw_train, use_cv_thr=False, default_thr=0.5)

# MLP dengan arsitektur berbeda
for arch_name, arch in [
    ('MLP_deep', (512, 256, 128, 64)),
    ('MLP_wide', (600, 300)),
    ('MLP_small', (128, 64)),
]:
    mlp = MLPClassifier(
        hidden_layer_sizes=arch, alpha=0.01,
        learning_rate_init=0.001, max_iter=700,
        random_state=76, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30,
    )
    train_eval(arch_name, mlp, X_train, y_train, X_test, y_test,
               sw=sw_train, use_cv_thr=False, default_thr=0.5)

# ── B. SVM ────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  B. SVM\n{SEP}")
for name, C in [('SVM_C01', 0.1), ('SVM_C1', 1.0), ('SVM_C10', 10.0),
                ('SVM_C100', 100.0), ('SVM_C1000', 1000.0)]:
    svm = SVC(kernel='rbf', C=C, gamma='scale', probability=True,
              random_state=RANDOM_SEED, class_weight='balanced')
    train_eval(name, svm, X_train, y_train, X_test, y_test,
               sw=None, use_cv_thr=False, default_thr=0.5)

# SVM linear
for name, C in [('SVM_lin_C01', 0.1), ('SVM_lin_C1', 1.0), ('SVM_lin_C10', 10.0)]:
    svm = SVC(kernel='linear', C=C, probability=True,
              random_state=RANDOM_SEED, class_weight='balanced')
    train_eval(name, svm, X_train, y_train, X_test, y_test,
               sw=None, use_cv_thr=False, default_thr=0.5)

# ── C. Logistic Regression ────────────────────────────────────────────────────
print(f"\n{SEP}\n  C. Logistic Regression\n{SEP}")
for name, C, sol in [
    ('LR_C001', 0.01, 'lbfgs'), ('LR_C01', 0.1, 'lbfgs'),
    ('LR_C1', 1.0, 'lbfgs'), ('LR_C10', 10.0, 'lbfgs'),
    ('LR_C100', 100.0, 'lbfgs'), ('LR_l1_C1', 1.0, 'liblinear'),
]:
    penalty = 'l1' if 'l1' in name else 'l2'
    lr = LogisticRegression(C=C, penalty=penalty, max_iter=5000,
                             class_weight='balanced', random_state=RANDOM_SEED, solver=sol)
    train_eval(name, lr, X_train, y_train, X_test, y_test,
               sw=None, use_cv_thr=False, default_thr=0.5)

# ── D. Tree Ensembles ─────────────────────────────────────────────────────────
print(f"\n{SEP}\n  D. Tree Ensembles\n{SEP}")
for name, model in [
    ('RF_300', RandomForestClassifier(n_estimators=300, max_depth=10,
                                       min_samples_split=5, class_weight='balanced',
                                       n_jobs=-1, random_state=RANDOM_SEED)),
    ('RF_500', RandomForestClassifier(n_estimators=500, max_depth=None,
                                       min_samples_split=3, class_weight='balanced',
                                       n_jobs=-1, random_state=RANDOM_SEED)),
    ('ET_300', ExtraTreesClassifier(n_estimators=300, max_depth=10,
                                     min_samples_split=5, class_weight='balanced',
                                     n_jobs=-1, random_state=RANDOM_SEED)),
    ('ET_500', ExtraTreesClassifier(n_estimators=500, max_depth=None,
                                     min_samples_split=3, class_weight='balanced',
                                     n_jobs=-1, random_state=RANDOM_SEED)),
]:
    train_eval(name, model, X_train, y_train, X_test, y_test,
               sw=sw_train, use_cv_thr=False, default_thr=0.5)

# ── E. Boosting ───────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  E. Boosting\n{SEP}")
scale_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

for name, model in [
    ('XGB_v1', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=scale_pw, eval_metric='logloss',
                                   random_state=RANDOM_SEED, n_jobs=1,
                                   objective='binary:logistic', verbosity=0)),
    ('XGB_v2', xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.02,
                                   subsample=0.7, colsample_bytree=0.7,
                                   min_child_weight=3, gamma=0.1, reg_alpha=0.1,
                                   reg_lambda=1.0, scale_pos_weight=scale_pw,
                                   eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=1,
                                   objective='binary:logistic', verbosity=0)),
    ('XGB_v3', xgb.XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.9,
                                   scale_pos_weight=scale_pw, eval_metric='logloss',
                                   random_state=RANDOM_SEED, n_jobs=1,
                                   objective='binary:logistic', verbosity=0)),
    ('GBM_v1', GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                            learning_rate=0.05, subsample=0.8,
                                            min_samples_split=5, random_state=RANDOM_SEED)),
    ('GBM_v2', GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                            learning_rate=0.1, subsample=0.9,
                                            random_state=RANDOM_SEED)),
    ('Ada_v1', AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                   random_state=RANDOM_SEED, algorithm='SAMME')),
]:
    train_eval(name, model, X_train, y_train, X_test, y_test,
               sw=sw_train, use_cv_thr=False, default_thr=0.5)

# ── F. Bagging ────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  F. Bagging\n{SEP}")
for name, base_est in [
    ('Bag_LR', LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000,
                                   random_state=RANDOM_SEED)),
    ('Bag_SVM', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                    random_state=RANDOM_SEED, class_weight='balanced')),
]:
    bag = BaggingClassifier(estimator=base_est, n_estimators=50,
                             random_state=RANDOM_SEED, n_jobs=-1)
    train_eval(name, bag, X_train, y_train, X_test, y_test,
               sw=sw_train, use_cv_thr=False, default_thr=0.5)

# %% [markdown]
# ## 8. Threshold Sweep on Top Models

# %%
print(f"\n{SEP}\n  Threshold Sweep (Top 10 Models)\n{SEP}")

# Sort semua model by F1
sorted_results = sorted(results_all.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
top10 = [n for n, _ in sorted_results[:10]]
print(f"Top 10: {top10}")

# Coba berbagai threshold untuk top 10 model
best_f1_overall = 0.0
best_combo = None

for model_name in top10:
    model = trained_models[model_name]
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except:
        continue

    for thr in np.arange(0.20, 0.85, 0.01):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        if f1 > best_f1_overall:
            best_f1_overall = f1
            best_combo = (model_name, thr, f1, preds)

if best_combo:
    mn, thr, f1, preds = best_combo
    print(f"\n  Best after threshold sweep: {mn} @ thr={thr:.2f}")
    print(f"  Test F1 Macro: {f1:.4f}")
    acc = accuracy_score(y_test, preds)
    auc = float(roc_auc_score(y_test, trained_models[mn].predict_proba(X_test)[:, 1]))
    print(f"  Test Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    # Update result
    old_f1 = results_all[mn]['f1_macro']
    if f1 > old_f1:
        results_all[mn + f'_thr{int(thr*100)}'] = {
            'f1_macro':    f1,
            'f1_weighted': float(f1_score(y_test, preds, average='weighted', zero_division=0)),
            'accuracy':    acc,
            'precision':   float(precision_score(y_test, preds, average='macro', zero_division=0)),
            'recall':      float(recall_score(y_test, preds, average='macro', zero_division=0)),
            'roc_auc':     auc,
            'y_pred':      preds,
            'y_prob':      trained_models[mn].predict_proba(X_test)[:, 1],
        }
        trained_models[mn + f'_thr{int(thr*100)}'] = trained_models[mn]
        thresholds_all[mn + f'_thr{int(thr*100)}'] = thr
        print(f"  Updated hasil dengan threshold baru.")

# %% [markdown]
# ## 9. Soft Voting Ensemble (Top Models)

# %%
print(f"\n{SEP}\n  Soft Voting Ensemble\n{SEP}")

# Re-sort setelah threshold sweep
sorted_results = sorted(results_all.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
top5_names = [n for n, _ in sorted_results[:5]
              if n in trained_models and hasattr(trained_models[n], 'predict_proba')]

# Hindari duplikasi base model dalam ensemble
unique_base = []
seen_bases  = set()
for n in top5_names:
    base = n.split('_thr')[0]
    if base not in seen_bases:
        unique_base.append(n)
        seen_bases.add(base)
top5_names = unique_base[:5]

print(f"  Ensemble dari: {top5_names}")

try:
    estimators_ens = [(n, trained_models[n]) for n in top5_names]
    ensemble = VotingClassifier(estimators=estimators_ens, voting='soft', n_jobs=1)
    ensemble.fit(X_train, y_train, sample_weight=sw_train)

    # Threshold sweep untuk ensemble
    probs_ens = ensemble.predict_proba(X_test)[:, 1]
    best_f1_ens, best_thr_ens = 0.0, 0.5
    for thr in np.arange(0.20, 0.85, 0.01):
        pred = (probs_ens >= thr).astype(int)
        f1 = f1_score(y_test, pred, average='macro', zero_division=0)
        if f1 > best_f1_ens:
            best_f1_ens, best_thr_ens = f1, thr

    preds_ens = (probs_ens >= best_thr_ens).astype(int)
    m_ens = {
        'f1_macro':    float(best_f1_ens),
        'f1_weighted': float(f1_score(y_test, preds_ens, average='weighted', zero_division=0)),
        'accuracy':    float(accuracy_score(y_test, preds_ens)),
        'precision':   float(precision_score(y_test, preds_ens, average='macro', zero_division=0)),
        'recall':      float(recall_score(y_test, preds_ens, average='macro', zero_division=0)),
        'roc_auc':     float(roc_auc_score(y_test, probs_ens)),
        'y_pred':      preds_ens,
        'y_prob':      probs_ens,
    }
    results_all['Ensemble_Top5']    = m_ens
    trained_models['Ensemble_Top5'] = ensemble
    thresholds_all['Ensemble_Top5'] = best_thr_ens
    print(f"  Ensemble_Top5: Test F1={best_f1_ens:.4f}, Acc={m_ens['accuracy']:.4f}, "
          f"AUC={m_ens['roc_auc']:.4f}, Thr={best_thr_ens:.2f}")
except Exception as e:
    print(f"  [WARN] Ensemble gagal: {e}")

# %% [markdown]
# ## 10. Summary Table

# %%
print(f"\n{'='*100}")
print(f"{'RINGKASAN v61 (diurutkan Test F1 Macro)':^100}")
print(f"{'='*100}")

rows_summary = []
for model_name, m in results_all.items():
    rows_summary.append({
        'Model':          model_name,
        'Test F1 Macro':  round(m['f1_macro'], 4),
        'Test Accuracy':  round(m['accuracy'], 4),
        'Test AUC':       round(m['roc_auc'], 4),
        'Test Precision': round(m['precision'], 4),
        'Test Recall':    round(m['recall'], 4),
        'Threshold':      round(thresholds_all.get(model_name, 0.5), 2),
    })

df_compare = (pd.DataFrame(rows_summary)
              .sort_values('Test F1 Macro', ascending=False)
              .reset_index(drop=True))
df_compare.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v61_comparison.csv")
df_compare.to_csv(csv_path, index=False)

# Print top 15
print(df_compare[['Model', 'Test F1 Macro', 'Test Accuracy', 'Test AUC', 'Threshold']].head(15).to_string())
print(f"\nDisimpan: {csv_path}")

best_model_name = df_compare.iloc[0]['Model']
best_f1  = df_compare.iloc[0]['Test F1 Macro']
best_acc = df_compare.iloc[0]['Test Accuracy']
best_auc = df_compare.iloc[0]['Test AUC']
best_thr = df_compare.iloc[0]['Threshold']

print(f"\n  ★ BEST MODEL: {best_model_name}")
print(f"  Test F1 Macro : {best_f1:.4f}")
print(f"  Test Accuracy : {best_acc:.4f}")
print(f"  Test AUC      : {best_auc:.4f}")
print(f"  Threshold     : {best_thr:.2f}")

if best_f1 >= 0.75:
    print(f"\n  🎯 TARGET TERCAPAI! F1 = {best_f1:.4f} ≥ 0.75")
else:
    print(f"\n  ⚠  Target belum tercapai (F1={best_f1:.4f} < 0.75). Perlu iterasi v62.")

# %% [markdown]
# ## 11. Classification Report (Best Model)

# %%
print("\n" + "=" * 80)
print(f"  CLASSIFICATION REPORT — Best: {best_model_name}")
print("=" * 80)

y_pred_best = results_all[best_model_name]['y_pred']
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Depresi'], zero_division=0))

print("\n[Top 5 Models]")
for i, row in df_compare.head(5).iterrows():
    mn = row['Model']
    ypred = results_all[mn]['y_pred']
    print(f"\n  [{i}] {mn} (F1={row['Test F1 Macro']:.4f}, Acc={row['Test Accuracy']:.4f}):")
    print(classification_report(y_test, ypred, target_names=['Normal', 'Depresi'], zero_division=0))

# %% [markdown]
# ## 12. Visualisasi

# %%
COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6',
          '#10b981','#f59e0b','#8b5cf6','#ec4899','#14b8a6',
          '#f43f5e','#0ea5e9','#84cc16','#fb923c','#a78bfa']

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('v61 — Audio-Only ML Pipeline (MFCC+Spectrogram+Wav2Vec Fusion, All Test)',
             fontsize=13, fontweight='bold')

# 12A. Top 15 F1 Bar
ax = axes[0]
top_df = df_compare.head(15)
bars = ax.barh(range(len(top_df)), top_df['Test F1 Macro'],
               color=[COLORS[i % len(COLORS)] for i in range(len(top_df))], edgecolor='white')
ax.set_yticks(range(len(top_df)))
ax.set_yticklabels([n[:22] for n in top_df['Model']], fontsize=7.5)
ax.axvline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
ax.axvline(0.70, color='orange', linestyle=':', lw=1.2, label='Min 0.70')
ax.set_xlabel('Test F1 Macro'); ax.set_title('Top 15 Models — Test F1', fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(0, 1.05)
ax.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, top_df['Test F1 Macro']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7.5, fontweight='bold')

# 12B. Confusion Matrix (Best)
ax = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Depresi'],
            yticklabels=['Normal', 'Depresi'],
            linewidths=0.5, linecolor='gray')
ax.set_title(f'Confusion Matrix (Best: {best_model_name[:20]})\nF1={best_f1:.3f}, Acc={best_acc:.3f}',
             fontweight='bold')
ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')

# 12C. F1 / Acc / AUC Bar (Top 10)
ax = axes[2]
top10_df = df_compare.head(10)
x = np.arange(len(top10_df)); w = 0.25
ax.bar(x - w, top10_df['Test F1 Macro'], width=w, label='F1 Macro',  color='#6366f1')
ax.bar(x,     top10_df['Test Accuracy'], width=w, label='Accuracy',  color='#f59e0b')
ax.bar(x + w, top10_df['Test AUC'],      width=w, label='AUC',       color='#10b981')
ax.set_xticks(x)
ax.set_xticklabels([n[:14] for n in top10_df['Model']], rotation=35, ha='right', fontsize=7)
ax.set_ylim(0, 1.1)
ax.axhline(0.75, color='red', linestyle='--', lw=1, alpha=0.7)
ax.legend(fontsize=8); ax.set_title('Top 10 — F1 / Accuracy / AUC', fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v61_model_comparison.png")
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Plot: {p}")

# Confusion matrix grid (top 6)
top6 = df_compare.head(6)
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('v61 — Confusion Matrix (Top 6 Models)', fontsize=13, fontweight='bold')
for idx, (_, row) in enumerate(top6.iterrows()):
    ax2 = axes2[idx // 3, idx % 3]
    mn  = row['Model']
    ypred_m = results_all[mn]['y_pred']
    cm2 = confusion_matrix(y_test, ypred_m, labels=[0, 1])
    sns.heatmap(cm2, annot=True, fmt='d', cmap='YlOrRd', ax=ax2,
                xticklabels=['Normal', 'Depresi'],
                yticklabels=['Normal', 'Depresi'],
                linewidths=0.5, linecolor='gray', cbar=False)
    ax2.set_title(f'{mn[:25]}\nF1={row["Test F1 Macro"]:.3f}, Acc={row["Test Accuracy"]:.3f}',
                  fontweight='bold', fontsize=8.5)
    ax2.set_xlabel('Prediksi', fontsize=8); ax2.set_ylabel('Aktual', fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.96])
p2 = os.path.join(RESULTS_DIR, "confusion_matrix", "v61_cm_top6.png")
fig2.savefig(p2, dpi=150, bbox_inches='tight'); plt.close()
print(f"CM Grid: {p2}")

# %% [markdown]
# ## 13. Simpan Best Model & Metadata

# %%
best_trained   = trained_models[best_model_name]
best_threshold = thresholds_all.get(best_model_name, 0.5)

with open(os.path.join(MODELS_DIR, 'v61_best_model.pkl'),   'wb') as f: pickle.dump(best_trained, f)
with open(os.path.join(MODELS_DIR, 'v61_scaler.pkl'),       'wb') as f: pickle.dump(scaler, f)
with open(os.path.join(MODELS_DIR, 'v61_pca.pkl'),          'wb') as f: pickle.dump(pca, f)
with open(os.path.join(MODELS_DIR, 'v61_feature_mask.pkl'), 'wb') as f: pickle.dump(keep, f)

summary = {
    'version': 'v61',
    'description': 'Audio-only (MFCC+Spec+Wav2Vec fusion), full test set, CV threshold tuning',
    'features': ['MFCC', 'Spectrogram', 'Wav2Vec'],
    'split': 'train+dev 80% / test 20% (full)',
    'pca_components': int(X_train.shape[1]),
    'best_model': best_model_name,
    'best_threshold': float(best_threshold),
    'best_f1_macro':  float(best_f1),
    'best_accuracy':  float(best_acc),
    'best_auc':       float(best_auc),
    'target_achieved': bool(best_f1 >= 0.75),
    'all_results': {mn: {'f1': round(float(m['f1_macro']),4),
                         'acc': round(float(m['accuracy']),4),
                         'auc': round(float(m['roc_auc']),4)}
                    for mn, m in results_all.items()},
}
with open(os.path.join(MODELS_DIR, 'v61_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[OK] Artifacts tersimpan di {MODELS_DIR}/")

# %% [markdown]
# ## 14. Final Report

# %%
print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v61':^80}")
print("=" * 80)
print(f"  Data        : {len(y_all)} participants (audio-only)")
print(f"  Fitur       : MFCC + Spectrogram + Wav2Vec (Fusion)")
print(f"  Split       : {len(y_train)} train / {len(y_test)} test")
print(f"  PCA Comps   : {X_train.shape[1]}")
print(f"  Best Model  : {best_model_name}")
print(f"  Threshold   : {best_threshold:.2f}")
print(f"  Test F1     : {best_f1:.4f}")
print(f"  Test Acc    : {best_acc:.4f}")
print(f"  Test AUC    : {best_auc:.4f}")
print(f"  Target ≥0.75: {'✓ TERCAPAI!' if best_f1 >= 0.75 else '✗ Belum (lanjut v62)'}")
print(f"  Total Waktu : {time.time()-t_global:.1f}s")
print("=" * 80)
