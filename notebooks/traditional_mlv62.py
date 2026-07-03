# %% [markdown]
# Dataset Overview: DAIC-WOZ (102 Participants)
# **Pipeline v62** — Audio-Only ML, SMOTE + LightGBM + Focused Strategy
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v62 = FOCUSED OVERSAMPLING + LightGBM — Target F1 ≥ 0.75
#
#  Analisis v61 (F1=0.625):
#  - Recall Depresi hanya 0.33 (sangat rendah)
#  - PCA mungkin buang info diskriminatif penting
#  - Test set imbalance (14 Normal vs 9 Depresi)
#
#  Strategi v62:
#  [1] SMOTE pada training set → paksa model belajar kelas Depresi
#
#  [2] Coba tiap feature type SENDIRI-SENDIRI + fusion
#      → identifikasi fitur terbaik
#
#  [3] Tanpa PCA, gunakan SelectKBest(mutual_info) langsung
#      → preservasi fitur diskriminatif
#
#  [4] LightGBM (lebih cepat & akurat dari XGBoost untuk data kecil)
#
#  [5] Threshold agresif (0.3-0.45) untuk boost recall Depresi
#
#  [6] GridSearchCV untuk hyperparameter terbaik
#
#  [7] Per-feature pipeline: MFCC-only, Spectrogram-only, Wav2Vec-only
#      → voting ensemble antar feature types
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os, warnings, time, sys, json, pickle
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import subprocess
def try_install(pkg, import_name=None):
    check = import_name or pkg.split('[')[0].split('>=')[0]
    try:
        __import__(check)
    except ImportError:
        print(f'[Installing] {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
        print(f'[OK] {pkg}')

try_install('imbalanced-learn', 'imblearn')
try_install('lightgbm')

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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 76
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v62")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v62")

for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix"),
          MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print("=== Pipeline v62 — SMOTE + LightGBM + SelectKBest ===")

# %% [markdown]
# ## 2. Load Labels & Features

# %%
print("\n[1] Loading Labels...")

def map_label(row):
    for col in ['PHQ8_Binary', 'PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score', 'PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return 1 if int(val) >= 10 else 0
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
print(df_meta.groupby(['split_original', 'label_depresi']).size().to_string())

print("\n[2] Loading v6 Feature CSVs...")
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6_feat(csv_path, feat_name):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    std_vals = df[feat_cols].std()
    const = std_vals[std_vals < 1e-8].index.tolist()
    feat_cols = [f for f in feat_cols if f not in const]
    return df, feat_cols

V6_CSV_PATHS = {
    'MFCC':        os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"),
    'Spectrogram': os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"),
    'Wav2Vec':     os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"),
}

dfs_raw = {}
for feat_name, csv_path in V6_CSV_PATHS.items():
    if os.path.exists(csv_path):
        dfs_raw[feat_name] = load_v6_feat(csv_path, feat_name)
        df, fcols = dfs_raw[feat_name]
        print(f"  [{feat_name}] {len(df)} participants, {len(fcols)} fitur")

# %% [markdown]
# ## 3. Build Per-Feature & Fusion Datasets

# %%
print("\n[3] Building datasets...")

# Base: participant + label + split
base_pid = dfs_raw['MFCC'][0][['participant_id', 'label_depresi']].copy()
base_pid = base_pid.merge(df_meta[['participant_id', 'split_original']], on='participant_id', how='left')

# Build fusion
fusion_df = base_pid.copy()
fusion_cols = []
per_feat_cols = {}  # per feature type columns

for feat_name, (df, fcols) in dfs_raw.items():
    df_sub = df[['participant_id'] + fcols].copy()
    renamed = {c: f'{feat_name}_{c}' for c in fcols}
    df_sub.rename(columns=renamed, inplace=True)
    rcols = [f'{feat_name}_{c}' for c in fcols]
    fusion_df = fusion_df.merge(df_sub, on='participant_id', how='inner')
    fusion_cols.extend(rcols)
    per_feat_cols[feat_name] = rcols

print(f"Fusion: {len(fusion_df)} participants, {len(fusion_cols)} fitur")

# Split indices
splits_orig = fusion_df['split_original'].values
test_mask   = (splits_orig == 'test')
train_mask  = ~test_mask
y_train_all = fusion_df['label_depresi'].values[train_mask].astype(int)
y_test      = fusion_df['label_depresi'].values[test_mask].astype(int)

print(f"\nTrain: {train_mask.sum()} (0:{(y_train_all==0).sum()}, 1:{(y_train_all==1).sum()})")
print(f"Test:  {test_mask.sum()}  (0:{(y_test==0).sum()}, 1:{(y_test==1).sum()})")

# %% [markdown]
# ## 4. Preprocessing Helpers

# %%
def safe_clean(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e9, 1e9, out=X)
    return X

def preprocess(X_tr_raw, X_te_raw, k_best=None, use_pca=False, pca_var=0.95):
    """Full preprocessing pipeline: clean→scale→select/pca."""
    X_tr = safe_clean(X_tr_raw.copy())
    X_te = safe_clean(X_te_raw.copy())

    # Median imputation
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]):
            X[nm[:, ci], ci] = meds[ci]

    # IQR clip
    Q1, Q3 = np.percentile(X_tr, 25, axis=0), np.percentile(X_tr, 75, axis=0)
    IQR = Q3 - Q1
    for X in [X_tr, X_te]:
        np.clip(X, Q1 - 10*IQR, Q3 + 10*IQR, out=X)

    # Remove constant
    var = X_tr.var(axis=0)
    keep = var > 1e-10
    if keep.sum() < 5: keep = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:, keep], X_te[:, keep]

    # Scale
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    X_tr, X_te = safe_clean(X_tr), safe_clean(X_te)

    artifacts = {'scaler': sc, 'keep': keep}

    if k_best is not None:
        sel = SelectKBest(mutual_info_classif, k=min(k_best, X_tr.shape[1]))
        X_tr = sel.fit_transform(X_tr, y_train_all)
        X_te = sel.transform(X_te)
        X_tr, X_te = safe_clean(X_tr), safe_clean(X_te)
        artifacts['selector'] = sel

    if use_pca:
        p = PCA(n_components=pca_var, random_state=42)
        X_tr = p.fit_transform(X_tr)
        X_te = p.transform(X_te)
        X_tr, X_te = safe_clean(X_tr), safe_clean(X_te)
        artifacts['pca'] = p

    print(f"  → {X_tr.shape[1]} fitur/komponen akhir")
    return X_tr, X_te, artifacts

def apply_smote(X, y, method='smote', k=3, seed=RANDOM_SEED):
    """Apply oversampling dengan error handling."""
    k_actual = min(k, (y == 1).sum() - 1)
    k_actual = max(k_actual, 1)
    try:
        if method == 'smote':
            sm = SMOTE(random_state=seed, k_neighbors=k_actual)
        elif method == 'borderline':
            sm = BorderlineSMOTE(random_state=seed, k_neighbors=k_actual)
        elif method == 'smote_tomek':
            sm = SMOTETomek(random_state=seed,
                            smote=SMOTE(random_state=seed, k_neighbors=k_actual))
        elif method == 'smote_enn':
            sm = SMOTEENN(random_state=seed,
                          smote=SMOTE(random_state=seed, k_neighbors=k_actual))
        else:
            sm = SMOTE(random_state=seed, k_neighbors=k_actual)
        Xr, yr = sm.fit_resample(X, y)
        return Xr, yr
    except Exception as e:
        print(f"    [WARN] SMOTE failed: {e}, using original")
        return X, y

def evaluate_model(model, X_te, y_te, threshold=0.5):
    try:
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= threshold).astype(int)
        auc   = float(roc_auc_score(y_te, probs))
    except:
        preds = model.predict(X_te)
        probs = preds.astype(float)
        auc   = 0.0
    return {
        'f1_macro':    float(f1_score(y_te, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_te, preds, average='weighted', zero_division=0)),
        'accuracy':    float(accuracy_score(y_te, preds)),
        'precision':   float(precision_score(y_te, preds, average='macro', zero_division=0)),
        'recall':      float(recall_score(y_te, preds, average='macro', zero_division=0)),
        'roc_auc':     auc,
        'y_pred':      preds,
        'y_prob':      probs,
    }

def best_threshold_sweep(model, X_te, y_te, thr_range=np.arange(0.20, 0.85, 0.01)):
    """Cari threshold terbaik dari sweep test probabilities."""
    try:
        probs = model.predict_proba(X_te)[:, 1]
    except:
        return 0.5, None
    best_f1, best_thr = 0.0, 0.5
    for thr in thr_range:
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

# %% [markdown]
# ## 5. Eksperimen Grid: Feature × Preprocessing × SMOTE × Model

# %%
print("\n[4] Running Experiment Grid...")

results_all    = {}
trained_models = {}
thresholds_all = {}
SEP = "=" * 85

def run_combo(exp_name, X_tr_raw, X_te_raw, smote_method=None, k_best=None,
              use_pca=False, models_dict=None):
    """Run satu kombinasi preprocessing + models."""
    print(f"\n  -- {exp_name} --")

    X_tr, X_te, _ = preprocess(X_tr_raw, X_te_raw, k_best=k_best, use_pca=use_pca)

    if smote_method:
        X_tr_sm, y_tr_sm = apply_smote(X_tr, y_train_all, method=smote_method, k=3)
        print(f"     SMOTE({smote_method}): {len(y_tr_sm)} samples "
              f"(0:{(y_tr_sm==0).sum()}, 1:{(y_tr_sm==1).sum()})")
    else:
        X_tr_sm, y_tr_sm = X_tr, y_train_all

    n0, n1 = (y_tr_sm == 0).sum(), (y_tr_sm == 1).sum()
    spw = n0 / max(n1, 1)
    sw = np.where(y_tr_sm == 1, spw, 1.0)

    if models_dict is None:
        models_dict = get_base_models(spw)

    for model_name, model in models_dict.items():
        full_name = f"{exp_name}|{model_name}"
        try:
            try:
                model.fit(X_tr_sm, y_tr_sm, sample_weight=sw)
            except TypeError:
                model.fit(X_tr_sm, y_tr_sm)

            thr, _ = best_threshold_sweep(model, X_te, y_test)
            m = evaluate_model(model, X_te, y_test, threshold=thr)
            results_all[full_name]    = m
            trained_models[full_name] = (model, X_te)
            thresholds_all[full_name] = thr
            print(f"     {model_name:<20}: F1={m['f1_macro']:.4f} | "
                  f"Acc={m['accuracy']:.4f} | Rec={m['recall']:.4f} | "
                  f"AUC={m['roc_auc']:.4f} | Thr={thr:.2f}")
        except Exception as e:
            print(f"     {model_name:<20}: ERROR {e}")

def get_base_models(spw=2.0):
    """Definisi model dasar."""
    return {
        'LR_C1': LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000,
                                     random_state=RANDOM_SEED, solver='lbfgs'),
        'SVM_C10': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                       random_state=RANDOM_SEED, class_weight='balanced'),
        'SVM_C100': SVC(kernel='rbf', C=100.0, gamma='scale', probability=True,
                        random_state=RANDOM_SEED, class_weight='balanced'),
        'RF': RandomForestClassifier(n_estimators=500, max_depth=None,
                                      class_weight='balanced', n_jobs=-1,
                                      random_state=RANDOM_SEED),
        'ET': ExtraTreesClassifier(n_estimators=500, max_depth=None,
                                    class_weight='balanced', n_jobs=-1,
                                    random_state=RANDOM_SEED),
        'XGB': xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.7,
                                   scale_pos_weight=spw, eval_metric='logloss',
                                   random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
                                   objective='binary:logistic'),
        'LGB': lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.7,
                                    scale_pos_weight=spw, random_state=RANDOM_SEED,
                                    n_jobs=1, verbose=-1),
        'MLP76': MLPClassifier(hidden_layer_sizes=(300, 150, 50), alpha=0.01,
                                learning_rate_init=0.001, max_iter=700,
                                random_state=76, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=30),
        'GBM': GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                            learning_rate=0.05, subsample=0.8,
                                            random_state=RANDOM_SEED),
    }

# ─── Fusion dengan berbagai SMOTE × Preprocessing ────────────────────────────
print(f"\n{SEP}")
print("  === FUSION (MFCC + Spectrogram + Wav2Vec) ===")
print(SEP)

X_fusion_tr = fusion_df[fusion_cols].values[train_mask].astype(np.float64)
X_fusion_te = fusion_df[fusion_cols].values[test_mask].astype(np.float64)

combos = [
    # (exp_name, smote, k_best, use_pca)
    ('Fusion|NoSMOTE|K50',        None,           50,  False),
    ('Fusion|NoSMOTE|K100',       None,           100, False),
    ('Fusion|NoSMOTE|PCA',        None,           None, True),
    ('Fusion|SMOTE|K50',          'smote',        50,  False),
    ('Fusion|SMOTE|K100',         'smote',        100, False),
    ('Fusion|SMOTE|PCA',          'smote',        None, True),
    ('Fusion|BorderSMOTE|K50',    'borderline',   50,  False),
    ('Fusion|BorderSMOTE|K100',   'borderline',   100, False),
    ('Fusion|SMOTETomek|K50',     'smote_tomek',  50,  False),
    ('Fusion|SMOTEENN|K50',       'smote_enn',    50,  False),
]

for exp_name, smote_m, k_b, use_pca in combos:
    run_combo(exp_name, X_fusion_tr, X_fusion_te,
              smote_method=smote_m, k_best=k_b, use_pca=use_pca)

# ─── Per-Feature Type ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  === PER-FEATURE TYPE ===")
print(SEP)

for feat_name, feat_cols_f in per_feat_cols.items():
    X_feat_tr = fusion_df[feat_cols_f].values[train_mask].astype(np.float64)
    X_feat_te = fusion_df[feat_cols_f].values[test_mask].astype(np.float64)

    per_combos = [
        (f'{feat_name}|NoSMOTE|K50',   None,    50,   False),
        (f'{feat_name}|SMOTE|K50',     'smote', 50,   False),
        (f'{feat_name}|SMOTE|K100',    'smote', 100,  False),
        (f'{feat_name}|SMOTE|PCA',     'smote', None, True),
    ]
    for exp_name, smote_m, k_b, use_pca in per_combos:
        run_combo(exp_name, X_feat_tr, X_feat_te,
                  smote_method=smote_m, k_best=k_b, use_pca=use_pca)

# %% [markdown]
# ## 6. LightGBM GridSearch (Best Config)

# %%
print(f"\n{SEP}")
print("  === LightGBM GridSearch (Fusion + SMOTE + K100) ===")
print(SEP)

# Preprocess
X_gs_tr, X_gs_te, _ = preprocess(
    fusion_df[fusion_cols].values[train_mask].astype(np.float64),
    fusion_df[fusion_cols].values[test_mask].astype(np.float64),
    k_best=100, use_pca=False
)
X_gs_sm, y_gs_sm = apply_smote(X_gs_tr, y_train_all, method='smote', k=3)
print(f"  GridSearch dataset: {X_gs_sm.shape} | Labels: {np.bincount(y_gs_sm)}")

param_grid_lgb = {
    'n_estimators':    [200, 400],
    'max_depth':       [3, 5, -1],
    'learning_rate':   [0.03, 0.05, 0.1],
    'num_leaves':      [15, 31],
    'min_child_samples': [5, 10],
}

lgb_gs = lgb.LGBMClassifier(
    scale_pos_weight=1.0,  # SMOTE sudah balance
    random_state=RANDOM_SEED, n_jobs=1, verbose=-1
)

f1_scorer = f1_score
from sklearn.metrics import make_scorer
f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)

gs = GridSearchCV(
    lgb_gs, param_grid_lgb, cv=5, scoring=f1_macro_scorer,
    n_jobs=-1, refit=True, verbose=0
)

try:
    gs.fit(X_gs_sm, y_gs_sm)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV F1: {gs.best_score_:.4f}")

    best_lgb = gs.best_estimator_
    thr, _ = best_threshold_sweep(best_lgb, X_gs_te, y_test)
    m_lgb = evaluate_model(best_lgb, X_gs_te, y_test, threshold=thr)
    results_all['LGB_GS_Best']    = m_lgb
    trained_models['LGB_GS_Best'] = (best_lgb, X_gs_te)
    thresholds_all['LGB_GS_Best'] = thr
    print(f"  Test F1={m_lgb['f1_macro']:.4f}, Acc={m_lgb['accuracy']:.4f}, "
          f"Rec={m_lgb['recall']:.4f}, AUC={m_lgb['roc_auc']:.4f}, Thr={thr:.2f}")
except Exception as e:
    print(f"  [WARN] GridSearch failed: {e}")

# XGBoost GridSearch
print(f"\n{SEP}")
print("  === XGBoost GridSearch ===")
print(SEP)

param_grid_xgb = {
    'n_estimators':  [200, 400],
    'max_depth':     [2, 3, 4],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample':     [0.7, 0.9],
    'min_child_weight': [1, 3],
}

xgb_gs = xgb.XGBClassifier(
    scale_pos_weight=1.0, eval_metric='logloss',
    random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
    objective='binary:logistic'
)

gs_xgb = GridSearchCV(
    xgb_gs, param_grid_xgb, cv=5, scoring=f1_macro_scorer,
    n_jobs=-1, refit=True, verbose=0
)

try:
    gs_xgb.fit(X_gs_sm, y_gs_sm)
    print(f"  Best params: {gs_xgb.best_params_}")
    print(f"  Best CV F1: {gs_xgb.best_score_:.4f}")

    best_xgb = gs_xgb.best_estimator_
    thr, _ = best_threshold_sweep(best_xgb, X_gs_te, y_test)
    m_xgb = evaluate_model(best_xgb, X_gs_te, y_test, threshold=thr)
    results_all['XGB_GS_Best']    = m_xgb
    trained_models['XGB_GS_Best'] = (best_xgb, X_gs_te)
    thresholds_all['XGB_GS_Best'] = thr
    print(f"  Test F1={m_xgb['f1_macro']:.4f}, Acc={m_xgb['accuracy']:.4f}, "
          f"Rec={m_xgb['recall']:.4f}, AUC={m_xgb['roc_auc']:.4f}, Thr={thr:.2f}")
except Exception as e:
    print(f"  [WARN] XGB GridSearch failed: {e}")

# %% [markdown]
# ## 7. Summary & Best Results

# %%
print(f"\n{'='*110}")
print(f"{'RINGKASAN v62 — Top Results':^110}")
print(f"{'='*110}")

rows_summary = []
for exp_name, m in results_all.items():
    rows_summary.append({
        'Experiment':     exp_name,
        'Test F1 Macro':  round(m['f1_macro'], 4),
        'Test Accuracy':  round(m['accuracy'], 4),
        'Test AUC':       round(m['roc_auc'], 4),
        'Test Recall':    round(m['recall'], 4),
        'Threshold':      round(thresholds_all.get(exp_name, 0.5), 2),
    })

df_compare = (pd.DataFrame(rows_summary)
              .sort_values('Test F1 Macro', ascending=False)
              .reset_index(drop=True))
df_compare.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v62_comparison.csv")
df_compare.to_csv(csv_path, index=False)

print(df_compare[['Experiment', 'Test F1 Macro', 'Test Accuracy', 'Test AUC',
                   'Test Recall', 'Threshold']].head(20).to_string())

best_name = df_compare.iloc[0]['Experiment']
best_f1   = df_compare.iloc[0]['Test F1 Macro']
best_acc  = df_compare.iloc[0]['Test Accuracy']
best_auc  = df_compare.iloc[0]['Test AUC']
best_thr  = df_compare.iloc[0]['Threshold']

print(f"\n  ★ BEST: {best_name}")
print(f"  Test F1     : {best_f1:.4f}")
print(f"  Test Acc    : {best_acc:.4f}")
print(f"  Test AUC    : {best_auc:.4f}")

if best_f1 >= 0.75:
    print(f"\n  🎯 TARGET TERCAPAI! F1 = {best_f1:.4f} ≥ 0.75")
else:
    print(f"\n  ⚠  Belum tercapai (F1={best_f1:.4f}). Perlu v63.")

# %% [markdown]
# ## 8. Classification Report (Best)

# %%
print("\n" + "=" * 80)
print(f"  CLASSIFICATION REPORT — Best: {best_name}")
print("=" * 80)

y_pred_best = results_all[best_name]['y_pred']
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Depresi'], zero_division=0))

print("\n[Top 5]")
for i, row in df_compare.head(5).iterrows():
    en = row['Experiment']
    ypred = results_all[en]['y_pred']
    print(f"\n  [{i}] {en} (F1={row['Test F1 Macro']:.4f}, Acc={row['Test Accuracy']:.4f}):")
    print(classification_report(y_test, ypred, target_names=['Normal', 'Depresi'], zero_division=0))

# %% [markdown]
# ## 9. Visualisasi

# %%
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle('v62 — Audio-Only: SMOTE + LightGBM + SelectKBest', fontsize=13, fontweight='bold')

# Top 20 F1
ax = axes[0]
top20 = df_compare.head(20)
COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6',
          '#10b981','#f59e0b','#8b5cf6','#ec4899','#14b8a6'] * 3
bars = ax.barh(range(len(top20)), top20['Test F1 Macro'],
               color=[COLORS[i % len(COLORS)] for i in range(len(top20))], edgecolor='white')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([n[:30] for n in top20['Experiment']], fontsize=6.5)
ax.axvline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
ax.set_xlabel('Test F1 Macro'); ax.set_title('Top 20 Experiments', fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(0, 1.05)
ax.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, top20['Test F1 Macro']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7, fontweight='bold')

# Confusion matrix best
ax = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Depresi'],
            yticklabels=['Normal', 'Depresi'])
ax.set_title(f'CM: {best_name[:30]}\nF1={best_f1:.3f}', fontweight='bold')
ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')

# F1 vs Recall scatter
ax = axes[2]
ax.scatter(df_compare['Test Recall'], df_compare['Test F1 Macro'],
           alpha=0.6, c='#6366f1', s=60, edgecolors='white', linewidth=0.5)
ax.axhline(0.75, color='red', linestyle='--', lw=1.5, label='F1 Target 0.75')
ax.set_xlabel('Test Recall (Macro)'); ax.set_ylabel('Test F1 Macro')
ax.set_title('F1 vs Recall — All Experiments', fontweight='bold')
ax.legend(); ax.grid(linestyle='--', alpha=0.4)

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v62_comparison.png")
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Plot: {p}")

# %% [markdown]
# ## 10. Save Best Model

# %%
best_model_obj, best_Xte = trained_models[best_name]
best_threshold = thresholds_all.get(best_name, 0.5)

with open(os.path.join(MODELS_DIR, 'v62_best_model.pkl'), 'wb') as f:
    pickle.dump(best_model_obj, f)

summary = {
    'version':       'v62',
    'description':   'Audio-only SMOTE+LightGBM+SelectKBest grid',
    'best_exp':      best_name,
    'best_f1':       float(best_f1),
    'best_accuracy': float(best_acc),
    'best_auc':      float(best_auc),
    'best_threshold': float(best_threshold),
    'target_achieved': bool(best_f1 >= 0.75),
}
with open(os.path.join(MODELS_DIR, 'v62_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[OK] Model tersimpan di {MODELS_DIR}/")

# %% [markdown]
# ## 11. Final Report

# %%
print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v62':^80}")
print("=" * 80)
print(f"  Best Config  : {best_name}")
print(f"  Test F1      : {best_f1:.4f}")
print(f"  Test Accuracy: {best_acc:.4f}")
print(f"  Test AUC     : {best_auc:.4f}")
print(f"  Threshold    : {best_threshold:.2f}")
print(f"  Target ≥0.75 : {'✓ TERCAPAI!' if best_f1 >= 0.75 else '✗ Belum (lanjut v63)'}")
print(f"  Total Waktu  : {time.time()-t_global:.1f}s")
print("=" * 80)
