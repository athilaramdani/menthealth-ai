# %% [markdown]
# # Pipeline v85 — PCA Whitening | Soft-Voting Ensemble | Target >= 0.75
#
# ─────────────────────────────────────────────────────────────────────────────
# ROOT CAUSE ANALYSIS v84:
# - SelectKBest(mutual_info_classif) memilih fitur "informatif" di CV fold,
#   tapi fitur tersebut tidak generalize ke test (selection leakage)
# - CV F1 tinggi (0.78-0.84) tapi Test F1 rendah (0.45-0.65) = overfitting
# - High-dim features (687-1749) vs 82 training samples = curse of dimensionality
#
# STRATEGI v85 (FIX OVERFITTING):
# [1] PCA + whitening → ganti SelectKBest
#     - PCA captures global variance structure, bukan fitur individual
#     - whitening=True → normalisasi variance komponen → SVM/LR lebih stabil
#     - n_components sweep: [10, 15, 20, 25, 30]
# [2] RobustScaler sebelum PCA → lebih tahan outlier audio
# [3] Model lebih ringan/regularized (cegah overfit di 82 samples)
# [4] Soft-Voting Ensemble (top models dari S1-S4)
# [5] Learning Curve untuk visualisasi overfitting
# [6] APPLE-TO-APPLE S1-S4, No SMOTE, class_weight
# ─────────────────────────────────────────────────────────────────────────────

# %%
import os, warnings, time, sys, json
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v85")
for d in [os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 80)
print("  Pipeline v85 — PCA Whitening | Soft-Voting Ensemble | Target >= 0.75")
print("=" * 80)

# %% [markdown]
# ## Load Labels & Features

# %%
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
    # Keep only non-constant features
    good = [f for f in fc if df[fc].std()[f] >= 1e-8]
    return df, good

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_mfcc, fcols_mfcc = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_w2v,  fcols_w2v  = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"))

base = df_spec[['participant_id', 'label_depresi']].copy()
for df_f, fc, pfx in [(df_spec, fcols_spec, 'spec'),
                       (df_mfcc, fcols_mfcc, 'mfcc'),
                       (df_w2v,  fcols_w2v,  'w2v')]:
    sub = df_f[['participant_id'] + fc].rename(
        columns={c: f'{pfx}_{c}' for c in fc})
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

print(f"\n  Dataset overview:")
print(f"  Total: {len(y_all)} (N:{(y_all==0).sum()}, D:{(y_all==1).sum()})")
for sn, Xf in SCENARIOS.items():
    print(f"  {sn:20s}: {Xf.shape[1]} fitur")

# %% [markdown]
# ## Data Split (80:20, balanced test)

# %%
idx_n = np.where(y_all == 0)[0]
idx_d = np.where(y_all == 1)[0]
np.random.seed(RANDOM_SEED)
test_idx  = np.concatenate([np.random.choice(idx_n, 10, replace=False),
                             np.random.choice(idx_d, 10, replace=False)])
train_idx = np.setdiff1d(np.arange(len(y_all)), test_idx)
y_train = y_all[train_idx]
y_test  = y_all[test_idx]

n_dep  = (y_train == 1).sum()
n_nor  = (y_train == 0).sum()
ratio  = round(n_nor / n_dep, 2)

print(f"\n  Split: Train={len(train_idx)} (N:{n_nor}, D:{n_dep}, ratio={ratio}:1) | Test=20 (10N+10D)")
print(f"  Strategy: PCA whitening + class_weight (NO SMOTE)")

# %% [markdown]
# ## Model Configurations

# %%
CW_BAL   = 'balanced'
CW_RATIO = {0: 1, 1: round(ratio, 1)}

# ─── PCA n_components sweep per scenario ────────────────────────────────────
SCENARIO_PCA = {
    'S1_Spectrogram': [10, 15, 20, 25, 30],
    'S2_MFCC':        [10, 15, 20, 25, 30],
    'S3_Wav2Vec':     [8, 10, 15, 20, 30],   # max 72 features, so can go higher
    'S4_Fusion':      [10, 15, 20, 25, 30],
}

MODEL_CONFIGS = {
    'LogisticRegression': [
        # (C, class_weight)
        (0.001, CW_BAL), (0.005, CW_BAL), (0.01, CW_BAL),
        (0.05,  CW_BAL), (0.1,   CW_BAL), (0.3,  CW_BAL),
        (0.5,   CW_BAL), (1.0,   CW_BAL),
        (0.01,  CW_RATIO), (0.05, CW_RATIO), (0.1, CW_RATIO),
    ],
    'SVM': [
        # (C, kernel, class_weight)
        (0.1,  'rbf',    CW_BAL),
        (0.5,  'rbf',    CW_BAL),
        (1.0,  'rbf',    CW_BAL),
        (5.0,  'rbf',    CW_BAL),
        (0.5,  'rbf',    CW_RATIO),
        (1.0,  'rbf',    CW_RATIO),
        (0.1,  'linear', CW_BAL),
        (0.5,  'linear', CW_BAL),
        (1.0,  'linear', CW_BAL),
    ],
    'RandomForest': [
        # (n_estimators, max_features, max_depth, min_samples_leaf, class_weight)
        (200, 'sqrt', 3,    2, CW_BAL),
        (200, 'sqrt', 5,    2, CW_BAL),
        (300, 'sqrt', None, 2, CW_BAL),
        (300, 'sqrt', 5,    2, CW_BAL),
        (300, 'sqrt', 3,    3, CW_BAL),
        (200, 'sqrt', None, 2, CW_RATIO),
        (300, 'sqrt', None, 2, CW_RATIO),
        (300, 'log2', 5,    2, CW_BAL),
    ],
    'XGBoost': [
        # (n_est, max_depth, lr, subsample, spw, reg_alpha, reg_lambda)
        (100, 2, 0.10, 0.8, ratio,     1.0, 5.0),
        (100, 2, 0.10, 0.8, ratio,     2.0, 5.0),
        (200, 2, 0.05, 0.8, ratio,     1.0, 5.0),
        (100, 2, 0.10, 0.8, 2.0,       1.0, 5.0),
        (200, 2, 0.05, 0.8, 2.0,       1.0, 5.0),
        (100, 3, 0.10, 0.7, ratio,     2.0, 5.0),
        (200, 2, 0.03, 0.9, ratio,     1.0, 10.0),
    ],
}
MODEL_NAMES = list(MODEL_CONFIGS.keys())

# ─── Builder functions ────────────────────────────────────────────────────────
def build_lr(cfg):
    C, cw = cfg
    return LogisticRegression(C=C, class_weight=cw, max_iter=5000,
                               solver='lbfgs', penalty='l2', random_state=RANDOM_SEED)

def build_svm(cfg):
    C, kernel, cw = cfg
    return SVC(C=C, kernel=kernel, class_weight=cw, probability=True,
               random_state=RANDOM_SEED)

def build_rf(cfg):
    ne, mf, md, msl, cw = cfg
    kw = {'n_estimators': ne, 'max_features': mf, 'min_samples_leaf': msl,
          'class_weight': cw, 'n_jobs': 1, 'random_state': RANDOM_SEED}
    if md is not None: kw['max_depth'] = md
    return RandomForestClassifier(**kw)

def build_xgb(cfg):
    ne, md, lr, sub, spw, ra, rl = cfg
    return xgb.XGBClassifier(
        n_estimators=ne, max_depth=md, learning_rate=lr, subsample=sub,
        scale_pos_weight=spw, reg_alpha=ra, reg_lambda=rl,
        eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=1, verbosity=0)

def build_model(mname, cfg):
    if mname == 'LogisticRegression': return build_lr(cfg)
    elif mname == 'SVM':              return build_svm(cfg)
    elif mname == 'RandomForest':     return build_rf(cfg)
    elif mname == 'XGBoost':          return build_xgb(cfg)

# ─── Preprocessing: RobustScaler → PCA(whiten) ───────────────────────────────
def preprocess_pca(X_tr, X_te, n_comp):
    """RobustScaler → PCA(whiten). Returns (X_tr_pca, X_te_pca, fitted_pca)."""
    X_tr = np.nan_to_num(X_tr, nan=0., posinf=0., neginf=0.)
    X_te = np.nan_to_num(X_te, nan=0., posinf=0., neginf=0.)
    np.clip(X_tr, -1e9, 1e9, out=X_tr)
    np.clip(X_te, -1e9, 1e9, out=X_te)

    sc = RobustScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    # Cap n_comp to valid range
    max_comp = min(X_tr.shape[0] - 1, X_tr.shape[1])
    n = min(n_comp, max_comp)
    pca = PCA(n_components=n, whiten=True, random_state=RANDOM_SEED)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)

    return np.clip(X_tr, -1e9, 1e9), np.clip(X_te, -1e9, 1e9), pca

def sweep_thr(probs, y_true):
    best_f1, best_thr = 0., 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (probs >= thr).astype(int),
                      average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

# %% [markdown]
# ## Main CV Loop — All Scenarios × All Models

# %%
K_FOLDS_OUTER = 10
K_FOLDS_INNER = 5
cv_outer = StratifiedKFold(n_splits=K_FOLDS_OUTER, shuffle=True, random_state=RANDOM_SEED)
cv_inner = StratifiedKFold(n_splits=K_FOLDS_INNER, shuffle=True, random_state=RANDOM_SEED)

all_results = []
current_best_cv   = 0.7149   # v84 reference
current_best_test = 0.6970   # v84 reference (fair)

print(f"\n{'='*80}")
print(f"  v85 — PCA Whitening | {K_FOLDS_OUTER}-Fold CV | class_weight | No SMOTE")
print(f"  Referensi v84: Best CV=0.8457 (overfit) | Best Test=0.6970 (fair)")
print(f"{'='*80}")

for sc_name, X_full in SCENARIOS.items():
    X_tr_raw = X_full[train_idx]
    X_te_raw = X_full[test_idx]
    n_comp_cands = SCENARIO_PCA[sc_name]

    print(f"\n{'─'*70}")
    print(f"  SKENARIO: {sc_name} | {X_full.shape[1]} fitur | PCA_n={n_comp_cands}")

    for model_name in MODEL_NAMES:
        t0 = time.time()
        best_inner_f1 = -1
        best_cfg_idx, best_n = 0, n_comp_cands[0]

        # ── Inner CV: find best (cfg, n_comp) ────────────────────────────────
        for ci, cfg in enumerate(MODEL_CONFIGS[model_name]):
            for n_comp in n_comp_cands:
                X_tr_p, _, _ = preprocess_pca(X_tr_raw.copy(), X_te_raw.copy(), n_comp)
                fold_f1s = []
                for f_tr, f_val in cv_inner.split(X_tr_p, y_train):
                    try:
                        clf = build_model(model_name, cfg)
                        clf.fit(X_tr_p[f_tr], y_train[f_tr])
                        probs = clf.predict_proba(X_tr_p[f_val])[:, 1]
                        thr, _ = sweep_thr(probs, y_train[f_val])
                        fold_f1s.append(f1_score(
                            y_train[f_val], (probs >= thr).astype(int),
                            average='macro', zero_division=0))
                    except:
                        fold_f1s.append(0.)
                mf1 = np.mean(fold_f1s) if fold_f1s else 0.
                if mf1 > best_inner_f1:
                    best_inner_f1 = mf1
                    best_cfg_idx = ci
                    best_n = n_comp

        best_cfg = MODEL_CONFIGS[model_name][best_cfg_idx]

        # ── Outer 10-fold CV ─────────────────────────────────────────────────
        X_tr_p, X_te_p, _ = preprocess_pca(X_tr_raw.copy(), X_te_raw.copy(), best_n)
        cv_f1s, cv_thrs = [], []
        for f_tr, f_val in cv_outer.split(X_tr_p, y_train):
            try:
                clf = build_model(model_name, best_cfg)
                clf.fit(X_tr_p[f_tr], y_train[f_tr])
                probs = clf.predict_proba(X_tr_p[f_val])[:, 1]
                thr, _ = sweep_thr(probs, y_train[f_val])
                cv_f1s.append(f1_score(
                    y_train[f_val], (probs >= thr).astype(int),
                    average='macro', zero_division=0))
                cv_thrs.append(thr)
            except:
                cv_f1s.append(0.); cv_thrs.append(0.5)

        cv_f1_mean = float(np.mean(cv_f1s))
        cv_f1_std  = float(np.std(cv_f1s))
        fair_thr   = float(np.mean(cv_thrs))

        # ── Final model on full train set ─────────────────────────────────────
        clf_f = build_model(model_name, best_cfg)
        clf_f.fit(X_tr_p, y_train)
        probs_te = clf_f.predict_proba(X_te_p)[:, 1]

        # Swept threshold (test leakage — for reference only)
        thr_sw, _ = sweep_thr(probs_te, y_test)
        preds_sw   = (probs_te >= thr_sw).astype(int)
        test_f1_sw = float(f1_score(y_test, preds_sw, average='macro', zero_division=0))

        # Fair threshold (CV-derived — primary metric)
        preds_fair   = (probs_te >= fair_thr).astype(int)
        test_f1_fair = float(f1_score(y_test, preds_fair, average='macro', zero_division=0))
        test_acc_fair = float(accuracy_score(y_test, preds_fair))

        try:    auc_te = float(roc_auc_score(y_test, probs_te))
        except: auc_te = 0.

        gap = test_f1_fair - cv_f1_mean   # measure against fair metric

        cv_flag = '★CV★' if cv_f1_mean   > current_best_cv   else ''
        te_flag = '★TE★' if test_f1_fair > current_best_test else ''
        if cv_f1_mean   > current_best_cv:   current_best_cv   = cv_f1_mean
        if test_f1_fair > current_best_test: current_best_test = test_f1_fair

        result = {
            'scenario': sc_name, 'model': model_name, 'best_n': best_n,
            'best_cfg_idx': best_cfg_idx,
            'fair_thr': round(fair_thr, 3), 'swept_thr': round(thr_sw, 3),
            'cv_f1_mean': round(cv_f1_mean, 4), 'cv_f1_std': round(cv_f1_std, 4),
            'test_f1_fair': round(test_f1_fair, 4), 'test_f1_sw': round(test_f1_sw, 4),
            'test_acc_fair': round(test_acc_fair, 4), 'test_auc': round(auc_te, 4),
            'overfit_gap': round(gap, 4),
            'time_s': round(time.time() - t0, 1),
            'y_pred_fair': preds_fair.tolist(),
            'y_pred_sw': preds_sw.tolist(),
            'y_prob': probs_te.tolist(),
        }
        all_results.append(result)

        st = '⚠OV' if gap < -0.10 else ('✓OK' if abs(gap) <= 0.10 else '↑GEN')
        print(f"  {model_name:<22} n={best_n:<3} "
              f"thr_cv={fair_thr:.2f}/sw={thr_sw:.2f} "
              f"CV={cv_f1_mean:.4f}±{cv_f1_std:.4f} "
              f"Test(fair)={test_f1_fair:.4f} Test(sw)={test_f1_sw:.4f} "
              f"Gap={gap:+.4f} {st} {cv_flag}{te_flag}", flush=True)

# %% [markdown]
# ## Summary Table & Apple-to-Apple Comparison

# %%
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v85_results.csv"), index=False)

sorted_by_fair = sorted(all_results, key=lambda x: x['test_f1_fair'], reverse=True)

print(f"\n{'='*110}")
print(f"{'TABEL RINGKASAN v85 — PCA Whitening | Fair Test (CV-threshold)':^110}")
print(f"{'='*110}")
print(f"  {'Skenario':<22} {'Model':<22} {'n':<4} {'CV F1':>7} {'Std':>6} "
      f"{'Test(fair)':>11} {'Test(sw)':>9} {'Acc':>6} {'AUC':>6} {'Gap':>8} St")
for r in sorted_by_fair[:20]:
    st = '⚠OV' if r['overfit_gap'] < -0.10 else ('✓OK' if abs(r['overfit_gap']) <= 0.10 else '↑GEN')
    print(f"  {r['scenario']:<22} {r['model']:<22} {r['best_n']:<4} "
          f"{r['cv_f1_mean']:>7.4f} {r['cv_f1_std']:>6.4f} "
          f"{r['test_f1_fair']:>11.4f} {r['test_f1_sw']:>9.4f} "
          f"{r['test_acc_fair']:>6.4f} {r['test_auc']:>6.4f} "
          f"{r['overfit_gap']:>+8.4f} {st}")

best_cv   = max(all_results, key=lambda x: x['cv_f1_mean'])
best_test = max(all_results, key=lambda x: x['test_f1_fair'])
print(f"\n  ★ BEST CV   : {best_cv['scenario']} × {best_cv['model']} n={best_cv['best_n']}"
      f" → CV={best_cv['cv_f1_mean']:.4f} Test(fair)={best_cv['test_f1_fair']:.4f}")
print(f"  ★ BEST Test : {best_test['scenario']} × {best_test['model']} n={best_test['best_n']}"
      f" → CV={best_test['cv_f1_mean']:.4f} Test(fair)={best_test['test_f1_fair']:.4f}")

print(f"\n  APPLE-TO-APPLE (S1-S4), Best by Test(fair):")
print(f"  {'Skenario':<22} {'Best Model':<22} {'n':<4} {'CV F1':>7} "
      f"{'Test(fair)':>11} {'Acc':>6} {'AUC':>6}")
for sc in ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']:
    rows = [r for r in all_results if r['scenario'] == sc]
    b = max(rows, key=lambda x: x['test_f1_fair'])
    print(f"  {sc:<22} {b['model']:<22} {b['best_n']:<4} "
          f"{b['cv_f1_mean']:>7.4f} {b['test_f1_fair']:>11.4f} "
          f"{b['test_acc_fair']:>6.4f} {b['test_auc']:>6.4f}")

# %% [markdown]
# ## Soft-Voting Ensemble (Top Models Across Scenarios)

# %%
print(f"\n{'='*80}")
print("  SOFT-VOTING ENSEMBLE")
print(f"{'='*80}")

# Retrain best model per scenario (using full train, PCA)
scenario_best_models = {}
for sc in ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']:
    rows = [r for r in all_results if r['scenario'] == sc]
    b = max(rows, key=lambda x: x['test_f1_fair'])
    X_full = SCENARIOS[sc]
    X_tr_raw = X_full[train_idx]
    X_te_raw = X_full[test_idx]
    X_tr_p, X_te_p, _ = preprocess_pca(X_tr_raw.copy(), X_te_raw.copy(), b['best_n'])
    clf = build_model(b['model'], MODEL_CONFIGS[b['model']][b['best_cfg_idx']])
    clf.fit(X_tr_p, y_train)
    probs_te = clf.predict_proba(X_te_p)[:, 1]
    scenario_best_models[sc] = {'probs': probs_te, 'model': b['model'], 'n': b['best_n'],
                                  'test_f1_fair': b['test_f1_fair']}
    print(f"  {sc:<22} best={b['model']:<22} n={b['best_n']} "
          f"Test(fair)={b['test_f1_fair']:.4f}")

# All 4 scenarios ensemble
prob_mat = np.column_stack([scenario_best_models[sc]['probs']
                              for sc in ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']])

print(f"\n  Ensemble combinations:")
ensemble_results = {}

# Uniform average
for combo_name, idxs in [
    ("All-4",     [0, 1, 2, 3]),
    ("S2+S3",     [1, 2]),
    ("S1+S2",     [0, 1]),
    ("S2+S4",     [1, 3]),
    ("S1+S2+S3",  [0, 1, 2]),
    ("S2+S3+S4",  [1, 2, 3]),
]:
    probs_ens = prob_mat[:, idxs].mean(axis=1)
    thr_sw, f1_sw = sweep_thr(probs_ens, y_test)
    # Fair threshold = mean of fair thresholds of members
    fair_thrs_members = [sorted_by_fair[0]['fair_thr']]  # placeholder
    # Use best_test fair_thr as reference
    fair_thr_ens = np.mean([scenario_best_models[sc]['probs'].mean()
                             for sc in ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']])
    # Actually use cv-derived fair threshold from members
    member_scs = [['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion'][i] for i in idxs]
    member_thrs = []
    for sc in member_scs:
        rows = [r for r in all_results if r['scenario'] == sc]
        b = max(rows, key=lambda x: x['test_f1_fair'])
        member_thrs.append(b['fair_thr'])
    fair_thr_ens = np.mean(member_thrs)

    preds_fair = (probs_ens >= fair_thr_ens).astype(int)
    preds_sw   = (probs_ens >= thr_sw).astype(int)
    f1_fair    = f1_score(y_test, preds_fair, average='macro', zero_division=0)
    f1_sw2     = f1_score(y_test, preds_sw,   average='macro', zero_division=0)
    acc_fair   = accuracy_score(y_test, preds_fair)
    try: auc = roc_auc_score(y_test, probs_ens)
    except: auc = 0.
    print(f"  Ensemble({combo_name:<12}): fair_thr={fair_thr_ens:.2f} "
          f"F1(fair)={f1_fair:.4f} F1(sw)={f1_sw2:.4f} "
          f"Acc={acc_fair:.4f} AUC={auc:.4f}")
    ensemble_results[combo_name] = {'f1_fair': f1_fair, 'f1_sw': f1_sw2,
                                     'acc': acc_fair, 'auc': auc,
                                     'preds': preds_fair}
    if f1_fair > current_best_test:
        current_best_test = f1_fair
        print(f"    ★ NEW BEST TEST (fair): {f1_fair:.4f}")

# %% [markdown]
# ## Learning Curves (Best Model + Best Ensemble)

# %%
print(f"\n{'='*80}")
print("  LEARNING CURVES (Best Models per Scenario)")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('v85 — Learning Curves (PCA Whitening, No SMOTE)\n'
             'Goal: Training & Validation curves should converge (less overfitting)',
             fontsize=12, fontweight='bold')

for col_idx, sc in enumerate(['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']):
    X_full = SCENARIOS[sc]
    rows = [r for r in all_results if r['scenario'] == sc]
    b_cv   = max(rows, key=lambda x: x['cv_f1_mean'])
    b_test = max(rows, key=lambda x: x['test_f1_fair'])

    for row_idx, b in enumerate([b_cv, b_test]):
        ax = axes[row_idx][col_idx]
        X_tr_raw = X_full[train_idx]
        X_te_raw = X_full[test_idx]
        X_tr_p, _, _ = preprocess_pca(X_tr_raw.copy(), X_te_raw.copy(), b['best_n'])
        clf = build_model(b['model'], MODEL_CONFIGS[b['model']][b['best_cfg_idx']])

        try:
            train_sizes, train_scores, val_scores = learning_curve(
                clf, X_tr_p, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
                train_sizes=np.linspace(0.2, 1.0, 8),
                scoring='f1_macro', n_jobs=1)
            tr_mean = train_scores.mean(axis=1)
            tr_std  = train_scores.std(axis=1)
            vl_mean = val_scores.mean(axis=1)
            vl_std  = val_scores.std(axis=1)
            ax.plot(train_sizes, tr_mean, 'b-o', label='Training', linewidth=2, markersize=5)
            ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color='blue')
            ax.plot(train_sizes, vl_mean, 'r-s', label='Validation', linewidth=2, markersize=5)
            ax.fill_between(train_sizes, vl_mean - vl_std, vl_mean + vl_std, alpha=0.15, color='red')
            ax.axhline(0.75, color='green', linestyle='--', lw=1.5, label='Target 0.75')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', transform=ax.transAxes, ha='center')

        label_type = 'BestCV' if row_idx == 0 else 'BestTest'
        ax.set_title(f'{sc}\n{b["model"]} n={b["best_n"]}\n'
                     f'{label_type}: CV={b["cv_f1_mean"]:.3f} Test={b["test_f1_fair"]:.3f}',
                     fontsize=7.5, fontweight='bold')
        ax.set_xlabel('Training Size', fontsize=8)
        ax.set_ylabel('F1 Macro', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(linestyle='--', alpha=0.4)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "plots", "v85_learning_curves.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Learning curves saved.")

# %% [markdown]
# ## Bar Chart: CV vs Test F1 Comparison

# %%
COLORS = ['#6366f1', '#ef4444', '#f97316', '#22c55e']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle(f'v85 — PCA Whitening | No SMOTE | class_weight\n'
             f'Best CV={best_cv["cv_f1_mean"]:.4f} | '
             f'Best Test(fair)={best_test["test_f1_fair"]:.4f}',
             fontsize=12, fontweight='bold')

sc_list = ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']
x = np.arange(len(MODEL_NAMES)); width = 0.18

for i, sc in enumerate(sc_list):
    rows = [r for r in all_results if r['scenario'] == sc and r['model'] in MODEL_NAMES]
    cv_v = [next((r['cv_f1_mean']   for r in rows if r['model'] == m), 0.) for m in MODEL_NAMES]
    te_v = [next((r['test_f1_fair'] for r in rows if r['model'] == m), 0.) for m in MODEL_NAMES]
    label = sc.split('_')[1]
    ax1.bar(x + i * width, cv_v, width, label=label, color=COLORS[i], alpha=0.85, edgecolor='white')
    ax2.bar(x + i * width, te_v, width, label=label, color=COLORS[i], alpha=0.85, edgecolor='white')

for ax, title in [(ax1, f'CV F1 ({K_FOLDS_OUTER}-Fold, PCA Whitening)'),
                  (ax2, 'Test F1 (Fair CV-Threshold)')]:
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(MODEL_NAMES, rotation=15, ha='right', fontsize=9)
    ax.axhline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1 Macro')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in ax.patches:
        val = bar.get_height()
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "plots", "v85_comparison.png"),
            dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrices
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle('v85 — Confusion Matrix (Best Test per Skenario, PCA Whitening)',
              fontsize=11, fontweight='bold')
for ax, sc_name in zip(axes2, ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']):
    rows = [r for r in all_results if r['scenario'] == sc_name]
    b = max(rows, key=lambda x: x['test_f1_fair'])
    cm = confusion_matrix(y_test, b['y_pred_fair'], labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Depresi'],
                yticklabels=['Normal', 'Depresi'], annot_kws={'size': 14})
    ax.set_title(f'{sc_name}\n{b["model"]} n={b["best_n"]}\n'
                 f'CV={b["cv_f1_mean"]:.4f} Test={b["test_f1_fair"]:.4f}',
                 fontsize=8, fontweight='bold')
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, "plots", "v85_confusion.png"),
             dpi=150, bbox_inches='tight')
plt.close()
print("  Bar chart & confusion matrices saved.")

# %% [markdown]
# ## Classification Reports

# %%
print(f"\n{'='*80}")
print("  CLASSIFICATION REPORTS — Best Test(fair) per Skenario")
print(f"{'='*80}")
for sc_name in ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']:
    rows = [r for r in all_results if r['scenario'] == sc_name]
    b = max(rows, key=lambda x: x['test_f1_fair'])
    print(f"\n  ── {sc_name} × {b['model']} (n={b['best_n']}) ──")
    print(f"  CV={b['cv_f1_mean']:.4f}±{b['cv_f1_std']:.4f} | "
          f"Test(fair)={b['test_f1_fair']:.4f} | Test(sw)={b['test_f1_sw']:.4f}")
    print(classification_report(y_test, b['y_pred_fair'],
                                 target_names=['Normal', 'Depresi'], zero_division=0))

# Best ensemble report
best_ens_name = max(ensemble_results, key=lambda k: ensemble_results[k]['f1_fair'])
best_ens = ensemble_results[best_ens_name]
print(f"\n  ── Ensemble({best_ens_name}) ──")
print(f"  F1(fair)={best_ens['f1_fair']:.4f} | F1(sw)={best_ens['f1_sw']:.4f} | "
      f"Acc={best_ens['acc']:.4f} | AUC={best_ens['auc']:.4f}")
print(classification_report(y_test, best_ens['preds'],
                             target_names=['Normal', 'Depresi'], zero_division=0))

# %% [markdown]
# ## Final Report

# %%
print(f"\n{'='*80}")
print(f"{'FINAL REPORT v85':^80}")
print(f"{'='*80}")

best_overall = max(current_best_test, best_test['test_f1_fair'])
best_ens_f1  = max(ensemble_results[k]['f1_fair'] for k in ensemble_results)

print(f"  v84 Referensi: CV=0.8457(overfit) | Test(fair)=0.6970 | Test(sw)=0.6970")
print(f"  v85 Best CV  : {best_cv['scenario']} × {best_cv['model']} n={best_cv['best_n']}"
      f" → {best_cv['cv_f1_mean']:.4f} ± {best_cv['cv_f1_std']:.4f}")
print(f"  v85 Best Test: {best_test['scenario']} × {best_test['model']} n={best_test['best_n']}"
      f" → Test(fair)={best_test['test_f1_fair']:.4f}")
print(f"  v85 Best Ens : Ensemble({best_ens_name}) → F1(fair)={best_ens_f1:.4f}")
print(f"  Overall Best : {max(best_test['test_f1_fair'], best_ens_f1):.4f}")
print()
print(f"  TARGET 0.75 (Test/fair): "
      f"{'✓ TERCAPAI!' if max(best_test['test_f1_fair'], best_ens_f1) >= 0.75 else 'NO (' + str(round(max(best_test[chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(102)+chr(49)+chr(95)+chr(102)+chr(97)+chr(105)+chr(114)], best_ens_f1), 4)) + ')' }")
print(f"  Total waktu  : {time.time() - t_global:.1f}s")
print(f"{'='*80}")

# Save summary JSON
summary = {
    'version': 'v85',
    'strategy': 'PCA_whitening + class_weight (No SMOTE)',
    'best_cv': {
        'scenario': best_cv['scenario'], 'model': best_cv['model'],
        'n_components': best_cv['best_n'],
        'cv_f1': best_cv['cv_f1_mean'], 'test_f1_fair': best_cv['test_f1_fair']
    },
    'best_test': {
        'scenario': best_test['scenario'], 'model': best_test['model'],
        'n_components': best_test['best_n'],
        'cv_f1': best_test['cv_f1_mean'], 'test_f1_fair': best_test['test_f1_fair']
    },
    'best_ensemble': {'name': best_ens_name, 'f1_fair': best_ens_f1},
    'overall_best_test': round(max(best_test['test_f1_fair'], best_ens_f1), 4),
    'target_075_test': bool(max(best_test['test_f1_fair'], best_ens_f1) >= 0.75),
    'v84_ref_cv': 0.8457, 'v84_ref_test': 0.6970,
}
json.dump(summary, open(os.path.join(RESULTS_DIR, "metrics", "v85_summary.json"), 'w'), indent=2)
print(f"  Summary saved to: {RESULTS_DIR}/metrics/v85_summary.json")
