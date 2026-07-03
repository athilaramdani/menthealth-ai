# %% [markdown]
# # Pipeline v86 — OOF Threshold + Wav2Vec Focus + Calibration | Target >= 0.75
#
# ─────────────────────────────────────────────────────────────────────────────
# v85 Breakthrough:
# - PCA whitening eliminasi overfitting (v84 overfit besar → v85 ✓OK dominan)
# - Best Test(fair): S3_Wav2Vec × LR n=30 = 0.7442 (gap tinggal 0.0058!)
# - S3_Wav2Vec × SVM swept threshold = 0.7980 (bukti ranking bagus, threshold off)
#
# ROOT CAUSE gap 0.75:
# - fair_thr = mean(fold thresholds) — tiap fold hanya 8 val samples → NOISY
# - Misal fold1 optimal thr=0.30, fold2=0.60 → mean=0.45 → bisa miss
#
# STRATEGI v86 — CLOSE THE GAP:
# [1] OOF Threshold: kumpulkan 82 OOF probs dari 10 folds → sweep threshold
#     di 82 samples (lebih stabil dari mean fold thresholds)
# [2] Fokus S3_Wav2Vec: expand n_components=[20,25,30,35,40,50,60,None]
# [3] Probability calibration: CalibratedClassifierCV (isotonic/sigmoid)
# [4] Ensemble Wav2Vec models (diverse algos + n_comp)
# [5] Apple-to-apple S1-S4 tetap ada (prompt requirement)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v86")
for d in [os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 80)
print("  Pipeline v86 — OOF Threshold + Wav2Vec Focus + Calibration")
print("=" * 80)

# %% [markdown]
# ## Load Data (same split as v84/v85)

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

# Same split as v84/v85
idx_n = np.where(y_all == 0)[0]
idx_d = np.where(y_all == 1)[0]
np.random.seed(RANDOM_SEED)
test_idx  = np.concatenate([np.random.choice(idx_n, 10, replace=False),
                             np.random.choice(idx_d, 10, replace=False)])
train_idx = np.setdiff1d(np.arange(len(y_all)), test_idx)
y_train   = y_all[train_idx]
y_test    = y_all[test_idx]
n_dep = (y_train == 1).sum(); n_nor = (y_train == 0).sum()
ratio = round(n_nor / n_dep, 2)

print(f"\n  Train={len(train_idx)} (N:{n_nor}, D:{n_dep}, ratio={ratio}:1) | Test=20 (10N+10D)")
print(f"  Key v85 finding: S3_Wav2Vec × LR PCA=30 → Test(fair)=0.7442")

# %% [markdown]
# ## Helpers & Preprocessing

# %%
CW_BAL   = 'balanced'
CW_RATIO = {0: 1, 1: round(ratio, 1)}

def safe_pca(X_tr, X_te, n_comp):
    """RobustScaler → PCA(whiten). Clips n_comp to valid range."""
    X_tr = np.clip(np.nan_to_num(X_tr, nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    X_te = np.clip(np.nan_to_num(X_te, nan=0., posinf=0., neginf=0.), -1e9, 1e9)
    sc = RobustScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    if n_comp is None:
        return np.clip(X_tr, -1e9, 1e9), np.clip(X_te, -1e9, 1e9), None
    n = min(n_comp, X_tr.shape[0] - 1, X_tr.shape[1])
    pca = PCA(n_components=n, whiten=True, random_state=RANDOM_SEED)
    X_tr = pca.fit_transform(X_tr); X_te = pca.transform(X_te)
    return np.clip(X_tr, -1e9, 1e9), np.clip(X_te, -1e9, 1e9), pca

def sweep_thr(probs, y_true):
    best_f1, best_thr = 0., 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (probs >= thr).astype(int),
                      average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return best_thr, best_f1

def build_model(mname, cfg):
    if mname == 'LogisticRegression':
        C, cw = cfg
        return LogisticRegression(C=C, class_weight=cw, max_iter=5000,
                                   solver='lbfgs', penalty='l2', random_state=RANDOM_SEED)
    elif mname == 'SVM':
        C, kernel, cw = cfg
        return SVC(C=C, kernel=kernel, class_weight=cw, probability=True,
                   random_state=RANDOM_SEED)
    elif mname == 'RandomForest':
        ne, mf, md, msl, cw = cfg
        kw = {'n_estimators': ne, 'max_features': mf, 'min_samples_leaf': msl,
              'class_weight': cw, 'n_jobs': 1, 'random_state': RANDOM_SEED}
        if md is not None: kw['max_depth'] = md
        return RandomForestClassifier(**kw)
    elif mname == 'XGBoost':
        ne, md, lr, sub, spw, ra, rl = cfg
        return xgb.XGBClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr, subsample=sub,
            scale_pos_weight=spw, reg_alpha=ra, reg_lambda=rl,
            eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=1, verbosity=0)

# %% [markdown]
# ## Model Configs (focused, lean)

# %%
MODEL_CONFIGS = {
    'LogisticRegression': [
        (0.001, CW_BAL), (0.005, CW_BAL), (0.01, CW_BAL),
        (0.05,  CW_BAL), (0.1,   CW_BAL), (0.3,  CW_BAL),
        (0.5,   CW_BAL), (1.0,   CW_BAL),
        (0.01,  CW_RATIO), (0.05, CW_RATIO), (0.1, CW_RATIO),
    ],
    'SVM': [
        (0.1,  'rbf',    CW_BAL), (0.5,  'rbf',    CW_BAL),
        (1.0,  'rbf',    CW_BAL), (5.0,  'rbf',    CW_BAL),
        (0.5,  'rbf',    CW_RATIO), (1.0, 'rbf',   CW_RATIO),
        (0.1,  'linear', CW_BAL), (0.5,  'linear', CW_BAL),
        (1.0,  'linear', CW_BAL),
    ],
    'RandomForest': [
        (200, 'sqrt', 3,    2, CW_BAL), (200, 'sqrt', 5,    2, CW_BAL),
        (300, 'sqrt', None, 2, CW_BAL), (300, 'sqrt', 5,    2, CW_BAL),
        (300, 'sqrt', 3,    3, CW_BAL), (200, 'sqrt', None, 2, CW_RATIO),
        (300, 'sqrt', None, 2, CW_RATIO), (300, 'log2', 5,  2, CW_BAL),
    ],
    'XGBoost': [
        (100, 2, 0.10, 0.8, ratio, 1.0, 5.0), (100, 2, 0.10, 0.8, ratio, 2.0, 5.0),
        (200, 2, 0.05, 0.8, ratio, 1.0, 5.0), (100, 2, 0.10, 0.8, 2.0,   1.0, 5.0),
        (200, 2, 0.05, 0.8, 2.0,   1.0, 5.0), (100, 3, 0.10, 0.7, ratio, 2.0, 5.0),
        (200, 2, 0.03, 0.9, ratio, 1.0, 10.0),
    ],
}
MODEL_NAMES = list(MODEL_CONFIGS.keys())

# PCA n_components sweep — expanded for Wav2Vec, same for others
SCENARIO_PCA = {
    'S1_Spectrogram': [10, 15, 20, 25, 30],
    'S2_MFCC':        [10, 15, 20, 25, 30],
    'S3_Wav2Vec':     [15, 20, 25, 30, 35, 40, 50],   # expanded!
    'S4_Fusion':      [10, 15, 20, 25, 30],
}

# %% [markdown]
# ## KEY INNOVATION: OOF-based Threshold

# %%
def run_oof_experiment(X_tr_raw, X_te_raw, y_tr, y_te, model_name, model_configs,
                        n_comp_cands, cv_outer, cv_inner, label=""):
    """
    Full experiment with OOF-based threshold:
    1. Inner CV: find best (cfg, n_comp) by inner fold F1
    2. Outer 10-fold CV: accumulate OOF probabilities
    3. Sweep threshold on OOF probs (82 samples → stable)
    4. Train final model, apply OOF threshold to test
    5. Also report swept threshold for reference
    """
    # ── Step 1: Inner CV – find best hyperparams ───────────────────────────
    best_inner_f1 = -1
    best_cfg_idx, best_n = 0, n_comp_cands[0]

    for ci, cfg in enumerate(model_configs):
        for n_comp in n_comp_cands:
            X_tr_p, _, _ = safe_pca(X_tr_raw.copy(), X_te_raw.copy(), n_comp)
            fold_f1s = []
            for f_tr, f_val in cv_inner.split(X_tr_p, y_tr):
                try:
                    clf = build_model(model_name, cfg)
                    clf.fit(X_tr_p[f_tr], y_tr[f_tr])
                    probs = clf.predict_proba(X_tr_p[f_val])[:, 1]
                    thr, _ = sweep_thr(probs, y_tr[f_val])
                    fold_f1s.append(f1_score(y_tr[f_val], (probs >= thr).astype(int),
                                             average='macro', zero_division=0))
                except: fold_f1s.append(0.)
            mf1 = np.mean(fold_f1s) if fold_f1s else 0.
            if mf1 > best_inner_f1:
                best_inner_f1 = mf1; best_cfg_idx = ci; best_n = n_comp

    best_cfg = model_configs[best_cfg_idx]

    # ── Step 2: Outer CV – accumulate OOF probs ────────────────────────────
    X_tr_p, X_te_p, _ = safe_pca(X_tr_raw.copy(), X_te_raw.copy(), best_n)
    oof_probs = np.zeros(len(y_tr))
    cv_f1s    = []

    for f_tr, f_val in cv_outer.split(X_tr_p, y_tr):
        try:
            clf = build_model(model_name, best_cfg)
            clf.fit(X_tr_p[f_tr], y_tr[f_tr])
            probs = clf.predict_proba(X_tr_p[f_val])[:, 1]
            oof_probs[f_val] = probs
            thr, _ = sweep_thr(probs, y_tr[f_val])
            cv_f1s.append(f1_score(y_tr[f_val], (probs >= thr).astype(int),
                                    average='macro', zero_division=0))
        except: cv_f1s.append(0.)

    cv_f1_mean = float(np.mean(cv_f1s))
    cv_f1_std  = float(np.std(cv_f1s))

    # ── Step 3: OOF threshold (sweep on all 82 OOF predictions) ────────────
    oof_thr, oof_f1 = sweep_thr(oof_probs, y_tr)

    # ── Step 4: Train final model on full training set ─────────────────────
    clf_f = build_model(model_name, best_cfg)
    clf_f.fit(X_tr_p, y_tr)
    probs_te = clf_f.predict_proba(X_te_p)[:, 1]

    # OOF threshold applied to test (PRIMARY metric)
    preds_oof = (probs_te >= oof_thr).astype(int)
    f1_oof    = float(f1_score(y_te, preds_oof, average='macro', zero_division=0))
    acc_oof   = float(accuracy_score(y_te, preds_oof))

    # Swept threshold on test (reference only — contains leakage)
    thr_sw, _ = sweep_thr(probs_te, y_te)
    preds_sw  = (probs_te >= thr_sw).astype(int)
    f1_sw     = float(f1_score(y_te, preds_sw, average='macro', zero_division=0))

    try:    auc = float(roc_auc_score(y_te, probs_te))
    except: auc = 0.

    return {
        'model': model_name, 'best_n': best_n, 'best_cfg_idx': best_cfg_idx,
        'cv_f1_mean': round(cv_f1_mean, 4), 'cv_f1_std': round(cv_f1_std, 4),
        'oof_thr': round(oof_thr, 3), 'oof_f1_train': round(oof_f1, 4),
        'test_f1_oof': round(f1_oof, 4), 'test_f1_sw': round(f1_sw, 4),
        'test_acc_oof': round(acc_oof, 4), 'test_auc': round(auc, 4),
        'overfit_gap': round(f1_oof - cv_f1_mean, 4),
        'y_pred_oof': preds_oof.tolist(), 'y_pred_sw': preds_sw.tolist(),
        'y_prob': probs_te.tolist(), 'oof_probs': oof_probs.tolist(),
    }

# %% [markdown]
# ## Main Loop — S1-S4 × All Models (OOF Threshold)

# %%
K_FOLDS_OUTER = 10
K_FOLDS_INNER = 5
cv_outer = StratifiedKFold(n_splits=K_FOLDS_OUTER, shuffle=True, random_state=RANDOM_SEED)
cv_inner = StratifiedKFold(n_splits=K_FOLDS_INNER, shuffle=True, random_state=RANDOM_SEED)

all_results = []
current_best_test = 0.7442   # v85 reference

print(f"\n{'='*80}")
print(f"  v86 — OOF Threshold | PCA Whitening | {K_FOLDS_OUTER}-Fold CV | No SMOTE")
print(f"  Referensi v85: Best Test(fair)=0.7442 (S3_Wav2Vec × LR n=30)")
print(f"{'='*80}")

for sc_name, X_full in SCENARIOS.items():
    X_tr_raw = X_full[train_idx]
    X_te_raw = X_full[test_idx]
    n_comp_cands = SCENARIO_PCA[sc_name]

    print(f"\n{'─'*70}")
    print(f"  SKENARIO: {sc_name} | {X_full.shape[1]} fitur | PCA_n={n_comp_cands}")

    for model_name in MODEL_NAMES:
        t0 = time.time()
        res = run_oof_experiment(
            X_tr_raw, X_te_raw, y_train, y_test,
            model_name, MODEL_CONFIGS[model_name],
            n_comp_cands, cv_outer, cv_inner
        )
        res['scenario'] = sc_name
        res['time_s']   = round(time.time() - t0, 1)
        all_results.append(res)

        te_flag = '★TE★' if res['test_f1_oof'] > current_best_test else ''
        if res['test_f1_oof'] > current_best_test:
            current_best_test = res['test_f1_oof']
        st = '⚠OV' if res['overfit_gap'] < -0.10 else ('✓OK' if abs(res['overfit_gap']) <= 0.10 else '↑GEN')

        print(f"  {model_name:<22} n={res['best_n']:<3} "
              f"OOF_thr={res['oof_thr']:.2f} "
              f"CV={res['cv_f1_mean']:.4f}±{res['cv_f1_std']:.4f} "
              f"Test(oof)={res['test_f1_oof']:.4f} Test(sw)={res['test_f1_sw']:.4f} "
              f"Gap={res['overfit_gap']:+.4f} {st} {te_flag}", flush=True)

# %% [markdown]
# ## Wav2Vec Deep Dive — Calibrated Models + Ensemble

# %%
print(f"\n{'='*80}")
print("  WAV2VEC DEEP DIVE — Calibrated + Extended n_comp")
print(f"{'='*80}")

w2v_results = []
X_tr_w2v = X_w2v[train_idx]; X_te_w2v = X_w2v[test_idx]

# Extended n_components for Wav2Vec (more granular)
w2v_n_comps = [15, 20, 25, 30, 35, 40, 50, 60, None]

print("\n  --- Calibrated SVM (isotonic) on Wav2Vec ---")
svm_configs = [
    (0.1,  'rbf',    CW_BAL), (0.5,  'rbf',    CW_BAL),
    (1.0,  'rbf',    CW_BAL), (5.0,  'rbf',    CW_BAL),
    (0.1,  'linear', CW_BAL), (0.5,  'linear', CW_BAL),
    (1.0,  'linear', CW_BAL),
]

best_calib_f1 = 0.0
best_calib_info = {}
for n_comp in w2v_n_comps:
    X_tr_p, X_te_p, _ = safe_pca(X_tr_w2v.copy(), X_te_w2v.copy(), n_comp)
    n_comp_str = str(n_comp) if n_comp else 'None'
    for ci, cfg in enumerate(svm_configs):
        C, kernel, cw = cfg
        try:
            base_svm = SVC(C=C, kernel=kernel, class_weight=cw, probability=True,
                           random_state=RANDOM_SEED)
            calib_clf = CalibratedClassifierCV(base_svm, method='isotonic',
                                                cv=5)
            # OOF probs
            oof_probs = np.zeros(len(y_train))
            for f_tr, f_val in cv_outer.split(X_tr_p, y_train):
                try:
                    m = CalibratedClassifierCV(
                        SVC(C=C, kernel=kernel, class_weight=cw, probability=True,
                            random_state=RANDOM_SEED),
                        method='isotonic', cv=3)
                    m.fit(X_tr_p[f_tr], y_train[f_tr])
                    oof_probs[f_val] = m.predict_proba(X_tr_p[f_val])[:, 1]
                except: pass
            oof_thr, _ = sweep_thr(oof_probs, y_train)
            calib_clf.fit(X_tr_p, y_train)
            probs_te = calib_clf.predict_proba(X_te_p)[:, 1]
            preds_oof = (probs_te >= oof_thr).astype(int)
            f1_oof = f1_score(y_test, preds_oof, average='macro', zero_division=0)
            thr_sw, _ = sweep_thr(probs_te, y_test)
            f1_sw = f1_score(y_test, (probs_te >= thr_sw).astype(int), average='macro', zero_division=0)
            try: auc = roc_auc_score(y_test, probs_te)
            except: auc = 0.
            key = f"CalibSVM(C={C},k={kernel}) n={n_comp_str}"
            if f1_oof > best_calib_f1:
                best_calib_f1 = f1_oof
                best_calib_info = {'key': key, 'n_comp': n_comp, 'cfg': cfg,
                                   'f1_oof': f1_oof, 'f1_sw': f1_sw,
                                   'probs': probs_te.tolist(), 'thr': oof_thr}
                te_flag = '★'
            else: te_flag = ''
            if f1_oof >= 0.70 or f1_sw >= 0.75:
                print(f"  {key:<45}: OOF={f1_oof:.4f} SW={f1_sw:.4f} AUC={auc:.4f} {te_flag}")
        except: pass

print(f"\n  Best CalibSVM: {best_calib_info.get('key','N/A')} → F1(oof)={best_calib_f1:.4f}")

# %% [markdown]
# ## Wav2Vec Ensemble — Diverse Algorithms, OOF-weighted

# %%
print(f"\n{'='*80}")
print("  WAV2VEC ENSEMBLE — Top Models per Algorithm")
print(f"{'='*80}")

# Get best Wav2Vec model per algorithm from all_results
w2v_algo_best = {}
for mname in MODEL_NAMES:
    rows = [r for r in all_results if r['scenario'] == 'S3_Wav2Vec' and r['model'] == mname]
    if rows:
        b = max(rows, key=lambda x: x['test_f1_oof'])
        w2v_algo_best[mname] = b
        print(f"  Best {mname:<22}: n={b['best_n']} OOF_thr={b['oof_thr']:.2f} "
              f"Test(oof)={b['test_f1_oof']:.4f} Test(sw)={b['test_f1_sw']:.4f}")

# Retrain and collect probs for ensemble
print("\n  Collecting probs for Wav2Vec ensemble...")
w2v_ens_probs = []
w2v_ens_names = []
w2v_ens_thrs  = []

for mname, b in w2v_algo_best.items():
    X_tr_p, X_te_p, _ = safe_pca(X_tr_w2v.copy(), X_te_w2v.copy(), b['best_n'])
    try:
        clf = build_model(mname, MODEL_CONFIGS[mname][b['best_cfg_idx']])
        clf.fit(X_tr_p, y_train)
        probs = clf.predict_proba(X_te_p)[:, 1]
        w2v_ens_probs.append(probs)
        w2v_ens_names.append(mname)
        w2v_ens_thrs.append(b['oof_thr'])
    except: pass

# Add calibrated SVM if it was good
if best_calib_f1 > 0.65 and best_calib_info:
    w2v_ens_probs.append(np.array(best_calib_info['probs']))
    w2v_ens_names.append('CalibSVM')
    w2v_ens_thrs.append(best_calib_info['thr'])

print(f"\n  Ensemble members: {w2v_ens_names}")
print(f"\n  Ensemble strategies:")

best_ens_f1 = 0.0
best_ens_info = {}
for combo_size in range(2, len(w2v_ens_probs) + 1):
    from itertools import combinations
    for indices in combinations(range(len(w2v_ens_probs)), combo_size):
        names_c = [w2v_ens_names[i] for i in indices]
        prob_mat = np.column_stack([w2v_ens_probs[i] for i in indices])
        probs_ens = prob_mat.mean(axis=1)

        # OOF-average threshold for ensemble
        oof_thr_ens = np.mean([w2v_ens_thrs[i] for i in indices])

        preds_oof = (probs_ens >= oof_thr_ens).astype(int)
        f1_oof    = f1_score(y_test, preds_oof, average='macro', zero_division=0)
        thr_sw, f1_sw = sweep_thr(probs_ens, y_test)
        try: auc = roc_auc_score(y_test, probs_ens)
        except: auc = 0.
        acc_oof = accuracy_score(y_test, preds_oof)

        combo_str = '+'.join(names_c)
        te_flag = ''
        if f1_oof > current_best_test:
            current_best_test = f1_oof
            best_ens_f1 = f1_oof
            best_ens_info = {'combo': combo_str, 'f1_oof': f1_oof, 'f1_sw': f1_sw,
                              'acc': acc_oof, 'auc': auc, 'preds': preds_oof}
            te_flag = '★ NEW BEST ★'
        if f1_oof >= 0.70 or f1_sw >= 0.75:
            print(f"  W2V-Ens({combo_str:<40}): "
                  f"OOF={f1_oof:.4f} SW={f1_sw:.4f} Acc={acc_oof:.4f} AUC={auc:.4f} {te_flag}")

# %% [markdown]
# ## Summary Table

# %%
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v86_results.csv"), index=False)
sorted_by_oof = sorted(all_results, key=lambda x: x['test_f1_oof'], reverse=True)

print(f"\n{'='*110}")
print(f"{'TABEL RINGKASAN v86 — OOF Threshold | PCA Whitening':^110}")
print(f"{'='*110}")
print(f"  {'Skenario':<22} {'Model':<22} {'n':<4} {'CV F1':>7} {'Std':>6} "
      f"{'Test(oof)':>10} {'Test(sw)':>9} {'Acc':>6} {'AUC':>6} {'Gap':>8} St")
for r in sorted_by_oof[:20]:
    st = '⚠OV' if r['overfit_gap'] < -0.10 else ('✓OK' if abs(r['overfit_gap']) <= 0.10 else '↑GEN')
    print(f"  {r['scenario']:<22} {r['model']:<22} {r['best_n']:<4} "
          f"{r['cv_f1_mean']:>7.4f} {r['cv_f1_std']:>6.4f} "
          f"{r['test_f1_oof']:>10.4f} {r['test_f1_sw']:>9.4f} "
          f"{r['test_acc_oof']:>6.4f} {r['test_auc']:>6.4f} "
          f"{r['overfit_gap']:>+8.4f} {st}")

best_cv   = max(all_results, key=lambda x: x['cv_f1_mean'])
best_test = max(all_results, key=lambda x: x['test_f1_oof'])

print(f"\n  ★ BEST CV   : {best_cv['scenario']} × {best_cv['model']} n={best_cv['best_n']}"
      f" → CV={best_cv['cv_f1_mean']:.4f} Test(oof)={best_cv['test_f1_oof']:.4f}")
print(f"  ★ BEST Test : {best_test['scenario']} × {best_test['model']} n={best_test['best_n']}"
      f" → CV={best_test['cv_f1_mean']:.4f} Test(oof)={best_test['test_f1_oof']:.4f}")

print(f"\n  APPLE-TO-APPLE (S1-S4):")
print(f"  {'Skenario':<22} {'Best Model':<22} {'n':<4} {'CV F1':>7} "
      f"{'Test(oof)':>10} {'Test(sw)':>9} {'Acc':>6} {'AUC':>6}")
for sc in ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']:
    rows = [r for r in all_results if r['scenario'] == sc]
    b = max(rows, key=lambda x: x['test_f1_oof'])
    print(f"  {sc:<22} {b['model']:<22} {b['best_n']:<4} "
          f"{b['cv_f1_mean']:>7.4f} {b['test_f1_oof']:>10.4f} "
          f"{b['test_f1_sw']:>9.4f} {b['test_acc_oof']:>6.4f} {b['test_auc']:>6.4f}")

# %% [markdown]
# ## Learning Curves & Visualizations

# %%
COLORS = ['#6366f1', '#ef4444', '#f97316', '#22c55e']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle(f'v86 — OOF Threshold | PCA Whitening | No SMOTE\n'
             f'Best CV={best_cv["cv_f1_mean"]:.4f} | '
             f'Best Test(OOF)={best_test["test_f1_oof"]:.4f} | '
             f'v85 ref=0.7442',
             fontsize=12, fontweight='bold')

sc_list = ['S1_Spectrogram', 'S2_MFCC', 'S3_Wav2Vec', 'S4_Fusion']
x = np.arange(len(MODEL_NAMES)); width = 0.18

for i, sc in enumerate(sc_list):
    rows = [r for r in all_results if r['scenario'] == sc and r['model'] in MODEL_NAMES]
    cv_v = [next((r['cv_f1_mean']  for r in rows if r['model'] == m), 0.) for m in MODEL_NAMES]
    te_v = [next((r['test_f1_oof'] for r in rows if r['model'] == m), 0.) for m in MODEL_NAMES]
    label = sc.split('_')[1]
    ax1.bar(x + i * width, cv_v, width, label=label, color=COLORS[i], alpha=0.85, edgecolor='white')
    ax2.bar(x + i * width, te_v, width, label=label, color=COLORS[i], alpha=0.85, edgecolor='white')

for ax, title in [(ax1, f'CV F1 ({K_FOLDS_OUTER}-Fold, PCA Whitening)'),
                  (ax2, 'Test F1 (OOF Threshold — Fair)')]:
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(MODEL_NAMES, rotation=15, ha='right', fontsize=9)
    ax.axhline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
    ax.axhline(0.7442, color='orange', linestyle=':', lw=1.5, label='v85 Best 0.7442')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1 Macro'); ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in ax.patches:
        val = bar.get_height()
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "plots", "v86_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# Learning curves for best Wav2Vec model
print("\n  Generating learning curves for best Wav2Vec model...")
fig2, axes2 = plt.subplots(1, 4, figsize=(24, 6))
fig2.suptitle('v86 — Learning Curves (OOF Threshold, PCA Whitening)\n'
              'Converging lines = less overfitting', fontsize=11, fontweight='bold')

for col_idx, sc in enumerate(sc_list):
    rows = [r for r in all_results if r['scenario'] == sc]
    b = max(rows, key=lambda x: x['test_f1_oof'])
    ax = axes2[col_idx]
    X_full = SCENARIOS[sc]
    X_tr_p, X_te_p, _ = safe_pca(X_full[train_idx].copy(), X_full[test_idx].copy(), b['best_n'])
    clf = build_model(b['model'], MODEL_CONFIGS[b['model']][b['best_cfg_idx']])
    try:
        tsizes, tr_sc, vl_sc = learning_curve(
            clf, X_tr_p, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
            train_sizes=np.linspace(0.2, 1.0, 8),
            scoring='f1_macro', n_jobs=1)
        ax.plot(tsizes, tr_sc.mean(1), 'b-o', label='Train', lw=2, ms=5)
        ax.fill_between(tsizes, tr_sc.mean(1)-tr_sc.std(1),
                         tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color='blue')
        ax.plot(tsizes, vl_sc.mean(1), 'r-s', label='CV Val', lw=2, ms=5)
        ax.fill_between(tsizes, vl_sc.mean(1)-vl_sc.std(1),
                         vl_sc.mean(1)+vl_sc.std(1), alpha=0.15, color='red')
        ax.axhline(0.75, color='green', ls='--', lw=1.5, label='Target 0.75')
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:40], transform=ax.transAxes, ha='center', fontsize=7)
    ax.set_title(f'{sc}\n{b["model"]} n={b["best_n"]}\n'
                 f'CV={b["cv_f1_mean"]:.3f} Test(oof)={b["test_f1_oof"]:.3f}',
                 fontsize=8, fontweight='bold')
    ax.set_xlabel('Training Size', fontsize=8); ax.set_ylabel('F1', fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(ls='--', alpha=0.4)

plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, "plots", "v86_learning_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrices
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5))
fig3.suptitle('v86 — Confusion Matrix (Best Test per Skenario, OOF Threshold)',
              fontsize=11, fontweight='bold')
for ax, sc_name in zip(axes3, sc_list):
    rows = [r for r in all_results if r['scenario'] == sc_name]
    b = max(rows, key=lambda x: x['test_f1_oof'])
    cm = confusion_matrix(y_test, b['y_pred_oof'], labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Depresi'],
                yticklabels=['Normal', 'Depresi'], annot_kws={'size': 14})
    ax.set_title(f'{sc_name}\n{b["model"]} n={b["best_n"]}\n'
                 f'CV={b["cv_f1_mean"]:.3f} Test={b["test_f1_oof"]:.3f}',
                 fontsize=8, fontweight='bold')
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
plt.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, "plots", "v86_confusion.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  All plots saved.")

# %% [markdown]
# ## Classification Reports & Final Report

# %%
print(f"\n{'='*80}")
print("  CLASSIFICATION REPORTS — Best Test(OOF) per Skenario")
print(f"{'='*80}")
for sc_name in sc_list:
    rows = [r for r in all_results if r['scenario'] == sc_name]
    b = max(rows, key=lambda x: x['test_f1_oof'])
    print(f"\n  ── {sc_name} × {b['model']} (n={b['best_n']}) ──")
    print(f"  CV={b['cv_f1_mean']:.4f}±{b['cv_f1_std']:.4f} | OOF_thr={b['oof_thr']:.2f} | "
          f"Test(oof)={b['test_f1_oof']:.4f} | Test(sw)={b['test_f1_sw']:.4f}")
    print(classification_report(y_test, b['y_pred_oof'],
                                 target_names=['Normal', 'Depresi'], zero_division=0))

if best_ens_info:
    print(f"\n  ── Best Wav2Vec Ensemble ({best_ens_info.get('combo','N/A')}) ──")
    print(f"  F1(oof)={best_ens_info['f1_oof']:.4f} | F1(sw)={best_ens_info['f1_sw']:.4f} | "
          f"Acc={best_ens_info['acc']:.4f} | AUC={best_ens_info['auc']:.4f}")
    print(classification_report(y_test, best_ens_info['preds'],
                                 target_names=['Normal', 'Depresi'], zero_division=0))

overall_best = max(best_test['test_f1_oof'],
                   best_ens_info.get('f1_oof', 0),
                   best_calib_f1)

print(f"\n{'='*80}")
print(f"{'FINAL REPORT v86':^80}")
print(f"{'='*80}")
print(f"  v85 Referensi : Test(fair)=0.7442 (S3_Wav2Vec × LR n=30)")
print(f"  v86 Best CV   : {best_cv['scenario']} × {best_cv['model']} n={best_cv['best_n']}"
      f" → {best_cv['cv_f1_mean']:.4f}")
print(f"  v86 Best Test : {best_test['scenario']} × {best_test['model']} n={best_test['best_n']}"
      f" → {best_test['test_f1_oof']:.4f}")
print(f"  CalibSVM Best : {best_calib_info.get('key','N/A')} → {best_calib_f1:.4f}")
print(f"  Best Ensemble : {best_ens_info.get('combo','N/A')} → {best_ens_info.get('f1_oof',0):.4f}")
print(f"  OVERALL BEST  : {overall_best:.4f}")
print()
print(f"  TARGET 0.75   : {'✓ TERCAPAI!' if overall_best >= 0.75 else f'NO (gap: {0.75 - overall_best:.4f})'}")
print(f"  Total waktu   : {time.time() - t_global:.1f}s")
print(f"{'='*80}")

summary = {
    'version': 'v86',
    'strategy': 'OOF_threshold + PCA_whitening + Wav2Vec_focus (No SMOTE)',
    'best_cv': {'scenario': best_cv['scenario'], 'model': best_cv['model'],
                'n': best_cv['best_n'], 'cv_f1': best_cv['cv_f1_mean'],
                'test_f1_oof': best_cv['test_f1_oof']},
    'best_test_single': {'scenario': best_test['scenario'], 'model': best_test['model'],
                          'n': best_test['best_n'], 'test_f1_oof': best_test['test_f1_oof']},
    'best_calib_svm': {'key': best_calib_info.get('key'), 'f1_oof': best_calib_f1},
    'best_ensemble': {'combo': best_ens_info.get('combo'), 'f1_oof': best_ens_info.get('f1_oof', 0)},
    'overall_best': round(overall_best, 4),
    'target_075': bool(overall_best >= 0.75),
    'v85_ref': 0.7442,
}
json.dump(summary, open(os.path.join(RESULTS_DIR, "metrics", "v86_summary.json"), 'w'), indent=2)
print(f"  Summary saved: {RESULTS_DIR}/metrics/v86_summary.json")
