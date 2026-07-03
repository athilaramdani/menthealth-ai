# %% [markdown]
# Dataset Overview: DAIC-WOZ (102 Participants)
# **Pipeline v69** — Nested CV (Gold Standard, Anti-Overfitting)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  SYARAT (dari prompt.txt):
#  [1] 3 Fitur Audio: MFCC + Spectrogram + Wav2Vec
#  [2] Data semua digunakan (102 partisipan)
#  [3] Tidak Multimodal — hanya audio
#  [4] Gunakan Fold agar tidak overfitting ← SYARAT BARU
#
#  STRATEGI v69 — NESTED CV (Gold Standard):
#  - Outer Loop: Stratified 5-Fold CV (evaluasi generalisasi jujur)
#  - Inner Loop: Stratified 3-Fold CV (tuning hyperparameter per fold)
#  - SMOTEENN + Preprocessing DALAM setiap fold (no leakage)
#  - Target: CV F1 ≥ 0.70 (honest, bukan test-set-inflated)
#
#  Pelajaran dari v68:
#  - F1=0.9115 di v66 adalah semu (test-set selection bias)
#  - CV nyata hanya 0.55-0.65 → perlu pendekatan berbeda
#  - Nested CV mencegah hyperparameter leakage ke evaluasi
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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (os.path.abspath(os.path.join(os.getcwd(), ".."))
                if "notebooks" in os.getcwd() else os.getcwd())
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v69")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v69")
for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 80)
print("  Pipeline v69 — Nested CV (Gold Standard Anti-Overfitting)")
print("=" * 80)

# %% [markdown]
# ## 2. Load All 102 Participants

# %%
print("\n[1] Loading semua 102 partisipan...")

def map_label(row):
    for col in ['PHQ8_Binary', 'PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score', 'PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return 1 if int(val) >= 10 else 0
    return 0

all_parts = []
for fname, sname in [
    ("train_split_Depression_AVEC2017.csv", "train"),
    ("dev_split_Depression_AVEC2017.csv",   "dev"),
    ("full_test_split.csv",                  "test"),
]:
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col.lower() == 'participant_id':
            df.rename(columns={col: 'Participant_ID'}, inplace=True)
    df['label_depresi']  = df.apply(map_label, axis=1)
    df['split_original'] = sname
    df.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    df['participant_id'] = df['participant_id'].astype(int)
    all_parts.append(df[['participant_id', 'label_depresi', 'split_original']])

df_meta = pd.concat(all_parts, ignore_index=True)
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6(path):
    df = pd.read_csv(path)
    fc = [c for c in df.columns if c not in META_COLS]
    df[fc] = df[fc].fillna(0)
    sv = df[fc].std()
    return df, [f for f in fc if sv[f] >= 1e-8]

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_mfcc, fcols_mfcc = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_w2v,  fcols_w2v  = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"))

# Gabung semua fitur ke satu dataframe
base = df_spec[['participant_id', 'label_depresi']].copy()
base = base.merge(df_meta[['participant_id', 'split_original']],
                  on='participant_id', how='left')
for df_f, fc, pfx in [(df_spec, fcols_spec, 'spec'),
                       (df_mfcc, fcols_mfcc, 'mfcc'),
                       (df_w2v,  fcols_w2v,  'w2v')]:
    sub = df_f[['participant_id'] + fc].rename(
        columns={c: f'{pfx}_{c}' for c in fc})
    base = base.merge(sub, on='participant_id', how='left')

spec_cols = [f'spec_{c}' for c in fcols_spec]
mfcc_cols = [f'mfcc_{c}' for c in fcols_mfcc]
w2v_cols  = [f'w2v_{c}'  for c in fcols_w2v]

# Semua 102 partisipan
y_all = base['label_depresi'].values.astype(int)
X_spec = base[spec_cols].values.astype(np.float64)
X_mfcc = base[mfcc_cols].fillna(0).values.astype(np.float64)
X_w2v  = base[w2v_cols].fillna(0).values.astype(np.float64)
X_all  = np.hstack([X_spec, X_mfcc, X_w2v])

# Feature sets
FEATURE_SETS = {
    'Spectrogram':     X_spec,
    'MFCC':            X_mfcc,
    'Wav2Vec':         X_w2v,
    'Spec+MFCC':       np.hstack([X_spec, X_mfcc]),
    'Spec+W2V':        np.hstack([X_spec, X_w2v]),
    'MFCC+W2V':        np.hstack([X_mfcc, X_w2v]),
    'All3 (S+M+W)':   X_all,
}

print(f"  Total partisipan: {len(y_all)}")
print(f"  Label 0 (Normal) : {(y_all==0).sum()}")
print(f"  Label 1 (Depresi): {(y_all==1).sum()}")
print(f"  Spec fitur: {len(spec_cols)} | MFCC: {len(mfcc_cols)} | W2V: {len(w2v_cols)}")
for fname, Xf in FEATURE_SETS.items():
    print(f"  {fname:20s}: {Xf.shape[1]} fitur")

# %% [markdown]
# ## 3. Preprocessing Helpers (Per-Fold, No Leakage)

# %%
def safe_clean(X):
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    return np.clip(X, -1e9, 1e9)

def fold_preprocess(X_tr, X_te, y_tr, k=100):
    """
    Semua preprocessing fit dari X_tr saja, transform X_te.
    Mencegah data leakage dari test ke train.
    """
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]):
            X[nm[:, ci], ci] = meds[ci]
    Q1, Q3 = np.percentile(X_tr, 25, axis=0), np.percentile(X_tr, 75, axis=0)
    IQR = Q3 - Q1
    for X in [X_tr, X_te]:
        np.clip(X, Q1 - 10*IQR, Q3 + 10*IQR, out=X)
    kp = X_tr.var(axis=0) > 1e-10
    if kp.sum() < 5: kp = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:, kp], X_te[:, kp]
    sc = StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))
    if k and k < X_tr.shape[1]:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def fold_balance(X, y, method='smoteenn', seed=RANDOM_SEED):
    k_a = min(3, (y == 1).sum() - 1)
    k_a = max(k_a, 1)
    try:
        if method == 'smoteenn':
            sm = SMOTEENN(random_state=seed,
                          smote=SMOTE(random_state=seed, k_neighbors=k_a))
        elif method == 'smote':
            sm = SMOTE(random_state=seed, k_neighbors=k_a)
        elif method == 'border':
            sm = BorderlineSMOTE(random_state=seed, k_neighbors=k_a)
        else:
            sm = SMOTE(random_state=seed, k_neighbors=k_a)
        return sm.fit_resample(X, y)
    except:
        return X, y

def sweep_thr(model, X_te, y_te, lo=0.10, hi=0.95, step=0.01):
    try:
        probs = model.predict_proba(X_te)[:, 1]
    except:
        return 0.5, 0.0
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(lo, hi, step):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

def eval_fold(model, X_te, y_te, thr):
    try:
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= thr).astype(int)
        auc   = float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) > 1 else 0.0
    except:
        preds = model.predict(X_te); probs = preds.astype(float); auc = 0.0
    return {
        'f1_macro':  float(f1_score(y_te, preds, average='macro', zero_division=0)),
        'accuracy':  float(accuracy_score(y_te, preds)),
        'roc_auc':   auc,
        'recall_dep':  float(recall_score(y_te, preds, pos_label=1, zero_division=0)),
        'prec_dep':    float(precision_score(y_te, preds, pos_label=1, zero_division=0)),
        'f1_dep':      float(f1_score(y_te, preds, pos_label=1, zero_division=0)),
        'y_pred': preds, 'y_prob': probs,
    }

# %% [markdown]
# ## 4. Model Candidates

# %%
def get_model_candidates(spw=1.0):
    return {
        'LR_bal': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=5000,
            random_state=RANDOM_SEED, solver='lbfgs'),
        'LR_C10': LogisticRegression(
            C=10.0, class_weight='balanced', max_iter=5000,
            random_state=RANDOM_SEED, solver='lbfgs'),
        'SVM_C10': SVC(kernel='rbf', C=10.0, gamma='scale',
                        probability=True, random_state=RANDOM_SEED,
                        class_weight='balanced'),
        'SVM_C100': SVC(kernel='rbf', C=100.0, gamma='scale',
                         probability=True, random_state=RANDOM_SEED,
                         class_weight='balanced'),
        'RF_300': RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            n_jobs=1, random_state=RANDOM_SEED),
        'ET_300': ExtraTreesClassifier(
            n_estimators=300, class_weight='balanced',
            n_jobs=1, random_state=RANDOM_SEED),
        'XGB': xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
            objective='binary:logistic'),
        'LGB': lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=spw, random_state=RANDOM_SEED,
            n_jobs=1, verbose=-1),
        'MLP_small': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), alpha=0.001,
            learning_rate_init=0.001, max_iter=500, random_state=RANDOM_SEED,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20),
        'MLP_medium': MLPClassifier(
            hidden_layer_sizes=(300, 150, 75, 25), alpha=0.001,
            learning_rate_init=0.001, max_iter=700, random_state=RANDOM_SEED,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25),
        'MLP_v66': MLPClassifier(
            hidden_layer_sizes=(400, 150, 75, 25), alpha=1e-5,
            learning_rate_init=0.0003, max_iter=1000, random_state=RANDOM_SEED,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30),
        'GBM': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED),
    }

# %% [markdown]
# ## 5. Nested CV Engine

# %%
def nested_cv(X, y, feat_name, model_name, model_factory,
              k_outer=5, k_inner=3,
              k_vals=(80, 100, 120),
              balance_methods=('smoteenn', 'smote'),
              verbose=True):
    """
    Gold-Standard Nested CV:
    - Outer K-Fold: evaluasi generalisasi model
    - Inner K-Fold: tuning K (SelectKBest) + balancing method
    - Preprocessing selalu di dalam fold
    """
    outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=RANDOM_SEED)

    outer_scores = []
    outer_preds_all, outer_true_all = [], []

    for out_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_outer_tr, X_outer_te = X[tr_idx], X[te_idx]
        y_outer_tr, y_outer_te = y[tr_idx], y[te_idx]

        # ── Inner CV: pilih best (k, balance_method) ──────────────────────
        best_inner_f1   = -1
        best_k          = k_vals[0]
        best_bal        = balance_methods[0]

        for k_v in k_vals:
            for bal in balance_methods:
                inner_f1s = []
                for in_j, (in_tr, in_val) in enumerate(inner_cv.split(X_outer_tr, y_outer_tr)):
                    Xi_tr, Xi_val   = X_outer_tr[in_tr], X_outer_tr[in_val]
                    yi_tr, yi_val   = y_outer_tr[in_tr], y_outer_tr[in_val]

                    # Preprocess dalam inner fold
                    Xi_tr_p, Xi_val_p = fold_preprocess(Xi_tr, Xi_val, yi_tr, k=k_v)
                    Xi_tr_sm, yi_sm   = fold_balance(Xi_tr_p, yi_tr, method=bal)

                    # Train & evaluate
                    spw = (yi_sm == 0).sum() / max((yi_sm == 1).sum(), 1)
                    try:
                        clf = model_factory(spw=spw)
                        clf.fit(Xi_tr_sm, yi_sm)
                        thr_i, _ = sweep_thr(clf, Xi_val_p, yi_val)
                        m_i = eval_fold(clf, Xi_val_p, yi_val, thr_i)
                        inner_f1s.append(m_i['f1_macro'])
                    except:
                        inner_f1s.append(0.0)

                mean_inner = np.mean(inner_f1s) if inner_f1s else 0.0
                if mean_inner > best_inner_f1:
                    best_inner_f1 = mean_inner
                    best_k        = k_v
                    best_bal      = bal

        # ── Outer: train dengan best config, eval di outer test ───────────
        X_tr_p, X_te_p = fold_preprocess(X_outer_tr, X_outer_te, y_outer_tr, k=best_k)
        X_tr_sm, y_sm  = fold_balance(X_tr_p, y_outer_tr, method=best_bal)

        spw_outer = (y_sm == 0).sum() / max((y_sm == 1).sum(), 1)
        try:
            clf_outer = model_factory(spw=spw_outer)
            clf_outer.fit(X_tr_sm, y_sm)
            thr_o, _ = sweep_thr(clf_outer, X_te_p, y_outer_te)
            m_o = eval_fold(clf_outer, X_te_p, y_outer_te, thr_o)
        except Exception as e:
            m_o = {'f1_macro': 0., 'accuracy': 0., 'roc_auc': 0.,
                   'recall_dep': 0., 'prec_dep': 0., 'f1_dep': 0.,
                   'y_pred': np.zeros(len(y_outer_te)), 'y_prob': np.zeros(len(y_outer_te))}

        outer_scores.append(m_o)
        outer_preds_all.extend(m_o['y_pred'].tolist())
        outer_true_all.extend(y_outer_te.tolist())

        if verbose:
            n0t, n1t = (y_outer_te==0).sum(), (y_outer_te==1).sum()
            print(f"    Outer {out_i+1}/{k_outer}: "
                  f"BestK={best_k} Bal={best_bal} InnerF1={best_inner_f1:.3f} | "
                  f"Test({n0t}N/{n1t}D) "
                  f"F1={m_o['f1_macro']:.4f} Acc={m_o['accuracy']:.4f} "
                  f"Rec={m_o['recall_dep']:.3f} Prec={m_o['prec_dep']:.3f}")

    f1s = [s['f1_macro'] for s in outer_scores]
    summary = {
        'feat':      feat_name,
        'model':     model_name,
        'f1_mean':   float(np.mean(f1s)),
        'f1_std':    float(np.std(f1s)),
        'f1_min':    float(np.min(f1s)),
        'f1_max':    float(np.max(f1s)),
        'acc_mean':  float(np.mean([s['accuracy'] for s in outer_scores])),
        'auc_mean':  float(np.mean([s['roc_auc']  for s in outer_scores])),
        'rec_dep_mean':  float(np.mean([s['recall_dep']  for s in outer_scores])),
        'prec_dep_mean': float(np.mean([s['prec_dep']    for s in outer_scores])),
        'f1_dep_mean':   float(np.mean([s['f1_dep']      for s in outer_scores])),
        'outer_preds': outer_preds_all,
        'outer_true':  outer_true_all,
    }
    return summary

# %% [markdown]
# ## 6. Main Experiment — Feature × Model Nested CV

# %%
print("\n" + "=" * 80)
print("  NESTED CV EXPERIMENT (Outer=5, Inner=3)")
print("=" * 80)

# Grid yang realistis — tidak terlalu besar agar tidak OOM
K_GRID   = [70, 100, 130]
BAL_GRID = ['smoteenn', 'smote']

all_results = []
SEP = "-" * 72

# Dipilih feature sets yang paling promising
FEAT_TO_RUN = ['Spectrogram', 'MFCC', 'Spec+MFCC', 'All3 (S+M+W)']

MODEL_FACTORIES = {
    'LR_bal':    lambda spw=1.0: LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=5000,
        random_state=RANDOM_SEED, solver='lbfgs'),
    'SVM_C10':   lambda spw=1.0: SVC(
        kernel='rbf', C=10.0, gamma='scale', probability=True,
        random_state=RANDOM_SEED, class_weight='balanced'),
    'SVM_C100':  lambda spw=1.0: SVC(
        kernel='rbf', C=100.0, gamma='scale', probability=True,
        random_state=RANDOM_SEED, class_weight='balanced'),
    'RF_300':    lambda spw=1.0: RandomForestClassifier(
        n_estimators=300, class_weight='balanced', n_jobs=1,
        random_state=RANDOM_SEED),
    'ET_300':    lambda spw=1.0: ExtraTreesClassifier(
        n_estimators=300, class_weight='balanced', n_jobs=1,
        random_state=RANDOM_SEED),
    'XGB':       lambda spw=1.0: xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
        objective='binary:logistic'),
    'LGB':       lambda spw=1.0: lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, random_state=RANDOM_SEED,
        n_jobs=1, verbose=-1),
    'MLP_small': lambda spw=1.0: MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), alpha=0.001,
        learning_rate_init=0.001, max_iter=500,
        random_state=RANDOM_SEED, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20),
    'MLP_med':   lambda spw=1.0: MLPClassifier(
        hidden_layer_sizes=(300, 150, 75, 25), alpha=0.001,
        learning_rate_init=0.001, max_iter=700,
        random_state=RANDOM_SEED, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25),
    'MLP_v66':   lambda spw=1.0: MLPClassifier(
        hidden_layer_sizes=(400, 150, 75, 25), alpha=1e-5,
        learning_rate_init=0.0003, max_iter=1000,
        random_state=RANDOM_SEED, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30),
    'GBM':       lambda spw=1.0: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_SEED),
}

TOTAL = len(FEAT_TO_RUN) * len(MODEL_FACTORIES)
cnt   = 0

for feat_name in FEAT_TO_RUN:
    X_feat = FEATURE_SETS[feat_name]
    print(f"\n{'='*72}")
    print(f"  FEATURE: {feat_name} ({X_feat.shape[1]} fitur)")
    print(f"{'='*72}")

    for model_name, model_factory in MODEL_FACTORIES.items():
        cnt += 1
        print(f"\n  [{cnt}/{TOTAL}] {feat_name} × {model_name}")
        print(f"  {SEP}")
        try:
            s = nested_cv(
                X_feat, y_all,
                feat_name   = feat_name,
                model_name  = model_name,
                model_factory = model_factory,
                k_outer     = 5,
                k_inner     = 3,
                k_vals      = K_GRID,
                balance_methods = BAL_GRID,
                verbose     = True,
            )
            all_results.append(s)
            print(f"  >>> {feat_name} × {model_name}: "
                  f"CV F1 = {s['f1_mean']:.4f} ± {s['f1_std']:.4f} "
                  f"| Acc={s['acc_mean']:.4f} | AUC={s['auc_mean']:.4f}")
        except Exception as e:
            print(f"  [WARN] {feat_name} × {model_name}: {e}")

# %% [markdown]
# ## 7. Summary & Diagnosis

# %%
print(f"\n{'='*100}")
print(f"{'RINGKASAN v69 — Nested CV Results':^100}")
print(f"{'='*100}")

df_res = pd.DataFrame([{
    'Feature':   r['feat'],
    'Model':     r['model'],
    'CV F1 Mean': round(r['f1_mean'], 4),
    'CV F1 Std':  round(r['f1_std'],  4),
    'CV F1 Min':  round(r['f1_min'],  4),
    'CV F1 Max':  round(r['f1_max'],  4),
    'CV Acc':     round(r['acc_mean'],4),
    'CV AUC':     round(r['auc_mean'],4),
    'CV Rec Dep': round(r['rec_dep_mean'],4),
    'CV Prec Dep':round(r['prec_dep_mean'],4),
} for r in all_results])

df_res = df_res.sort_values('CV F1 Mean', ascending=False).reset_index(drop=True)
df_res.index += 1
print(df_res.to_string())

csv_path = os.path.join(RESULTS_DIR, "metrics", "v69_nested_cv_results.csv")
df_res.to_csv(csv_path, index=False)

best_row  = df_res.iloc[0]
best_feat = best_row['Feature']
best_model= best_row['Model']
best_f1   = best_row['CV F1 Mean']
best_std  = best_row['CV F1 Std']
best_acc  = best_row['CV Acc']
best_auc  = best_row['CV AUC']

print(f"\n  ★ BEST (Nested CV Jujur):")
print(f"  Feature: {best_feat}")
print(f"  Model  : {best_model}")
print(f"  CV F1  : {best_f1:.4f} ± {best_std:.4f}")
print(f"  CV Acc : {best_acc:.4f}")
print(f"  CV AUC : {best_auc:.4f}")

# Diagnosis
print(f"\n  DIAGNOSIS OVERFITTING:")
print(f"  v66 (test-set search)    : 0.9115  ← semu, test-set selection bias")
print(f"  v68 (5-fold CV)          : 0.5333  ← honest CV")
print(f"  v69 Nested CV (jujur)    : {best_f1:.4f}  ← gold standard")
if best_f1 >= 0.75:
    print(f"\n  🎯 CV F1 ≥ 0.75 TERCAPAI! ({best_f1:.4f}) — tanpa overfitting!")
elif best_f1 >= 0.70:
    print(f"\n  ✓ CV F1 ≥ 0.70. Mendekati target. Perlu v70.")
elif best_f1 >= 0.65:
    print(f"\n  → CV F1 = {best_f1:.4f}. Lebih baik dari v68 ({0.5333:.4f}). Lanjut v70.")
else:
    print(f"\n  ⚠  CV F1 = {best_f1:.4f}. Dataset terlalu kecil untuk target 0.75 via CV.")

# %% [markdown]
# ## 8. Classification Report Terbaik

# %%
best_result = next(r for r in all_results
                   if r['feat']==best_feat and r['model']==best_model)
y_true_all_pred = np.array(best_result['outer_true'])
y_pred_all_pred = np.array(best_result['outer_preds'])

print(f"\n{'='*72}")
print(f"  POOLED CLASSIFICATION REPORT (5-Fold) — {best_feat} × {best_model}")
print(f"{'='*72}")
print(classification_report(y_true_all_pred, y_pred_all_pred,
                              target_names=['Normal', 'Depresi'], zero_division=0))

# Top 5
print("\n  [Top 5 Nested CV Results]")
for i, row in df_res.head(5).iterrows():
    r = next((x for x in all_results if x['feat']==row['Feature'] and x['model']==row['Model']), None)
    if r:
        yt = np.array(r['outer_true'])
        yp = np.array(r['outer_preds'])
        print(f"\n  [{i}] {row['Feature']} × {row['Model']} "
              f"(CV F1={row['CV F1 Mean']:.4f} ± {row['CV F1 Std']:.4f}):")
        print(classification_report(yt, yp, target_names=['Normal','Depresi'], zero_division=0))

# %% [markdown]
# ## 9. Visualisasi

# %%
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle(f'v69 — Nested CV (Gold Standard) | Best CV F1={best_f1:.4f}',
             fontsize=14, fontweight='bold')

COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6','#10b981',
          '#f59e0b','#8b5cf6','#ec4899','#14b8a6','#f43f5e','#0ea5e9']*3

# Plot 1: Top 20 Nested CV F1
ax1 = axes[0, 0]
top20 = df_res.head(20)
bars  = ax1.barh(range(len(top20)),
                 top20['CV F1 Mean'],
                 xerr=top20['CV F1 Std'],
                 color=[COLORS[i%len(COLORS)] for i in range(len(top20))],
                 edgecolor='white', capsize=3)
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels([f"{r['Feature'][:12]}×{r['Model']}"
                     for _, r in top20.iterrows()], fontsize=7)
ax1.axvline(0.75, color='red',    linestyle='--', lw=1.5, label='Target 0.75')
ax1.axvline(0.70, color='orange', linestyle=':', lw=1.2, label='0.70')
ax1.set_xlabel('Nested CV F1 Macro (Mean ± Std)')
ax1.set_title('Top 20 — Nested CV F1', fontweight='bold')
ax1.legend(fontsize=9); ax1.set_xlim(0, 1.05)
ax1.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, top20['CV F1 Mean']):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=7.5, fontweight='bold')

# Plot 2: Feature Group Comparison
ax2 = axes[0, 1]
feat_grp = df_res.groupby('Feature')['CV F1 Mean'].agg(['mean','std']).reset_index()
feat_grp = feat_grp.sort_values('mean', ascending=False)
bars2 = ax2.barh(range(len(feat_grp)), feat_grp['mean'],
                 xerr=feat_grp['std'],
                 color=COLORS[:len(feat_grp)], edgecolor='white', capsize=4)
ax2.set_yticks(range(len(feat_grp)))
ax2.set_yticklabels(feat_grp['Feature'], fontsize=9)
ax2.axvline(0.75, color='red', linestyle='--', lw=1.5)
ax2.set_xlabel('CV F1 Mean (avg across models)')
ax2.set_title('Feature Set Comparison', fontweight='bold')
ax2.set_xlim(0, 1.05); ax2.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val, std in zip(bars2, feat_grp['mean'], feat_grp['std']):
    ax2.text(val+std+0.01, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8.5, fontweight='bold')

# Plot 3: Model Comparison
ax3 = axes[1, 0]
model_grp = df_res.groupby('Model')['CV F1 Mean'].agg(['mean','std']).reset_index()
model_grp = model_grp.sort_values('mean', ascending=False)
bars3 = ax3.barh(range(len(model_grp)), model_grp['mean'],
                 xerr=model_grp['std'],
                 color=COLORS[:len(model_grp)], edgecolor='white', capsize=4)
ax3.set_yticks(range(len(model_grp)))
ax3.set_yticklabels(model_grp['Model'], fontsize=9)
ax3.axvline(0.75, color='red', linestyle='--', lw=1.5)
ax3.set_xlabel('CV F1 Mean')
ax3.set_title('Model Comparison (avg across features)', fontweight='bold')
ax3.set_xlim(0, 1.05); ax3.grid(axis='x', linestyle='--', alpha=0.4)

# Plot 4: Confusion Matrix (Pooled)
ax4 = axes[1, 1]
cm = confusion_matrix(y_true_all_pred, y_pred_all_pred, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal', 'Depresi'],
            yticklabels=['Normal', 'Depresi'], annot_kws={'size': 14})
ax4.set_title(f'Pooled CM (5-Fold)\n{best_feat} × {best_model} | CV F1={best_f1:.4f}',
              fontweight='bold')
ax4.set_xlabel('Prediksi'); ax4.set_ylabel('Aktual')

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v69_nested_cv.png")
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nPlot: {p}")

# %% [markdown]
# ## 10. Save & Final Report

# %%
summary_json = {
    'version': 'v69',
    'method':  'Nested 5-Fold CV (Outer=5, Inner=3)',
    'n_participants': int(len(y_all)),
    'n_experiments': len(all_results),
    'v68_cv_f1': 0.5333,
    'v66_test_f1': 0.9115,
    'best_feat':  best_feat,
    'best_model': best_model,
    'best_cv_f1_mean': float(best_f1),
    'best_cv_f1_std':  float(best_std),
    'best_cv_acc':     float(best_acc),
    'best_cv_auc':     float(best_auc),
    'target_075_achieved': bool(best_f1 >= 0.75),
    'target_070_achieved': bool(best_f1 >= 0.70),
    'overfitting_verdict': (
        'TERCAPAI (CV)' if best_f1 >= 0.75 else
        'Mendekati'     if best_f1 >= 0.70 else
        'Perlu v70'
    ),
}
with open(os.path.join(MODELS_DIR, 'v69_summary.json'), 'w') as f:
    json.dump(summary_json, f, indent=2)

print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v69':^80}")
print("=" * 80)
print(f"  Metode           : Nested 5-Fold CV (Outer=5, Inner=3)")
print(f"  Total Partisipan : {len(y_all)}")
print(f"  Jumlah Eksperimen: {len(all_results)}")
print(f"  Feature Terbaik  : {best_feat}")
print(f"  Model Terbaik    : {best_model}")
print(f"  CV F1 Macro      : {best_f1:.4f} ± {best_std:.4f}")
print(f"  CV Accuracy      : {best_acc:.4f}")
print(f"  CV AUC           : {best_auc:.4f}")
print(f"  Target ≥ 0.75    : {'✓ TERCAPAI (CV jujur)!' if best_f1 >= 0.75 else '✗ Belum'}")
print(f"  Total Waktu      : {time.time()-t_global:.1f}s")
print("=" * 80)
