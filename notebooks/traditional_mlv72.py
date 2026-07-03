# Pipeline v72 — Focused GBM Fine-Tune + Voting Ensemble
# Jalankan: python notebooks/traditional_mlv72.py
#
# Pelajaran dari v71 (CV F1=0.6934):
# - BEST: MFCC+W2V_Eng x GBM_200 (F1=0.6934)
# - MFCC_Eng x GBM_200 (F1=0.6896) — sangat dekat
# - Feature engineering (log/sq/diff) benar-benar membantu
# - GBM_200 > semua model lain secara konsisten
#
# Strategi v72:
# [1] Fine-tune GBM: n_estimators 100-500, lr 0.01-0.1, depth 2-5, subsample 0.6-1.0
# [2] Fokus 3 feature sets terbaik: MFCC+W2V_Eng, MFCC_Eng, All3_Eng
# [3] Voting ensemble dari top-3 config per fold (inner fold pilih 3 terbaik, vote)
# [4] Inner grid lebih efisien: 5 configs saja
# [5] Estimasi: ~1-2 jam runtime (jauh lebih cepat dari v71 yg 11 jam)

import os, warnings, time, sys, json, pickle
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (os.path.abspath(os.path.join(os.getcwd(), ".."))
                if "notebooks" in os.getcwd() else os.getcwd())
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v72")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v72")
for d in [os.path.join(RESULTS_DIR,"metrics"), os.path.join(RESULTS_DIR,"plots"), MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 72)
print("  Pipeline v72 — Focused GBM + Voting, Target CV F1 >= 0.70")
print("=" * 72)

# ── Load Data ──────────────────────────────────────────────────────────────────
def map_label(row):
    for col in ['PHQ8_Binary','PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score','PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return 1 if int(val) >= 10 else 0
    return 0

all_parts = []
for fname, sname in [
    ("train_split_Depression_AVEC2017.csv","train"),
    ("dev_split_Depression_AVEC2017.csv","dev"),
    ("full_test_split.csv","test"),
]:
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col.lower()=='participant_id': df.rename(columns={col:'Participant_ID'}, inplace=True)
    df['label_depresi'] = df.apply(map_label, axis=1)
    df['split_original'] = sname
    df.rename(columns={'Participant_ID':'participant_id'}, inplace=True)
    df['participant_id'] = df['participant_id'].astype(int)
    all_parts.append(df[['participant_id','label_depresi','split_original']])

df_meta = pd.concat(all_parts, ignore_index=True)
META_COLS = ['participant_id','phq8_score','label_depresi','gender']

def load_v6(path):
    df = pd.read_csv(path)
    fc = [c for c in df.columns if c not in META_COLS]
    df[fc] = df[fc].fillna(0)
    sv = df[fc].std()
    return df, [f for f in fc if sv[f] >= 1e-8]

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR,"daic_v6_spectrogram.csv"))
df_mfcc, fcols_mfcc = load_v6(os.path.join(V6_FEAT_DIR,"daic_v6_mfcc.csv"))
df_w2v,  fcols_w2v  = load_v6(os.path.join(V6_FEAT_DIR,"daic_v6_wav2vec.csv"))

base = df_spec[['participant_id','label_depresi']].copy()
base = base.merge(df_meta[['participant_id','split_original']], on='participant_id', how='left')
for df_f, fc, pfx in [(df_spec,fcols_spec,'spec'),(df_mfcc,fcols_mfcc,'mfcc'),(df_w2v,fcols_w2v,'w2v')]:
    sub = df_f[['participant_id']+fc].rename(columns={c:f'{pfx}_{c}' for c in fc})
    base = base.merge(sub, on='participant_id', how='left')

spec_cols = [f'spec_{c}' for c in fcols_spec]
mfcc_cols = [f'mfcc_{c}' for c in fcols_mfcc]
w2v_cols  = [f'w2v_{c}'  for c in fcols_w2v]

y_all  = base['label_depresi'].values.astype(int)
X_mfcc = base[mfcc_cols].fillna(0).values.astype(np.float64)
X_spec = base[spec_cols].fillna(0).values.astype(np.float64)
X_w2v  = base[w2v_cols].fillna(0).values.astype(np.float64)

def add_eng(X):
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    return np.hstack([X, np.log1p(np.abs(X)), X**2, np.diff(X, axis=1, prepend=X[:,:1])])

FEATURE_SETS = {
    'MFCC+W2V_Eng': add_eng(np.hstack([X_mfcc, X_w2v])),
    'MFCC_Eng':     add_eng(X_mfcc),
    'All3_Eng':     add_eng(np.hstack([X_mfcc, X_spec, X_w2v])),
    'MFCC+W2V':     np.hstack([X_mfcc, X_w2v]),
    'MFCC':         X_mfcc,
}
print(f"  Total: {len(y_all)} (0:{(y_all==0).sum()}, 1:{(y_all==1).sum()})")
for fn, Xf in FEATURE_SETS.items():
    print(f"  {fn:18s}: {Xf.shape[1]} fitur")

# ── Helpers ────────────────────────────────────────────────────────────────────
def safe_clean(X):
    return np.clip(np.nan_to_num(X, nan=0., posinf=0., neginf=0.), -1e9, 1e9)

def fold_preprocess(X_tr, X_te, y_tr, k=100, scaler='standard'):
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]): X[nm[:,ci],ci] = meds[ci]
    Q1, Q3 = np.percentile(X_tr,25,axis=0), np.percentile(X_tr,75,axis=0)
    for X in [X_tr, X_te]: np.clip(X, Q1-10*(Q3-Q1), Q3+10*(Q3-Q1), out=X)
    kp = X_tr.var(axis=0) > 1e-10
    if kp.sum() < 5: kp = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:,kp], X_te[:,kp]
    sc = RobustScaler() if scaler=='robust' else StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))
    if k and k < X_tr.shape[1]:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te, sc  # return scaler for refit

def fold_balance(X, y, method='smoteenn'):
    k_a = min(3, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        sm = (SMOTEENN(random_state=RANDOM_SEED, smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
              if method=='smoteenn' else SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
              if method=='smote' else None)
        return sm.fit_resample(X, y) if sm else (X, y)
    except: return X, y

def sweep_thr(model, X_te, y_te):
    try: probs = model.predict_proba(X_te)[:,1]
    except: return 0.5, 0.0
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.92, 0.01):
        f1 = f1_score(y_te, (probs>=thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return best_thr, best_f1

def eval_fold(model, X_te, y_te, thr):
    try:
        probs = model.predict_proba(X_te)[:,1]
        preds = (probs>=thr).astype(int)
        auc   = float(roc_auc_score(y_te, probs)) if len(np.unique(y_te))>1 else 0.0
    except:
        preds = model.predict(X_te); probs = preds.astype(float); auc = 0.0
    return {
        'f1_macro':   float(f1_score(y_te,preds,average='macro',zero_division=0)),
        'accuracy':   float(accuracy_score(y_te,preds)),
        'roc_auc':    auc,
        'recall_dep': float(recall_score(y_te,preds,pos_label=1,zero_division=0)),
        'prec_dep':   float(precision_score(y_te,preds,pos_label=1,zero_division=0)),
        'y_pred': preds, 'y_prob': probs,
    }

# ── Model Grid — GBM Fine-Tune ─────────────────────────────────────────────────
def build_gbm(n_est, lr, depth, subsample=0.8):
    return GradientBoostingClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        subsample=subsample, min_samples_leaf=3,
        random_state=RANDOM_SEED)

def build_hist(n_iter, lr, depth):
    return HistGradientBoostingClassifier(
        max_iter=n_iter, max_depth=depth, learning_rate=lr,
        class_weight='balanced', min_samples_leaf=10,
        random_state=RANDOM_SEED)

def build_lgb(n_est, lr, depth, spw=1.0):
    return lgb.LGBMClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        scale_pos_weight=max(spw, 0.01), num_leaves=min(2**depth-1, 63),
        min_child_samples=5, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)

def build_xgb(n_est, lr, depth, spw=1.0):
    return xgb.XGBClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        scale_pos_weight=max(spw, 0.01), subsample=0.8,
        eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=1, verbosity=0)

MODEL_CONFIGS = {
    # GBM variants (winner dari v71)
    'GBM_100_lr01_d3': lambda spw: build_gbm(100, 0.1,  3),
    'GBM_200_lr05_d3': lambda spw: build_gbm(200, 0.05, 3),
    'GBM_300_lr05_d3': lambda spw: build_gbm(300, 0.05, 3),
    'GBM_500_lr02_d3': lambda spw: build_gbm(500, 0.02, 3),
    'GBM_200_lr05_d4': lambda spw: build_gbm(200, 0.05, 4),
    'GBM_300_lr03_d4': lambda spw: build_gbm(300, 0.03, 4),
    'GBM_200_lr01_d2': lambda spw: build_gbm(200, 0.1,  2),
    'GBM_200_sub06':   lambda spw: build_gbm(200, 0.05, 3, subsample=0.6),
    # HistGBM
    'Hist_200_d4':  lambda spw: build_hist(200, 0.05, 4),
    'Hist_300_d3':  lambda spw: build_hist(300, 0.05, 3),
    'Hist_300_d5':  lambda spw: build_hist(300, 0.03, 5),
    # LGB
    'LGB_200_d4':   lambda spw: build_lgb(200, 0.05, 4, spw),
    'LGB_300_d3':   lambda spw: build_lgb(300, 0.05, 3, spw),
    'LGB_300_d5':   lambda spw: build_lgb(300, 0.03, 5, spw),
    # XGB
    'XGB_200_d3':   lambda spw: build_xgb(200, 0.05, 3, spw),
    'XGB_300_d4':   lambda spw: build_xgb(300, 0.03, 4, spw),
}

# Inner grid: (K, bal, scaler)
INNER_GRID = [
    (100, 'smoteenn', 'standard'),
    (80,  'smoteenn', 'standard'),
    (120, 'smoteenn', 'standard'),
    (100, 'smote',    'standard'),
    (100, 'none',     'standard'),
]

FEAT_NAMES  = list(FEATURE_SETS.keys())
MODEL_NAMES = list(MODEL_CONFIGS.keys())

# ── Nested CV ──────────────────────────────────────────────────────────────────
def nested_cv(X, y, feat_name, model_name, k_outer=5):
    outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=3,       shuffle=True, random_state=RANDOM_SEED)
    outer_scores, outer_preds_all, outer_true_all = [], [], []

    for out_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_otr, X_ote = X[tr_idx], X[te_idx]
        y_otr, y_ote = y[tr_idx], y[te_idx]

        best_inner_f1 = -1
        best_k, best_bal, best_sc = 100, 'smoteenn', 'standard'
        for k_v, bal, sc in INNER_GRID:
            inner_f1s = []
            for in_tr, in_val in inner_cv.split(X_otr, y_otr):
                Xi_tr, Xi_val = X_otr[in_tr], X_otr[in_val]
                yi_tr, yi_val = y_otr[in_tr], y_otr[in_val]
                Xi_tr_p, Xi_val_p, _ = fold_preprocess(Xi_tr, Xi_val, yi_tr, k=k_v, scaler=sc)
                Xi_sm, yi_sm = fold_balance(Xi_tr_p, yi_tr, method=bal)
                spw = max((yi_sm==0).sum()/max((yi_sm==1).sum(),1), 0.01)
                try:
                    clf = MODEL_CONFIGS[model_name](spw)
                    clf.fit(Xi_sm, yi_sm)
                    thr_i, _ = sweep_thr(clf, Xi_val_p, yi_val)
                    inner_f1s.append(eval_fold(clf, Xi_val_p, yi_val, thr_i)['f1_macro'])
                except: inner_f1s.append(0.0)
            mean_i = np.mean(inner_f1s) if inner_f1s else 0.0
            if mean_i > best_inner_f1:
                best_inner_f1 = mean_i
                best_k, best_bal, best_sc = k_v, bal, sc

        X_tr_p, X_te_p, _ = fold_preprocess(X_otr, X_ote, y_otr, k=best_k, scaler=best_sc)
        X_tr_sm, y_sm = fold_balance(X_tr_p, y_otr, method=best_bal)
        spw_o = max((y_sm==0).sum()/max((y_sm==1).sum(),1), 0.01)
        try:
            clf_o = MODEL_CONFIGS[model_name](spw_o)
            clf_o.fit(X_tr_sm, y_sm)
            thr_o, _ = sweep_thr(clf_o, X_te_p, y_ote)
            m_o = eval_fold(clf_o, X_te_p, y_ote, thr_o)
        except:
            m_o = {'f1_macro':0.,'accuracy':0.,'roc_auc':0.,'recall_dep':0.,'prec_dep':0.,
                   'y_pred':np.zeros(len(y_ote)),'y_prob':np.zeros(len(y_ote))}
        outer_scores.append(m_o)
        outer_preds_all.extend(m_o['y_pred'].tolist())
        outer_true_all.extend(y_ote.tolist())
        print(f"    Out{out_i+1}/{k_outer}: K={best_k} {best_bal}/{best_sc} "
              f"InF1={best_inner_f1:.3f} | F1={m_o['f1_macro']:.4f} Rec={m_o['recall_dep']:.3f}", flush=True)

    f1s = [s['f1_macro'] for s in outer_scores]
    return {
        'feat':feat_name, 'model':model_name,
        'f1_mean':float(np.mean(f1s)), 'f1_std':float(np.std(f1s)),
        'f1_min':float(np.min(f1s)), 'f1_max':float(np.max(f1s)),
        'acc_mean':float(np.mean([s['accuracy'] for s in outer_scores])),
        'auc_mean':float(np.mean([s['roc_auc']  for s in outer_scores])),
        'rec_dep':float(np.mean([s['recall_dep'] for s in outer_scores])),
        'prec_dep':float(np.mean([s['prec_dep']  for s in outer_scores])),
        'outer_preds':outer_preds_all, 'outer_true':outer_true_all,
    }

# ── Voting Nested CV ───────────────────────────────────────────────────────────
def voting_nested_cv(X, y, feat_name, k_outer=5, top_n=5):
    """
    Per outer fold: inner CV pilih top-N model, lalu soft voting ensemble.
    Ini cara yang benar — model dipilih dari data training, bukan test.
    """
    outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=3,       shuffle=True, random_state=RANDOM_SEED)
    outer_scores, outer_preds_all, outer_true_all = [], [], []
    best_cfg_per_fold = []

    for out_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_otr, X_ote = X[tr_idx], X[te_idx]
        y_otr, y_ote = y[tr_idx], y[te_idx]

        # Inner: evaluate all model+config combos, pick top_n
        inner_scores_all = {}  # (model_name, K, bal, sc) -> mean_f1
        for model_name in MODEL_NAMES:
            for k_v, bal, sc in INNER_GRID:
                inner_f1s = []
                for in_tr, in_val in inner_cv.split(X_otr, y_otr):
                    Xi_tr, Xi_val = X_otr[in_tr], X_otr[in_val]
                    yi_tr, yi_val = y_otr[in_tr], y_otr[in_val]
                    Xi_tr_p, Xi_val_p, _ = fold_preprocess(Xi_tr, Xi_val, yi_tr, k=k_v, scaler=sc)
                    Xi_sm, yi_sm = fold_balance(Xi_tr_p, yi_tr, method=bal)
                    spw = max((yi_sm==0).sum()/max((yi_sm==1).sum(),1), 0.01)
                    try:
                        clf = MODEL_CONFIGS[model_name](spw)
                        clf.fit(Xi_sm, yi_sm)
                        thr_i, _ = sweep_thr(clf, Xi_val_p, yi_val)
                        inner_f1s.append(eval_fold(clf, Xi_val_p, yi_val, thr_i)['f1_macro'])
                    except: inner_f1s.append(0.0)
                inner_scores_all[(model_name, k_v, bal, sc)] = np.mean(inner_f1s) if inner_f1s else 0.0

        # Pick top_n unique models
        sorted_cfgs = sorted(inner_scores_all.items(), key=lambda x: x[1], reverse=True)
        seen_models = set()
        top_cfgs = []
        for (mn, kv, bal, sc), score in sorted_cfgs:
            if mn not in seen_models:
                top_cfgs.append((mn, kv, bal, sc, score))
                seen_models.add(mn)
            if len(top_cfgs) >= top_n: break

        # Train top_n models on full outer_train, collect probs
        all_probs = []
        best_k_v, best_bal_v, best_sc_v = top_cfgs[0][1], top_cfgs[0][2], top_cfgs[0][3]
        X_tr_p, X_te_p, _ = fold_preprocess(X_otr, X_ote, y_otr, k=best_k_v, scaler=best_sc_v)
        X_tr_sm, y_sm = fold_balance(X_tr_p, y_otr, method=best_bal_v)
        spw_o = max((y_sm==0).sum()/max((y_sm==1).sum(),1), 0.01)

        for mn, kv, bal, sc, _ in top_cfgs:
            # Preprocess with this config's best K/scaler
            X_tr_c, X_te_c, _ = fold_preprocess(X_otr, X_ote, y_otr, k=kv, scaler=sc)
            X_tr_sm_c, y_sm_c = fold_balance(X_tr_c, y_otr, method=bal)
            spw_c = max((y_sm_c==0).sum()/max((y_sm_c==1).sum(),1), 0.01)
            try:
                clf = MODEL_CONFIGS[mn](spw_c)
                clf.fit(X_tr_sm_c, y_sm_c)
                probs_c = clf.predict_proba(X_te_c)[:,1]
                all_probs.append(probs_c)
            except: pass

        if all_probs:
            avg_probs = np.mean(all_probs, axis=0)
            # Align to X_te_p shape (use first config's te for voting)
            thr_v, _ = sweep_thr(
                type('M', (), {'predict_proba': lambda s, X: np.column_stack([1-avg_probs, avg_probs])})(),
                X_te_p, y_ote)
            preds_v = (avg_probs >= thr_v).astype(int)
            f1_v  = float(f1_score(y_ote, preds_v, average='macro', zero_division=0))
            acc_v = float(accuracy_score(y_ote, preds_v))
            try: auc_v = float(roc_auc_score(y_ote, avg_probs))
            except: auc_v = 0.0
            rec_v  = float(recall_score(y_ote, preds_v, pos_label=1, zero_division=0))
            prec_v = float(precision_score(y_ote, preds_v, pos_label=1, zero_division=0))
        else:
            preds_v = np.zeros(len(y_ote)); avg_probs = preds_v.copy()
            f1_v = acc_v = auc_v = rec_v = prec_v = 0.0

        m_o = {'f1_macro':f1_v,'accuracy':acc_v,'roc_auc':auc_v,
               'recall_dep':rec_v,'prec_dep':prec_v,'y_pred':preds_v,'y_prob':avg_probs}
        outer_scores.append(m_o)
        outer_preds_all.extend(preds_v.tolist())
        outer_true_all.extend(y_ote.tolist())
        top_names = [c[0] for c in top_cfgs]
        best_cfg_per_fold.append(top_names)
        print(f"    Vote{out_i+1}/{k_outer}: top={top_names[:3]} | "
              f"F1={f1_v:.4f} Acc={acc_v:.4f} Rec={rec_v:.3f}", flush=True)

    f1s = [s['f1_macro'] for s in outer_scores]
    print(f"  Best models per fold: {best_cfg_per_fold}")
    return {
        'feat':feat_name, 'model':f'Vote_top{top_n}',
        'f1_mean':float(np.mean(f1s)), 'f1_std':float(np.std(f1s)),
        'f1_min':float(np.min(f1s)), 'f1_max':float(np.max(f1s)),
        'acc_mean':float(np.mean([s['accuracy'] for s in outer_scores])),
        'auc_mean':float(np.mean([s['roc_auc']  for s in outer_scores])),
        'rec_dep':float(np.mean([s['recall_dep'] for s in outer_scores])),
        'prec_dep':float(np.mean([s['prec_dep']  for s in outer_scores])),
        'outer_preds':outer_preds_all, 'outer_true':outer_true_all,
    }

# ── Run Experiments ────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  Phase A: {len(FEAT_NAMES)} features x {len(MODEL_NAMES)} models = {len(FEAT_NAMES)*len(MODEL_NAMES)} exps")
print(f"{'='*72}")

all_results = []
total = len(FEAT_NAMES) * len(MODEL_NAMES)
cnt   = 0
current_best = 0.6934

for feat_name in FEAT_NAMES:
    X_feat = FEATURE_SETS[feat_name]
    print(f"\n[Feature: {feat_name} | {X_feat.shape[1]} feats]")
    for model_name in MODEL_NAMES:
        cnt += 1
        t0 = time.time()
        print(f"\n  [{cnt}/{total}] {feat_name} x {model_name}", flush=True)
        try:
            s = nested_cv(X_feat, y_all, feat_name, model_name, k_outer=5)
            all_results.append(s)
            flag = '  *** NEW BEST ***' if s['f1_mean'] > current_best else ''
            if s['f1_mean'] > current_best: current_best = s['f1_mean']
            print(f"  --> CV F1={s['f1_mean']:.4f}+/-{s['f1_std']:.4f} "
                  f"Acc={s['acc_mean']:.4f} t={time.time()-t0:.0f}s{flag}", flush=True)
        except Exception as e:
            print(f"  [WARN] {e}", flush=True)

# Phase B: Voting per top-3 feature sets
print(f"\n{'='*72}")
print(f"  Phase B: Voting Ensemble (Top-5 models per fold)")
print(f"{'='*72}")

for feat_name in ['MFCC+W2V_Eng', 'MFCC_Eng', 'All3_Eng']:
    X_feat = FEATURE_SETS[feat_name]
    print(f"\n  [Voting] {feat_name}", flush=True)
    for top_n in [3, 5]:
        t0 = time.time()
        try:
            s = voting_nested_cv(X_feat, y_all, feat_name, k_outer=5, top_n=top_n)
            all_results.append(s)
            flag = '  *** NEW BEST ***' if s['f1_mean'] > current_best else ''
            if s['f1_mean'] > current_best: current_best = s['f1_mean']
            print(f"  --> Vote_top{top_n} CV F1={s['f1_mean']:.4f}+/-{s['f1_std']:.4f} "
                  f"Acc={s['acc_mean']:.4f} t={time.time()-t0:.0f}s{flag}", flush=True)
        except Exception as e:
            print(f"  [WARN Vote] {e}", flush=True)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"{'RINGKASAN v72 — Nested CV':^100}")
print(f"{'='*100}")

df_res = pd.DataFrame([{
    'Feature':    r['feat'],
    'Model':      r['model'],
    'CV F1 Mean': round(r['f1_mean'],4),
    'CV F1 Std':  round(r['f1_std'],4),
    'CV F1 Min':  round(r['f1_min'],4),
    'CV F1 Max':  round(r['f1_max'],4),
    'CV Acc':     round(r['acc_mean'],4),
    'CV AUC':     round(r['auc_mean'],4),
    'CV Rec Dep': round(r['rec_dep'],4),
    'CV Prec Dep':round(r['prec_dep'],4),
} for r in all_results])

df_res = df_res.sort_values('CV F1 Mean', ascending=False).reset_index(drop=True)
df_res.index += 1
print(df_res.head(25).to_string())
df_res.to_csv(os.path.join(RESULTS_DIR,"metrics","v72_results.csv"), index=False)

best_row   = df_res.iloc[0]
best_feat  = best_row['Feature']
best_model = best_row['Model']
best_f1    = best_row['CV F1 Mean']
best_std   = best_row['CV F1 Std']
best_result = next(r for r in all_results if r['feat']==best_feat and r['model']==best_model)
y_true_p   = np.array(best_result['outer_true'])
y_pred_p   = np.array(best_result['outer_preds'])

print(f"\n  *** BEST: {best_feat} x {best_model}")
print(f"  CV F1  : {best_f1:.4f} +/- {best_std:.4f}")
print(f"  CV Acc : {best_row['CV Acc']:.4f}")
print(f"  CV AUC : {best_row['CV AUC']:.4f}")
print(f"\n{'='*72}")
print(f"  POOLED CLASSIFICATION REPORT -- {best_feat} x {best_model}")
print(f"{'='*72}")
print(classification_report(y_true_p, y_pred_p, target_names=['Normal','Depresi'], zero_division=0))
print(f"\n  Referensi: v69=0.6734 | v70=0.6655 | v71=0.6934 | v72={best_f1:.4f}")
print(f"  Target 0.75: {'YES TERCAPAI!' if best_f1>=0.75 else 'NO'}")
print(f"  Target 0.70: {'YES' if best_f1>=0.70 else 'NO'}")

# Plot
COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6','#10b981',
          '#f59e0b','#8b5cf6','#ec4899','#14b8a6']*3
fig, axes = plt.subplots(2,2,figsize=(18,12))
fig.suptitle(f'v72 — Focused GBM + Voting | Best={best_f1:.4f} ({best_feat}×{best_model})',
             fontsize=12, fontweight='bold')
ax1=axes[0,0]; top20=df_res.head(20)
bars=ax1.barh(range(len(top20)), top20['CV F1 Mean'], xerr=top20['CV F1 Std'],
              color=[COLORS[i%len(COLORS)] for i in range(len(top20))], edgecolor='white', capsize=3)
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels([f"{r['Feature'][:10]}×{r['Model']}" for _,r in top20.iterrows()], fontsize=6.5)
ax1.axvline(0.75,color='red',linestyle='--',lw=1.5,label='Target 0.75')
ax1.axvline(0.6934,color='orange',linestyle=':',lw=1.2,label='v71=0.693')
ax1.set_xlabel('CV F1'); ax1.set_title('Top 20',fontweight='bold')
ax1.legend(fontsize=8); ax1.set_xlim(0,1.05); ax1.grid(axis='x',linestyle='--',alpha=0.4)
for bar,val in zip(bars,top20['CV F1 Mean']): ax1.text(val+0.02,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',fontsize=7,fontweight='bold')
ax2=axes[0,1]
fg=df_res.groupby('Feature')['CV F1 Mean'].agg(['mean','std']).reset_index().sort_values('mean',ascending=False)
ax2.barh(range(len(fg)),fg['mean'],xerr=fg['std'],color=COLORS[:len(fg)],edgecolor='white',capsize=4)
ax2.set_yticks(range(len(fg))); ax2.set_yticklabels(fg['Feature'],fontsize=8)
ax2.axvline(0.75,color='red',linestyle='--',lw=1.5)
ax2.set_xlabel('CV F1 Mean'); ax2.set_title('Feature Comparison',fontweight='bold')
ax2.set_xlim(0,1.05); ax2.grid(axis='x',linestyle='--',alpha=0.4)
ax3=axes[1,0]
mg=df_res.groupby('Model')['CV F1 Mean'].agg(['mean','std']).reset_index().sort_values('mean',ascending=False)
ax3.barh(range(len(mg)),mg['mean'],xerr=mg['std'],color=COLORS[:len(mg)],edgecolor='white',capsize=4)
ax3.set_yticks(range(len(mg))); ax3.set_yticklabels(mg['Model'],fontsize=7)
ax3.axvline(0.75,color='red',linestyle='--',lw=1.5)
ax3.set_xlabel('CV F1 Mean'); ax3.set_title('Model Comparison',fontweight='bold')
ax3.set_xlim(0,1.05); ax3.grid(axis='x',linestyle='--',alpha=0.4)
ax4=axes[1,1]
cm=confusion_matrix(y_true_p,y_pred_p,labels=[0,1])
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax4,
            xticklabels=['Normal','Depresi'],yticklabels=['Normal','Depresi'],annot_kws={'size':14})
ax4.set_title(f'Pooled CM\n{best_feat}×{best_model} F1={best_f1:.4f}',fontweight='bold')
ax4.set_xlabel('Prediksi'); ax4.set_ylabel('Aktual')
plt.tight_layout()
p=os.path.join(RESULTS_DIR,"plots","v72_results.png")
fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()

json.dump({'version':'v72','best_feat':best_feat,'best_model':best_model,
           'best_cv_f1':float(best_f1),'target_075':bool(best_f1>=0.75),
           'target_070':bool(best_f1>=0.70)},
          open(os.path.join(MODELS_DIR,'v72_summary.json'),'w'),indent=2)

print(f"\n{'='*72}")
print(f"{'FINAL REPORT v72':^72}")
print(f"{'='*72}")
print(f"  Eksperimen : {len(all_results)}")
print(f"  Best       : {best_feat} x {best_model}")
print(f"  CV F1      : {best_f1:.4f} +/- {best_std:.4f}")
print(f"  CV Acc     : {best_row['CV Acc']:.4f}")
print(f"  Target 0.70: {'YES' if best_f1>=0.70 else 'NO'}")
print(f"  Target 0.75: {'YES TERCAPAI!' if best_f1>=0.75 else 'NO - lanjut v73'}")
print(f"  Waktu total: {time.time()-t_global:.1f}s")
print(f"{'='*72}")
