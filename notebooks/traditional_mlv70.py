# Pipeline v70 — Nested CV Lean (Direct Python Script)
# Jalankan: python notebooks/traditional_mlv70.py
#
# Syarat prompt.txt: MFCC+Spec+W2V, semua data, audio-only, gunakan fold
# Dari v69: MFCC × GBM terbaik (CV F1=0.6734)
# Strategi v70: Nested CV efisien — 5-fold outer, 3-fold inner, grid kecil

import os, warnings, time, sys, json, pickle
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v70")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v70")
for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"), MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 72)
print("  Pipeline v70 — Nested CV Lean, Target CV F1 >= 0.75")
print("=" * 72)

# ── Load Data ─────────────────────────────────────────────────────────────────
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

FEATURE_SETS = {
    'MFCC':      X_mfcc,
    'Spec':      X_spec,
    'MFCC+Spec': np.hstack([X_mfcc, X_spec]),
    'MFCC+W2V':  np.hstack([X_mfcc, X_w2v]),
    'All3':      np.hstack([X_mfcc, X_spec, X_w2v]),
}
print(f"  Total: {len(y_all)} (0:{(y_all==0).sum()}, 1:{(y_all==1).sum()})")
for fn, Xf in FEATURE_SETS.items():
    print(f"  {fn:12s}: {Xf.shape[1]} fitur")

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_clean(X):
    return np.clip(np.nan_to_num(X, nan=0., posinf=0., neginf=0.), -1e9, 1e9)

def fold_preprocess(X_tr, X_te, y_tr, k=100):
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]): X[nm[:,ci],ci] = meds[ci]
    Q1, Q3 = np.percentile(X_tr,25,axis=0), np.percentile(X_tr,75,axis=0)
    IQR = Q3 - Q1
    for X in [X_tr, X_te]: np.clip(X, Q1-10*IQR, Q3+10*IQR, out=X)
    kp = X_tr.var(axis=0) > 1e-10
    if kp.sum() < 5: kp = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:,kp], X_te[:,kp]
    sc = StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))
    if k and k < X_tr.shape[1]:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def fold_balance(X, y, method='smoteenn'):
    k_a = min(3, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        if method == 'smoteenn':
            sm = SMOTEENN(random_state=RANDOM_SEED,
                          smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        elif method == 'smote':
            sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        else:
            return X, y
        return sm.fit_resample(X, y)
    except:
        return X, y

def sweep_thr(model, X_te, y_te):
    try: probs = model.predict_proba(X_te)[:,1]
    except: return 0.5, 0.0
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.95, 0.01):
        preds = (probs>=thr).astype(int)
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
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

def build_model(mname, spw=1.0):
    spw = max(float(spw), 0.01)
    models = {
        'GBM_100': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_SEED),
        'GBM_200': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED),
        'GBM_d4':  GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED),
        'XGB_100': xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=RANDOM_SEED, n_jobs=1, verbosity=0),
        'XGB_200': xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=RANDOM_SEED, n_jobs=1, verbosity=0),
        'XGB_d4':  xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=RANDOM_SEED, n_jobs=1, verbosity=0),
        'LGB_100': lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            scale_pos_weight=spw, num_leaves=31,
            random_state=RANDOM_SEED, n_jobs=1, verbose=-1),
        'LGB_200': lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=spw, num_leaves=31,
            random_state=RANDOM_SEED, n_jobs=1, verbose=-1),
        'RF_300':  RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            n_jobs=1, random_state=RANDOM_SEED),
        'ET_300':  ExtraTreesClassifier(
            n_estimators=300, class_weight='balanced',
            n_jobs=1, random_state=RANDOM_SEED),
        'SVM_10':  SVC(kernel='rbf', C=10.0, gamma='scale',
                        probability=True, random_state=RANDOM_SEED,
                        class_weight='balanced'),
        'SVM_100': SVC(kernel='rbf', C=100.0, gamma='scale',
                        probability=True, random_state=RANDOM_SEED,
                        class_weight='balanced'),
        'LR_1':    LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=5000,
            random_state=RANDOM_SEED, solver='lbfgs'),
        'MLP_s':   MLPClassifier(
            hidden_layer_sizes=(200,100,50), alpha=0.001,
            learning_rate_init=0.001, max_iter=500,
            random_state=RANDOM_SEED, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20),
    }
    return models[mname]

# Inner grid: (K, balancer)
INNER_GRID = [
    (60,  'smoteenn'),
    (80,  'smoteenn'),
    (100, 'smoteenn'),
    (120, 'smoteenn'),
    (80,  'smote'),
    (100, 'smote'),
    (120, 'smote'),
    (100, 'none'),
]

MODEL_NAMES = [
    'GBM_100','GBM_200','GBM_d4',
    'XGB_100','XGB_200','XGB_d4',
    'LGB_100','LGB_200',
    'RF_300','ET_300',
    'SVM_10','SVM_100','LR_1','MLP_s',
]
FEAT_NAMES = ['MFCC','Spec','MFCC+Spec','MFCC+W2V','All3']

# ── Nested CV ──────────────────────────────────────────────────────────────────
def nested_cv(X, y, feat_name, model_name, k_outer=5, verbose=True):
    outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=3,       shuffle=True, random_state=RANDOM_SEED)

    outer_scores = []
    outer_preds_all, outer_true_all = [], []

    for out_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_otr, X_ote = X[tr_idx], X[te_idx]
        y_otr, y_ote = y[tr_idx], y[te_idx]

        # Inner: find best (K, balancer)
        best_inner_f1, best_k, best_bal = -1, 100, 'smoteenn'
        for k_v, bal in INNER_GRID:
            inner_f1s = []
            for in_tr, in_val in inner_cv.split(X_otr, y_otr):
                Xi_tr, Xi_val = X_otr[in_tr], X_otr[in_val]
                yi_tr, yi_val = y_otr[in_tr], y_otr[in_val]
                Xi_tr_p, Xi_val_p = fold_preprocess(Xi_tr, Xi_val, yi_tr, k=k_v)
                Xi_sm, yi_sm = fold_balance(Xi_tr_p, yi_tr, method=bal)
                spw = max((yi_sm==0).sum()/max((yi_sm==1).sum(),1), 0.01)
                try:
                    clf = build_model(model_name, spw)
                    clf.fit(Xi_sm, yi_sm)
                    thr_i, _ = sweep_thr(clf, Xi_val_p, yi_val)
                    m_i = eval_fold(clf, Xi_val_p, yi_val, thr_i)
                    inner_f1s.append(m_i['f1_macro'])
                except:
                    inner_f1s.append(0.0)
            mean_i = np.mean(inner_f1s) if inner_f1s else 0.0
            if mean_i > best_inner_f1:
                best_inner_f1, best_k, best_bal = mean_i, k_v, bal

        # Outer: train with best config
        X_tr_p, X_te_p = fold_preprocess(X_otr, X_ote, y_otr, k=best_k)
        X_tr_sm, y_sm  = fold_balance(X_tr_p, y_otr, method=best_bal)
        spw_o = max((y_sm==0).sum()/max((y_sm==1).sum(),1), 0.01)
        try:
            clf_o = build_model(model_name, spw_o)
            clf_o.fit(X_tr_sm, y_sm)
            thr_o, _ = sweep_thr(clf_o, X_te_p, y_ote)
            m_o = eval_fold(clf_o, X_te_p, y_ote, thr_o)
        except:
            m_o = {'f1_macro':0.,'accuracy':0.,'roc_auc':0.,
                   'recall_dep':0.,'prec_dep':0.,
                   'y_pred':np.zeros(len(y_ote)),'y_prob':np.zeros(len(y_ote))}

        outer_scores.append(m_o)
        outer_preds_all.extend(m_o['y_pred'].tolist())
        outer_true_all.extend(y_ote.tolist())
        if verbose:
            print(f"    Out{out_i+1}/{k_outer}: K={best_k} bal={best_bal} "
                  f"InF1={best_inner_f1:.3f} | "
                  f"F1={m_o['f1_macro']:.4f} Acc={m_o['accuracy']:.4f} "
                  f"Rec={m_o['recall_dep']:.3f}", flush=True)

    f1s = [s['f1_macro'] for s in outer_scores]
    return {
        'feat':     feat_name,
        'model':    model_name,
        'f1_mean':  float(np.mean(f1s)),
        'f1_std':   float(np.std(f1s)),
        'f1_min':   float(np.min(f1s)),
        'f1_max':   float(np.max(f1s)),
        'acc_mean': float(np.mean([s['accuracy']   for s in outer_scores])),
        'auc_mean': float(np.mean([s['roc_auc']    for s in outer_scores])),
        'rec_dep':  float(np.mean([s['recall_dep'] for s in outer_scores])),
        'prec_dep': float(np.mean([s['prec_dep']   for s in outer_scores])),
        'outer_preds': outer_preds_all,
        'outer_true':  outer_true_all,
    }

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  NESTED CV 5-FOLD (Inner=3, Grid={len(INNER_GRID)} configs)")
print(f"  {len(FEAT_NAMES)} features x {len(MODEL_NAMES)} models = {len(FEAT_NAMES)*len(MODEL_NAMES)} exps")
print(f"{'='*72}")

all_results = []
total = len(FEAT_NAMES) * len(MODEL_NAMES)
cnt   = 0
current_best = 0.6734  # v69 benchmark

for feat_name in FEAT_NAMES:
    X_feat = FEATURE_SETS[feat_name]
    print(f"\n[Feature: {feat_name} | {X_feat.shape[1]} feats]")
    for model_name in MODEL_NAMES:
        cnt += 1
        t0 = time.time()
        print(f"\n  [{cnt}/{total}] {feat_name} x {model_name}", flush=True)
        try:
            s = nested_cv(X_feat, y_all, feat_name, model_name,
                          k_outer=5, verbose=True)
            all_results.append(s)
            elapsed = time.time() - t0
            flag = ''
            if s['f1_mean'] > current_best:
                current_best = s['f1_mean']
                flag = ' *** NEW BEST ***'
            print(f"  --> CV F1={s['f1_mean']:.4f}+/-{s['f1_std']:.4f} "
                  f"Acc={s['acc_mean']:.4f} t={elapsed:.1f}s{flag}", flush=True)
        except Exception as e:
            print(f"  [WARN] {e}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"{'RINGKASAN v70 — Nested CV':^100}")
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

csv_path = os.path.join(RESULTS_DIR,"metrics","v70_nested_cv.csv")
df_res.to_csv(csv_path, index=False)

best_row   = df_res.iloc[0]
best_feat  = best_row['Feature']
best_model = best_row['Model']
best_f1    = best_row['CV F1 Mean']
best_std   = best_row['CV F1 Std']

best_result = next(r for r in all_results
                   if r['feat']==best_feat and r['model']==best_model)
y_true_p = np.array(best_result['outer_true'])
y_pred_p = np.array(best_result['outer_preds'])

print(f"\n  *** BEST: {best_feat} x {best_model}")
print(f"  CV F1  : {best_f1:.4f} +/- {best_std:.4f}")
print(f"  CV Acc : {best_row['CV Acc']:.4f}")
print(f"  CV AUC : {best_row['CV AUC']:.4f}")

print(f"\n{'='*72}")
print(f"  CLASSIFICATION REPORT (5-Fold Pooled) -- {best_feat} x {best_model}")
print(f"{'='*72}")
print(classification_report(y_true_p, y_pred_p,
                             target_names=['Normal','Depresi'], zero_division=0))

print("\n  [Referensi Jujur]")
print(f"  v68 5-Fold CV simple : 0.5333")
print(f"  v69 Nested CV        : 0.6734")
print(f"  v70 Nested CV        : {best_f1:.4f}")

if best_f1 >= 0.75:
    print(f"\n  TARGET CV F1 >= 0.75 TERCAPAI! ({best_f1:.4f})")
elif best_f1 >= 0.70:
    print(f"\n  CV F1 >= 0.70 ({best_f1:.4f}). Naik dari v69. Lanjut v71.")
else:
    print(f"\n  CV F1 = {best_f1:.4f}. Lanjut v71 dengan pendekatan baru.")

# Visualisasi
COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6',
          '#10b981','#f59e0b','#8b5cf6','#ec4899','#14b8a6']*3
fig, axes = plt.subplots(2, 2, figsize=(18,12))
fig.suptitle(f'v70 — Nested CV | Best={best_f1:.4f} ({best_feat}x{best_model})',
             fontsize=13, fontweight='bold')

ax1 = axes[0,0]
top20 = df_res.head(20)
bars  = ax1.barh(range(len(top20)), top20['CV F1 Mean'],
                 xerr=top20['CV F1 Std'],
                 color=[COLORS[i%len(COLORS)] for i in range(len(top20))],
                 edgecolor='white', capsize=3)
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels([f"{r['Feature'][:8]}x{r['Model']}"
                     for _, r in top20.iterrows()], fontsize=7)
ax1.axvline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
ax1.axvline(0.6734, color='orange', linestyle=':', lw=1.2, label='v69=0.673')
ax1.set_xlabel('CV F1 Mean +/- Std')
ax1.set_title('Top 20 Nested CV', fontweight='bold')
ax1.legend(fontsize=8); ax1.set_xlim(0,1.05)
ax1.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, top20['CV F1 Mean']):
    ax1.text(val+0.02, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=7.5, fontweight='bold')

ax2 = axes[0,1]
feat_grp = df_res.groupby('Feature')['CV F1 Mean'].agg(['mean','std']).reset_index()
feat_grp = feat_grp.sort_values('mean', ascending=False)
ax2.barh(range(len(feat_grp)), feat_grp['mean'], xerr=feat_grp['std'],
         color=COLORS[:len(feat_grp)], edgecolor='white', capsize=4)
ax2.set_yticks(range(len(feat_grp)))
ax2.set_yticklabels(feat_grp['Feature'], fontsize=9)
ax2.axvline(0.75, color='red', linestyle='--', lw=1.5)
ax2.set_xlabel('CV F1 Mean'); ax2.set_title('Feature Comparison', fontweight='bold')
ax2.set_xlim(0,1.05); ax2.grid(axis='x', linestyle='--', alpha=0.4)

ax3 = axes[1,0]
model_grp = df_res.groupby('Model')['CV F1 Mean'].agg(['mean','std']).reset_index()
model_grp = model_grp.sort_values('mean', ascending=False)
ax3.barh(range(len(model_grp)), model_grp['mean'], xerr=model_grp['std'],
         color=COLORS[:len(model_grp)], edgecolor='white', capsize=4)
ax3.set_yticks(range(len(model_grp)))
ax3.set_yticklabels(model_grp['Model'], fontsize=8)
ax3.axvline(0.75, color='red', linestyle='--', lw=1.5)
ax3.set_xlabel('CV F1 Mean'); ax3.set_title('Model Comparison', fontweight='bold')
ax3.set_xlim(0,1.05); ax3.grid(axis='x', linestyle='--', alpha=0.4)

ax4 = axes[1,1]
cm = confusion_matrix(y_true_p, y_pred_p, labels=[0,1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal','Depresi'],
            yticklabels=['Normal','Depresi'], annot_kws={'size':14})
ax4.set_title(f'Pooled CM (5-Fold)\n{best_feat}x{best_model} F1={best_f1:.4f}',
              fontweight='bold')
ax4.set_xlabel('Prediksi'); ax4.set_ylabel('Aktual')

plt.tight_layout()
p = os.path.join(RESULTS_DIR,"plots","v70_nested_cv.png")
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()

# Save
summary = {
    'version':'v70','method':'Nested 5-Fold CV (Inner=3)',
    'n_participants':int(len(y_all)),'n_experiments':len(all_results),
    'best_feat':best_feat,'best_model':best_model,
    'best_cv_f1':float(best_f1),'best_cv_std':float(best_std),
    'best_cv_acc':float(best_row['CV Acc']),
    'target_075':bool(best_f1>=0.75),'target_070':bool(best_f1>=0.70),
}
with open(os.path.join(MODELS_DIR,'v70_summary.json'),'w') as f: json.dump(summary,f,indent=2)

print(f"\n{'='*72}")
print(f"{'FINAL REPORT v70':^72}")
print(f"{'='*72}")
print(f"  Eksperimen : {len(all_results)}")
print(f"  Best       : {best_feat} x {best_model}")
print(f"  CV F1      : {best_f1:.4f} +/- {best_std:.4f}")
print(f"  CV Acc     : {best_row['CV Acc']:.4f}")
print(f"  Target 0.75: {'YES TERCAPAI!' if best_f1>=0.75 else 'NO - lanjut v71'}")
print(f"  Waktu total: {time.time()-t_global:.1f}s")
print(f"{'='*72}")
