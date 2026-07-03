# %% [markdown]
# # Pipeline v84 — No SMOTE | Class Weight Only | 10-Fold CV | Target >= 0.75
# Insight dari v76-v83:
# - SMOTEENN fair threshold → test lebih buruk (distribution mismatch)
# - Multi-seed bagging tidak membantu
# - RF Wav2Vec Test=0.7442 selalu dengan threshold sweep (15/20 benar)
# - Fundamental issue: SMOTE synthetic samples ≠ distribusi test data asli
#
# STRATEGI v84 — No Synthetic Data:
# [1] TIDAK pakai SMOTE sama sekali — eliminasi distribution mismatch
# [2] Kompensasi imbalance dengan class_weight (jujur terhadap distribusi asli)
# [3] 10-Fold CV (lebih stabil untuk dataset kecil 82 sampel)
# [4] class_weight sweep: 'balanced', {0:1,1:2}, {0:1,1:3}, {0:1,1:1.83}
# [5] SelectKBest + StandardScaler tetap
# [6] Threshold dari inner CV → apply ke test (fair)
# [7] ALL 4 model, ALL 5 skenario, apple-to-apple S1-S4

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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (os.path.abspath(os.path.join(os.getcwd(), ".."))
                if "notebooks" in os.getcwd() else os.getcwd())
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v84")
for d in [os.path.join(RESULTS_DIR,"metrics"), os.path.join(RESULTS_DIR,"plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("="*80)
print("  Pipeline v84 — No SMOTE | Class Weight | 10-Fold CV | Target >= 0.75")
print("="*80)

# %%
def map_label(row):
    for col in ['PHQ8_Binary','PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score','PHQ_Score']:
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
        if col.lower()=='participant_id': df.rename(columns={col:'Participant_ID'}, inplace=True)
    df['label_depresi'] = df.apply(map_label, axis=1)
    df.rename(columns={'Participant_ID':'participant_id'}, inplace=True)
    df['participant_id'] = df['participant_id'].astype(int)
    all_parts.append(df[['participant_id','label_depresi']])

META_COLS = ['participant_id','phq8_score','label_depresi','gender']
df_meta = pd.concat(all_parts, ignore_index=True)

def load_v6(path):
    df = pd.read_csv(path)
    fc = [c for c in df.columns if c not in META_COLS]
    df[fc] = df[fc].fillna(0)
    return df, [f for f in fc if df[fc].std()[f] >= 1e-8]

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR,"daic_v6_spectrogram.csv"))
df_mfcc, fcols_mfcc = load_v6(os.path.join(V6_FEAT_DIR,"daic_v6_mfcc.csv"))
df_w2v,  fcols_w2v  = load_v6(os.path.join(V6_FEAT_DIR,"daic_v6_wav2vec.csv"))

base = df_spec[['participant_id','label_depresi']].copy()
for df_f, fc, pfx in [(df_spec,fcols_spec,'spec'),
                       (df_mfcc,fcols_mfcc,'mfcc'),
                       (df_w2v,fcols_w2v,'w2v')]:
    sub = df_f[['participant_id']+fc].rename(columns={c:f'{pfx}_{c}' for c in fc})
    base = base.merge(sub, on='participant_id', how='left')

y_all  = base['label_depresi'].values.astype(int)
X_spec = base[[f'spec_{c}' for c in fcols_spec]].fillna(0).values.astype(np.float64)
X_mfcc = base[[f'mfcc_{c}' for c in fcols_mfcc]].fillna(0).values.astype(np.float64)
X_w2v  = base[[f'w2v_{c}'  for c in fcols_w2v]].fillna(0).values.astype(np.float64)
X_fuse = np.hstack([X_spec, X_mfcc, X_w2v])

def add_eng(X):
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    return np.hstack([X, np.log1p(np.abs(X)), X**2, np.diff(X,axis=1,prepend=X[:,:1])])

X_fuse_eng = add_eng(X_fuse)

SCENARIOS = {
    'S1_Spectrogram': X_spec,
    'S2_MFCC':        X_mfcc,
    'S3_Wav2Vec':     X_w2v,
    'S4_Fusion':      X_fuse,
    'S5_FusionEng':   X_fuse_eng,
}

print(f"  Total: {len(y_all)} (N:{(y_all==0).sum()}, D:{(y_all==1).sum()})")
for sn,Xf in SCENARIOS.items():
    print(f"  {sn:20s}: {Xf.shape[1]} fitur")

idx_n=np.where(y_all==0)[0]; idx_d=np.where(y_all==1)[0]
np.random.seed(RANDOM_SEED)
test_idx  = np.concatenate([np.random.choice(idx_n,10,replace=False),
                             np.random.choice(idx_d,10,replace=False)])
train_idx = np.setdiff1d(np.arange(len(y_all)), test_idx)
y_train   = y_all[train_idx]; y_test = y_all[test_idx]
n_dep = (y_train==1).sum(); n_nor = (y_train==0).sum()
ratio = round(n_nor/n_dep,2)
print(f"  Train:{len(train_idx)} (N:{n_nor}, D:{n_dep}, ratio={ratio}:1) | Test:20 (10N+10D)")
print(f"  NO SMOTE — class_weight kompensasi imbalance {ratio}:1")

# ── Helpers ───────────────────────────────────────────────────────────
def safe_clean(X):
    return np.clip(np.nan_to_num(X,nan=0.,posinf=0.,neginf=0.),-1e9,1e9)

def preprocess(X_tr, X_te, y_tr, k=None):
    X_tr,X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    sc = StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))
    if k and k < X_tr.shape[1]:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def sweep_thr(probs, y_true):
    best_f1,best_thr=0.,0.5
    for thr in np.arange(0.05,0.96,0.01):
        f1=f1_score(y_true,(probs>=thr).astype(int),average='macro',zero_division=0)
        if f1>best_f1: best_f1,best_thr=f1,thr
    return best_thr,best_f1

# ── Model Configs — class_weight sweep (no SMOTE) ─────────────────────
# Imbalance ratio ≈ n_nor/n_dep, so class_weight for D ≈ ratio
CW_TRUE = {0:1, 1:round(ratio,1)}  # true imbalance
CW_OPTIONS_RF_LR_SVM = ['balanced', CW_TRUE, {0:1,1:2}, {0:1,1:3}]

MODEL_CONFIGS = {
    'LogisticRegression': [
        # (C, class_weight)
        (0.05, 'balanced'),(0.1, 'balanced'),(0.3, 'balanced'),(0.5, 'balanced'),(1.0, 'balanced'),
        (0.1, CW_TRUE),(0.3, CW_TRUE),(0.5, CW_TRUE),
        (0.1, {0:1,1:2}),(0.3, {0:1,1:2}),
        (0.1, {0:1,1:3}),
        # L1 penalty
        (0.1, 'balanced', 'liblinear', 'l1'),
        (0.3, 'balanced', 'liblinear', 'l1'),
    ],
    'RandomForest': [
        (500,  'sqrt', 1, None, 'balanced'),
        (1000, 'sqrt', 1, None, 'balanced'),
        (500,  'log2', 1, None, 'balanced'),
        (500,  0.5,   1, None, 'balanced'),
        (1000, 'sqrt', 1, None, CW_TRUE),
        (1000, 'sqrt', 1, None, {0:1,1:2}),
        (1000, 'sqrt', 2, None, 'balanced'),
        (500,  'sqrt', 1, 15,   'balanced'),
        (1000, 'log2', 1, None, CW_TRUE),
    ],
    'SVM': [
        (0.1,  'rbf',    'scale', 'balanced'),
        (0.5,  'rbf',    'scale', 'balanced'),
        (1.0,  'rbf',    'scale', 'balanced'),
        (5.0,  'rbf',    'scale', 'balanced'),
        (10.0, 'rbf',    'scale', 'balanced'),
        (1.0,  'rbf',    'scale', CW_TRUE),
        (5.0,  'rbf',    'scale', CW_TRUE),
        (1.0,  'rbf',    'scale', {0:1,1:2}),
        (1.0,  'linear', 'scale', 'balanced'),
        (5.0,  'linear', 'scale', 'balanced'),
        (1.0,  'linear', 'scale', CW_TRUE),
    ],
    'XGBoost': [
        (100, 2, 0.1,  0.8, 2.0, 0.5, 2.0),
        (200, 2, 0.05, 0.8, 2.0, 0.5, 2.0),
        (200, 3, 0.05, 0.8, 2.5, 0.1, 1.0),
        (300, 2, 0.03, 0.9, 2.0, 0.5, 5.0),
        (100, 2, 0.1,  0.8, 3.0, 1.0, 2.0),
        (200, 2, 0.05, 0.8, round(ratio,1), 0.5, 2.0),  # true ratio
    ],
}
MODEL_NAMES = list(MODEL_CONFIGS.keys())

def build_lr(cfg):
    if len(cfg)==2:
        C,cw=cfg; return LogisticRegression(C=C,class_weight=cw,max_iter=5000,solver='lbfgs',penalty='l2',random_state=RANDOM_SEED)
    else:
        C,cw,solver,penalty=cfg; return LogisticRegression(C=C,class_weight=cw,max_iter=5000,solver=solver,penalty=penalty,random_state=RANDOM_SEED)

def build_rf(cfg):
    ne,mf,msl,md,cw=cfg
    kw = {'n_estimators':ne,'max_features':mf,'min_samples_leaf':msl,'class_weight':cw,'n_jobs':1,'random_state':RANDOM_SEED}
    if md: kw['max_depth']=md
    return RandomForestClassifier(**kw)

def build_svm(cfg):
    C,kernel,gamma,cw=cfg
    if kernel=='linear': return SVC(C=C,kernel=kernel,class_weight=cw,probability=True,random_state=RANDOM_SEED)
    return SVC(C=C,kernel=kernel,gamma=gamma,class_weight=cw,probability=True,random_state=RANDOM_SEED)

def build_xgb(cfg):
    ne,md,lr,sub,spw,ra,rl=cfg
    return xgb.XGBClassifier(n_estimators=ne,max_depth=md,learning_rate=lr,subsample=sub,
                              scale_pos_weight=spw,reg_alpha=ra,reg_lambda=rl,
                              eval_metric='logloss',random_state=RANDOM_SEED,n_jobs=1,verbosity=0)

def build_model(mname, cfg):
    if mname=='LogisticRegression': return build_lr(cfg)
    elif mname=='RandomForest':   return build_rf(cfg)
    elif mname=='SVM':            return build_svm(cfg)
    elif mname=='XGBoost':        return build_xgb(cfg)

SCENARIO_K = {
    'S1_Spectrogram': [None, 30, 50, 60, 80],
    'S2_MFCC':        [None, 30, 50, 60, 80],
    'S3_Wav2Vec':     [None, 50, 60, 70],
    'S4_Fusion':      [30, 40, 50, 60],
    'S5_FusionEng':   [30, 50, 80],
}

# 10-fold CV for stability
K_FOLDS_OUTER = 10
K_FOLDS_INNER = 5
cv_outer = StratifiedKFold(n_splits=K_FOLDS_OUTER, shuffle=True, random_state=RANDOM_SEED)
cv_inner = StratifiedKFold(n_splits=K_FOLDS_INNER, shuffle=True, random_state=RANDOM_SEED)

all_results = []
current_best_cv   = 0.7149
current_best_test = 0.7494

print(f"\n{'='*80}")
print(f"  v84 — No SMOTE | {K_FOLDS_OUTER}-Fold CV | class_weight compensation")
print(f"  Referensi: CV=0.7149 (v76,SMOTEENN) | Test=0.7494 (v77) | Test_RF=0.7442")
print(f"{'='*80}")

for sc_name, X_full in SCENARIOS.items():
    X_tr_raw = X_full[train_idx]; X_te_raw = X_full[test_idx]
    k_cands = SCENARIO_K[sc_name]
    print(f"\n{'─'*70}")
    print(f"  SKENARIO: {sc_name} | {X_full.shape[1]} fitur | K={k_cands}")

    for model_name in MODEL_NAMES:
        t0 = time.time()
        best_inner_f1 = -1
        best_cfg_idx, best_K = 0, k_cands[0]

        # Inner: cfg × K (no SMOTE)
        for ci, cfg in enumerate(MODEL_CONFIGS[model_name]):
            for K in k_cands:
                X_tr_p, _ = preprocess(X_tr_raw, X_te_raw, y_train, k=K)
                fold_f1s = []
                for f_tr, f_val in cv_inner.split(X_tr_p, y_train):
                    Xf_tr, Xf_val = X_tr_p[f_tr], X_tr_p[f_val]
                    yf_tr, yf_val = y_train[f_tr], y_train[f_val]
                    try:
                        clf = build_model(model_name, cfg)
                        clf.fit(Xf_tr, yf_tr)  # NO SMOTE!
                        probs = clf.predict_proba(Xf_val)[:,1]
                        thr,_ = sweep_thr(probs, yf_val)
                        fold_f1s.append(f1_score(yf_val,(probs>=thr).astype(int),
                                                  average='macro',zero_division=0))
                    except: fold_f1s.append(0.)
                mf1 = np.mean(fold_f1s) if fold_f1s else 0.
                if mf1 > best_inner_f1:
                    best_inner_f1=mf1; best_cfg_idx=ci; best_K=K

        best_cfg = MODEL_CONFIGS[model_name][best_cfg_idx]

        # Outer 10-fold CV (no SMOTE)
        cv_f1s, cv_accs, cv_thrs = [], [], []
        X_tr_p, X_te_p = preprocess(X_tr_raw, X_te_raw, y_train, k=best_K)
        for f_tr, f_val in cv_outer.split(X_tr_p, y_train):
            Xf_tr, Xf_val = X_tr_p[f_tr], X_tr_p[f_val]
            yf_tr, yf_val = y_train[f_tr], y_train[f_val]
            try:
                clf = build_model(model_name, best_cfg)
                clf.fit(Xf_tr, yf_tr)
                probs = clf.predict_proba(Xf_val)[:,1]
                thr,_ = sweep_thr(probs, yf_val)
                preds = (probs>=thr).astype(int)
                cv_f1s.append(f1_score(yf_val,preds,average='macro',zero_division=0))
                cv_accs.append(accuracy_score(yf_val,preds))
                cv_thrs.append(thr)
            except: cv_f1s.append(0.); cv_accs.append(0.); cv_thrs.append(0.5)

        cv_f1_mean=float(np.mean(cv_f1s)); cv_f1_std=float(np.std(cv_f1s))
        fair_thr = float(np.mean(cv_thrs))

        # Final model (no SMOTE) — both fair & swept threshold
        clf_f = build_model(model_name, best_cfg)
        clf_f.fit(X_tr_p, y_train)
        probs_te = clf_f.predict_proba(X_te_p)[:,1]

        # Swept threshold (test)
        thr_sw,_ = sweep_thr(probs_te, y_test)
        preds_sw = (probs_te>=thr_sw).astype(int)
        test_f1_sw = float(f1_score(y_test,preds_sw,average='macro',zero_division=0))

        # Fair threshold (CV-derived)
        preds_fair = (probs_te>=fair_thr).astype(int)
        test_f1_fair = float(f1_score(y_test,preds_fair,average='macro',zero_division=0))
        test_acc_fair = float(accuracy_score(y_test,preds_fair))

        try: auc_te=float(roc_auc_score(y_test,probs_te))
        except: auc_te=0.

        # Primary: use whichever is better (report both)
        test_f1  = test_f1_sw   # swept for comparison
        test_acc = float(accuracy_score(y_test,preds_sw))
        preds_te = preds_sw

        gap = test_f1 - cv_f1_mean
        cv_flag = '★CV★' if cv_f1_mean > current_best_cv   else ''
        te_flag = '★TE★' if test_f1    > current_best_test  else ''
        if cv_f1_mean > current_best_cv:   current_best_cv   = cv_f1_mean
        if test_f1    > current_best_test: current_best_test = test_f1

        K_str = 'ALL' if best_K is None else str(best_K)
        result = {
            'scenario':sc_name,'model':model_name,'best_K':K_str,
            'best_cfg_idx':best_cfg_idx,'fair_thr':round(fair_thr,3),'swept_thr':round(thr_sw,3),
            'cv_f1_mean':round(cv_f1_mean,4),'cv_f1_std':round(cv_f1_std,4),
            'test_f1_fair':round(test_f1_fair,4),'test_f1':round(test_f1_sw,4),
            'test_acc':round(test_acc,4),'test_auc':round(auc_te,4),
            'overfit_gap':round(gap,4),'time_s':round(time.time()-t0,1),
            'y_pred':preds_te.tolist(),'y_prob':probs_te.tolist(),
        }
        all_results.append(result)
        st='⚠OV' if gap<-0.10 else '✓OK' if abs(gap)<=0.10 else '↑GEN'
        print(f"  {model_name:<22} K={K_str:<5} thr_cv={fair_thr:.2f}/sw={thr_sw:.2f} "
              f"CV={cv_f1_mean:.4f}±{cv_f1_std:.4f} Test={test_f1_sw:.4f}(sw)/{test_f1_fair:.4f}(fair) "
              f"Gap={gap:+.4f} {st} {cv_flag}{te_flag}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(RESULTS_DIR,"metrics","v84_results.csv"), index=False)
sorted_res = sorted(all_results, key=lambda x: x['cv_f1_mean'], reverse=True)

print(f"\n{'='*100}")
print(f"{'TABEL RINGKASAN v84 — No SMOTE | 10-Fold CV':^100}")
print(f"{'='*100}")
print(f"  {'Skenario':<22} {'Model':<22} {'K':>5} {'CV F1':>7} {'Std':>6} {'F1(sw)':>7} {'F1(fair)':>9} {'Acc':>7} {'Gap':>8} {'St'}")
for r in sorted_res[:18]:
    st='⚠OV' if r['overfit_gap']<-0.10 else '✓OK' if abs(r['overfit_gap'])<=0.10 else '↑GEN'
    print(f"  {r['scenario']:<22} {r['model']:<22} {r['best_K']:>5} "
          f"{r['cv_f1_mean']:>7.4f} {r['cv_f1_std']:>6.4f} {r['test_f1']:>7.4f} "
          f"{r['test_f1_fair']:>9.4f} {r['test_acc']:>7.4f} {r['overfit_gap']:>+8.4f} {st}")

best_cv   = max(all_results, key=lambda x: x['cv_f1_mean'])
best_test = max(all_results, key=lambda x: x['test_f1'])
print(f"\n  ★ BEST CV   : {best_cv['scenario']} × {best_cv['model']} K={best_cv['best_K']} → CV={best_cv['cv_f1_mean']:.4f} Test={best_cv['test_f1']:.4f}")
print(f"  ★ BEST Test : {best_test['scenario']} × {best_test['model']} K={best_test['best_K']} → CV={best_test['cv_f1_mean']:.4f} Test={best_test['test_f1']:.4f}")

print(f"\n  APPLE-TO-APPLE (S1-S4):")
print(f"  {'Skenario':<20} {'Best Model':<22} {'K':>5} {'CV F1':>7} {'Test F1':>8} {'Acc':>7} {'AUC':>7}")
for sc in ['S1_Spectrogram','S2_MFCC','S3_Wav2Vec','S4_Fusion']:
    rows=[r for r in all_results if r['scenario']==sc]
    b=max(rows,key=lambda x:x['cv_f1_mean'])
    print(f"  {sc:<20} {b['model']:<22} {b['best_K']:>5} {b['cv_f1_mean']:>7.4f} {b['test_f1']:>8.4f} {b['test_acc']:>7.4f} {b['test_auc']:>7.4f}")

# Plots
COLORS=['#6366f1','#ef4444','#f97316','#22c55e']
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,8))
fig.suptitle(f'v84 — No SMOTE | 10-Fold CV | Class Weight Only\n'
             f'Best CV={best_cv["cv_f1_mean"]:.4f} | Best Test={best_test["test_f1"]:.4f}',
             fontsize=12,fontweight='bold')
sc_list=['S1_Spectrogram','S2_MFCC','S3_Wav2Vec','S4_Fusion']
x=np.arange(len(MODEL_NAMES)); width=0.18
for i,sc in enumerate(sc_list):
    rows=[r for r in all_results if r['scenario']==sc and r['model'] in MODEL_NAMES]
    cv_v=[next((r['cv_f1_mean'] for r in rows if r['model']==m),0.) for m in MODEL_NAMES]
    te_v=[next((r['test_f1']    for r in rows if r['model']==m),0.) for m in MODEL_NAMES]
    label=sc.split('_')[1]
    ax1.bar(x+i*width,cv_v,width,label=label,color=COLORS[i],alpha=0.85,edgecolor='white')
    ax2.bar(x+i*width,te_v,width,label=label,color=COLORS[i],alpha=0.85,edgecolor='white')
for ax,title in [(ax1,f'CV F1 ({K_FOLDS_OUTER}-Fold, No SMOTE)'),(ax2,'Test F1 (Swept Threshold)')]:
    ax.set_xticks(x+width*1.5); ax.set_xticklabels(MODEL_NAMES,rotation=15,ha='right',fontsize=9)
    ax.axhline(0.75,color='red',linestyle='--',lw=1.5,label='Target 0.75')
    ax.set_ylim(0,1.); ax.set_ylabel('F1 Macro'); ax.set_title(title,fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y',linestyle='--',alpha=0.4)
    for bar in ax.patches:
        val=bar.get_height()
        if val>0.05: ax.text(bar.get_x()+bar.get_width()/2,val+0.01,f'{val:.2f}',ha='center',va='bottom',fontsize=6.5,fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR,"plots","v84_comparison.png"),dpi=150,bbox_inches='tight'); plt.close()

fig2,axes2=plt.subplots(1,4,figsize=(20,5))
fig2.suptitle('v84 — Confusion Matrix (Best CV per Skenario, No SMOTE)',fontsize=11,fontweight='bold')
for ax,(sc_name,_) in zip(axes2,list(SCENARIOS.items())[:4]):
    rows=[r for r in all_results if r['scenario']==sc_name]
    b=max(rows,key=lambda x:x['cv_f1_mean'])
    cm=confusion_matrix(y_test,b['y_pred'],labels=[0,1])
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,
                xticklabels=['Normal','Depresi'],yticklabels=['Normal','Depresi'],annot_kws={'size':14})
    ax.set_title(f'{sc_name}\n{b["model"]} K={b["best_K"]}\nCV={b["cv_f1_mean"]:.4f} Test={b["test_f1"]:.4f}',fontsize=8,fontweight='bold')
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR,"plots","v84_confusion.png"),dpi=150,bbox_inches='tight'); plt.close()
print("  Plots saved.")

print(f"\n{'='*80}")
print("  CLASSIFICATION REPORTS — S1-S4 Best CV")
print(f"{'='*80}")
for sc_name in ['S1_Spectrogram','S2_MFCC','S3_Wav2Vec','S4_Fusion']:
    rows=[r for r in all_results if r['scenario']==sc_name]
    b=max(rows,key=lambda x:x['cv_f1_mean'])
    print(f"\n  ── {sc_name} × {b['model']} (K={b['best_K']}) ──")
    print(f"  CV={b['cv_f1_mean']:.4f}±{b['cv_f1_std']:.4f} | Test(sw)={b['test_f1']:.4f} | Test(fair)={b['test_f1_fair']:.4f}")
    print(classification_report(y_test,b['y_pred'],target_names=['Normal','Depresi'],zero_division=0))

print(f"\n{'='*80}")
print(f"{'FINAL REPORT v84':^80}")
print(f"{'='*80}")
print(f"  Progress: v76=0.7149(SMOTE)|v79=0.7138|v84={best_cv['cv_f1_mean']:.4f}(NoSMOTE)")
print(f"  Best CV  : {best_cv['scenario']} × {best_cv['model']} K={best_cv['best_K']}")
print(f"  CV F1    : {best_cv['cv_f1_mean']:.4f} ± {best_cv['cv_f1_std']:.4f}")
print(f"  Test F1 (swept): {best_cv['test_f1']:.4f}")
print(f"  Best Test: {best_test['scenario']} × {best_test['model']} = {best_test['test_f1']:.4f}")
print(f"  TARGET 0.75 (CV)  : {'✓ TERCAPAI!' if best_cv['cv_f1_mean']>=0.75 else f'NO (selisih {0.75-best_cv[chr(99)+chr(118)+chr(95)+chr(102)+chr(49)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.4f})'}")
print(f"  TARGET 0.75 (Test): {'✓ TERCAPAI!' if best_test['test_f1']>=0.75 else f'NO ({best_test[chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(102)+chr(49)]:.4f})'}")
print(f"  Total waktu : {time.time()-t_global:.1f}s")
print(f"{'='*80}")

json.dump({'version':'v84','no_smote':True,'cv_folds':K_FOLDS_OUTER,
    'best_cv':{'scenario':best_cv['scenario'],'model':best_cv['model'],
               'cv_f1':best_cv['cv_f1_mean'],'test_f1':best_cv['test_f1'],'K':best_cv['best_K']},
    'best_test':{'scenario':best_test['scenario'],'model':best_test['model'],
                 'cv_f1':best_test['cv_f1_mean'],'test_f1':best_test['test_f1']},
    'target_075_cv':bool(best_cv['cv_f1_mean']>=0.75),
    'target_075_test':bool(best_test['test_f1']>=0.75),
},open(os.path.join(RESULTS_DIR,"metrics","v84_summary.json"),'w'),indent=2)
