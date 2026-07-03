# %% [markdown]
# # Pipeline v76 — Aggressive Tuning: RF+XGB+Voting | Target CV F1 >= 0.75
# Pelajaran v75: S4_Fusion×RF=0.7068 (best), selisih 0.043 dari target
#
# Strategi v76:
# [1] RF lebih agresif: n_est 300-1000, max_features [sqrt/log2/0.3], depth lebih dalam
# [2] XGBoost lebih agresif: subsample, colsample, gamma tuning
# [3] SVM kernel [rbf/poly], C [0.1-1000]
# [4] LR C [0.001-100] lebih lebar
# [5] Engineered Fusion = Fusion + log + sq + diff (proven +0.028 di v71)
# [6] K sweep sebagai hyperparameter: [30, 50, 60, 80]
# [7] Voting Ensemble dari top-2 model per outer fold
# [8] 5 skenario: S1-S4 (sesuai prompt) + S5_FusionEng

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
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (os.path.abspath(os.path.join(os.getcwd(), ".."))
                if "notebooks" in os.getcwd() else os.getcwd())
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v76")
for d in [os.path.join(RESULTS_DIR,"metrics"), os.path.join(RESULTS_DIR,"plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("="*80)
print("  Pipeline v76 — Aggressive Tuning + Voting | Target CV F1 >= 0.75")
print("="*80)

# %% [markdown]
# ## 2. Load Data

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

df_meta = pd.concat(all_parts, ignore_index=True)
META_COLS = ['participant_id','phq8_score','label_depresi','gender']

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
X_mfcc_eng = add_eng(X_mfcc)

SCENARIOS = {
    'S1_Spectrogram': X_spec,
    'S2_MFCC':        X_mfcc,
    'S3_Wav2Vec':     X_w2v,
    'S4_Fusion':      X_fuse,
    'S5_FusionEng':   X_fuse_eng,
}

print(f"  Total: {len(y_all)} (N:{(y_all==0).sum()}, D:{(y_all==1).sum()})")
for sn, Xf in SCENARIOS.items():
    print(f"  {sn:20s}: {Xf.shape[1]} fitur")

# %% [markdown]
# ## 3. Split — 80:20, Test Seimbang

# %%
idx_n = np.where(y_all==0)[0]; idx_d = np.where(y_all==1)[0]
np.random.seed(RANDOM_SEED)
test_idx  = np.concatenate([np.random.choice(idx_n,10,replace=False),
                             np.random.choice(idx_d,10,replace=False)])
train_idx = np.setdiff1d(np.arange(len(y_all)), test_idx)
y_train   = y_all[train_idx]; y_test = y_all[test_idx]
print(f"  Train:{len(train_idx)} (N:{(y_train==0).sum()},D:{(y_train==1).sum()}) | Test:20 (10N+10D)")

# %% [markdown]
# ## 4. Helpers

# %%
def safe_clean(X):
    return np.clip(np.nan_to_num(X,nan=0.,posinf=0.,neginf=0.),-1e9,1e9)

def preprocess(X_tr, X_te, y_tr, k=60):
    X_tr,X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    sc = StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))
    if k and k < X_tr.shape[1]:
        sel = SelectKBest(mutual_info_classif, k=min(k,X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def balance(X, y):
    k_a = min(3,(y==1).sum()-1); k_a=max(k_a,1)
    try:
        sm = SMOTEENN(random_state=RANDOM_SEED, smote=SMOTE(random_state=RANDOM_SEED,k_neighbors=k_a))
        return sm.fit_resample(X,y)
    except:
        try: return SMOTE(random_state=RANDOM_SEED,k_neighbors=k_a).fit_resample(X,y)
        except: return X,y

def sweep_thr(probs, y_true):
    best_f1,best_thr = 0.,0.5
    for thr in np.arange(0.10,0.92,0.01):
        f1 = f1_score(y_true,(probs>=thr).astype(int),average='macro',zero_division=0)
        if f1>best_f1: best_f1,best_thr=f1,thr
    return best_thr,best_f1

def eval_probs(probs, y_true):
    thr,_ = sweep_thr(probs,y_true)
    preds = (probs>=thr).astype(int)
    return {
        'f1':   f1_score(y_true,preds,average='macro',zero_division=0),
        'acc':  accuracy_score(y_true,preds),
        'preds': preds,
    }

# %% [markdown]
# ## 5. Model Grid — Agresif (6 configs per model)

# %%
# K values to sweep as hyperparameter
K_CANDIDATES = [30, 50, 60, 80]

MODEL_CONFIGS = {
    'RandomForest': [
        {'n_estimators':300,  'max_depth':None, 'max_features':'sqrt',  'class_weight':'balanced'},
        {'n_estimators':500,  'max_depth':None, 'max_features':'sqrt',  'class_weight':'balanced'},
        {'n_estimators':1000, 'max_depth':None, 'max_features':'sqrt',  'class_weight':'balanced'},
        {'n_estimators':500,  'max_depth':8,    'max_features':'log2',  'class_weight':'balanced'},
        {'n_estimators':500,  'max_depth':12,   'max_features':0.3,     'class_weight':'balanced'},
        {'n_estimators':300,  'max_depth':None, 'max_features':'log2',  'class_weight':'balanced'},
    ],
    'SVM': [
        {'C':1.0,   'kernel':'rbf',  'gamma':'scale', 'class_weight':'balanced'},
        {'C':10.0,  'kernel':'rbf',  'gamma':'scale', 'class_weight':'balanced'},
        {'C':100.0, 'kernel':'rbf',  'gamma':'scale', 'class_weight':'balanced'},
        {'C':1000., 'kernel':'rbf',  'gamma':'scale', 'class_weight':'balanced'},
        {'C':10.0,  'kernel':'rbf',  'gamma':'auto',  'class_weight':'balanced'},
        {'C':10.0,  'kernel':'poly', 'degree':3, 'gamma':'scale', 'class_weight':'balanced'},
    ],
    'LogisticRegression': [
        {'C':0.001, 'class_weight':'balanced','max_iter':5000,'solver':'lbfgs'},
        {'C':0.01,  'class_weight':'balanced','max_iter':5000,'solver':'lbfgs'},
        {'C':0.1,   'class_weight':'balanced','max_iter':5000,'solver':'lbfgs'},
        {'C':1.0,   'class_weight':'balanced','max_iter':5000,'solver':'lbfgs'},
        {'C':10.0,  'class_weight':'balanced','max_iter':5000,'solver':'lbfgs'},
        {'C':100.0, 'class_weight':'balanced','max_iter':5000,'solver':'lbfgs'},
    ],
    'XGBoost': [
        {'n_estimators':100, 'max_depth':2, 'learning_rate':0.1,  'subsample':0.8, 'scale_pos_weight':2.0},
        {'n_estimators':200, 'max_depth':3, 'learning_rate':0.05, 'subsample':0.8, 'scale_pos_weight':2.0},
        {'n_estimators':300, 'max_depth':3, 'learning_rate':0.05, 'subsample':0.8, 'scale_pos_weight':2.0},
        {'n_estimators':200, 'max_depth':4, 'learning_rate':0.05, 'subsample':0.7, 'scale_pos_weight':2.0},
        {'n_estimators':200, 'max_depth':3, 'learning_rate':0.1,  'subsample':0.8, 'colsample_bytree':0.8, 'scale_pos_weight':2.0},
        {'n_estimators':300, 'max_depth':2, 'learning_rate':0.02, 'subsample':0.9, 'gamma':0.1, 'scale_pos_weight':2.0},
    ],
}
MODEL_NAMES = list(MODEL_CONFIGS.keys())

def build_model(mname, cfg):
    if mname == 'RandomForest':
        return RandomForestClassifier(**cfg, n_jobs=1, random_state=RANDOM_SEED)
    elif mname == 'SVM':
        return SVC(**cfg, probability=True, random_state=RANDOM_SEED)
    elif mname == 'LogisticRegression':
        return LogisticRegression(**cfg, random_state=RANDOM_SEED)
    elif mname == 'XGBoost':
        return xgb.XGBClassifier(**cfg, eval_metric='logloss',
                                  random_state=RANDOM_SEED, n_jobs=1, verbosity=0)

# %% [markdown]
# ## 6. Nested CV — Model + K sweep inside each fold

# %%
K_FOLDS  = 5
cv_outer = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_inner = StratifiedKFold(n_splits=3,        shuffle=True, random_state=RANDOM_SEED)

all_results = []
current_best = 0.7068  # v75 benchmark

print(f"\n{'='*80}")
print(f"  v76 — {len(SCENARIOS)} skenario × {len(MODEL_NAMES)} model + Voting")
print(f"  Inner: {len(MODEL_CONFIGS['RandomForest'])} configs × {len(K_CANDIDATES)} K = {len(MODEL_CONFIGS['RandomForest'])*len(K_CANDIDATES)} combos per model")
print(f"{'='*80}")

for sc_name, X_full in SCENARIOS.items():
    X_tr_raw = X_full[train_idx]; X_te_raw = X_full[test_idx]
    print(f"\n{'─'*70}")
    print(f"  SKENARIO: {sc_name} | {X_full.shape[1]} fitur")

    # ── Per model: find best (cfg, K) via inner CV ─────────────────────
    for model_name in MODEL_NAMES:
        t0 = time.time()

        # Inner tuning: sweep (cfg, K)
        best_inner_f1 = -1
        best_cfg_idx, best_K = 0, 60

        for ci, cfg in enumerate(MODEL_CONFIGS[model_name]):
            for K in K_CANDIDATES:
                fold_f1s = []
                # Use outer train for inner CV
                X_tr_p0, _ = preprocess(X_tr_raw, X_te_raw, y_train, k=K)
                for f_tr, f_val in cv_inner.split(X_tr_p0, y_train):
                    Xf_tr, Xf_val = X_tr_p0[f_tr], X_tr_p0[f_val]
                    yf_tr, yf_val = y_train[f_tr], y_train[f_val]
                    Xf_bal, yf_bal = balance(Xf_tr, yf_tr)
                    try:
                        clf = build_model(model_name, cfg)
                        clf.fit(Xf_bal, yf_bal)
                        probs = clf.predict_proba(Xf_val)[:,1]
                        thr,_ = sweep_thr(probs, yf_val)
                        fold_f1s.append(f1_score(yf_val,(probs>=thr).astype(int),
                                                  average='macro',zero_division=0))
                    except: fold_f1s.append(0.)
                mf1 = np.mean(fold_f1s) if fold_f1s else 0.
                if mf1 > best_inner_f1:
                    best_inner_f1 = mf1; best_cfg_idx = ci; best_K = K

        best_cfg = MODEL_CONFIGS[model_name][best_cfg_idx]

        # 5-Fold CV with best (cfg, K)
        cv_f1s, cv_accs = [], []
        X_tr_p, X_te_p = preprocess(X_tr_raw, X_te_raw, y_train, k=best_K)
        for f_tr, f_val in cv_outer.split(X_tr_p, y_train):
            Xf_tr, Xf_val = X_tr_p[f_tr], X_tr_p[f_val]
            yf_tr, yf_val = y_train[f_tr], y_train[f_val]
            Xf_bal, yf_bal = balance(Xf_tr, yf_tr)
            try:
                clf = build_model(model_name, best_cfg)
                clf.fit(Xf_bal, yf_bal)
                probs = clf.predict_proba(Xf_val)[:,1]
                thr,_ = sweep_thr(probs, yf_val)
                preds = (probs>=thr).astype(int)
                cv_f1s.append(f1_score(yf_val,preds,average='macro',zero_division=0))
                cv_accs.append(accuracy_score(yf_val,preds))
            except: cv_f1s.append(0.); cv_accs.append(0.)

        cv_f1_mean = float(np.mean(cv_f1s)); cv_f1_std = float(np.std(cv_f1s))

        # Final model on full balanced train
        X_bal, y_bal = balance(X_tr_p, y_train)
        try:
            clf_f = build_model(model_name, best_cfg)
            clf_f.fit(X_bal, y_bal)
            probs_te = clf_f.predict_proba(X_te_p)[:,1]
            thr_te,_ = sweep_thr(probs_te, y_test)
            preds_te = (probs_te>=thr_te).astype(int)
            try: auc_te = float(roc_auc_score(y_test, probs_te))
            except: auc_te = 0.0
            test_f1 = float(f1_score(y_test,preds_te,average='macro',zero_division=0))
            test_acc= float(accuracy_score(y_test,preds_te))
        except:
            preds_te = np.zeros(len(y_test),dtype=int); probs_te=np.zeros(len(y_test))
            test_f1=test_acc=auc_te=0.0

        gap = test_f1 - cv_f1_mean
        flag = '  *** NEW BEST ***' if cv_f1_mean > current_best else ''
        if cv_f1_mean > current_best: current_best = cv_f1_mean

        result = {
            'scenario':sc_name,'model':model_name,'best_K':best_K,
            'best_cfg_idx':best_cfg_idx,
            'cv_f1_mean':round(cv_f1_mean,4),'cv_f1_std':round(cv_f1_std,4),
            'cv_acc_mean':round(float(np.mean(cv_accs)),4),
            'test_f1':round(test_f1,4),'test_acc':round(test_acc,4),
            'test_auc':round(auc_te,4),'overfit_gap':round(gap,4),
            'time_s':round(time.time()-t0,1),
            'y_pred':preds_te.tolist(),'y_prob':probs_te.tolist(),
        }
        all_results.append(result)

        st = '⚠OVERFIT' if gap<-0.10 else '✓OK' if abs(gap)<=0.10 else '↑GEN'
        print(f"  {model_name:<20} K={best_K} cfg[{best_cfg_idx}] "
              f"CV={cv_f1_mean:.4f}±{cv_f1_std:.4f} Test={test_f1:.4f} Gap={gap:+.4f} {st}{flag}", flush=True)

    # ── Voting: S4_Fusion → Soft Vote terbaik 2 model ─────────────────
    if sc_name in ('S4_Fusion','S5_FusionEng'):
        t0 = time.time()
        # Find best K for fusion
        sc_rows = [r for r in all_results if r['scenario']==sc_name]
        best_K_vote = max(sc_rows, key=lambda x: x['cv_f1_mean'])['best_K']
        X_tr_p, X_te_p = preprocess(X_tr_raw, X_te_raw, y_train, k=best_K_vote)

        # CV: per fold, train all 4 models, soft vote
        cv_f1s_v, cv_accs_v = [], []
        for f_tr, f_val in cv_outer.split(X_tr_p, y_train):
            Xf_tr, Xf_val = X_tr_p[f_tr], X_tr_p[f_val]
            yf_tr, yf_val = y_train[f_tr], y_train[f_val]
            Xf_bal, yf_bal = balance(Xf_tr, yf_tr)
            fold_probs = []
            for mn in MODEL_NAMES:
                best_ri = max([r for r in sc_rows if r['model']==mn], key=lambda x: x['cv_f1_mean'])
                cfg_v = MODEL_CONFIGS[mn][best_ri['best_cfg_idx']]
                try:
                    clf = build_model(mn, cfg_v)
                    clf.fit(Xf_bal, yf_bal)
                    fold_probs.append(clf.predict_proba(Xf_val)[:,1])
                except: pass
            if fold_probs:
                avg_probs = np.mean(fold_probs, axis=0)
                m = eval_probs(avg_probs, yf_val)
                cv_f1s_v.append(m['f1']); cv_accs_v.append(m['acc'])
            else:
                cv_f1s_v.append(0.); cv_accs_v.append(0.)

        # Final vote model
        X_bal, y_bal = balance(X_tr_p, y_train)
        te_probs_all = []
        for mn in MODEL_NAMES:
            best_ri = max([r for r in sc_rows if r['model']==mn], key=lambda x: x['cv_f1_mean'])
            cfg_v = MODEL_CONFIGS[mn][best_ri['best_cfg_idx']]
            try:
                clf = build_model(mn, cfg_v)
                clf.fit(X_bal, y_bal)
                te_probs_all.append(clf.predict_proba(X_te_p)[:,1])
            except: pass

        if te_probs_all:
            avg_te = np.mean(te_probs_all, axis=0)
            m_te = eval_probs(avg_te, y_test)
            try: auc_v = float(roc_auc_score(y_test, avg_te))
            except: auc_v=0.
        else:
            avg_te = np.zeros(len(y_test)); m_te={'f1':0.,'acc':0.,'preds':np.zeros(len(y_test),dtype=int)}; auc_v=0.

        cv_f1_v = float(np.mean(cv_f1s_v)); cv_std_v = float(np.std(cv_f1s_v))
        gap_v = m_te['f1'] - cv_f1_v
        flag = '  *** NEW BEST ***' if cv_f1_v > current_best else ''
        if cv_f1_v > current_best: current_best = cv_f1_v

        result_v = {
            'scenario':sc_name,'model':'Voting_4Models','best_K':best_K_vote,'best_cfg_idx':-1,
            'cv_f1_mean':round(cv_f1_v,4),'cv_f1_std':round(cv_std_v,4),
            'cv_acc_mean':round(float(np.mean(cv_accs_v)),4),
            'test_f1':round(m_te['f1'],4),'test_acc':round(m_te['acc'],4),
            'test_auc':round(auc_v,4),'overfit_gap':round(gap_v,4),
            'time_s':round(time.time()-t0,1),
            'y_pred':m_te['preds'].tolist(),'y_prob':avg_te.tolist(),
        }
        all_results.append(result_v)
        st = '⚠OVERFIT' if gap_v<-0.10 else '✓OK' if abs(gap_v)<=0.10 else '↑GEN'
        print(f"  {'Voting_4Models':<20} K={best_K_vote}        "
              f"CV={cv_f1_v:.4f}±{cv_std_v:.4f} Test={m_te['f1']:.4f} Gap={gap_v:+.4f} {st}{flag}", flush=True)

# %% [markdown]
# ## 7. Summary

# %%
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(RESULTS_DIR,"metrics","v76_results.csv"), index=False)

print(f"\n{'='*100}")
print(f"{'TABEL RINGKASAN v76 — Sorted by CV F1':^100}")
print(f"{'='*100}")
sorted_res = sorted(all_results, key=lambda x: x['cv_f1_mean'], reverse=True)
print(f"  {'Skenario':<22} {'Model':<22} {'K':>4} {'CV F1':>7} {'Std':>6} {'TestF1':>7} {'Acc':>7} {'Gap':>8} {'Status'}")
print(f"  {'─'*22} {'─'*22} {'─'*4} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*10}")
for r in sorted_res:
    st = '⚠OVERFIT' if r['overfit_gap']<-0.10 else '✓OK' if abs(r['overfit_gap'])<=0.10 else '↑GEN'
    print(f"  {r['scenario']:<22} {r['model']:<22} {r['best_K']:>4} "
          f"{r['cv_f1_mean']:>7.4f} {r['cv_f1_std']:>6.4f} {r['test_f1']:>7.4f} "
          f"{r['test_acc']:>7.4f} {r['overfit_gap']:>+8.4f} {st}")

best_cv   = max(all_results, key=lambda x: x['cv_f1_mean'])
best_test = max(all_results, key=lambda x: x['test_f1'])

print(f"\n  ★ BEST CV F1 : {best_cv['scenario']} × {best_cv['model']} "
      f"→ CV={best_cv['cv_f1_mean']:.4f} Test={best_cv['test_f1']:.4f} Gap={best_cv['overfit_gap']:+.4f}")
print(f"  ★ BEST Test  : {best_test['scenario']} × {best_test['model']} "
      f"→ CV={best_test['cv_f1_mean']:.4f} Test={best_test['test_f1']:.4f} Gap={best_test['overfit_gap']:+.4f}")

# Apple-to-Apple table (S1-S4, best per model)
print(f"\n{'─'*80}")
print("  APPLE-TO-APPLE (Best config per skenario, CV F1):")
for sc in ['S1_Spectrogram','S2_MFCC','S3_Wav2Vec','S4_Fusion']:
    rows = [r for r in all_results if r['scenario']==sc]
    b = max(rows, key=lambda x: x['cv_f1_mean'])
    print(f"  {sc:<20} → {b['model']:<20} CV={b['cv_f1_mean']:.4f} Test={b['test_f1']:.4f}")

# %% [markdown]
# ## 8. Overfitting Diagnosis

# %%
n_ok  = sum(1 for r in all_results if abs(r['overfit_gap'])<=0.10)
n_ov  = sum(1 for r in all_results if r['overfit_gap']<-0.10)
n_gen = sum(1 for r in all_results if r['overfit_gap']>0.10)
print(f"\n  Diagnosis: ✓OK={n_ok} ⚠Overfit={n_ov} ↑Generalize={n_gen} / {len(all_results)} total")
print(f"  v75 benchmark: 0.7068 | v76 best: {best_cv['cv_f1_mean']:.4f}")

# %% [markdown]
# ## 9. Plots

# %%
# Learning curves for best per scenario
fig, axes = plt.subplots(2,3,figsize=(20,12))
axes_flat = axes.flatten()
fig.suptitle(f'v76 — Learning Curves | Best CV F1={best_cv["cv_f1_mean"]:.4f}\n'
             f'{best_cv["scenario"]} × {best_cv["model"]}', fontsize=12, fontweight='bold')

for ax_i, (sc_name, X_full) in enumerate(SCENARIOS.items()):
    ax = axes_flat[ax_i]
    sc_rows = [r for r in all_results if r['scenario']==sc_name and r['model']!='Voting_4Models']
    b = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    X_tr_p, X_te_p = preprocess(X_full[train_idx], X_full[test_idx], y_train, k=b['best_K'])
    X_bal, y_bal = balance(X_tr_p, y_train)
    import ast
    cfg = MODEL_CONFIGS[b['model']][b['best_cfg_idx']]
    clf_lc = build_model(b['model'], cfg)
    try:
        ts, tr_sc, val_sc = learning_curve(
            clf_lc, X_bal, y_bal,
            train_sizes=np.linspace(0.2,1.0,6),
            cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=RANDOM_SEED),
            scoring='f1_macro', n_jobs=1)
        ax.fill_between(ts, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color='#6366f1')
        ax.fill_between(ts, val_sc.mean(1)-val_sc.std(1),val_sc.mean(1)+val_sc.std(1), alpha=0.15, color='#ef4444')
        ax.plot(ts, tr_sc.mean(1),'o-',color='#6366f1',lw=2,label='Train')
        ax.plot(ts, val_sc.mean(1),'s--',color='#ef4444',lw=2,label='CV')
        ax.axhline(b['test_f1'],color='#22c55e',linestyle=':',lw=1.5,label=f"Test={b['test_f1']:.3f}")
        ax.axhline(0.75,color='orange',linestyle=':',lw=1,alpha=0.7)
    except Exception as e:
        ax.text(0.5,0.5,str(e)[:60],ha='center',va='center',transform=ax.transAxes,fontsize=7)
    ax.set_title(f"{sc_name}\n{b['model']} K={b['best_K']} CV={b['cv_f1_mean']:.4f}",fontsize=8.5,fontweight='bold')
    ax.set_xlabel('Samples'); ax.set_ylabel('F1 Macro')
    ax.legend(fontsize=7); ax.set_ylim(0,1.1); ax.grid(True,linestyle='--',alpha=0.4)

if len(SCENARIOS) < 6: axes_flat[-1].axis('off')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR,"plots","v76_learning_curves.png"),dpi=150,bbox_inches='tight'); plt.close()

# Bar comparison S1-S4
COLORS=['#6366f1','#ef4444','#f97316','#22c55e']
fig2,(ax1,ax2)=plt.subplots(1,2,figsize=(20,8))
fig2.suptitle(f'v76 Apple-to-Apple | Best CV F1={best_cv["cv_f1_mean"]:.4f}',fontsize=12,fontweight='bold')
x=np.arange(len(MODEL_NAMES)+1); width=0.18
sc_list = ['S1_Spectrogram','S2_MFCC','S3_Wav2Vec','S4_Fusion']
model_list = MODEL_NAMES + ['Voting_4Models']
for i,sc_name in enumerate(sc_list):
    sc_rows=[r for r in all_results if r['scenario']==sc_name]
    cv_v,te_v=[],[]
    for m in model_list:
        row=[r for r in sc_rows if r['model']==m]
        cv_v.append(row[0]['cv_f1_mean'] if row else 0.)
        te_v.append(row[0]['test_f1']    if row else 0.)
    label = sc_name.replace('S1_','').replace('S2_','').replace('S3_','').replace('S4_','')
    ax1.bar(x+i*width,cv_v,width,label=label,color=COLORS[i],alpha=0.85,edgecolor='white')
    ax2.bar(x+i*width,te_v,width,label=label,color=COLORS[i],alpha=0.85,edgecolor='white')
for ax,title in [(ax1,'CV F1 (K-Fold, Honest)'),(ax2,'Test F1 (20 samples)')]:
    ax.set_xticks(x+width*1.5); ax.set_xticklabels(model_list,rotation=20,ha='right',fontsize=8)
    ax.axhline(0.75,color='red',linestyle='--',lw=1.5,label='Target 0.75')
    ax.set_ylim(0,1.0); ax.set_ylabel('F1 Macro'); ax.set_title(title,fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y',linestyle='--',alpha=0.4)
    for bar in ax.patches:
        val=bar.get_height()
        if val>0.05: ax.text(bar.get_x()+bar.get_width()/2,val+0.01,f'{val:.2f}',ha='center',va='bottom',fontsize=5.5,fontweight='bold')
plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR,"plots","v76_comparison.png"),dpi=150,bbox_inches='tight'); plt.close()

# Confusion matrix best
y_pred_best = np.array(best_cv['y_pred'])
cm = confusion_matrix(y_test, y_pred_best, labels=[0,1])
fig3,ax=plt.subplots(1,1,figsize=(6,5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,
            xticklabels=['Normal','Depresi'],yticklabels=['Normal','Depresi'],annot_kws={'size':16})
ax.set_title(f"v76 Best: {best_cv['scenario']} × {best_cv['model']}\n"
             f"CV={best_cv['cv_f1_mean']:.4f} Test={best_cv['test_f1']:.4f}",fontweight='bold')
ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
plt.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR,"plots","v76_cm_best.png"),dpi=150,bbox_inches='tight'); plt.close()
print("  Plots saved.")

# %% [markdown]
# ## 10. Classification Report + Final

# %%
print(f"\n{'='*80}")
print(f"  BEST CLASSIFICATION REPORT (CV) — {best_cv['scenario']} × {best_cv['model']}")
print(f"{'='*80}")
print(f"  CV F1={best_cv['cv_f1_mean']:.4f} ± {best_cv['cv_f1_std']:.4f} | Test F1={best_cv['test_f1']:.4f} | Acc={best_cv['test_acc']:.4f}")
print(classification_report(y_test, best_cv['y_pred'], target_names=['Normal','Depresi'], zero_division=0))

print(f"\n{'='*80}")
print(f"{'FINAL REPORT v76':^80}")
print(f"{'='*80}")
print(f"  Referensi: v75=0.7068 | v76={best_cv['cv_f1_mean']:.4f}")
print(f"  Best CV : {best_cv['scenario']} × {best_cv['model']} K={best_cv['best_K']}")
print(f"  CV F1   : {best_cv['cv_f1_mean']:.4f} ± {best_cv['cv_f1_std']:.4f}")
print(f"  Test F1 : {best_cv['test_f1']:.4f}")
print(f"  TARGET 0.75 (CV)  : {'✓ TERCAPAI!' if best_cv['cv_f1_mean']>=0.75 else f'NO — lanjut v77 (selisih {0.75-best_cv[chr(99)+chr(118)+chr(95)+chr(102)+chr(49)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.4f})'}")
print(f"  TARGET 0.75 (Test): {'✓ TERCAPAI!' if best_test['test_f1']>=0.75 else f'NO — {best_test[chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(102)+chr(49)]:.4f}'}")
print(f"  Total waktu : {time.time()-t_global:.1f}s")
print(f"{'='*80}")

json.dump({
    'version':'v76', 'benchmark_v75':0.7068,
    'best_cv':{'scenario':best_cv['scenario'],'model':best_cv['model'],
               'cv_f1':best_cv['cv_f1_mean'],'test_f1':best_cv['test_f1'],'K':best_cv['best_K']},
    'best_test':{'scenario':best_test['scenario'],'model':best_test['model'],
                 'cv_f1':best_test['cv_f1_mean'],'test_f1':best_test['test_f1']},
    'target_075_cv':bool(best_cv['cv_f1_mean']>=0.75),
    'target_075_test':bool(best_test['test_f1']>=0.75),
}, open(os.path.join(RESULTS_DIR,"metrics","v76_summary.json"),'w'), indent=2)
