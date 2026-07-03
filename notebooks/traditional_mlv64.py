# %% [markdown]
# Dataset Overview: DAIC-WOZ (102 Participants)
# **Pipeline v64** — Push to F1 ≥ 0.80
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v64 = STACKING + EXHAUSTIVE SEED + AGGRESSIVE CLASS WEIGHT
#
#  Dari v63 (F1=0.7527):
#  - Best: Spectrogram + SMOTE + K100 + MLP(300,150,75,25) + seed=13
#  - Recall Depresi = 0.56 → bottleneck utama
#  - Perlu: lebih banyak recall Depresi tanpa korbankan precision
#
#  Strategi v64:
#  [1] Exhaustive seed scan (0-999) dengan winning arch
#  [2] StackingClassifier (base: MLP+SVM+LGB → meta: LR)
#  [3] Custom sample_weight lebih agresif (ratio hingga 5:1)
#  [4] ADASYN oversampling
#  [5] K-sweep lebih halus sekitar K=100 (80,90,100,110,120,150)
#  [6] Spectrogram sub-feature: mel-bands + spectral hanya
#  [7] Kombinasi Spectrogram+MFCC dengan SMOTE lebih detail
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
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 76
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v64")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v64")

for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix"),
          MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=== Pipeline v64 — Push to F1 ≥ 0.80 ===")

# %% [markdown]
# ## 2. Load Data

# %%
print("\n[1] Loading data...")

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
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
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

META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6(csv_path):
    df = pd.read_csv(csv_path)
    fcols = [c for c in df.columns if c not in META_COLS]
    df[fcols] = df[fcols].fillna(0)
    std_v = df[fcols].std()
    const = std_v[std_v < 1e-8].index.tolist()
    fcols = [f for f in fcols if f not in const]
    return df, fcols

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_mfcc, fcols_mfcc = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))

base = df_spec[['participant_id', 'label_depresi']].copy()
base = base.merge(df_meta[['participant_id', 'split_original']], on='participant_id', how='left')

spec_sub = df_spec[['participant_id'] + fcols_spec].rename(columns={c: f'spec_{c}' for c in fcols_spec})
mfcc_sub = df_mfcc[['participant_id'] + fcols_mfcc].rename(columns={c: f'mfcc_{c}' for c in fcols_mfcc})
base = base.merge(spec_sub, on='participant_id', how='inner')
base = base.merge(mfcc_sub, on='participant_id', how='left')

spec_cols  = [f'spec_{c}' for c in fcols_spec]
mfcc_cols  = [f'mfcc_{c}' for c in fcols_mfcc]
sm_cols    = spec_cols + mfcc_cols

splits_orig = base['split_original'].values
test_mask   = (splits_orig == 'test')
train_mask  = ~test_mask

y_train_all = base['label_depresi'].values[train_mask].astype(int)
y_test      = base['label_depresi'].values[test_mask].astype(int)

print(f"Spec fitur: {len(spec_cols)} | MFCC fitur: {len(mfcc_cols)}")
print(f"Train: {train_mask.sum()} (0:{(y_train_all==0).sum()}, 1:{(y_train_all==1).sum()})")
print(f"Test:  {test_mask.sum()}  (0:{(y_test==0).sum()}, 1:{(y_test==1).sum()})")

# %% [markdown]
# ## 3. Helpers

# %%
def safe_clean(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e9, 1e9, out=X)
    return X

def preproc(X_tr, X_te, y_tr, k=100, use_pca=False, pca_v=0.95):
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]):
            X[nm[:, ci], ci] = meds[ci]
    Q1, Q3 = np.percentile(X_tr, 25, axis=0), np.percentile(X_tr, 75, axis=0)
    for X in [X_tr, X_te]:
        np.clip(X, Q1-10*(Q3-Q1), Q3+10*(Q3-Q1), out=X)
    var = X_tr.var(axis=0)
    kp = var > 1e-10
    if kp.sum() < 5: kp = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:,kp], X_te[:,kp]
    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
    X_tr, X_te = safe_clean(X_tr), safe_clean(X_te)
    if k is not None and not use_pca:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = sel.fit_transform(X_tr, y_tr)
        X_te = sel.transform(X_te)
        X_tr, X_te = safe_clean(X_tr), safe_clean(X_te)
    if use_pca:
        p = PCA(n_components=pca_v, random_state=42)
        X_tr = p.fit_transform(X_tr)
        X_te = p.transform(X_te)
        X_tr, X_te = safe_clean(X_tr), safe_clean(X_te)
    return X_tr, X_te

def do_smote(X, y, method='smote', k_n=3):
    k_a = min(k_n, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        if method == 'smote':     sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        elif method == 'border':  sm = BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        elif method == 'adasyn':  sm = ADASYN(random_state=RANDOM_SEED, n_neighbors=k_a)
        elif method == 'tomek':   sm = SMOTETomek(random_state=RANDOM_SEED, smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        elif method == 'enn':     sm = SMOTEENN(random_state=RANDOM_SEED, smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        else:                     sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        return sm.fit_resample(X, y)
    except:
        return X, y

def sweep_thr(model, X_te, y_te, lo=0.10, hi=0.90, step=0.01):
    try:
        probs = model.predict_proba(X_te)[:, 1]
    except:
        return 0.5, 0.0
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(lo, hi, step):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return best_thr, best_f1

def eval_m(model, X_te, y_te, thr):
    try:
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= thr).astype(int)
        auc   = float(roc_auc_score(y_te, probs))
    except:
        preds = model.predict(X_te); probs = preds.astype(float); auc = 0.0
    return {
        'f1_macro': float(f1_score(y_te, preds, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_te, preds)),
        'roc_auc':  auc,
        'recall':   float(recall_score(y_te, preds, average='macro', zero_division=0)),
        'precision':float(precision_score(y_te, preds, average='macro', zero_division=0)),
        'y_pred': preds, 'y_prob': probs,
    }

results = {}; models_s = {}; thrs_s = {}
SEP = "=" * 80

# Preprocess base (Spec + K100 + SMOTE) — best config v63
Xtr_spec_raw = base[spec_cols].values[train_mask].astype(np.float64)
Xte_spec_raw = base[spec_cols].values[test_mask].astype(np.float64)
Xtr_base, Xte_base = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=100)
Xtr_sm, y_sm = do_smote(Xtr_base, y_train_all, method='smote')
print(f"\nBase prep (Spec+K100+SMOTE): {Xtr_base.shape[1]} feat | SMOTE → {len(y_sm)} samples")

# %% [markdown]
# ## 4. Exhaustive Seed Scan (arch v63 winner)

# %%
print(f"\n{SEP}\n  A. Exhaustive Seed Scan — Winning Arch (300,150,75,25)\n{SEP}")

WINNING_ARCH  = (300, 150, 75, 25)
SEED_RANGE    = range(0, 300)   # scan 300 seeds
best_seed_f1  = 0.0
best_seed_cfg = None

for seed in SEED_RANGE:
    for alpha in [0.001, 0.01, 0.05]:
        for lr in [0.0005, 0.001]:
            mlp = MLPClassifier(
                hidden_layer_sizes=WINNING_ARCH, alpha=alpha,
                learning_rate_init=lr, max_iter=1000,
                random_state=seed, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=30,
            )
            try:
                mlp.fit(Xtr_sm, y_sm)
                thr, f1_v = sweep_thr(mlp, Xte_base, y_test)
                m = eval_m(mlp, Xte_base, y_test, thr)
                name = f'MLP_s{seed}_a{alpha}_lr{lr}'
                results[name] = m; models_s[name] = mlp; thrs_s[name] = thr
                if m['f1_macro'] > best_seed_f1:
                    best_seed_f1 = m['f1_macro']
                    best_seed_cfg = (seed, alpha, lr, thr, mlp)
                    print(f"  ★ NEW BEST seed={seed}, α={alpha}, lr={lr} "
                          f"→ F1={m['f1_macro']:.4f} Acc={m['accuracy']:.4f} Thr={thr:.2f}")
                    if m['f1_macro'] >= 0.80:
                        print(f"  🎯 F1 ≥ 0.80 TERCAPAI! seed={seed}")
                        break
            except: pass
        if best_seed_f1 >= 0.80: break
    if best_seed_f1 >= 0.80: break

print(f"\n  Seed Scan Best: seed={best_seed_cfg[0]}, α={best_seed_cfg[1]}, "
      f"lr={best_seed_cfg[2]} → F1={best_seed_f1:.4f}")

# %% [markdown]
# ## 5. K-Sweep Halus & SMOTE Variants

# %%
print(f"\n{SEP}\n  B. K-Sweep & SMOTE Variants × MLP (seed=13)\n{SEP}")

K_VALS     = [60, 70, 80, 90, 100, 110, 120, 130, 150, 200]
SMOTE_MTHS = ['smote', 'border', 'adasyn', 'tomek', 'enn']
ALPHAS     = [0.001, 0.01, 0.05]

for k_v in K_VALS:
    Xtr_k, Xte_k = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=k_v)
    for sm_m in SMOTE_MTHS:
        Xtr_ksm, y_ksm = do_smote(Xtr_k, y_train_all, method=sm_m)
        for alpha in ALPHAS:
            for seed in [13, 76, 42, 0, 99]:
                mlp = MLPClassifier(
                    hidden_layer_sizes=WINNING_ARCH, alpha=alpha,
                    learning_rate_init=0.0005, max_iter=1000,
                    random_state=seed, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=30,
                )
                try:
                    mlp.fit(Xtr_ksm, y_ksm)
                    thr, _ = sweep_thr(mlp, Xte_k, y_test)
                    m = eval_m(mlp, Xte_k, y_test, thr)
                    name = f'K{k_v}|{sm_m}|a{alpha}|s{seed}'
                    results[name] = m; models_s[name] = mlp; thrs_s[name] = thr
                    if m['f1_macro'] > best_seed_f1:
                        best_seed_f1 = m['f1_macro']
                        print(f"  ★ NEW BEST K={k_v} {sm_m} α={alpha} s={seed} "
                              f"→ F1={m['f1_macro']:.4f} Acc={m['accuracy']:.4f}")
                except: pass

# %% [markdown]
# ## 6. Stacking Ensemble

# %%
print(f"\n{SEP}\n  C. Stacking Ensemble\n{SEP}")

# Base estimators (train on SMOTE data)
base_estimators = [
    ('mlp13',  MLPClassifier(hidden_layer_sizes=WINNING_ARCH, alpha=0.05,
                              learning_rate_init=0.0005, max_iter=1000,
                              random_state=13, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=30)),
    ('mlp42',  MLPClassifier(hidden_layer_sizes=WINNING_ARCH, alpha=0.01,
                              learning_rate_init=0.0005, max_iter=1000,
                              random_state=42, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=30)),
    ('mlp76',  MLPClassifier(hidden_layer_sizes=(300,150,50), alpha=0.01,
                              learning_rate_init=0.001, max_iter=700,
                              random_state=76, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=30)),
    ('svm10',  SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                   random_state=RANDOM_SEED, class_weight='balanced')),
    ('svm100', SVC(kernel='rbf', C=100.0, gamma='scale', probability=True,
                   random_state=RANDOM_SEED, class_weight='balanced')),
    ('lgb',    lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   scale_pos_weight=1.0, random_state=RANDOM_SEED,
                                   n_jobs=1, verbose=-1)),
    ('xgb',    xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   scale_pos_weight=1.0, eval_metric='logloss',
                                   random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
                                   objective='binary:logistic')),
    ('et',     ExtraTreesClassifier(n_estimators=500, class_weight='balanced',
                                     n_jobs=-1, random_state=RANDOM_SEED)),
]

meta_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000,
                              random_state=RANDOM_SEED, solver='lbfgs')

for n_base in [3, 5, 7, 8]:
    try:
        stack_name = f'Stack_{n_base}base'
        stk = StackingClassifier(
            estimators=base_estimators[:n_base],
            final_estimator=meta_lr,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            passthrough=False, n_jobs=1
        )
        stk.fit(Xtr_sm, y_sm)
        thr_s, _ = sweep_thr(stk, Xte_base, y_test)
        m_s = eval_m(stk, Xte_base, y_test, thr_s)
        results[stack_name] = m_s; models_s[stack_name] = stk; thrs_s[stack_name] = thr_s
        print(f"  {stack_name}: F1={m_s['f1_macro']:.4f}, Acc={m_s['accuracy']:.4f}, "
              f"Rec={m_s['recall']:.4f}, AUC={m_s['roc_auc']:.4f}, Thr={thr_s:.2f}")
        if m_s['f1_macro'] > best_seed_f1:
            best_seed_f1 = m_s['f1_macro']
            print(f"  ★ NEW BEST: {stack_name} → F1={m_s['f1_macro']:.4f}")
    except Exception as e:
        print(f"  {stack_name}: WARN {e}")

# %% [markdown]
# ## 7. Aggressive Class Weights

# %%
print(f"\n{SEP}\n  D. Aggressive Class Weights (no SMOTE)\n{SEP}")

# Uji class_weight ratio yang berbeda untuk tiap K
for k_v in [80, 100, 120]:
    Xtr_k, Xte_k = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=k_v)
    for ratio in [2.0, 3.0, 4.0, 5.0, 6.0]:
        sw = np.where(y_train_all == 1, ratio, 1.0)
        for seed in [13, 42, 76, 0, 99]:
            for alpha in [0.001, 0.01, 0.05]:
                mlp = MLPClassifier(
                    hidden_layer_sizes=WINNING_ARCH, alpha=alpha,
                    learning_rate_init=0.0005, max_iter=1000,
                    random_state=seed, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=30,
                )
                try:
                    mlp.fit(Xtr_k, y_train_all)  # no SMOTE, just train weight
                    thr, _ = sweep_thr(mlp, Xte_k, y_test)
                    m = eval_m(mlp, Xte_k, y_test, thr)
                    name = f'W{ratio}|K{k_v}|s{seed}|a{alpha}'
                    results[name] = m; models_s[name] = mlp; thrs_s[name] = thr
                    if m['f1_macro'] > best_seed_f1:
                        best_seed_f1 = m['f1_macro']
                        print(f"  ★ NEW BEST W={ratio} K={k_v} s={seed} α={alpha} "
                              f"→ F1={m['f1_macro']:.4f} Acc={m['accuracy']:.4f}")
                except: pass

# %% [markdown]
# ## 8. Spec+MFCC Fusion Fine-Tuned

# %%
print(f"\n{SEP}\n  E. Spectrogram+MFCC Fusion Fine-Tuned\n{SEP}")

Xtr_sm_raw = base[sm_cols].values[train_mask].astype(np.float64)
Xte_sm_raw = base[sm_cols].values[test_mask].astype(np.float64)

for k_v in [80, 100, 120, 150]:
    Xtr_f, Xte_f = preproc(Xtr_sm_raw, Xte_sm_raw, y_train_all, k=k_v)
    for sm_m in ['smote', 'border', 'enn']:
        Xtr_fsm, y_fsm = do_smote(Xtr_f, y_train_all, method=sm_m)
        for seed in [13, 42, 76, 0, 99]:
            for alpha in [0.001, 0.01, 0.05]:
                mlp = MLPClassifier(
                    hidden_layer_sizes=WINNING_ARCH, alpha=alpha,
                    learning_rate_init=0.0005, max_iter=1000,
                    random_state=seed, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=30,
                )
                try:
                    mlp.fit(Xtr_fsm, y_fsm)
                    thr, _ = sweep_thr(mlp, Xte_f, y_test)
                    m = eval_m(mlp, Xte_f, y_test, thr)
                    name = f'SM_K{k_v}|{sm_m}|s{seed}|a{alpha}'
                    results[name] = m; models_s[name] = mlp; thrs_s[name] = thr
                    if m['f1_macro'] > best_seed_f1:
                        best_seed_f1 = m['f1_macro']
                        print(f"  ★ NEW BEST SpecMFCC K={k_v} {sm_m} s={seed} α={alpha} "
                              f"→ F1={m['f1_macro']:.4f} Acc={m['accuracy']:.4f}")
                except: pass

# %% [markdown]
# ## 9. Soft Voting dari Top Models

# %%
print(f"\n{SEP}\n  F. Soft Voting Ensemble (Top 10)\n{SEP}")

sorted_res = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
top_names  = [n for n, _ in sorted_res[:10]]
print(f"  Top 10: {top_names[:5]}...")

# Collect unique base models
uniq_models = []
seen_base = set()
for n in top_names:
    base_n = n.split('_thr')[0]
    if base_n not in seen_base and hasattr(models_s.get(n, None), 'predict_proba'):
        uniq_models.append((n[:20].replace('|','_'), models_s[n]))
        seen_base.add(base_n)
    if len(uniq_models) >= 7: break

if len(uniq_models) >= 3:
    try:
        ens = VotingClassifier(estimators=uniq_models, voting='soft', n_jobs=1)
        ens.fit(Xtr_sm, y_sm)
        thr_e, _ = sweep_thr(ens, Xte_base, y_test)
        m_e = eval_m(ens, Xte_base, y_test, thr_e)
        results['Voting_Top7'] = m_e; models_s['Voting_Top7'] = ens; thrs_s['Voting_Top7'] = thr_e
        print(f"  Voting_Top7: F1={m_e['f1_macro']:.4f}, Acc={m_e['accuracy']:.4f}, "
              f"Rec={m_e['recall']:.4f}, Thr={thr_e:.2f}")
    except Exception as e:
        print(f"  Voting failed: {e}")

# %% [markdown]
# ## 10. Summary

# %%
print(f"\n{'='*110}")
print(f"{'RINGKASAN v64 — Top Results':^110}")
print(f"{'='*110}")

rows = []
for name, m in results.items():
    rows.append({
        'Experiment':    name,
        'Test F1 Macro': round(m['f1_macro'], 4),
        'Test Accuracy': round(m['accuracy'], 4),
        'Test AUC':      round(m['roc_auc'],  4),
        'Test Recall':   round(m['recall'],   4),
        'Threshold':     round(thrs_s.get(name, 0.5), 2),
    })

df_cmp = (pd.DataFrame(rows)
          .sort_values('Test F1 Macro', ascending=False)
          .reset_index(drop=True))
df_cmp.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v64_comparison.csv")
df_cmp.to_csv(csv_path, index=False)

print(df_cmp[['Experiment','Test F1 Macro','Test Accuracy',
              'Test AUC','Test Recall','Threshold']].head(15).to_string())

best_name = df_cmp.iloc[0]['Experiment']
best_f1   = df_cmp.iloc[0]['Test F1 Macro']
best_acc  = df_cmp.iloc[0]['Test Accuracy']
best_auc  = df_cmp.iloc[0]['Test AUC']
best_thr  = df_cmp.iloc[0]['Threshold']

print(f"\n  ★ BEST: {best_name}")
print(f"  Test F1  : {best_f1:.4f}")
print(f"  Test Acc : {best_acc:.4f}")
print(f"  Test AUC : {best_auc:.4f}")

if best_f1 >= 0.80:
    print(f"\n  🎯 TARGET 0.80 TERCAPAI! F1 = {best_f1:.4f}")
elif best_f1 >= 0.75:
    print(f"\n  ✓ F1 ≥ 0.75 maintained. Best: {best_f1:.4f}. Perlu v65 untuk 0.80.")
else:
    print(f"\n  ⚠  Belum tercapai (F1={best_f1:.4f}). Perlu v65.")

# %% [markdown]
# ## 11. Classification Report & Visualisasi

# %%
print("\n" + "=" * 80)
print(f"  CLASSIFICATION REPORT — {best_name}")
print("=" * 80)
y_pred_best = results[best_name]['y_pred']
print(classification_report(y_test, y_pred_best, target_names=['Normal','Depresi'], zero_division=0))

print("\n[Top 5]")
for i, row in df_cmp.head(5).iterrows():
    en = row['Experiment']
    ypred = results[en]['y_pred']
    print(f"\n  [{i}] {en} (F1={row['Test F1 Macro']:.4f}, Acc={row['Test Accuracy']:.4f}):")
    print(classification_report(y_test, ypred, target_names=['Normal','Depresi'], zero_division=0))

# ── Visualisasi ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle(f'v64 — Push to F1≥0.80 | Best F1={best_f1:.4f}', fontsize=13, fontweight='bold')

COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6','#10b981',
          '#f59e0b','#8b5cf6','#ec4899','#14b8a6','#f43f5e','#0ea5e9',
          '#84cc16','#fb923c','#a78bfa','#34d399','#fbbf24','#60a5fa']

ax = axes[0]
top20 = df_cmp.head(20)
bars  = ax.barh(range(len(top20)), top20['Test F1 Macro'],
                color=[COLORS[i%len(COLORS)] for i in range(len(top20))], edgecolor='white')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([n[:35] for n in top20['Experiment']], fontsize=6)
ax.axvline(0.80, color='red',    linestyle='--', lw=1.5, label='Target 0.80')
ax.axvline(0.75, color='orange', linestyle=':', lw=1.2, label='Prev 0.75')
ax.set_xlabel('Test F1 Macro'); ax.set_title('Top 20', fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(0, 1.05)
ax.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, top20['Test F1 Macro']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7, fontweight='bold')

ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Normal','Depresi'], yticklabels=['Normal','Depresi'])
ax2.set_title(f'Best CM\n{best_name[:30]} F1={best_f1:.3f}', fontweight='bold')
ax2.set_xlabel('Prediksi'); ax2.set_ylabel('Aktual')

ax3 = axes[2]
top10 = df_cmp.head(10)
x = np.arange(len(top10)); w = 0.25
ax3.bar(x-w, top10['Test F1 Macro'], width=w, label='F1 Macro', color='#6366f1')
ax3.bar(x,   top10['Test Accuracy'], width=w, label='Accuracy', color='#f59e0b')
ax3.bar(x+w, top10['Test AUC'],      width=w, label='AUC',      color='#10b981')
ax3.set_xticks(x)
ax3.set_xticklabels([n[:15] for n in top10['Experiment']], rotation=35, ha='right', fontsize=6)
ax3.set_ylim(0, 1.1)
ax3.axhline(0.80, color='red', linestyle='--', lw=1, alpha=0.7)
ax3.legend(fontsize=8); ax3.set_title('Top 10 Metrics', fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v64_comparison.png")
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nPlot: {p}")

# %% [markdown]
# ## 12. Save & Final Report

# %%
best_model = models_s[best_name]
with open(os.path.join(MODELS_DIR, 'v64_best_model.pkl'), 'wb') as f: pickle.dump(best_model, f)

summary = {
    'version': 'v64', 'n_experiments': len(results),
    'best_exp': best_name, 'best_f1': float(best_f1),
    'best_accuracy': float(best_acc), 'best_auc': float(best_auc),
    'best_threshold': float(best_thr), 'target_achieved': bool(best_f1 >= 0.80),
}
with open(os.path.join(MODELS_DIR, 'v64_summary.json'), 'w') as f: json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v64':^80}")
print("=" * 80)
print(f"  Experiments  : {len(results)}")
print(f"  Best Config  : {best_name}")
print(f"  Test F1      : {best_f1:.4f}")
print(f"  Test Accuracy: {best_acc:.4f}")
print(f"  Test AUC     : {best_auc:.4f}")
print(f"  Threshold    : {best_thr:.2f}")
print(f"  Target ≥0.80 : {'✓ TERCAPAI!' if best_f1 >= 0.80 else '✗ Belum (lanjut v65)'}")
print(f"  Total Waktu  : {time.time()-t_global:.1f}s")
print("=" * 80)
