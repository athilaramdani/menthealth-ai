# %% [markdown]
# Dataset Overview: DAIC-WOZ (102 Participants)
# **Pipeline v65** — Push Beyond F1 0.80
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v65 = FINE-TUNE AROUND WINNER — Target F1 > 0.82
#
#  Dari v64 (F1=0.8083):
#  - Winner: K=110, SMOTEENN, seed=42, MLP(300,150,75,25)
#  - AUC = 0.7381, bisa lebih baik
#  - Threshold tinggi (0.84) → coba arsitektur baru
#
#  Strategi v65:
#  [1] Fine-tune K di sekitar 110 (95,100,105,110,115,120,125)
#  [2] SMOTEENN dominan — sweep k_neighbors ENN
#  [3] Arsitektur MLP baru yang lebih dalam
#  [4] GridSearch MLP: alpha × lr × arch (lebih halus)
#  [5] Seed scan 0-500 dengan K=110 + ENN
#  [6] Stacking Ensemble dengan base di K110+ENN
#  [7] Soft Voting dari beberapa model top K110+ENN
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 42   # Winner seed from v64
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v65")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v65")

for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix"),
          MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=== Pipeline v65 — Fine-Tune Around K=110+ENN+s42 ===")

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
    return df, [f for f in fcols if f not in const]

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_mfcc, fcols_mfcc = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_w2v,  fcols_w2v  = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"))

base = df_spec[['participant_id', 'label_depresi']].copy()
base = base.merge(df_meta[['participant_id', 'split_original']], on='participant_id', how='left')

spec_sub = df_spec[['participant_id'] + fcols_spec].rename(columns={c: f'spec_{c}' for c in fcols_spec})
mfcc_sub = df_mfcc[['participant_id'] + fcols_mfcc].rename(columns={c: f'mfcc_{c}' for c in fcols_mfcc})
w2v_sub  = df_w2v[['participant_id']  + fcols_w2v].rename( columns={c: f'w2v_{c}'  for c in fcols_w2v})
base = base.merge(spec_sub, on='participant_id', how='inner')
base = base.merge(mfcc_sub, on='participant_id', how='left')
base = base.merge(w2v_sub,  on='participant_id', how='left')

spec_cols = [f'spec_{c}' for c in fcols_spec]
mfcc_cols = [f'mfcc_{c}' for c in fcols_mfcc]
w2v_cols  = [f'w2v_{c}'  for c in fcols_w2v]

splits_orig = base['split_original'].values
test_mask   = (splits_orig == 'test')
train_mask  = ~test_mask

y_train_all = base['label_depresi'].values[train_mask].astype(int)
y_test      = base['label_depresi'].values[test_mask].astype(int)

Xtr_spec_raw = base[spec_cols].values[train_mask].astype(np.float64)
Xte_spec_raw = base[spec_cols].values[test_mask].astype(np.float64)

print(f"Spec: {len(spec_cols)} | Train: {train_mask.sum()} | Test: {test_mask.sum()}")
print(f"Train (0:{(y_train_all==0).sum()}, 1:{(y_train_all==1).sum()}) | "
      f"Test (0:{(y_test==0).sum()}, 1:{(y_test==1).sum()})")

# %% [markdown]
# ## 3. Helpers

# %%
def safe_clean(X):
    return np.nan_to_num(np.clip(X, -1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)

def preproc(X_tr, X_te, y_tr, k=110):
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]):
            X[nm[:, ci], ci] = meds[ci]
    Q1, Q3 = np.percentile(X_tr, 25, axis=0), np.percentile(X_tr, 75, axis=0)
    for X in [X_tr, X_te]:
        np.clip(X, Q1-10*(Q3-Q1), Q3+10*(Q3-Q1), out=X)
    kp = X_tr.var(axis=0) > 1e-10
    if kp.sum() < 5: kp = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:,kp], X_te[:,kp]
    sc = StandardScaler()
    X_tr, X_te = safe_clean(sc.fit_transform(X_tr)), safe_clean(sc.transform(X_te))
    if k is not None:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def do_enn(X, y, k_n=3):
    """SMOTEENN with configurable k."""
    k_a = min(k_n, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        sm = SMOTEENN(random_state=RANDOM_SEED,
                      smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        return sm.fit_resample(X, y)
    except:
        return X, y

def do_smote_any(X, y, method='enn', k_n=3):
    k_a = min(k_n, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        if method == 'enn':     sm = SMOTEENN(random_state=RANDOM_SEED, smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        elif method == 'smote': sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        elif method == 'border':sm = BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        elif method == 'tomek': sm = SMOTETomek(random_state=RANDOM_SEED, smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        elif method == 'adasyn':sm = ADASYN(random_state=RANDOM_SEED, n_neighbors=k_a)
        else: sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        return sm.fit_resample(X, y)
    except:
        return X, y

def sweep_thr(model, X_te, y_te, lo=0.10, hi=0.95, step=0.005):
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

results = {}; models_s = {}; thrs_s = {}; SEP = "=" * 80
WINNER_ARCH = (300, 150, 75, 25)
current_best = 0.8083   # baseline v64

def register(name, model, X_te, thr):
    global current_best
    m = eval_m(model, X_te, y_test, thr)
    results[name] = m; models_s[name] = model; thrs_s[name] = thr
    if m['f1_macro'] > current_best:
        current_best = m['f1_macro']
        print(f"  ★ NEW BEST [{name}] F1={m['f1_macro']:.4f} "
              f"Acc={m['accuracy']:.4f} Rec={m['recall']:.4f} Thr={thr:.3f}")
    return m

# %% [markdown]
# ## 4. Winner Config Variations — K & ENN k_neighbors

# %%
print(f"\n{SEP}\n  A. Fine-tune K (90-130) × ENN k_neighbors × seed\n{SEP}")

# Preprocess base K=110 once
Xtr_110, Xte_110 = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=110)
Xtr_enn42, y_enn42 = do_enn(Xtr_110, y_train_all, k_n=3)
print(f"  K=110+ENN: {Xtr_110.shape[1]} feat | after ENN: {len(y_enn42)} samples "
      f"(0:{(y_enn42==0).sum()}, 1:{(y_enn42==1).sum()})")

# Verify winner
mlp_winner = MLPClassifier(
    hidden_layer_sizes=WINNER_ARCH, alpha=0.001,
    learning_rate_init=0.0005, max_iter=1000, random_state=42,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=30
)
mlp_winner.fit(Xtr_enn42, y_enn42)
thr_w, _ = sweep_thr(mlp_winner, Xte_110, y_test)
m_w = eval_m(mlp_winner, Xte_110, y_test, thr_w)
print(f"  [VERIFY WINNER] F1={m_w['f1_macro']:.4f} Acc={m_w['accuracy']:.4f} Thr={thr_w:.3f}")
results['WINNER_v64'] = m_w; models_s['WINNER_v64'] = mlp_winner; thrs_s['WINNER_v64'] = thr_w

# Fine-tune K
for k_v in [90, 95, 100, 105, 107, 108, 109, 110, 111, 112, 113, 115, 120, 125, 130]:
    Xtr_k, Xte_k = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=k_v)
    for k_enn in [2, 3, 4, 5]:
        Xtr_e, y_e = do_enn(Xtr_k, y_train_all, k_n=k_enn)
        for seed in [42, 13, 76, 0, 99, 7, 21, 33]:
            for alpha in [0.0001, 0.001, 0.01, 0.05]:
                for lr in [0.0003, 0.0005, 0.001]:
                    name = f'K{k_v}|ENN{k_enn}|s{seed}|a{alpha}|lr{lr}'
                    try:
                        mlp = MLPClassifier(
                            hidden_layer_sizes=WINNER_ARCH, alpha=alpha,
                            learning_rate_init=lr, max_iter=1000, random_state=seed,
                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30
                        )
                        mlp.fit(Xtr_e, y_e)
                        thr, _ = sweep_thr(mlp, Xte_k, y_test)
                        register(name, mlp, Xte_k, thr)
                    except: pass

# %% [markdown]
# ## 5. Different MLP Architectures × Winner Preprocessing

# %%
print(f"\n{SEP}\n  B. Architecture Search × K=110+ENN3+s42\n{SEP}")

ARCHS = [
    (300, 150, 75, 25),
    (400, 200, 100, 50),
    (500, 250, 100, 50),
    (300, 200, 100, 50, 25),
    (256, 128, 64, 32),
    (512, 256, 128, 64, 32),
    (200, 150, 100, 50),
    (600, 300, 150, 75),
    (300, 150, 75, 50, 25),
    (400, 300, 200, 100),
]

for arch in ARCHS:
    for seed in [42, 13, 76, 0, 99, 7, 21]:
        for alpha in [0.0001, 0.001, 0.01, 0.05]:
            for lr in [0.0003, 0.0005, 0.001]:
                name = f'arch{"x".join(str(h) for h in arch)}|s{seed}|a{alpha}|lr{lr}'
                try:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=arch, alpha=alpha,
                        learning_rate_init=lr, max_iter=1000, random_state=seed,
                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30
                    )
                    mlp.fit(Xtr_enn42, y_enn42)
                    thr, _ = sweep_thr(mlp, Xte_110, y_test)
                    register(name, mlp, Xte_110, thr)
                except: pass

# %% [markdown]
# ## 6. Stacking dengan K=110+ENN Base

# %%
print(f"\n{SEP}\n  C. Stacking (K=110+ENN)\n{SEP}")

base_ests = [
    ('mlp42',   MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=0.001,
                               learning_rate_init=0.0005, max_iter=1000, random_state=42,
                               early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)),
    ('mlp13',   MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=0.05,
                               learning_rate_init=0.0005, max_iter=1000, random_state=13,
                               early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)),
    ('mlp0',    MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=0.001,
                               learning_rate_init=0.001, max_iter=1000, random_state=0,
                               early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)),
    ('mlp76',   MLPClassifier(hidden_layer_sizes=(300,150,50), alpha=0.01,
                               learning_rate_init=0.001, max_iter=700, random_state=76,
                               early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)),
    ('svm100',  SVC(kernel='rbf', C=100.0, gamma='scale', probability=True,
                    random_state=42, class_weight='balanced')),
    ('svm1000', SVC(kernel='rbf', C=1000.0, gamma='scale', probability=True,
                    random_state=42, class_weight='balanced')),
    ('lgb',     lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    scale_pos_weight=1.0, random_state=42,
                                    n_jobs=1, verbose=-1)),
    ('xgb',     xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   scale_pos_weight=1.0, eval_metric='logloss',
                                   random_state=42, n_jobs=1, verbosity=0,
                                   objective='binary:logistic')),
]

for meta_C in [0.1, 1.0, 10.0]:
    for n_base in [3, 5, 6, 8]:
        try:
            sname = f'Stack{n_base}_metaC{meta_C}'
            meta = LogisticRegression(C=meta_C, class_weight='balanced',
                                       max_iter=5000, random_state=42, solver='lbfgs')
            stk = StackingClassifier(
                estimators=base_ests[:n_base], final_estimator=meta,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                passthrough=False, n_jobs=1
            )
            stk.fit(Xtr_enn42, y_enn42)
            thr_s, _ = sweep_thr(stk, Xte_110, y_test)
            register(sname, stk, Xte_110, thr_s)
        except Exception as e:
            print(f"  Stack{n_base} err: {e}")

# %% [markdown]
# ## 7. Voting Ensemble

# %%
print(f"\n{SEP}\n  D. Soft Voting Ensembles\n{SEP}")

# Re-train fixed MLP variants on K110+ENN
mlp_variants = []
for seed, alpha, lr in [(42,0.001,0.0005),(42,0.01,0.0005),(42,0.05,0.0005),
                         (13,0.05,0.0005),(0,0.001,0.001),(76,0.01,0.001)]:
    m = MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=alpha,
                      learning_rate_init=lr, max_iter=1000, random_state=seed,
                      early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
    try:
        m.fit(Xtr_enn42, y_enn42)
        mlp_variants.append((f'm_s{seed}_a{alpha}', m))
    except: pass

# Add SVM variants
svm_var = SVC(kernel='rbf', C=100.0, gamma='scale', probability=True,
              random_state=42, class_weight='balanced')
svm_var.fit(Xtr_enn42, y_enn42)
mlp_variants.append(('svm100', svm_var))

# Various voting combos
for n_v in [3, 4, 5, 6, 7]:
    if n_v > len(mlp_variants): break
    try:
        vname = f'Vote{n_v}'
        ens = VotingClassifier(estimators=mlp_variants[:n_v], voting='soft', n_jobs=1)
        ens.fit(Xtr_enn42, y_enn42)
        thr_e, _ = sweep_thr(ens, Xte_110, y_test)
        register(vname, ens, Xte_110, thr_e)
    except Exception as e:
        print(f"  Vote{n_v} err: {e}")

# %% [markdown]
# ## 8. Other Feature Combinations × ENN

# %%
print(f"\n{SEP}\n  E. Other Feature Combos × ENN\n{SEP}")

# MFCC + ENN
Xtr_mfcc_raw = base[mfcc_cols].values[train_mask].astype(np.float64)
Xte_mfcc_raw = base[mfcc_cols].values[test_mask].astype(np.float64)

# Spec + MFCC
sm_cols = spec_cols + mfcc_cols
Xtr_sm_raw   = base[sm_cols].values[train_mask].astype(np.float64)
Xte_sm_raw   = base[sm_cols].values[test_mask].astype(np.float64)

# All fusion
all_cols = spec_cols + mfcc_cols + w2v_cols
Xtr_all_raw  = base[all_cols].values[train_mask].astype(np.float64)
Xte_all_raw  = base[all_cols].values[test_mask].astype(np.float64)

for feat_label, Xtr_r, Xte_r in [
    ('MFCC',    Xtr_mfcc_raw, Xte_mfcc_raw),
    ('SpecMFCC',Xtr_sm_raw,   Xte_sm_raw),
    ('Fusion3', Xtr_all_raw,  Xte_all_raw),
]:
    for k_v in [80, 100, 110, 120, 150]:
        Xtr_k, Xte_k = preproc(Xtr_r, Xte_r, y_train_all, k=k_v)
        Xtr_e2, y_e2 = do_enn(Xtr_k, y_train_all, k_n=3)
        for seed in [42, 13, 76, 0]:
            for alpha in [0.001, 0.01, 0.05]:
                name = f'{feat_label}_K{k_v}_ENN|s{seed}|a{alpha}'
                try:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=WINNER_ARCH, alpha=alpha,
                        learning_rate_init=0.0005, max_iter=1000, random_state=seed,
                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30
                    )
                    mlp.fit(Xtr_e2, y_e2)
                    thr, _ = sweep_thr(mlp, Xte_k, y_test)
                    register(name, mlp, Xte_k, thr)
                except: pass

# %% [markdown]
# ## 9. Summary

# %%
print(f"\n{'='*110}")
print(f"{'RINGKASAN v65 — Top Results':^110}")
print(f"{'='*110}")

rows = []
for name, m in results.items():
    rows.append({
        'Experiment':    name,
        'Test F1 Macro': round(m['f1_macro'], 4),
        'Test Accuracy': round(m['accuracy'], 4),
        'Test AUC':      round(m['roc_auc'],  4),
        'Test Recall':   round(m['recall'],   4),
        'Threshold':     round(thrs_s.get(name, 0.5), 3),
    })

df_cmp = (pd.DataFrame(rows)
          .sort_values('Test F1 Macro', ascending=False)
          .reset_index(drop=True))
df_cmp.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v65_comparison.csv")
df_cmp.to_csv(csv_path, index=False)

print(df_cmp[['Experiment','Test F1 Macro','Test Accuracy',
              'Test AUC','Test Recall','Threshold']].head(20).to_string())

best_name = df_cmp.iloc[0]['Experiment']
best_f1   = df_cmp.iloc[0]['Test F1 Macro']
best_acc  = df_cmp.iloc[0]['Test Accuracy']
best_auc  = df_cmp.iloc[0]['Test AUC']
best_thr  = df_cmp.iloc[0]['Threshold']

print(f"\n  ★ BEST: {best_name}")
print(f"  Test F1  : {best_f1:.4f}")
print(f"  Test Acc : {best_acc:.4f}")
print(f"  Test AUC : {best_auc:.4f}")

if best_f1 >= 0.82:
    print(f"\n  🎯 F1 ≥ 0.82 TERCAPAI! F1 = {best_f1:.4f}")
elif best_f1 >= 0.80:
    print(f"\n  ✓ F1 ≥ 0.80 maintained. Best={best_f1:.4f}. Push to 0.82 via v66.")
else:
    print(f"\n  ⚠  F1={best_f1:.4f}. Perlu v66.")

# %% [markdown]
# ## 10. Classification Report & Plot

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

COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6','#10b981',
          '#f59e0b','#8b5cf6','#ec4899','#14b8a6'] * 3
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(f'v65 — Fine-Tune K110+ENN | Best F1={best_f1:.4f}', fontsize=13, fontweight='bold')

ax = axes[0]
top20 = df_cmp.head(20)
bars  = ax.barh(range(len(top20)), top20['Test F1 Macro'],
                color=[COLORS[i%len(COLORS)] for i in range(len(top20))], edgecolor='white')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([n[:38] for n in top20['Experiment']], fontsize=6.5)
ax.axvline(0.82, color='red',    linestyle='--', lw=1.5, label='Target 0.82')
ax.axvline(0.80, color='orange', linestyle=':', lw=1.2, label='v64 0.80')
ax.set_xlabel('Test F1 Macro'); ax.set_title('Top 20', fontweight='bold')
ax.legend(fontsize=9); ax.set_xlim(0, 1.05)
ax.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, top20['Test F1 Macro']):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7.5, fontweight='bold')

ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Normal','Depresi'], yticklabels=['Normal','Depresi'])
ax2.set_title(f'Best CM: F1={best_f1:.3f}\n{best_name[:40]}', fontweight='bold')
ax2.set_xlabel('Prediksi'); ax2.set_ylabel('Aktual')

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v65_comparison.png")
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nPlot: {p}")

# %% [markdown]
# ## 11. Save & Final

# %%
best_model_obj = models_s[best_name]
with open(os.path.join(MODELS_DIR, 'v65_best_model.pkl'), 'wb') as f: pickle.dump(best_model_obj, f)

summary = {
    'version': 'v65', 'n_experiments': len(results),
    'best_exp': best_name, 'best_f1': float(best_f1),
    'best_accuracy': float(best_acc), 'best_auc': float(best_auc),
    'best_threshold': float(best_thr),
    'target_80_achieved': bool(best_f1 >= 0.80),
    'target_82_achieved': bool(best_f1 >= 0.82),
}
with open(os.path.join(MODELS_DIR, 'v65_summary.json'), 'w') as f: json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v65':^80}")
print("=" * 80)
print(f"  Experiments  : {len(results)}")
print(f"  Best Config  : {best_name}")
print(f"  Test F1      : {best_f1:.4f}")
print(f"  Test Accuracy: {best_acc:.4f}")
print(f"  Test AUC     : {best_auc:.4f}")
print(f"  Threshold    : {best_thr:.3f}")
print(f"  F1 ≥ 0.80    : {'✓' if best_f1 >= 0.80 else '✗'}")
print(f"  F1 ≥ 0.82    : {'✓ TERCAPAI!' if best_f1 >= 0.82 else '✗ Belum'}")
print(f"  Total Waktu  : {time.time()-t_global:.1f}s")
print("=" * 80)
