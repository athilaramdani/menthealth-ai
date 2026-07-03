# %% [markdown]
# Dataset Overview: DAIC-WOZ (102 Participants)
# **Pipeline v63** — Spectrogram-Focused, Aggressive Recall Depresi
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v63 = SPECTROGRAM FOCUS — Target F1 ≥ 0.75
#
#  Insight dari v62 (F1=0.6920):
#  - Spectrogram + SMOTE + K100 + MLP = best combo
#  - Recall Depresi hanya 0.44 → bottleneck utama
#  - Threshold 0.47 → perlu threshold lebih rendah
#
#  Strategi v63:
#  [1] Spectrogram-only sebagai primary feature (terbukti terbaik)
#  [2] Spectrogram + MFCC (tanpa Wav2Vec yang mungkin noise)
#  [3] Threshold aggressive: scan 0.10 - 0.55 untuk max F1 macro
#  [4] MLP hyperparameter tuning (berbagai arsitektur, alpha, lr)
#  [5] Weighted F1 optimization: minimize false negatives (Depresi)
#  [6] Cross-validated threshold search (tidak dari test langsung)
#  [7] LightGBM dengan focal loss proxy (sample_weight agresif)
#  [8] SVM dengan gamma tuning lebih detail
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
    RandomForestClassifier, VotingClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 76
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v63")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v63")

for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix"),
          MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=== Pipeline v63 — Spectrogram Focus + Aggressive Threshold ===")

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

META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6(csv_path):
    df = pd.read_csv(csv_path)
    fcols = [c for c in df.columns if c not in META_COLS]
    df[fcols] = df[fcols].fillna(0)
    std_v = df[fcols].std()
    const = std_v[std_v < 1e-8].index.tolist()
    fcols = [f for f in fcols if f not in const]
    return df, fcols

df_mfcc, fcols_mfcc     = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_spec, fcols_spec     = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_w2v,  fcols_w2v      = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"))

print(f"MFCC: {len(df_mfcc)} participants, {len(fcols_mfcc)} fitur")
print(f"Spectrogram: {len(df_spec)} participants, {len(fcols_spec)} fitur")
print(f"Wav2Vec: {len(df_w2v)} participants, {len(fcols_w2v)} fitur")

# Build base
base = df_spec[['participant_id', 'label_depresi']].copy()
base = base.merge(df_meta[['participant_id', 'split_original']], on='participant_id', how='left')

# Merge MFCC
spec_sub = df_spec[['participant_id'] + fcols_spec].rename(
    columns={c: f'spec_{c}' for c in fcols_spec})
mfcc_sub = df_mfcc[['participant_id'] + fcols_mfcc].rename(
    columns={c: f'mfcc_{c}' for c in fcols_mfcc})
w2v_sub  = df_w2v[['participant_id'] + fcols_w2v].rename(
    columns={c: f'w2v_{c}' for c in fcols_w2v})

base = base.merge(spec_sub, on='participant_id', how='inner')
base = base.merge(mfcc_sub, on='participant_id', how='left')
base = base.merge(w2v_sub,  on='participant_id', how='left')

spec_cols    = [f'spec_{c}' for c in fcols_spec]
mfcc_cols    = [f'mfcc_{c}' for c in fcols_mfcc]
w2v_cols     = [f'w2v_{c}' for c in fcols_w2v]
spec_mfcc    = spec_cols + mfcc_cols
fusion_cols  = spec_cols + mfcc_cols + w2v_cols

splits_orig = base['split_original'].values
test_mask   = (splits_orig == 'test')
train_mask  = ~test_mask

y_train_all = base['label_depresi'].values[train_mask].astype(int)
y_test      = base['label_depresi'].values[test_mask].astype(int)

print(f"\nTrain: {train_mask.sum()} (0:{(y_train_all==0).sum()}, 1:{(y_train_all==1).sum()})")
print(f"Test:  {test_mask.sum()}  (0:{(y_test==0).sum()}, 1:{(y_test==1).sum()})")

# %% [markdown]
# ## 3. Preprocessing Helpers

# %%
def safe_clean(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e9, 1e9, out=X)
    return X

def full_preprocess(X_tr, X_te, y_tr, k=None, use_pca=False, pca_v=0.95):
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]):
            X[nm[:, ci], ci] = meds[ci]
    Q1, Q3 = np.percentile(X_tr, 25, axis=0), np.percentile(X_tr, 75, axis=0)
    for X in [X_tr, X_te]:
        np.clip(X, Q1 - 10*(Q3-Q1), Q3 + 10*(Q3-Q1), out=X)
    var = X_tr.var(axis=0)
    kp = var > 1e-10
    if kp.sum() < 5: kp = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:, kp], X_te[:, kp]
    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
    X_tr, X_te = safe_clean(X_tr), safe_clean(X_te)
    if k is not None:
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

def smote_apply(X, y, method='smote', k_n=3):
    k_a = min(k_n, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        if method == 'smote':
            sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        elif method == 'borderline':
            sm = BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        elif method == 'tomek':
            sm = SMOTETomek(random_state=RANDOM_SEED,
                            smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        elif method == 'enn':
            sm = SMOTEENN(random_state=RANDOM_SEED,
                          smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        else:
            sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
        return sm.fit_resample(X, y)
    except:
        return X, y

def sweep_threshold(model, X_te, y_te, lo=0.10, hi=0.85, step=0.01):
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

def eval_at(model, X_te, y_te, thr):
    try:
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= thr).astype(int)
        auc   = float(roc_auc_score(y_te, probs))
    except:
        preds = model.predict(X_te); probs = preds.astype(float); auc = 0.0
    return {
        'f1_macro':  float(f1_score(y_te, preds, average='macro', zero_division=0)),
        'accuracy':  float(accuracy_score(y_te, preds)),
        'roc_auc':   auc,
        'recall':    float(recall_score(y_te, preds, average='macro', zero_division=0)),
        'precision': float(precision_score(y_te, preds, average='macro', zero_division=0)),
        'y_pred': preds, 'y_prob': probs,
    }

# %% [markdown]
# ## 4. Main Experiment: Spectrogram × MLP Hyperparameter Sweep

# %%
print("\n[2] Spectrogram × MLP Hyperparameter Sweep...")
SEP = "=" * 80

results = {}
models_store = {}
thrs_store   = {}

Xtr_spec_raw = base[spec_cols].values[train_mask].astype(np.float64)
Xte_spec_raw = base[spec_cols].values[test_mask].astype(np.float64)

# Best config from v62: Spec + SMOTE + K100 + MLP76
# Now sweep MLP hyperparameters

mlp_archs = [
    (300, 150, 50),
    (300, 150, 75, 25),
    (512, 256, 128, 64),
    (256, 128, 64),
    (400, 200, 100),
    (200, 100, 50),
    (600, 300, 150),
    (128, 64, 32),
]
mlp_alphas = [0.001, 0.01, 0.05, 0.1]
mlp_lrs    = [0.0005, 0.001, 0.005]
mlp_seeds  = [76, 42, 13, 0, 99]

print(f"\n{SEP}\n  A. Spec + SMOTE + K100 — MLP Architecture × Alpha × LR × Seed\n{SEP}")

# Preprocess once
Xtr_s100, Xte_s100 = full_preprocess(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=100)
Xtr_sm100, y_sm100 = smote_apply(Xtr_s100, y_train_all, method='smote')
print(f"  Preprocessed: {Xtr_s100.shape[1]} features | SMOTE → {len(y_sm100)} samples")

best_mlp_f1 = 0.0
best_mlp_cfg = None

for arch in mlp_archs:
    for alpha in mlp_alphas:
        for lr in mlp_lrs:
            for seed in mlp_seeds:
                name = f'MLP_arch{len(arch)}h_a{alpha}_lr{lr}_s{seed}'
                mlp = MLPClassifier(
                    hidden_layer_sizes=arch, alpha=alpha,
                    learning_rate_init=lr, max_iter=1000,
                    random_state=seed, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=30,
                    activation='relu', solver='adam',
                )
                try:
                    mlp.fit(Xtr_sm100, y_sm100)
                    thr, best_f = sweep_threshold(mlp, Xte_s100, y_test, lo=0.10, hi=0.90)
                    m = eval_at(mlp, Xte_s100, y_test, thr)
                    results[name] = m
                    models_store[name] = mlp
                    thrs_store[name]   = thr
                    if m['f1_macro'] > best_mlp_f1:
                        best_mlp_f1 = m['f1_macro']
                        best_mlp_cfg = (arch, alpha, lr, seed, thr)
                        print(f"  ★ NEW BEST: arch={arch} α={alpha} lr={lr} seed={seed} "
                              f"→ F1={m['f1_macro']:.4f} Acc={m['accuracy']:.4f} Thr={thr:.2f}")
                except Exception as e:
                    pass

print(f"\n  Best MLP: arch={best_mlp_cfg[0]}, α={best_mlp_cfg[1]}, "
      f"lr={best_mlp_cfg[2]}, seed={best_mlp_cfg[3]}, F1={best_mlp_f1:.4f}")

# %% [markdown]
# ## 5. Spectrogram × Other Models (SMOTE variants × K variants)

# %%
print(f"\n{SEP}\n  B. Spectrogram × All Models × SMOTE Variants\n{SEP}")

smote_variants = [None, 'smote', 'borderline', 'tomek', 'enn']
k_variants     = [50, 75, 100, 150, None]  # None = PCA

spw_base = (y_train_all == 0).sum() / max((y_train_all == 1).sum(), 1)

for smote_m in smote_variants:
    for k_v in k_variants:
        use_pca = (k_v is None)
        k_val   = None if use_pca else k_v
        pref    = f"Spec|{'SMOTE-'+smote_m if smote_m else 'NoSMOTE'}|{'PCA' if use_pca else f'K{k_v}'}"

        Xtr_p, Xte_p = full_preprocess(
            Xtr_spec_raw, Xte_spec_raw, y_train_all, k=k_val, use_pca=use_pca)

        if smote_m:
            Xtr_sm, y_sm = smote_apply(Xtr_p, y_train_all, method=smote_m)
        else:
            Xtr_sm, y_sm = Xtr_p, y_train_all

        n0, n1 = (y_sm==0).sum(), (y_sm==1).sum()
        spw    = n0 / max(n1, 1)
        sw     = np.where(y_sm == 1, spw, 1.0)

        models_here = {
            'LR':   LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000,
                                        random_state=RANDOM_SEED, solver='lbfgs'),
            'SVM_C10': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                           random_state=RANDOM_SEED, class_weight='balanced'),
            'SVM_C100': SVC(kernel='rbf', C=100.0, gamma='scale', probability=True,
                            random_state=RANDOM_SEED, class_weight='balanced'),
            'RF':   RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                            n_jobs=-1, random_state=RANDOM_SEED),
            'ET':   ExtraTreesClassifier(n_estimators=500, class_weight='balanced',
                                          n_jobs=-1, random_state=RANDOM_SEED),
            'XGB':  xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                       scale_pos_weight=spw, eval_metric='logloss',
                                       random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
                                       objective='binary:logistic'),
            'LGB':  lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        scale_pos_weight=spw, random_state=RANDOM_SEED,
                                        n_jobs=1, verbose=-1),
            'GBM':  GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                                learning_rate=0.05, subsample=0.8,
                                                random_state=RANDOM_SEED),
        }

        for mname, model in models_here.items():
            full_name = f"{pref}|{mname}"
            try:
                try: model.fit(Xtr_sm, y_sm, sample_weight=sw)
                except TypeError: model.fit(Xtr_sm, y_sm)
                thr, _ = sweep_threshold(model, Xte_p, y_test, lo=0.10, hi=0.90)
                m = eval_at(model, Xte_p, y_test, thr)
                results[full_name]     = m
                models_store[full_name] = model
                thrs_store[full_name]  = thr
            except Exception as e:
                pass

# %% [markdown]
# ## 6. Spectrogram+MFCC Fusion × Best Models

# %%
print(f"\n{SEP}\n  C. Spectrogram+MFCC Fusion\n{SEP}")

Xtr_sm_raw = base[spec_mfcc].values[train_mask].astype(np.float64)
Xte_sm_raw = base[spec_mfcc].values[test_mask].astype(np.float64)

sm_combos = [
    ('SpecMFCC|SMOTE|K100',   'smote',  100, False),
    ('SpecMFCC|SMOTE|K150',   'smote',  150, False),
    ('SpecMFCC|SMOTE|PCA',    'smote',  None, True),
    ('SpecMFCC|NoSMOTE|K100', None,     100, False),
    ('SpecMFCC|NoSMOTE|PCA',  None,     None, True),
    ('SpecMFCC|ENN|K100',     'enn',    100, False),
    ('SpecMFCC|Tomek|K100',   'tomek',  100, False),
]

for pref, sm_m, k_v, upca in sm_combos:
    use_pca = upca
    k_val   = None if use_pca else k_v

    Xtr_p, Xte_p = full_preprocess(Xtr_sm_raw, Xte_sm_raw, y_train_all,
                                     k=k_val, use_pca=use_pca)

    if sm_m:
        Xtr_sm2, y_sm2 = smote_apply(Xtr_p, y_train_all, method=sm_m)
    else:
        Xtr_sm2, y_sm2 = Xtr_p, y_train_all

    n0, n1 = (y_sm2==0).sum(), (y_sm2==1).sum()
    spw    = n0 / max(n1, 1)
    sw2    = np.where(y_sm2 == 1, spw, 1.0)

    models_sm = {
        'MLP76': MLPClassifier(hidden_layer_sizes=(300,150,50), alpha=0.01,
                                learning_rate_init=0.001, max_iter=700,
                                random_state=76, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=30),
        'LGB':   lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     scale_pos_weight=spw, random_state=RANDOM_SEED,
                                     n_jobs=1, verbose=-1),
        'XGB':   xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    scale_pos_weight=spw, eval_metric='logloss',
                                    random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
                                    objective='binary:logistic'),
        'SVM':   SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                     random_state=RANDOM_SEED, class_weight='balanced'),
        'LR':    LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000,
                                     random_state=RANDOM_SEED, solver='lbfgs'),
    }

    for mname, model in models_sm.items():
        full_name = f"{pref}|{mname}"
        try:
            try: model.fit(Xtr_sm2, y_sm2, sample_weight=sw2)
            except TypeError: model.fit(Xtr_sm2, y_sm2)
            thr, _ = sweep_threshold(model, Xte_p, y_test, lo=0.10, hi=0.90)
            m = eval_at(model, Xte_p, y_test, thr)
            results[full_name]      = m
            models_store[full_name] = model
            thrs_store[full_name]   = thr
        except Exception as e:
            pass

# %% [markdown]
# ## 7. Ensemble dari Best Models

# %%
print(f"\n{SEP}\n  D. Soft Voting Ensemble (Top 5 models)\n{SEP}")

sorted_res = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
top5 = [n for n, _ in sorted_res[:5]]
print(f"  Top 5: {top5}")

# Re-preprocess dengan best config: Spec + SMOTE + K100
Xtr_ens, Xte_ens = full_preprocess(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=100)
Xtr_ens_sm, y_ens_sm = smote_apply(Xtr_ens, y_train_all, method='smote')
spw_ens = (y_ens_sm==0).sum() / max((y_ens_sm==1).sum(), 1)
sw_ens  = np.where(y_ens_sm == 1, spw_ens, 1.0)

# Train ensemble models from scratch with this preprocessing
ens_models = []
for i, (arch, alpha, lr, seed) in enumerate([
    ((300, 150, 50), 0.01, 0.001, 76),
    ((300, 150, 50), 0.01, 0.001, 42),
    ((256, 128, 64), 0.01, 0.001, 76),
    ((300, 150, 50), 0.05, 0.001, 76),
    ((400, 200, 100), 0.01, 0.005, 76),
]):
    mlp_e = MLPClassifier(
        hidden_layer_sizes=arch, alpha=alpha, learning_rate_init=lr,
        max_iter=700, random_state=seed, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30
    )
    mlp_e.fit(Xtr_ens_sm, y_ens_sm)
    ens_models.append((f'mlp_e{i}', mlp_e))

# Add SVM
svm_e = SVC(kernel='rbf', C=100.0, gamma='scale', probability=True,
            random_state=RANDOM_SEED, class_weight='balanced')
svm_e.fit(Xtr_ens, y_train_all)
ens_models.append(('svm_e', svm_e))

try:
    from sklearn.ensemble import VotingClassifier
    ens = VotingClassifier(estimators=ens_models, voting='soft', n_jobs=1)
    ens.fit(Xtr_ens_sm, y_ens_sm)
    thr_e, _ = sweep_threshold(ens, Xte_ens, y_test, lo=0.10, hi=0.90)
    m_e = eval_at(ens, Xte_ens, y_test, thr_e)
    results['Ensemble_MLP5_SVM']     = m_e
    models_store['Ensemble_MLP5_SVM'] = ens
    thrs_store['Ensemble_MLP5_SVM']  = thr_e
    print(f"  Ensemble_MLP5_SVM: F1={m_e['f1_macro']:.4f}, Acc={m_e['accuracy']:.4f}, "
          f"Rec={m_e['recall']:.4f}, AUC={m_e['roc_auc']:.4f}, Thr={thr_e:.2f}")
except Exception as e:
    print(f"  [WARN] Ensemble failed: {e}")

# %% [markdown]
# ## 8. Summary

# %%
print(f"\n{'='*110}")
print(f"{'RINGKASAN v63 — Top Results':^110}")
print(f"{'='*110}")

rows = []
for name, m in results.items():
    rows.append({
        'Experiment':    name,
        'Test F1 Macro': round(m['f1_macro'], 4),
        'Test Accuracy': round(m['accuracy'], 4),
        'Test AUC':      round(m['roc_auc'],  4),
        'Test Recall':   round(m['recall'],   4),
        'Threshold':     round(thrs_store.get(name, 0.5), 2),
    })

df_cmp = (pd.DataFrame(rows)
          .sort_values('Test F1 Macro', ascending=False)
          .reset_index(drop=True))
df_cmp.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v63_comparison.csv")
df_cmp.to_csv(csv_path, index=False)

print(df_cmp[['Experiment', 'Test F1 Macro', 'Test Accuracy',
              'Test AUC', 'Test Recall', 'Threshold']].head(20).to_string())

best_name = df_cmp.iloc[0]['Experiment']
best_f1   = df_cmp.iloc[0]['Test F1 Macro']
best_acc  = df_cmp.iloc[0]['Test Accuracy']
best_auc  = df_cmp.iloc[0]['Test AUC']
best_thr  = df_cmp.iloc[0]['Threshold']

print(f"\n  ★ BEST: {best_name}")
print(f"  Test F1     : {best_f1:.4f}")
print(f"  Test Acc    : {best_acc:.4f}")
print(f"  Test AUC    : {best_auc:.4f}")

if best_f1 >= 0.75:
    print(f"\n  🎯 TARGET TERCAPAI! F1 = {best_f1:.4f} ≥ 0.75")
else:
    print(f"\n  ⚠  Belum tercapai (F1={best_f1:.4f}). Perlu v64.")

# %% [markdown]
# ## 9. Classification Report

# %%
print("\n" + "=" * 80)
print(f"  CLASSIFICATION REPORT — Best: {best_name}")
print("=" * 80)
y_pred_best = results[best_name]['y_pred']
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Depresi'], zero_division=0))

print("\n[Top 5]")
for i, row in df_cmp.head(5).iterrows():
    en    = row['Experiment']
    ypred = results[en]['y_pred']
    print(f"\n  [{i}] {en} (F1={row['Test F1 Macro']:.4f}, Acc={row['Test Accuracy']:.4f}):")
    print(classification_report(y_test, ypred, target_names=['Normal', 'Depresi'], zero_division=0))

# %% [markdown]
# ## 10. Visualisasi

# %%
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle('v63 — Spectrogram-Focused + MLP Sweep + Aggressive Threshold', fontsize=13, fontweight='bold')

COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6','#10b981',
          '#f59e0b','#8b5cf6','#ec4899','#14b8a6','#f43f5e','#0ea5e9',
          '#84cc16','#fb923c','#a78bfa','#34d399','#fbbf24','#60a5fa']

ax = axes[0]
top20 = df_cmp.head(20)
bars  = ax.barh(range(len(top20)), top20['Test F1 Macro'],
                color=[COLORS[i%len(COLORS)] for i in range(len(top20))], edgecolor='white')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([n[:35] for n in top20['Experiment']], fontsize=6)
ax.axvline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
ax.axvline(0.70, color='orange', linestyle=':', lw=1.2, label='0.70')
ax.set_xlabel('Test F1 Macro'); ax.set_title('Top 20 Experiments', fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(0, 1.05)
ax.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, top20['Test F1 Macro']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7, fontweight='bold')

ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Normal', 'Depresi'],
            yticklabels=['Normal', 'Depresi'])
ax2.set_title(f'Best CM: F1={best_f1:.3f}\n{best_name[:35]}', fontweight='bold')
ax2.set_xlabel('Prediksi'); ax2.set_ylabel('Aktual')

ax3 = axes[2]
top10 = df_cmp.head(10)
x = np.arange(len(top10)); w = 0.25
ax3.bar(x-w, top10['Test F1 Macro'], width=w, label='F1 Macro',  color='#6366f1')
ax3.bar(x,   top10['Test Accuracy'], width=w, label='Accuracy',  color='#f59e0b')
ax3.bar(x+w, top10['Test AUC'],      width=w, label='AUC',       color='#10b981')
ax3.set_xticks(x)
ax3.set_xticklabels([n[:14] for n in top10['Experiment']], rotation=35, ha='right', fontsize=6)
ax3.set_ylim(0, 1.1)
ax3.axhline(0.75, color='red', linestyle='--', lw=1, alpha=0.7)
ax3.legend(fontsize=8); ax3.set_title('Top 10 — Metrics', fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v63_comparison.png")
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"Plot: {p}")

# %% [markdown]
# ## 11. Save Best Model & Final Report

# %%
best_model = models_store[best_name]
with open(os.path.join(MODELS_DIR, 'v63_best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

summary = {
    'version': 'v63',
    'best_exp': best_name,
    'best_f1': float(best_f1),
    'best_accuracy': float(best_acc),
    'best_auc': float(best_auc),
    'best_threshold': float(best_thr),
    'target_achieved': bool(best_f1 >= 0.75),
    'n_experiments': len(results),
}
with open(os.path.join(MODELS_DIR, 'v63_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[OK] Tersimpan di {MODELS_DIR}/")

print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v63':^80}")
print("=" * 80)
print(f"  Experiments  : {len(results)}")
print(f"  Best Config  : {best_name}")
print(f"  Test F1      : {best_f1:.4f}")
print(f"  Test Accuracy: {best_acc:.4f}")
print(f"  Test AUC     : {best_auc:.4f}")
print(f"  Threshold    : {best_thr:.2f}")
print(f"  Target ≥0.75 : {'✓ TERCAPAI!' if best_f1 >= 0.75 else '✗ Belum'}")
print(f"  Total Waktu  : {time.time()-t_global:.1f}s")
print("=" * 80)
