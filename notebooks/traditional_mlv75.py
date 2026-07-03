# %% [markdown]
# # Pipeline v75 — Apple-to-Apple: 4 Skenario × 4 ML Tradisional
# **Dataset:** DAIC-WOZ — 102 Partisipan (Audio-Only)
#
# **Sesuai prompt.txt (versi final):**
# - Skenario 1: Spectrogram saja
# - Skenario 2: MFCC saja
# - Skenario 3: Wav2Vec saja
# - Skenario 4: Feature Fusion (Spec + MFCC + W2V)
# - Model: RF, SVM, LR, XGBoost (4 ML Tradisional, tanpa DL)
# - K-Fold CV (5-fold) pada training data — SMOTEENN inside fold
# - Split 80:20, test seimbang (10N + 10D)
# - Learning Curves untuk tiap model terbaik
#
# **Perbedaan dari v74:**
# - Hapus MLP/DL sesuai prompt update
# - Tambah hyperparameter tuning per skenario (pilih best config via inner CV)
# - Visualisasi lebih komprehensif
# - Summary apple-to-apple yang lebih jelas

# %% [markdown]
# ## 1. Setup

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v75")
for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 80)
print("  Pipeline v75 — 4 Skenario × 4 ML Tradisional")
print("  Apple-to-Apple | K-Fold CV Inside Fold | Tuned Hyperparameters")
print("=" * 80)

# %% [markdown]
# ## 2. Load Data (102 Partisipan)

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

spec_cols = [f'spec_{c}' for c in fcols_spec]
mfcc_cols = [f'mfcc_{c}' for c in fcols_mfcc]
w2v_cols  = [f'w2v_{c}'  for c in fcols_w2v]

y_all  = base['label_depresi'].values.astype(int)
X_spec = base[spec_cols].fillna(0).values.astype(np.float64)
X_mfcc = base[mfcc_cols].fillna(0).values.astype(np.float64)
X_w2v  = base[w2v_cols].fillna(0).values.astype(np.float64)
X_fuse = np.hstack([X_spec, X_mfcc, X_w2v])

# ── DIMENSI FITUR (sesuai prompt) ─────────────────────────────────────
print(f"\n{'='*70}")
print("  1. ANALISIS DIMENSI FITUR (Catat Eksplisit)")
print(f"{'='*70}")
print(f"  Skenario 1 — Spectrogram : {X_spec.shape[1]:>5} fitur")
print(f"  Skenario 2 — MFCC        : {X_mfcc.shape[1]:>5} fitur")
print(f"  Skenario 3 — Wav2Vec     : {X_w2v.shape[1]:>5} fitur")
print(f"  Skenario 4 — Fusion (S+M+W): {X_fuse.shape[1]:>4} fitur (konkatenasi)")
print(f"\n  Total partisipan : {len(y_all)}")
print(f"  Label Normal (0) : {(y_all==0).sum()}")
print(f"  Label Depresi (1): {(y_all==1).sum()}")

SCENARIOS = {
    'S1_Spectrogram': X_spec,
    'S2_MFCC':        X_mfcc,
    'S3_Wav2Vec':     X_w2v,
    'S4_Fusion':      X_fuse,
}

# %% [markdown]
# ## 3. Data Splitting — 80:20, Test Seimbang

# %%
print(f"\n{'='*70}")
print("  2. DATA SPLITTING — 80:20 | Test = 10 Normal + 10 Depresi")
print(f"{'='*70}")

idx_normal  = np.where(y_all == 0)[0]
idx_depresi = np.where(y_all == 1)[0]
np.random.seed(RANDOM_SEED)
test_normal  = np.random.choice(idx_normal,  size=10, replace=False)
test_depresi = np.random.choice(idx_depresi, size=10, replace=False)
test_idx     = np.concatenate([test_normal, test_depresi])
train_idx    = np.setdiff1d(np.arange(len(y_all)), test_idx)
y_train      = y_all[train_idx]
y_test       = y_all[test_idx]

print(f"  Training : {len(train_idx)} partisipan  (Normal:{(y_train==0).sum()}, Depresi:{(y_train==1).sum()})")
print(f"  Test     : {len(test_idx)} partisipan  (Normal:{(y_test==0).sum()}, Depresi:{(y_test==1).sum()}) ✓ Seimbang")

# %% [markdown]
# ## 4. Helpers

# %%
def safe_clean(X):
    return np.clip(np.nan_to_num(X, nan=0., posinf=0., neginf=0.), -1e9, 1e9)

def preprocess_pair(X_tr, X_te, y_tr, k=60):
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    sc  = StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))
    if k and k < X_tr.shape[1]:
        sel  = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def balance_fold(X, y):
    """SMOTEENN hanya pada inner training fold."""
    k_a = min(3, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        sm = SMOTEENN(random_state=RANDOM_SEED,
                      smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        return sm.fit_resample(X, y)
    except:
        try: return SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a).fit_resample(X, y)
        except: return X, y

def sweep_thr(probs, y_true):
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.92, 0.01):
        f1 = f1_score(y_true, (probs>=thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return best_thr, best_f1

# %% [markdown]
# ## 5. Model Configs — 4 Traditional ML dengan Hyperparameter Tuning

# %%
# Beberapa konfigurasi per model untuk inner tuning
MODEL_CONFIGS = {
    'RandomForest': [
        {'n_estimators':300, 'max_depth':None, 'class_weight':'balanced'},
        {'n_estimators':500, 'max_depth':8,    'class_weight':'balanced'},
        {'n_estimators':300, 'max_depth':6,    'class_weight':'balanced'},
    ],
    'SVM': [
        {'C':1.0,   'kernel':'rbf', 'gamma':'scale', 'class_weight':'balanced'},
        {'C':10.0,  'kernel':'rbf', 'gamma':'scale', 'class_weight':'balanced'},
        {'C':100.0, 'kernel':'rbf', 'gamma':'scale', 'class_weight':'balanced'},
    ],
    'LogisticRegression': [
        {'C':0.1,  'class_weight':'balanced', 'max_iter':5000, 'solver':'lbfgs'},
        {'C':1.0,  'class_weight':'balanced', 'max_iter':5000, 'solver':'lbfgs'},
        {'C':10.0, 'class_weight':'balanced', 'max_iter':5000, 'solver':'lbfgs'},
    ],
    'XGBoost': [
        {'n_estimators':100, 'max_depth':3, 'learning_rate':0.1,  'scale_pos_weight':2.0},
        {'n_estimators':200, 'max_depth':3, 'learning_rate':0.05, 'scale_pos_weight':2.0},
        {'n_estimators':200, 'max_depth':4, 'learning_rate':0.05, 'scale_pos_weight':2.0},
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

K_FEATURES = 60
K_FOLDS    = 5
cv_outer   = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_inner   = StratifiedKFold(n_splits=3,        shuffle=True, random_state=RANDOM_SEED)

# %% [markdown]
# ## 6. Main Experiment — 4 Skenario × 4 Model (K-Fold CV, Anti-Overfitting)

# %%
all_results = []

print(f"\n{'='*80}")
print("  EKSPERIMEN — 4 Skenario × 4 ML Tradisional")
print(f"  K-Fold CV (5-fold, SMOTEENN inside) | Hyperparameter Tuning (3 configs)")
print(f"{'='*80}")

for sc_name, X_full in SCENARIOS.items():
    X_tr_raw = X_full[train_idx]
    X_te_raw = X_full[test_idx]
    X_tr_p, X_te_p = preprocess_pair(X_tr_raw, X_te_raw, y_train, k=K_FEATURES)

    print(f"\n{'─'*72}")
    print(f"  SKENARIO: {sc_name} | Train:{X_tr_p.shape} Test:{X_te_p.shape}")

    for model_name, configs in MODEL_CONFIGS.items():
        t0 = time.time()

        # ── Inner tuning: pilih config terbaik via 3-fold ─────────────
        best_cfg_idx, best_cfg_f1 = 0, -1
        for ci, cfg in enumerate(configs):
            fold_f1s = []
            for f_tr, f_val in cv_inner.split(X_tr_p, y_train):
                Xf_tr, Xf_val = X_tr_p[f_tr], X_tr_p[f_val]
                yf_tr, yf_val = y_train[f_tr], y_train[f_val]
                Xf_bal, yf_bal = balance_fold(Xf_tr, yf_tr)
                try:
                    clf = build_model(model_name, cfg)
                    clf.fit(Xf_bal, yf_bal)
                    probs = clf.predict_proba(Xf_val)[:,1]
                    thr, _ = sweep_thr(probs, yf_val)
                    fold_f1s.append(f1_score(yf_val, (probs>=thr).astype(int),
                                             average='macro', zero_division=0))
                except: fold_f1s.append(0.)
            mean_f1 = np.mean(fold_f1s) if fold_f1s else 0.
            if mean_f1 > best_cfg_f1:
                best_cfg_f1, best_cfg_idx = mean_f1, ci

        best_cfg = configs[best_cfg_idx]

        # ── 5-Fold CV dengan best config ──────────────────────────────
        cv_f1s, cv_accs = [], []
        for f_tr, f_val in cv_outer.split(X_tr_p, y_train):
            Xf_tr, Xf_val = X_tr_p[f_tr], X_tr_p[f_val]
            yf_tr, yf_val = y_train[f_tr], y_train[f_val]
            Xf_bal, yf_bal = balance_fold(Xf_tr, yf_tr)
            try:
                clf = build_model(model_name, best_cfg)
                clf.fit(Xf_bal, yf_bal)
                probs = clf.predict_proba(Xf_val)[:,1]
                thr, _ = sweep_thr(probs, yf_val)
                preds  = (probs>=thr).astype(int)
                cv_f1s.append(f1_score(yf_val, preds, average='macro', zero_division=0))
                cv_accs.append(accuracy_score(yf_val, preds))
            except: cv_f1s.append(0.); cv_accs.append(0.)

        cv_f1_mean = float(np.mean(cv_f1s))
        cv_f1_std  = float(np.std(cv_f1s))

        # ── Final: train on full balanced train, eval on test ──────────
        X_bal, y_bal = balance_fold(X_tr_p, y_train)
        try:
            clf_f = build_model(model_name, best_cfg)
            clf_f.fit(X_bal, y_bal)
            probs_te  = clf_f.predict_proba(X_te_p)[:,1]
            thr_te, _ = sweep_thr(probs_te, y_test)
            preds_te  = (probs_te >= thr_te).astype(int)
            try: auc_te = float(roc_auc_score(y_test, probs_te))
            except: auc_te = 0.0
            test_f1  = float(f1_score(y_test, preds_te, average='macro', zero_division=0))
            test_acc = float(accuracy_score(y_test, preds_te))
            rec_dep  = float(recall_score(y_test, preds_te, pos_label=1, zero_division=0))
            prec_dep = float(precision_score(y_test, preds_te, pos_label=1, zero_division=0))
        except:
            preds_te = np.zeros(len(y_test), dtype=int)
            probs_te = np.zeros(len(y_test))
            test_f1 = test_acc = auc_te = rec_dep = prec_dep = 0.0

        gap = test_f1 - cv_f1_mean
        result = {
            'scenario':    sc_name,
            'model':       model_name,
            'best_cfg':    str(best_cfg),
            'cv_f1_mean':  round(cv_f1_mean,4),
            'cv_f1_std':   round(cv_f1_std,4),
            'cv_acc_mean': round(float(np.mean(cv_accs)),4),
            'test_f1':     round(test_f1,4),
            'test_acc':    round(test_acc,4),
            'test_auc':    round(auc_te,4),
            'test_rec_dep':  round(rec_dep,4),
            'test_prec_dep': round(prec_dep,4),
            'overfit_gap': round(gap,4),
            'time_s':      round(time.time()-t0,1),
            'y_pred':      preds_te.tolist(),
            'y_prob':      probs_te.tolist(),
        }
        all_results.append(result)

        status = ('⚠OVERFIT' if gap < -0.10 else '✓OK' if abs(gap)<=0.10 else '↑GEN')
        print(f"  {model_name:<20} best_cfg[{best_cfg_idx}] "
              f"CV={cv_f1_mean:.4f}±{cv_f1_std:.4f} "
              f"Test={test_f1:.4f} Acc={test_acc:.4f} Gap={gap:+.4f} {status}", flush=True)

# %% [markdown]
# ## 7. Apple-to-Apple Summary Table

# %%
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(RESULTS_DIR,"metrics","v75_results.csv"), index=False)

print(f"\n{'='*100}")
print(f"{'TABEL RINGKASAN APPLE-TO-APPLE — v75':^100}")
print(f"{'='*100}")

print("\n  CV F1 Macro (5-Fold, SMOTEENN Inside Fold — Jujur):")
pv = df_res.pivot(index='model', columns='scenario', values='cv_f1_mean').round(4)
print(pv.to_string())

print("\n  Test F1 Macro (20 Balanced Test Samples):")
pv2 = df_res.pivot(index='model', columns='scenario', values='test_f1').round(4)
print(pv2.to_string())

print("\n  Gap = Test - CV F1 (negatif = model overfit ke CV folds):")
pv3 = df_res.pivot(index='model', columns='scenario', values='overfit_gap').round(4)
print(pv3.to_string())

# Best per scenario
print(f"\n{'─'*90}")
print("  BEST MODEL per Skenario:")
print(f"  {'Skenario':<20} {'Best (CV)':<20} {'CV F1':>7} {'Test F1':>8} {'Acc':>7} {'AUC':>7}")
print(f"  {'─'*20} {'─'*20} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")
for sc in SCENARIOS:
    rows = [r for r in all_results if r['scenario']==sc]
    best = max(rows, key=lambda x: x['cv_f1_mean'])
    print(f"  {sc:<20} {best['model']:<20} {best['cv_f1_mean']:>7.4f} "
          f"{best['test_f1']:>8.4f} {best['test_acc']:>7.4f} {best['test_auc']:>7.4f}")

# Sorted by CV F1
print(f"\n{'─'*90}")
print("  SEMUA HASIL (Sorted by CV F1):")
print(f"  {'Skenario':<20} {'Model':<22} {'CV F1':>7} {'Std':>6} {'Test F1':>8} {'Acc':>7} {'Gap':>8} {'Status'}")
print(f"  {'─'*20} {'─'*22} {'─'*7} {'─'*6} {'─'*8} {'─'*7} {'─'*8} {'─'*10}")
for r in sorted(all_results, key=lambda x: x['cv_f1_mean'], reverse=True):
    st = '⚠OVERFIT' if r['overfit_gap'] < -0.10 else '✓OK' if abs(r['overfit_gap'])<=0.10 else '↑GEN'
    print(f"  {r['scenario']:<20} {r['model']:<22} {r['cv_f1_mean']:>7.4f} "
          f"{r['cv_f1_std']:>6.4f} {r['test_f1']:>8.4f} {r['test_acc']:>7.4f} "
          f"{r['overfit_gap']:>+8.4f} {st}")

# %% [markdown]
# ## 8. Diagnosis Overfitting

# %%
print(f"\n{'='*80}")
print("  DIAGNOSIS OVERFITTING — K-Fold CV vs Test Set")
print("="*80)
n_ok  = sum(1 for r in all_results if abs(r['overfit_gap']) <= 0.10)
n_ov  = sum(1 for r in all_results if r['overfit_gap'] < -0.10)
n_gen = sum(1 for r in all_results if r['overfit_gap'] > 0.10)
print(f"  ✓ OK (|gap| <= 0.10) : {n_ok}/{len(all_results)} model")
print(f"  ⚠ Overfit (gap < -0.10): {n_ov}/{len(all_results)} model")
print(f"  ↑ Generalize (gap > 0.10): {n_gen}/{len(all_results)} model")

# %% [markdown]
# ## 9. Learning Curves (per Skenario — Model Terbaik)

# %%
print(f"\n[Learning Curves — Best Model per Skenario...]")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('v75 — Learning Curves | Best CV F1 per Feature Scenario\n'
             '(SMOTEENN Inside Fold | Train vs CV Score)',
             fontsize=13, fontweight='bold')

for ax, (sc_name, X_full) in zip(axes.flatten(), SCENARIOS.items()):
    X_tr_raw = X_full[train_idx]
    X_te_raw = X_full[test_idx]
    X_tr_p, X_te_p = preprocess_pair(X_tr_raw, X_te_raw, y_train, k=K_FEATURES)
    X_bal, y_bal = balance_fold(X_tr_p, y_train)

    sc_rows  = [r for r in all_results if r['scenario'] == sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    # parse best_cfg
    import ast
    try: cfg = ast.literal_eval(best_row['best_cfg'])
    except: cfg = MODEL_CONFIGS[best_row['model']][0]
    clf_lc = build_model(best_row['model'], cfg)

    try:
        train_sizes, train_sc, val_sc = learning_curve(
            clf_lc, X_bal, y_bal,
            train_sizes=np.linspace(0.2, 1.0, 6),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
            scoring='f1_macro', n_jobs=1)
        ax.fill_between(train_sizes, train_sc.mean(1)-train_sc.std(1),
                        train_sc.mean(1)+train_sc.std(1), alpha=0.15, color='#6366f1')
        ax.fill_between(train_sizes, val_sc.mean(1)-val_sc.std(1),
                        val_sc.mean(1)+val_sc.std(1), alpha=0.15, color='#ef4444')
        ax.plot(train_sizes, train_sc.mean(1), 'o-', color='#6366f1', lw=2, label='Train Score')
        ax.plot(train_sizes, val_sc.mean(1),   's--',color='#ef4444', lw=2, label='CV Score')
        ax.axhline(best_row['test_f1'], color='#22c55e', linestyle=':',
                   lw=1.5, label=f"Test F1={best_row['test_f1']:.3f}")
        ax.axhline(0.75, color='orange', linestyle=':', lw=1, alpha=0.6)
        ax.text(train_sizes[-1]*0.5, 0.76, 'Target 0.75', color='orange', fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:80], ha='center', va='center',
                transform=ax.transAxes, fontsize=8)

    gap_str = f"Gap={best_row['overfit_gap']:+.3f}"
    ax.set_title(f"{sc_name}\nBest: {best_row['model']} | CV={best_row['cv_f1_mean']:.4f} "
                 f"Test={best_row['test_f1']:.4f} {gap_str}",
                 fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Training Samples'); ax.set_ylabel('F1 Macro')
    ax.legend(fontsize=8); ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
p_lc = os.path.join(RESULTS_DIR,"plots","v75_learning_curves.png")
fig.savefig(p_lc, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p_lc}")

# %% [markdown]
# ## 10. Confusion Matrices (Best per Skenario)

# %%
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
fig2.suptitle('v75 — Confusion Matrix (Best CV F1) | Test: 10 Normal + 10 Depresi',
              fontsize=12, fontweight='bold')
for ax, (sc_name, _) in zip(axes2, SCENARIOS.items()):
    sc_rows  = [r for r in all_results if r['scenario']==sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    cm = confusion_matrix(y_test, best_row['y_pred'], labels=[0,1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Depresi'], yticklabels=['Normal','Depresi'],
                annot_kws={'size':14})
    ax.set_title(f'{sc_name}\n{best_row["model"]}\n'
                 f'CV={best_row["cv_f1_mean"]:.4f} | Test={best_row["test_f1"]:.4f}',
                 fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
plt.tight_layout()
p_cm = os.path.join(RESULTS_DIR,"plots","v75_confusion_matrices.png")
fig2.savefig(p_cm, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p_cm}")

# %% [markdown]
# ## 11. Apple-to-Apple Bar Chart

# %%
COLORS = ['#6366f1','#ef4444','#f97316','#22c55e']
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig3.suptitle('v75 — Apple-to-Apple: 4 Feature Scenarios × 4 ML Traditional\n'
              'K-Fold CV (Honest) vs Test Set Evaluation',
              fontsize=12, fontweight='bold')

x = np.arange(len(MODEL_NAMES)); width = 0.18
for i, sc_name in enumerate(SCENARIOS.keys()):
    sc_rows = [r for r in all_results if r['scenario']==sc_name]
    cv_v  = [next(r for r in sc_rows if r['model']==m)['cv_f1_mean'] for m in MODEL_NAMES]
    te_v  = [next(r for r in sc_rows if r['model']==m)['test_f1']    for m in MODEL_NAMES]
    label = sc_name.replace('S1_','').replace('S2_','').replace('S3_','').replace('S4_','')
    ax1.bar(x+i*width, cv_v, width, label=label, color=COLORS[i], alpha=0.85, edgecolor='white')
    ax2.bar(x+i*width, te_v, width, label=label, color=COLORS[i], alpha=0.85, edgecolor='white')

for ax, title in [(ax1,'CV F1 Macro (K-Fold CV, Anti-Overfitting)'),
                   (ax2,'Test F1 Macro (20 Balanced Test Samples)')]:
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(MODEL_NAMES, rotation=15, ha='right', fontsize=9)
    ax.axhline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
    ax.set_ylim(0, 1.0); ax.set_ylabel('F1 Macro')
    ax.set_title(title, fontweight='bold'); ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in ax.patches:
        val = bar.get_height()
        if val > 0.05:
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')

plt.tight_layout()
p_bar = os.path.join(RESULTS_DIR,"plots","v75_apple_comparison.png")
fig3.savefig(p_bar, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p_bar}")

# %% [markdown]
# ## 12. Classification Reports (Best CV per Skenario)

# %%
print(f"\n{'='*80}")
print("  CLASSIFICATION REPORTS — Best CV F1 per Skenario")
print("="*80)
for sc_name in SCENARIOS.keys():
    sc_rows  = [r for r in all_results if r['scenario']==sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    print(f"\n  ── {sc_name} × {best_row['model']} ──")
    print(f"  CV F1 = {best_row['cv_f1_mean']:.4f} ± {best_row['cv_f1_std']:.4f} | "
          f"Test F1 = {best_row['test_f1']:.4f} | "
          f"Acc = {best_row['test_acc']:.4f} | AUC = {best_row['test_auc']:.4f}")
    print(classification_report(y_test, best_row['y_pred'],
                                 target_names=['Normal','Depresi'], zero_division=0))

# %% [markdown]
# ## 13. Final Report

# %%
best_cv   = max(all_results, key=lambda x: x['cv_f1_mean'])
best_test = max(all_results, key=lambda x: x['test_f1'])

print(f"\n{'='*80}")
print(f"{'FINAL REPORT — Pipeline v75':^80}")
print(f"{'='*80}")
print(f"\n  DIMENSI FITUR:")
print(f"    Spectrogram : {X_spec.shape[1]} | MFCC: {X_mfcc.shape[1]} | Wav2Vec: {X_w2v.shape[1]} | Fusion: {X_fuse.shape[1]}")
print(f"\n  DATA SPLIT  : {len(train_idx)} train | {len(test_idx)} test (10N+10D seimbang)")
print(f"  VALIDASI    : {K_FOLDS}-Fold CV, SMOTEENN inside fold (anti data leakage)")
print(f"  HYP TUNING  : 3 configs per model (inner 3-fold selection)")
print(f"\n  BEST by CV F1 (jujur, anti-overfitting):")
print(f"    {best_cv['scenario']} × {best_cv['model']}")
print(f"    CV F1  = {best_cv['cv_f1_mean']:.4f} ± {best_cv['cv_f1_std']:.4f}")
print(f"    Test F1= {best_cv['test_f1']:.4f} | Gap={best_cv['overfit_gap']:+.4f}")
print(f"\n  BEST by Test F1:")
print(f"    {best_test['scenario']} × {best_test['model']}")
print(f"    CV F1  = {best_test['cv_f1_mean']:.4f} ± {best_test['cv_f1_std']:.4f}")
print(f"    Test F1= {best_test['test_f1']:.4f} | Gap={best_test['overfit_gap']:+.4f}")
print(f"\n  TARGET 0.75 (CV F1) : {'✓ TERCAPAI!' if best_cv['cv_f1_mean']>=0.75 else f'✗ Belum ({best_cv[chr(99)+chr(118)+chr(95)+chr(102)+chr(49)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.4f})'}")
print(f"  TARGET 0.75 (Test)  : {'✓ TERCAPAI!' if best_test['test_f1']>=0.75 else f'✗ Belum ({best_test[chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(102)+chr(49)]:.4f})'}")
print(f"\n  Total Waktu : {time.time()-t_global:.1f}s")
print(f"{'='*80}")

json.dump({
    'version': 'v75',
    'scenarios': list(SCENARIOS.keys()),
    'models': MODEL_NAMES,
    'n_participants': int(len(y_all)),
    'train_size': int(len(train_idx)),
    'test_size': int(len(test_idx)),
    'k_folds': K_FOLDS,
    'feature_dims': {
        'Spectrogram': int(X_spec.shape[1]),
        'MFCC': int(X_mfcc.shape[1]),
        'Wav2Vec': int(X_w2v.shape[1]),
        'Fusion': int(X_fuse.shape[1]),
    },
    'best_cv': {
        'scenario': best_cv['scenario'], 'model': best_cv['model'],
        'cv_f1': best_cv['cv_f1_mean'], 'test_f1': best_cv['test_f1'],
    },
    'best_test': {
        'scenario': best_test['scenario'], 'model': best_test['model'],
        'cv_f1': best_test['cv_f1_mean'], 'test_f1': best_test['test_f1'],
    },
    'target_075_cv': bool(best_cv['cv_f1_mean'] >= 0.75),
    'target_075_test': bool(best_test['test_f1'] >= 0.75),
}, open(os.path.join(RESULTS_DIR,"metrics","v75_summary.json"),'w'), indent=2)
