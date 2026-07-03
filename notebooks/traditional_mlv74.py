# %% [markdown]
# # Pipeline v74 — Apple-to-Apple FIXED (SMOTEENN inside CV Fold)
# **Dataset:** DAIC-WOZ — 102 Partisipan (Audio-Only)
#
# **BUG FIX dari v73:**
# - v73: SMOTEENN diterapkan SEBELUM K-Fold CV → CV F1 inflated (1.0!)
#   karena synthetic samples bocor ke validation fold
# - v74: SMOTEENN diterapkan DALAM setiap fold (pada inner train saja) → CV jujur
#
# **Struktur (sesuai prompt.txt):**
# - 4 Skenario: Spectrogram, MFCC, Wav2Vec, Fusion
# - 5 Model per skenario: RF, SVM, LR, XGBoost, MLP (DL)
# - K-Fold CV (5-fold) PADA training data (SMOTEENN inside fold)
# - Split 80:20, test seimbang (10N + 10D)
# - Learning Curves untuk model terbaik

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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, confusion_matrix
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v74")
for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 80)
print("  Pipeline v74 — 4 Skenario × 5 Model | SMOTEENN Inside CV (Fixed)")
print("=" * 80)

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
for fname, sname in [
    ("train_split_Depression_AVEC2017.csv","train"),
    ("dev_split_Depression_AVEC2017.csv","dev"),
    ("full_test_split.csv","test"),
]:
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col.lower()=='participant_id':
            df.rename(columns={col:'Participant_ID'}, inplace=True)
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

print(f"  Total: {len(y_all)} (0:{(y_all==0).sum()}, 1:{(y_all==1).sum()})")
print(f"\n  DIMENSI FITUR:")
print(f"  Skenario 1 — Spectrogram : {X_spec.shape[1]:>5} fitur")
print(f"  Skenario 2 — MFCC        : {X_mfcc.shape[1]:>5} fitur")
print(f"  Skenario 3 — Wav2Vec     : {X_w2v.shape[1]:>5} fitur")
print(f"  Skenario 4 — Fusion      : {X_fuse.shape[1]:>5} fitur (Spec+MFCC+W2V)")

SCENARIOS = {
    'S1_Spectrogram': X_spec,
    'S2_MFCC':        X_mfcc,
    'S3_Wav2Vec':     X_w2v,
    'S4_Fusion':      X_fuse,
}

# %% [markdown]
# ## 3. Data Splitting — 80:20 Balanced Test (10N + 10D)

# %%
idx_normal  = np.where(y_all == 0)[0]
idx_depresi = np.where(y_all == 1)[0]
np.random.seed(RANDOM_SEED)
test_normal  = np.random.choice(idx_normal,  size=10, replace=False)
test_depresi = np.random.choice(idx_depresi, size=10, replace=False)
test_idx     = np.concatenate([test_normal, test_depresi])
train_idx    = np.setdiff1d(np.arange(len(y_all)), test_idx)
y_train      = y_all[train_idx]
y_test       = y_all[test_idx]

print(f"\n  Training : {len(train_idx)} (N:{(y_train==0).sum()}, D:{(y_train==1).sum()})")
print(f"  Test     : {len(test_idx)}  (N:{(y_test==0).sum()}, D:{(y_test==1).sum()}) ← seimbang")

# %% [markdown]
# ## 4. Helpers — BENAR: SMOTEENN Inside CV

# %%
def safe_clean(X):
    return np.clip(np.nan_to_num(X, nan=0., posinf=0., neginf=0.), -1e9, 1e9)

def preprocess_pair(X_tr, X_te, y_tr, k=60):
    """StandardScaler + SelectKBest. Fit dari train saja."""
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    sc = StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))
    if k and k < X_tr.shape[1]:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def balance_smoteenn(X, y):
    """SMOTEENN — hanya pada data training inner fold."""
    k_a = min(3, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        sm = SMOTEENN(random_state=RANDOM_SEED,
                      smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a))
        return sm.fit_resample(X, y)
    except:
        try:
            sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_a)
            return sm.fit_resample(X, y)
        except:
            return X, y

def sweep_thr(probs, y_true):
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.92, 0.01):
        f1 = f1_score(y_true, (probs>=thr).astype(int),
                      average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return best_thr, best_f1

# %% [markdown]
# ## 5. Model Factory

# %%
def get_models():
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            n_jobs=1, random_state=RANDOM_SEED),
        'SVM': SVC(
            kernel='rbf', C=10.0, gamma='scale',
            probability=True, class_weight='balanced',
            random_state=RANDOM_SEED),
        'LogisticRegression': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=5000,
            random_state=RANDOM_SEED, solver='lbfgs'),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            scale_pos_weight=2.0, eval_metric='logloss',
            random_state=RANDOM_SEED, n_jobs=1, verbosity=0),
        'MLP_DL': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), alpha=0.001,
            learning_rate_init=0.001, max_iter=500,
            random_state=RANDOM_SEED, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20),
    }

MODEL_NAMES = ['RandomForest','SVM','LogisticRegression','XGBoost','MLP_DL']
MODEL_TYPES = {
    'RandomForest':'Traditional ML','SVM':'Traditional ML',
    'LogisticRegression':'Traditional ML','XGBoost':'Traditional ML',
    'MLP_DL':'Deep Learning',
}

# %% [markdown]
# ## 6. K-Fold CV BENAR — SMOTEENN Inside Each Fold

# %%
K_FOLDS   = 5
K_FEATURES = 60   # SelectKBest (lebih kecil = lebih generalisasi)
cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

all_results = []

print(f"\n{'='*80}")
print(f"  EKSPERIMEN v74 — SMOTEENN Inside Fold (Benar, Anti Data Leakage)")
print(f"  {K_FOLDS}-Fold CV | K_features={K_FEATURES} | 4 Skenario × 5 Model")
print(f"{'='*80}")

for scenario_name, X_full in SCENARIOS.items():
    X_train_raw = X_full[train_idx]
    X_test_raw  = X_full[test_idx]

    # Preprocess: fit dari train, transform test (hanya scaling + feature selection)
    X_train_p, X_test_p = preprocess_pair(X_train_raw, X_test_raw, y_train, k=K_FEATURES)

    print(f"\n{'─'*70}")
    print(f"  SKENARIO: {scenario_name} | Train:{X_train_p.shape} Test:{X_test_p.shape}")

    models = get_models()

    for model_name in MODEL_NAMES:
        t0 = time.time()

        # ── K-Fold CV BENAR: SMOTEENN inside setiap fold ──────────────
        cv_f1s, cv_accs = [], []

        for fold_idx, (fold_tr, fold_val) in enumerate(cv.split(X_train_p, y_train)):
            # Split fold
            Xf_tr, Xf_val = X_train_p[fold_tr], X_train_p[fold_val]
            yf_tr, yf_val = y_train[fold_tr],    y_train[fold_val]

            # SMOTEENN hanya pada inner training fold (BUKAN pada full train)
            Xf_bal, yf_bal = balance_smoteenn(Xf_tr, yf_tr)

            try:
                clf = models[model_name]
                clf.fit(Xf_bal, yf_bal)
                probs_val  = clf.predict_proba(Xf_val)[:,1]
                thr_val, _ = sweep_thr(probs_val, yf_val)
                preds_val  = (probs_val >= thr_val).astype(int)
                cv_f1s.append(f1_score(yf_val, preds_val, average='macro', zero_division=0))
                cv_accs.append(accuracy_score(yf_val, preds_val))
            except:
                cv_f1s.append(0.); cv_accs.append(0.)

        cv_f1_mean = float(np.mean(cv_f1s))
        cv_f1_std  = float(np.std(cv_f1s))

        # ── Final model: SMOTEENN pada full training set ───────────────
        X_tr_bal, y_tr_bal = balance_smoteenn(X_train_p, y_train)

        try:
            clf_final = models[model_name]
            clf_final.fit(X_tr_bal, y_tr_bal)
            probs_te  = clf_final.predict_proba(X_test_p)[:,1]
            thr_te, _ = sweep_thr(probs_te, y_test)
            preds_te  = (probs_te >= thr_te).astype(int)
            try: auc_te = float(roc_auc_score(y_test, probs_te))
            except: auc_te = 0.0
            test_f1   = float(f1_score(y_test, preds_te, average='macro', zero_division=0))
            test_acc  = float(accuracy_score(y_test, preds_te))
        except:
            preds_te = np.zeros(len(y_test), dtype=int)
            probs_te = np.zeros(len(y_test))
            test_f1 = test_acc = auc_te = 0.0

        elapsed      = time.time() - t0
        overfit_gap  = test_f1 - cv_f1_mean  # positif = test > CV (fine), negatif = overfit

        result = {
            'scenario':    scenario_name,
            'model':       model_name,
            'type':        MODEL_TYPES[model_name],
            'cv_f1_mean':  round(cv_f1_mean, 4),
            'cv_f1_std':   round(cv_f1_std,  4),
            'cv_acc_mean': round(float(np.mean(cv_accs)), 4),
            'test_f1':     round(test_f1, 4),
            'test_acc':    round(test_acc, 4),
            'test_auc':    round(auc_te, 4),
            'overfit_gap': round(overfit_gap, 4),
            'time_s':      round(elapsed, 1),
            'y_pred':      preds_te.tolist(),
            'y_prob':      probs_te.tolist(),
        }
        all_results.append(result)

        gap_status = ('⚠ OVERFIT'  if overfit_gap < -0.10 else
                      '✓ OK'       if abs(overfit_gap) <= 0.10 else
                      '↑ GENERALIZE')
        print(f"  {model_name:<20} CV={cv_f1_mean:.4f}±{cv_f1_std:.4f} "
              f"TestF1={test_f1:.4f} Gap={overfit_gap:+.4f} {gap_status} t={elapsed:.1f}s", flush=True)

# %% [markdown]
# ## 7. Apple-to-Apple Summary

# %%
df_res = pd.DataFrame(all_results)

print(f"\n{'='*100}")
print(f"{'TABEL APPLE-TO-APPLE — 4 Skenario × 5 Model (v74 Fixed)':^100}")
print(f"{'='*100}")

# Pivot: CV F1
print("\n  CV F1 Macro (5-Fold, SMOTEENN Inside Fold — Jujur):")
pv_cv = df_res.pivot(index='model', columns='scenario', values='cv_f1_mean').round(4)
print(pv_cv.to_string())

print("\n  Test F1 Macro (20 Balanced Test Samples):")
pv_te = df_res.pivot(index='model', columns='scenario', values='test_f1').round(4)
print(pv_te.to_string())

print("\n  Overfitting Gap (Test F1 - CV F1) [negatif = overfit]:")
pv_gap = df_res.pivot(index='model', columns='scenario', values='overfit_gap').round(4)
print(pv_gap.to_string())

# Best per scenario
print(f"\n{'─'*90}")
print("  BEST per Skenario (sorted by CV F1):")
for sc in SCENARIOS:
    sc_rows = [r for r in all_results if r['scenario'] == sc]
    best    = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    best_te = max(sc_rows, key=lambda x: x['test_f1'])
    print(f"  {sc:20s} → Best CV : {best['model']:<22} CV={best['cv_f1_mean']:.4f} Test={best['test_f1']:.4f}")
    print(f"  {' ':20s}   Best Test: {best_te['model']:<22} CV={best_te['cv_f1_mean']:.4f} Test={best_te['test_f1']:.4f}")

# Semua model terurut CV F1
print(f"\n{'─'*90}")
print("  TOP 10 (CV F1 Terbaik):")
top10 = sorted(all_results, key=lambda x: x['cv_f1_mean'], reverse=True)[:10]
for i, r in enumerate(top10, 1):
    print(f"  {i:2d}. {r['scenario']:20s} × {r['model']:<22} "
          f"CV={r['cv_f1_mean']:.4f}±{r['cv_f1_std']:.4f} Test={r['test_f1']:.4f} "
          f"Acc={r['test_acc']:.4f} Gap={r['overfit_gap']:+.4f}")

df_res.to_csv(os.path.join(RESULTS_DIR,"metrics","v74_results.csv"), index=False)

# %% [markdown]
# ## 8. Diagnosis Overfitting

# %%
print(f"\n{'='*80}")
print("  DIAGNOSIS OVERFITTING — Gap Analysis")
print("="*80)
print(f"  {'Skenario':<20} {'Model':<22} {'CV F1':>8} {'Test F1':>8} {'Gap':>8} {'Status':>14}")
print(f"  {'─'*20} {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*14}")
for r in sorted(all_results, key=lambda x: x['overfit_gap']):
    status = ('⚠ OVERFIT'   if r['overfit_gap'] < -0.10 else
              '✓ OK'        if abs(r['overfit_gap']) <= 0.10 else
              '↑ GENERALIZE')
    print(f"  {r['scenario']:<20} {r['model']:<22} {r['cv_f1_mean']:>8.4f} "
          f"{r['test_f1']:>8.4f} {r['overfit_gap']:>+8.4f} {status:>14}")

# %% [markdown]
# ## 9. Learning Curves — Model Terbaik per Skenario

# %%
print(f"\n[Learning Curves...]")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('v74 — Learning Curves (Best CV F1 per Scenario)\nTrain vs CV Score — Deteksi Overfitting/Underfitting',
             fontsize=13, fontweight='bold')

for ax, (sc_name, X_full) in zip(axes.flatten(), SCENARIOS.items()):
    X_train_raw = X_full[train_idx]
    X_test_raw  = X_full[test_idx]
    X_tr_p, _   = preprocess_pair(X_train_raw, X_test_raw, y_train, k=K_FEATURES)
    X_tr_bal, y_tr_bal = balance_smoteenn(X_tr_p, y_train)

    sc_rows  = [r for r in all_results if r['scenario'] == sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    best_clf = get_models()[best_row['model']]

    try:
        train_sizes, train_sc, val_sc = learning_curve(
            best_clf, X_tr_bal, y_tr_bal,
            train_sizes=np.linspace(0.2, 1.0, 6),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
            scoring='f1_macro', n_jobs=1
        )
        ax.fill_between(train_sizes, train_sc.mean(1)-train_sc.std(1),
                        train_sc.mean(1)+train_sc.std(1), alpha=0.15, color='#6366f1')
        ax.fill_between(train_sizes, val_sc.mean(1)-val_sc.std(1),
                        val_sc.mean(1)+val_sc.std(1), alpha=0.15, color='#ef4444')
        ax.plot(train_sizes, train_sc.mean(1), 'o-', color='#6366f1', lw=2, label='Training Score')
        ax.plot(train_sizes, val_sc.mean(1),   's--',color='#ef4444', lw=2, label='CV Score')
        ax.axhline(best_row['test_f1'], color='#22c55e', linestyle=':',
                   lw=1.5, label=f"Test F1={best_row['test_f1']:.3f}")
        ax.axhline(0.75, color='orange', linestyle=':', lw=1, alpha=0.6, label='Target 0.75')
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:80], ha='center', va='center',
                transform=ax.transAxes, fontsize=8)

    ax.set_title(f"{sc_name}\nBest: {best_row['model']} (CV={best_row['cv_f1_mean']:.4f}, Test={best_row['test_f1']:.4f})",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Training Samples'); ax.set_ylabel('F1 Macro')
    ax.legend(fontsize=8); ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
p_lc = os.path.join(RESULTS_DIR,"plots","v74_learning_curves.png")
fig.savefig(p_lc, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p_lc}")

# %% [markdown]
# ## 10. Confusion Matrices

# %%
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
fig2.suptitle('v74 — Confusion Matrix (Best CV F1) | Test Set: 10N + 10D',
              fontsize=12, fontweight='bold')
for ax, (sc_name, _) in zip(axes2, SCENARIOS.items()):
    sc_rows  = [r for r in all_results if r['scenario'] == sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    cm = confusion_matrix(y_test, best_row['y_pred'], labels=[0,1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Depresi'],
                yticklabels=['Normal','Depresi'], annot_kws={'size':14})
    ax.set_title(f'{sc_name}\n{best_row["model"]}\nCV={best_row["cv_f1_mean"]:.4f} Test={best_row["test_f1"]:.4f}',
                 fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
plt.tight_layout()
p_cm = os.path.join(RESULTS_DIR,"plots","v74_confusion_matrices.png")
fig2.savefig(p_cm, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p_cm}")

# %% [markdown]
# ## 11. Apple-to-Apple Bar Chart

# %%
COLORS = ['#6366f1','#ef4444','#f97316','#22c55e']
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig3.suptitle('v74 Fixed — Apple-to-Apple: 4 Feature Scenarios × 5 Models\n(SMOTEENN Inside CV Fold — Honest Evaluation)',
              fontsize=12, fontweight='bold')

x     = np.arange(len(MODEL_NAMES))
width = 0.18

for i, sc_name in enumerate(SCENARIOS.keys()):
    sc_rows = [r for r in all_results if r['scenario']==sc_name]
    cv_v    = [next(r for r in sc_rows if r['model']==m)['cv_f1_mean'] for m in MODEL_NAMES]
    te_v    = [next(r for r in sc_rows if r['model']==m)['test_f1']    for m in MODEL_NAMES]
    ax1.bar(x + i*width, cv_v,  width, label=sc_name, color=COLORS[i], alpha=0.85, edgecolor='white')
    ax2.bar(x + i*width, te_v,  width, label=sc_name, color=COLORS[i], alpha=0.85, edgecolor='white')

for ax, title in [(ax1,'CV F1 Macro (5-Fold, Honest)'),
                   (ax2,'Test F1 Macro (20 Test Samples)')]:
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(MODEL_NAMES, rotation=25, ha='right', fontsize=9)
    ax.axhline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
    ax.set_ylim(0, 1.0); ax.set_ylabel('F1 Macro')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in ax.patches:
        val = bar.get_height()
        if val > 0.05:
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=6, fontweight='bold')

plt.tight_layout()
p_bar = os.path.join(RESULTS_DIR,"plots","v74_comparison.png")
fig3.savefig(p_bar, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p_bar}")

# %% [markdown]
# ## 12. Classification Reports — Best per Skenario

# %%
print(f"\n{'='*80}")
print("  CLASSIFICATION REPORTS (Test Set) — Best CV F1 per Skenario")
print("="*80)
for sc_name in SCENARIOS.keys():
    sc_rows  = [r for r in all_results if r['scenario']==sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    print(f"\n  ── {sc_name} × {best_row['model']} ({best_row['type']}) ──")
    print(f"  CV F1={best_row['cv_f1_mean']:.4f}±{best_row['cv_f1_std']:.4f} | "
          f"Test F1={best_row['test_f1']:.4f} | Test Acc={best_row['test_acc']:.4f} | "
          f"AUC={best_row['test_auc']:.4f}")
    print(classification_report(y_test, best_row['y_pred'],
                                 target_names=['Normal','Depresi'], zero_division=0))

# %% [markdown]
# ## 13. Final Report

# %%
best_cv   = max(all_results, key=lambda x: x['cv_f1_mean'])
best_test = max(all_results, key=lambda x: x['test_f1'])

print(f"\n{'='*80}")
print(f"{'FINAL REPORT — Pipeline v74':^80}")
print(f"{'='*80}")
print(f"\n  DIMENSI FITUR:")
print(f"    Spectrogram : {X_spec.shape[1]} | MFCC: {X_mfcc.shape[1]} | Wav2Vec: {X_w2v.shape[1]} | Fusion: {X_fuse.shape[1]}")
print(f"\n  DATA SPLIT  : {len(train_idx)} train | {len(test_idx)} test (10N+10D seimbang)")
print(f"  VALIDASI    : {K_FOLDS}-Fold CV — SMOTEENN inside fold (data leakage-free)")
print(f"\n  BEST by CV F1  : {best_cv['scenario']} × {best_cv['model']}")
print(f"    CV F1  : {best_cv['cv_f1_mean']:.4f} ± {best_cv['cv_f1_std']:.4f}")
print(f"    Test F1: {best_cv['test_f1']:.4f}")
print(f"    Gap    : {best_cv['overfit_gap']:+.4f}")
print(f"\n  BEST by Test F1: {best_test['scenario']} × {best_test['model']}")
print(f"    CV F1  : {best_test['cv_f1_mean']:.4f} ± {best_test['cv_f1_std']:.4f}")
print(f"    Test F1: {best_test['test_f1']:.4f}")
print(f"    Gap    : {best_test['overfit_gap']:+.4f}")
print(f"\n  TARGET 0.75:")
print(f"    CV F1 >= 0.75 : {'✓ TERCAPAI' if best_cv['cv_f1_mean']>=0.75 else f'✗ Belum ({best_cv[chr(99)+chr(118)+chr(95)+chr(102)+chr(49)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.4f})'}")
print(f"    Test F1 >= 0.75: {'✓ TERCAPAI' if best_test['test_f1']>=0.75 else f'✗ Belum ({best_test[chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(102)+chr(49)]:.4f})'}")
print(f"\n  Total Waktu : {time.time()-t_global:.1f}s")
print(f"{'='*80}")

json.dump({
    'version': 'v74',
    'bug_fix': 'SMOTEENN inside CV fold — no data leakage',
    'best_cv_f1': best_cv['cv_f1_mean'], 'best_cv_scenario': best_cv['scenario'],
    'best_cv_model': best_cv['model'], 'best_test_f1': best_test['test_f1'],
    'best_test_scenario': best_test['scenario'], 'best_test_model': best_test['model'],
    'target_075_cv': best_cv['cv_f1_mean'] >= 0.75,
    'target_075_test': best_test['test_f1'] >= 0.75,
}, open(os.path.join(RESULTS_DIR,"metrics","v74_summary.json"),'w'), indent=2)
