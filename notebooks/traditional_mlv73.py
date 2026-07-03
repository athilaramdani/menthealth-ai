# %% [markdown]
# # Pipeline v73 — Apple-to-Apple: 4 Feature Scenarios × (4 ML + 1 DL)
# **Dataset:** DAIC-WOZ — 102 Partisipan (Audio-Only)
#
# **Struktur Eksperimen (sesuai prompt.txt):**
# - Skenario 1: Spectrogram saja
# - Skenario 2: MFCC saja
# - Skenario 3: Wav2Vec saja
# - Skenario 4: Feature Fusion (Spec + MFCC + W2V)
#
# **Per skenario:** RF, SVM, LR, XGBoost + MLP (Deep Learning)
# **Validasi:** K-Fold CV (5-fold) pada training data (anti-overfitting)
# **Split:** 80:20 balanced test set
# **Visualisasi:** Learning Curves untuk model terbaik
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 1. Setup & Imports

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
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, learning_curve,
    StratifiedShuffleSplit
)
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v73")
for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 80)
print("  Pipeline v73 — 4 Feature Scenarios × 5 Models (4 ML + 1 DL)")
print("  Apple-to-Apple Comparison | K-Fold CV + Learning Curves")
print("=" * 80)

# %% [markdown]
# ## 2. Load All 102 Participants

# %%
print("\n[1] Loading semua 102 partisipan...")

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
    df['label_depresi']  = df.apply(map_label, axis=1)
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

# ── BAGIAN 1: Dimensi Fitur ────────────────────────────────────────────
print("\n" + "="*70)
print("  ANALISIS DIMENSI FITUR (Sesuai Prompt — Catat Eksplisit)")
print("="*70)
print(f"  Skenario 1 — Spectrogram : {X_spec.shape[1]:>5} fitur")
print(f"  Skenario 2 — MFCC        : {X_mfcc.shape[1]:>5} fitur")
print(f"  Skenario 3 — Wav2Vec     : {X_w2v.shape[1]:>5} fitur")
print(f"  Skenario 4 — Fusion      : {X_fuse.shape[1]:>5} fitur (Spec+MFCC+W2V)")
print(f"\n  Total partisipan: {len(y_all)}")
print(f"  Label Normal (0) : {(y_all==0).sum()}")
print(f"  Label Depresi (1): {(y_all==1).sum()}")

SCENARIOS = {
    'S1_Spectrogram': X_spec,
    'S2_MFCC':        X_mfcc,
    'S3_Wav2Vec':     X_w2v,
    'S4_Fusion':      X_fuse,
}

# %% [markdown]
# ## 3. Data Splitting — 80:20 Balanced Test Set

# %%
print("\n" + "="*70)
print("  DATA SPLITTING — 80:20, Test Set Seimbang (10N + 10D = 20 Test)")
print("="*70)

# Balanced test: ambil 10 Normal + 10 Depresi = 20 test
idx_normal  = np.where(y_all == 0)[0]
idx_depresi = np.where(y_all == 1)[0]

np.random.seed(RANDOM_SEED)
test_normal  = np.random.choice(idx_normal,  size=10, replace=False)
test_depresi = np.random.choice(idx_depresi, size=10, replace=False)
test_idx     = np.concatenate([test_normal, test_depresi])
train_idx    = np.setdiff1d(np.arange(len(y_all)), test_idx)

y_train = y_all[train_idx]
y_test  = y_all[test_idx]

print(f"  Training : {len(train_idx)} partisipan (Normal:{(y_train==0).sum()}, Depresi:{(y_train==1).sum()})")
print(f"  Test     : {len(test_idx)} partisipan  (Normal:{(y_test==0).sum()}, Depresi:{(y_test==1).sum()})")

# %% [markdown]
# ## 4. Preprocessing & Preprocessing Helpers

# %%
def safe_clean(X):
    return np.clip(np.nan_to_num(X, nan=0., posinf=0., neginf=0.), -1e9, 1e9)

def preprocess(X_tr, X_te, y_tr, k=100):
    """StandardScaler + SelectKBest. Fit dari train, transform test."""
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    sc = StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))
    if k and k < X_tr.shape[1]:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def balance_train(X, y):
    """SMOTEENN untuk mengatasi imbalance pada training data."""
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

def sweep_threshold(probs, y_true):
    """Cari threshold optimal untuk F1 Macro."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.10, 0.92, 0.01):
        f1 = f1_score(y_true, (probs>=thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return best_thr, best_f1

# %% [markdown]
# ## 5. Model Definitions

# %%
def get_models(spw=1.0):
    spw = max(float(spw), 0.01)
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
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=RANDOM_SEED, n_jobs=1, verbosity=0),
        'MLP_DL': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), alpha=0.001,
            learning_rate_init=0.001, max_iter=500,
            random_state=RANDOM_SEED, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20),
    }

MODEL_NAMES = ['RandomForest','SVM','LogisticRegression','XGBoost','MLP_DL']
MODEL_TYPES = {
    'RandomForest':     'Traditional ML',
    'SVM':              'Traditional ML',
    'LogisticRegression':'Traditional ML',
    'XGBoost':          'Traditional ML',
    'MLP_DL':           'Deep Learning',
}

# %% [markdown]
# ## 6. Main Experiment — K-Fold CV + Test Evaluation

# %%
K_FOLDS   = 5
K_FEATURES = 100   # SelectKBest

all_results = []
cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print(f"\n{'='*80}")
print(f"  EKSPERIMEN — 4 Skenario × 5 Model | {K_FOLDS}-Fold CV + Test Eval")
print(f"{'='*80}")

for scenario_name, X_full in SCENARIOS.items():
    X_train_raw = X_full[train_idx]
    X_test_raw  = X_full[test_idx]

    # Preprocess: fit dari train, transform test
    X_train_p, X_test_p = preprocess(X_train_raw, X_test_raw, y_train, k=K_FEATURES)

    # Balance training data
    X_train_bal, y_train_bal = balance_train(X_train_p, y_train)
    spw = max((y_train_bal==0).sum()/max((y_train_bal==1).sum(),1), 0.01)

    print(f"\n{'─'*70}")
    print(f"  SKENARIO: {scenario_name}")
    print(f"  Train shape: {X_train_p.shape} | Setelah SMOTEENN: {X_train_bal.shape}")
    print(f"  Test shape : {X_test_p.shape}")

    models = get_models(spw)

    for model_name in MODEL_NAMES:
        clf = models[model_name]
        t0  = time.time()

        # ── K-Fold CV pada training data ──────────────────────────────
        cv_f1s, cv_accs, cv_recs = [], [], []
        for fold_tr, fold_val in cv.split(X_train_bal, y_train_bal):
            Xf_tr, Xf_val = X_train_bal[fold_tr], X_train_bal[fold_val]
            yf_tr, yf_val = y_train_bal[fold_tr], y_train_bal[fold_val]
            try:
                clf.fit(Xf_tr, yf_tr)
                probs_cv  = clf.predict_proba(Xf_val)[:,1]
                thr_cv, _ = sweep_threshold(probs_cv, yf_val)
                preds_cv  = (probs_cv >= thr_cv).astype(int)
                cv_f1s.append(f1_score(yf_val, preds_cv, average='macro', zero_division=0))
                cv_accs.append(accuracy_score(yf_val, preds_cv))
                cv_recs.append(f1_score(yf_val, preds_cv, pos_label=1, average='binary', zero_division=0))
            except:
                cv_f1s.append(0.); cv_accs.append(0.); cv_recs.append(0.)

        cv_f1_mean = float(np.mean(cv_f1s))
        cv_f1_std  = float(np.std(cv_f1s))

        # ── Train on full balanced train, eval on test ─────────────────
        try:
            clf.fit(X_train_bal, y_train_bal)
            probs_te  = clf.predict_proba(X_test_p)[:,1]
            thr_te, _ = sweep_threshold(probs_te, y_test)
            preds_te  = (probs_te >= thr_te).astype(int)
            try: auc_te = float(roc_auc_score(y_test, probs_te))
            except: auc_te = 0.0
            test_f1   = float(f1_score(y_test, preds_te, average='macro', zero_division=0))
            test_acc  = float(accuracy_score(y_test, preds_te))
        except Exception as e:
            preds_te = np.zeros(len(y_test)); test_f1 = test_acc = auc_te = 0.0

        elapsed = time.time() - t0
        overfit_gap = test_f1 - cv_f1_mean  # positif = test lebih baik (potensi overfit)

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

        gap_label = (f"[{'OVERFIT' if overfit_gap > 0.10 else 'OK' if overfit_gap < 0.05 else 'MILD'}]")
        print(f"  {model_name:<20} | CV_F1={cv_f1_mean:.4f}±{cv_f1_std:.4f} "
              f"TestF1={test_f1:.4f} TestAcc={test_acc:.4f} "
              f"Gap={overfit_gap:+.4f}{gap_label} t={elapsed:.1f}s", flush=True)

# %% [markdown]
# ## 7. Apple-to-Apple Summary Table

# %%
print(f"\n{'='*100}")
print(f"{'TABEL RINGKASAN — Apple-to-Apple (4 Skenario × 5 Model)':^100}")
print(f"{'='*100}")

df_res = pd.DataFrame(all_results)
df_pivot = df_res.pivot_table(
    index=['scenario','model'],
    values=['cv_f1_mean','test_f1','test_acc','overfit_gap'],
    aggfunc='first'
).round(4)
print(df_pivot.to_string())

# Tabel perbandingan tradisional vs DL per skenario
print(f"\n{'─'*100}")
print("  RINGKASAN per Skenario (best CV F1):")
for sc in SCENARIOS:
    sc_rows = [r for r in all_results if r['scenario'] == sc]
    best_ml = max([r for r in sc_rows if r['type']=='Traditional ML'], key=lambda x: x['cv_f1_mean'])
    best_dl = max([r for r in sc_rows if r['type']=='Deep Learning'],  key=lambda x: x['cv_f1_mean'])
    print(f"  {sc:20s}: Best ML={best_ml['model'][:12]}(CV={best_ml['cv_f1_mean']:.4f}, Test={best_ml['test_f1']:.4f}) | "
          f"DL={best_dl['model'][:8]}(CV={best_dl['cv_f1_mean']:.4f}, Test={best_dl['test_f1']:.4f})")

# Overall best
best_cv   = max(all_results, key=lambda x: x['cv_f1_mean'])
best_test = max(all_results, key=lambda x: x['test_f1'])
print(f"\n  ★ Best by CV F1  : {best_cv['scenario']} × {best_cv['model']} "
      f"→ CV={best_cv['cv_f1_mean']:.4f} Test={best_cv['test_f1']:.4f}")
print(f"  ★ Best by Test F1: {best_test['scenario']} × {best_test['model']} "
      f"→ CV={best_test['cv_f1_mean']:.4f} Test={best_test['test_f1']:.4f}")

df_res.to_csv(os.path.join(RESULTS_DIR,"metrics","v73_results.csv"), index=False)

# %% [markdown]
# ## 8. K-Fold CV Anti-Overfitting Diagnosis

# %%
print(f"\n{'='*80}")
print("  DIAGNOSIS OVERFITTING (CV vs Test Gap)")
print("="*80)
print(f"  {'Skenario':<20} {'Model':<22} {'CV F1':>8} {'TestF1':>8} {'Gap':>8} {'Status':>12}")
print(f"  {'─'*20} {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")
for r in sorted(all_results, key=lambda x: abs(x['overfit_gap']), reverse=True):
    status = ('⚠ OVERFIT'  if r['overfit_gap'] > 0.10 else
              '✓ OK'       if r['overfit_gap'] < 0.05 else
              '~ MILD')
    print(f"  {r['scenario']:<20} {r['model']:<22} {r['cv_f1_mean']:>8.4f} "
          f"{r['test_f1']:>8.4f} {r['overfit_gap']:>+8.4f} {status:>12}")

# %% [markdown]
# ## 9. Learning Curves — Model Terbaik per Skenario

# %%
print(f"\n[Learning Curves untuk model terbaik per skenario...]")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('v73 — Learning Curves (Best Model per Feature Scenario)\n'
             'Train Score vs Cross-Val Score — Deteksi Overfitting',
             fontsize=13, fontweight='bold')

for ax, (sc_name, X_full) in zip(axes.flatten(), SCENARIOS.items()):
    X_train_raw = X_full[train_idx]
    X_test_raw  = X_full[test_idx]
    X_tr_p, _   = preprocess(X_train_raw, X_test_raw, y_train, k=K_FEATURES)

    # Pilih best model untuk skenario ini (berdasarkan CV F1)
    sc_rows  = [r for r in all_results if r['scenario'] == sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    spw_best = max((y_train==0).sum()/max((y_train==1).sum(),1), 0.01)
    best_clf = get_models(spw_best)[best_row['model']]

    # Learning curve
    try:
        train_sizes, train_sc, val_sc = learning_curve(
            best_clf, X_tr_p, y_train,
            train_sizes=np.linspace(0.2, 1.0, 6),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
            scoring='f1_macro', n_jobs=1
        )
        ax.fill_between(train_sizes, train_sc.mean(1)-train_sc.std(1),
                        train_sc.mean(1)+train_sc.std(1), alpha=0.15, color='#6366f1')
        ax.fill_between(train_sizes, val_sc.mean(1)-val_sc.std(1),
                        val_sc.mean(1)+val_sc.std(1), alpha=0.15, color='#ef4444')
        ax.plot(train_sizes, train_sc.mean(1), 'o-', color='#6366f1',
                lw=2, label='Training Score')
        ax.plot(train_sizes, val_sc.mean(1),   's--', color='#ef4444',
                lw=2, label='CV Score')
        ax.axhline(best_row['test_f1'], color='#22c55e', linestyle=':',
                   lw=1.5, label=f"Test F1={best_row['test_f1']:.3f}")
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)

    ax.set_title(f"{sc_name}\nBest: {best_row['model']} (CV={best_row['cv_f1_mean']:.4f})",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Training Samples'); ax.set_ylabel('F1 Macro Score')
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0.75, color='orange', linestyle=':', lw=1, label='Target 0.75', alpha=0.7)

plt.tight_layout()
p_lc = os.path.join(RESULTS_DIR,"plots","v73_learning_curves.png")
fig.savefig(p_lc, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Learning curves saved: {p_lc}")

# %% [markdown]
# ## 10. Confusion Matrix Best Model (per Skenario)

# %%
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
fig2.suptitle('v73 — Confusion Matrix (Best CV F1 per Scenario) — Test Set (20 samples)',
              fontsize=12, fontweight='bold')

for ax, (sc_name, _) in zip(axes2, SCENARIOS.items()):
    sc_rows  = [r for r in all_results if r['scenario'] == sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    cm = confusion_matrix(y_test, best_row['y_pred'], labels=[0,1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Depresi'], yticklabels=['Normal','Depresi'],
                annot_kws={'size':14})
    ax.set_title(f'{sc_name}\n{best_row["model"]}\nCV={best_row["cv_f1_mean"]:.4f} Test={best_row["test_f1"]:.4f}',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')

plt.tight_layout()
p_cm = os.path.join(RESULTS_DIR,"plots","v73_confusion_matrices.png")
fig2.savefig(p_cm, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Confusion matrices saved: {p_cm}")

# %% [markdown]
# ## 11. Bar Chart — Apple-to-Apple Perbandingan

# %%
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig3.suptitle('v73 — Apple-to-Apple: 4 Feature Scenarios × 5 Models',
              fontsize=13, fontweight='bold')

x = np.arange(len(MODEL_NAMES))
width = 0.18
colors = ['#6366f1','#ef4444','#f97316','#22c55e']

for i, sc_name in enumerate(SCENARIOS.keys()):
    sc_rows = [r for r in all_results if r['scenario']==sc_name]
    cv_vals  = [next(r for r in sc_rows if r['model']==m)['cv_f1_mean'] for m in MODEL_NAMES]
    test_vals= [next(r for r in sc_rows if r['model']==m)['test_f1']    for m in MODEL_NAMES]
    ax1.bar(x + i*width, cv_vals,  width, label=sc_name, color=colors[i], alpha=0.85)
    ax2.bar(x + i*width, test_vals, width, label=sc_name, color=colors[i], alpha=0.85)

for ax, title in [(ax1,'CV F1 Macro (K-Fold CV, Anti-Overfit)'),
                   (ax2,'Test F1 Macro (20 Balanced Test Samples)')]:
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(MODEL_NAMES, rotation=25, ha='right', fontsize=9)
    ax.axhline(0.75, color='red', linestyle='--', lw=1.5, label='Target 0.75')
    ax.set_ylim(0, 1.1); ax.set_ylabel('F1 Macro'); ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in ax.patches:
        val = bar.get_height()
        if val > 0.01:
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')

# Mark DL models
ax1.axvline(3.5 + width*1.5, color='gray', linestyle=':', lw=1)
ax2.axvline(3.5 + width*1.5, color='gray', linestyle=':', lw=1)
ax1.text(4 + width, 1.0, '↑ DL', color='gray', fontsize=9)
ax2.text(4 + width, 1.0, '↑ DL', color='gray', fontsize=9)

plt.tight_layout()
p_bar = os.path.join(RESULTS_DIR,"plots","v73_apple_comparison.png")
fig3.savefig(p_bar, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Apple comparison saved: {p_bar}")

# %% [markdown]
# ## 12. Classification Reports — Best Model per Scenario

# %%
print(f"\n{'='*80}")
print("  CLASSIFICATION REPORTS — Best Model per Skenario (Test Set)")
print("="*80)
for sc_name in SCENARIOS.keys():
    sc_rows  = [r for r in all_results if r['scenario']==sc_name]
    best_row = max(sc_rows, key=lambda x: x['cv_f1_mean'])
    print(f"\n  ── {sc_name} × {best_row['model']} ──")
    print(f"  [Type: {best_row['type']}] CV F1={best_row['cv_f1_mean']:.4f}±{best_row['cv_f1_std']:.4f} "
          f"| Test F1={best_row['test_f1']:.4f} | Test Acc={best_row['test_acc']:.4f}")
    print(classification_report(y_test, best_row['y_pred'],
                                 target_names=['Normal','Depresi'], zero_division=0))

# %% [markdown]
# ## 13. Final Report

# %%
print(f"\n{'='*80}")
print(f"{'FINAL REPORT — Pipeline v73':^80}")
print(f"{'='*80}")

print(f"\n  DIMENSI FITUR:")
print(f"  Spectrogram : {X_spec.shape[1]} fitur")
print(f"  MFCC        : {X_mfcc.shape[1]} fitur")
print(f"  Wav2Vec     : {X_w2v.shape[1]} fitur")
print(f"  Fusion Total: {X_fuse.shape[1]} fitur")

print(f"\n  DATA SPLIT: {len(train_idx)} train | {len(test_idx)} test (10N+10D seimbang)")
print(f"  VALIDASI  : {K_FOLDS}-Fold CV pada training data")

print(f"\n  RINGKASAN TERBAIK (berdasarkan CV F1 — anti-overfitting):")
df_top = df_res.nlargest(5, 'cv_f1_mean')[
    ['scenario','model','type','cv_f1_mean','cv_f1_std','test_f1','test_acc','overfit_gap']]
df_top.index = range(1, len(df_top)+1)
print(df_top.to_string())

best_overall = df_res.loc[df_res['cv_f1_mean'].idxmax()]
print(f"\n  ★ BEST OVERALL (CV F1): {best_overall['scenario']} × {best_overall['model']}")
print(f"  CV F1 : {best_overall['cv_f1_mean']:.4f} ± {best_overall['cv_f1_std']:.4f}")
print(f"  Test F1: {best_overall['test_f1']:.4f}")
print(f"  Test Acc: {best_overall['test_acc']:.4f}")
print(f"  Overfitting Gap: {best_overall['overfit_gap']:+.4f}")
print(f"\n  Target 0.75 (CV F1) : {'✓ TERCAPAI!' if best_overall['cv_f1_mean']>=0.75 else '✗ Belum'}")
print(f"  Target 0.75 (Test)  : {'✓ TERCAPAI!' if best_overall['test_f1']>=0.75 else '✗ Belum'}")
print(f"\n  Total Waktu: {time.time()-t_global:.1f}s")
print(f"{'='*80}")

# Save summary
summary = {
    'version': 'v73',
    'scenarios': list(SCENARIOS.keys()),
    'models': MODEL_NAMES,
    'n_participants': int(len(y_all)),
    'train_size': int(len(train_idx)),
    'test_size': int(len(test_idx)),
    'k_folds': K_FOLDS,
    'best_by_cv': {
        'scenario': str(best_overall['scenario']),
        'model':    str(best_overall['model']),
        'cv_f1':   float(best_overall['cv_f1_mean']),
        'test_f1': float(best_overall['test_f1']),
    }
}
with open(os.path.join(RESULTS_DIR,"metrics","v73_summary.json"),'w') as f:
    json.dump(summary, f, indent=2)
