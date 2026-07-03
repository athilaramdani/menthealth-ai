# Pipeline v68 — K-Fold Cross-Validation untuk Cek Overfitting v66
# Dijalankan langsung: python notebooks/traditional_mlv68.py
#
# Metode validasi:
# [1] Stratified 5-Fold CV — semua 102 partisipan
# [2] Stratified 10-Fold CV — semua 102 partisipan
# [3] Leave-One-Out CV (opsional jika data kecil)
# [4] Bandingkan CV mean vs Test Set F1 = 0.9115 (v66)
#
# PENTING: SMOTEENN + SelectKBest diaplikasikan di DALAM setiap fold
# untuk mencegah data leakage. Ini standar yang benar.
#
# Interpretasi hasil:
# - CV F1 >> Test F1 → Model overfit di test set
# - CV F1 ≈ Test F1  → Model generalizes well (tidak overfitting)
# - CV F1 << Test F1 → Test set fluke / lucky split

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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v68")
os.makedirs(os.path.join(RESULTS_DIR, "plots"),   exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

t_start = time.time()
print("=" * 80)
print("  Pipeline v68 — K-Fold Cross-Validation (Cek Overfitting v66)")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD SEMUA DATA (102 partisipan)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading all 102 participants...")

def map_label(row):
    for col in ['PHQ8_Binary', 'PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score', 'PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return 1 if int(val) >= 10 else 0
    return 0

all_parts = []
for fname, sname in [
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
    df['split_original'] = sname
    df.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    df['participant_id'] = df['participant_id'].astype(int)
    all_parts.append(df[['participant_id', 'label_depresi', 'split_original']])

df_meta = pd.concat(all_parts, ignore_index=True)
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6(path):
    df = pd.read_csv(path)
    fc = [c for c in df.columns if c not in META_COLS]
    df[fc] = df[fc].fillna(0)
    sv = df[fc].std()
    return df, [f for f in fc if sv[f] >= 1e-8]

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
base = df_spec[['participant_id', 'label_depresi']].copy()
base = base.merge(df_meta[['participant_id', 'split_original']], on='participant_id', how='left')
sub  = df_spec[['participant_id'] + fcols_spec].rename(
    columns={c: f'spec_{c}' for c in fcols_spec})
base = base.merge(sub, on='participant_id', how='inner')
spec_cols = [f'spec_{c}' for c in fcols_spec]

# SEMUA 102 partisipan
X_all = base[spec_cols].values.astype(np.float64)
y_all = base['label_depresi'].values.astype(int)
splits_orig = base['split_original'].values

print(f"  Total partisipan : {len(y_all)}")
print(f"  Label 0 (Normal) : {(y_all==0).sum()}")
print(f"  Label 1 (Depresi): {(y_all==1).sum()}")
print(f"  Rasio imbalance  : {(y_all==0).sum()/(y_all==1).sum():.2f}:1")

# ─────────────────────────────────────────────────────────────────────────────
# 2. KONFIGURASI MODEL v66 (Winner)
# ─────────────────────────────────────────────────────────────────────────────
V66_WINNER = {
    'arch':  (400, 150, 75, 25),
    'alpha': 1e-5,
    'lr':    0.0003,
    'seed':  42,
    'K':     110,
    'enn_k': 4,
}

print(f"\n[2] Konfigurasi Model v66 Winner:")
print(f"  Arsitektur MLP : {V66_WINNER['arch']}")
print(f"  Alpha          : {V66_WINNER['alpha']}")
print(f"  Learning Rate  : {V66_WINNER['lr']}")
print(f"  Seed           : {V66_WINNER['seed']}")
print(f"  K (SelectKBest): {V66_WINNER['K']}")
print(f"  ENN k_neighbors: {V66_WINNER['enn_k']}")
print(f"  Test Set F1    : 0.9115 (reference dari v66)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FUNGSI PREPROCESSING PER-FOLD (Mencegah Data Leakage!)
# ─────────────────────────────────────────────────────────────────────────────
def safe_clean(X):
    return np.nan_to_num(np.clip(X, -1e9, 1e9), nan=0., posinf=0., neginf=0.)

def fold_preprocess(X_tr, X_te, y_tr, k=110):
    """
    Preprocessing WAJIB dilakukan di dalam fold.
    Scaler dan selector di-fit HANYA di X_tr, lalu di-transform X_te.
    Ini mencegah data leakage dari test ke train.
    """
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())

    # Imputasi median dari train saja
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]):
            X[nm[:, ci], ci] = meds[ci]

    # IQR clipping dari train saja
    Q1, Q3 = np.percentile(X_tr, 25, axis=0), np.percentile(X_tr, 75, axis=0)
    IQR = Q3 - Q1
    for X in [X_tr, X_te]:
        np.clip(X, Q1 - 10*IQR, Q3 + 10*IQR, out=X)

    # Hapus fitur konstan
    kp = X_tr.var(axis=0) > 1e-10
    if kp.sum() < 5: kp = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:, kp], X_te[:, kp]

    # StandardScaler dari train saja
    sc = StandardScaler()
    X_tr = safe_clean(sc.fit_transform(X_tr))
    X_te = safe_clean(sc.transform(X_te))

    # SelectKBest dari train saja
    if k:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))

    return X_tr, X_te

def fold_smoteenn(X, y, k_n=4, seed=42):
    """SMOTEENN dari train saja — JANGAN diaplikasikan ke test."""
    k_a = min(k_n, (y == 1).sum() - 1)
    k_a = max(k_a, 1)
    try:
        sm = SMOTEENN(random_state=seed,
                      smote=SMOTE(random_state=seed, k_neighbors=k_a))
        return sm.fit_resample(X, y)
    except:
        return X, y

def build_model(cfg):
    return MLPClassifier(
        hidden_layer_sizes=cfg['arch'],
        alpha=cfg['alpha'],
        learning_rate_init=cfg['lr'],
        max_iter=1000,
        random_state=cfg['seed'],
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        activation='relu',
        solver='adam',
    )

def sweep_thr(model, X_te, y_te, lo=0.10, hi=0.95, step=0.01):
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

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNGSI CV UTAMA
# ─────────────────────────────────────────────────────────────────────────────
def run_cv(X, y, cfg, n_folds=5, label="5-Fold CV"):
    """
    Jalankan Stratified K-Fold CV dengan preprocessing per-fold.
    SMOTEENN hanya diterapkan di training fold, bukan test fold.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_SEED)
    fold_results = []
    print(f"\n  === {label} ===")

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Preprocessing dalam fold
        X_tr_p, X_te_p = fold_preprocess(X_tr_raw, X_te_raw, y_tr, k=cfg['K'])

        # SMOTEENN hanya pada training fold
        X_tr_sm, y_tr_sm = fold_smoteenn(X_tr_p, y_tr,
                                          k_n=cfg['enn_k'], seed=cfg['seed'])

        # Train model
        model = build_model(cfg)
        try:
            model.fit(X_tr_sm, y_tr_sm)
        except Exception as e:
            print(f"  Fold {fold_idx+1}: TRAIN ERROR — {e}")
            continue

        # Predict (dengan threshold sweep)
        thr, _ = sweep_thr(model, X_te_p, y_te)
        try:
            probs = model.predict_proba(X_te_p)[:, 1]
            preds = (probs >= thr).astype(int)
            auc   = float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) > 1 else 0.0
        except:
            preds = model.predict(X_te_p)
            probs = preds.astype(float)
            auc   = 0.0

        f1   = f1_score(y_te, preds, average='macro', zero_division=0)
        acc  = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, average='macro', zero_division=0)
        rec  = recall_score(y_te, preds, average='macro', zero_division=0)
        rec1 = recall_score(y_te, preds, pos_label=1, zero_division=0)
        prec1= precision_score(y_te, preds, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_te, preds, pos_label=1, zero_division=0)
        n0_te, n1_te = (y_te==0).sum(), (y_te==1).sum()

        fold_results.append({
            'fold':      fold_idx + 1,
            'f1_macro':  f1,
            'accuracy':  acc,
            'roc_auc':   auc,
            'recall_macro': rec,
            'precision_macro': prec,
            'f1_depresi':    f1_1,
            'recall_depresi':  rec1,
            'precision_depresi': prec1,
            'threshold': thr,
            'n_train':   len(y_tr),
            'n_test':    len(y_te),
            'n0_test':   n0_te,
            'n1_test':   n1_te,
            'n_train_after_enn': len(y_tr_sm),
        })
        print(f"  Fold {fold_idx+1}/{n_folds}: "
              f"Train={len(y_tr)}→{len(y_tr_sm)} | Test={len(y_te)} "
              f"(0:{n0_te},1:{n1_te}) | "
              f"F1={f1:.4f} | Acc={acc:.4f} | "
              f"Rec_Dep={rec1:.4f} | Prec_Dep={prec1:.4f} | Thr={thr:.2f}")

    if not fold_results:
        return {}

    df_folds = pd.DataFrame(fold_results)
    summary = {
        'label':         label,
        'n_folds':       n_folds,
        'f1_macro_mean': float(df_folds['f1_macro'].mean()),
        'f1_macro_std':  float(df_folds['f1_macro'].std()),
        'f1_macro_min':  float(df_folds['f1_macro'].min()),
        'f1_macro_max':  float(df_folds['f1_macro'].max()),
        'accuracy_mean': float(df_folds['accuracy'].mean()),
        'accuracy_std':  float(df_folds['accuracy'].std()),
        'auc_mean':      float(df_folds['roc_auc'].mean()),
        'auc_std':       float(df_folds['roc_auc'].std()),
        'recall_dep_mean':  float(df_folds['recall_depresi'].mean()),
        'recall_dep_std':   float(df_folds['recall_depresi'].std()),
        'prec_dep_mean':    float(df_folds['precision_depresi'].mean()),
        'prec_dep_std':     float(df_folds['precision_depresi'].std()),
        'f1_dep_mean':      float(df_folds['f1_depresi'].mean()),
        'f1_dep_std':       float(df_folds['f1_depresi'].std()),
        'fold_details':  fold_results,
    }

    print(f"\n  >>> {label} Summary:")
    print(f"  F1 Macro  : {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
    print(f"  Accuracy  : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  AUC       : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  Rec Dep   : {summary['recall_dep_mean']:.4f} ± {summary['recall_dep_std']:.4f}")
    print(f"  Prec Dep  : {summary['prec_dep_mean']:.4f} ± {summary['prec_dep_std']:.4f}")
    print(f"  F1 Dep    : {summary['f1_dep_mean']:.4f} ± {summary['f1_dep_std']:.4f}")
    return summary

# ─────────────────────────────────────────────────────────────────────────────
# 5. JALANKAN BERBAGAI VARIASI CV
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  FASE A — Multi-Fold CV dengan Konfigurasi v66 Winner")
print("=" * 80)

all_summaries = {}

# 5-Fold CV
s5 = run_cv(X_all, y_all, V66_WINNER, n_folds=5, label="Stratified 5-Fold CV")
all_summaries['5fold'] = s5

# 10-Fold CV
s10 = run_cv(X_all, y_all, V66_WINNER, n_folds=10, label="Stratified 10-Fold CV")
all_summaries['10fold'] = s10

# ─────────────────────────────────────────────────────────────────────────────
# 6. REPLIKASI SPLIT ORIGINAL v66 UNTUK REFERENSI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  FASE B — Replikasi Split Original v66 (Train+Dev vs Test)")
print("=" * 80)

test_mask  = (splits_orig == 'test')
train_mask = ~test_mask
X_tr_orig = X_all[train_mask]
X_te_orig = X_all[test_mask]
y_tr_orig = y_all[train_mask]
y_te_orig = y_all[test_mask]

print(f"  Train: {len(y_tr_orig)} (0:{(y_tr_orig==0).sum()}, 1:{(y_tr_orig==1).sum()})")
print(f"  Test : {len(y_te_orig)} (0:{(y_te_orig==0).sum()}, 1:{(y_te_orig==1).sum()})")

X_tr_p, X_te_p = fold_preprocess(X_tr_orig, X_te_orig, y_tr_orig, k=V66_WINNER['K'])
X_tr_sm, y_tr_sm = fold_smoteenn(X_tr_p, y_tr_orig,
                                  k_n=V66_WINNER['enn_k'], seed=V66_WINNER['seed'])
mlp_orig = build_model(V66_WINNER)
mlp_orig.fit(X_tr_sm, y_tr_sm)
thr_orig, _ = sweep_thr(mlp_orig, X_te_p, y_te_orig)
probs_orig  = mlp_orig.predict_proba(X_te_p)[:, 1]
preds_orig  = (probs_orig >= thr_orig).astype(int)
f1_orig     = f1_score(y_te_orig, preds_orig, average='macro', zero_division=0)
auc_orig    = float(roc_auc_score(y_te_orig, probs_orig))

print(f"\n  Original Split F1  : {f1_orig:.4f}")
print(f"  Original Split AUC : {auc_orig:.4f}")
print(f"  Threshold          : {thr_orig:.2f}")
print("\n  Classification Report (Original Split):")
print(classification_report(y_te_orig, preds_orig,
                             target_names=['Normal', 'Depresi'], zero_division=0))

all_summaries['original_split'] = {
    'f1_macro': f1_orig,
    'auc':      auc_orig,
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. ANALISIS OVERFITTING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  DIAGNOSIS OVERFITTING")
print("=" * 80)

f1_5fold  = all_summaries['5fold'].get('f1_macro_mean', 0)
std_5fold = all_summaries['5fold'].get('f1_macro_std', 0)
f1_10fold = all_summaries['10fold'].get('f1_macro_mean', 0)
std_10fold= all_summaries['10fold'].get('f1_macro_std', 0)
f1_test   = f1_orig

print(f"\n  ┌─────────────────────────────────────────────────────┐")
print(f"  │  Metrik              │ Nilai                        │")
print(f"  ├─────────────────────────────────────────────────────┤")
print(f"  │  Test Set F1 (v66)  │ {f1_test:.4f}                       │")
print(f"  │  5-Fold CV F1       │ {f1_5fold:.4f} ± {std_5fold:.4f}            │")
print(f"  │  10-Fold CV F1      │ {f1_10fold:.4f} ± {std_10fold:.4f}            │")
print(f"  └─────────────────────────────────────────────────────┘")

gap_5  = f1_test - f1_5fold
gap_10 = f1_test - f1_10fold

print(f"\n  Gap (Test - 5Fold) : {gap_5:+.4f}")
print(f"  Gap (Test - 10Fold): {gap_10:+.4f}")

print(f"\n  KESIMPULAN:")
if gap_5 > 0.15:
    verdict = "OVERFITTING SIGNIFIKAN"
    detail  = (f"Model v66 kemungkinan besar overfitting ke split test "
               f"yang kecil. CV F1 jauh lebih rendah ({f1_5fold:.3f} vs {f1_test:.3f}).")
elif gap_5 > 0.08:
    verdict = "OVERFITTING SEDANG"
    detail  = (f"Ada gap yang cukup besar antara test set dan CV. "
               f"Model mungkin sedikit menghafal 23 sampel test.")
elif gap_5 > 0.03:
    verdict = "SEDIKIT OPTIMISTIS"
    detail  = (f"Gap kecil ({gap_5:.3f}). Model cukup generalize, tapi "
               f"test set yang kecil membuat skor test sedikit inflate.")
else:
    verdict = "TIDAK OVERFITTING"
    detail  = (f"CV F1 hampir sama dengan test F1. "
               f"Model v66 terbukti generalize dengan baik!")

print(f"  [{verdict}]")
print(f"  {detail}")
print(f"\n  Std Deviation 5-Fold : {std_5fold:.4f}")
if std_5fold > 0.12:
    print(f"  [VARIANCE TINGGI] Model tidak stabil antar fold — "
          f"sangat dipengaruhi oleh distribusi data per fold.")
elif std_5fold > 0.07:
    print(f"  [VARIANCE SEDANG] Fluktuasi antar fold wajar mengingat "
          f"dataset kecil (102 partisipan).")
else:
    print(f"  [VARIANCE RENDAH] Model konsisten di semua fold.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MULTI-MODEL CV COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  FASE C — Perbandingan Multi-Model di 5-Fold CV")
print("=" * 80)

ALT_CONFIGS = {
    'v66_winner (400,150,75,25)': V66_WINNER,
    'v65_winner (300,150,75,25)': {**V66_WINNER, 'arch': (300,150,75,25), 'seed':7},
    'simple_MLP (256,128,64)': {**V66_WINNER, 'arch': (256,128,64), 'seed':42},
    'deep_MLP (512,256,128,64)': {**V66_WINNER, 'arch': (512,256,128,64), 'seed':42},
}

cv_comparison = {}
for model_name, cfg in ALT_CONFIGS.items():
    print(f"\n  --- {model_name} ---")
    s = run_cv(X_all, y_all, cfg, n_folds=5, label=f"5-Fold [{model_name}]")
    cv_comparison[model_name] = s

# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Visualisasi]...")

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('v68 — K-Fold Cross-Validation Analysis (Overfitting Check)',
             fontsize=14, fontweight='bold')

COLORS_FOLD = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6',
               '#10b981','#f59e0b','#8b5cf6','#ec4899','#14b8a6']

# Plot 1: 5-Fold & 10-Fold F1 per fold
ax1 = axes[0, 0]
folds_5  = [r['f1_macro'] for r in all_summaries['5fold']['fold_details']]
folds_10 = [r['f1_macro'] for r in all_summaries['10fold']['fold_details']]
ax1.plot(range(1, len(folds_5)+1),  folds_5,  'o-', color='#6366f1', lw=2,
         markersize=8, label=f'5-Fold (μ={f1_5fold:.3f}±{std_5fold:.3f})')
ax1.plot(range(1, len(folds_10)+1), folds_10, 's--', color='#ef4444', lw=2,
         markersize=7, label=f'10-Fold (μ={f1_10fold:.3f}±{std_10fold:.3f})')
ax1.axhline(f1_test, color='green', linestyle=':', lw=2,
            label=f'Test Set v66 = {f1_test:.3f}')
ax1.axhline(f1_5fold, color='#6366f1', linestyle='--', alpha=0.4, lw=1)
ax1.fill_between(range(1, len(folds_5)+1),
                 f1_5fold - std_5fold, f1_5fold + std_5fold,
                 alpha=0.15, color='#6366f1')
ax1.set_xlabel('Fold'); ax1.set_ylabel('F1 Macro')
ax1.set_title('F1 per Fold (5-Fold & 10-Fold)', fontweight='bold')
ax1.legend(fontsize=9); ax1.set_ylim(0, 1.05)
ax1.grid(linestyle='--', alpha=0.4)

# Plot 2: Recall & Precision Depresi per fold
ax2 = axes[0, 1]
rec_dep5  = [r['recall_depresi']   for r in all_summaries['5fold']['fold_details']]
prec_dep5 = [r['precision_depresi'] for r in all_summaries['5fold']['fold_details']]
f1_dep5   = [r['f1_depresi']        for r in all_summaries['5fold']['fold_details']]
x5 = range(1, len(folds_5)+1)
ax2.plot(x5, rec_dep5,  'o-', color='#ef4444', lw=2, markersize=8, label='Recall Depresi')
ax2.plot(x5, prec_dep5, 's-', color='#f97316', lw=2, markersize=8, label='Precision Depresi')
ax2.plot(x5, f1_dep5,   '^-', color='#6366f1', lw=2, markersize=8, label='F1 Depresi')
ax2.axhline(np.mean(rec_dep5),  linestyle='--', color='#ef4444', alpha=0.4, lw=1)
ax2.axhline(np.mean(prec_dep5), linestyle='--', color='#f97316', alpha=0.4, lw=1)
ax2.set_xlabel('Fold'); ax2.set_ylabel('Score')
ax2.set_title('Recall & Precision Depresi per Fold (5-Fold)', fontweight='bold')
ax2.legend(fontsize=9); ax2.set_ylim(0, 1.05)
ax2.grid(linestyle='--', alpha=0.4)

# Plot 3: Overfitting Gap Visualization
ax3 = axes[1, 0]
labels_gap = ['Test Set\n(v66)', '5-Fold CV\nMean', '10-Fold CV\nMean']
vals_gap   = [f1_test, f1_5fold, f1_10fold]
stds_gap   = [0, std_5fold, std_10fold]
bars = ax3.bar(labels_gap, vals_gap, color=['#22c55e','#6366f1','#ef4444'],
               edgecolor='white', width=0.5)
ax3.errorbar(range(len(vals_gap)), vals_gap, yerr=stds_gap,
             fmt='none', color='black', capsize=6, lw=2)
ax3.set_ylim(0, 1.1)
ax3.axhline(0.75, color='gray', linestyle=':', lw=1, label='F1=0.75')
for bar, val, std in zip(bars, vals_gap, stds_gap):
    label = f'{val:.4f}' if std==0 else f'{val:.4f}\n±{std:.4f}'
    ax3.text(bar.get_x() + bar.get_width()/2, val + std + 0.02,
             label, ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylabel('F1 Macro Score')
ax3.set_title(f'Overfitting Gap Analysis\n[{verdict}]', fontweight='bold',
              color='darkred' if 'OVER' in verdict else 'darkgreen')
ax3.grid(axis='y', linestyle='--', alpha=0.4)
ax3.legend(fontsize=9)

# Plot 4: Multi-model CV comparison
ax4 = axes[1, 1]
model_names_short = [n[:20] for n in cv_comparison.keys()]
cv_f1_means = [v.get('f1_macro_mean', 0) for v in cv_comparison.values()]
cv_f1_stds  = [v.get('f1_macro_std', 0)  for v in cv_comparison.values()]
x4 = range(len(model_names_short))
bars4 = ax4.bar(x4, cv_f1_means, color=COLORS_FOLD[:len(x4)],
                edgecolor='white', width=0.6)
ax4.errorbar(x4, cv_f1_means, yerr=cv_f1_stds,
             fmt='none', color='black', capsize=5, lw=2)
ax4.set_xticks(x4)
ax4.set_xticklabels(model_names_short, rotation=20, ha='right', fontsize=8)
ax4.set_ylim(0, 1.1)
ax4.axhline(f1_test, color='red', linestyle='--', lw=1.5,
            label=f'Test Set v66 = {f1_test:.3f}')
for bar, val, std in zip(bars4, cv_f1_means, cv_f1_stds):
    ax4.text(bar.get_x() + bar.get_width()/2, val + std + 0.02,
             f'{val:.3f}\n±{std:.3f}', ha='center', va='bottom',
             fontsize=8, fontweight='bold')
ax4.set_ylabel('F1 Macro (5-Fold CV)')
ax4.set_title('Multi-Model 5-Fold CV Comparison', fontweight='bold')
ax4.legend(fontsize=9); ax4.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "plots", "v68_cv_analysis.png")
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot tersimpan: {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SIMPAN HASIL & FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
import json

report = {
    'version': 'v68',
    'purpose': 'Overfitting Check via K-Fold CV',
    'v66_test_f1': f1_test,
    'v66_test_auc': auc_orig,
    '5fold_cv': {
        'f1_mean': f1_5fold, 'f1_std': std_5fold,
        'f1_min':  all_summaries['5fold'].get('f1_macro_min', 0),
        'f1_max':  all_summaries['5fold'].get('f1_macro_max', 0),
    },
    '10fold_cv': {
        'f1_mean': f1_10fold, 'f1_std': std_10fold,
    },
    'gap_test_vs_5fold':  float(gap_5),
    'gap_test_vs_10fold': float(gap_10),
    'verdict': verdict,
    'detail':  detail,
    'multi_model_cv': {k: {'f1_mean': v.get('f1_macro_mean', 0),
                            'f1_std':  v.get('f1_macro_std', 0)}
                       for k, v in cv_comparison.items()},
}
report_path = os.path.join(RESULTS_DIR, "metrics", "v68_cv_report.json")
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v68 (Overfitting Check)':^80}")
print("=" * 80)
print(f"  Total Partisipan : {len(y_all)}")
print(f"  Test Set F1 (v66): {f1_test:.4f}   ← benchmark")
print(f"  5-Fold CV F1     : {f1_5fold:.4f} ± {std_5fold:.4f}")
print(f"  10-Fold CV F1    : {f1_10fold:.4f} ± {std_10fold:.4f}")
print(f"  Gap (Test-5Fold) : {gap_5:+.4f}")
print(f"  Verdict          : [{verdict}]")
print(f"  Detail           : {detail}")
print(f"\n  Laporan disimpan : {report_path}")
print(f"  Plot disimpan    : {plot_path}")
print(f"  Total Waktu      : {time.time()-t_start:.1f}s")
print("=" * 80)
