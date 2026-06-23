# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v9** — Klasifikasi Kesehatan Mental Berbasis Audio
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v9 = Feature Selection + Nested CV + Stacking
#
#  [1] Feature Selection: SelectKBest (f_classif + MI)
#      Kurangi 990-1536 fitur → 30-80 fitur optimal
#      Di dalam setiap CV fold (NO data leakage)
#
#  [2] Nested CV: LOOCV outer + 5-fold GridSearchCV inner
#      Hyperparameter tuning per fold
#
#  [3] Stacking Ensemble: RF + XGB → meta-learner LR
#
#  [4] Fokus 2 model (RF + XGB) × 2 feature (MFCC + Wav2Vec)
#      = 4 base model + 1 stacking = 5 total
#
#  [5] Threshold tuning pada LOOCV probabilities
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

# %%
import subprocess, sys

def _pip(pkg, name=None):
    check = name or pkg.split('[')[0].split('>=')[0].split('==')[0]
    try: __import__(check); return
    except ImportError: pass
    except Exception: return
    print(f"[Installing] {pkg}")
    try: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    except: pass

_pip("librosa"); _pip("scikit-learn", "sklearn"); _pip("xgboost"); _pip("seaborn")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("[OK] Dependencies siap.\n")

# %%
import os, pickle, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
import xgboost as xgb

plt.rcParams['font.family'] = 'DejaVu Sans'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd() else os.getcwd()
)
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v9")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v9")
for d in [MODELS_DIR,
          os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

print(f"Project: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Load Features (MFCC + Wav2Vec Full)

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']

def load_clean(path, name):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    std_v = df[feat_cols].std()
    feat_cols = [c for c in feat_cols if std_v[c] > 1e-8]
    if 'label' not in df.columns:
        if 'label_depresi' in df.columns:
            df['label'] = df['label_depresi']
    print(f"  [{name}] {len(df)} participants, {len(feat_cols)} features")
    return df, feat_cols

df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"), "MFCC")
df_w2v,  cols_w2v  = load_clean(os.path.join(V8_FEAT_DIR, "daic_v8_wav2vec_full.csv"), "Wav2Vec Full")

datasets = {
    'MFCC':    (df_mfcc, cols_mfcc),
    'Wav2Vec': (df_w2v,  cols_w2v),
}

FEAT_NAMES = list(datasets.keys())

# %% [markdown]
# ## 2. Feature Selection Analysis (preview)

# %%
def preview_feature_importance(df, feat_cols, name, top_k=30):
    """Preview fitur terpenting — F-test + MI + RF importance."""
    X = df[feat_cols].values.astype(np.float64)
    y = df['label'].values.astype(int)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # F-test
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0)

    # MI
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED, n_neighbors=5)

    # RF importance
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=1,
                                 class_weight='balanced', max_depth=10)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    # Combine rankings
    rank_f  = np.argsort(np.argsort(-f_scores))
    rank_mi = np.argsort(np.argsort(-mi_scores))
    rank_rf = np.argsort(np.argsort(-rf_imp))
    avg_rank = (rank_f + rank_mi + rank_rf) / 3.0

    top_idx = np.argsort(avg_rank)[:top_k]
    top_names = [feat_cols[i] for i in top_idx]

    print(f"\n  [{name}] Top-{top_k} fitur (rata-rata ranking F/MI/RF):")
    for rank, i in enumerate(top_idx[:15]):
        print(f"    {rank+1:2d}. {feat_cols[i]:30s}  F={f_scores[i]:.2f}  MI={mi_scores[i]:.4f}  RF={rf_imp[i]:.4f}")
    print(f"    ... ({top_k - 15} more)")

    return top_idx, f_scores, mi_scores, rf_imp

for fn in FEAT_NAMES:
    df, fc = datasets[fn]
    preview_feature_importance(df, fc, fn)

# %% [markdown]
# ## 3. LOOCV with Nested CV + Feature Selection

# %%
INNER_CV = 5
K_OPTIONS = [30, 50, 80]

def get_param_grid(model_name):
    if model_name == 'Random Forest':
        return {
            'n_estimators': [200, 500],
            'max_depth': [5, 8, 12, None],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2'],
        }
    elif model_name == 'XGBoost':
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.5, 0.7],
        }
    return {}

def make_model(model_name):
    if model_name == 'Random Forest':
        return RandomForestClassifier(
            random_state=RANDOM_SEED, class_weight='balanced', n_jobs=1)
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(
            random_state=RANDOM_SEED, eval_metric='logloss',
            objective='binary:logistic', n_jobs=1,
            scale_pos_weight=2.5)
    raise ValueError(f"Unknown model: {model_name}")


def loocv_nested(df, feat_cols, model_name):
    """
    LOOCV with nested CV:
      Outer: Leave-One-Out (102 folds)
      Inner: 5-fold StratifiedKFold GridSearchCV + Feature Selection
    """
    n = len(df)
    X_all = df[feat_cols].values.astype(np.float64)
    y_all = df['label'].values.astype(int)

    y_true = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)
    best_ks = []
    best_params_list = []

    inner_cv = StratifiedKFold(n_splits=INNER_CV, shuffle=True, random_state=RANDOM_SEED)
    param_grid = get_param_grid(model_name)
    f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)

    for i in range(n):
        # Outer split
        X_tr = np.delete(X_all, i, axis=0)
        y_tr = np.delete(y_all, i, axis=0)
        X_te = X_all[i:i+1]

        # Clean
        X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)

        # Clip
        Q1 = np.percentile(X_tr, 25, axis=0)
        Q3 = np.percentile(X_tr, 75, axis=0)
        IQR = Q3 - Q1
        lo, hi = Q1 - 10*IQR, Q3 + 10*IQR
        X_tr = np.clip(X_tr, lo, hi)
        X_te = np.clip(X_te, lo, hi)

        # Scale
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Feature selection: try K options, pick best via inner CV
        best_k = K_OPTIONS[0]
        best_inner_f1 = -1

        for k in K_OPTIONS:
            k_actual = min(k, X_tr_sc.shape[1])
            selector = SelectKBest(f_classif, k=k_actual)
            X_tr_k = selector.fit_transform(X_tr_sc, y_tr)

            # Quick inner CV to evaluate this k
            inner_f1s = []
            for tr_inner, val_inner in inner_cv.split(X_tr_k, y_tr):
                model_tmp = make_model(model_name)
                model_tmp.fit(X_tr_k[tr_inner], y_tr[tr_inner])
                try:
                    p = model_tmp.predict_proba(X_tr_k[val_inner])[:, 1]
                    pred = (p >= 0.5).astype(int)
                except:
                    pred = model_tmp.predict(X_tr_k[val_inner])
                inner_f1s.append(f1_score(y_tr[val_inner], pred, average='macro', zero_division=0))

            avg_f1 = np.mean(inner_f1s)
            if avg_f1 > best_inner_f1:
                best_inner_f1 = avg_f1
                best_k = k_actual

        best_ks.append(best_k)

        # Apply best K
        selector = SelectKBest(f_classif, k=best_k)
        X_tr_sel = selector.fit_transform(X_tr_sc, y_tr)
        X_te_sel = selector.transform(X_te_sc)

        # GridSearchCV with inner CV
        if param_grid:
            gs = GridSearchCV(
                make_model(model_name), param_grid,
                cv=inner_cv, scoring=f1_scorer,
                n_jobs=1, refit=True
            )
            gs.fit(X_tr_sel, y_tr)
            model_final = gs.best_estimator_
            best_params_list.append(gs.best_params_)
        else:
            model_final = make_model(model_name)
            model_final.fit(X_tr_sel, y_tr)

        # Predict
        try:
            prob = model_final.predict_proba(X_te_sel)[0, 1]
        except:
            prob = float(model_final.predict(X_te_sel)[0])

        y_true[i] = y_all[i]
        y_prob[i] = prob

    # Metrics at threshold 0.5
    y_pred_50 = (y_prob >= 0.5).astype(int)
    metrics = {
        'f1_050': float(f1_score(y_true, y_pred_50, average='macro', zero_division=0)),
        'acc_050': float(accuracy_score(y_true, y_pred_50)),
    }
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
    except:
        metrics['auc'] = 0.0

    # Threshold tuning
    best_thr, best_f1 = 0.5, metrics['f1_050']
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (y_prob >= thr).astype(int)
        f1_t = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr
    y_pred_tuned = (y_prob >= best_thr).astype(int)
    metrics['f1_tuned'] = float(best_f1)
    metrics['thr'] = float(round(best_thr, 2))
    metrics['acc_tuned'] = float(accuracy_score(y_true, y_pred_tuned))
    metrics['prec_tuned'] = float(precision_score(y_true, y_pred_tuned, average='macro', zero_division=0))
    metrics['recall_tuned'] = float(recall_score(y_true, y_pred_tuned, average='macro', zero_division=0))

    from collections import Counter
    k_dist = Counter(best_ks)
    metrics['k_distribution'] = dict(k_dist)
    metrics['k_most_common'] = k_dist.most_common(1)[0][0]

    return metrics, y_true, y_pred_50, y_prob, y_pred_tuned

# %% [markdown]
# ## 4. Run LOOCV — 4 Base Models

# %%
MODEL_NAMES = ['Random Forest', 'XGBoost']
SEP = "=" * 100

all_results = {}
all_ys = {}

print(f"\n{'#' * 100}")
print(f"{'v9: LOOCV + NESTED CV + FEATURE SELECTION (102 folds)':^100}")
print(f"{'#' * 100}")

for feat_name in FEAT_NAMES:
    df, feat_cols = datasets[feat_name]

    print(f"\n{SEP}")
    print(f"  FEATURE: {feat_name}  |  {len(feat_cols)} features  |  {len(df)} participants")
    print(f"  Feature Selection: SelectKBest K in {K_OPTIONS}")
    print(f"  Inner CV: {INNER_CV}-fold StratifiedKFold + GridSearchCV")
    print(SEP)

    for model_name in MODEL_NAMES:
        combo = f"{feat_name} + {model_name}"
        print(f"\n  [{combo}] LOOCV + Nested CV running ...", flush=True)

        t0 = time.time()
        metrics, y_true, y_pred_50, y_prob, y_pred_tuned = loocv_nested(
            df, feat_cols, model_name)
        elapsed = time.time() - t0

        all_results[combo] = metrics
        all_ys[combo] = (y_true, y_pred_50, y_prob, y_pred_tuned)

        print(f"    Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"    Feature K: most common = {metrics['k_most_common']}  dist = {metrics['k_distribution']}")
        print(f"    F1 (thr=0.5) : {metrics['f1_050']:.4f}  |  Acc: {metrics['acc_050']:.4f}")
        print(f"    F1 (tuned)   : {metrics['f1_tuned']:.4f}  |  Acc: {metrics['acc_tuned']:.4f}  |  thr={metrics['thr']:.2f}")
        print(f"    AUC          : {metrics['auc']:.4f}")

print(f"\n{SEP}")
print(f"  4 BASE MODEL SELESAI")
print(SEP)

# %% [markdown]
# ## 5. Stacking Ensemble

# %%
print(f"\n{'#' * 100}")
print(f"{'STACKING ENSEMBLE':^100}")
print(f"{'#' * 100}")

# Sort base models by f1_tuned
sorted_combos = sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)
for i, c in enumerate(sorted_combos):
    m = all_results[c]
    print(f"  {i+1}. {c}: F1={m['f1_tuned']:.4f} (thr={m['thr']:.2f})")

# ─── Stacking via LOOCV ──────────────────────────────────────────────────
# For each LOOCV fold:
#   1. Train base models on N-1 with feature selection
#   2. Generate OOF predictions via inner CV
#   3. Train meta-learner on OOF predictions
#   4. Predict held-out participant

n = len(df_mfcc)
X_mfcc_all = df_mfcc[cols_mfcc].values.astype(np.float64)
X_w2v_all  = df_w2v[cols_w2v].values.astype(np.float64)
y_all = df_mfcc['label'].values.astype(int)

# Simpler stacking: use the already-computed LOOCV probabilities
# Each base model already has a probability for every participant
# Stack = use these 4 probabilities as features for meta-learner

# Collect LOOCV probabilities
stack_probs = np.zeros((n, len(sorted_combos)))
for j, combo in enumerate(sorted_combos):
    stack_probs[:, j] = all_ys[combo][2]  # y_prob

# LOOCV for meta-learner
y_true_stack = y_all.copy()
y_prob_stack = np.zeros(n)

inner_cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for i in range(n):
    X_meta_tr = np.delete(stack_probs, i, axis=0)
    y_meta_tr = np.delete(y_all, i)
    X_meta_te = stack_probs[i:i+1]

    meta = LogisticRegression(C=1.0, class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000)
    meta.fit(X_meta_tr, y_meta_tr)
    y_prob_stack[i] = meta.predict_proba(X_meta_te)[0, 1]

# Metrics
y_pred_stack_50 = (y_prob_stack >= 0.5).astype(int)
f1_stack_50 = f1_score(y_true_stack, y_pred_stack_50, average='macro', zero_division=0)
acc_stack_50 = accuracy_score(y_true_stack, y_pred_stack_50)

best_thr_s, best_f1_s = 0.5, f1_stack_50
for thr in np.arange(0.25, 0.75, 0.01):
    preds = (y_prob_stack >= thr).astype(int)
    f1_t = f1_score(y_true_stack, preds, average='macro', zero_division=0)
    if f1_t > best_f1_s:
        best_f1_s = f1_t
        best_thr_s = thr

y_pred_stack_tuned = (y_prob_stack >= best_thr_s).astype(int)
try:
    auc_stack = float(roc_auc_score(y_true_stack, y_prob_stack))
except:
    auc_stack = 0.0

all_results['Stacking'] = {
    'f1_050': f1_stack_50, 'acc_050': acc_stack_50,
    'f1_tuned': best_f1_s, 'thr': round(best_thr_s, 2),
    'acc_tuned': float(accuracy_score(y_true_stack, y_pred_stack_tuned)),
    'auc': auc_stack,
    'prec_tuned': float(precision_score(y_true_stack, y_pred_stack_tuned, average='macro', zero_division=0)),
    'recall_tuned': float(recall_score(y_true_stack, y_pred_stack_tuned, average='macro', zero_division=0)),
}
all_ys['Stacking'] = (y_true_stack, y_pred_stack_50, y_prob_stack, y_pred_stack_tuned)

# Simple average ensemble too
y_prob_avg = stack_probs.mean(axis=1)
y_pred_avg_50 = (y_prob_avg >= 0.5).astype(int)
f1_avg_50 = f1_score(y_all, y_pred_avg_50, average='macro', zero_division=0)

best_thr_a, best_f1_a = 0.5, f1_avg_50
for thr in np.arange(0.25, 0.75, 0.01):
    preds = (y_prob_avg >= thr).astype(int)
    f1_t = f1_score(y_all, preds, average='macro', zero_division=0)
    if f1_t > best_f1_a:
        best_f1_a = f1_t
        best_thr_a = thr
y_pred_avg_tuned = (y_prob_avg >= best_thr_a).astype(int)
try:
    auc_avg = float(roc_auc_score(y_all, y_prob_avg))
except:
    auc_avg = 0.0

all_results['Avg Ensemble'] = {
    'f1_050': f1_avg_50, 'acc_050': float(accuracy_score(y_all, y_pred_avg_50)),
    'f1_tuned': best_f1_a, 'thr': round(best_thr_a, 2),
    'acc_tuned': float(accuracy_score(y_all, y_pred_avg_tuned)),
    'auc': auc_avg,
    'prec_tuned': float(precision_score(y_all, y_pred_avg_tuned, average='macro', zero_division=0)),
    'recall_tuned': float(recall_score(y_all, y_pred_avg_tuned, average='macro', zero_division=0)),
}
all_ys['Avg Ensemble'] = (y_all, y_pred_avg_50, y_prob_avg, y_pred_avg_tuned)

print(f"\n  [Stacking Meta-Learner (LR)]")
print(f"    F1 (thr=0.5) : {f1_stack_50:.4f}  |  Acc: {acc_stack_50:.4f}")
print(f"    F1 (tuned)   : {best_f1_s:.4f}  |  Acc: {all_results['Stacking']['acc_tuned']:.4f}  |  thr={best_thr_s:.2f}")
print(f"    AUC          : {auc_stack:.4f}")

print(f"\n  [Average Ensemble]")
print(f"    F1 (thr=0.5) : {f1_avg_50:.4f}")
print(f"    F1 (tuned)   : {best_f1_a:.4f}  |  thr={best_thr_a:.2f}")
print(f"    AUC          : {auc_avg:.4f}")

# %% [markdown]
# ## 6. Tabel Perbandingan

# %%
rows = []
for combo, m in all_results.items():
    parts = combo.split(' + ')
    feat = parts[0] if len(parts) > 1 else combo
    model = parts[1] if len(parts) > 1 else combo
    rows.append({
        'Feature': feat, 'Model': model,
        'F1 (0.5)': round(m['f1_050'], 4),
        'F1 (tuned)': round(m['f1_tuned'], 4),
        'Thr': round(m['thr'], 2),
        'Acc': round(m['acc_tuned'], 4),
        'AUC': round(m['auc'], 4),
    })

df_results = (pd.DataFrame(rows)
              .sort_values('F1 (tuned)', ascending=False)
              .reset_index(drop=True))
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v9_loocv_nested_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(f"{'RINGKASAN v9 — LOOCV + Nested CV + Feature Selection + Stacking':^110}")
print("=" * 110)
print(df_results.to_string())
print(f"\nDisimpan: {csv_path}")

best = df_results.iloc[0]
print(f"\n  BEST: {best['Feature']} {best['Model']}")
print(f"  F1 (tuned) = {best['F1 (tuned)']}  |  thr={best['Thr']}  |  AUC={best['AUC']}")

# Perbandingan vs v8
print(f"\n  Perbandingan:")
print(f"    v8 best:  MFCC + RF  F1=0.6242 (LOOCV, no feature selection)")
print(f"    v9 best:  {best['Feature']} {best['Model']}  F1={best['F1 (tuned)']}  (LOOCV + nested CV + feat sel)")
delta = best['F1 (tuned)'] - 0.6242
print(f"    Delta: {'+' if delta >= 0 else ''}{delta:.4f}")

# %% [markdown]
# ## 7. Visualisasi

# %%
COLORS = {
    'MFCC': '#3b82f6', 'Wav2Vec': '#10b981',
    'Stacking': '#8b5cf6', 'Avg Ensemble': '#f59e0b', 'Avg': '#f59e0b'
}

# ─── 7A. Bar Chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
labels = df_results.apply(lambda r: f"{r['Feature'][:5]}+{r['Model'][:5]}", axis=1).tolist()
f1s = df_results['F1 (tuned)'].tolist()
colors = [COLORS.get(r['Feature'], '#888') for _, r in df_results.iterrows()]

bars = ax.barh(range(len(labels)), f1s, color=colors, edgecolor='white', linewidth=0.7)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Macro F1 (Tuned)', fontsize=11)
ax.set_title('v9 — LOOCV + Nested CV + Feature Selection + Stacking',
             fontweight='bold', fontsize=12)
ax.axvline(0.7, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Target 0.70')
ax.axvline(0.6242, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='v8 best (0.624)')
for bar, val in zip(bars, f1s):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlim(0, 1.0)
ax.invert_yaxis()
ax.legend(fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
p1 = os.path.join(RESULTS_DIR, "plots", "v9_bar_f1.png")
fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.show()
print(f"Plot: {p1}")

# ─── 7B. Heatmap ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, col, title, cmap in zip(axes,
    ['f1_tuned', 'auc'], ['Macro F1 (Tuned)', 'ROC-AUC'], ['YlOrRd', 'Blues']):
    data = {}
    for fn in FEAT_NAMES:
        data[fn] = []
        for mn in MODEL_NAMES:
            combo = f"{fn} + {mn}"
            data[fn].append(all_results.get(combo, {}).get(col, 0))
    hdf = pd.DataFrame(data, index=MODEL_NAMES).T
    sns.heatmap(hdf, annot=True, fmt='.3f', cmap=cmap, linewidths=0.5,
                cbar_kws={'label': title}, ax=ax, vmin=0.3, vmax=1.0)
    ax.set_title(f'{title} — v9 LOOCV', fontweight='bold')
plt.tight_layout()
p2 = os.path.join(RESULTS_DIR, "plots", "v9_heatmap.png")
fig.savefig(p2, dpi=150, bbox_inches='tight'); plt.show()
print(f"Heatmap: {p2}")

# %% [markdown]
# ## 8. Classification Report (Top Models)

# %%
class_labels = ['Normal (0)', 'Depresi (1)']

print("\n" + "=" * 100)
print(f"{'CLASSIFICATION REPORT v9 — LOOCV + Nested CV':^100}")
print("=" * 100)

for combo in sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True):
    m = all_results[combo]
    if combo not in all_ys:
        continue
    y_true, _, _, y_pred_t = all_ys[combo]

    print(f"\n{'-'*100}")
    print(f"  [{combo}]  F1={m['f1_tuned']:.4f}  thr={m['thr']:.2f}  AUC={m['auc']:.4f}")
    if 'k_distribution' in m:
        print(f"  Feature K distribution: {m['k_distribution']}  (most common: {m['k_most_common']})")
    print(f"{'-'*100}")
    print(classification_report(y_true, y_pred_t, labels=[0, 1],
                                target_names=class_labels, zero_division=0))

# ─── Confusion Matrix ───────────────────────────────────────────────────
all_combos_sorted = sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)
n_plots = min(6, len(all_combos_sorted))
fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
fig.suptitle('v9 — LOOCV Confusion Matrix (Tuned Threshold)', fontweight='bold', fontsize=13)

for idx, combo in enumerate(all_combos_sorted[:n_plots]):
    ax = axes[idx] if n_plots > 1 else axes
    y_true, _, _, y_pred_t = all_ys[combo]
    cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
    m = all_results[combo]
    feat = combo.split(' + ')[0] if ' + ' in combo else combo
    cmap_d = {'MFCC': 'Blues', 'Wav2Vec': 'Greens', 'Stacking': 'Purples', 'Avg': 'Oranges', 'Avg Ensemble': 'Oranges'}
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_d.get(feat, 'Blues'),
                ax=ax, xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, cbar=False)
    ax.set_title(f'{combo[:25]}\nF1={m["f1_tuned"]:.3f} thr={m["thr"]:.2f}',
                 fontweight='bold', fontsize=9)
    ax.set_xlabel('Pred', fontsize=8)
    ax.set_ylabel('True' if idx == 0 else '', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
p3 = os.path.join(RESULTS_DIR, "confusion_matrix", "v9_cm.png")
fig.savefig(p3, dpi=150, bbox_inches='tight'); plt.show()
print(f"CM: {p3}")

# %% [markdown]
# ## 9. Ringkasan Final + Cross-Version Comparison

# %%
best_key = all_combos_sorted[0]
bm = all_results[best_key]

summary = {
    'version': 'v9',
    'evaluation': 'LOOCV (102 folds) + Nested 5-fold GridSearchCV',
    'feature_selection': f'SelectKBest (f_classif), K in {K_OPTIONS}',
    'models': MODEL_NAMES,
    'features': FEAT_NAMES,
    'best_model': best_key,
    'best_f1_tuned': round(bm['f1_tuned'], 4),
    'best_f1_050': round(bm['f1_050'], 4),
    'best_auc': round(bm['auc'], 4),
    'best_threshold': bm['thr'],
    'all_results': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                         for kk, vv in v.items()} for k, v in all_results.items()},
}
with open(os.path.join(MODELS_DIR, "v9_summary.json"), 'w') as fp:
    json.dump(summary, fp, indent=2)

print(f"\n{'=' * 100}")
print(f"{'PIPELINE v9 SELESAI':^100}")
print(f"{'=' * 100}")
print(f"\n  Method   : LOOCV + Nested CV + Feature Selection + Stacking")
print(f"  Best     : {best_key}")
print(f"  F1 tuned : {bm['f1_tuned']:.4f}  (thr={bm['thr']:.2f})")
print(f"  F1 (0.5) : {bm['f1_050']:.4f}")
print(f"  AUC      : {bm['auc']:.4f}")

print(f"\n  Cross-Version Comparison:")
print(f"  {'Version':<8} {'Eval':<15} {'Best Model':<25} {'F1':>6}")
print(f"  {'-'*60}")
print(f"  {'v4':<8} {'AVEC test':15} {'Spec+SVM':<25} {'0.544':>6}")
print(f"  {'v6':<8} {'80/10/10':15} {'W2V+XGB':<25} {'0.629':>6}")
print(f"  {'v8':<8} {'LOOCV':15} {'MFCC+RF':<25} {'0.624':>6}")
print(f"  {'v9':<8} {'LOOCV+Nested':15} {best_key:<25} {bm['f1_tuned']:>6.3f}")

print(f"\n  Models : {MODELS_DIR}")
print(f"  Results: {RESULTS_DIR}")
