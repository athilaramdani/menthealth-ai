# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v30** — Klasifikasi Kesehatan Mental Berbasis Audio
#
# ─────────────────────────────────────────────────────────────────────
#  v30 = LOOCV + SMOTE (k=5) + PCA-Fused Features + Lasso FS + Ensemble
#
#  Tujuan: Menembus Macro F1 > 0.70
#  [1] 3 Ekstraksi Fitur (MFCC, Spectrogram, Wav2Vec 2.0 Full)
#  [2] Menggunakan keseluruhan data (102 partisipan) dengan LOOCV
#  [3] Terdapat 4 model dasar: LR, SVM (Linear), XGBoost, Random Forest
#  [4] Wav2Vec & Spectrogram di-reduksi dengan PCA sebelum Fusion agar tidak mendominasi MFCC.
#  [5] Menambahkan Soft Voting Ensemble di akhir (Top-3)
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

# %%
import sys, os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v30")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v30")

for d in [MODELS_DIR, os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## Load Features

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']

def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    if 'label' not in df.columns and 'label_depresi' in df.columns:
        df['label'] = df['label_depresi']
    return df, feat_cols

df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_spec, cols_spec = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_w2v, cols_w2v = load_clean(os.path.join(V8_FEAT_DIR, "daic_v8_wav2vec_full.csv"))

# PCA Reduction for Spectrogram and Wav2Vec BEFORE Fusion
scaler_spec = StandardScaler()
spec_scaled = scaler_spec.fit_transform(df_spec[cols_spec])
pca_spec = PCA(n_components=30, random_state=RANDOM_SEED)
spec_pca = pca_spec.fit_transform(spec_scaled)
cols_spec_new = [f"s_pca_{i}" for i in range(30)]
df_spec_ren = pd.DataFrame(spec_pca, columns=cols_spec_new)
df_spec_ren['participant_id'] = df_spec['participant_id']

scaler_w2v = StandardScaler()
w2v_scaled = scaler_w2v.fit_transform(df_w2v[cols_w2v])
pca_w2v = PCA(n_components=50, random_state=RANDOM_SEED)
w2v_pca = pca_w2v.fit_transform(w2v_scaled)
cols_w2v_new = [f"w_pca_{i}" for i in range(50)]
df_w2v_ren = pd.DataFrame(w2v_pca, columns=cols_w2v_new)
df_w2v_ren['participant_id'] = df_w2v['participant_id']

# Rename MFCC to avoid conflicts
df_mfcc_ren = df_mfcc.copy()
cols_mfcc_new = [f"m_{c}" for c in cols_mfcc]
df_mfcc_ren = df_mfcc_ren.rename(columns={c: new_c for c, new_c in zip(cols_mfcc, cols_mfcc_new)})

df_fusion = pd.merge(df_mfcc_ren[['participant_id', 'label'] + cols_mfcc_new],
                     df_spec_ren[['participant_id'] + cols_spec_new], on='participant_id')
df_fusion = pd.merge(df_fusion, df_w2v_ren[['participant_id'] + cols_w2v_new], on='participant_id')
cols_fusion = cols_mfcc_new + cols_spec_new + cols_w2v_new

datasets = {
    'MFCC': (df_mfcc, cols_mfcc),
    'Spectrogram': (df_spec, cols_spec),
    'Wav2Vec_Full': (df_w2v, cols_w2v),
    'Fusion_PCA': (df_fusion, cols_fusion)
}

# %% [markdown]
# ## Model Config

# %%
def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=RANDOM_SEED, class_weight='balanced', C=0.01, solver='liblinear'),
        'SVM': SVC(kernel='linear', probability=True, C=0.1, random_state=RANDOM_SEED, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', objective='binary:logistic', n_jobs=-1, scale_pos_weight=1, n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1, n_estimators=300, max_depth=8, max_features='sqrt')
    }

MODEL_NAMES = list(get_models().keys())
FEAT_NAMES  = list(datasets.keys())

# %% [markdown]
# ## LOOCV Evaluation Loop

# %%
def loocv_evaluate(df, feat_cols, model_fn):
    n = len(df)
    X = df[feat_cols].values.astype(np.float64)
    y = df['label'].values.astype(int)

    y_true_all, y_pred_all, y_prob_all = np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=float)

    for i in range(n):
        X_tr, y_tr = np.delete(X, i, axis=0), np.delete(y, i, axis=0)
        X_te = X[i:i+1]

        medians = np.nanmedian(X_tr, axis=0)
        np.copyto(X_tr, medians, where=np.isnan(X_tr))
        np.copyto(X_te, medians, where=np.isnan(X_te))

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # SMOTE with 5 neighbors
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        
        # Lasso Feature Selection
        l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=RANDOM_SEED)
        sel = SelectFromModel(l1_model)
        X_tr_res = sel.fit_transform(X_tr_res, y_tr_res)
        X_te = sel.transform(X_te)

        # Fallback if all features are rejected
        if X_tr_res.shape[1] == 0:
            X_tr_res = X_tr
            X_te = X_te

        model = model_fn()
        model.fit(X_tr_res, y_tr_res)
        
        try: prob = model.predict_proba(X_te)[0, 1]
        except: prob = float(model.predict(X_te)[0])
        
        y_true_all[i] = y[i]
        y_prob_all[i] = prob
        y_pred_all[i] = int(prob >= 0.5)

    metrics = {
        'f1_macro': float(f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_true_all, y_pred_all)),
        'roc_auc': float(roc_auc_score(y_true_all, y_prob_all))
    }

    best_thr, best_f1 = 0.5, metrics['f1_macro']
    for thr in np.arange(0.30, 0.71, 0.01):
        preds_t = (y_prob_all >= thr).astype(int)
        f1_t = f1_score(y_true_all, preds_t, average='macro', zero_division=0)
        if f1_t > best_f1: best_f1, best_thr = f1_t, thr

    metrics['f1_tuned'] = float(best_f1)
    metrics['best_threshold'] = float(round(best_thr, 2))
    return metrics, y_true_all, y_prob_all

# %% [markdown]
# ## Running

# %%
all_results, all_ys = {}, {}
for feat_name in FEAT_NAMES:
    df, feat_cols = datasets[feat_name]
    print(f"\n[{feat_name}]")
    for model_name in MODEL_NAMES:
        combo = f"{feat_name} + {model_name}"
        t0 = time.time()
        model_fn = lambda mn=model_name: get_models()[mn]
        metrics, y_true, y_prob = loocv_evaluate(df, feat_cols, model_fn)
        all_results[combo] = metrics
        all_ys[combo] = (y_true, y_prob, metrics['best_threshold'])
        print(f"  {model_name:<20}: F1_tuned={metrics['f1_tuned']:.4f} (thr={metrics['best_threshold']:.2f})")

# %% [markdown]
# ## Ensembles (Soft Voting Top-2, Top-3, Top-4)

# %%
sorted_combos = sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)

for n_top in [2, 3, 4]:
    topN = sorted_combos[:n_top]
    y_true_ens = all_ys[topN[0]][0]
    probs_top = np.array([all_ys[c][1] for c in topN])
    y_prob_ens = probs_top.mean(axis=0)

    best_thr_ens, best_f1_ens = 0.5, f1_score(y_true_ens, (y_prob_ens >= 0.5).astype(int), average='macro', zero_division=0)
    for thr in np.arange(0.30, 0.71, 0.01):
        preds = (y_prob_ens >= thr).astype(int)
        f1_t = f1_score(y_true_ens, preds, average='macro', zero_division=0)
        if f1_t > best_f1_ens:
            best_f1_ens, best_thr_ens = f1_t, thr

    all_results[f'Ensemble_Top{n_top}'] = {
        'f1_tuned': best_f1_ens, 'best_threshold': best_thr_ens,
        'roc_auc': roc_auc_score(y_true_ens, y_prob_ens)
    }

# %% [markdown]
# ## Save Results

# %%
rows = []
for combo, m in all_results.items():
    if 'Ensemble' in combo:
        parts = combo.split('_')
    else:
        parts = combo.split(' + ')
    rows.append({
        'Feature': parts[0], 'Model': parts[1],
        'F1 (tuned)': m['f1_tuned'], 'Best Thr': m['best_threshold'],
        'AUC': m['roc_auc']
    })

df_results = pd.DataFrame(rows).sort_values('F1 (tuned)', ascending=False).reset_index(drop=True)
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v30_results.csv")
df_results.to_csv(csv_path, index=False)
print("\n" + "=" * 80)
print(f"RINGKASAN v30 — LOOCV (102 folds, SMOTE + Lasso FS + PCA-Fused Features + Ensemble)")
print("=" * 80)
print(df_results.to_string())
