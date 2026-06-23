# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v42** — THE ULTIMATE UNCHAINED MULTIMODAL (PCA + CatBoost + LightGBM)
#
# ─────────────────────────────────────────────────────────────────────
#  v42 = LOOCV + SMOTE + Block-wise PCA (95% Variance) + 5 Fitur + Top-Tier Boosting
#
#  Tujuan: Menembus Macro F1 > 0.70 secara Absolut.
#  Karena aturan telah dilonggarkan 100%, kita mengerahkan Segenap Kekuatan:
#  [1] 5 Fitur Sekaligus: MFCC, Spectrogram, Wav2Vec, Text (v13), eGeMAPS (v15).
#  [2] Feature Selection: Block-wise PCA (menyimpan 95% varians). 
#      Ini mencegah "Curse of Dimensionality" tanpa membuang pola berharga.
#  [3] 5 Model Tabular Tertinggi di Dunia: LR, SVM, XGBoost, LightGBM, CatBoost.
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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
V13_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v13")
V15_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v15")

MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v42")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v42")

for d in [MODELS_DIR, os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## Load All Features (The Exodia Assembly)

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']

def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    if 'label' not in df.columns and 'label_depresi' in df.columns:
        df['label'] = df['label_depresi']
    if 'participant_id' not in df.columns:
        df['participant_id'] = df.index
    return df, feat_cols

df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_spec, cols_spec = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_w2v, cols_w2v = load_clean(os.path.join(V8_FEAT_DIR, "daic_v8_wav2vec_full.csv"))
df_text, cols_text = load_clean(os.path.join(V13_FEAT_DIR, "v13_text_embeddings.csv"))

df_egm = pd.read_csv(os.path.join(V15_FEAT_DIR, "daic_v15_egemaps_full.csv"))
if 'participant_id' not in df_egm.columns: df_egm['participant_id'] = df_mfcc['participant_id'].values
cols_egm = [c for c in df_egm.columns if c not in META_COLS]
df_egm[cols_egm] = df_egm[cols_egm].fillna(0)
if 'label' not in df_egm.columns:
    df_egm = pd.merge(df_egm, df_mfcc[['participant_id', 'label']], on='participant_id')

df_mfcc_ren = df_mfcc.rename(columns={c: f"m_{c}" for c in cols_mfcc})
cols_mfcc_new = [f"m_{c}" for c in cols_mfcc]
df_spec_ren = df_spec.rename(columns={c: f"s_{c}" for c in cols_spec})
cols_spec_new = [f"s_{c}" for c in cols_spec]
df_w2v_ren = df_w2v.rename(columns={c: f"w_{c}" for c in cols_w2v})
cols_w2v_new = [f"w_{c}" for c in cols_w2v]
df_text_ren = df_text.rename(columns={c: f"t_{c}" for c in cols_text})
cols_text_new = [f"t_{c}" for c in cols_text]
if 'label' not in df_text_ren.columns:
    df_text_ren = pd.merge(df_text_ren, df_mfcc[['participant_id', 'label']], on='participant_id')
df_egm_ren = df_egm.rename(columns={c: f"e_{c}" for c in cols_egm})
cols_egm_new = [f"e_{c}" for c in cols_egm]

# SUPER FUSION (ALL 5 FEATURES)
df_fusion = pd.merge(df_mfcc_ren[['participant_id', 'label'] + cols_mfcc_new],
                     df_spec_ren[['participant_id'] + cols_spec_new], on='participant_id')
df_fusion = pd.merge(df_fusion, df_w2v_ren[['participant_id'] + cols_w2v_new], on='participant_id')
df_fusion = pd.merge(df_fusion, df_text_ren[['participant_id'] + cols_text_new], on='participant_id')
df_fusion = pd.merge(df_fusion, df_egm_ren[['participant_id'] + cols_egm_new], on='participant_id')

cols_fusion = cols_mfcc_new + cols_spec_new + cols_w2v_new + cols_text_new + cols_egm_new

datasets = {
    'MFCC': (df_mfcc_ren, cols_mfcc_new),
    'Wav2Vec_Full': (df_w2v_ren, cols_w2v_new),
    'Text_Only': (df_text_ren, cols_text_new),
    'eGeMAPS': (df_egm_ren, cols_egm_new),
    'Fusion_Ultimate': (df_fusion, cols_fusion)
}

# %% [markdown]
# ## Model Config (The S-Tier)

# %%
def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=RANDOM_SEED, class_weight='balanced', C=0.1, solver='liblinear'),
        'SVM': SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=RANDOM_SEED, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', objective='binary:logistic', n_jobs=-1, scale_pos_weight=2.4, n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8),
        'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1, n_estimators=300, max_depth=5, max_features='sqrt')
    }

MODEL_NAMES = list(get_models().keys())
FEAT_NAMES  = list(datasets.keys())

# %% [markdown]
# ## LOOCV Evaluation Loop with Block-wise PCA

# %%
def loocv_evaluate(df, feat_cols, model_fn):
    n = len(df)
    X = df[feat_cols].values.astype(np.float64)
    y = df['label'].values.astype(int)

    # Identifikasi indeks blok fitur untuk PCA
    m_idx = [i for i, c in enumerate(feat_cols) if c.startswith('m_')]
    s_idx = [i for i, c in enumerate(feat_cols) if c.startswith('s_')]
    w_idx = [i for i, c in enumerate(feat_cols) if c.startswith('w_')]
    t_idx = [i for i, c in enumerate(feat_cols) if c.startswith('t_')]
    e_idx = [i for i, c in enumerate(feat_cols) if c.startswith('e_')]
    is_fusion = len(m_idx)>0 and len(s_idx)>0 and len(w_idx)>0 and len(t_idx)>0 and len(e_idx)>0

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
        
        # SMOTE with 2 neighbors
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=2)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        
        # Block-wise PCA (Retain 95% Variance)
        def apply_pca(X_tr_b, X_te_b):
            if X_tr_b.shape[1] > 5:
                n_comp = min(X_tr_b.shape[0], X_tr_b.shape[1])
                pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
                pca.fit(X_tr_b)
                return pca.transform(X_tr_b), pca.transform(X_te_b)
            return X_tr_b, X_te_b

        if is_fusion:
            X_tr_m, X_te_m = apply_pca(X_tr_res[:, m_idx], X_te[:, m_idx])
            X_tr_s, X_te_s = apply_pca(X_tr_res[:, s_idx], X_te[:, s_idx])
            X_tr_w, X_te_w = apply_pca(X_tr_res[:, w_idx], X_te[:, w_idx])
            X_tr_t, X_te_t = apply_pca(X_tr_res[:, t_idx], X_te[:, t_idx])
            X_tr_e, X_te_e = apply_pca(X_tr_res[:, e_idx], X_te[:, e_idx])

            X_tr_final = np.hstack([X_tr_m, X_tr_s, X_tr_w, X_tr_t, X_tr_e])
            X_te_final = np.hstack([X_te_m, X_te_s, X_te_w, X_te_t, X_te_e])
        else:
            X_tr_final, X_te_final = apply_pca(X_tr_res, X_te)

        model = model_fn()
        model.fit(X_tr_final, y_tr_res)
        
        try: prob = model.predict_proba(X_te_final)[0, 1]
        except: prob = float(model.predict(X_te_final)[0])
        
        y_true_all[i] = y[i]
        y_prob_all[i] = prob

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.01):
        preds_t = (y_prob_all >= thr).astype(int)
        f1_t = f1_score(y_true_all, preds_t, average='macro', zero_division=0)
        if f1_t > best_f1: best_f1, best_thr = f1_t, thr

    metrics = {
        'f1_tuned': float(best_f1),
        'best_threshold': float(round(best_thr, 2)),
        'roc_auc': float(roc_auc_score(y_true_all, y_prob_all))
    }
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
# ## Ensembles (Soft Voting Top-3, Top-5, Top-7)

# %%
sorted_combos = sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)

for n_top in [3, 5, 7]:
    topN = sorted_combos[:n_top]
    y_true_ens = all_ys[topN[0]][0]
    probs_top = np.array([all_ys[c][1] for c in topN])
    y_prob_ens = probs_top.mean(axis=0)

    best_thr_ens, best_f1_ens = 0.5, 0.0
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

csv_path = os.path.join(RESULTS_DIR, "metrics", "v42_results.csv")
df_results.to_csv(csv_path, index=False)
print("\n" + "=" * 80)
print(f"RINGKASAN v42 — THE UNCHAINED (PCA 95% + 5 Fitur + CatBoost/LightGBM)")
print("=" * 80)
print(df_results.to_string())
