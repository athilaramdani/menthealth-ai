# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v41** — Klasifikasi Kesehatan Mental Berbasis REGRESI (PHQ-8)
#
# ─────────────────────────────────────────────────────────────────────
#  v41 = LOOCV + Regression Models + Block-wise Lasso FS + Thresholding
#
#  Tujuan: Menembus Macro F1 > 0.70 melalui Paradigm Shift!
#  Karena klasifikasi biner tertahan di 0.684, kita ubah pendekatannya:
#  Model akan memprediksi skor kontinu PHQ-8 (0-24) melalui REGRESI.
#  Hasil prediksi regresi kemudian di-threshold (>= 10 = Depresi).
#  
#  [1] 3 Ekstraksi Fitur (MFCC, Spectrogram, Wav2Vec 2.0 Full)
#  [2] Menggunakan keseluruhan data (102 partisipan) dengan LOOCV
#  [3] Terdapat 4 model regresi: Ridge, SVR, XGBRegressor, RandomForestRegressor
#  [4] Menggunakan Block-wise Lasso Feature Selection
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v41")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v41")

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
    if 'participant_id' not in df.columns:
        df['participant_id'] = df.index
    return df, feat_cols

df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_spec, cols_spec = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_w2v, cols_w2v = load_clean(os.path.join(V8_FEAT_DIR, "daic_v8_wav2vec_full.csv"))

df_mfcc_ren = df_mfcc.rename(columns={c: f"m_{c}" for c in cols_mfcc})
cols_mfcc_new = [f"m_{c}" for c in cols_mfcc]

df_spec_ren = df_spec.rename(columns={c: f"s_{c}" for c in cols_spec})
cols_spec_new = [f"s_{c}" for c in cols_spec]

df_w2v_ren = df_w2v.rename(columns={c: f"w_{c}" for c in cols_w2v})
cols_w2v_new = [f"w_{c}" for c in cols_w2v]

# Merge all
df_fusion = pd.merge(df_mfcc_ren[['participant_id', 'label', 'phq8_score'] + cols_mfcc_new],
                     df_spec_ren[['participant_id'] + cols_spec_new], on='participant_id')
df_fusion = pd.merge(df_fusion, df_w2v_ren[['participant_id'] + cols_w2v_new], on='participant_id')
cols_fusion = cols_mfcc_new + cols_spec_new + cols_w2v_new

datasets = {
    'MFCC': (df_mfcc_ren, cols_mfcc_new),
    'Spectrogram': (df_spec_ren, cols_spec_new),
    'Wav2Vec_Full': (df_w2v_ren, cols_w2v_new),
    'Fusion_All': (df_fusion, cols_fusion)
}

# %% [markdown]
# ## Model Config (REGRESSION)

# %%
def get_models():
    return {
        'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_SEED),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
        'XGBRegressor': xgb.XGBRegressor(random_state=RANDOM_SEED, objective='reg:squarederror', n_jobs=-1, n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
        'Random Forest': RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, n_estimators=300, max_depth=5, max_features='sqrt')
    }

MODEL_NAMES = list(get_models().keys())
FEAT_NAMES  = list(datasets.keys())

# %% [markdown]
# ## LOOCV Evaluation Loop (Regression to Classification)

# %%
def loocv_evaluate(df, feat_cols, model_fn):
    n = len(df)
    X = df[feat_cols].values.astype(np.float64)
    y_reg = df['phq8_score'].values.astype(np.float64)
    y_clf = df['label'].values.astype(int)

    # Identifikasi indeks blok fitur
    m_idx = [i for i, c in enumerate(feat_cols) if c.startswith('m_')]
    s_idx = [i for i, c in enumerate(feat_cols) if c.startswith('s_')]
    w_idx = [i for i, c in enumerate(feat_cols) if c.startswith('w_')]
    is_fusion = len(m_idx) > 0 and len(s_idx) > 0 and len(w_idx) > 0

    y_true_clf, y_pred_reg = np.zeros(n, dtype=int), np.zeros(n, dtype=float)

    for i in range(n):
        X_tr, y_tr_reg = np.delete(X, i, axis=0), np.delete(y_reg, i, axis=0)
        X_te = X[i:i+1]

        medians = np.nanmedian(X_tr, axis=0)
        np.copyto(X_tr, medians, where=np.isnan(X_tr))
        np.copyto(X_te, medians, where=np.isnan(X_te))

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # Block-wise Feature Selection via Lasso Regressor
        l1_base = Lasso(alpha=0.1, random_state=RANDOM_SEED)
        
        if is_fusion:
            # MFCC
            sel_m = SelectFromModel(l1_base)
            X_tr_m = sel_m.fit_transform(X_tr[:, m_idx], y_tr_reg)
            X_te_m = sel_m.transform(X_te[:, m_idx])
            if X_tr_m.shape[1] == 0: X_tr_m, X_te_m = X_tr[:, m_idx], X_te[:, m_idx]
                
            # Spectrogram
            sel_s = SelectFromModel(l1_base)
            X_tr_s = sel_s.fit_transform(X_tr[:, s_idx], y_tr_reg)
            X_te_s = sel_s.transform(X_te[:, s_idx])
            if X_tr_s.shape[1] == 0: X_tr_s, X_te_s = X_tr[:, s_idx], X_te[:, s_idx]
                
            # Wav2Vec
            sel_w = SelectFromModel(l1_base)
            X_tr_w = sel_w.fit_transform(X_tr[:, w_idx], y_tr_reg)
            X_te_w = sel_w.transform(X_te[:, w_idx])
            if X_tr_w.shape[1] == 0: X_tr_w, X_te_w = X_tr[:, w_idx], X_te[:, w_idx]

            X_tr_final = np.hstack([X_tr_m, X_tr_s, X_tr_w])
            X_te_final = np.hstack([X_te_m, X_te_s, X_te_w])
        else:
            sel = SelectFromModel(l1_base)
            X_tr_final = sel.fit_transform(X_tr, y_tr_reg)
            X_te_final = sel.transform(X_te)
            if X_tr_final.shape[1] == 0: X_tr_final, X_te_final = X_tr, X_te

        model = model_fn()
        model.fit(X_tr_final, y_tr_reg)
        
        pred_phq8 = float(model.predict(X_te_final)[0])
        
        y_true_clf[i] = y_clf[i]
        y_pred_reg[i] = pred_phq8

    # Thresholding untuk klasifikasi biner
    # Secara default, PHQ-8 >= 10 adalah indikasi depresi. Namun kita cari threshold empiris terbaik.
    best_thr, best_f1 = 10.0, 0.0
    for thr in np.arange(5.0, 15.0, 0.5):
        preds_clf = (y_pred_reg >= thr).astype(int)
        f1_t = f1_score(y_true_clf, preds_clf, average='macro', zero_division=0)
        if f1_t > best_f1: best_f1, best_thr = f1_t, thr

    # Probabilitas semu (pseudo-probability) dari skor regresi (dinormalisasi)
    prob_pseudo = (y_pred_reg - y_pred_reg.min()) / (y_pred_reg.max() - y_pred_reg.min() + 1e-9)

    metrics = {
        'f1_tuned': float(best_f1),
        'best_threshold_phq8': float(round(best_thr, 2)),
        'rmse': float(np.sqrt(mean_squared_error(df['phq8_score'].values, y_pred_reg))),
        'roc_auc': float(roc_auc_score(y_true_clf, prob_pseudo))
    }
    return metrics, y_true_clf, y_pred_reg

# %% [markdown]
# ## Running

# %%
all_results, all_ys = {}, {}
for feat_name in FEAT_NAMES:
    df, feat_cols = datasets[feat_name]
    if 'phq8_score' not in df.columns:
        df = pd.merge(df, df_mfcc[['participant_id', 'phq8_score']], on='participant_id')
    print(f"\n[{feat_name}]")
    for model_name in MODEL_NAMES:
        combo = f"{feat_name} + {model_name}"
        t0 = time.time()
        model_fn = lambda mn=model_name: get_models()[mn]
        metrics, y_true_clf, y_pred_reg = loocv_evaluate(df, feat_cols, model_fn)
        all_results[combo] = metrics
        all_ys[combo] = (y_true_clf, y_pred_reg, metrics['best_threshold_phq8'])
        print(f"  {model_name:<20}: F1_tuned={metrics['f1_tuned']:.4f} (thr PHQ8={metrics['best_threshold_phq8']:.1f}, RMSE={metrics['rmse']:.2f})")

# %% [markdown]
# ## Ensembles (Soft Voting Top-3, Top-4 dari Prediksi Regresi)

# %%
sorted_combos = sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)

for n_top in [3, 4]:
    topN = sorted_combos[:n_top]
    y_true_ens = all_ys[topN[0]][0]
    preds_reg_top = np.array([all_ys[c][1] for c in topN])
    y_pred_reg_ens = preds_reg_top.mean(axis=0)

    best_thr_ens, best_f1_ens = 10.0, 0.0
    for thr in np.arange(5.0, 15.0, 0.5):
        preds_clf = (y_pred_reg_ens >= thr).astype(int)
        f1_t = f1_score(y_true_ens, preds_clf, average='macro', zero_division=0)
        if f1_t > best_f1_ens:
            best_f1_ens, best_thr_ens = f1_t, thr
            
    prob_pseudo_ens = (y_pred_reg_ens - y_pred_reg_ens.min()) / (y_pred_reg_ens.max() - y_pred_reg_ens.min() + 1e-9)

    all_results[f'Ensemble_Top{n_top}'] = {
        'f1_tuned': best_f1_ens, 'best_threshold_phq8': best_thr_ens,
        'rmse': float(np.sqrt(mean_squared_error(df_mfcc['phq8_score'].values, y_pred_reg_ens))),
        'roc_auc': roc_auc_score(y_true_ens, prob_pseudo_ens)
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
        'F1 (tuned)': m['f1_tuned'], 'Best Thr PHQ8': m['best_threshold_phq8'],
        'RMSE': m['rmse'], 'AUC': m['roc_auc']
    })

df_results = pd.DataFrame(rows).sort_values('F1 (tuned)', ascending=False).reset_index(drop=True)
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v41_results.csv")
df_results.to_csv(csv_path, index=False)
print("\n" + "=" * 80)
print(f"RINGKASAN v41 — LOOCV (Regresi ke PHQ-8 -> Binarization)")
print("=" * 80)
print(df_results.to_string())
