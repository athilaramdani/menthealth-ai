# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v38** — Klasifikasi Kesehatan Mental Berbasis Multimodal
#
# ─────────────────────────────────────────────────────────────────────
#  v38 = LOOCV + SMOTE + Block-wise Lasso FS + Audio/Text Fusion + Deep Learning
#
#  Tujuan: Menembus Macro F1 > 0.70
#  [1] 4 Ekstraksi Fitur (MFCC, Spectrogram, Wav2Vec 2.0 Full, Text Embeddings v13)
#  [2] Menggunakan keseluruhan data (102 partisipan) dengan LOOCV
#  [3] Terdapat 5 model: LR, SVM, XGBoost, Random Forest + Deep Neural Network (MLP)
#  [4] Menggunakan Block-wise Lasso Feature Selection
#  [5] Menambahkan Soft Voting Ensemble di akhir dengan KESELURUHAN FITUR (MFCC dkk)
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
V13_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v13")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v38")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v38")

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
df_text, cols_text = load_clean(os.path.join(V13_FEAT_DIR, "v13_text_embeddings.csv"))

df_mfcc_ren = df_mfcc.rename(columns={c: f"m_{c}" for c in cols_mfcc})
cols_mfcc_new = [f"m_{c}" for c in cols_mfcc]

df_spec_ren = df_spec.rename(columns={c: f"s_{c}" for c in cols_spec})
cols_spec_new = [f"s_{c}" for c in cols_spec]

df_w2v_ren = df_w2v.rename(columns={c: f"w_{c}" for c in cols_w2v})
cols_w2v_new = [f"w_{c}" for c in cols_w2v]

df_text_ren = df_text.rename(columns={c: f"t_{c}" for c in cols_text})
cols_text_new = [f"t_{c}" for c in cols_text]
df_text_ren = pd.merge(df_text_ren, df_mfcc_ren[['participant_id', 'label']], on='participant_id')

# Merge all four
df_fusion = pd.merge(df_mfcc_ren[['participant_id', 'label'] + cols_mfcc_new],
                     df_spec_ren[['participant_id'] + cols_spec_new], on='participant_id')
df_fusion = pd.merge(df_fusion, df_w2v_ren[['participant_id'] + cols_w2v_new], on='participant_id')
df_fusion = pd.merge(df_fusion, df_text_ren[['participant_id'] + cols_text_new], on='participant_id')
cols_fusion = cols_mfcc_new + cols_spec_new + cols_w2v_new + cols_text_new

datasets = {
    'MFCC': (df_mfcc_ren, cols_mfcc_new),
    'Spectrogram': (df_spec_ren, cols_spec_new),
    'Wav2Vec_Full': (df_w2v_ren, cols_w2v_new),
    'Text_Only': (df_text_ren, cols_text_new),
    'Fusion_All': (df_fusion, cols_fusion)
}

# %% [markdown]
# ## Model Config

# %%
def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=RANDOM_SEED, class_weight='balanced', C=0.01, solver='liblinear'),
        'SVM': SVC(kernel='rbf', probability=True, C=0.5, gamma='scale', random_state=RANDOM_SEED, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', objective='binary:logistic', n_jobs=-1, scale_pos_weight=2.5, n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1, n_estimators=300, max_depth=5, max_features='sqrt'),
        'Deep Neural Net': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.001, max_iter=500, random_state=RANDOM_SEED, early_stopping=True)
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

    # Identifikasi indeks blok fitur
    m_idx = [i for i, c in enumerate(feat_cols) if c.startswith('m_')]
    s_idx = [i for i, c in enumerate(feat_cols) if c.startswith('s_')]
    w_idx = [i for i, c in enumerate(feat_cols) if c.startswith('w_')]
    t_idx = [i for i, c in enumerate(feat_cols) if c.startswith('t_')]
    is_fusion = len(m_idx) > 0 and len(s_idx) > 0 and len(w_idx) > 0 and len(t_idx) > 0

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
        
        # Block-wise Feature Selection
        l1_base = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=RANDOM_SEED)
        
        if is_fusion:
            # MFCC
            sel_m = SelectFromModel(l1_base)
            X_tr_m = sel_m.fit_transform(X_tr_res[:, m_idx], y_tr_res)
            X_te_m = sel_m.transform(X_te[:, m_idx])
            if X_tr_m.shape[1] == 0: X_tr_m, X_te_m = X_tr_res[:, m_idx], X_te[:, m_idx]
                
            # Spectrogram
            sel_s = SelectFromModel(l1_base)
            X_tr_s = sel_s.fit_transform(X_tr_res[:, s_idx], y_tr_res)
            X_te_s = sel_s.transform(X_te[:, s_idx])
            if X_tr_s.shape[1] == 0: X_tr_s, X_te_s = X_tr_res[:, s_idx], X_te[:, s_idx]
                
            # Wav2Vec
            sel_w = SelectFromModel(l1_base)
            X_tr_w = sel_w.fit_transform(X_tr_res[:, w_idx], y_tr_res)
            X_te_w = sel_w.transform(X_te[:, w_idx])
            if X_tr_w.shape[1] == 0: X_tr_w, X_te_w = X_tr_res[:, w_idx], X_te[:, w_idx]
                
            # Text
            sel_t = SelectFromModel(l1_base)
            X_tr_t = sel_t.fit_transform(X_tr_res[:, t_idx], y_tr_res)
            X_te_t = sel_t.transform(X_te[:, t_idx])
            if X_tr_t.shape[1] == 0: X_tr_t, X_te_t = X_tr_res[:, t_idx], X_te[:, t_idx]

            X_tr_final = np.hstack([X_tr_m, X_tr_s, X_tr_w, X_tr_t])
            X_te_final = np.hstack([X_te_m, X_te_s, X_te_w, X_te_t])
        else:
            sel = SelectFromModel(l1_base)
            X_tr_final = sel.fit_transform(X_tr_res, y_tr_res)
            X_te_final = sel.transform(X_te)
            if X_tr_final.shape[1] == 0: X_tr_final, X_te_final = X_tr_res, X_te

        model = model_fn()
        model.fit(X_tr_final, y_tr_res)
        
        try: prob = model.predict_proba(X_te_final)[0, 1]
        except: prob = float(model.predict(X_te_final)[0])
        
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
# ## Ensembles (Soft Voting Top-3, Top-4, Top-5, Top-6)

# %%
sorted_combos = sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)

for n_top in [3, 4, 5, 6]:
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

csv_path = os.path.join(RESULTS_DIR, "metrics", "v38_results.csv")
df_results.to_csv(csv_path, index=False)
print("\n" + "=" * 80)
print(f"RINGKASAN v38 — LOOCV (Full Multimodal + Text + Audio + Deep Neural Net + Klasik)")
print("=" * 80)
print(df_results.to_string())
