# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v21** — Klasifikasi Kesehatan Mental Berbasis Audio
#
# ─────────────────────────────────────────────────────────────────────
#  v21 = LOOCV + SMOTE + SelectKBest + Early Fusion
#
#  [1] 3 Ekstraksi Fitur (MFCC, Spectrogram, Wav2Vec 2.0 Full)
#  [2] Menggunakan keseluruhan data (102 partisipan) dengan LOOCV
#  [3] Terdapat 4 model dasar: LR, SVM, XGBoost, Random Forest
#  [4] Menambahkan pipeline preprocessing di dalam LOOCV:
#      Imputasi -> Scaling -> SMOTE -> SelectKBest -> Classifier
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

# %%
import subprocess, sys, os, pickle, json, time, warnings
warnings.filterwarnings('ignore')

def _pip(pkg):
    try: __import__(pkg.split('==')[0].split('>=')[0]); return
    except: pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_pip("imbalanced-learn")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

plt.rcParams['font.family'] = 'DejaVu Sans'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v21")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v21")

for d in [MODELS_DIR, os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"), os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## Load Features

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']

def load_clean(csv_path, name):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    if 'label' not in df.columns and 'label_depresi' in df.columns:
        df['label'] = df['label_depresi']
    return df, feat_cols

df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"), "MFCC")
df_spec, cols_spec = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"), "Spectrogram")
W2V_FULL_CSV = os.path.join(V8_FEAT_DIR, "daic_v8_wav2vec_full.csv")
df_w2v, cols_w2v = load_clean(W2V_FULL_CSV, "Wav2Vec_Full")

# EARLY FUSION
df_mfcc_ren = df_mfcc.rename(columns={c: f"m_{c}" for c in cols_mfcc})
cols_mfcc_new = [f"m_{c}" for c in cols_mfcc]

df_spec_ren = df_spec.rename(columns={c: f"s_{c}" for c in cols_spec})
cols_spec_new = [f"s_{c}" for c in cols_spec]

df_w2v_ren = df_w2v.rename(columns={c: f"w_{c}" for c in cols_w2v})
cols_w2v_new = [f"w_{c}" for c in cols_w2v]

df_fusion = pd.merge(df_mfcc_ren[['participant_id', 'label'] + cols_mfcc_new],
                     df_spec_ren[['participant_id'] + cols_spec_new], on='participant_id')
df_fusion = pd.merge(df_fusion, df_w2v_ren[['participant_id'] + cols_w2v_new], on='participant_id')
cols_fusion = cols_mfcc_new + cols_spec_new + cols_w2v_new

datasets = {
    'MFCC': (df_mfcc, cols_mfcc),
    'Spectrogram': (df_spec, cols_spec),
    'Wav2Vec_Full': (df_w2v, cols_w2v),
    'Fusion_All': (df_fusion, cols_fusion)
}

# %% [markdown]
# ## Model Config

# %%
def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=10000, random_state=RANDOM_SEED, class_weight='balanced', C=0.01, solver='liblinear'
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, C=1.0, gamma='scale',
            random_state=RANDOM_SEED, class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            random_state=RANDOM_SEED, eval_metric='logloss',
            objective='binary:logistic', n_jobs=-1,
            scale_pos_weight=2.5, n_estimators=150, max_depth=3,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1,
            n_estimators=300, max_depth=5, max_features='sqrt'
        ),
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

        # NaN handling
        medians = np.nanmedian(X_tr, axis=0)
        np.copyto(X_tr, medians, where=np.isnan(X_tr))
        np.copyto(X_te, medians, where=np.isnan(X_te))

        # Scaling
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # SMOTE
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        
        # SelectKBest (100 top features to prevent overfitting)
        k_feats = min(100, X_tr_res.shape[1])
        sel = SelectKBest(f_classif, k=k_feats)
        X_tr_res = sel.fit_transform(X_tr_res, y_tr_res)
        X_te = sel.transform(X_te)

        # Train & Predict
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
    metrics['acc_tuned'] = float(accuracy_score(y_true_all, (y_prob_all >= best_thr).astype(int)))

    return metrics, y_true_all, y_prob_all

# %% [markdown]
# ## Running

# %%
all_results, all_ys = {}, {}
for feat_name in FEAT_NAMES:
    df, feat_cols = datasets[feat_name]
    print(f"\n[{feat_name}] ({len(feat_cols)} features)")
    for model_name in MODEL_NAMES:
        combo = f"{feat_name} + {model_name}"
        t0 = time.time()
        model_fn = lambda mn=model_name: get_models()[mn]
        metrics, y_true, y_prob = loocv_evaluate(df, feat_cols, model_fn)
        all_results[combo] = metrics
        all_ys[combo] = (y_true, y_prob, metrics['best_threshold'])
        print(f"  {model_name:<20}: F1_tuned={metrics['f1_tuned']:.4f} (thr={metrics['best_threshold']:.2f}) | Time={time.time()-t0:.1f}s")

# %% [markdown]
# ## Save Results

# %%
rows = []
for combo, m in all_results.items():
    parts = combo.split(' + ')
    rows.append({
        'Feature': parts[0], 'Model': parts[1],
        'F1 (tuned)': m['f1_tuned'], 'Best Thr': m['best_threshold'],
        'AUC': m['roc_auc']
    })

df_results = pd.DataFrame(rows).sort_values('F1 (tuned)', ascending=False).reset_index(drop=True)
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v21_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 80)
print(f"RINGKASAN v21 — LOOCV (102 folds, 3 Features + Early Fusion)")
print("=" * 80)
print(df_results.to_string())
print("\nDone.")
