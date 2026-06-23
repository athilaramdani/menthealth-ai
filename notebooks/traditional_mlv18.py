# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v18** — Lasso (L1) Feature Selection + XGBoost
#
# ─────────────────────────────────────────────────────────────────────
#  v18 = THE BREAKTHROUGH STRATEGY
#
#  Masalah dengan pendekatan sebelumnya (v9, v11, v16):
#  Kita menggunakan `SelectKBest`, yang menyeleksi fitur secara 
#  independen. Karena fitur audio saling berkorelasi erat (redundant),
#  `SelectKBest` akan memilih 50 fitur yang informasinya itu-itu saja,
#  dan mengabaikan fitur unik dari eGeMAPS atau Text.
#
#  Solusi v18:
#  Gabungkan seluruh 1091 fitur (MFCC + eGeMAPS + Text).
#  Gunakan L1 Regularization (Lasso) via `SelectFromModel(LogisticRegression)`.
#  Lasso secara matematis memaksa bobot fitur yang redundant menjadi 0, 
#  sehingga hanya fitur-fitur independen yang paling prediktif 
#  (gabungan harmonis dari suara & teks) yang lolos ke XGBoost.
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

# %%
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
)
import xgboost as xgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm

plt.rcParams['font.family'] = 'DejaVu Sans'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd() else os.getcwd()
)
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V15_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v15")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v18")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v18")

for d in [MODELS_DIR, os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## 1. Load All Features

# %%
print("Memuat seluruh 1091 fitur...")

# 1. MFCC
df_mfcc = pd.read_csv(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
cols_mfcc = [c for c in df_mfcc.columns if c not in ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']]
df_mfcc['label'] = df_mfcc['label_depresi']
df_mfcc = df_mfcc[['participant_id', 'label'] + cols_mfcc].fillna(0)
print(f"  [MFCC] {len(cols_mfcc)} fitur.")

# 2. eGeMAPS
df_egemaps = pd.read_csv(os.path.join(V15_FEAT_DIR, "daic_v15_egemaps_full.csv"))
cols_egemaps = [c for c in df_egemaps.columns if c not in ['participant_id', 'label']]
df_egemaps = df_egemaps[['participant_id'] + cols_egemaps].fillna(0)
print(f"  [eGeMAPS] {len(cols_egemaps)} fitur.")

# 3. Dense Text
analyzer = SentimentIntensityAnalyzer()
def extract_dense_text_features():
    rows = []
    label_dict = dict(zip(df_mfcc['participant_id'], df_mfcc['label']))
    for d in os.listdir(RAW_DIR):
        if not d.endswith('_P'): continue
        pid = int(d.split('_')[0])
        if pid not in label_dict: continue
        tpath = os.path.join(RAW_DIR, d, f"{pid}_TRANSCRIPT.csv")
        if not os.path.exists(tpath): continue
        try: df = pd.read_csv(tpath, sep='\t')
        except:
            df = pd.read_csv(tpath)
            if len(df.columns) < 4: df = pd.read_csv(tpath, sep=None, engine='python')
        df.columns = [c.lower().strip() for c in df.columns]
        if 'speaker' not in df.columns or 'value' not in df.columns: continue
        p_df = df[df['speaker'].str.lower().str.strip() == 'participant']
        words = p_df['value'].dropna().astype(str).tolist()
        text = " ".join(words).lower()
        words_lower = text.split()
        word_count = len(words_lower)
        if word_count == 0: continue
        rows.append({
            'participant_id': pid,
            'word_count': word_count,
            'unique_ratio': len(set(words_lower)) / word_count,
            'i_ratio': sum(1 for w in words_lower if w in ['i', 'me', 'my', 'mine', 'myself']) / word_count,
            'we_ratio': sum(1 for w in words_lower if w in ['we', 'us', 'our', 'ours', 'ourselves']) / word_count,
            'you_ratio': sum(1 for w in words_lower if w in ['you', 'your', 'yours']) / word_count,
            'hesitation_ratio': sum(1 for w in words_lower if w in ['um', 'uh', 'er', 'ah', 'like']) / word_count,
            'negation_ratio': sum(1 for w in words_lower if w in ['no', 'not', 'never', 'none', 'nothing']) / word_count,
            'vader_pos': analyzer.polarity_scores(text)['pos'],
            'vader_neg': analyzer.polarity_scores(text)['neg'],
            'vader_neu': analyzer.polarity_scores(text)['neu'],
            'vader_comp': analyzer.polarity_scores(text)['compound'],
            'blob_subj': TextBlob(text).sentiment.subjectivity,
            'blob_pol': TextBlob(text).sentiment.polarity
        })
    return pd.DataFrame(rows)

df_text = extract_dense_text_features()
cols_text = [c for c in df_text.columns if c != 'participant_id']
print(f"  [Text] {len(cols_text)} fitur.")

# FUSION ALL 3
df_all = pd.merge(df_mfcc, df_egemaps, on='participant_id', how='inner')
df_all = pd.merge(df_all, df_text, on='participant_id', how='inner')
ALL_COLS = cols_mfcc + cols_egemaps + cols_text
print(f"  [SuperFusion] {len(df_all)} partisipan dengan TOTAL {len(ALL_COLS)} fitur.")

# %% [markdown]
# ## 2. LOOCV + Lasso Feature Selection

# %%
def get_param_grid(model_name):
    if model_name == 'Random Forest':
        return {'n_estimators': [200, 300], 'max_depth': [5, 10, None]}
    elif model_name == 'XGBoost':
        return {'n_estimators': [100, 150], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05]}
    return {}

def make_model(model_name):
    if model_name == 'Random Forest':
        return RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=1)
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', scale_pos_weight=2.5, n_jobs=1)

def loocv_lasso(df, cols, model_name):
    n = len(df)
    y_all = df['label'].values.astype(int)
    X_all = df[cols].values.astype(np.float64)

    y_true = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_grid = get_param_grid(model_name)
    f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)
    
    selected_counts = []

    for i in tqdm(range(n), desc=f"LOOCV {model_name}"):
        y_tr = np.delete(y_all, i, axis=0)
        X_tr = np.delete(X_all, i, axis=0)
        X_te = X_all[i:i+1]
        
        # Scaling
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(np.nan_to_num(X_tr))
        X_te_scaled = scaler.transform(np.nan_to_num(X_te))
        
        # Lasso L1 Feature Selection (C parameter controls sparsity. Lower C = fewer features selected)
        # We use C=0.05 to aggressively force sparsity (around 20-50 features selected)
        lasso = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.05, random_state=RANDOM_SEED)
        selector = SelectFromModel(lasso)
        
        X_tr_final = selector.fit_transform(X_tr_scaled, y_tr)
        X_te_final = selector.transform(X_te_scaled)
        
        num_selected = X_tr_final.shape[1]
        selected_counts.append(num_selected)

        # Tuning & Predict
        if num_selected > 0:
            gs = GridSearchCV(make_model(model_name), param_grid, cv=inner_cv, scoring=f1_scorer, n_jobs=1)
            gs.fit(X_tr_final, y_tr)
            try: y_prob[i] = gs.best_estimator_.predict_proba(X_te_final)[0, 1]
            except: y_prob[i] = float(gs.best_estimator_.predict(X_te_final)[0])
        else:
            y_prob[i] = 0.5 # Fail safe if 0 features selected
            
        y_true[i] = y_all[i]

    print(f"  [Lasso] Rata-rata fitur yang dipertahankan: {np.mean(selected_counts):.1f} dari {len(cols)}")

    # EVALUATION
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (y_prob >= thr).astype(int)
        f1_t = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1_t > best_f1: best_f1, best_thr = f1_t, thr
            
    y_pred_tuned = (y_prob >= best_thr).astype(int)
    
    return {
        'f1_tuned': float(best_f1),
        'thr': float(round(best_thr, 2)),
        'acc_tuned': float(accuracy_score(y_true, y_pred_tuned)),
        'auc': float(roc_auc_score(y_true, y_prob))
    }, y_true, y_prob, y_pred_tuned

# %% [markdown]
# ## 3. Run Pipeline

# %%
all_results, all_ys = {}, {}

print(f"\n{'#' * 100}")
print(f"{'v18: LASSO FEATURE SELECTION + XGBOOST/RF':^100}")
print(f"{'#' * 100}")

for m_name in ['Random Forest', 'XGBoost']:
    print(f"\n  [AllFeatures (1091) + {m_name}] Running ...", flush=True)
    t0 = time.time()
    metrics, y_true, y_prob, y_pred_tuned = loocv_lasso(df_all, ALL_COLS, m_name)
    all_results[m_name] = metrics
    all_ys[m_name] = (y_true, y_prob, y_pred_tuned)
    print(f"    F1 (tuned): {metrics['f1_tuned']:.4f} (thr={metrics['thr']:.2f}) | AUC: {metrics['auc']:.4f} | Time: {time.time()-t0:.0f}s")

# %% [markdown]
# ## 4. Final Summary

# %%
rows = []
for model, m in all_results.items():
    rows.append({'Model': model, 'F1': m['f1_tuned'], 'Thr': m['thr'], 'AUC': m['auc']})

df_results = pd.DataFrame(rows).sort_values('F1', ascending=False).reset_index(drop=True)
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v18_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(f"{'RINGKASAN v18 — LASSO + ALL MODALITIES':^110}")
print("=" * 110)
print(df_results.to_string())

best = df_results.iloc[0]
with open(os.path.join(MODELS_DIR, "v18_summary.json"), 'w') as fp:
    json.dump({'best_f1': best['F1'], 'best_model': best['Model']}, fp)
    
best_name = best['Model']
best_y_true, best_y_prob, best_y_pred = all_ys[best_name]
cm = confusion_matrix(best_y_true, best_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Depresi'], yticklabels=['Normal', 'Depresi'])
plt.title(f"Confusion Matrix ({best_name})")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix", "v18_cm.png")
plt.savefig(cm_path, bbox_inches='tight')
plt.close()
print(f"\nConfusion matrix tersimpan di: {cm_path}")
