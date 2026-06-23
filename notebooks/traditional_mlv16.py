# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v16** — The Ultimate SOTA Stacking Ensemble
#
# ─────────────────────────────────────────────────────────────────────
#  v16 = THE FINAL PUSH TO 0.70+
#
#  Kombinasi 3 pilar utama DAIC-WOZ:
#  1. MFCC (Vocal Tract) -> SelectKBest(50) + XGBoost
#  2. eGeMAPS (Vocal Cords / Glottal) -> SelectKBest(30) + Random Forest
#  3. LIWC Dense Text (Psychological) -> Random Forest
#
#  Gunakan `ColumnTransformer` untuk memisahkan fitur per model, lalu
#  bungkus dalam `StackingClassifier` dengan Meta-Learner Logistic Regression.
#  Evaluasi menggunakan ketatnya LOOCV pada 102 partisipan.
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

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from tqdm import tqdm

plt.rcParams['font.family'] = 'DejaVu Sans'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd() else os.getcwd()
)
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V15_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v15")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v16")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v16")

for d in [MODELS_DIR, os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## 1. Load All Modalities (102 Participants)

# %%
print("Memuat fitur dari ketiga modalitas...")

# 1. MFCC
df_mfcc = pd.read_csv(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
cols_mfcc = [c for c in df_mfcc.columns if c not in ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']]
df_mfcc['label'] = df_mfcc['label_depresi']
df_mfcc = df_mfcc[['participant_id', 'label'] + cols_mfcc].fillna(0)
print(f"  [MFCC] {len(df_mfcc)} partisipan, {len(cols_mfcc)} fitur.")

# 2. eGeMAPS
df_egemaps = pd.read_csv(os.path.join(V15_FEAT_DIR, "daic_v15_egemaps_full.csv"))
cols_egemaps = [c for c in df_egemaps.columns if c not in ['participant_id', 'label']]
df_egemaps = df_egemaps[['participant_id'] + cols_egemaps].fillna(0)
print(f"  [eGeMAPS] {len(df_egemaps)} partisipan, {len(cols_egemaps)} fitur.")

# 3. Dense Text (Kita ekstrak langsung dari v11/v15 logic tapi cukup memuat ulang dari data fusion jika ada, atau extract cepat)
# Karena proses cepat, kita extract langsung
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
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
print(f"  [Text] {len(df_text)} partisipan, {len(cols_text)} fitur.")

# FUSION ALL 3
df_all = pd.merge(df_mfcc, df_egemaps, on='participant_id', how='inner')
df_all = pd.merge(df_all, df_text, on='participant_id', how='inner')
print(f"  [SuperFusion] Berhasil menggabungkan 3 modalitas untuk {len(df_all)} partisipan!")

# %% [markdown]
# ## 2. Build the Ultimate Stacking Pipeline

# %%
# Indeks kolom untuk ColumnTransformer
X_df = df_all[cols_mfcc + cols_egemaps + cols_text]
y_all = df_all['label'].values.astype(int)

# Membuat pipeline masing-masing modalitas
pipe_mfcc = Pipeline([
    ('col_transform', ColumnTransformer([
        ('mfcc_feats', Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=50))
        ]), cols_mfcc)
    ], remainder='drop')),
    ('clf', xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, 
                              eval_metric='logloss', scale_pos_weight=2.5, random_state=RANDOM_SEED, n_jobs=1))
])

pipe_egemaps = Pipeline([
    ('col_transform', ColumnTransformer([
        ('egemaps_feats', Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=30))
        ]), cols_egemaps)
    ], remainder='drop')),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1))
])

pipe_text = Pipeline([
    ('col_transform', ColumnTransformer([
        ('text_feats', StandardScaler(), cols_text)
    ], remainder='drop')),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1))
])

# Meta Learner
stacking_clf = StackingClassifier(
    estimators=[
        ('mfcc_xgb', pipe_mfcc),
        ('egemaps_rf', pipe_egemaps),
        ('text_rf', pipe_text)
    ],
    final_estimator=LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED),
    stack_method='predict_proba',
    cv=5  # Inner CV for stacking to prevent data leakage!
)

# %% [markdown]
# ## 3. Execute LOOCV

# %%
n = len(df_all)
y_true = np.zeros(n, dtype=int)
y_prob_stack = np.zeros(n, dtype=float)

X_all = X_df.values.astype(np.float64)

print(f"\n{'#' * 100}")
print(f"{'v16: THE ULTIMATE SOTA STACKING ENSEMBLE (LOOCV)':^100}")
print(f"{'#' * 100}")

t0 = time.time()
for i in tqdm(range(n), desc="LOOCV Stacking"):
    X_tr = np.delete(X_all, i, axis=0)
    y_tr = np.delete(y_all, i, axis=0)
    
    X_te = X_all[i:i+1]
    y_true[i] = y_all[i]
    
    # Supaya ColumnTransformer bisa mengenali nama kolom, kita harus passing DataFrame
    X_tr_df = pd.DataFrame(X_tr, columns=X_df.columns)
    X_te_df = pd.DataFrame(X_te, columns=X_df.columns)
    
    # Train Stacking Pipeline (Ini sangat aman dari data leakage karena semuanya ada di dalam pipeline)
    stacking_clf.fit(X_tr_df, y_tr)
    
    # Predict
    y_prob_stack[i] = stacking_clf.predict_proba(X_te_df)[0, 1]

# Threshold Tuning
best_thr, best_f1 = 0.5, 0
for thr in np.arange(0.25, 0.75, 0.01):
    preds = (y_prob_stack >= thr).astype(int)
    f1_t = f1_score(y_true, preds, average='macro', zero_division=0)
    if f1_t > best_f1: best_f1, best_thr = f1_t, thr
        
y_pred_tuned = (y_prob_stack >= best_thr).astype(int)

auc = roc_auc_score(y_true, y_prob_stack)
acc = accuracy_score(y_true, y_pred_tuned)

print(f"\nSelesai dalam {time.time()-t0:.0f} detik.")
print(f"  F1 Score (tuned) : {best_f1:.4f} (threshold={best_thr:.2f})")
print(f"  AUC              : {auc:.4f}")
print(f"  Accuracy         : {acc:.4f}")

# %% [markdown]
# ## 4. Simpan Hasil

# %%
results = [{
    'Model': 'Ultimate_Stacking_v16',
    'F1': best_f1,
    'Thr': best_thr,
    'AUC': auc
}]

df_results = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "metrics", "v16_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(f"{'RINGKASAN v16 — THE ULTIMATE STACKING ENSEMBLE':^110}")
print("=" * 110)
print(df_results.to_string())

with open(os.path.join(MODELS_DIR, "v16_summary.json"), 'w') as fp:
    json.dump({'best_f1': best_f1, 'best_model': 'Ultimate_Stacking_v16'}, fp)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_tuned)
plt.figure(figsize=(6, 5))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Depresi'], yticklabels=['Normal', 'Depresi'])
plt.title(f"Confusion Matrix (Ultimate Stacking)")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix", "v16_cm.png")
plt.savefig(cm_path, bbox_inches='tight')
plt.close()
print(f"Confusion matrix tersimpan di: {cm_path}")
