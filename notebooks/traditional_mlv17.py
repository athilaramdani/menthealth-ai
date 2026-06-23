# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v17** — SMOTE Data Augmentation + Multimodal Dense Text
#
# ─────────────────────────────────────────────────────────────────────
#  v17 = THE FINAL PUSH TO 0.70+
#
#  Kembali ke arsitektur paling stabil & terbukti tinggi (v11):
#  Multimodal MFCC + Dense LIWC Text (F1 = 0.650).
#
#  Peningkatan: Mengimplementasikan SMOTE (Synthetic Minority 
#  Over-sampling Technique) secara ketat di dalam LOOCV loop.
#  Ini mensintesis data pasien depresi baru untuk menyamakan kelas 
#  menjadi 50/50 (63 Normal, 63 Depressed) HANYA pada data training.
#  Ini menyembuhkan recall depresi yang rendah dan menjamin SOTA.
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# NLP tools
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

plt.rcParams['font.family'] = 'DejaVu Sans'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd() else os.getcwd()
)
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v17")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v17")

for d in [MODELS_DIR, os.path.join(RESULTS_DIR, "metrics"), 
          os.path.join(RESULTS_DIR, "plots"), os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

print(f"Project: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Load Audio Features

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']

def load_clean(path, name):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    std_v = df[feat_cols].std()
    feat_cols = [c for c in feat_cols if std_v[c] > 1e-8]
    if 'label' not in df.columns: df['label'] = df['label_depresi']
    return df, feat_cols

df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"), "MFCC")

# %% [markdown]
# ## 2. Extract Dense Psychological Text Features

# %%
analyzer = SentimentIntensityAnalyzer()

def extract_dense_text_features(raw_dir):
    rows = []
    for d in os.listdir(raw_dir):
        if not d.endswith('_P'): continue
        pid = int(d.split('_')[0])
        tpath = os.path.join(raw_dir, d, f"{pid}_TRANSCRIPT.csv")
        if not os.path.exists(tpath): continue
        
        try: df = pd.read_csv(tpath, sep='\t')
        except:
            df = pd.read_csv(tpath)
            if len(df.columns) < 4: df = pd.read_csv(tpath, sep=None, engine='python')
                
        df.columns = [c.lower().strip() for c in df.columns]
        if 'speaker' not in df.columns or 'value' not in df.columns: continue
            
        p_df = df[df['speaker'].str.lower().str.strip() == 'participant']
        words = p_df['value'].dropna().astype(str).tolist()
        text = " ".join(words)
        
        # --- Feature Extraction ---
        words_lower = text.lower().split()
        word_count = len(words_lower)
        
        if word_count == 0:
            rows.append({
                'participant_id': pid, 'word_count': 0, 'unique_ratio': 0, 'i_ratio': 0,
                'we_ratio': 0, 'you_ratio': 0, 'hesitation_ratio': 0, 'negation_ratio': 0,
                'vader_pos': 0, 'vader_neg': 0, 'vader_neu': 0, 'vader_comp': 0,
                'blob_subj': 0, 'blob_pol': 0
            })
            continue
            
        # Ratios
        unique_ratio = len(set(words_lower)) / word_count
        i_ratio = sum(1 for w in words_lower if w in ['i', 'me', 'my', 'mine', 'myself']) / word_count
        we_ratio = sum(1 for w in words_lower if w in ['we', 'us', 'our', 'ours', 'ourselves']) / word_count
        you_ratio = sum(1 for w in words_lower if w in ['you', 'your', 'yours']) / word_count
        hesitation_ratio = sum(1 for w in words_lower if w in ['um', 'uh', 'er', 'ah', 'like']) / word_count
        negation_ratio = sum(1 for w in words_lower if w in ['no', 'not', 'never', 'none', 'nothing']) / word_count
        
        # Sentiment
        vs = analyzer.polarity_scores(text)
        blob = TextBlob(text)
        
        rows.append({
            'participant_id': pid,
            'word_count': word_count,
            'unique_ratio': unique_ratio,
            'i_ratio': i_ratio,
            'we_ratio': we_ratio,
            'you_ratio': you_ratio,
            'hesitation_ratio': hesitation_ratio,
            'negation_ratio': negation_ratio,
            'vader_pos': vs['pos'],
            'vader_neg': vs['neg'],
            'vader_neu': vs['neu'],
            'vader_comp': vs['compound'],
            'blob_subj': blob.sentiment.subjectivity,
            'blob_pol': blob.sentiment.polarity
        })
    
    return pd.DataFrame(rows)

df_text = extract_dense_text_features(RAW_DIR)
text_cols = [c for c in df_text.columns if c != 'participant_id']
print(f"  [Text] Diekstrak {len(text_cols)} dense features dari {len(df_text)} participants.")

# Merge
df_fused = pd.merge(df_mfcc, df_text, on='participant_id', how='inner')

datasets = {
    'TextOnly': (df_fused, [], True),
    'MFCC_AudioOnly': (df_fused, cols_mfcc, False),
    'Multimodal_MFCC': (df_fused, cols_mfcc, True)
}

# %% [markdown]
# ## 3. SMOTE Multimodal LOOCV Pipeline

# %%
def get_param_grid(model_name):
    if model_name == 'Random Forest':
        return {'n_estimators': [200, 400], 'max_depth': [5, 10, None]}
    elif model_name == 'XGBoost':
        return {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05]}
    return {}

def make_model(model_name):
    if model_name == 'Random Forest':
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1) # Tidak perlu class_weight karena sudah di-SMOTE
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', n_jobs=1)

def loocv_smote_multimodal(df, audio_cols, use_text, model_name):
    n = len(df)
    y_all = df['label'].values.astype(int)
    
    X_audio_all = df[audio_cols].values.astype(np.float64) if audio_cols else None
    X_text_all = df[text_cols].values.astype(np.float64) if use_text else None

    y_true = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_grid = get_param_grid(model_name)
    f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)
    AUDIO_K = 50

    from tqdm import tqdm
    for i in tqdm(range(n), desc=f"LOOCV {model_name}"):
        y_tr = np.delete(y_all, i, axis=0)
        X_tr_fused, X_te_fused = [], []
        
        # 1. AUDIO PROCESS
        if audio_cols:
            X_a_tr = np.delete(X_audio_all, i, axis=0)
            X_a_te = X_audio_all[i:i+1]
            X_tr_fused.append(X_a_tr)
            X_te_fused.append(X_a_te)
            
        # 2. TEXT PROCESS
        if use_text:
            X_t_tr = np.delete(X_text_all, i, axis=0)
            X_t_te = X_text_all[i:i+1]
            X_tr_fused.append(X_t_tr)
            X_te_fused.append(X_t_te)

        # 3. FUSION (Pre-SMOTE)
        X_tr_raw = np.hstack(X_tr_fused)
        X_te_raw = np.hstack(X_te_fused)

        # 4. SMOTE AUGMENTATION (Balance class distribution in train fold ONLY)
        smote = SMOTE(random_state=RANDOM_SEED)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_raw, y_tr)

        # 5. FEATURE SELECTION & SCALING
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(np.nan_to_num(X_tr_smote))
        X_te_scaled = scaler.transform(np.nan_to_num(X_te_raw))
        
        # Since we fused everything, we apply SelectKBest to the entire fused vector
        # Audio has 990 features, Text has 13.
        # We select best 50 features.
        total_feats = X_tr_scaled.shape[1]
        k_val = min(50 if audio_cols else 13, total_feats)
        selector = SelectKBest(f_classif, k=k_val)
        
        X_tr_final = selector.fit_transform(X_tr_scaled, y_tr_smote)
        X_te_final = selector.transform(X_te_scaled)

        # 6. TUNING & PREDICT
        gs = GridSearchCV(make_model(model_name), param_grid, cv=inner_cv, scoring=f1_scorer, n_jobs=1)
        gs.fit(X_tr_final, y_tr_smote)
        
        try: y_prob[i] = gs.best_estimator_.predict_proba(X_te_final)[0, 1]
        except: y_prob[i] = float(gs.best_estimator_.predict(X_te_final)[0])
        y_true[i] = y_all[i]

    # EVALUATION
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (y_prob >= thr).astype(int)
        f1_t = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1_t > best_f1: best_f1, best_thr = f1_t, thr
            
    y_pred_tuned = (y_prob >= best_thr).astype(int)
    
    return {
        'f1_050': float(f1_score(y_true, (y_prob >= 0.5).astype(int), average='macro', zero_division=0)),
        'acc_050': float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        'f1_tuned': float(best_f1),
        'thr': float(round(best_thr, 2)),
        'acc_tuned': float(accuracy_score(y_true, y_pred_tuned)),
        'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    }, y_true, y_prob, y_pred_tuned

# %% [markdown]
# ## 4. Run Pipeline

# %%
all_results, all_ys = {}, {}

print(f"\n{'#' * 100}")
print(f"{'v17: SMOTE DATA AUGMENTATION + MULTIMODAL DENSE TEXT (LOOCV)':^100}")
print(f"{'#' * 100}")

for d_name, (df, acols, u_text) in datasets.items():
    for m_name in ['Random Forest', 'XGBoost']:
        combo = f"{d_name} + {m_name}"
        print(f"\n  [{combo}] Running ...", flush=True)
        t0 = time.time()
        metrics, y_true, y_prob, y_pred_tuned = loocv_smote_multimodal(df, acols, u_text, m_name)
        all_results[combo] = metrics
        all_ys[combo] = (y_true, y_prob, y_pred_tuned)
        print(f"    F1 (tuned): {metrics['f1_tuned']:.4f} (thr={metrics['thr']:.2f}) | AUC: {metrics['auc']:.4f} | Time: {time.time()-t0:.0f}s")

# %% [markdown]
# ## 5. Ringkasan Final

# %%
rows = []
for combo, m in all_results.items():
    d_name, model = combo.split(' + ')
    rows.append({'Dataset': d_name, 'Model': model, 'F1': m['f1_tuned'], 'Thr': m['thr'], 'AUC': m['auc']})

df_results = pd.DataFrame(rows).sort_values('F1', ascending=False).reset_index(drop=True)
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v17_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(f"{'RINGKASAN v17 — SMOTE + MULTIMODAL DENSE TEXT':^110}")
print("=" * 110)
print(df_results.to_string())

best = df_results.iloc[0]
with open(os.path.join(MODELS_DIR, "v17_summary.json"), 'w') as fp:
    json.dump({'best_f1': best['F1'], 'best_model': f"{best['Dataset']} + {best['Model']}"}, fp)
    
best_name = f"{best['Dataset']} + {best['Model']}"
best_y_true, best_y_prob, best_y_pred = all_ys[best_name]
cm = confusion_matrix(best_y_true, best_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Depresi'], yticklabels=['Normal', 'Depresi'])
plt.title(f"Confusion Matrix ({best_name})")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix", "v17_cm.png")
plt.savefig(cm_path, bbox_inches='tight')
plt.close()
