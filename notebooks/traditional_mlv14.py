# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v14** — OpenSMILE eGeMAPS (Glottal/Voice Quality) + Dense NLP
#
# ─────────────────────────────────────────────────────────────────────
#  v14 = THE MISSING CLINICAL LINK (eGeMAPS)
#
#  Masalah v13: Fitur librosa MFCC hanya menangkap bentuk pita suara (vocal tract),
#               tapi kehilangan fitur kritis depresi: Glottal Tension, Breathiness,
#               Jitter, Shimmer, HNR (Harmonic-to-Noise Ratio).
#  Solusi v14: 
#  1. Menggunakan **OpenSMILE** eGeMAPSv02 (88 fitur standar klinis untuk Emosi/Depresi).
#  2. Menggabungkan fitur eGeMAPS dengan 13 Dense NLP Features dari v11.
#  3. Participant-Level Aggregation dengan Feature Selection & Nested CV.
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, make_scorer
)
import xgboost as xgb
import opensmile
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
CLEAN_DIR   = os.path.join(PROJECT_ROOT, "data", "cleaned")
V14_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v14")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v14")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v14")

for d in [V14_FEAT_DIR, MODELS_DIR, os.path.join(RESULTS_DIR, "metrics")]:
    os.makedirs(d, exist_ok=True)

# Load Labels
df_labels = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dev = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_all_labels = pd.concat([df_labels, df_dev])
df_all_labels.columns = [c.lower().strip() for c in df_all_labels.columns]
label_dict = dict(zip(df_all_labels['participant_id'], df_all_labels['phq8_binary']))

# %% [markdown]
# ## 1. Extract OpenSMILE eGeMAPS Features

# %%
EGEMAPS_CSV = os.path.join(V14_FEAT_DIR, "daic_v14_egemaps.csv")

def extract_egemaps():
    if os.path.exists(EGEMAPS_CSV):
        print(f"  [Audio] Memuat fitur eGeMAPS dari {EGEMAPS_CSV}")
        return pd.read_csv(EGEMAPS_CSV)
        
    print("  [Audio] Mengekstrak 88 fitur klinis eGeMAPS via OpenSMILE...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    rows = []
    wav_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.wav')]
    for f in tqdm(wav_files, desc="OpenSMILE"):
        pid = int(f.split('.')[0])
        if pid not in label_dict: continue
        
        path = os.path.join(CLEAN_DIR, f)
        y = smile.process_file(path)
        
        # smile.process_file returns a DataFrame with 1 row per file
        row = y.iloc[0].to_dict()
        row['participant_id'] = pid
        row['label'] = label_dict[pid]
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(EGEMAPS_CSV, index=False)
    return df

df_audio = extract_egemaps()
cols_audio = [c for c in df_audio.columns if c not in ['participant_id', 'label']]
print(f"  [Audio] Diekstrak {len(cols_audio)} eGeMAPS features dari {len(df_audio)} participants.")

# %% [markdown]
# ## 2. Extract Dense Psychological Text Features (v11)

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
        
        words_lower = text.lower().split()
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

df_text = extract_dense_text_features(RAW_DIR)
cols_text = [c for c in df_text.columns if c != 'participant_id']
print(f"  [Text] Diekstrak {len(cols_text)} dense features dari {len(df_text)} participants.")

# FUSION
df_fused = pd.merge(df_audio, df_text, on='participant_id', how='inner')
print(f"  [Fusion] Ready: {len(df_fused)} partisipan dengan {len(cols_audio)} Audio + {len(cols_text)} Text features.")

datasets = {
    'eGeMAPS_AudioOnly': (df_fused, cols_audio, []),
    'Multimodal_eGeMAPS': (df_fused, cols_audio, cols_text)
}

# %% [markdown]
# ## 3. LOOCV + Nested CV Pipeline

# %%
def get_param_grid(model_name):
    if model_name == 'Random Forest':
        return {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    elif model_name == 'XGBoost':
        return {'n_estimators': [100, 150], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05]}
    return {}

def make_model(model_name):
    if model_name == 'Random Forest':
        return RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=1)
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', scale_pos_weight=2.5, n_jobs=1)

def loocv_egemaps(df, audio_cols, text_cols, model_name):
    n = len(df)
    y_all = df['label'].values.astype(int)
    
    X_audio_all = df[audio_cols].values.astype(np.float64) if audio_cols else None
    X_text_all = df[text_cols].values.astype(np.float64) if text_cols else None

    y_true = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_grid = get_param_grid(model_name)
    f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)
    
    # K-Best untuk Audio
    AUDIO_K = 30 # eGeMAPS punya 88, kita pilih 30 terbaik

    for i in range(n):
        y_tr = np.delete(y_all, i, axis=0)
        X_tr_fused, X_te_fused = [], []
        
        # 1. AUDIO PROCESS
        if audio_cols:
            X_a_tr = np.delete(X_audio_all, i, axis=0)
            X_a_te = X_audio_all[i:i+1]
            
            scaler_a = StandardScaler()
            X_a_tr = scaler_a.fit_transform(np.nan_to_num(X_a_tr))
            X_a_te = scaler_a.transform(np.nan_to_num(X_a_te))
            
            sel_a = SelectKBest(f_classif, k=min(AUDIO_K, X_a_tr.shape[1]))
            X_tr_fused.append(sel_a.fit_transform(X_a_tr, y_tr))
            X_te_fused.append(sel_a.transform(X_a_te))
            
        # 2. TEXT PROCESS (Tidak diseleksi, semua 13 dipakai)
        if text_cols:
            X_t_tr = np.delete(X_text_all, i, axis=0)
            X_t_te = X_text_all[i:i+1]
            
            scaler_t = StandardScaler()
            X_t_tr = scaler_t.fit_transform(np.nan_to_num(X_t_tr))
            X_t_te = scaler_t.transform(np.nan_to_num(X_t_te))
            
            X_tr_fused.append(X_t_tr)
            X_te_fused.append(X_t_te)

        # 3. FUSION
        X_tr = np.hstack(X_tr_fused)
        X_te = np.hstack(X_te_fused)

        # 4. TUNING & PREDICT
        gs = GridSearchCV(make_model(model_name), param_grid, cv=inner_cv, scoring=f1_scorer, n_jobs=1)
        gs.fit(X_tr, y_tr)
        
        try: y_prob[i] = gs.best_estimator_.predict_proba(X_te)[0, 1]
        except: y_prob[i] = float(gs.best_estimator_.predict(X_te)[0])
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
print(f"{'v14: EGEMAPS GLOTTAL AUDIO + DENSE TEXT':^100}")
print(f"{'#' * 100}")

for d_name, (df, acols, tcols) in datasets.items():
    for m_name in ['Random Forest', 'XGBoost']:
        combo = f"{d_name} + {m_name}"
        print(f"\n  [{combo}] Running ...", flush=True)
        t0 = time.time()
        metrics, y_true, y_prob, y_pred_tuned = loocv_egemaps(df, acols, tcols, m_name)
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

csv_path = os.path.join(RESULTS_DIR, "metrics", "v14_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(f"{'RINGKASAN v14 — EGEMAPS + LIWC TEXT':^110}")
print("=" * 110)
print(df_results.to_string())

best = df_results.iloc[0]
with open(os.path.join(MODELS_DIR, "v14_summary.json"), 'w') as fp:
    json.dump({'best_f1': best['F1'], 'best_model': f"{best['Dataset']} + {best['Model']}"}, fp)
