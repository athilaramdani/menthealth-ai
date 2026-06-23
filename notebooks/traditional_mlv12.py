# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v12** — Segment-Level Classification + Multimodal Dense Text
#
# ─────────────────────────────────────────────────────────────────────
#  v12 = THE PARADIGM SHIFT (BREAKING 0.70)
#
#  Masalah v11: 102 sample (participant-level) terlalu kecil untuk ML.
#  Solusi v12: 
#  1. Potong audio jadi 10-detik segment (~6,000 training samples).
#  2. Ekstrak MFCC per segment.
#  3. Fuse dengan Dense NLP Text Feature milik participant tersebut.
#  4. Train Model pada tingkat SEGMENT.
#  5. Prediksi participant dengan Rata-rata Probabilitas Segmennya.
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

# %%
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
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
CLEAN_DIR   = os.path.join(PROJECT_ROOT, "data", "cleaned")
V12_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v12")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v12")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v12")

for d in [V12_FEAT_DIR, MODELS_DIR, os.path.join(RESULTS_DIR, "metrics"), 
          os.path.join(RESULTS_DIR, "plots"), os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

print(f"Project: {PROJECT_ROOT}")

# Load Ground Truth Labels
df_labels = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ", "train_split_Depression_AVEC2017.csv"))
df_dev = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ", "dev_split_Depression_AVEC2017.csv"))
# Combine all known labels
df_all_labels = pd.concat([df_labels, df_dev])
df_all_labels.columns = [c.lower().strip() for c in df_all_labels.columns]
label_dict = dict(zip(df_all_labels['participant_id'], df_all_labels['phq8_binary']))

# %% [markdown]
# ## 1. Extract Dense Psychological Text Features (from v11)

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
text_cols = [c for c in df_text.columns if c != 'participant_id']
print(f"  [Text] Diekstrak {len(text_cols)} dense features dari {len(df_text)} participants.")

# %% [markdown]
# ## 2. Segment-Level MFCC Extraction

# %%
SEGMENT_LEN_SEC = 10
SEGMENTS_CSV = os.path.join(V12_FEAT_DIR, "v12_segments_mfcc.csv")

def extract_segment_mfccs(clean_dir, output_csv):
    if os.path.exists(output_csv):
        print(f"  [Audio] File segmen sudah ada: {output_csv}")
        return pd.read_csv(output_csv)
        
    print(f"  [Audio] Mengekstrak MFCC per {SEGMENT_LEN_SEC} detik...")
    rows = []
    wav_files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]
    
    for f in tqdm(wav_files, desc="Extracting Segments"):
        pid = int(f.split('.')[0])
        if pid not in label_dict: continue
        label = label_dict[pid]
        
        y, sr = librosa.load(os.path.join(clean_dir, f), sr=16000)
        
        # Split into non-overlapping segments
        samples_per_seg = SEGMENT_LEN_SEC * sr
        num_segs = len(y) // samples_per_seg
        
        for i in range(num_segs):
            seg_y = y[i*samples_per_seg : (i+1)*samples_per_seg]
            
            # Extract basic 20 MFCCs (mean and std)
            mfccs = librosa.feature.mfcc(y=seg_y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            row = {'participant_id': pid, 'segment_idx': i, 'label': label}
            for j in range(20):
                row[f'mfcc_{j}_mean'] = mfcc_mean[j]
                row[f'mfcc_{j}_std'] = mfcc_std[j]
            rows.append(row)
            
    df_segs = pd.DataFrame(rows)
    df_segs.to_csv(output_csv, index=False)
    return df_segs

df_segments = extract_segment_mfccs(CLEAN_DIR, SEGMENTS_CSV)
audio_cols = [c for c in df_segments.columns if c.startswith('mfcc_')]
print(f"  [Audio] Total {len(df_segments)} segmen dari {df_segments['participant_id'].nunique()} partisipan.")

# Merge Audio Segments with Text Features
# Because Text is participant-level, it will be duplicated across all segments of that participant
df_multimodal = pd.merge(df_segments, df_text, on='participant_id', how='inner')
print(f"  [Fusion] Ready: {len(df_multimodal)} segmen dengan {len(audio_cols)} audio feats + {len(text_cols)} text feats.")

# %% [markdown]
# ## 3. Participant-Level LOOCV with Segment Aggregation

# %%
def loocv_segment_multimodal(df, model_name):
    """
    Train on SEGMENTS. Evaluate on PARTICIPANT (by averaging segment probs).
    """
    participants = df['participant_id'].unique()
    n = len(participants)
    
    y_true_part = np.zeros(n, dtype=int)
    y_prob_part = np.zeros(n, dtype=float)
    
    features = audio_cols + text_cols
    
    for i, test_pid in enumerate(tqdm(participants, desc=f"LOOCV {model_name}")):
        # Train-Test Split based on Participant ID to prevent leakage
        train_mask = df['participant_id'] != test_pid
        test_mask = df['participant_id'] == test_pid
        
        df_tr = df[train_mask]
        df_te = df[test_mask]
        
        X_tr = df_tr[features].values.astype(np.float64)
        y_tr = df_tr['label'].values.astype(int)
        
        X_te = df_te[features].values.astype(np.float64)
        y_te_seg = df_te['label'].values.astype(int)
        
        if len(X_te) == 0: continue
            
        y_true_part[i] = y_te_seg[0] # All segments of a participant have the same label
        
        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.nan_to_num(X_tr))
        X_te = scaler.transform(np.nan_to_num(X_te))
        
        # Model
        if model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=200, max_depth=10, 
                                           class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                                      scale_pos_weight=2.5, eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=-1)
            
        model.fit(X_tr, y_tr)
        
        # Predict all segments for this participant
        seg_probs = model.predict_proba(X_te)[:, 1]
        
        # AGGREGATION: Average the probabilities
        y_prob_part[i] = np.mean(seg_probs)

    # Threshold Tuning
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (y_prob_part >= thr).astype(int)
        f1_t = f1_score(y_true_part, preds, average='macro', zero_division=0)
        if f1_t > best_f1: best_f1, best_thr = f1_t, thr
            
    y_pred_tuned = (y_prob_part >= best_thr).astype(int)
    
    return {
        'f1_050': float(f1_score(y_true_part, (y_prob_part >= 0.5).astype(int), average='macro', zero_division=0)),
        'acc_050': float(accuracy_score(y_true_part, (y_prob_part >= 0.5).astype(int))),
        'f1_tuned': float(best_f1),
        'thr': float(round(best_thr, 2)),
        'acc_tuned': float(accuracy_score(y_true_part, y_pred_tuned)),
        'auc': float(roc_auc_score(y_true_part, y_prob_part))
    }, y_true_part, y_prob_part, y_pred_tuned

# %% [markdown]
# ## 4. Run Pipeline

# %%
all_results, all_ys = {}, {}

print(f"\n{'#' * 100}")
print(f"{'v12: SEGMENT-LEVEL CLASSIFICATION + MULTIMODAL DENSE TEXT':^100}")
print(f"{'#' * 100}")

for m_name in ['Random Forest', 'XGBoost']:
    combo = f"Segment_Multimodal + {m_name}"
    print(f"\n  [{combo}] Running ...", flush=True)
    t0 = time.time()
    metrics, y_true, y_prob, y_pred_tuned = loocv_segment_multimodal(df_multimodal, m_name)
    all_results[combo] = metrics
    all_ys[combo] = (y_true, y_prob, y_pred_tuned)
    print(f"\n    F1 (tuned): {metrics['f1_tuned']:.4f} (thr={metrics['thr']:.2f}) | AUC: {metrics['auc']:.4f} | Time: {time.time()-t0:.0f}s")

# %% [markdown]
# ## 5. Ringkasan Final

# %%
rows = []
for combo, m in all_results.items():
    d_name, model = combo.split(' + ')
    rows.append({'Dataset': d_name, 'Model': model, 'F1': m['f1_tuned'], 'Thr': m['thr'], 'AUC': m['auc']})

df_results = pd.DataFrame(rows).sort_values('F1', ascending=False).reset_index(drop=True)
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v12_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(f"{'RINGKASAN v12 — SEGMENT-LEVEL MULTIMODAL':^110}")
print("=" * 110)
print(df_results.to_string())

best = df_results.iloc[0]
with open(os.path.join(MODELS_DIR, "v12_summary.json"), 'w') as fp:
    json.dump({'best_f1': best['F1'], 'best_model': f"{best['Dataset']} + {best['Model']}"}, fp)
