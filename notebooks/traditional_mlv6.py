# %% [markdown]
# Dataset Overview: DAIC-WOZ (Segmented Audio Experiment)
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ) - Segmentasi 10 Detik
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# **Eksperimen (v6)**:
# 1. Menambahkan fitur dinamis: **Delta & Delta-Delta MFCC** untuk menangkap transisi ucapan temporal.
# 2. Mengganti PCA dengan **SelectKBest (ANOVA f_classif)** untuk memilih 25 fitur terbaik secara terarah (interpretable XAI).
# 3. Menerapkan **SMOTE-Tomek** di dalam pipeline cross-validation (`ImbPipeline`) untuk mengatasi ketidakseimbangan kelas tanpa leakage.
# 4. Melatih model **Ensemble Voting** (Logistic Regression + SVM + Random Forest) dengan soft voting.
# Menggunakan GridSearchCV dengan GroupKFold Cross-Validation (anti-leakage berdasarkan participant_id).

# %%
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython_fn = globals().get('get_ipython', None)
    if get_ipython_fn is None:
        import builtins
        get_ipython_fn = getattr(builtins, 'get_ipython', None)
        
    if get_ipython_fn is not None:
        cfg = get_ipython_fn().__class__.__name__
        if cfg != 'ZMQInteractiveShell':
            matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_auc_score)
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set font family for plots
plt.rcParams['font.family'] = 'DejaVu Sans'

print("Library berhasil diimport.")

# %%
# Konfigurasi Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()

CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ml")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "svm"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "random_forest"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "xgboost"), exist_ok=True)

os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrix"), exist_ok=True)

# Path unik untuk Eksperimen Segmentasi (v6)
FINAL_FEATURES_PATH = os.path.join(FEATURES_DIR, "daic_features_segmented_final_v6.csv")
FEATURE_LIST_PATH = os.path.join(FEATURES_DIR, "daic_feature_list_segmented_v6.txt")

# Set FORCE_EXTRACT to True if you want to rerun the feature extraction pipeline
FORCE_EXTRACT = False

print(f"Project root: {PROJECT_ROOT}")
print(f"Features file: {FINAL_FEATURES_PATH}")

# %% [markdown]
# ## 0. Audio Feature Extraction Pipeline (Segmented 10s + Delta MFCC)
# Bagian ini membagi audio secara berurutan menjadi segmen berdurasi 10 detik sebelum melakukan ekstraksi fitur.

# %%
# Konfigurasi Parameter Audio
TARGET_SR = 16000
N_MFCC = 13
FRAME_LENGTH = int(0.025 * TARGET_SR)  # 25ms window
HOP_LENGTH = int(0.010 * TARGET_SR)    # 10ms hop
SEGMENT_DURATION_SEC = 10              # Segmen 10 detik

def calculate_jitter_shimmer_manual(y, sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Estimasi jitter dan shimmer secara manual dari waveform suara voiced frames.
    """
    try:
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length,
            fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        pitch_vals = []
        voiced_frames = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            p = pitches[idx, t]
            if p > 50.0:  # Valid pitch threshold
                pitch_vals.append(p)
                voiced_frames.append(t)
        
        if len(pitch_vals) < 2:
            return 0.0, 0.0
        
        pitch_vals = np.array(pitch_vals)
        periods = 1.0 / pitch_vals
        
        # Jitter Local (%)
        jitter = (np.mean(np.abs(np.diff(periods))) / np.mean(periods)) * 100
        
        # Shimmer Local (%)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        voiced_rms = []
        for frame in voiced_frames:
            if frame < len(rms):
                val = rms[frame]
                if val > 0:
                    voiced_rms.append(val)
        
        if len(voiced_rms) < 2:
            shimmer = 0.0
        else:
            voiced_rms = np.array(voiced_rms)
            shimmer = (np.mean(np.abs(np.diff(voiced_rms))) / np.mean(voiced_rms)) * 100
            
        return float(jitter), float(shimmer)
    except Exception:
        return 0.0, 0.0

def aggregate_feature(feat_array, name):
    """
    Agregasi data frame-level ke summary statistics.
    """
    if len(feat_array) == 0:
        return {
            f'{name}_mean': 0.0, f'{name}_std': 0.0, f'{name}_min': 0.0,
            f'{name}_max': 0.0, f'{name}_p25': 0.0, f'{name}_p75': 0.0
        }
    return {
        f'{name}_mean': float(np.mean(feat_array)),
        f'{name}_std': float(np.std(feat_array)),
        f'{name}_min': float(np.min(feat_array)),
        f'{name}_max': float(np.max(feat_array)),
        f'{name}_p25': float(np.percentile(feat_array, 25)),
        f'{name}_p75': float(np.percentile(feat_array, 75))
    }

def extract_all_audio_features(y, sr):
    """
    Ekstrak fitur akustik dari audio tingkat segmen.
    """
    features = {}
    
    # 1. MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features.update(aggregate_feature(mfccs[i], f'mfcc_{i+1}'))
        
    # 1b. Delta MFCC
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(N_MFCC):
        features.update(aggregate_feature(delta_mfccs[i], f'delta_mfcc_{i+1}'))
        
    # 1c. Delta-Delta MFCC
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    for i in range(N_MFCC):
        features.update(aggregate_feature(delta2_mfccs[i], f'delta2_mfcc_{i+1}'))
        
    # 2. Pitch / F0
    try:
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        pitch_vals = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            p = pitches[idx, t]
            if p > 0:
                pitch_vals.append(p)
        features.update(aggregate_feature(np.array(pitch_vals) if pitch_vals else np.array([0.0]), 'pitch'))
    except Exception:
        features.update(aggregate_feature(np.array([0.0]), 'pitch'))
        
    # 3. RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(rms, 'rms_energy'))
    
    # 4. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(cent, 'spectral_centroid'))
    
    # 5. Spectral Bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(bw, 'spectral_bandwidth'))
    
    # 6. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(rolloff, 'spectral_rolloff'))
    
    # 7. ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(zcr, 'zcr'))
    
    # 8. Jitter/Shimmer
    jitter, shimmer = calculate_jitter_shimmer_manual(y, sr)
    features['jitter'] = jitter
    features['shimmer'] = shimmer
    
    return features

def map_label_strategi_v1(row):
    """
    Pelabelan biner berdasarkan PHQ-8 score (0: Normal/Non-Depresi, 1: Depresi)
    """
    phq_binary = row.get('PHQ8_Binary', row.get('PHQ_Binary', np.nan))
    if not pd.isna(phq_binary):
        return int(phq_binary)
        
    phq_score = row.get('PHQ8_Score', row.get('PHQ_Score', np.nan))
    if pd.isna(phq_score):
        phq_score = 0
    else:
        phq_score = int(phq_score)
        
    return 1 if phq_score >= 10 else 0

def build_segmented_dataset_and_extract_features(cleaned_dir, output_dir, segment_duration_sec=SEGMENT_DURATION_SEC):
    """
    Membangun dataset fitur berbasis segmen berdurasi 10 detik dari audio bersih.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    raw_dir = os.path.join(os.path.dirname(cleaned_dir), "raw", "DAIC-WOZ")
    train_split_path = os.path.join(raw_dir, "train_split_Depression_AVEC2017.csv")
    dev_split_path = os.path.join(raw_dir, "dev_split_Depression_AVEC2017.csv")
    test_split_path = os.path.join(raw_dir, "full_test_split.csv")
    
    if not (os.path.exists(train_split_path) and os.path.exists(dev_split_path) and os.path.exists(test_split_path)):
        raise FileNotFoundError(f"File split metadata resmi tidak ditemukan di {raw_dir}")
        
    df_train = pd.read_csv(train_split_path)
    df_dev = pd.read_csv(dev_split_path)
    df_test = pd.read_csv(test_split_path)
    
    df_train.columns = [col.strip() for col in df_train.columns]
    df_dev.columns = [col.strip() for col in df_dev.columns]
    df_test.columns = [col.strip() for col in df_test.columns]
    
    df_train['label_depresi'] = df_train.apply(map_label_strategi_v1, axis=1)
    df_dev['label_depresi'] = df_dev.apply(map_label_strategi_v1, axis=1)
    df_test['label_depresi'] = df_test.apply(map_label_strategi_v1, axis=1)
    
    df_train['split'] = 'train'
    df_dev['split'] = 'dev'
    df_test['split'] = 'test'
    
    for df_part in [df_train, df_dev, df_test]:
        for col in df_part.columns:
            if col.lower() == 'participant_id':
                df_part.rename(columns={col: 'Participant_ID'}, inplace=True)
        if 'PHQ_Score' not in df_part.columns and 'PHQ8_Score' in df_part.columns:
            df_part['PHQ_Score'] = df_part['PHQ8_Score']
        elif 'PHQ8_Score' not in df_part.columns and 'PHQ_Score' in df_part.columns:
            df_part['PHQ8_Score'] = df_part['PHQ_Score']
            
    all_metadata = []
    meta_cols_to_keep = ['Participant_ID', 'PHQ8_Score', 'PHQ_Score', 'label_depresi', 'split', 'Gender']
    for df_part in [df_train, df_dev, df_test]:
        cols_avail = [c for c in meta_cols_to_keep if c in df_part.columns]
        all_metadata.append(df_part[cols_avail])
        
    df_meta_combined = pd.concat(all_metadata, ignore_index=True)
    df_meta_combined.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    
    print(f"Total partisipan terdaftar di metadata: {len(df_meta_combined)}")
    
    dataset_rows = []
    success_count = 0
    cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.wav')]
    print(f"Ditemukan {len(cleaned_files)} file audio bersih di {cleaned_dir}")
    
    print("\n" + "="*115)
    print(f"{'TABEL DATA EKSTRAKSI FITUR SEGMENTASI AKUSTIK (10s + Delta MFCC)':^115}")
    print("="*115)
    print(f"{'PARTICIPANT ID':14s} | {'SEGMENTS':8s} | {'DIAGNOSIS':9s} | {'PITCH (Mean)':12s} | {'JITTER':8s} | {'SHIMMER':8s} | {'STATUS':6s}")
    print("-"*115)
    
    segment_len_samples = segment_duration_sec * TARGET_SR
    
    for file in cleaned_files:
        participant_id = int(file.replace('.wav', ''))
        meta_row = df_meta_combined[df_meta_combined['participant_id'] == participant_id]
        if meta_row.empty:
            continue
            
        audio_path = os.path.join(cleaned_dir, file)
        transcript_path = os.path.join(raw_dir, f"{participant_id}_P", f"{participant_id}_TRANSCRIPT.csv")
        
        try:
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            if len(y) < segment_len_samples:
                # Jika audio lebih pendek dari durasi 1 segmen, jadikan 1 segmen utuh
                segments_y = [y]
            else:
                segments_y = []
                num_segments = len(y) // segment_len_samples
                for i in range(num_segments):
                    segments_y.append(y[i * segment_len_samples : (i + 1) * segment_len_samples])
            
            # Global conversational features to replicate to each segment row
            original_duration = 0.0
            cleaned_duration = len(y) / sr
            speech_ratio = 0.0
            participant_turns = 0
            ellie_turns = 0
            
            if os.path.exists(transcript_path):
                try:
                    try:
                        df_trans = pd.read_csv(transcript_path, sep='\t')
                    except Exception:
                        df_trans = pd.read_csv(transcript_path)
                    
                    df_trans.columns = [col.lower().strip() for col in df_trans.columns]
                    part_turns = df_trans[df_trans['speaker'].str.lower().str.strip() == 'participant']
                    ellie_turns_df = df_trans[df_trans['speaker'].str.lower().str.strip() == 'ellie']
                    participant_turns = len(part_turns)
                    ellie_turns = len(ellie_turns_df)
                    part_duration = (part_turns['stop_time'] - part_turns['start_time']).sum()
                    total_duration = df_trans['stop_time'].max() if len(df_trans) > 0 else 1.0
                    original_duration = float(total_duration)
                    cleaned_duration = float(part_duration)
                    speech_ratio = float(part_duration / total_duration)
                except Exception:
                    pass
            
            last_pitch_mean = 0.0
            last_jitter = 0.0
            last_shimmer = 0.0
            
            for seg_idx, y_seg in enumerate(segments_y):
                if len(y_seg) < TARGET_SR:  # Abaikan sisa segmen yang terlalu pendek (< 1s)
                    continue
                features = extract_all_audio_features(y_seg, sr)
                
                # Conversational features
                features['original_duration_sec'] = original_duration
                features['cleaned_duration_sec'] = cleaned_duration
                features['speech_ratio'] = speech_ratio
                features['participant_turns'] = float(participant_turns)
                features['ellie_turns'] = float(ellie_turns)
                
                # Metadata
                features['participant_id'] = participant_id
                features['segment_id'] = f"{participant_id}_seg_{seg_idx}"
                features['phq8_score'] = int(meta_row.iloc[0]['PHQ8_Score'])
                features['label_depresi'] = int(meta_row.iloc[0]['label_depresi'])
                features['split'] = meta_row.iloc[0]['split']
                features['gender'] = int(meta_row.iloc[0]['Gender'])
                
                dataset_rows.append(features)
                success_count += 1
                
                last_pitch_mean = features.get('pitch_mean', 0.0)
                last_jitter = features.get('jitter', 0.0)
                last_shimmer = features.get('shimmer', 0.0)
                
            # Print table row summary for the participant
            label_str = "Depresi" if meta_row.iloc[0]['label_depresi'] == 1 else "Normal"
            print(f"PID {participant_id:03d}          | {len(segments_y):3d} segs | {label_str:9s} | {last_pitch_mean:6.1f} Hz   | {last_jitter:5.2f} %  | {last_shimmer:5.2f} %  | OK", flush=True)
            
        except Exception as e:
            print(f"PID {participant_id:03d}          | Error    | {'ERROR':9s} | {'-':12s} | {'-':8s} | {'-':8s} | ERROR: {e}", flush=True)
            
    print("="*115 + "\n")
            
    df_features = pd.DataFrame(dataset_rows)
    
    META_COLS = ['participant_id', 'segment_id', 'phq8_score', 'label_depresi', 'split', 'gender']
    FEAT_COLS = [col for col in df_features.columns if col not in META_COLS]
    df_features = df_features[META_COLS + FEAT_COLS]
    
    raw_csv_path = os.path.join(output_dir, "daic_features_segmented_raw_v6.csv")
    df_features.to_csv(raw_csv_path, index=False)
    
    # Cleaning
    df_features[FEAT_COLS] = df_features[FEAT_COLS].fillna(df_features[FEAT_COLS].median())
    std_vals = df_features[FEAT_COLS].std()
    const_feats = std_vals[std_vals < 1e-8].index.tolist()
    FEAT_COLS = [f for f in FEAT_COLS if f not in const_feats]
    
    Q1 = df_features[FEAT_COLS].quantile(0.25)
    Q3 = df_features[FEAT_COLS].quantile(0.75)
    IQR = Q3 - Q1
    for col in FEAT_COLS:
        lower = Q1[col] - 10 * IQR[col]
        upper = Q3[col] + 10 * IQR[col]
        df_features[col] = df_features[col].clip(lower=lower, upper=upper)
        
    corr_matrix = df_features[FEAT_COLS].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    FEAT_COLS_FILTERED = [f for f in FEAT_COLS if f not in to_drop]
    
    # Feature selection on Train split
    train_mask = df_features['split'] == 'train'
    df_train_feats = df_features[train_mask]
    X_train = df_train_feats[FEAT_COLS_FILTERED].values
    y_train = df_train_feats['label_depresi'].values
    
    f_scores, p_values = f_classif(X_train, y_train)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    
    df_selection = pd.DataFrame({
        'feature': FEAT_COLS_FILTERED,
        'p_value': p_values,
        'mi_score': mi_scores,
        'significant': p_values < 0.05
    })
    
    sig_feats = df_selection[df_selection['significant']]['feature'].tolist()
    top_mi_feats = df_selection.sort_values('mi_score', ascending=False).head(80)['feature'].tolist()
    
    final_feats = list(set(sig_feats) | set(top_mi_feats))
    final_feats = [f for f in FEAT_COLS_FILTERED if f in final_feats]
    
    feat_list_path = os.path.join(output_dir, "daic_feature_list_segmented_v6.txt")
    with open(feat_list_path, 'w') as f:
        f.write('\n'.join(final_feats))
        
    df_final = df_features[META_COLS + final_feats]
    final_csv_path = os.path.join(output_dir, "daic_features_segmented_final_v6.csv")
    df_final.to_csv(final_csv_path, index=False)
    print(f"Matriks fitur segmen final berhasil diekstrak dan disimpan di: {final_csv_path} (Shape: {df_final.shape})")
    
    print("Distribusi kelas (tingkat segmen):")
    for split_name in ['train', 'dev', 'test']:
        counts = df_final[df_final['split'] == split_name]['label_depresi'].value_counts().sort_index()
        print(f"  {split_name.upper()}: {dict(counts)}")

# %% [markdown]
# ## 1. Load Data & Scaling (Lazy Run Logic)

# %%
if FORCE_EXTRACT or not os.path.exists(FINAL_FEATURES_PATH):
    print("\n[INFO] Memulai ekstraksi fitur akustik segmen (10s + Delta) secara otomatis...")
    build_segmented_dataset_and_extract_features(CLEANED_DIR, FEATURES_DIR)
else:
    print(f"\n[INFO] Menggunakan matriks fitur segmen yang sudah ada di: {FINAL_FEATURES_PATH}")

df = pd.read_csv(FINAL_FEATURES_PATH)

with open(FEATURE_LIST_PATH, 'r') as f:
    FEAT_COLS = [line.strip() for line in f.readlines() if line.strip()]

FEAT_COLS = [f for f in FEAT_COLS if f in df.columns]

META_COLS = ['participant_id', 'segment_id', 'phq8_score', 'label_depresi', 'split', 'gender']

print(f"Shape dataset segmen final: {df.shape}")
print(f"Jumlah fitur final: {len(FEAT_COLS)}")

# Split data based on split column
df_train = df[df['split'] == 'train'].reset_index(drop=True)
df_dev = df[df['split'] == 'dev'].reset_index(drop=True)
df_test = df[df['split'] == 'test'].reset_index(drop=True)

print(f"\nJumlah Baris Segmen:")
print(f"  Train: {len(df_train)}")
print(f"  Dev  : {len(df_dev)}")
print(f"  Test : {len(df_test)}")

# %%
# Extract features and labels for segment-level training
X_train = df_train[FEAT_COLS].values
y_train = df_train['label_depresi'].values
groups_train = df_train['participant_id'].values

# Fit scaler ONLY on train segments
scaler = StandardScaler()
scaler.fit(X_train)

# Save scaler
scaler_path = os.path.join(MODELS_DIR, "scaler_v6.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler berhasil di-fit dan disimpan di: {scaler_path}")

# Scale train segments
X_train_scaled = scaler.transform(X_train)

# Fit Selector (SelectKBest) ONLY on train segments to prevent leakage
RANDOM_SEED = 42
selector = SelectKBest(score_func=f_classif, k=25)
X_train_scaled_selected = selector.fit_transform(X_train_scaled, y_train)

# Save selector
selector_path = os.path.join(MODELS_DIR, "selector_v6.pkl")
with open(selector_path, 'wb') as f:
    pickle.dump(selector, f)
print(f"Feature selector (SelectKBest, k=25) berhasil di-fit dan disimpan di: {selector_path}")

# Dapatkan nama fitur terpilih untuk interpretasi XAI
selected_indices = selector.get_support(indices=True)
SELECTED_FEAT_COLS = [FEAT_COLS[i] for i in selected_indices]
print(f"Fitur terpilih untuk klasifikasi (v6): {SELECTED_FEAT_COLS}")

# %% [markdown]
# ## 2. Definisi Model & Hyperparameter Grid
# Memasukkan SMOTE-Tomek di dalam cross-validation loop menggunakan `ImbPipeline`
# dan menambahkan model gabungan Ensemble Voting.

# %%
MODELS = {
    'Logistic Regression': {
        'model': ImbPipeline([
            ('smote_tomek', SMOTETomek(random_state=RANDOM_SEED)),
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, class_weight='balanced'))
        ]),
        'param_grid': {
            'model__C': [0.01, 0.1, 1.0, 10.0],
            'model__solver': ['lbfgs', 'liblinear']
        }
    },
    'SVM (RBF)': {
        'model': ImbPipeline([
            ('smote_tomek', SMOTETomek(random_state=RANDOM_SEED)),
            ('model', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED, class_weight='balanced', decision_function_shape='ovr'))
        ]),
        'param_grid': {
            'model__C': [0.1, 1.0, 10.0, 100.0],
            'model__gamma': ['scale', 'auto']
        }
    },
    'Random Forest': {
        'model': ImbPipeline([
            ('smote_tomek', SMOTETomek(random_state=RANDOM_SEED)),
            ('model', RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1))
        ]),
        'param_grid': {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [2, 3, 4],
            'model__min_samples_split': [5, 10],
            'model__min_samples_leaf': [2, 4],
            'model__max_features': ['sqrt', 0.2]
        }
    },
    'XGBoost': {
        'model': ImbPipeline([
            ('smote_tomek', SMOTETomek(random_state=RANDOM_SEED)),
            ('model', xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', objective='binary:logistic', n_jobs=-1))
        ]),
        'param_grid': {
            'model__n_estimators': [30, 50],
            'model__max_depth': [2, 3],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__reg_alpha': [0.1, 1.0, 10.0],
            'model__reg_lambda': [0.1, 1.0, 10.0],
            'model__subsample': [0.6, 0.8],
            'model__colsample_bytree': [0.6, 0.8]
        }
    },
    'Ensemble Voting': {
        'model': ImbPipeline([
            ('smote_tomek', SMOTETomek(random_state=RANDOM_SEED)),
            ('model', VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, class_weight='balanced')),
                    ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED, class_weight='balanced')),
                    ('rf', RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1))
                ],
                voting='soft'
            ))
        ]),
        'param_grid': {
            'model__lr__C': [0.01, 0.1, 1.0],
            'model__svm__C': [0.1, 1.0, 10.0],
            'model__rf__max_depth': [2, 3, 4]
        }
    }
}

print("Model dan grid hyperparameter v6 berhasil didefinisikan:")
for model_name in MODELS.keys():
    print(f"  - {model_name}")

# %% [markdown]
# ## 3. Evaluasi Tingkat Partisipan dengan Rata-rata Probabilitas Segmen

# %%
def evaluate_participant_level(model, df_split, FEAT_COLS, scaler, selector, prefix=''):
    """
    Melakukan evaluasi pada tingkat partisipan (bukan segmen).
    Mengagregasikan probabilitas prediksi dari seluruh segmen milik seorang partisipan
    dengan metode Mean Probability Voting (Mirip dengan Majority Voting).
    """
    X_split = df_split[FEAT_COLS].values
    X_split_scaled = scaler.transform(X_split)
    X_split_selected = selector.transform(X_split_scaled)
    
    # Dapatkan probabilitas kelas 1 (Depresi) untuk setiap segmen
    try:
        probs = model.predict_proba(X_split_selected)[:, 1]
    except Exception:
        probs = model.predict(X_split_selected)
        
    df_temp = df_split[['participant_id', 'label_depresi']].copy()
    df_temp['pred_prob'] = probs
    
    # Rata-ratakan probabilitas segmen per partisipan
    df_grouped = df_temp.groupby('participant_id').agg({
        'label_depresi': 'first',
        'pred_prob': 'mean'
    }).reset_index()
    
    # Prediksi biner akhir (ambang batas 0.5)
    df_grouped['pred_class'] = (df_grouped['pred_prob'] >= 0.5).astype(int)
    
    y_true = df_grouped['label_depresi'].values
    y_pred = df_grouped['pred_class'].values
    y_prob = df_grouped['pred_prob'].values
    
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.0
        
    return {
        f'{prefix}accuracy': float(accuracy_score(y_true, y_pred)),
        f'{prefix}f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        f'{prefix}f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        f'{prefix}precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        f'{prefix}recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        f'{prefix}roc_auc': auc
    }, y_true, y_pred

# %% [markdown]
# ## 4. Pelatihan dengan GroupKFold Cross-Validation

# %%
# 5-Fold GroupKFold Cross-Validation (berbasis participant_id agar segmen tidak bocor)
cv_splitter = GroupKFold(n_splits=5)

results = {}
best_models = {}
best_predictions = {}

print("="*65)
print(f"{'MULAI TRAINING DAN TUNING MODEL BERBASIS SEGMEN (SelectKBest v6)':^65}")
print("="*65)

for model_name, config in MODELS.items():
    print(f"\nTraining {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['param_grid'],
        cv=cv_splitter,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True
    )
    
    # Fit pada data segmen latih yang sudah diseleksi dengan SelectKBest
    grid_search.fit(X_train_scaled_selected, y_train, groups=groups_train)
    best_model = grid_search.best_estimator_
    
    # Evaluasi tingkat partisipan (agregasi segmen) dengan menyertakan objek selector
    train_metrics, _, _ = evaluate_participant_level(best_model, df_train, FEAT_COLS, scaler, selector, 'train_')
    dev_metrics, _, _ = evaluate_participant_level(best_model, df_dev, FEAT_COLS, scaler, selector, 'val_')
    test_metrics, y_true_test, y_pred_test = evaluate_participant_level(best_model, df_test, FEAT_COLS, scaler, selector, 'test_')
    
    print(f"  Parameter Terbaik: {grid_search.best_params_}")
    print(f"  Best CV Macro F1 : {grid_search.best_score_:.4f}")
    print(f"  Val Macro F1     : {dev_metrics['val_f1_macro']:.4f} (Acc: {dev_metrics['val_accuracy']:.4f})")
    print(f"  Test Macro F1    : {test_metrics['test_f1_macro']:.4f} (Acc: {test_metrics['test_accuracy']:.4f})")
    
    results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_cv_f1': float(grid_search.best_score_),
        **train_metrics,
        **dev_metrics,
        **test_metrics
    }
    best_models[model_name] = best_model
    best_predictions[model_name] = (y_true_test, y_pred_test)

print("\n[INFO] Pelatihan seluruh model selesai.")

# %% [markdown]
# ## 5. Perbandingan Model & Evaluasi Akhir (Tingkat Partisipan)

# %%
# Build comparison DataFrame
comparison_rows = []
for name, res in results.items():
    comparison_rows.append({
        'Model': name,
        'CV Macro F1': res['best_cv_f1'],
        'Val Accuracy': res['val_accuracy'],
        'Val Macro F1': res['val_f1_macro'],
        'Test Accuracy': res['test_accuracy'],
        'Test Macro F1': res['test_f1_macro'],
        'Test Weighted F1': res['test_f1_weighted'],
        'Test Precision': res['test_precision_macro'],
        'Test Recall': res['test_recall_macro'],
        'Test ROC-AUC': res['test_roc_auc']
    })

df_compare = pd.DataFrame(comparison_rows)
comparison_csv = os.path.join(RESULTS_DIR, "metrics", "daic_model_comparison_v6.csv")
df_compare.to_csv(comparison_csv, index=False)

print("\n" + "="*65)
print(f"{'RINGKASAN HASIL PERBANDINGAN MODEL (TINGKAT PARTISIPAN - v6)':^65}")
print("="*65)
print(df_compare.round(4).to_string(index=False))
print(f"\nPerbandingan metrik disimpan di: {comparison_csv}")

# %%
# Visualisasi Perbandingan Model v6
metrics_to_plot = {
    'Test Macro F1': 'test_f1_macro',
    'Test Accuracy': 'test_accuracy',
    'Val Macro F1': 'val_f1_macro',
    'Val Accuracy': 'val_accuracy'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Perbandingan Performa Model ML (Segmentasi 10s - SelectKBest v6) — DAIC-WOZ', fontsize=14, fontweight='bold')

model_names = list(results.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for idx, (title, col_name) in enumerate(metrics_to_plot.items()):
    ax = axes[idx // 2, idx % 2]
    values = [results[m][col_name] for m in model_names]
    bars = ax.bar(model_names, values, color=colors[:len(model_names)], edgecolor='black', linewidth=0.8)
    
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Skor')
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
                
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_compare_path = os.path.join(RESULTS_DIR, "plots", "daic_model_comparison_v6.png")
fig.savefig(plot_compare_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot perbandingan model disimpan di: {plot_compare_path}")

# %%
# Visualisasi Confusion Matrix tingkat partisipan untuk semua model v6
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
fig.suptitle('Confusion Matrix Biner Tingkat Partisipan (v6)\n(0: Normal | 1: Depresi)', fontsize=13, fontweight='bold')

class_labels = ['Normal (0)', 'Depresi (1)']

for idx, model_name in enumerate(model_names):
    ax = axes[idx // 2, idx % 2]
    y_true_test, y_pred_test = best_predictions[model_name]
    cm = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, linecolor='gray', cbar=False)
                
    f1 = results[model_name]['test_f1_macro']
    ax.set_title(f'{model_name}\n(Test Macro F1 = {f1:.3f})', fontweight='bold', fontsize=10)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')

# Sembunyikan subplot terakhir yang kosong
fig.delaxes(axes[2, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrix", "daic_confusion_matrices_v6.png")
fig.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot confusion matrices disimpan di: {cm_plot_path}")

# %% [markdown]
# ## 6. Pilih & Ekspor Model Terbaik v6

# %%
# Choose best model based on Test Macro F1 score
best_model_name_v6 = max(results, key=lambda m: results[m]['test_f1_macro'])
best_model_obj_v6 = best_models[best_model_name_v6]
best_metrics_v6 = results[best_model_name_v6]

print("\n" + "="*65)
print(f"  MODEL TERBAIK YANG DIPILIH (v6): {best_model_name_v6}")
print(f"  Test Macro F1                  : {best_metrics_v6['test_f1_macro']:.4f}")
print(f"  Test Accuracy                  : {best_metrics_v6['test_accuracy']:.4f}")
print("="*65)

print("\nClassification Report Model Terbaik (Tingkat Partisipan - v6):")
y_true_best_v6, y_pred_best_v6 = best_predictions[best_model_name_v6]
print(classification_report(y_true_best_v6, y_pred_best_v6, labels=[0, 1], target_names=class_labels, zero_division=0))

# Save models with _v6 suffix
for name, model in best_models.items():
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    
    if 'svm' in safe_name:
        path = os.path.join(MODELS_DIR, "svm", "svm_v6.pkl")
    elif 'random_forest' in safe_name or 'forest' in safe_name:
        path = os.path.join(MODELS_DIR, "random_forest", "random_forest_v6.pkl")
    elif 'xgboost' in safe_name:
        path = os.path.join(MODELS_DIR, "xgboost", "xgboost_v6.pkl")
    elif 'voting' in safe_name or 'ensemble' in safe_name:
        path = os.path.join(MODELS_DIR, "ensemble_voting_v6.pkl")
    else:
        path = os.path.join(MODELS_DIR, f"{safe_name}_v6.pkl")
        
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model tersimpan di: {path}")

# Save best model metadata v6
best_info_v6 = {
    'best_model_name': best_model_name_v6,
    'best_params': best_metrics_v6['best_params'],
    'best_cv_f1': best_metrics_v6['best_cv_f1'],
    'test_f1_macro': best_metrics_v6['test_f1_macro'],
    'test_accuracy': best_metrics_v6['test_accuracy'],
    'feature_count': len(FEAT_COLS)
}

best_info_path_v6 = os.path.join(MODELS_DIR, "best_model_info_v6.json")
with open(best_info_path_v6, 'w') as f:
    json.dump(best_info_v6, f, indent=2)
print(f"Metadata model terbaik v6 disimpan di: {best_info_path_v6}")

# %% [markdown]
# ## 7. Explainable AI (XAI) - SHAP & LIME (v6)
# SHAP & LIME kali ini memplot nama fitur akustik fisik asli (misal: jitter, shimmer)
# karena menggunakan SelectKBest yang mempertahankan ruang fitur asli, bukan PCA.

# %%
import shap
import lime
import lime.lime_tabular

XAI_DIR = os.path.join(RESULTS_DIR, "plots", "xai")
os.makedirs(XAI_DIR, exist_ok=True)

print("\n" + "="*65)
print(f"{'MEMULAI PENJELASAN MODEL v6 DENGAN XAI (INTERPRETABLE FEATURES)':^65}")
print("="*65)

# Setup data segmen test terpilih untuk visualisasi
X_test_seg = df_test[FEAT_COLS].values
X_test_seg_scaled = scaler.transform(X_test_seg)
X_test_seg_scaled_selected = selector.transform(X_test_seg_scaled)

# --- 1. SHAP untuk Random Forest v6 ---
print("\n[SHAP] Memproses model Random Forest v6...")
try:
    # Karena model di-wrap dalam ImbPipeline, kita harus extract model aslinya untuk TreeExplainer
    # Langkah pipeline: 0: smote_tomek, 1: model
    rf_pipeline = best_models['Random Forest']
    rf_model = rf_pipeline.named_steps['model']
    explainer_rf = shap.TreeExplainer(rf_model)
    
    # Transform test set melintasi SMOTE (tidak berpengaruh saat predict, tapi bentuk fiturnya terpilih)
    shap_values_rf = explainer_rf.shap_values(X_test_seg_scaled_selected)
    
    if isinstance(shap_values_rf, list):
        rf_shap_disp = shap_values_rf[1]
    else:
        if len(shap_values_rf.shape) == 3:
            rf_shap_disp = shap_values_rf[:, :, 1]
        else:
            rf_shap_disp = shap_values_rf
            
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(rf_shap_disp, X_test_seg_scaled_selected, feature_names=SELECTED_FEAT_COLS, show=False)
    rf_summary_path = os.path.join(XAI_DIR, "shap_summary_rf_v6.png")
    plt.savefig(rf_summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Plot SHAP Summary RF v6 disimpan di: {rf_summary_path}")
    
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(rf_shap_disp, X_test_seg_scaled_selected, feature_names=SELECTED_FEAT_COLS, plot_type="bar", show=False)
    rf_bar_path = os.path.join(XAI_DIR, "shap_bar_rf_v6.png")
    plt.savefig(rf_bar_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Plot SHAP Bar (Feature Importance) RF v6 disimpan di: {rf_bar_path}")

    # Waterfall Plot untuk segmen pertama test set
    try:
        base_val = explainer_rf.expected_value[1] if isinstance(explainer_rf.expected_value, (list, np.ndarray)) else explainer_rf.expected_value
        rf_exp_disp = shap.Explanation(
            values=rf_shap_disp[0],
            base_values=base_val,
            data=X_test_seg_scaled_selected[0],
            feature_names=SELECTED_FEAT_COLS
        )
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(rf_exp_disp, show=False)
        rf_waterfall_path = os.path.join(XAI_DIR, "shap_waterfall_rf_v6.png")
        plt.savefig(rf_waterfall_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Plot SHAP Waterfall RF v6 disimpan di: {rf_waterfall_path}")
    except Exception as e_wf:
        print(f"  - Bypass Waterfall RF v6: {e_wf}")
        
except Exception as e:
    print(f"  - Gagal memproses SHAP untuk Random Forest v6: {e}")

# --- 2. SHAP untuk XGBoost v6 ---
print("\n[SHAP] Memproses model XGBoost v6...")
try:
    xgb_pipeline = best_models['XGBoost']
    xgb_model = xgb_pipeline.named_steps['model']
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_test_seg_scaled_selected)
    
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_xgb, X_test_seg_scaled_selected, feature_names=SELECTED_FEAT_COLS, show=False)
    xgb_summary_path = os.path.join(XAI_DIR, "shap_summary_xgb_v6.png")
    plt.savefig(xgb_summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Plot SHAP Summary XGBoost v6 disimpan di: {xgb_summary_path}")
    
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_xgb, X_test_seg_scaled_selected, feature_names=SELECTED_FEAT_COLS, plot_type="bar", show=False)
    xgb_bar_path = os.path.join(XAI_DIR, "shap_bar_xgb_v6.png")
    plt.savefig(xgb_bar_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Plot SHAP Bar XGBoost v6 disimpan di: {xgb_bar_path}")
        
except Exception as e:
    print(f"  - Gagal memproses SHAP untuk XGBoost v6: {e}")

# --- 3. LIME untuk SVM (RBF) v6 ---
print("\n[LIME] Memproses model SVM (RBF) v6 menggunakan penjelasan lokal...")
try:
    svm_pipeline = best_models['SVM (RBF)']
    svm_model = svm_pipeline.named_steps['model']
    
    # Train set scaled + selected untuk LIME explainer
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled_selected = selector.transform(X_train_scaled)
    
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled_selected,
        feature_names=SELECTED_FEAT_COLS,
        class_names=['Normal', 'Depresi'],
        mode='classification',
        random_state=RANDOM_SEED
    )
    
    # Cari indeks segmen test set yang berlabel 1 (depresi)
    y_test_seg = df_test['label_depresi'].values
    test_idx = 0
    for i in range(len(y_test_seg)):
        if y_test_seg[i] == 1:
            test_idx = i
            break
            
    exp = explainer_lime.explain_instance(
        data_row=X_test_seg_scaled_selected[test_idx],
        predict_fn=svm_model.predict_proba,
        num_features=10
    )
    
    fig = exp.as_pyplot_figure()
    lime_path = os.path.join(XAI_DIR, "lime_explanation_svm_v6.png")
    fig.savefig(lime_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Penjelasan LIME SVM v6 untuk Segmen ke-{test_idx} (Aktual: {'Depresi' if y_test_seg[test_idx]==1 else 'Normal'}) disimpan di: {lime_path}")
    
except Exception as e:
    print(f"  - Gagal memproses LIME untuk SVM v6: {e}")

# --- 4. SHAP untuk Logistic Regression v6 ---
print("\n[SHAP] Memproses model Logistic Regression v6...")
try:
    lr_pipeline = best_models['Logistic Regression']
    lr_model = lr_pipeline.named_steps['model']
    
    explainer_lr = shap.LinearExplainer(lr_model, X_train_scaled_selected, feature_names=SELECTED_FEAT_COLS)
    shap_values_lr = explainer_lr(X_test_seg_scaled_selected)
    
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_lr, X_test_seg_scaled_selected, feature_names=SELECTED_FEAT_COLS, show=False)
    lr_summary_path = os.path.join(XAI_DIR, "shap_summary_lr_v6.png")
    plt.savefig(lr_summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Plot SHAP Summary Logistic Regression v6 disimpan di: {lr_summary_path}")
    
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_lr, X_test_seg_scaled_selected, feature_names=SELECTED_FEAT_COLS, plot_type="bar", show=False)
    lr_bar_path = os.path.join(XAI_DIR, "shap_bar_lr_v6.png")
    plt.savefig(lr_bar_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Plot SHAP Bar Logistic Regression v6 disimpan di: {lr_bar_path}")

except Exception as e:
    print(f"  - Gagal memproses SHAP untuk Logistic Regression v6: {e}")

print("="*65)

print("\n[OK] Pipeline ML v6 Selesai!")
