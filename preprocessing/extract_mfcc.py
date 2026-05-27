import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, lfilter
from sklearn.feature_selection import f_classif, mutual_info_classif

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DAIC-WOZ-FeatureExtractor")

TARGET_SR = 16000
N_MFCC = 13
FRAME_LENGTH = int(0.025 * TARGET_SR)  # 25ms window
HOP_LENGTH = int(0.010 * TARGET_SR)    # 10ms hop

def calculate_jitter_shimmer_manual(y, sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Manually estimates jitter and shimmer from the waveform using fundamental frequency (F0) tracking.
    """
    try:
        # Extract fundamental frequency (F0) using piptrack
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
        # Pitch period T = 1 / F0
        periods = 1.0 / pitch_vals
        
        # Jitter Local (%) = (mean(|T_i - T_i+1|) / mean(T)) * 100
        jitter = (np.mean(np.abs(np.diff(periods))) / np.mean(periods)) * 100
        
        # Calculate RMS energy for shimmer
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Extract amplitude peaks corresponding to voiced frames
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
            # Shimmer Local (%) = (mean(|A_i - A_i+1|) / mean(A)) * 100
            shimmer = (np.mean(np.abs(np.diff(voiced_rms))) / np.mean(voiced_rms)) * 100
            
        return float(jitter), float(shimmer)
    except Exception as e:
        logger.warning(f"Gagal menghitung Jitter/Shimmer: {e}")
        return 0.0, 0.0

def aggregate_feature(feat_array, name):
    """
    Aggregates 1D frame-level feature array into summary statistics.
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
    Extracts complete set of 116 features from a cleaned audio waveform.
    """
    features = {}
    
    # 1. MFCC (13 Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features.update(aggregate_feature(mfccs[i], f'mfcc_{i+1}'))
        
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
    
    # 7. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    features.update(aggregate_feature(zcr, 'zcr'))
    
    # 8. Voice Quality: Jitter and Shimmer
    jitter, shimmer = calculate_jitter_shimmer_manual(y, sr)
    features['jitter'] = jitter
    features['shimmer'] = shimmer
    
    return features

def map_label_strategi_v1(row):
    """
    Applies hierarchical rule-based labeling from docs/strategiv1.md and maps to 4 classes:
    Class 0: NORMAL
    Class 1: STRES
    Class 2: CEMAS
    Class 3: DEPRESI
    """
    # Extract scores
    phq_score = row.get('PHQ8_Score', row.get('PHQ_Score', np.nan))
    if pd.isna(phq_score):
        phq_score = 0
    else:
        phq_score = int(phq_score)
        
    # Check if individual questionnaire item columns are present (Train & Dev splits)
    # The test split only has PHQ_Score
    has_items = 'PHQ8_Depressed' in row and not pd.isna(row['PHQ8_Depressed'])
    
    if has_items:
        def get_val(key):
            val = row.get(key, 0)
            return 0 if pd.isna(val) else int(val)
            
        dep = get_val('PHQ8_Depressed')
        noi = get_val('PHQ8_NoInterest')
        fail = get_val('PHQ8_Failure')
        mov = get_val('PHQ8_Moving')
        conc = get_val('PHQ8_Concentrating')
        sleep = get_val('PHQ8_Sleep')
        tired = get_val('PHQ8_Tired')
        
        # 1. Prioritas Pertama: DEPRESI
        if phq_score >= 10 or (dep >= 2 and noi >= 2 and fail >= 2):
            return 3  # DEPRESI
            
        # 2. Prioritas Kedua: CEMAS
        if phq_score >= 5 and (mov >= 1 or conc >= 2):
            return 2  # CEMAS
            
        # 3. Prioritas Ketiga: STRES
        if phq_score >= 5 and (sleep >= 2 or tired >= 2):
            return 1  # STRES
            
        # 4. Prioritas Keempat: NORMAL
        if phq_score <= 4 and dep <= 1 and noi <= 1:
            return 0  # NORMAL
            
        # 5. Fallback
        return 1  # STRES
    else:
        # Fallback logic for test set using only the score thresholds
        if phq_score >= 10:
            return 3
        elif phq_score <= 4:
            return 0
        else:
            return 1  # Fallback to STRES for score 5-9 when items are missing

def build_dataset_and_extract_features(cleaned_dir, output_dir):
    """
    Loads official splits, aligns participant IDs, labels them, extracts acoustic features,
    applies feature cleaning & selection, and outputs feature matrices.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load official CSV splits from raw directory
    raw_dir = "data/raw/DAIC-WOZ"
    train_split_path = os.path.join(raw_dir, "train_split_Depression_AVEC2017.csv")
    dev_split_path = os.path.join(raw_dir, "dev_split_Depression_AVEC2017.csv")
    test_split_path = os.path.join(raw_dir, "full_test_split.csv")
    
    if not (os.path.exists(train_split_path) and os.path.exists(dev_split_path) and os.path.exists(test_split_path)):
        raise FileNotFoundError("File split metadata resmi tidak ditemukan di data/raw/DAIC-WOZ/")
        
    df_train = pd.read_csv(train_split_path)
    df_dev = pd.read_csv(dev_split_path)
    df_test = pd.read_csv(test_split_path)
    
    # Normalize column names
    df_train.columns = [col.strip() for col in df_train.columns]
    df_dev.columns = [col.strip() for col in df_dev.columns]
    df_test.columns = [col.strip() for col in df_test.columns]
    
    # Label splits using map_label_strategi_v1
    logger.info("Melabeli data train, dev, dan test sesuai docs/strategiv1.md...")
    df_train['label_3kelas'] = df_train.apply(map_label_strategi_v1, axis=1)
    df_dev['label_3kelas'] = df_dev.apply(map_label_strategi_v1, axis=1)
    df_test['label_3kelas'] = df_test.apply(map_label_strategi_v1, axis=1)
    
    # Add split column
    df_train['split'] = 'train'
    df_dev['split'] = 'dev'
    df_test['split'] = 'test'
    
    # Rename test ID column to match Train/Dev
    if 'Participant_ID' in df_test.columns:
        df_test.rename(columns={'Participant_ID': 'Participant_ID'}, inplace=True)
    elif 'participant_ID' in df_test.columns:
        df_test.rename(columns={'participant_ID': 'Participant_ID'}, inplace=True)
        
    # Combine metadata
    meta_cols_to_keep = ['Participant_ID', 'PHQ8_Score', 'PHQ_Score', 'label_3kelas', 'split', 'Gender']
    all_metadata = []
    
    for df_part in [df_train, df_dev, df_test]:
        temp_df = df_part.copy()
        # Make sure Participant_ID column is named correctly
        for col in temp_df.columns:
            if col.lower() == 'participant_id':
                temp_df.rename(columns={col: 'Participant_ID'}, inplace=True)
        # Handle PHQ_Score or PHQ8_Score
        if 'PHQ_Score' not in temp_df.columns and 'PHQ8_Score' in temp_df.columns:
            temp_df['PHQ_Score'] = temp_df['PHQ8_Score']
        elif 'PHQ8_Score' not in temp_df.columns and 'PHQ_Score' in temp_df.columns:
            temp_df['PHQ8_Score'] = temp_df['PHQ_Score']
            
        cols_avail = [c for c in meta_cols_to_keep if c in temp_df.columns]
        all_metadata.append(temp_df[cols_avail])
        
    df_meta_combined = pd.concat(all_metadata, ignore_index=True)
    df_meta_combined.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    
    logger.info(f"Total partisipan terdaftar di metadata: {len(df_meta_combined)}")
    
    dataset_rows = []
    success_count = 0
    
    # Process cleaned files
    cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.wav')]
    logger.info(f"Ditemukan {len(cleaned_files)} file audio bersih di {cleaned_dir}")
    
    for file in cleaned_files:
        participant_id = int(file.replace('.wav', ''))
        
        # Check if participant is in metadata splits
        meta_row = df_meta_combined[df_meta_combined['participant_id'] == participant_id]
        if meta_row.empty:
            continue
            
        audio_path = os.path.join(cleaned_dir, file)
        
        try:
            # Load cleaned audio (already resampled to 16kHz, mono)
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            if len(y) < TARGET_SR:  # minimum 1 second of audio
                logger.warning(f"Audio PID {participant_id} terlalu pendek, skip.")
                continue
                
            features = extract_all_audio_features(y, sr)
            features['participant_id'] = participant_id
            features['phq8_score'] = int(meta_row.iloc[0]['PHQ8_Score'])
            features['label_3kelas'] = int(meta_row.iloc[0]['label_3kelas'])
            features['split'] = meta_row.iloc[0]['split']
            features['gender'] = int(meta_row.iloc[0]['Gender'])
            
            dataset_rows.append(features)
            success_count += 1
            if success_count % 10 == 0:
                logger.info(f"  Berhasil mengekstrak {success_count} partisipan...")
        except Exception as e:
            logger.error(f"Gagal memproses PID {participant_id}: {e}")
            
    df_features = pd.DataFrame(dataset_rows)
    logger.info(f"Berhasil mengekstrak total {len(df_features)} baris fitur.")
    
    # Save Raw Feature Matrix
    META_COLS = ['participant_id', 'phq8_score', 'label_3kelas', 'split', 'gender']
    FEAT_COLS = [col for col in df_features.columns if col not in META_COLS]
    
    # Rearrange columns
    df_features = df_features[META_COLS + FEAT_COLS]
    raw_csv_path = os.path.join(output_dir, "daic_features_raw.csv")
    df_features.to_csv(raw_csv_path, index=False)
    logger.info(f"Matriks fitur mentah disimpan di: {raw_csv_path}")
    
    # Apply Feature Cleaning & Normalization on Feature Selection
    logger.info("Melakukan pembersihan fitur...")
    
    # 1. Fill NaNs with median
    nan_counts = df_features[FEAT_COLS].isnull().sum()
    logger.info(f"Fitur mengandung NaN: {len(nan_counts[nan_counts > 0])}")
    df_features[FEAT_COLS] = df_features[FEAT_COLS].fillna(df_features[FEAT_COLS].median())
    
    # 2. Drop constant features
    std_vals = df_features[FEAT_COLS].std()
    const_feats = std_vals[std_vals < 1e-8].index.tolist()
    logger.info(f"Fitur konstan dihapus: {len(const_feats)}")
    FEAT_COLS = [f for f in FEAT_COLS if f not in const_feats]
    
    # 3. Clip outliers to 10 * IQR
    Q1 = df_features[FEAT_COLS].quantile(0.25)
    Q3 = df_features[FEAT_COLS].quantile(0.75)
    IQR = Q3 - Q1
    for col in FEAT_COLS:
        lower = Q1[col] - 10 * IQR[col]
        upper = Q3[col] + 10 * IQR[col]
        df_features[col] = df_features[col].clip(lower=lower, upper=upper)
        
    # 4. Remove redundant features (correlation > 0.95)
    logger.info("Menyaring fitur redundan (korelasi > 0.95)...")
    corr_matrix = df_features[FEAT_COLS].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    logger.info(f"Dihapus karena multikolinieritas tinggi: {len(to_drop)} fitur")
    FEAT_COLS_FILTERED = [f for f in FEAT_COLS if f not in to_drop]
    
    # 5. Feature Selection using ANOVA & Mutual Information on Train Split
    logger.info("Melakukan seleksi fitur menggunakan ANOVA F-test dan Mutual Information...")
    train_mask = df_features['split'] == 'train'
    df_train_feats = df_features[train_mask]
    
    X_train = df_train_feats[FEAT_COLS_FILTERED].values
    y_train = df_train_feats['label_3kelas'].values
    
    f_scores, p_values = f_classif(X_train, y_train)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    
    df_selection = pd.DataFrame({
        'feature': FEAT_COLS_FILTERED,
        'p_value': p_values,
        'mi_score': mi_scores,
        'significant': p_values < 0.05
    })
    
    # Union of statistically significant features (p < 0.05) and top 50 MI features
    sig_feats = df_selection[df_selection['significant']]['feature'].tolist()
    top_mi_feats = df_selection.sort_values('mi_score', ascending=False).head(50)['feature'].tolist()
    
    final_feats = list(set(sig_feats) | set(top_mi_feats))
    # Preserve order
    final_feats = [f for f in FEAT_COLS_FILTERED if f in final_feats]
    
    logger.info(f"Hasil Seleksi Fitur:")
    logger.info(f"  Signifikan (p < 0.05): {len(sig_feats)}")
    logger.info(f"  Top 50 MI            : {len(top_mi_feats)}")
    logger.info(f"  Kombinasi (Final)    : {len(final_feats)}")
    
    # Save Final Feature List
    feat_list_path = os.path.join(output_dir, "daic_feature_list.txt")
    with open(feat_list_path, 'w') as f:
        f.write('\n'.join(final_feats))
    logger.info(f"Daftar fitur final disimpan di: {feat_list_path}")
    
    # Save Final Feature Matrix
    df_final = df_features[META_COLS + final_feats]
    final_csv_path = os.path.join(output_dir, "daic_features_final.csv")
    df_final.to_csv(final_csv_path, index=False)
    logger.info(f"Matriks fitur final disimpan di: {final_csv_path} (Shape: {df_final.shape})")
    
    # Print distribution
    logger.info("Distribusi Kelas:")
    for split_name in ['train', 'dev', 'test']:
        counts = df_final[df_final['split'] == split_name]['label_3kelas'].value_counts().sort_index()
        logger.info(f"  {split_name.upper()}: {dict(counts)}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
    
    logger.info("Memulai ekstraksi fitur akustik...")
    start_t = time.time()
    build_dataset_and_extract_features(CLEANED_DIR, OUTPUT_DIR)
    end_t = time.time()
    logger.info(f"Selesai mengekstrak fitur dalam {end_t - start_t:.2f} detik.")
