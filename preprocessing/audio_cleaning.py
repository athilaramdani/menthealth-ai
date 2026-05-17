# %% [markdown]
# # DAIC-WOZ Audio Preprocessing Pipeline
# 
# Project: Mental Health Classification based on Audio
# Location: `menthealth-ai-main/preprocessing/audio_cleaning.py`
# 
# This notebook/script performs complete preprocessing of the DAIC-WOZ audio dataset. 
# It includes resampling, mono conversion, noise reduction, transcript-based speaker isolation (Voice Activity Detection/Diarization), silence trimming, optional bandpass filtering, and amplitude normalization.
# 
# ## Technical Analysis of Metadata Files
# Before building the pipeline, we technically analyzed the usefulness of the accompanying metadata files:
# 
# 1. **`*_TRANSCRIPT.csv` (Extremely High Relevance - Critical)**:
#    - **Structure**: Tab-separated values with `start_time`, `stop_time`, `speaker`, and `value`.
#    - **Role**: This is the most crucial file for preprocessing. The raw audio contains dialogue between the virtual interviewer (`Ellie`) and the `Participant`. Ellie's voice is synthetic and structurally constant, meaning it acts as a significant confounding variable for deep learning models.
#    - **Use Case**: We use the transcript timestamps to build a **Speaker Isolation Mask**. By extracting and concatenating only the segments where `speaker == 'Participant'`, we achieve perfect speaker diarization and VAD, removing Ellie's voice and turn-taking silence completely.
# 
# 2. **`*_COVAREP.csv` (Low Relevance for Preprocessing, High for Feature Extraction)**:
#    - **Structure**: 74 acoustic features (F0, VUV - Voiced/Unvoiced Decision, H1-H2, NAQ, QOQ, MFCCs) at a 10ms frame rate.
#    - **Role**: While the Voiced/Unvoiced Decision (VUV) column could theoretically act as a speech mask, it does **not** distinguish between Ellie and the Participant. Additionally, VUV masks frame-level vocal cord vibration. Applying a VUV mask directly to the waveform would clip unvoiced consonants (like /s/, /t/, /f/) and micro-pauses, heavily distorting the phonetic structure of the audio.
#    - **Use Case**: Ignored for audio preprocessing; best used in the downstream feature extraction or model training stage.
# 
# 3. **`*_FORMANT.csv` (Irrelevant for Preprocessing)**:
#    - **Structure**: 5 columns representing formant frequencies (F1 to F5) at a frame-level.
#    - **Role**: Vocal tract resonance frequencies have no timing, noise, or diarization utility.
#    - **Use Case**: Ignored entirely during preprocessing.
# 
# ---
# 
# Let's begin the preprocessing script implementation.

# %%
import os
import sys
import time
import logging
import warnings
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
from tqdm import tqdm

# Handle optional dependencies gracefully
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    warnings.warn("Library 'noisereduce' tidak ditemukan. Tahap Noise Reduction akan dilewati atau menggunakan alternatif sederhana.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocessing_pipeline.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("DAIC-WOZ-Preprocessor")

# Configuration Constants
TARGET_SR = 16000        # 16kHz
TARGET_CHANNELS = 1      # Mono
TARGET_NORM_DB = -20.0   # Target dBFS for peak/RMS normalization
DC_CUTOFF_HZ = 80.0      # Butterworth highpass filter cutoff

# %% [markdown]
# ## Technical Implementation of Preprocessing Stages
# 
# We implement the preprocessing pipeline stages modularly:
# 1. **Resampling & Mono Conversion**: Handled efficiently during loading with `librosa.load(sr=16000, mono=True)`.
# 2. **Speaker Isolation (Transcript VAD)**: Parses `TRANSCRIPT.csv` to crop only Participant audio segments, removing Ellie and turn-taking dead space.
# 3. **Optional Smoothing/Filtering (High-pass Filter)**: Butterworth high-pass filter at 80Hz to eliminate low-frequency rumble, hum, and DC offset.
# 4. **Noise Reduction**: Uses spectral gating via `noisereduce` to clean background hum and static.
# 5. **Trim Silence**: Removes leading and trailing absolute silence from the participant speech turns.
# 6. **Amplitude Normalization**: Peak normalization to -1.0 to 1.0 range, scaled to 0.9 maximum amplitude to ensure volume consistency across speakers.

# %%
def apply_butterworth_highpass(y, sr, cutoff=80.0, order=5):
    """
    Apply a Butterworth high-pass filter to remove DC offset and low-frequency rumble.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_y = lfilter(b, a, y)
    return filtered_y

def isolate_participant_speech(y, sr, transcript_path):
    """
    Isolates and concatenates speech turns belonging only to the Participant 
    using timestamps from the TRANSCRIPT.csv file.
    
    Returns:
        np.ndarray: Concatenated participant speech waveform.
        dict: Turning statistics ( Ellie turns, Participant turns, etc. )
    """
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript tidak ditemukan di: {transcript_path}")
        
    # Read the tab-separated transcript
    try:
        df = pd.read_csv(transcript_path, sep='\t')
    except Exception as e:
        # Fallback to standard comma/semicolon separation if tab fails
        df = pd.read_csv(transcript_path)
        if len(df.columns) < 4:
            df = pd.read_csv(transcript_path, sep=None, engine='python')
            
    # Normalize column names to lowercase and strip whitespace
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Ensure correct columns exist
    required_cols = {'start_time', 'stop_time', 'speaker'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Kolom transcript tidak valid. Kolom ditemukan: {list(df.columns)}")
        
    # Filter only Participant speech segments
    participant_df = df[df['speaker'].str.lower().str.strip() == 'participant']
    ellie_df = df[df['speaker'].str.lower().str.strip() == 'ellie']
    
    participant_segments = []
    total_audio_length_samples = len(y)
    
    for _, row in participant_df.iterrows():
        start_sec = float(row['start_time'])
        stop_sec = float(row['stop_time'])
        
        # Convert seconds to sample indices
        start_idx = int(start_sec * sr)
        stop_idx = int(stop_sec * sr)
        
        # Bounds checking to avoid index errors
        start_idx = max(0, min(start_idx, total_audio_length_samples))
        stop_idx = max(0, min(stop_idx, total_audio_length_samples))
        
        if stop_idx > start_idx:
            segment = y[start_idx:stop_idx]
            participant_segments.append(segment)
            
    if not participant_segments:
        logger.warning(f"Tidak ada segmen participant yang ditemukan dalam transcript {os.path.basename(transcript_path)}!")
        return np.array([], dtype=np.float32), {
            "ellie_turns": len(ellie_df),
            "participant_turns": 0,
            "speech_ratio": 0.0
        }
        
    # Concatenate all isolated participant segments
    isolated_audio = np.concatenate(participant_segments)
    
    stats = {
        "ellie_turns": len(ellie_df),
        "participant_turns": len(participant_df),
        "speech_ratio": len(isolated_audio) / total_audio_length_samples if total_audio_length_samples > 0 else 0.0
    }
    
    return isolated_audio, stats

def clean_audio(audio_path, transcript_path):
    """
    Full audio preprocessing pipeline for a single participant.
    
    Stages:
      1. Load audio, resample to 16kHz and convert to mono.
      2. High-pass filter (>80Hz) to remove low-frequency rumble & DC offset.
      3. Isolate participant speech turns using TRANSCRIPT.csv timestamps.
      4. Apply spectral-gating Noise Reduction using noisereduce.
      5. Trim silence from leading/trailing parts of isolated speech.
      6. Peak normalize to max amplitude of 0.9.
      
    Returns:
        np.ndarray: Preprocessed and cleaned audio waveform.
        dict: Preprocessing metrics & metadata.
    """
    logger.info(f"Memulai preprocessing untuk file: {os.path.basename(audio_path)}")
    metrics = {
        "status": "FAILED",
        "original_duration_sec": 0.0,
        "cleaned_duration_sec": 0.0,
        "participant_turns": 0,
        "ellie_turns": 0,
        "speech_ratio": 0.0,
        "max_amplitude_before": 0.0,
        "max_amplitude_after": 0.0,
        "has_nan_inf": False,
        "error_msg": ""
    }
    
    try:
        # Stage 1: Load and Resample/Mono-convert
        # (Using sr=TARGET_SR automatically resamples; mono=True averages channels if stereo)
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        metrics["original_duration_sec"] = float(len(y) / sr)
        metrics["max_amplitude_before"] = float(np.max(np.abs(y)))
        
        if len(y) == 0:
            raise ValueError("File audio kosong.")
            
        # Stage 2: Butterworth High-pass filtering
        y_filtered = apply_butterworth_highpass(y, sr, cutoff=DC_CUTOFF_HZ)
        
        # Stage 3: Speaker Isolation (Transcript Diarization VAD)
        y_participant, turn_stats = isolate_participant_speech(y_filtered, sr, transcript_path)
        metrics["participant_turns"] = turn_stats["participant_turns"]
        metrics["ellie_turns"] = turn_stats["ellie_turns"]
        metrics["speech_ratio"] = turn_stats["speech_ratio"]
        
        if len(y_participant) == 0:
            raise ValueError("Hasil isolasi suara participant kosong.")
            
        # Stage 4: Noise Reduction (Spectral Gating)
        if HAS_NOISEREDUCE:
            # We estimate noise from the beginning of the original clip or a silent part,
            # or let noisereduce auto-estimate over the waveform.
            y_denoised = nr.reduce_noise(y=y_participant, sr=sr, prop_decrease=0.85)
        else:
            # Simple fallback: mild gate or identity
            y_denoised = y_participant
            
        # Stage 5: Trim silence (leading/trailing absolute silence)
        # top_db=30 means anything 30dB below reference is considered silence
        y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=30)
        
        if len(y_trimmed) == 0:
            logger.warning("Trimming silence menghasilkan audio kosong. Menggunakan audio sebelum trim.")
            y_trimmed = y_denoised
            
        # Stage 6: Peak Normalization to max 0.90 amplitude
        max_val = np.max(np.abs(y_trimmed))
        if max_val > 0:
            y_normalized = y_trimmed * (0.90 / max_val)
        else:
            y_normalized = y_trimmed
            
        # Validation checks
        has_nan = np.isnan(y_normalized).any()
        has_inf = np.isinf(y_normalized).any()
        metrics["has_nan_inf"] = bool(has_nan or has_inf)
        
        if has_nan or has_inf:
            raise ValueError("Output audio mengandung NaN atau Infinite values.")
            
        metrics["cleaned_duration_sec"] = float(len(y_normalized) / sr)
        metrics["max_amplitude_after"] = float(np.max(np.abs(y_normalized)))
        metrics["status"] = "SUCCESS"
        
        logger.info(f"Berhasil memproses {os.path.basename(audio_path)}. "
                    f"Durasi: {metrics['original_duration_sec']:.2f}s -> {metrics['cleaned_duration_sec']:.2f}s "
                    f"(Speech ratio: {metrics['speech_ratio'] * 100:.1f}%)")
        
        return y_normalized, metrics
        
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Gagal memproses {os.path.basename(audio_path)}: {err_msg}")
        metrics["error_msg"] = err_msg
        return None, metrics

# %% [markdown]
# ## Global Execution Loop
# 
# We implement the runner that scans `data/raw/DAIC-WOZ`, matches metadata, skips corrupt folders gracefully, and generates a robust log.

# %%
def run_dataset_preprocessing(raw_dir, cleaned_dir):
    """
    Scans the raw directory, processes all matched participant audio, 
    saves cleaned audio, and saves a global CSV preprocessing report.
    """
    os.makedirs(cleaned_dir, exist_ok=True)
    
    if not os.path.exists(raw_dir):
        logger.error(f"Directory raw data tidak ditemukan: {raw_dir}")
        return
        
    # Scan all directories like 300_P, 301_P, etc.
    participant_folders = [
        d for d in os.listdir(raw_dir) 
        if os.path.isdir(os.path.join(raw_dir, d)) and d.endswith('_P')
    ]
    participant_folders = sorted(participant_folders)
    
    logger.info(f"Ditemukan {len(participant_folders)} folder participant di {raw_dir}")
    
    preprocessing_records = []
    success_count = 0
    failure_count = 0
    
    # Preprocess each participant
    for folder in tqdm(participant_folders, desc="Preprocessing DAIC-WOZ Patients"):
        participant_id = folder.split('_')[0]
        folder_path = os.path.join(raw_dir, folder)
        
        # Construct expected paths
        audio_name = f"{participant_id}_AUDIO.wav"
        transcript_name = f"{participant_id}_TRANSCRIPT.csv"
        covarep_name = f"{participant_id}_COVAREP.csv"
        formant_name = f"{participant_id}_FORMANT.csv"
        
        audio_path = os.path.join(folder_path, audio_name)
        transcript_path = os.path.join(folder_path, transcript_name)
        covarep_path = os.path.join(folder_path, covarep_name)
        formant_path = os.path.join(folder_path, formant_name)
        
        record = {
            "participant_id": participant_id,
            "folder_name": folder,
            "has_covarep": os.path.exists(covarep_path),
            "has_formant": os.path.exists(formant_path),
            "original_duration_sec": 0.0,
            "cleaned_duration_sec": 0.0,
            "speech_ratio": 0.0,
            "participant_turns": 0,
            "ellie_turns": 0,
            "status": "SKIPPED",
            "error_message": ""
        }
        
        # Basic validation of files
        if not os.path.exists(audio_path):
            record["status"] = "FAILED"
            record["error_message"] = f"Audio file {audio_name} tidak ditemukan."
            preprocessing_records.append(record)
            failure_count += 1
            continue
            
        if not os.path.exists(transcript_path):
            record["status"] = "FAILED"
            record["error_message"] = f"Transcript file {transcript_name} tidak ditemukan."
            preprocessing_records.append(record)
            failure_count += 1
            continue
            
        # Run the cleaning pipeline
        cleaned_y, metrics = clean_audio(audio_path, transcript_path)
        
        # Update records with metrics
        record.update({
            "original_duration_sec": metrics["original_duration_sec"],
            "cleaned_duration_sec": metrics["cleaned_duration_sec"],
            "speech_ratio": metrics["speech_ratio"],
            "participant_turns": metrics["participant_turns"],
            "ellie_turns": metrics["ellie_turns"],
            "status": metrics["status"],
            "error_message": metrics["error_msg"]
        })
        
        if metrics["status"] == "SUCCESS" and cleaned_y is not None:
            # Save the processed clean audio
            output_filename = f"{participant_id}.wav"
            output_path = os.path.join(cleaned_dir, output_filename)
            
            try:
                sf.write(output_path, cleaned_y, TARGET_SR, subtype='PCM_16')
                success_count += 1
            except Exception as e:
                record["status"] = "FAILED"
                record["error_message"] = f"Gagal menulis audio file: {str(e)}"
                failure_count += 1
        else:
            failure_count += 1
            
        preprocessing_records.append(record)
        
    # Write a comprehensive log file to cleaned directory
    log_df = pd.DataFrame(preprocessing_records)
    log_csv_path = os.path.join(cleaned_dir, "preprocessing_log.csv")
    log_df.to_csv(log_csv_path, index=False)
    
    # Calculate statistics
    logger.info("=== PREPROCESSING STATISTICS SUMMARY ===")
    logger.info(f"Jumlah file diproses: {len(preprocessing_records)}")
    logger.info(f"Berhasil: {success_count}")
    logger.info(f"Gagal: {failure_count}")
    
    if success_count > 0:
        success_df = log_df[log_df['status'] == 'SUCCESS']
        avg_orig = success_df['original_duration_sec'].mean()
        avg_clean = success_df['cleaned_duration_sec'].mean()
        avg_ratio = success_df['speech_ratio'].mean()
        
        logger.info(f"Rata-rata Durasi Audio Asli: {avg_orig:.2f} detik (~{avg_orig/60:.2f} menit)")
        logger.info(f"Rata-rata Durasi Audio Bersih (Hanya Participant): {avg_clean:.2f} detik (~{avg_clean/60:.2f} menit)")
        logger.info(f"Rata-rata Speech Ratio Participant: {avg_ratio * 100:.1f}%")
        logger.info(f"Total durasi audio bersih yang siap dipakai: {success_df['cleaned_duration_sec'].sum() / 3600:.2f} jam")
        
    logger.info(f"Log preprocessing disimpan di: {log_csv_path}")

# %% [markdown]
# ## Main Block
# Executes the script when run directly or converted into notebook.

# %%
if __name__ == "__main__":
    # Define absolute or relative paths aligned with directory structure
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
    CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
    
    logger.info("Memulai pipeline preprocessing audio DAIC-WOZ...")
    logger.info(f"Path Input (Raw): {RAW_DATA_DIR}")
    logger.info(f"Path Output (Cleaned): {CLEANED_DATA_DIR}")
    
    start_time = time.time()
    run_dataset_preprocessing(RAW_DATA_DIR, CLEANED_DATA_DIR)
    end_time = time.time()
    
    logger.info(f"Pipeline selesai dalam {end_time - start_time:.2f} detik.")
