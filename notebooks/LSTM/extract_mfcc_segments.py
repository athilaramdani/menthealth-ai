"""
LSTM Step 1 - MFCC Extraction with 10-Second Segmentation
===========================================================
Project : Mental Health Audio Classification (NORMAL vs DEPRESI)
Script  : notebooks/LSTM/extract_mfcc_segments.py

📝 DEEP LEARNING PIPELINE NOTE: MFCC & SEGMENTATION FEATURE ENGINEERING
- Role: Prepares sequential features for the BiLSTM model.
- Key Operations:
  1. 10-Second Segmentation: Splits cleaned participant audio into non-overlapping 10s segments.
  2. Feature Extraction: Extracts 40 MFCC coefficients, calculates Delta and Delta-Delta (total 120 features).
  3. Normalization: Z-score normalization per coefficient.
  4. Final Dimension: (T_frames=313, 120 features).
  5. Multiplier Effect: Boosts sample count from 189 patients to ~3,000-6,000 segments.

Strategi segmentasi:
  - Setiap file cleaned audio (.wav) dipotong menjadi segmen NON-OVERLAPPING 10 detik.
  - Segmen terakhir yang lebih pendek dari MIN_SEG_SEC detik akan dibuang.
  - Setiap segmen menghasilkan 1 file .npy berisi matriks MFCC shape (T_frames, n_mfcc).
  - Label segmen = label participant aslinya (segment-level label = patient-level label).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# ── Path Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CLEANED_DIR  = PROJECT_ROOT / "data" / "cleaned"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
MFCC_OUT_DIR = PROJECT_ROOT / "data" / "features" / "mfcc"
LOG_FILE     = SCRIPT_DIR / "extract_mfcc.log"

# ── Konfigurasi MFCC & Segmentasi ─────────────────────────────────────────────
TARGET_SR    = 16000    # Hz — konsisten dengan preprocessing pipeline
N_MFCC       = 40       # jumlah koefisien MFCC
HOP_LENGTH   = 512      # hop length untuk MFCC frames (~32ms @ 16kHz)
N_FFT        = 1024     # FFT window size (~64ms @ 16kHz)
SEG_SEC      = 10.0     # durasi setiap segmen dalam detik
MIN_SEG_SEC  = 4.0      # segmen lebih pendek dari ini akan dibuang
CLASS_NAMES  = ["NORMAL", "DEPRESI"]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Paksa stdout encoding utf-8 di Windows agar karakter non-ASCII aman
import io
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
logger = logging.getLogger("MFCC-Extractor")


def extract_mfcc_from_segment(y_seg: np.ndarray, sr: int) -> np.ndarray:
    """
    Ekstrak fitur MFCC dari segmen audio.

    Pipeline:
      1. Hitung MFCC (n_mfcc koefisien) + Delta + Delta-Delta.
      2. Normalisasi per-koefisien (z-score) — mencegah dominasi koefisien besar.
      3. Transpose ke shape (T_frames, n_mfcc * 3) untuk input LSTM.

    Returns:
        np.ndarray: shape (T_frames, n_mfcc * 3), dtype float32
    """
    # Raw MFCC
    mfcc = librosa.feature.mfcc(
        y=y_seg, sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )                                           # shape: (n_mfcc, T)

    # Delta dan Delta-Delta — menangkap dinamika temporal
    delta1 = librosa.feature.delta(mfcc)       # shape: (n_mfcc, T)
    delta2 = librosa.feature.delta(mfcc, order=2)  # shape: (n_mfcc, T)

    # Gabungkan → (n_mfcc*3, T)
    features = np.concatenate([mfcc, delta1, delta2], axis=0)

    # Z-score normalisasi per koefisien (per baris)
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True) + 1e-8
    features = (features - mean) / std

    # Transpose → (T, n_mfcc*3) — format input LSTM: (timesteps, features)
    return features.T.astype(np.float32)


def segment_and_extract(wav_path: Path, participant_id: str, label_name: str,
                         out_dir: Path) -> dict:
    """
    Muat audio cleaned, potong jadi segmen 10 detik, ekstrak MFCC tiap segmen,
    simpan sebagai .npy terpisah.

    Returns:
        dict: statistik proses (n_segments, total_duration, status, error)
    """
    stats = {
        "participant_id": participant_id,
        "label": label_name,
        "status": "FAILED",
        "n_segments_saved": 0,
        "n_segments_dropped": 0,
        "total_duration_sec": 0.0,
        "error": "",
    }

    try:
        # Load audio (sudah di-resample ke 16kHz saat cleaning)
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        total_sec = len(y) / sr
        stats["total_duration_sec"] = round(total_sec, 2)

        if total_sec < MIN_SEG_SEC:
            stats["error"] = f"Durasi terlalu pendek ({total_sec:.1f}s < {MIN_SEG_SEC}s)"
            return stats

        # Panjang segmen dalam samples
        seg_samples = int(SEG_SEC * sr)
        min_samples  = int(MIN_SEG_SEC * sr)

        n_segments = int(total_sec // SEG_SEC)
        saved = 0
        dropped = 0

        for seg_idx in range(n_segments + 1):  # +1 untuk handle sisa akhir
            start = seg_idx * seg_samples
            end   = start + seg_samples

            if start >= len(y):
                break

            y_seg = y[start:end]

            # Buang segmen yang terlalu pendek
            if len(y_seg) < min_samples:
                dropped += 1
                continue

            # Pad segmen terakhir bila perlu (lebih pendek dari 10 detik tapi > MIN)
            if len(y_seg) < seg_samples:
                pad_len = seg_samples - len(y_seg)
                y_seg = np.pad(y_seg, (0, pad_len), mode="constant")

            # Ekstrak MFCC
            mfcc_features = extract_mfcc_from_segment(y_seg, sr)

            # Simpan .npy
            out_filename = f"{participant_id}_seg{seg_idx:03d}.npy"
            out_path = out_dir / out_filename
            np.save(str(out_path), mfcc_features)
            saved += 1

        stats["n_segments_saved"]  = saved
        stats["n_segments_dropped"] = dropped
        stats["status"] = "SUCCESS"

        logger.info(
            f"[{participant_id}] {label_name}: {total_sec:.0f}s "
            f"-> {saved} segmen disimpan, {dropped} dibuang"
        )

    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"[{participant_id}] Gagal: {e}")

    return stats


def run_extraction():
    """Pipeline utama: baca label CSV → proses semua participant → simpan log."""

    # Buat folder output
    for cls in CLASS_NAMES:
        (MFCC_OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    # Baca label
    label_path = SPLITS_DIR / "custom_2class_labels.csv"
    if not label_path.exists():
        logger.error(f"File label tidak ditemukan: {label_path}")
        sys.exit(1)

    df = pd.read_csv(label_path)
    # Kolom: participant_id, label_depresi, split
    label_map = dict(zip(df["participant_id"].astype(str), df["label_depresi"].astype(int)))
    logger.info(f"Total participant di label CSV: {len(label_map)}")

    # Scan cleaned WAV
    wav_files = sorted(CLEANED_DIR.glob("*.wav"))
    logger.info(f"Total WAV di cleaned/: {len(wav_files)}")

    all_stats = []
    total_saved = 0

    for wav_path in tqdm(wav_files, desc="Extracting MFCC Segments"):
        pid = wav_path.stem  # mis. "303"

        if pid not in label_map:
            logger.warning(f"Participant {pid} tidak ada di label CSV — dilewati.")
            continue

        label_int  = label_map[pid]
        label_name = "DEPRESI" if label_int == 1 else "NORMAL"
        out_dir    = MFCC_OUT_DIR / label_name

        stats = segment_and_extract(wav_path, pid, label_name, out_dir)
        all_stats.append(stats)
        total_saved += stats["n_segments_saved"]

    # Simpan log CSV
    log_df = pd.DataFrame(all_stats)
    log_path = MFCC_OUT_DIR / "extraction_log.csv"
    log_df.to_csv(log_path, index=False)

    # Ringkasan
    success = log_df[log_df["status"] == "SUCCESS"]
    logger.info("\n" + "=" * 60)
    logger.info("MFCC EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Participant berhasil  : {len(success)} / {len(log_df)}")
    logger.info(f"Total segmen tersimpan: {total_saved}")

    if len(success) > 0:
        normal_count  = len(list((MFCC_OUT_DIR / "NORMAL").glob("*.npy")))
        depresi_count = len(list((MFCC_OUT_DIR / "DEPRESI").glob("*.npy")))
        logger.info(f"Segmen NORMAL  : {normal_count}")
        logger.info(f"Segmen DEPRESI : {depresi_count}")
        logger.info(f"Total segmen   : {normal_count + depresi_count}")
        logger.info(
            f"Feature shape tiap segmen: "
            f"({int(SEG_SEC * TARGET_SR / HOP_LENGTH) + 1}, {N_MFCC * 3})  "
            f"→ (timesteps, n_mfcc*3)"
        )
    logger.info(f"Log disimpan di: {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("Memulai ekstraksi MFCC dengan segmentasi 10 detik...")
    logger.info(f"Cleaned dir : {CLEANED_DIR}")
    logger.info(f"Output dir  : {MFCC_OUT_DIR}")
    logger.info(f"Seg duration: {SEG_SEC}s | Min seg: {MIN_SEG_SEC}s")
    logger.info(f"n_mfcc      : {N_MFCC} (+delta+delta2 → {N_MFCC * 3} features)")
    run_extraction()
