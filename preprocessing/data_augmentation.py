import logging
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==========================================
# KONFIGURASI PATH & LOGGING
# ==========================================

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DATA_DIR    = PROJECT_ROOT / "data" / "raw" / "DAIC-WOZ"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
FEATURES_DIR    = PROJECT_ROOT / "data" / "features" / "spectrogram"
SPLITS_DIR      = PROJECT_ROOT / "data" / "splits"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "data_augmentation.log"),
        logging.StreamHandler()
    ]
)

# Kelas minoritas yang mendapat augmentasi ekstra
MINORITY_LABELS = {'STRES', 'CEMAS'}

# max_len harus konsisten dengan generate_spectrogram.py dan dataloader
MAX_LEN = 800

# ==========================================
# FUNGSI AUGMENTASI
# ==========================================

def add_noise(y, noise_factor=0.005):
    """Gaussian White Noise."""
    noise = np.random.randn(len(y))
    return (y + noise_factor * noise).astype(np.float32)


def apply_pitch_shift(y, sr, n_steps=2):
    """Pitch shift naik/turun sebesar n_steps semitone."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def apply_time_stretch(y, rate=1.1):
    """Time stretch: >1 = lebih cepat, <1 = lebih lambat."""
    return librosa.effects.time_stretch(y=y, rate=rate)


def extract_and_save_spectrogram(y, sr, output_path, n_mels=128, max_len=MAX_LEN):
    """
    Generate Mel-Spectrogram, Z-score normalisasi, padding/truncate, dan save.
    Konsisten dengan generate_spectrogram.py.
    """
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Z-score normalisasi
    S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)

    if max_len is not None:
        if S_dB.shape[1] > max_len:
            S_dB = S_dB[:, :max_len]
        else:
            pad_width = max_len - S_dB.shape[1]
            S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')

    np.save(output_path, S_dB)


# ==========================================
# HELPER: BACA LABEL DICT
# ==========================================

def get_label_dict():
    """Membaca label dari custom_4class_labels.csv."""
    label_path = SPLITS_DIR / "custom_4class_labels.csv"
    if not label_path.exists():
        logging.error(
            "custom_4class_labels.csv tidak ditemukan! "
            "Jalankan generate_spectrogram.py terlebih dahulu."
        )
        return None

    df     = pd.read_csv(label_path)
    id_col = 'Participant_ID' if 'Participant_ID' in df.columns else 'participant_ID'
    return dict(zip(df[id_col].astype(str), df['Custom_Label']))


# ==========================================
# PIPELINE AUGMENTASI
# ==========================================

def process_augmentations():
    """
    Generate augmentasi untuk semua kelas (noise, pitch, stretch).
    Kelas minoritas (STRES, CEMAS) mendapat 3 augmentasi tambahan
    untuk mengurangi ketidakseimbangan kelas.

    Catatan desain:
    - Augmentasi hanya dilakukan pada data CLEANED (bukan pada .npy),
      agar proses bebas dari kebocoran val set.
    - Pastikan dataloader hanya membaca augmentasi dari folder TRAIN split,
      bukan dari val/test.
    """
    label_dict = get_label_dict()
    if label_dict is None:
        return

    audio_files = list(CLEANED_DATA_DIR.glob("*.wav"))
    logging.info(f"Ditemukan {len(audio_files)} file audio di {CLEANED_DATA_DIR}")

    success_count = 0
    skipped_count = 0
    error_count   = 0

    for audio_path in tqdm(audio_files, desc="Data Augmentation"):
        patient_id = audio_path.stem

        if patient_id not in label_dict:
            skipped_count += 1
            continue

        custom_label = label_dict[patient_id]
        class_dir    = FEATURES_DIR / custom_label

        # Paths augmentasi standar (SEMUA KELAS dapat 6 augmentasi)
        noise_path        = class_dir / f"{patient_id}_mel_noise.npy"
        noise2_path       = class_dir / f"{patient_id}_mel_noise2.npy"
        pitch_path        = class_dir / f"{patient_id}_mel_pitch.npy"
        pitch_down_path   = class_dir / f"{patient_id}_mel_pitch_down.npy"
        stretch_path      = class_dir / f"{patient_id}_mel_stretch.npy"
        stretch_slow_path = class_dir / f"{patient_id}_mel_stretch_slow.npy"

        all_aug_paths = [noise_path, noise2_path, pitch_path, pitch_down_path, stretch_path, stretch_slow_path]
        all_done = all([p.exists() for p in all_aug_paths])

        if all_done:
            logging.debug(f"{patient_id}: Semua augmentasi sudah ada, dilewati.")
            success_count += 1
            continue

        try:
            logging.info(f"Augmentasi {patient_id} -> [{custom_label}]")
            y, sr = librosa.load(audio_path, sr=16000)

            # --- 6 Augmentasi untuk SEMUA KELAS ---
            if not noise_path.exists():
                extract_and_save_spectrogram(add_noise(y, 0.005), sr, noise_path)

            if not noise2_path.exists():
                extract_and_save_spectrogram(add_noise(y, 0.008), sr, noise2_path)

            if not pitch_path.exists():
                extract_and_save_spectrogram(apply_pitch_shift(y, sr, n_steps=2), sr, pitch_path)

            if not pitch_down_path.exists():
                extract_and_save_spectrogram(apply_pitch_shift(y, sr, n_steps=-2), sr, pitch_down_path)

            if not stretch_path.exists():
                extract_and_save_spectrogram(apply_time_stretch(y, rate=1.1), sr, stretch_path)

            if not stretch_slow_path.exists():
                extract_and_save_spectrogram(apply_time_stretch(y, rate=0.9), sr, stretch_slow_path)

            success_count += 1

        except Exception as e:
            logging.error(f"GAGAL augmentasi {patient_id}: {e}")
            error_count += 1

    logging.info("=" * 50)
    logging.info("DATA AUGMENTATION SELESAI")
    logging.info(f"Berhasil    : {success_count} pasien")
    logging.info(f"Gagal       : {error_count} pasien")
    logging.info(f"Tanpa label : {skipped_count} pasien")
    logging.info("=" * 50)


if __name__ == "__main__":
    process_augmentations()