"""
Dataloader untuk Wav2Vec2 - Mental Health Audio Classification
Membaca langsung dari data/cleaned/{pid}.wav

📝 DEEP LEARNING PIPELINE NOTE: WAV2VEC2 DATALOADER & PREPROCESSING
- Role: Loads raw mono audio waveforms directly for fine-tuning the Wav2Vec2 model.
- Key Operations:
  1. Resampling & Mono: Ensures 16,000 Hz, mono.
  2. Amplitude Normalization: Peak scaling to [-1.0, 1.0].
  3. Padding/Truncating: Formats waveform length to exactly 30 seconds (30s * 16kHz = 480,000 samples).
  4. Anti-Leakage Split: Patient-level splits to guarantee separate training and validation patients.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CLEANED_DIR  = PROJECT_ROOT / "data" / "cleaned"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"

CLASS_NAMES  = ["NORMAL", "DEPRESI"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

AUG_TAGS     = ["_noise", "_pitch", "_stretch", "_combo"]


# ── Dataset ──────────────────────────────────────────────────────────────────
class Wav2VecDataset(Dataset):
    """
    Membaca raw waveform dari data/cleaned/{pid}.wav
    Output: dict dengan 'waveform' (float32 tensor 1D) dan 'label'
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],   # list of (wav_path, label_idx)
        sample_rate: int = 16000,
        max_duration_sec: float = 30.0,
    ):
        self.samples        = samples
        self.labels         = [s[1] for s in samples]
        self.sample_rate    = sample_rate
        self.max_len        = int(max_duration_sec * sample_rate)

        from collections import Counter
        dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter(self.labels).items())}
        logging.info(f"Dataset: {len(samples)} samples | {dist}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]

        waveform, sr = sf.read(str(wav_path))

        # Stereo → mono
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(np.float32)

        # Resample jika perlu
        if sr != self.sample_rate:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        # Normalize
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val

        # Crop / pad ke max_len
        if len(waveform) > self.max_len:
            waveform = waveform[:self.max_len]
        elif len(waveform) < self.max_len:
            waveform = np.pad(waveform, (0, self.max_len - len(waveform)), mode="constant")

        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "label":    torch.tensor(label, dtype=torch.long),
        }


# ── Label loading ─────────────────────────────────────────────────────────────
def load_patient_labels(label_csv: Optional[Path] = None) -> Dict[str, str]:
    """Return dict {pid_str: 'NORMAL'|'DEPRESI'}"""
    if label_csv is None:
        label_csv = SPLITS_DIR / "custom_2class_labels.csv"

    df     = pd.read_csv(label_csv)
    id_col = "Participant_ID" if "Participant_ID" in df.columns else "participant_id"

    if "label_depresi" in df.columns:
        mapping = df["label_depresi"].astype(int).map({0: "NORMAL", 1: "DEPRESI"})
    elif "Custom_Label" in df.columns:
        mapping = df["Custom_Label"].astype(str).str.upper()
    else:
        raise KeyError(f"Kolom label tidak ditemukan. Kolom ada: {list(df.columns)}")

    return dict(zip(df[id_col].astype(str), mapping))


# ── Collect all wav files ─────────────────────────────────────────────────────
def collect_all_files(
    cleaned_dir: Optional[Path] = None,
    label_csv: Optional[Path] = None,
) -> Tuple[List[str], List[int], List[Tuple[Path, str, int]]]:
    """
    Return:
        patients_list  : sorted list of patient IDs
        patients_labels: label index per patient
        all_files      : list of (wav_path, pid, label_idx)
    """
    if cleaned_dir is None:
        cleaned_dir = CLEANED_DIR

    patient_to_label = load_patient_labels(label_csv)

    all_files  = []
    available  = set()

    for wav_path in sorted(cleaned_dir.glob("*.wav")):
        pid = wav_path.stem   # format: "300", "301", ...
        if pid not in patient_to_label:
            continue
        label_name = patient_to_label[pid]
        if label_name not in CLASS_TO_IDX:
            continue
        label_idx = CLASS_TO_IDX[label_name]
        all_files.append((wav_path, pid, label_idx))
        available.add(pid)

    patients_list   = sorted(available)
    patients_labels = [CLASS_TO_IDX[patient_to_label[p]] for p in patients_list]

    from collections import Counter
    dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter(patients_labels).items())}
    logging.info(f"Total pasien: {len(patients_list)} | Distribusi: {dist}")
    logging.info(f"Total files : {len(all_files)}")

    return patients_list, patients_labels, all_files


# ── Make DataLoader ───────────────────────────────────────────────────────────
def make_loader(
    samples: List[Tuple[Path, int]],
    batch_size: int,
    shuffle: bool = True,
    weighted: bool = False,
    sample_rate: int = 16000,
    max_duration_sec: float = 30.0,
    num_workers: int = 2,
) -> DataLoader:
    ds = Wav2VecDataset(samples, sample_rate=sample_rate, max_duration_sec=max_duration_sec)

    if weighted and shuffle:
        labels_t = torch.tensor(ds.labels, dtype=torch.long)
        counts   = torch.bincount(labels_t, minlength=2).float().clamp(min=1)
        weights  = 1.0 / counts[labels_t]
        weights  = weights / weights.sum()
        sampler  = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        return DataLoader(
            ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ── 5-Fold CV split ───────────────────────────────────────────────────────────
def get_cv_splits(
    patients_list: List[str],
    patients_labels: List[int],
    all_files: List[Tuple[Path, str, int]],
    n_folds: int = 5,
    test_size: float = 0.10,
    random_state: int = 42,
):
    """
    Return:
        test_samples  : list of (wav_path, label_idx) — fixed test set
        fold_splits   : list of (train_samples, val_samples) per fold
    """
    # 1. Pisahkan test set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(sss.split(patients_list, patients_labels))

    trainval_patients = [patients_list[i] for i in trainval_idx]
    trainval_labels   = [patients_labels[i] for i in trainval_idx]
    test_pids         = {patients_list[i] for i in test_idx}

    # Test: hanya original files
    test_samples = [
        (p, cls) for p, pid, cls in all_files
        if pid in test_pids
    ]
    logging.info(f"Test set: {len(test_samples)} samples dari {len(test_pids)} pasien")

    # 2. 5-Fold pada trainval
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_splits = []

    for tr_idx, val_idx in skf.split(trainval_patients, trainval_labels):
        train_pids = {trainval_patients[i] for i in tr_idx}
        val_pids   = {trainval_patients[i] for i in val_idx}

        train_s = [(p, cls) for p, pid, cls in all_files if pid in train_pids]
        val_s   = [(p, cls) for p, pid, cls in all_files if pid in val_pids]

        fold_splits.append((train_s, val_s))

    return test_samples, fold_splits


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    patients, labels, files = collect_all_files()
    test_samples, fold_splits = get_cv_splits(patients, labels, files)

    print(f"\nTest set  : {len(test_samples)} samples")
    for i, (tr, vl) in enumerate(fold_splits, 1):
        from collections import Counter
        tr_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter([s[1] for s in tr]).items())}
        vl_dist = {CLASS_NAMES[k]: v for k, v in sorted(Counter([s[1] for s in vl]).items())}
        print(f"Fold {i}: Train {tr_dist} | Val {vl_dist}")
