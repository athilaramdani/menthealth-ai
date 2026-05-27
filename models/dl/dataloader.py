import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# ==========================================
# KONFIGURASI PATH
# ==========================================

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "spectrogram"

CLASS_NAMES    = ['NORMAL', 'STRES', 'CEMAS', 'DEPRESI']
CLASS_TO_IDX   = {name: idx for idx, name in enumerate(CLASS_NAMES)}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# DATASET
# ==========================================

class MelSpectrogramDataset(Dataset):
    """
    Dataset yang membaca file .npy dari folder:
        features/spectrogram/NORMAL/
        features/spectrogram/STRES/
        features/spectrogram/CEMAS/
        features/spectrogram/DEPRESI/

    Setiap file .npy berisi Mel-Spectrogram shape (n_mels, max_len).
    Output tensor shape: (1, n_mels, max_len) — 1 channel untuk Conv2d.
    """

    def __init__(self, features_dir=FEATURES_DIR, max_len=800, samples=None):
        self.max_len = max_len
        if samples is not None:
            self.samples = samples
            self.labels  = [s[1] for s in samples]
        else:
            self.samples = []   # list of (path, label_idx)
            self.labels  = []   # list of label_idx (untuk StratifiedSplit)

            for class_name, class_idx in CLASS_TO_IDX.items():
                class_dir = Path(features_dir) / class_name
                if not class_dir.exists():
                    logging.warning(f"Folder kelas tidak ditemukan: {class_dir}")
                    continue

                npy_files = sorted(class_dir.glob("*.npy"))
                for npy_path in npy_files:
                    self.samples.append((npy_path, class_idx))
                    self.labels.append(class_idx)

            if len(self.samples) == 0:
                raise RuntimeError(
                    f"Tidak ada file .npy ditemukan di {features_dir}. "
                    "Pastikan generate_spectrogram.py dan data_augmentation.py sudah dijalankan."
                )

        logging.info(f"Total sampel ditemukan: {len(self.samples)}")
        self._log_distribution()

    def _log_distribution(self):
        from collections import Counter
        counts = Counter(self.labels)
        dist   = {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}
        logging.info(f"Distribusi dataset penuh: {dist}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]

        spec = np.load(npy_path)  # shape: (n_mels, time)

        # Padding / truncating (jaga-jaga file augmentasi punya panjang berbeda)
        if spec.shape[1] > self.max_len:
            spec = spec[:, :self.max_len]
        elif spec.shape[1] < self.max_len:
            pad_width = self.max_len - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')

        # Tambah channel dimension: (1, n_mels, max_len)
        spec_tensor  = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spec_tensor, label_tensor


# ==========================================
# DATALOADER FACTORY
# ==========================================

def get_dataloaders(
    batch_size=16,
    max_len=800,
    val_size=0.15,
    test_size=0.10,
    random_state=42,
    features_dir=FEATURES_DIR,
    use_weighted_sampler=False
):
    """
    Membuat train, val, dan test DataLoader dengan stratified split berbasis pasien (Patient-wise).

    Perbaikan utama vs versi sebelumnya:
    - Mencegah kebocoran data (data leakage) dengan melakukan StratifiedShuffleSplit pada Patient ID unik.
    - Train set mendapatkan data original + semua versi data augmentasi pasien tersebut.
    - Val & Test set hanya mendapatkan data original (*_mel.npy) agar evaluasi objektif dan realistis.
    - `use_weighted_sampler=True` dapat menyeimbangkan batch training untuk kelas minoritas ekstrem.
    """
    import pandas as pd

    # 1. Load label mapping untuk semua pasien
    label_path = PROJECT_ROOT / "data" / "splits" / "custom_4class_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(
            f"File label custom_4class_labels.csv tidak ditemukan di {label_path}. "
            "Pastikan generate_spectrogram.py sudah dijalankan."
        )

    df = pd.read_csv(label_path)
    id_col = 'Participant_ID' if 'Participant_ID' in df.columns else 'participant_ID'
    patient_to_label = dict(zip(df[id_col].astype(str), df['Custom_Label']))

    # 2. Temukan semua file .npy di folder spectrogram dan petakan ke pasien
    all_files = [] # list of (path, patient_id, label_idx)
    available_patients = set()

    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = Path(features_dir) / class_name
        if not class_dir.exists():
            continue

        npy_files = sorted(class_dir.glob("*.npy"))
        for npy_path in npy_files:
            # Format nama file: {patient_id}_mel.npy atau {patient_id}_mel_noise.npy
            patient_id = npy_path.stem.split('_')[0]
            if patient_id in patient_to_label:
                all_files.append((npy_path, patient_id, class_idx))
                available_patients.add(patient_id)
            else:
                logging.warning(f"File {npy_path.name} dilewati karena label pasien tidak ditemukan di CSV.")

    if not all_files:
        raise RuntimeError(f"Tidak ada file spectrogram valid ditemukan di {features_dir}.")

    logging.info(f"Total pasien unik ditemukan: {len(available_patients)}")

    # 3. Urutkan pasien untuk pembagian deterministik dan stratification
    patients_list = sorted(list(available_patients))
    patients_labels = [CLASS_TO_IDX[patient_to_label[pid]] for pid in patients_list]

    # 4. Stratified Split pada tingkat pasien
    # Step A: Pisahkan test set terlebih dahulu
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    trainval_idx, test_idx = next(sss_test.split(patients_list, patients_labels))

    # Step B: Dari trainval, pisahkan val set
    val_size_adjusted = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_adjusted, random_state=random_state
    )
    train_idx, val_idx = next(
        sss_val.split(trainval_idx, [patients_labels[i] for i in trainval_idx])
    )
    # Kembalikan ke index global
    train_idx = trainval_idx[train_idx]
    val_idx   = trainval_idx[val_idx]

    # Kelompokkan patient IDs untuk masing-masing split
    train_patients = {patients_list[i] for i in train_idx}
    val_patients = {patients_list[i] for i in val_idx}
    test_patients = {patients_list[i] for i in test_idx}

    # 5. Distribusikan file ke masing-masing split
    train_samples = []
    val_samples = []
    test_samples = []

    for npy_path, pid, class_idx in all_files:
        # File original: {pid}_mel.npy
        # File augmented: {pid}_mel_noise.npy, {pid}_mel_pitch.npy, dll
        filename = npy_path.stem
        is_original = filename.endswith('_mel') and not any(
            aug in filename for aug in ['_noise', '_pitch', '_stretch']
        )

        if pid in train_patients:
            # Train set mendapatkan data asli DAN semua hasil augmentasi
            train_samples.append((npy_path, class_idx))
        elif pid in val_patients:
            # Val set HANYA mendapatkan data original (mencegah bias)
            if is_original:
                val_samples.append((npy_path, class_idx))
        elif pid in test_patients:
            # Test set HANYA mendapatkan data original (mencegah bias)
            if is_original:
                test_samples.append((npy_path, class_idx))

    # 6. Buat objek dataset untuk masing-masing split
    train_dataset = MelSpectrogramDataset(samples=train_samples, max_len=max_len)
    val_dataset   = MelSpectrogramDataset(samples=val_samples, max_len=max_len)
    test_dataset  = MelSpectrogramDataset(samples=test_samples, max_len=max_len)

    # Log distribusi setiap split
    _log_split_distribution("Train", np.array(train_dataset.labels))
    _log_split_distribution("Val",   np.array(val_dataset.labels))
    _log_split_distribution("Test",  np.array(test_dataset.labels))

    # 7. Siapkan PyTorch DataLoaders
    if use_weighted_sampler:
        logging.info("Mengaktifkan WeightedRandomSampler untuk train loader.")
        train_labels_tensor = torch.tensor(train_dataset.labels, dtype=torch.long)
        class_counts = torch.bincount(train_labels_tensor, minlength=len(CLASS_NAMES)).float().clamp(min=1.0)
        # Stronger balancing: each class has equal total probability in the sampler.
        sample_weights = 1.0 / class_counts[train_labels_tensor]
        sample_weights = sample_weights / sample_weights.sum()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False  # PENTING: dataset kecil, jangan buang batch terakhir
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False  # PENTING: dataset kecil, jangan buang batch terakhir
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    logging.info(
        f"DataLoader siap — Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def _log_split_distribution(split_name, split_labels):
    from collections import Counter
    counts = Counter(split_labels.tolist())
    dist   = {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}
    logging.info(f"Distribusi {split_name:5s}: {dist}")


# ==========================================
# QUICK TEST
# ==========================================

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, max_len=800)

    # Cek satu batch
    for specs, labels in train_loader:
        print(f"Batch spec shape : {specs.shape}")   # (16, 1, 128, 800)
        print(f"Batch label shape: {labels.shape}")  # (16,)
        print(f"Label sample     : {labels[:8].tolist()}")
        break