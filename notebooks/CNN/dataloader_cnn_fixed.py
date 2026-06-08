# Standalone fixed dataloader extracted from preprocessing/cnn_dataloader_fixed.py

import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent
# NOTE: file ini ada di notebooks/CNN/, jadi project root adalah 2 level di atas
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "spectrogram"

CLASS_NAMES = ["NORMAL", "DEPRESI"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MelSpectrogramDataset(Dataset):
    """Read mel spectrogram .npy and return (1, n_mels, max_len) + label."""

    def __init__(self, features_dir=FEATURES_DIR, max_len=800, samples=None):
        self.max_len = max_len

        if samples is not None:
            self.samples = samples
            self.labels = [s[1] for s in samples]
        else:
            self.samples = []
            self.labels = []
            for class_name, class_idx in CLASS_TO_IDX.items():
                class_dir = Path(features_dir) / class_name
                if not class_dir.exists():
                    logging.warning(f"Folder kelas tidak ditemukan: {class_dir}")
                    continue

                for npy_path in sorted(class_dir.glob("*.npy")):
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
        dist = {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}
        logging.info(f"Distribusi dataset penuh: {dist}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        spec = np.load(npy_path)

        if spec.shape[1] > self.max_len:
            spec = spec[:, : self.max_len]
        elif spec.shape[1] < self.max_len:
            pad_width = self.max_len - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")

        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return spec_tensor, label_tensor


def _log_split_distribution(split_name, split_labels):
    from collections import Counter

    counts = Counter(split_labels.tolist())
    dist = {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}
    logging.info(f"Distribusi {split_name:5s}: {dist}")


def get_dataloaders(
    batch_size=16,
    max_len=800,
    val_size=0.15,
    test_size=0.10,
    random_state=42,
    features_dir=FEATURES_DIR,
    use_weighted_sampler=False,
):
    import pandas as pd

    label_path = PROJECT_ROOT / "data" / "splits" / "custom_2class_labels.csv"
    fallback_path = PROJECT_ROOT / "data" / "splits" / "custom_2class_labels_broken_out.csv"

    if not label_path.exists():
        if fallback_path.exists():
            label_path = fallback_path

    if not label_path.exists():
        raise FileNotFoundError(
            "File label tidak ditemukan. "
            f"Coba cek: {PROJECT_ROOT / 'data' / 'splits' / 'custom_2class_labels.csv'} "
            f"atau {fallback_path}"
        )


    df = pd.read_csv(label_path)

    id_col_candidates = ["Participant_ID", "participant_id"]
    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    if id_col is None:
        raise KeyError(f"custom_2class_labels.csv tidak punya kolom id yang dikenal. Kolom ada: {list(df.columns)}")

    if "label_depresi" not in df.columns:
        if "Custom_Label" in df.columns:
            patient_to_label = dict(zip(df[id_col].astype(str), df["Custom_Label"].astype(str)))
        else:
            raise KeyError(
                "custom_2class_labels.csv harus punya kolom 'label_depresi' atau 'Custom_Label'. "
                f"Kolom ada: {list(df.columns)}"
            )
    else:
        patient_to_label = dict(
            zip(
                df[id_col].astype(str),
                df["label_depresi"].astype(int).map({0: "NORMAL", 1: "DEPRESI"}),
            )
        )

    all_files = []
    available_patients = set()

    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = Path(features_dir) / class_name
        if not class_dir.exists():
            continue

        for npy_path in sorted(class_dir.glob("*.npy")):
            pid = npy_path.stem.split("_")[0]
            if pid in patient_to_label:
                all_files.append((npy_path, pid, class_idx))
                available_patients.add(pid)

    if not all_files:
        raise RuntimeError(f"Tidak ada file spectrogram valid ditemukan di {features_dir}.")

    patients_list = sorted(list(available_patients))
    patients_labels = [CLASS_TO_IDX[patient_to_label[pid]] for pid in patients_list]

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(sss_test.split(patients_list, patients_labels))

    val_size_adjusted = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    train_idx, val_idx = next(sss_val.split(trainval_idx, [patients_labels[i] for i in trainval_idx]))

    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    train_patients = {patients_list[i] for i in train_idx}
    val_patients = {patients_list[i] for i in val_idx}
    test_patients = {patients_list[i] for i in test_idx}

    train_samples, val_samples, test_samples = [], [], []

    for npy_path, pid, class_idx in all_files:
        filename = npy_path.stem
        # Original: hanya file yang berakhiran _mel tanpa suffix augmentasi
        is_original = (
            filename.endswith("_mel")
            and not any(aug in filename for aug in ["_noise", "_pitch", "_stretch", "_combo"])
        )

        if pid in train_patients:
            # Train: semua file (original + augmentasi)
            train_samples.append((npy_path, class_idx))
        elif pid in val_patients:
            # Val: hanya original — evaluasi objektif
            if is_original:
                val_samples.append((npy_path, class_idx))
        elif pid in test_patients:
            # Test: hanya original — evaluasi objektif
            if is_original:
                test_samples.append((npy_path, class_idx))

    train_dataset = MelSpectrogramDataset(samples=train_samples, max_len=max_len)
    val_dataset = MelSpectrogramDataset(samples=val_samples, max_len=max_len)
    test_dataset = MelSpectrogramDataset(samples=test_samples, max_len=max_len)

    _log_split_distribution("Train", np.array(train_dataset.labels))
    _log_split_distribution("Val", np.array(val_dataset.labels))
    _log_split_distribution("Test", np.array(test_dataset.labels))

    if use_weighted_sampler:
        train_labels_tensor = torch.tensor(train_dataset.labels, dtype=torch.long)
        class_counts = torch.bincount(train_labels_tensor, minlength=len(CLASS_NAMES)).float().clamp(min=1.0)
        sample_weights = 1.0 / class_counts[train_labels_tensor]
        sample_weights = sample_weights / sample_weights.sum()

        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    logging.info(
        f"DataLoader siap — Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader


def _print_one_batch(loader, split_name="Train"):
    import numpy as _np

    for specs, labels in loader:
        print(f"{split_name} batch specs shape : {specs.shape}")
        print(f"{split_name} batch labels shape: {labels.shape}")
        print(f"{split_name} label sample      : {labels[:8].cpu().numpy().tolist()}")
        return


if __name__ == "__main__":
    # Run sederhana untuk memastikan dataloader benar-benar memuat data
    print("Running dataloader_cnn_fixed.py ...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, max_len=800)

    logging.info("Ambil 1 batch untuk verifikasi tensor")
    _print_one_batch(train_loader, "Train")
    _print_one_batch(val_loader, "Val")
    _print_one_batch(test_loader, "Test")


