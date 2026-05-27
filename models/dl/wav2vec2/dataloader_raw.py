import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import soundfile as sf

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Struktur 4 Kelas Utama (Sesuai kesepakatan Master Pipeline)
CLASS_NAMES = ["NORMAL", "STRES", "CEMAS", "DEPRESI"]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

@dataclass
class RawAudioSample:
    audio_path: Path
    label_idx: int

class Wav2Vec2RawDataset(Dataset):
    """
    Dataset untuk memuat waveform mentah dari DAIC-WOZ.
    Memetakan Participant_ID ke file audio dan mengonversinya ke bentuk tensor 4 kelas.
    """
    def __init__(
        self, 
        raw_root: Path, 
        label_csv: Path, 
        max_duration_seconds: float = 8.0, 
        sample_rate: int = 16000,
        limit_samples: Optional[int] = None,
    ):
        super().__init__()
        self.raw_root = Path(raw_root)
        self.label_csv = Path(label_csv)
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate

        if not self.raw_root.exists():
            raise FileNotFoundError(f"raw_root tidak ditemukan: {self.raw_root}")
        if not self.label_csv.exists():
            raise FileNotFoundError(f"label_csv tidak ditemukan: {self.label_csv}")

        # Fix KeyError: Menggunakan Custom_Label sesuai struktur CSV kamu
        self.participant_to_label = self._load_labels(self.label_csv)
        self.samples: List[RawAudioSample] = self._collect_samples(self.raw_root)

        if limit_samples is not None:
            self.samples = self.samples[: int(limit_samples)]

        logging.info(f"[Wav2Vec2RawDataset] Total sampel valid ditemukan: {len(self.samples)}")
        self._log_distribution()

    def _log_distribution(self):
        """Menampilkan distribusi jumlah data per kelas di terminal."""
        from collections import Counter
        counts = Counter([s.label_idx for s in self.samples])
        dist = {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}
        logging.info(f"[Wav2Vec2RawDataset] Distribusi dataset saat ini: {dist}")

    @staticmethod
    def _load_labels(label_csv: Path) -> Dict[str, str]:
        df = pd.read_csv(label_csv)

        # Menentukan kolom ID Partisipan
        if "Participant_ID" in df.columns:
            id_col = "Participant_ID"
        else:
            id_col = df.columns[0]

        # Proteksi pengecekan nama kolom label
        if "Custom_Label" not in df.columns:
            raise ValueError("Kolom 'Custom_Label' tidak ditemukan di custom_4class_labels.csv!")

        # Mengonversi string label menjadi huruf besar (Uppercase) agar tidak sensitif huruf besar/kecil
        mapping = dict(zip(df[id_col].astype(str), df["Custom_Label"].str.upper()))
        return mapping

    def _heuristic_participant_id_from_path(self, audio_path: Path) -> Optional[str]:
        # Pola utama: Mengambil ID dari nama folder parent (contoh: '300_P' -> '300')
        parent = audio_path.parent.name
        if parent:
            digits = "".join(ch for ch in parent if ch.isdigit())
            if digits:
                return digits

        # Fallback: Mengambil ID dari nama depan file audio
        name = audio_path.stem
        digits = "".join(ch for ch in name if ch.isdigit())
        if digits:
            return digits

        return None

    def _collect_samples(self, raw_root: Path) -> List[RawAudioSample]:
        # Mencari seluruh file berakhiran _AUDIO.wav secara rekursif
        wav_paths = sorted(raw_root.rglob("*_AUDIO.wav"))
        if len(wav_paths) == 0:
            raise RuntimeError(f"Tidak menemukan file audio '*_AUDIO.wav' di direktori {raw_root}")

        samples: List[RawAudioSample] = []
        for wav_path in wav_paths:
            pid = self._heuristic_participant_id_from_path(wav_path)
            if pid is None:
                continue
            if pid not in self.participant_to_label:
                logging.warning(f"ID Partisipan {pid} tidak terdaftar di file CSV label. File dilewati: {wav_path.name}")
                continue
            
            label_name = self.participant_to_label[pid]
            if label_name not in CLASS_TO_IDX:
                logging.warning(f"Label '{label_name}' pada ID {pid} tidak dikenali dalam kategori 4 kelas.")
                continue

            label_idx = CLASS_TO_IDX[label_name]
            samples.append(RawAudioSample(audio_path=wav_path, label_idx=label_idx))

        return sorted(samples, key=lambda s: str(s.audio_path))

    def _resample_if_needed(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.sample_rate:
            return waveform

        waveform_t = torch.tensor(waveform, dtype=torch.float32)
        n = waveform_t.numel()
        new_n = int(n * (self.sample_rate / orig_sr))
        idx = torch.linspace(0, n - 1, new_n)
        waveform_resampled = torch.interp(idx, torch.arange(n, dtype=torch.float32), waveform_t)
        return waveform_resampled.numpy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        audio_path = sample.audio_path
        
        waveform, sr = sf.read(str(audio_path))

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        waveform = self._resample_if_needed(waveform, sr)

        # Standarisasi panjang audio (Padding / Truncate)
        max_len = int(self.max_duration_seconds * self.sample_rate)
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]
        elif waveform.shape[0] < max_len:
            pad_width = max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(torch.tensor(waveform, dtype=torch.float32), (0, pad_width)).numpy()

        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "label": torch.tensor(sample.label_idx, dtype=torch.long),
            "audio_path": str(audio_path),
        }

# =====================================================================
# BLOK JALANKAN OTOMATIS (SANITY CHECK)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("      MENGUJI INTEGRASI PIPELINE DATALOADER RAW (4 KELAS)     ")
    print("="*60)
    
    PATH_RAW = Path(r"C:\Users\raiha\OneDrive\Documents\GitHub\menthealth-ai\data\raw").resolve()
    PATH_CSV = Path(r"C:\Users\raiha\OneDrive\Documents\GitHub\menthealth-ai\data\splits\custom_4class_labels.csv").resolve()
    
    try:
        # Load dataset secara penuh tanpa limit untuk memvalidasi distribusi target kelas
        ds = Wav2Vec2RawDataset(raw_root=PATH_RAW, label_csv=PATH_CSV)
        print(f"\n🎉 Sukses! Total file yang terhubung: {len(ds)} sampel.\n")
        
        if len(ds) > 0:
            print("--- Menampilkan 3 Sampel Data Teratas ---")
            for i in range(min(3, len(ds))):
                item = ds[i]
                print(f"Sampel ke-{i}:")
                print(f"  - Path Audio : {Path(item['audio_path']).name}")
                print(f"  - Shape      : {item['waveform'].shape}")
                print(f"  - Label Indeks: {item['label'].item()} -> Kategori: {CLASS_NAMES[item['label'].item()]}")
                print("-" * 50)
    except Exception as e:
        print(f"❌ Sanity check gagal karena eror: {e}")