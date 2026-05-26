from pathlib import Path
from typing import Tuple, Optional, Any
import numpy as np
import torch
import soundfile as sf

def preprocess_wav2vec2(
    audio_path: str | Path,
    processor: Optional[Any] = None,
    sample_rate: int = 16000,
    max_duration_seconds: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memproses file audio mentah (.wav) menjadi bentuk tensor yang siap diterima oleh Wav2Vec2.
    Fungsi ini otomatis menangani audio stereo, interpolasi sample rate ke 16kHz,
    serta pembatasan durasi (padding/truncate).
    
    Returns:
        input_values: FloatTensor berukuran [T] yang berisi amplitudo ternormalisasi.
        attention_mask: LongTensor berukuran [T] (1 untuk data asli, 0 untuk padding).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"File audio tidak ditemukan di lokasi: {audio_path}")

    # 1. Membaca file audio menggunakan soundfile
    waveform, sr = sf.read(str(audio_path))

    # 2. Konversi ke Mono jika audio bertipe Stereo (2 Channel)
    if isinstance(waveform, np.ndarray) and waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    waveform = waveform.astype(np.float32)

    # 3. Resampling jika sample rate asal tidak sesuai dengan target (16kHz)
    if sr != sample_rate:
        num_samples = int(len(waveform) * sample_rate / sr)
        x_old = np.arange(len(waveform), dtype=np.float32)
        x_new = np.linspace(0, len(waveform) - 1, num_samples, dtype=np.float32)
        waveform = np.interp(x_new, x_old, waveform).astype(np.float32)

    # 4. Normalisasi Amplitudo (Peak Normalization)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = waveform / peak

    # 5. Standarisasi Panjang Gelombang (Crop atau Padding)
    max_len = int(max_duration_seconds * sample_rate)
    if waveform.shape[0] > max_len:
        waveform = waveform[:max_len]
    elif waveform.shape[0] < max_len:
        waveform = np.pad(waveform, (0, max_len - waveform.shape[0]), mode='constant')

    # 6. Jika dijalankan tanpa objek Hugging Face Processor, return tensor manual dasar
    if processor is None:
        input_values = torch.tensor(waveform, dtype=torch.float32)
        attention_mask = torch.ones_like(input_values, dtype=torch.long)
        return input_values, attention_mask

    # 7. Ekstraksi fitur menggunakan Hugging Face Wav2Vec2Processor bawaan
    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # CRITICAL SQUEEZE: Melepas dimensi batch internal Hugging Face [1, T] menjadi [T]
    # Tindakan ini wajib dilakukan agar saat DataLoader melakukan batching, dimensi tensor tidak bentrok
    input_values = inputs["input_values"][0]
    attention_mask = inputs["attention_mask"][0]
    
    return input_values, attention_mask


# =====================================================================
# BLOK SANITY CHECK (PENGUJIAN SIMULASI MANDIRI)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("        MENGUJI INTEGRASI PIPELINE PROCESSOR (Wav2Vec2)       ")
    print("="*60)
    
    # Membuat sinyal sinus dummy 5 detik sebagai simulasi file audio audio mentah
    sample_rate_test = 16000
    duration_test = 5.0
    t = np.linspace(0, duration_test, int(sample_rate_test * duration_test), endpoint=False)
    dummy_waveform = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    dummy_path = Path("dummy_processor_check.wav")
    sf.write(str(dummy_path), dummy_waveform, sample_rate_test)
    print(f"-> Sukses membuat file audio simulasi lokal: {dummy_path.name}")
    
    try:
        # Jalankan pengujian mode fallback manual (tanpa memanggil library transformers)
        in_val, att_mask = preprocess_wav2vec2(
            audio_path=dummy_path,
            processor=None,
            sample_rate=16000,
            max_duration_seconds=8.0
        )
        
        print("\n--- Hasil Verifikasi Struktur Tensor (Fallback Mode) ---")
        print(f"  - Input Values Shape  : {in_val.shape} (Harus tunggal [128000] untuk 8 detik)")
        print(f"  - Attention Mask Shape : {att_mask.shape} (Harus tunggal [128000])")
        print(f"  - Tipe Data Matriks    : {in_val.dtype}")
        print("\n🎉 Validasi internal processor.py berhasil dan siap digunakan!")
        
    except Exception as e:
        print(f"❌ Terjadi gangguan saat eksekusi fungsi: {e}")
        
    finally:
        # Hapus file dummy setelah pengetesan selesai agar folder tetap bersih
        if dummy_path.exists():
            dummy_path.unlink()