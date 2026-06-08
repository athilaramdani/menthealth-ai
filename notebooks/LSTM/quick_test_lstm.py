"""
LSTM Step 3 – Quick Sanity Check
==================================
Project : Mental Health Audio Classification
Script  : notebooks/LSTM/quick_test_lstm.py

Jalankan ini SEBELUM training untuk memverifikasi bahwa:
  1. File .npy MFCC sudah ada dan bisa dibaca.
  2. Shape tensor sesuai (MAX_T, N_FEATURES).
  3. Model BiLSTM bisa forward pass tanpa error.
  4. DataLoader berjalan tanpa crash.
  5. Distribusi kelas terlihat wajar.

Usage:
    python quick_test_lstm.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import Counter

# Paksa stdout encoding utf-8 di Windows agar karakter non-ASCII aman
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

MFCC_DIR = PROJECT_ROOT / "data" / "features" / "mfcc"


def check_mfcc_files():
    print("\n" + "="*60)
    print("1. CEK FILE .NPY MFCC")
    print("="*60)

    total = 0
    shapes = []
    for cls in ["NORMAL", "DEPRESI"]:
        cls_dir = MFCC_DIR / cls
        if not cls_dir.exists():
            print(f"  ❌ Folder tidak ada: {cls_dir}")
            continue
        files = sorted(cls_dir.glob("*.npy"))
        print(f"  {cls}: {len(files)} file .npy")
        total += len(files)
        # Sample 5 files untuk cek shape
        for f in files[:5]:
            arr = np.load(str(f))
            shapes.append(arr.shape)
            print(f"     {f.name}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}")

    print(f"\n  Total segmen: {total}")

    if shapes:
        t_vals = [s[0] for s in shapes]
        f_vals = [s[1] for s in shapes]
        print(f"\n  Timestep range: {min(t_vals)} – {max(t_vals)}")
        print(f"  Features      : {set(f_vals)}")
        print(f"  ✅ Cek shapes selesai")
    else:
        print("  ❌ Tidak ada file .npy! Jalankan extract_mfcc_segments.py dulu.")
        return False
    return True


def check_dataloader():
    print("\n" + "="*60)
    print("2. CEK DATALOADER")
    print("="*60)

    try:
        from train_bilstm_mfcc import (
            load_patient_data, MFCCSegmentDataset,
            make_weighted_loader, MAX_T, N_FEATURES, CLASS_NAMES
        )

        patients_list, patients_labels, all_files, _ = load_patient_data()
        print(f"  Total patient: {len(patients_list)}")
        print(f"  Total segmen : {len(all_files)}")
        dist = Counter(patients_labels)
        print(f"  Distribusi patient: NORMAL={dist[0]}, DEPRESI={dist[1]}")

        # Coba buat loader dari 50 sampel pertama
        samples_test = [(p, c) for p, pid, c in all_files[:50]]
        loader = make_weighted_loader(samples_test, batch_size=8, shuffle=False)
        feats, labels = next(iter(loader))
        print(f"\n  Batch feats  : {feats.shape}   (expected: [8, {MAX_T}, {N_FEATURES}])")
        print(f"  Batch labels : {labels.shape}  values={labels.tolist()}")
        print(f"  Feats dtype  : {feats.dtype}")
        print(f"  ✅ DataLoader OK")
        return True
    except Exception as e:
        print(f"  ❌ DataLoader error: {e}")
        import traceback; traceback.print_exc()
        return False


def check_model():
    print("\n" + "="*60)
    print("3. CEK MODEL FORWARD PASS")
    print("="*60)

    try:
        from train_bilstm_mfcc import BiLSTMMFCC, MAX_T, N_FEATURES, HIDDEN_DIM, NUM_LAYERS

        model = BiLSTMMFCC()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model       : BiLSTMMFCC")
        print(f"  Parameters  : {n_params:,}")
        print(f"  Architecture: Input({N_FEATURES}) → BiLSTM({HIDDEN_DIM}×{NUM_LAYERS}) → Attn → FC → 2")

        # Dummy forward pass
        dummy = torch.randn(4, MAX_T, N_FEATURES)
        out   = model(dummy)
        print(f"\n  Input shape : {dummy.shape}")
        print(f"  Output shape: {out.shape}  (expected: [4, 2])")

        # Cek attention weights
        attn = model.get_attention_weights(dummy)
        print(f"  Attn shape  : {attn.shape}  (expected: [4, {MAX_T}])")
        print(f"  Attn sum    : {attn.sum(dim=1).mean().item():.4f}  (harus ~1.0 per sampel)")
        print(f"  ✅ Model forward pass OK")
        return True
    except Exception as e:
        print(f"  ❌ Model error: {e}")
        import traceback; traceback.print_exc()
        return False


def check_gpu():
    print("\n" + "="*60)
    print("4. CEK GPU / DEVICE")
    print("="*60)

    if torch.cuda.is_available():
        print(f"  ✅ CUDA tersedia: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️  CUDA tidak tersedia — training akan berjalan di CPU (lebih lambat)")
        print("     Gunakan Google Colab / server GPU untuk training yang lebih cepat.")


def estimate_segments():
    print("\n" + "="*60)
    print("5. ESTIMASI JUMLAH SEGMEN (setelah ekstraksi)")
    print("="*60)

    cleaned_dir = PROJECT_ROOT / "data" / "cleaned"
    wav_files   = list(cleaned_dir.glob("*.wav"))
    print(f"  Total cleaned WAV: {len(wav_files)}")

    # Estimasi kasar berdasarkan ukuran file
    # 16kHz mono PCM16 = 32 KB/detik → 1 MB ≈ 31 detik
    total_estimated_seg = 0
    for f in wav_files:
        size_mb   = f.stat().st_size / 1e6
        duration_s = size_mb * 31      # estimasi kasar
        n_segs    = max(0, int(duration_s // 10))  # 10-detik segmen
        total_estimated_seg += n_segs

    print(f"  Estimasi total segmen (~10 detik): {total_estimated_seg}")
    print(f"  vs sebelumnya (1 file = 1 sampel): {len(wav_files)}")
    print(f"  Multiplikasi data: ~{total_estimated_seg/max(1,len(wav_files)):.0f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("BiLSTM MFCC — QUICK SANITY CHECK")
    print("=" * 60)

    check_gpu()
    estimate_segments()
    ok1 = check_mfcc_files()
    if ok1:
        ok2 = check_dataloader()
        ok3 = check_model()

        print("\n" + "=" * 60)
        if ok1 and ok2 and ok3:
            print("✅ SEMUA CEK PASSED — Siap menjalankan training!")
            print("\nLangkah selanjutnya:")
            print("   python train_bilstm_mfcc.py")
        else:
            print("❌ Ada cek yang gagal — perbaiki sebelum training.")
        print("=" * 60)
    else:
        print("\n⚠️  Jalankan dulu: python extract_mfcc_segments.py")
