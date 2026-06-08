"""
Script untuk membuat ZIP file yang siap diupload ke Google Colab.

Isi ZIP (menthealth_colab.zip):
  data/features/spectrogram/NORMAL/   -> semua .npy NORMAL  (~868 files)
  data/features/spectrogram/DEPRESI/  -> semua .npy DEPRESI (~845 files)
  data/splits/custom_2class_labels.csv
  notebooks/CNN/dataloader_cnn_fixed.py
  notebooks/CNN/train_2d_cnn_colab.ipynb

Jalankan:
  python prepare_colab_zip.py
"""

import zipfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP   = PROJECT_ROOT / "menthealth_colab.zip"

# File & folder yang akan dimasukkan ke ZIP
ITEMS = [
    # (path_di_disk, path_di_dalam_zip)
    (
        PROJECT_ROOT / "data" / "features" / "spectrogram" / "NORMAL",
        "data/features/spectrogram/NORMAL",
    ),
    (
        PROJECT_ROOT / "data" / "features" / "spectrogram" / "DEPRESI",
        "data/features/spectrogram/DEPRESI",
    ),
    (
        PROJECT_ROOT / "data" / "splits" / "custom_2class_labels.csv",
        "data/splits/custom_2class_labels.csv",
    ),
    (
        PROJECT_ROOT / "notebooks" / "CNN" / "dataloader_cnn_fixed.py",
        "notebooks/CNN/dataloader_cnn_fixed.py",
    ),
    (
        PROJECT_ROOT / "notebooks" / "CNN" / "train_2d_cnn_colab.ipynb",
        "notebooks/CNN/train_2d_cnn_colab.ipynb",
    ),
]


def make_zip():
    print(f"Membuat ZIP: {OUTPUT_ZIP}")
    total_files = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc_name in ITEMS:
            src = Path(src)

            if not src.exists():
                print(f"  [SKIP] Tidak ditemukan: {src}")
                continue

            if src.is_dir():
                files = sorted(src.glob("*.npy"))
                print(f"  [DIR ] {src.name}/ → {len(files)} files")
                for f in files:
                    zf.write(f, f"{arc_name}/{f.name}")
                    total_files += 1
            else:
                print(f"  [FILE] {src.name}")
                zf.write(src, arc_name)
                total_files += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n✅ ZIP selesai!")
    print(f"   File  : {OUTPUT_ZIP.name}")
    print(f"   Total : {total_files} files")
    print(f"   Ukuran: {size_mb:.1f} MB")
    print(f"\nLangkah selanjutnya:")
    print(f"  1. Upload '{OUTPUT_ZIP.name}' ke Google Colab")
    print(f"  2. Buka 'notebooks/CNN/train_2d_cnn_colab.ipynb' di Colab")
    print(f"  3. Runtime → Change runtime type → T4 GPU")
    print(f"  4. Jalankan semua cell dari atas ke bawah")


if __name__ == "__main__":
    make_zip()
