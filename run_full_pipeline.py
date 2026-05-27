import subprocess
import logging
import sys
import time
from pathlib import Path

# ==========================================
# PATH SETUP
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
CNN_DIR           = PROJECT_ROOT / "models" / "dl" / "cnn"

# Script paths
GENERATE_SPEC = PREPROCESSING_DIR / "generate_spectrogram.py"
AUGMENT       = PREPROCESSING_DIR / "data_augmentation.py"
TRAIN_CNN     = CNN_DIR           / "train_2d_cnn.py"

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "run_full_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ==========================================
# RUNNER
# ==========================================

def run_step(label: str, script: Path, cwd: Path = PROJECT_ROOT) -> bool:
    """
    Jalankan satu script Python dan return True jika berhasil.
    Streaming stdout/stderr ke log secara real-time.
    """
    if not script.exists():
        logging.error(f"[{label}] Script tidak ditemukan: {script}")
        return False

    logging.info(f"{'='*60}")
    logging.info(f"[{label}] Memulai: {script}")
    logging.info(f"{'='*60}")

    start = time.time()
    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # gabungkan stderr ke stdout agar urutan log terjaga
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # Stream output baris per baris agar progress tqdm terlihat di log
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logging.info(f"  [{label}] {line}")

    process.wait()
    elapsed = time.time() - start

    if process.returncode != 0:
        logging.error(f"[{label}] GAGAL (exit code {process.returncode}) — {elapsed:.1f}s")
        return False

    logging.info(f"[{label}] SELESAI — {elapsed:.1f}s")
    return True


# ==========================================
# PIPELINE STEPS
# ==========================================

STEPS = [
    # (label, script, wajib_sukses)
    ("1/3 Generate Spectrogram", GENERATE_SPEC, True),
    ("2/3 Data Augmentation",    AUGMENT,       True),
    ("3/3 Train 2D CNN",         TRAIN_CNN,     True),
    # Tambahkan quick test setelah training (optional, tidak wajib sukses)
    # ("4/4 Quick Test", CNN_DIR / "quick_test.py", False),
]

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    logging.info("Pipeline dimulai")
    logging.info(f"Project root : {PROJECT_ROOT}")
    logging.info(f"Python       : {sys.executable}")

    pipeline_start = time.time()
    failed_steps   = []

    for label, script, must_succeed in STEPS:
        ok = run_step(label, script)
        if not ok:
            failed_steps.append(label)
            if must_succeed:
                logging.error(
                    f"Step '{label}' gagal dan wajib sukses. "
                    f"Pipeline dihentikan."
                )
                break

    # --- Ringkasan ---
    total_elapsed = time.time() - pipeline_start
    logging.info("=" * 60)
    if not failed_steps:
        logging.info(f"PIPELINE SELESAI TANPA ERROR — {total_elapsed:.1f}s")
        logging.info("Output ada di: results/metrics/, results/plots/, results/confusion_matrix/")
    else:
        logging.error(f"PIPELINE SELESAI DENGAN ERROR — {total_elapsed:.1f}s")
        logging.error(f"Step yang gagal: {failed_steps}")
        sys.exit(1)