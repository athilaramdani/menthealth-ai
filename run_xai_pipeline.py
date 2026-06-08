#!/usr/bin/env python
"""
Unified XAI Runner Pipeline
==========================
Project : Mental Health Audio Classification (NORMAL vs DEPRESI)
Script  : run_xai_pipeline.py

Skrip ini menyatukan dan menjalankan seluruh pipeline Explainable AI (XAI)
untuk model-model Deep Learning (CNN, BiLSTM, Wav2Vec 2.0).

Cara menjalankan:
  python run_xai_pipeline.py
  python run_xai_pipeline.py --n_samples 5 --device cuda
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Color codes for premium output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner():
    banner = f"""
{BLUE}{BOLD}==================================================================
        MENTAL HEALTH AUDIO CLASSIFICATION — XAI RUNNER
=================================================================={RESET}
Proyek  : Klasifikasi NORMAL vs DEPRESI Berbasis Audio
Fase 4  : Implementasi Explainable AI (XAI) untuk Deep Learning
Laporan : docs/xai_workflow_expectations.md
{BLUE}=================================================================={RESET}
"""
    print(banner)

def run_script(script_path: Path, args_list: list):
    """Menjalankan script python eksternal secara robust."""
    script_name = script_path.name
    print(f"\n{YELLOW}{BOLD}[RUNNING]{RESET} Menjalankan {BOLD}{script_name}{RESET}...")
    start_time = time.time()
    
    cmd = [sys.executable, str(script_path)] + args_list
    
    try:
        # Menjalankan subprocess dan meneruskan output secara langsung ke stdout/stderr
        result = subprocess.run(cmd, check=True, text=True)
        elapsed = time.time() - start_time
        print(f"{GREEN}{BOLD}[SUCCESS]{RESET} {script_name} selesai dalam {elapsed:.2f} detik.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}{BOLD}[ERROR]{RESET} {script_name} gagal dengan exit code {e.returncode}.")
        return False
    except Exception as e:
        print(f"{RED}{BOLD}[ERROR]{RESET} Gagal menjalankan {script_name}: {e}")
        return False

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Unified Explainable AI (XAI) Pipeline Runner"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "cnn", "bilstm", "wav2vec"],
        help="Model XAI mana yang ingin dijalankan (default: all)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="Jumlah sampel audio untuk setiap analisis XAI (default: 4 = 2 NORMAL + 2 DEPRESI, minimum: 2)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device komputasi (default: cpu)",
    )
    args = parser.parse_args()
    
    # Validasi jumlah sampel
    if args.n_samples < 2:
        print(f"{RED}{BOLD}[ERROR]{RESET} Jumlah sampel (--n_samples) harus minimal 2 untuk memastikan representasi kedua kelas.")
        print(f"        Anda meminta: {args.n_samples} sampel")
        print(f"        Gunakan --n_samples 2 atau lebih (default: 4)")
        sys.exit(1)
    
    # Peringatan jika jumlah sampel ganjil
    if args.n_samples % 2 != 0:
        print(f"{YELLOW}{BOLD}[WARNING]{RESET} Jumlah sampel ganjil ({args.n_samples}). Sampel akan dibagi tidak merata antar kelas.")
        print(f"          Disarankan gunakan jumlah genap untuk pembagian seimbang (contoh: 4, 6, 8)")
        import time
        time.sleep(2)
    
    # Validasi path script
    cnn_gradcam_path  = PROJECT_ROOT / "notebooks" / "CNN" / "xai_gradcam_cnn.py"
    shap_dl_path      = PROJECT_ROOT / "notebooks" / "xai_shap_dl.py"
    wav2vec_attn_path = PROJECT_ROOT / "notebooks" / "WAV2VEC" / "xai_attention_wav2vec.py"

    # Path model eksplisit
    CNN_MODEL_PATH    = PROJECT_ROOT / "models" / "dl" / "cnn"  / "best_model_1;1.pt"
    LSTM_MODEL_PATH   = PROJECT_ROOT / "models" / "dl" / "lstm" / "best_bilstm_mfcc.pt"
    WAV2VEC_MODEL_PATH= PROJECT_ROOT / "models" / "dl" / "wav2vec" / "best_model_v2.pt"

    # Validasi ketersediaan model sebelum menjalankan
    model_check = {
        "CNN"    : (CNN_MODEL_PATH,     args.model in ("all", "cnn")),
        "BiLSTM" : (LSTM_MODEL_PATH,    args.model in ("all", "bilstm")),
        "Wav2Vec": (WAV2VEC_MODEL_PATH, args.model in ("all", "wav2vec")),
    }
    missing = [name for name, (path, needed) in model_check.items() if needed and not path.exists()]
    if missing:
        for name in missing:
            path = model_check[name][0]
            print(f"{RED}{BOLD}[ERROR]{RESET} Model {name} tidak ditemukan: {path}")
        sys.exit(1)
    
    tasks = []
    
    # 1. Grad-CAM (CNN)
    if args.model in ("all", "cnn"):
        tasks.append((cnn_gradcam_path, [
            "--n_samples",  str(args.n_samples),
            "--model_path", str(CNN_MODEL_PATH),
        ]))
        
    # 2. SHAP (CNN & BiLSTM) — dijalankan terpisah untuk isolasi lebih baik
    if args.model in ("all", "cnn", "bilstm"):
        shap_base = ["--n_samples", str(args.n_samples), "--device", args.device]
        if args.model in ("all", "cnn"):
            tasks.append((shap_dl_path, shap_base + [
                "--model", "cnn",
                "--cnn_model_path", str(CNN_MODEL_PATH),
            ]))
        if args.model in ("all", "bilstm"):
            tasks.append((shap_dl_path, shap_base + [
                "--model", "bilstm",
                "--bilstm_model_path", str(LSTM_MODEL_PATH),
            ]))
        
    # 3. Attention Visualization (Wav2Vec 2.0)
    if args.model in ("all", "wav2vec"):
        tasks.append((wav2vec_attn_path, [
            "--n_samples",  str(args.n_samples),
            "--device",     args.device,
            "--model_path", str(WAV2VEC_MODEL_PATH),
        ]))
        
    # Jalankan semua tugas yang dijadwalkan
    success_count = 0
    for script_path, script_args in tasks:
        if not script_path.exists():
            print(f"{RED}{BOLD}[WARNING]{RESET} File script tidak ditemukan: {script_path}")
            continue
        
        success = run_script(script_path, script_args)
        if success:
            success_count += 1
            
    print(f"\n{BLUE}{BOLD}=================================================================={RESET}")
    print(f"{GREEN}{BOLD}PIPELINE XAI SELESAI!{RESET}")
    print(f"Berhasil menjalankan {success_count} dari {len(tasks)} modul XAI.")
    print(f"Hasil visualisasi disimpan di direktori berikut:")
    if args.model in ("all", "cnn"):
        print(f"  • Grad-CAM   : {PROJECT_ROOT / 'results' / 'xai' / 'gradcam'}")
    if args.model in ("all", "cnn", "bilstm"):
        print(f"  • SHAP Value : {PROJECT_ROOT / 'results' / 'xai' / 'shap'}")
    if args.model in ("all", "wav2vec"):
        print(f"  • Attention  : {PROJECT_ROOT / 'results' / 'xai' / 'attention'}")
    print(f"Untuk penjelasan cara membaca grafik, silakan baca:")
    print(f"  {PROJECT_ROOT / 'docs' / 'xai_workflow_expectations.md'}")
    print(f"{BLUE}{BOLD}=================================================================={RESET}\n")

if __name__ == "__main__":
    main()
