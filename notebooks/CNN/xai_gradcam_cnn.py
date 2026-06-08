"""
XAI — Grad-CAM Heatmaps untuk 2D CNN Mel-Spectrogram
======================================================
Project : Mental Health Audio Classification (NORMAL vs DEPRESI)
Script  : notebooks/CNN/xai_gradcam_cnn.py

📝 PENJELASAN GRAD-CAM:
Grad-CAM (Gradient-weighted Class Activation Mapping) mengidentifikasi
region MANA pada mel-spectrogram yang paling berpengaruh terhadap prediksi
model. Gradient dari output kelas target di-backprop ke feature map
konvolusi terakhir, lalu di-average → bobot kepentingan kanal.

Output:
  results/xai/gradcam/
    ├── gradcam_<pid>_<kelas>_pred<kelas>.png   ← overlay heatmap
    ├── gradcam_summary_grid.png                 ← ringkasan multi-sample
    └── gradcam_report.txt                       ← metrik agregat

Cara menjalankan:
  python notebooks/CNN/xai_gradcam_cnn.py
  python notebooks/CNN/xai_gradcam_cnn.py --n_samples 20 --model_path models/dl/cnn/best_model.pt
"""

import argparse
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import librosa
import librosa.display
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import model dari training script
from notebooks.CNN.train_2d_cnn import (
    MelSpectrogram2DCNN, MelSpectrogramDataset, load_patient_data,
    CLASS_NAMES, CLASS_TO_IDX, FEATURES_DIR, MODEL_DIR, MAX_LEN,
)

RESULTS_XAI  = PROJECT_ROOT / "results" / "xai" / "gradcam"
RESULTS_XAI.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM untuk 2D CNN mel-spectrogram.

    Cara kerja:
    1. Forward pass → simpan feature map layer target via hook
    2. Backward pass dari skor kelas target → simpan gradients via hook
    3. Global average-pool gradients → bobot kepentingan tiap channel
    4. Weighted sum feature maps → raw CAM
    5. ReLU → resize ke ukuran input
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Simpan feature map saat forward pass."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Simpan gradients saat backward pass."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor : (1, 1, H, W) — satu sampel spectrogram
            class_idx    : indeks kelas target (None = predicted class)

        Returns:
            cam      : (H, W) heatmap dinormalisasi [0, 1]
            pred_cls : kelas yang diprediksi
            pred_conf: confidence score prediksi
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        logits = self.model(input_tensor)           # (1, n_classes)
        probs  = torch.softmax(logits, dim=-1)
        pred_cls  = int(torch.argmax(probs, dim=-1).item())
        pred_conf = float(probs[0, pred_cls].item())

        # Gunakan kelas target (default = predicted class)
        target = class_idx if class_idx is not None else pred_cls

        # Backward hanya pada skor kelas target
        self.model.zero_grad()
        score = logits[0, target]
        score.backward(retain_graph=True)

        # Grad-CAM computation
        grads   = self._gradients[0]           # (C, H', W')
        acts    = self._activations[0]         # (C, H', W')

        # Global average pooling of gradients → importance weights
        weights = grads.mean(dim=[1, 2])       # (C,)

        # Weighted combination of feature maps
        cam = torch.zeros(acts.shape[1:], device=acts.device)  # (H', W')
        for c, w in enumerate(weights):
            cam += w * acts[c]

        # ReLU → hanya aktivasi positif
        cam = torch.relu(cam)

        # Normalize ke [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, pred_cls, pred_conf

    def remove_hooks(self):
        """Hapus hooks setelah selesai."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_gradcam_overlay(
    spec        : np.ndarray,
    cam         : np.ndarray,
    true_label  : str,
    pred_label  : str,
    pred_conf   : float,
    sample_name : str,
    save_path   : Path,
    sr          : int = 16000,
    hop_length  : int = 512,
    n_mels      : int = 128,
):
    """
    Plot mel-spectrogram asli + Grad-CAM overlay berdampingan.

    Layout (3 panel):
    [Mel-Spectrogram] | [Grad-CAM Heatmap] | [Overlay]
    """
    # Resize CAM ke ukuran spectrogram
    from scipy.ndimage import zoom
    zoom_h = spec.shape[0] / cam.shape[0]
    zoom_w = spec.shape[1] / cam.shape[1]
    cam_resized = zoom(cam, (zoom_h, zoom_w), order=1)
    cam_resized = np.clip(cam_resized, 0, 1)

    # Normalisasi spectrogram untuk display
    spec_db = librosa.power_to_db(spec + 1e-9, ref=np.max)

    # Colormap untuk heatmap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # (H, W, 3) RGB

    # Spec ke RGB
    spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-9)
    spec_rgb  = plt.cm.magma(spec_norm)[:, :, :3]

    # Overlay: blend heatmap di atas spectrogram
    alpha   = 0.55
    overlay = np.clip(spec_rgb * (1 - alpha) + heatmap * alpha, 0, 1)

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#ffffff")

    time_axis = np.linspace(0, spec.shape[1] * hop_length / sr, spec.shape[1])
    freq_axis = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr // 2)

    # Label panel yang lebih awam
    titles = [
        "Suara Asli\n(Pola Frekuensi vs Waktu)",
        "Area Penting\n(Merah = Paling Diperhatikan Model)",
        "Gabungan\n(Suara + Sorotan Area Penting)",
    ]
    images = [spec_rgb, heatmap, overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(
            img, aspect="auto", origin="lower",
            extent=[time_axis[0], time_axis[-1], freq_axis[0] / 1000, freq_axis[-1] / 1000],
        )
        ax.set_title(title, color="#111111", fontsize=10, fontweight="bold")
        ax.set_xlabel("Waktu (detik)", color="#555555", fontsize=9)
        ax.set_ylabel("Frekuensi Suara (kHz)", color="#555555", fontsize=9)
        ax.tick_params(colors="#777777")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_facecolor("#f7f7f7")

    # Anotasi label frekuensi awam di sumbu Y panel pertama
    ax0 = axes[0]
    ax0.axhline(y=0.3,  color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    ax0.axhline(y=1.2,  color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    ax0.axhline(y=3.5,  color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    ax0.text(time_axis[-1]*0.01, 0.05,  "Nada Dasar",      color="#333333", fontsize=6.5, va="bottom")
    ax0.text(time_axis[-1]*0.01, 0.35,  "Vokal / Ucapan",  color="#333333", fontsize=6.5, va="bottom")
    ax0.text(time_axis[-1]*0.01, 1.25,  "Kejelasan Bicara",color="#333333", fontsize=6.5, va="bottom")
    ax0.text(time_axis[-1]*0.01, 3.6,   "Kecerahan Suara", color="#333333", fontsize=6.5, va="bottom")

    # Colorbar panel tengah (heatmap)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Tingkat Kepentingan\n(Biru=Rendah → Merah=Tinggi)",
                    color="#111111", fontsize=8)
    cbar1.ax.yaxis.set_tick_params(color="#111111")
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color="#111111")

    # Colorbar panel kanan (overlay) — referensi yang sama
    sm2 = plt.cm.ScalarMappable(cmap="jet", norm=mcolors.Normalize(vmin=0, vmax=1))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Tingkat Kepentingan\n(Biru=Rendah → Merah=Tinggi)",
                    color="#111111", fontsize=8)
    cbar2.ax.yaxis.set_tick_params(color="#111111")
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#111111")

    # Judul utama
    correct  = true_label == pred_label
    status   = "✓ PREDIKSI BENAR" if correct else "✗ PREDIKSI SALAH"
    color_s  = "#2ca02c" if correct else "#d62728"
    suptitle = (
        f"Analisis Area Penting Suara (Grad-CAM) — Pasien {sample_name}\n"
        f"Label Sebenarnya: {true_label}  |  Prediksi Model: {pred_label} "
        f"(keyakinan {pred_conf:.1%})  |  {status}"
    )
    fig.suptitle(suptitle, color=color_s, fontsize=12, fontweight="bold", y=1.01)

    # Caption penjelasan awam — SEBELUM tight_layout agar tidak terpotong
    caption = (
        "📌  Cara membaca grafik ini:\n"
        "     • Panel kiri   : rekaman suara asli — sumbu X = waktu bicara, sumbu Y = tinggi/rendah nada\n"
        "     • Panel tengah : area yang paling diperhatikan model — MERAH/KUNING = sangat penting, BIRU = kurang penting\n"
        "     • Panel kanan  : gabungan keduanya — terlihat bagian mana dari suara yang mempengaruhi keputusan model"
    )
    fig.text(
        0.01, 0.01, caption,
        ha="left", va="bottom",
        fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                  edgecolor="#555555", alpha=0.92),
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"  Saved: {save_path.name}")


def plot_summary_grid(results: list, save_path: Path):
    """
    Plot ringkasan grid Grad-CAM: beberapa sampel dalam satu figure.
    results = list of dict {spec, cam, true_label, pred_label, pred_conf, name}
    """
    n    = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4 + 1.2))
    fig.patch.set_facecolor("#f7f7f7")
    plt.subplots_adjust(top=0.88, bottom=0.10, hspace=0.50, wspace=0.08)

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    from scipy.ndimage import zoom

    for idx, res in enumerate(results):
        r, c = divmod(idx, cols)
        ax   = axes[r, c]

        spec   = res["spec"]
        cam    = res["cam"]
        spec_db = librosa.power_to_db(spec + 1e-9, ref=np.max)
        spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-9)
        spec_rgb  = plt.cm.magma(spec_norm)[:, :, :3]

        zoom_h    = spec.shape[0] / cam.shape[0]
        zoom_w    = spec.shape[1] / cam.shape[1]
        cam_r     = np.clip(zoom(cam, (zoom_h, zoom_w), order=1), 0, 1)
        heatmap   = cm.jet(cam_r)[:, :, :3]
        alpha     = 0.55
        overlay   = np.clip(spec_rgb * (1 - alpha) + heatmap * alpha, 0, 1)

        ax.imshow(overlay, aspect="auto", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

        correct = res["true_label"] == res["pred_label"]
        color_s = "#2ca02c" if correct else "#d62728"
        verdict = "✓ Benar" if correct else "✗ Salah"
        ax.set_title(
            f"Pasien {res['name']}  {verdict}\n"
            f"Aktual: {res['true_label']} | Prediksi: {res['pred_label']} ({res['pred_conf']:.0%})",
            color=color_s, fontsize=8, fontweight="bold", pad=4,
        )

    # Matikan axes yang tidak terpakai
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    # Colorbar horizontal sebagai legend warna
    import matplotlib.colors as mcolors_inner
    cax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
    sm  = plt.cm.ScalarMappable(
        cmap="jet", norm=mcolors_inner.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(
        "← Kurang Penting                Tingkat Kepentingan Area Suara                Sangat Penting →",
        color="#111111", fontsize=8,
    )
    cbar.ax.xaxis.set_tick_params(color="#111111")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="#111111")
    cax.set_facecolor("#f7f7f7")

    fig.suptitle(
        "Ringkasan Area Penting Suara (Grad-CAM) — CNN Mel-Spectrogram\n"
        "Warna MERAH/KUNING = bagian suara yang paling mempengaruhi keputusan model",
        color="#111111", fontsize=13, fontweight="bold", y=0.97,
    )
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"Summary grid saved: {save_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# ANALISIS AGREGAT
# ──────────────────────────────────────────────────────────────────────────────

def analyze_cam_regions(cam: np.ndarray, n_mels: int = 128, max_len: int = MAX_LEN):
    """
    Analisis CAM: identifikasi region frekuensi dan temporal yang paling aktif.

    Returns dict berisi:
    - top_freq_band : band frekuensi (low/mid/high) dengan aktivasi tertinggi
    - top_time_pct  : persentase temporal (0-100%) dengan aktivasi tertinggi
    - mean_activation: rata-rata aktivasi CAM
    """
    from scipy.ndimage import zoom

    # Resize ke ukuran input
    zoom_h = n_mels / cam.shape[0]
    zoom_w = max_len / cam.shape[1]
    cam_full = zoom(cam, (zoom_h, zoom_w), order=1)

    # Frekuensi: bagi jadi 3 band (low/mid/high)
    h = cam_full.shape[0]
    low  = cam_full[:h//3, :].mean()
    mid  = cam_full[h//3:2*h//3, :].mean()
    high = cam_full[2*h//3:, :].mean()
    band_scores = {"low": float(low), "mid": float(mid), "high": float(high)}
    top_freq    = max(band_scores, key=band_scores.get)

    # Temporal: rata-rata aktivasi per frame, ambil top 20%
    temporal_mean = cam_full.mean(axis=0)
    top_20_pct    = np.percentile(temporal_mean, 80)
    active_frames = np.where(temporal_mean >= top_20_pct)[0]
    time_pct_start = float(active_frames.min() / cam_full.shape[1] * 100) if len(active_frames) else 0
    time_pct_end   = float(active_frames.max() / cam_full.shape[1] * 100) if len(active_frames) else 100

    return {
        "freq_bands"        : band_scores,
        "top_freq_band"     : top_freq,
        "active_time_start" : time_pct_start,
        "active_time_end"   : time_pct_end,
        "mean_activation"   : float(cam_full.mean()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_gradcam(
    model_path : Optional[str] = None,
    n_samples  : int = 16,
    class_filter: Optional[str] = None,
    target_class: Optional[int] = None,
):
    """
    Jalankan Grad-CAM pada sejumlah sampel dari dataset.

    Args:
        model_path   : path ke file .pt (default: models/dl/cnn/best_model.pt)
        n_samples    : jumlah sampel yang di-explain (default: 16)
        class_filter : filter kelas sampel ('NORMAL', 'DEPRESI', atau None=semua)
        target_class : kelas target Grad-CAM (None = predicted class)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    if model_path is None:
        model_path = MODEL_DIR / "best_model_1;1.pt"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        logging.error(f"Model tidak ditemukan: {model_path}")
        logging.error("Pastikan sudah menjalankan train_2d_cnn.py terlebih dahulu.")
        sys.exit(1)

    model = MelSpectrogram2DCNN(num_classes=2).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Model loaded dari: {model_path}")

    # ── Target layer: conv2 (layer konvolusi terakhir sebelum GAP) ────────────
    # Arsitektur: conv1 → conv2 → global_pool → fc
    # conv2[-2] = ReLU (aktivasi akhir conv2) → ini yang paling informatif
    target_layer = model.conv2[-2]   # nn.ReLU setelah BatchNorm2d di conv2
    gradcam      = GradCAM(model, target_layer)
    logging.info(f"Target layer Grad-CAM: {target_layer.__class__.__name__} di conv2")

    # ── Load data ─────────────────────────────────────────────────────────────
    _, _, all_files, _ = load_patient_data()

    # Filter hanya file original (tanpa augmentasi)
    AUG_TAGS   = ["_noise", "_pitch", "_stretch", "_combo"]
    orig_files = [
        (p, pid, cls_idx) for p, pid, cls_idx in all_files
        if p.stem.endswith("_mel") and not any(t in p.stem for t in AUG_TAGS)
    ]

    # Filter kelas
    if class_filter is not None:
        cls_idx_filter = CLASS_TO_IDX.get(class_filter.upper())
        if cls_idx_filter is None:
            logging.warning(f"Kelas tidak dikenal: {class_filter}. Gunakan semua kelas.")
        else:
            orig_files = [(p, pid, cls) for p, pid, cls in orig_files if cls == cls_idx_filter]

    # Ambil n_samples sampel (seimbang antar kelas jika memungkinkan)
    import random
    random.seed(42)
    # Pisahkan per kelas dulu
    by_class = {0: [], 1: []}
    for entry in orig_files:
        by_class[entry[2]].append(entry)
    
    # PERBAIKAN: Hitung n_each dengan lebih baik untuk jumlah sampel sedikit
    # Jika n_samples ganjil, tambahkan 1 ke salah satu kelas
    n_each = n_samples // 2
    n_extra = n_samples % 2  # Sisa untuk class pertama jika ganjil
    
    selected = []
    for i, cls_idx in enumerate([0, 1]):
        cls_samples = by_class[cls_idx]
        random.shuffle(cls_samples)
        # Class pertama dapat extra sample jika n_samples ganjil
        n_take = n_each + (n_extra if i == 0 else 0)
        # Pastikan tidak melebihi jumlah sampel yang tersedia
        n_take = min(n_take, len(cls_samples))
        selected.extend(cls_samples[:n_take])
    
    # Jika masih kurang, ambil sampel tambahan dari kelas yang punya lebih banyak data
    if len(selected) < n_samples:
        remaining_needed = n_samples - len(selected)
        for cls_idx in [0, 1]:
            if remaining_needed <= 0:
                break
            cls_samples = by_class[cls_idx]
            already_taken = sum(1 for s in selected if s[2] == cls_idx)
            available = cls_samples[already_taken:]
            take_more = min(remaining_needed, len(available))
            selected.extend(available[:take_more])
            remaining_needed -= take_more

    logging.info(f"Total sampel untuk Grad-CAM: {len(selected)} (target: {n_samples})")
    # Hitung distribusi kelas
    class_dist = {0: 0, 1: 0}
    for _, _, cls_idx in selected:
        class_dist[cls_idx] += 1
    logging.info(f"  - NORMAL  : {class_dist[0]} sampel")
    logging.info(f"  - DEPRESI : {class_dist[1]} sampel")

    # ── Jalankan Grad-CAM ─────────────────────────────────────────────────────
    results      = []
    report_lines = [
        "=== Grad-CAM Analysis Report — 2D CNN Mel-Spectrogram ===\n",
        f"Model       : {model_path}\n",
        f"Target Layer: conv2[ReLU]\n",
        f"n_samples   : {len(selected)}\n\n",
        f"{'Sample':<30} {'True':>8} {'Pred':>8} {'Conf':>7} {'TopFreq':>8} {'TimeStart':>10} {'TimeEnd':>9}\n",
        "-" * 85 + "\n",
    ]

    correct_count = 0

    for path, pid, true_cls in selected:
        # Load spectrogram
        spec     = np.load(path)
        if spec.shape[1] > MAX_LEN:
            spec = spec[:, :MAX_LEN]
        else:
            spec = np.pad(spec, ((0,0),(0, MAX_LEN - spec.shape[1])), mode="constant")

        input_t  = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Generate CAM
        cam, pred_cls, pred_conf = gradcam.generate(input_t, class_idx=target_class)

        true_label = CLASS_NAMES[true_cls]
        pred_label = CLASS_NAMES[pred_cls]
        correct    = (true_cls == pred_cls)
        if correct:
            correct_count += 1

        # Analisis region
        region = analyze_cam_regions(cam)

        # Plot individual
        sample_name = f"{pid}_{path.stem}"
        save_path   = RESULTS_XAI / f"gradcam_{pid}_{true_label}_pred{pred_label}.png"
        plot_gradcam_overlay(
            spec        = spec,
            cam         = cam,
            true_label  = true_label,
            pred_label  = pred_label,
            pred_conf   = pred_conf,
            sample_name = sample_name,
            save_path   = save_path,
        )

        # Simpan untuk summary grid
        results.append({
            "spec"       : spec,
            "cam"        : cam,
            "true_label" : true_label,
            "pred_label" : pred_label,
            "pred_conf"  : pred_conf,
            "name"       : f"{pid}",
        })

        # Report line
        report_lines.append(
            f"{sample_name[:29]:<30} {true_label:>8} {pred_label:>8} {pred_conf:>7.1%} "
            f"{region['top_freq_band']:>8} {region['active_time_start']:>10.1f}% "
            f"{region['active_time_end']:>9.1f}%\n"
        )

        logging.info(
            f"[{true_label}→{pred_label}] {path.stem} | conf={pred_conf:.1%} | "
            f"freq={region['top_freq_band']} | time={region['active_time_start']:.0f}%-{region['active_time_end']:.0f}%"
        )

    # ── Summary grid ─────────────────────────────────────────────────────────
    if results:
        plot_summary_grid(results, RESULTS_XAI / "gradcam_summary_grid.png")

    # ── Report ────────────────────────────────────────────────────────────────
    accuracy = correct_count / max(1, len(selected))
    report_lines.extend([
        "\n" + "=" * 85 + "\n",
        f"Akurasi pada {len(selected)} sampel: {correct_count}/{len(selected)} ({accuracy:.1%})\n",
        "\nINTERPRETASI:\n",
        "- Heatmap merah/kuning  = region PALING BERPENGARUH pada keputusan model\n",
        "- Heatmap biru/hijau    = region kurang berpengaruh\n",
        "- Top Freq Band         : band frekuensi dominan (low=0-3kHz, mid=3-6kHz, high=6-8kHz)\n",
        "- Time Start/End        : persentase posisi temporal area paling aktif\n",
        "\nCatatan untuk depresi detection:\n",
        "- Literatur menunjukkan fitur prosodik (pitch, energy) lebih dominan\n",
        "- Frekuensi rendah (formant F0, F1) umumnya lebih informatif\n",
        "- Segmen awal dan akhir ucapan cenderung lebih diagnostik\n",
    ])

    report_path = RESULTS_XAI / "gradcam_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    gradcam.remove_hooks()
    logging.info(f"\n✅ Grad-CAM selesai!")
    logging.info(f"   Hasil disimpan di: {RESULTS_XAI}")
    logging.info(f"   Akurasi: {accuracy:.1%} ({correct_count}/{len(selected)} benar)")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grad-CAM XAI untuk 2D CNN Mel-Spectrogram (Mental Health)"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path ke file .pt model CNN (default: models/dl/cnn/best_model_1;1.pt)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="Jumlah sampel yang di-explain (default: 4 = 2 NORMAL + 2 DEPRESI)",
    )
    parser.add_argument(
        "--class_filter", type=str, default=None,
        choices=["NORMAL", "DEPRESI"],
        help="Filter kelas sampel input (default: semua kelas)",
    )
    parser.add_argument(
        "--target_class", type=int, default=None,
        choices=[0, 1],
        help="Kelas target Grad-CAM: 0=NORMAL, 1=DEPRESI (default: predicted class)",
    )
    args = parser.parse_args()

    run_gradcam(
        model_path   = args.model_path,
        n_samples    = args.n_samples,
        class_filter = args.class_filter,
        target_class = args.target_class,
    )
