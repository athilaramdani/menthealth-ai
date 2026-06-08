"""
XAI — Visualisasi Attention Weights Wav2Vec 2.0
================================================
Project : Mental Health Audio Classification (NORMAL vs DEPRESI)
Script  : notebooks/WAV2VEC/xai_attention_wav2vec.py

📝 PENJELASAN ATTENTION VISUALIZATION:
Wav2Vec2 menggunakan Transformer dengan Multi-Head Self-Attention.
Setiap head attention mempelajari hubungan antar frame audio yang berbeda.
Dengan memvisualisasikan attention weights, kita dapat melihat:

1. SELF-ATTENTION MAPS       : frame mana yang "memerhatikan" frame lain
2. TEMPORAL ATTENTION PROFILE: seberapa besar perhatian model pada
                               setiap posisi temporal dalam audio
3. HEAD COMPARISON           : variasi pola antar attention head
4. LAYER COMPARISON          : evolusi representasi dari layer bawah ke atas
5. TOKEN IMPORTANCE          : token audio mana yang paling informatif

Output:
  results/xai/attention/
    ├── attn_<pid>_head_heatmap.png      ← self-attention matrix per head
    ├── attn_<pid>_temporal_profile.png  ← profil kepentingan temporal
    ├── attn_<pid>_layer_evolution.png   ← evolusi attention antar layer
    ├── attn_summary_grid.png            ← ringkasan multi-sample
    └── attn_report.txt                  ← laporan agregat

Cara menjalankan:
  python notebooks/WAV2VEC/xai_attention_wav2vec.py
  python notebooks/WAV2VEC/xai_attention_wav2vec.py --n_samples 10 --n_heads 4
"""

import argparse
import logging
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

RESULTS_XAI = PROJECT_ROOT / "results" / "xai" / "attention"
RESULTS_XAI.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ── Import model Wav2Vec2 ─────────────────────────────────────────────────────
try:
    from transformers import Wav2Vec2Model
    TRANSFORMERS_OK = True
except ImportError:
    logging.error("transformers tidak terinstall. Jalankan: pip install transformers")
    TRANSFORMERS_OK = False

try:
    from train_wav2vec import (
        Wav2Vec2Classifier, PRETRAINED, CLASS_NAMES, MODEL_DIR,
        SAMPLE_RATE, MAX_DURATION,
    )
    from dataloader_wav2vec import collect_all_files, Wav2VecDataset
    MAX_DURATION_SEC = MAX_DURATION
    WAV2VEC_AVAILABLE = True
except ImportError:
    try:
        from notebooks.WAV2VEC.train_wav2vec import (
            Wav2Vec2Classifier, PRETRAINED, CLASS_NAMES, MODEL_DIR,
            SAMPLE_RATE, MAX_DURATION,
        )
        from notebooks.WAV2VEC.dataloader_wav2vec import collect_all_files, Wav2VecDataset
        MAX_DURATION_SEC = MAX_DURATION
        WAV2VEC_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"Wav2Vec2 module tidak bisa di-import: {e}")
        WAV2VEC_AVAILABLE = False
        # Fallback constants
        CLASS_NAMES      = ["NORMAL", "DEPRESI"]
        PRETRAINED       = "facebook/wav2vec2-base"
        MODEL_DIR        = PROJECT_ROOT / "models" / "dl" / "wav2vec"
        SAMPLE_RATE      = 16000
        MAX_DURATION_SEC = 30.0


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class Wav2Vec2AttentionExtractor:
    """
    Ekstrak attention weights dari semua layer Transformer Wav2Vec2.

    Cara kerja:
    1. Register forward hooks pada setiap layer attention
    2. Forward pass → hooks menyimpan attention weights
    3. Return attention weights per layer dan per head

    Attention shape: (batch, n_heads, T, T)
    """

    def __init__(self, model: nn.Module):
        self.model   = model
        self._attns  = {}   # layer_idx → attention tensor
        self._hooks  = []

        # Cari semua attention layer di Transformer encoder
        wav2vec2 = model.wav2vec2
        for layer_idx, layer in enumerate(wav2vec2.encoder.layers):
            hook = layer.attention.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

        logging.info(
            f"Hooks dipasang pada {len(wav2vec2.encoder.layers)} Transformer layers"
        )

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output dari SelfAttention: (hidden_states, attn_weights, ...)
            # attn_weights shape: (batch, heads, T, T)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self._attns[layer_idx] = output[1].detach().cpu()
            else:
                # Beberapa versi transformers tidak return attn_weights by default
                self._attns[layer_idx] = None
        return hook

    def extract(
        self,
        input_values  : torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass dengan output_attentions=True untuk mendapat attention weights.

        Returns:
            dict berisi:
            - logits     : (1, n_classes)
            - probs      : (1, n_classes)
            - attentions : dict {layer_idx: (1, n_heads, T, T)}
            - hidden     : (1, T, H) — last hidden states
        """
        self.model.eval()
        self._attns.clear()

        with torch.no_grad():
            # Gunakan wav2vec2 dengan output_attentions=True
            wav2vec2_out = self.model.wav2vec2(
                input_values   = input_values,
                attention_mask = attention_mask,
                output_attentions = True,
            )

            hidden   = wav2vec2_out.last_hidden_state   # (1, T, H)
            attentions_tuple = wav2vec2_out.attentions  # tuple of (1, heads, T, T) per layer

            # Override hooks dengan output langsung (lebih reliable)
            if attentions_tuple is not None:
                for i, attn in enumerate(attentions_tuple):
                    self._attns[i] = attn.detach().cpu()

            # Pooling + classifier head
            pooled = hidden.mean(dim=1)
            # Gunakan forward() model langsung agar kompatibel dengan semua arsitektur
            logits = self.model.classifier(pooled)
            probs  = torch.softmax(logits, dim=-1)

        return {
            "logits"    : logits.cpu(),
            "probs"     : probs.cpu(),
            "attentions": dict(self._attns),
            "hidden"    : hidden.cpu(),
        }

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# ANALISIS ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_attention_rollout(attentions: dict, n_layers: int) -> np.ndarray:
    """
    Attention Rollout: hitung aliran attention dari input ke output
    dengan mengalikan attention matrices antar layer.

    Ini memberikan estimasi "effective attention" yang memperhitungkan
    residual connections, berbeda dari raw attention yang mungkin misleading.

    Reference: Abnar & Zuidema, 2020 — "Quantifying Attention Flow in Transformers"

    Returns:
        rollout : (T,) — effective attention dari semua token ke [CLS] (atau mean)
    """
    if not attentions:
        return np.array([])

    T = None
    for layer_idx in range(n_layers):
        if layer_idx in attentions and attentions[layer_idx] is not None:
            T = attentions[layer_idx].shape[-1]
            break

    if T is None:
        return np.array([])

    # Inisialisasi dengan identity
    rollout = np.eye(T)

    for layer_idx in range(n_layers):
        if layer_idx not in attentions or attentions[layer_idx] is None:
            continue
        attn = attentions[layer_idx][0].numpy()   # (heads, T, T)
        # Rata-rata semua heads
        attn_mean = attn.mean(axis=0)              # (T, T)
        # Tambahkan residual identity (untuk residual connections)
        attn_aug  = attn_mean + np.eye(T)
        attn_aug  = attn_aug / attn_aug.sum(axis=-1, keepdims=True)
        rollout   = attn_aug @ rollout

    # Ambil baris terakhir atau rata-rata semua posisi
    rollout_score = rollout.mean(axis=0)  # (T,)
    rollout_score = (rollout_score - rollout_score.min()) / (rollout_score.max() - rollout_score.min() + 1e-9)
    return rollout_score


def compute_mean_head_attention(attentions: dict, layer_idx: int = -1) -> np.ndarray:
    """
    Hitung mean attention matrix dari semua heads pada layer tertentu.

    Returns:
        mean_attn : (T, T)
    """
    layers = sorted(attentions.keys())
    if not layers:
        return np.array([])
    if layer_idx == -1:
        layer_idx = layers[-1]

    attn = attentions.get(layer_idx)
    if attn is None:
        return np.array([])

    return attn[0].numpy().mean(axis=0)   # (T, T)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISASI
# ══════════════════════════════════════════════════════════════════════════════

def plot_attention_heatmap(
    attentions  : dict,
    pred_label  : str,
    true_label  : str,
    pred_conf   : float,
    sample_name : str,
    save_path   : Path,
    n_heads_show: int = 4,
    layer_show  : int = -1,   # -1 = last layer
):
    """
    Plot self-attention matrix per head dari 1 layer.

    Layout: grid n_heads_show × 1 attention heatmap (T × T)
    """
    layers = sorted(attentions.keys())
    if not layers:
        logging.warning("Tidak ada attention data tersedia.")
        return

    if layer_show == -1:
        layer_show = layers[-1]

    attn = attentions.get(layer_show)
    if attn is None:
        return

    attn_np  = attn[0].numpy()   # (n_heads, T, T)
    n_heads  = min(n_heads_show, attn_np.shape[0])
    T        = attn_np.shape[-1]

    # Subsample T jika terlalu besar (> 100 frames)
    step     = max(1, T // 100)
    t_slice  = slice(None, None, step)

    fig, axes = plt.subplots(1, n_heads, figsize=(n_heads * 5, 6))
    plt.subplots_adjust(top=0.82, bottom=0.12, wspace=0.30)
    fig.patch.set_facecolor("#f7f7f7")
    if n_heads == 1:
        axes = [axes]

    for h in range(n_heads):
        ax   = axes[h]
        data = attn_np[h][t_slice, :][:, t_slice]

        im = ax.imshow(data, cmap="inferno", aspect="auto",
                       vmin=0, vmax=data.max())
        ax.set_title(f"Sudut Pandang {h+1}\n(Attention Head {h+1})",
                     color="#111111", fontsize=9, fontweight="bold")
        ax.set_xlabel("Frame Audio (Sumber)", color="#555555", fontsize=8)
        ax.set_ylabel("Frame Audio (Tujuan)", color="#555555", fontsize=8)
        ax.tick_params(colors="#777777", labelsize=7)
        ax.set_facecolor("#f7f7f7")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Kekuatan\nPerhatian", color="#111111", fontsize=7)
        cbar.ax.yaxis.set_tick_params(color="#111111")

    correct = true_label == pred_label
    color_s = "#2ca02c" if correct else "#d62728"
    status  = "✓ Benar" if correct else "✗ Salah"
    fig.suptitle(
        f"Pola Perhatian Model (Self-Attention) — Layer Terakhir | Pasien {sample_name}\n"
        f"Sebenarnya: {true_label}  |  Prediksi: {pred_label} ({pred_conf:.1%})  |  {status}\n"
        "Setiap panel = satu 'sudut pandang' model saat mendengarkan audio. "
        "Warna terang = koneksi kuat antar bagian suara.",
        color=color_s, fontsize=10, fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"  Saved: {save_path.name}")


def plot_temporal_attention_profile(
    attentions      : dict,
    rollout_score   : np.ndarray,
    waveform        : np.ndarray,
    sample_rate     : int,
    pred_label      : str,
    true_label      : str,
    pred_conf       : float,
    sample_name     : str,
    save_path       : Path,
):
    """
    Plot temporal attention profile: seberapa besar perhatian model
    pada setiap frame audio.

    Layout 3 panel:
    [Waveform] | [Rollout Attention] | [Layer-wise Mean Attention]
    """
    layers    = sorted(attentions.keys())
    n_layers  = len(layers)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.patch.set_facecolor("#f7f7f7")

    def style_ax(ax, title):
        ax.set_title(title, color="#111111", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#777777")
        ax.set_facecolor("#f7f7f7")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

    # Panel 1: Waveform
    time_wav = np.linspace(0, len(waveform) / sample_rate, len(waveform))
    axes[0].fill_between(time_wav, waveform, alpha=0.7, color="#1f77b4")
    axes[0].plot(time_wav, waveform, color="#1f77b4", linewidth=0.5)
    axes[0].set_xlabel("Waktu (detik)", color="#555555", fontsize=8)
    axes[0].set_ylabel("Keras/Pelan Suara", color="#555555", fontsize=8)
    style_ax(axes[0], "🎙️  Rekaman Suara Asli")

    # Panel 2: Attention Rollout (frame-level importance)
    if len(rollout_score) > 0:
        frame_times = np.linspace(0, len(waveform) / sample_rate, len(rollout_score))
        axes[1].fill_between(frame_times, rollout_score, alpha=0.6, color="#d62728")
        axes[1].plot(frame_times, rollout_score, color="#d62728", linewidth=1.5)

        threshold = np.percentile(rollout_score, 90)
        axes[1].axhline(threshold, color="yellow", linestyle="--", alpha=0.6, linewidth=1)
        axes[1].fill_between(
            frame_times, rollout_score, threshold,
            where=rollout_score >= threshold,
            alpha=0.5, color="yellow",
            label="10% momen paling penting",
        )
        axes[1].legend(loc="upper right", fontsize=8,
                       facecolor="#f0f0f0", edgecolor="#bbbbbb", labelcolor="#111111")
    axes[1].set_xlabel("Waktu (detik)", color="#555555", fontsize=8)
    axes[1].set_ylabel("Tingkat Perhatian Model", color="#555555", fontsize=8)
    style_ax(axes[1],
             "🔍  Momen yang Paling Diperhatikan Model\n"
             "     (Puncak = bagian suara paling menentukan prediksi)")

    # Panel 3: Layer-wise mean attention heatmap
    layer_attns = []
    for l in layers:
        if attentions[l] is not None:
            a = attentions[l][0].numpy()
            layer_attns.append(a.mean(axis=(0, 1)))

    if layer_attns:
        layer_attn_matrix = np.stack(layer_attns, axis=0)
        layer_attn_matrix = (layer_attn_matrix - layer_attn_matrix.min(axis=1, keepdims=True)) / \
                            (layer_attn_matrix.max(axis=1, keepdims=True) -
                             layer_attn_matrix.min(axis=1, keepdims=True) + 1e-9)

        im = axes[2].imshow(
            layer_attn_matrix, aspect="auto", cmap="plasma",
            extent=[0, len(waveform) / sample_rate, 0, len(layer_attns)],
        )
        axes[2].set_xlabel("Waktu (detik)", color="#555555", fontsize=8)
        axes[2].set_ylabel("Lapisan Pemrosesan\n(Bawah=Dasar → Atas=Tingkat Lanjut)",
                           color="#555555", fontsize=8)
        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
        cbar.set_label("Intensitas\nPerhatian", color="#111111", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#111111")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#111111")

    style_ax(axes[2],
             "📊  Perhatian per Lapisan Pemrosesan\n"
             "     (Lapisan bawah = pola dasar, lapisan atas = makna keseluruhan)")

    correct = true_label == pred_label
    color_s = "#2ca02c" if correct else "#d62728"
    status  = "✓ Prediksi Benar" if correct else "✗ Prediksi Salah"
    fig.suptitle(
        f"Analisis Perhatian Model (Wav2Vec2) — Pasien {sample_name}\n"
        f"Label Sebenarnya: {true_label}  |  Prediksi: {pred_label} "
        f"(keyakinan {pred_conf:.1%})  |  {status}",
        color=color_s, fontsize=12, fontweight="bold", y=1.01,
    )

    # Caption awam
    caption = (
        "📌  Cara membaca grafik ini:\n"
        "     • Panel atas   : gelombang suara asli — naik-turun = keras-pelannya suara\n"
        "     • Panel tengah : bagian suara yang paling diperhatikan model — puncak kuning = momen paling menentukan\n"
        "     • Panel bawah  : bagaimana perhatian berubah di setiap lapisan pemrosesan model"
    )
    fig.text(
        0.01, 0.005, caption,
        ha="left", va="bottom",
        fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                  edgecolor="#555555", alpha=0.92),
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"  Saved: {save_path.name}")


def plot_layer_evolution(
    attentions  : dict,
    sample_name : str,
    save_path   : Path,
    n_layers_show: int = 6,
):
    """
    Visualisasi evolusi attention antar layer:
    - Bagaimana representasi audio berubah dari layer rendah ke tinggi
    - Layer rendah: attention lokal (frame-ke-frame terdekat)
    - Layer tinggi: attention global (long-range dependencies)
    """
    layers   = sorted(attentions.keys())
    if not layers:
        return

    # Pilih subset layer yang representatif
    step        = max(1, len(layers) // n_layers_show)
    show_layers = layers[::step][:n_layers_show]
    n_show      = len(show_layers)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 4.5, 9))
    plt.subplots_adjust(hspace=0.45, wspace=0.15, top=0.88, bottom=0.06)
    fig.patch.set_facecolor("#f7f7f7")
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col, l in enumerate(show_layers):
        if attentions[l] is None:
            continue
        attn_np = attentions[l][0].numpy()   # (heads, T, T)
        T       = attn_np.shape[-1]
        step_T  = max(1, T // 80)

        # Row 0: Mean attention matrix (T × T)
        mean_attn = attn_np.mean(axis=0)[::step_T, :][:, ::step_T]
        im0 = axes[0, col].imshow(mean_attn, cmap="inferno", aspect="auto")
        axes[0, col].set_title(
            f"Lapisan {l}\n({'Dasar' if l < 4 else 'Menengah' if l < 9 else 'Tingkat Lanjut'})",
            color="#111111", fontsize=9, fontweight="bold")
        axes[0, col].tick_params(colors="#777777", labelsize=6)
        axes[0, col].set_facecolor("#f7f7f7")

        # Row 1: Attention entropy
        entropies = []
        for h in range(attn_np.shape[0]):
            attn_h   = attn_np[h] + 1e-9
            attn_h   = attn_h / attn_h.sum(axis=-1, keepdims=True)
            entropy  = -(attn_h * np.log(attn_h)).sum(axis=-1)
            entropies.append(entropy)
        entropy_mean = np.stack(entropies).mean(axis=0)

        axes[1, col].fill_between(range(T), entropy_mean, alpha=0.6, color="#4c72b0")
        axes[1, col].plot(entropy_mean, color="#4c72b0", linewidth=1)
        axes[1, col].set_ylabel(
            "Sebaran\nPerhatian" if col == 0 else "", color="#555555", fontsize=8)
        axes[1, col].set_xlabel("Frame Audio", color="#555555", fontsize=8)
        axes[1, col].tick_params(colors="#777777", labelsize=6)
        axes[1, col].set_facecolor("#f7f7f7")
        for sp in axes[1, col].spines.values():
            sp.set_edgecolor("#333333")

    axes[0, 0].set_ylabel("Pola Koneksi\nAntar Frame", color="#555555", fontsize=8)

    fig.suptitle(
        f"Evolusi Pemrosesan Model (Wav2Vec2) — Pasien {sample_name}\n"
        "Baris atas: pola koneksi antar bagian suara  |  "
        "Baris bawah: seberapa luas perhatian model (tinggi = global, rendah = lokal)\n"
        "Lapisan Dasar → menangkap bunyi dasar  |  Lapisan Tingkat Lanjut → memahami makna keseluruhan",
        color="#111111", fontsize=10, fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"  Saved: {save_path.name}")


def plot_summary_grid_attention(results: list, save_path: Path):
    """
    Grid ringkasan: temporal attention profile semua sampel dalam 1 figure.
    """
    n    = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3 + 1.0))
    plt.subplots_adjust(top=0.88, bottom=0.10, hspace=0.55, wspace=0.08)
    fig.patch.set_facecolor("#f7f7f7")

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, res in enumerate(results):
        r, c = divmod(idx, cols)
        ax   = axes[r, c]

        rollout = res.get("rollout", np.array([]))
        if len(rollout) > 0:
            ax.fill_between(range(len(rollout)), rollout, alpha=0.6, color="#d62728")
            ax.plot(rollout, color="#d62728", linewidth=1)
            threshold = np.percentile(rollout, 90)
            ax.axhline(threshold, color="yellow", linestyle="--", alpha=0.5, linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "No attention\ndata", ha="center", va="center",
                    color="#111111", transform=ax.transAxes, fontsize=9)

        correct = res["true_label"] == res["pred_label"]
        color_s = "#2ca02c" if correct else "#d62728"
        verdict = "✓ Benar" if correct else "✗ Salah"
        ax.set_title(
            f"Pasien {res['name']}  {verdict}\n"
            f"Sebenarnya: {res['true_label']} | Prediksi: {res['pred_label']} ({res['pred_conf']:.0%})",
            color=color_s, fontsize=7.5, fontweight="bold",
        )
        ax.set_xlabel("Waktu (detik)", color="#777777", fontsize=6)
        ax.set_ylabel("Perhatian", color="#777777", fontsize=6)
        ax.tick_params(colors="#777777", labelsize=6)
        ax.set_facecolor("#f7f7f7")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(
        "Ringkasan Momen Penting Suara (Wav2Vec2 Attention Rollout)\n"
        "Setiap grafik = satu rekaman. Puncak = momen suara yang paling menentukan prediksi model.",
        color="#111111", fontsize=12, fontweight="bold", y=0.97,
    )
    # Colorbar horizontal sebagai referensi
    import matplotlib.colors as _mcolors
    cax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
    sm  = plt.cm.ScalarMappable(
        cmap="RdPu", norm=_mcolors.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(
        "← Perhatian Rendah                        Tingkat Perhatian Model                        Perhatian Tinggi →",
        color="#111111", fontsize=7.5,
    )
    cbar.ax.xaxis.set_tick_params(color="#111111")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="#111111")
    cax.set_facecolor("#f7f7f7")
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"Summary grid saved: {save_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_attention_visualization(
    model_path  : Optional[Path] = None,
    n_samples   : int = 8,
    n_heads_show: int = 4,
    device_str  : str = "cpu",
):
    """
    Jalankan visualisasi attention Wav2Vec2.
    """
    if not WAV2VEC_AVAILABLE or not TRANSFORMERS_OK:
        logging.error(
            "Module wav2vec2 atau transformers tidak tersedia.\n"
            "Pastikan train_wav2vec.py dan dataloader_wav2vec.py bisa di-import."
        )
        return

    device = torch.device(device_str)
    logging.info("\n" + "="*60)
    logging.info("Wav2Vec2 Attention Visualization — XAI")
    logging.info("="*60)

    # Load model
    if model_path is None:
        model_path = MODEL_DIR / "best_model_v2.pt"
    if not model_path.exists():
        logging.error(f"Model Wav2Vec2 tidak ditemukan: {model_path}")
        logging.error("Jalankan train_wav2vec.py terlebih dahulu.")
        return

    model = Wav2Vec2Classifier(PRETRAINED, num_classes=2).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # ── Auto-detect arsitektur classifier dari state_dict ────────────────────
    # Model lama (v1/v2): classifier.weight + classifier.bias (Linear tunggal)
    # Model baru        : classifier.0.weight … (Sequential kompleks)
    is_legacy = "classifier.weight" in state and "classifier.bias" in state

    if is_legacy:
        # Bangun ulang model dengan arsitektur lama agar cocok dengan weights
        logging.info("Terdeteksi arsitektur classifier LAMA (Linear tunggal). Menyesuaikan model...")
        hidden_size = model.wav2vec2.config.hidden_size
        model.classifier = nn.Linear(hidden_size, 2).to(device)
        model.dropout    = nn.Dropout(0.1).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"Keys tidak ada di checkpoint (tidak masalah jika minor): {missing}")
    if unexpected:
        logging.warning(f"Keys tidak dikenali di checkpoint (diabaikan): {unexpected}")
    model.eval()
    logging.info(f"Model Wav2Vec2 loaded: {model_path}")

    n_transformer_layers = len(model.wav2vec2.encoder.layers)
    logging.info(f"Transformer layers: {n_transformer_layers}")

    # Extractor
    extractor = Wav2Vec2AttentionExtractor(model)

    # Load data samples
    import random
    random.seed(42)
    _, _, all_files = collect_all_files()

    # Pisah per kelas untuk sampling seimbang (n/2 NORMAL + n/2 DEPRESI)
    by_class = {0: [], 1: []}
    for entry in all_files:
        by_class[entry[2]].append(entry)
    for cls_list in by_class.values():
        random.shuffle(cls_list)

    n_each  = n_samples // 2
    n_extra = n_samples % 2
    selected = []
    for i, cls_idx in enumerate([0, 1]):
        n_take = n_each + (n_extra if i == 0 else 0)
        n_take = min(n_take, len(by_class[cls_idx]))
        selected.extend(by_class[cls_idx][:n_take])
    # Jika masih kurang (kelas tidak cukup), ambil sisa dari kelas lain
    if len(selected) < n_samples:
        remaining = n_samples - len(selected)
        for cls_idx in [0, 1]:
            if remaining <= 0:
                break
            already = sum(1 for s in selected if s[2] == cls_idx)
            extra   = by_class[cls_idx][already:already + remaining]
            selected.extend(extra)
            remaining -= len(extra)

    class_dist = {CLASS_NAMES[0]: 0, CLASS_NAMES[1]: 0}
    for _, _, cls_idx in selected:
        class_dist[CLASS_NAMES[cls_idx]] += 1

    logging.info(f"Jumlah sampel: {len(selected)} — {class_dist}")

    report_lines = [
        "=== Wav2Vec2 Attention Visualization Report ===\n\n",
        f"Model         : {model_path}\n",
        f"Pretrained    : {PRETRAINED}\n",
        f"n_layers      : {n_transformer_layers}\n",
        f"n_samples     : {len(selected)}\n\n",
        f"{'Sample':<30} {'True':>8} {'Pred':>8} {'Conf':>7} {'PeakAttn%':>10} {'AvgEntropy':>11}\n",
        "-" * 80 + "\n",
    ]

    summary_results = []
    correct_count   = 0

    for path, pid, cls_idx in selected:
        true_label = CLASS_NAMES[cls_idx]
        name       = f"{pid}_{Path(path).stem}"

        # Load waveform
        import soundfile as sf
        import librosa as _lib
        try:
            wav, sr = sf.read(str(path))
            if sr != SAMPLE_RATE:
                wav = _lib.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
                sr  = SAMPLE_RATE
        except Exception as e:
            logging.warning(f"  Gagal load audio {path}: {e}")
            continue

        # Truncate/pad
        max_samples = int(MAX_DURATION_SEC * SAMPLE_RATE)
        if len(wav) > max_samples:
            wav = wav[:max_samples]

        waveform  = wav.astype(np.float32)
        input_t   = torch.tensor(waveform[np.newaxis, :], dtype=torch.float32).to(device)
        mask      = torch.ones(1, len(waveform), dtype=torch.long).to(device)

        logging.info(f"  Processing: {name} [{true_label}] ({len(waveform)/SAMPLE_RATE:.1f}s)")

        # Extract attention
        result = extractor.extract(input_t, mask)

        probs       = result["probs"]
        pred_idx    = int(torch.argmax(probs, dim=-1).item())
        pred_conf   = float(probs[0, pred_idx].item())
        pred_label  = CLASS_NAMES[pred_idx]
        attentions  = result["attentions"]

        if not attentions or all(v is None for v in attentions.values()):
            logging.warning(f"  Tidak ada attention data untuk {name}. Skip.")
            continue

        if pred_label == true_label:
            correct_count += 1

        # Attention rollout
        rollout = compute_attention_rollout(attentions, n_transformer_layers)

        # 1. Heatmap self-attention per head (last layer)
        save_attn = RESULTS_XAI / f"attn_{pid}_head_heatmap.png"
        plot_attention_heatmap(
            attentions   = attentions,
            pred_label   = pred_label,
            true_label   = true_label,
            pred_conf    = pred_conf,
            sample_name  = name,
            save_path    = save_attn,
            n_heads_show = n_heads_show,
            layer_show   = -1,
        )

        # 2. Temporal attention profile
        save_temp = RESULTS_XAI / f"attn_{pid}_temporal_profile.png"
        plot_temporal_attention_profile(
            attentions   = attentions,
            rollout_score= rollout,
            waveform     = waveform,
            sample_rate  = SAMPLE_RATE,
            pred_label   = pred_label,
            true_label   = true_label,
            pred_conf    = pred_conf,
            sample_name  = name,
            save_path    = save_temp,
        )

        # 3. Layer evolution
        save_evol = RESULTS_XAI / f"attn_{pid}_layer_evolution.png"
        plot_layer_evolution(
            attentions    = attentions,
            sample_name   = name,
            save_path     = save_evol,
            n_layers_show = min(6, n_transformer_layers),
        )

        # Statistik untuk report
        peak_pct    = float(np.percentile(rollout, 90) * 100) if len(rollout) > 0 else 0.0
        avg_entropy = 0.0
        for l, a in attentions.items():
            if a is None: continue
            a_np     = a[0].numpy()  # (heads, T, T)
            a_norm   = a_np + 1e-9
            a_norm   = a_norm / a_norm.sum(axis=-1, keepdims=True)
            entropy  = -(a_norm * np.log(a_norm)).sum(axis=-1).mean()
            avg_entropy += float(entropy)
        avg_entropy /= max(1, len([v for v in attentions.values() if v is not None]))

        report_lines.append(
            f"{name[:29]:<30} {true_label:>8} {pred_label:>8} {pred_conf:>7.1%} "
            f"{peak_pct:>10.2f}% {avg_entropy:>11.4f}\n"
        )

        summary_results.append({
            "rollout"    : rollout,
            "true_label" : true_label,
            "pred_label" : pred_label,
            "pred_conf"  : pred_conf,
            "name"       : str(pid),
        })

        logging.info(
            f"  [{true_label}→{pred_label}] conf={pred_conf:.1%} | "
            f"peak={peak_pct:.1f}% | entropy={avg_entropy:.4f}"
        )

    # Summary grid
    if summary_results:
        plot_summary_grid_attention(
            summary_results,
            RESULTS_XAI / "attn_summary_grid.png",
        )

    # Report
    accuracy = correct_count / max(1, len(summary_results))
    report_lines.extend([
        "\n" + "=" * 80 + "\n",
        f"Akurasi    : {correct_count}/{len(summary_results)} ({accuracy:.1%})\n",
        "\nINTERPRETASI:\n",
        "- Peak Attn% : persentil-90 dari rollout attention (semakin tinggi = lebih fokus)\n",
        "- Avg Entropy: rata-rata entropy attention (tinggi = global/difus, rendah = lokal/fokus)\n",
        "- Layer rendah (1-4) umumnya menangkap pola fonetik lokal\n",
        "- Layer tinggi (8-12) menangkap pola prosodik jangka panjang (relevan untuk depresi)\n",
        "\nCatatan:\n",
        "- Attention weights bukanlah penjelasan sempurna — mereka korelasional, bukan kausal\n",
        "- Gunakan Attention Rollout (bukan raw attention) untuk estimasi yang lebih akurat\n",
        "- Kombinasikan dengan SHAP untuk interpretasi yang lebih robust\n",
    ])

    report_path = RESULTS_XAI / "attn_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    extractor.remove_hooks()
    logging.info(f"\n✅ Attention visualization selesai!")
    logging.info(f"   Hasil di: {RESULTS_XAI}")
    logging.info(f"   Akurasi : {accuracy:.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisasi Attention Wav2Vec2 — XAI Mental Health"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path ke model Wav2Vec2 .pt (default: models/dl/wav2vec/best_model_v2.pt)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="Jumlah sampel audio (default: 4 = 2 NORMAL + 2 DEPRESI)",
    )
    parser.add_argument(
        "--n_heads", type=int, default=4,
        help="Jumlah attention heads yang divisualisasi (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device komputasi (default: cpu)",
    )
    args = parser.parse_args()

    run_attention_visualization(
        model_path   = Path(args.model_path) if args.model_path else None,
        n_samples    = args.n_samples,
        n_heads_show = args.n_heads,
        device_str   = args.device,
    )
