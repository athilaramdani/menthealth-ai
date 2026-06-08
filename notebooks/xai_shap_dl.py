"""
XAI — SHAP DeepExplainer untuk Model Deep Learning
====================================================
Project : Mental Health Audio Classification (NORMAL vs DEPRESI)
Script  : notebooks/xai_shap_dl.py

Visualisasi standar SHAP (Bar + Beeswarm/Summary + Waterfall) untuk:
  - BiLSTM MFCC  : fitur bernama → Bar, Beeswarm, Waterfall per sampel
  - CNN Spectrogram: fitur pixel → Bar (per freq-band), Beeswarm (aggregated), Waterfall

Output:
  results/xai/shap/
    ├── shap_bilstm_bar.png              ← global bar chart (mean |SHAP|)
    ├── shap_bilstm_beeswarm.png         ← beeswarm summary plot
    ├── shap_bilstm_waterfall_<pid>.png  ← waterfall per sampel
    ├── shap_bilstm_report.txt
    ├── shap_cnn_bar.png
    ├── shap_cnn_beeswarm.png
    ├── shap_cnn_waterfall_<pid>.png
    └── shap_cnn_report.txt

Cara menjalankan:
  python notebooks/xai_shap_dl.py
  python notebooks/xai_shap_dl.py --model bilstm --n_samples 8
  python notebooks/xai_shap_dl.py --model cnn --n_background 20
"""

import argparse
import logging
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Light theme global ────────────────────────────────────────────────────────
DARK_BG      = "#ffffff"   # figure background → putih
DARK_AX      = "#f7f7f7"   # axes background   → abu sangat muda
DARK_TEXT    = "#111111"   # teks              → hitam
DARK_GRID    = "#dddddd"   # grid lines        → abu muda
ACCENT_RED   = "#d62728"   # merah standar
ACCENT_GREEN = "#2ca02c"   # hijau standar
ACCENT_BLUE  = "#1f77b4"   # biru standar
import matplotlib.cm as cm
import shap
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_XAI        = PROJECT_ROOT / "results" / "xai" / "shap"
RESULTS_XAI_LSTM   = RESULTS_XAI / "LSTM" / "BILSTM_V2"
RESULTS_XAI_CNN    = RESULTS_XAI / "CNN"  / "CNN_V2"
for _p in [RESULTS_XAI_LSTM, RESULTS_XAI_CNN]:
    _p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ── Import model dan data loaders ─────────────────────────────────────────────
try:
    from notebooks.CNN.train_2d_cnn import (
        MelSpectrogram2DCNN,
        load_patient_data as cnn_load_data,
        CLASS_NAMES as CNN_CLASS_NAMES,
        MODEL_DIR as CNN_MODEL_DIR,
        MAX_LEN as CNN_MAX_LEN,
    )
    CNN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"CNN module tidak bisa di-import: {e}")
    CNN_AVAILABLE = False

try:
    from notebooks.LSTM.train_bilstm_mfcc import (
        BiLSTMMFCC,
        load_patient_data as lstm_load_data,
        CLASS_NAMES as LSTM_CLASS_NAMES,
        MODEL_DIR as LSTM_MODEL_DIR,
        MAX_T, N_FEATURES, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    )
    LSTM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BiLSTM module tidak bisa di-import: {e}")
    LSTM_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY BiLSTMMFCC — Arsitektur lama untuk kompatibilitas checkpoint v1
# Keys: input_bn, lstm, attn_pool.attn.weight, fc.0-4
# ══════════════════════════════════════════════════════════════════════════════

class _BiLSTMMFCC_Legacy(nn.Module):
    """
    Arsitektur BiLSTMMFCC versi lama (sebelum refactor).
    Digunakan saat checkpoint tidak cocok dengan arsitektur terbaru.
    """
    def __init__(self, n_features: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        # attn_pool versi lama: satu weight matrix (T, 1)
        self.attn_pool = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        x_bn = self.input_bn(x.reshape(B * T, F)).reshape(B, T, F)
        lstm_out, _ = self.lstm(x_bn)            # (B, T, 2H)
        attn_w = torch.softmax(self.attn_pool(lstm_out), dim=1)  # (B, T, 1)
        pooled = (lstm_out * attn_w).sum(dim=1)  # (B, 2H)
        return self.fc(pooled)

    def load_legacy_state_dict(self, state_dict: dict):
        """Map key lama ke nama field di class ini."""
        new_sd = {}
        for k, v in state_dict.items():
            if k == "attn_pool.attn.weight":
                new_sd["attn_pool.weight"] = v
            else:
                new_sd[k] = v
        self.load_state_dict(new_sd, strict=True)


# ══════════════════════════════════════════════════════════════════════════════
# WRAPPER — raw logits (SHAP DeepExplainer butuh logits, bukan softmax)
# ══════════════════════════════════════════════════════════════════════════════

class CNNWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)   # raw logits


class BiLSTMWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)   # raw logits


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — parse output shap_values() lintas versi SHAP
# ══════════════════════════════════════════════════════════════════════════════

def _parse_shap_output(shap_values, n_classes: int = 2):
    """
    Normalkan output shap_values() ke shape (n_classes, *input_shape).

    SHAP >= 0.46 : ndarray (*input_shape, n_classes)  — last dim = kelas
    SHAP <  0.46 : list[n_classes] of ndarray(*input_shape)
    """
    if isinstance(shap_values, np.ndarray):
        # Pindahkan axis kelas dari last ke first
        # contoh CNN  : (1, 1, H, W, 2) → squeeze batch → (1, H, W, 2) → (2, 1, H, W)
        # contoh LSTM : (1, T, F, 2)    → squeeze batch → (T, F, 2)    → (2, T, F)
        sv = shap_values[0]                          # buang batch dim
        sv = np.moveaxis(sv, -1, 0)                  # kelas ke depan
        return sv                                    # (n_classes, ...)
    elif isinstance(shap_values, list):
        return np.stack([v[0] for v in shap_values], axis=0)
    else:
        raise ValueError(f"Unexpected shap_values type: {type(shap_values)}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — nama fitur MFCC (ramah awam)
# ══════════════════════════════════════════════════════════════════════════════

# Deskripsi singkat tiap koefisien MFCC dalam bahasa awam
_MFCC_DESC = {
    1:  "Energi Suara",          # overall loudness / energy
    2:  "Nada Dasar",            # fundamental pitch shape
    3:  "Kecerahan Suara",       # spectral brightness
    4:  "Kejelasan Ucapan",      # speech clarity / formant 1
    5:  "Resonansi Vokal",       # vocal resonance / formant 2
    6:  "Tekstur Suara-1",
    7:  "Tekstur Suara-2",
    8:  "Tekstur Suara-3",
    9:  "Tekstur Suara-4",
    10: "Tekstur Suara-5",
    11: "Detail Spektrum-1",
    12: "Detail Spektrum-2",
    13: "Detail Spektrum-3",
    14: "Detail Spektrum-4",
    15: "Detail Spektrum-5",
    16: "Detail Spektrum-6",
    17: "Detail Spektrum-7",
    18: "Detail Spektrum-8",
    19: "Detail Spektrum-9",
    20: "Detail Spektrum-10",
}

def _mfcc_label(i: int) -> str:
    """Label awam untuk koefisien MFCC ke-i (1-indexed)."""
    return _MFCC_DESC.get(i, f"Pola Suara-{i}")

def get_feature_names(n_mfcc: int = 40, n_features: int = 120) -> list:
    """
    Nama fitur MFCC dalam bahasa awam:
      - MFCC raw   → deskripsi karakteristik suara
      - Delta      → "Perubahan [nama]"  (dinamika temporal)
      - Delta-delta→ "Kecepatan Ubah [nama]"  (akselerasi)
    """
    raw   = [_mfcc_label(i+1)                    for i in range(n_mfcc)]
    delta = [f"Perubahan {_mfcc_label(i+1)}"     for i in range(n_mfcc)]
    ddelta= [f"Kecepatan Ubah {_mfcc_label(i+1)}" for i in range(n_mfcc)]
    return (raw + delta + ddelta)[:n_features]


# ══════════════════════════════════════════════════════════════════════════════
# SHAP PLOTS — Bar, Beeswarm, Waterfall (menggunakan shap.Explanation)
# ══════════════════════════════════════════════════════════════════════════════

# Teks penjelasan awam yang ditempel di bawah setiap jenis plot
_CAPTION_BAR = (
    "📊  Grafik ini menunjukkan FITUR SUARA mana yang paling sering diperhatikan model.\n"
    "     Batang lebih panjang = fitur tersebut lebih sering menentukan hasil prediksi."
)
_CAPTION_BEESWARM = (
    "🐝  Setiap titik = satu rekaman suara.  Warna merah = nilai fitur tinggi, biru = rendah.\n"
    "     Titik ke kanan (nilai SHAP +) → mendorong prediksi ke DEPRESI.\n"
    "     Titik ke kiri  (nilai SHAP −) → mendorong prediksi ke NORMAL."
)
_CAPTION_WATERFALL = (
    "🌊  Grafik ini menjelaskan prediksi untuk SATU rekaman.\n"
    "     Batang merah → fitur yang mendorong ke arah DEPRESI.\n"
    "     Batang biru  → fitur yang mendorong ke arah NORMAL.\n"
    "     Panjang batang = seberapa kuat pengaruhnya."
)


def _save_shap_fig(save_path: Path, dpi: int = 150):
    """Simpan figure matplotlib aktif lalu tutup."""
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=plt.gcf().get_facecolor())
    plt.close("all")
    logging.info(f"  Saved: {save_path.name}")


def _add_caption(fig: plt.Figure, caption: str):
    """Tambahkan teks caption awam di bagian bawah figure (light theme)."""
    fig.text(
        0.01, 0.01, caption,
        ha="left", va="bottom",
        fontsize=8.5, color="#333333",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                  edgecolor="#555555", alpha=0.95),
    )


def plot_bar(explanation: shap.Explanation, title: str, save_path: Path,
             max_display: int = 20):
    """Bar chart — mean |SHAP| per fitur (global importance)."""
    fig = plt.figure(figsize=(13, max(7, max_display * 0.42)))
    fig.patch.set_facecolor(DARK_BG)
    ax = plt.gca()
    ax.set_facecolor(DARK_AX)

    shap.plots.bar(explanation, max_display=max_display, show=False)

    # Re-style setelah SHAP menggambar
    ax = plt.gca()
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=DARK_TEXT, labelsize=9)
    ax.set_xlabel(
        "Rata-rata pengaruh fitur terhadap prediksi (semakin panjang = semakin penting)",
        fontsize=9, color=DARK_TEXT,
    )
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    # Ganti warna teks label y-axis
    for label in ax.get_yticklabels():
        label.set_color(DARK_TEXT)
    for label in ax.get_xticklabels():
        label.set_color(DARK_TEXT)

    ax.grid(axis="x", linestyle="--", alpha=0.3, color=DARK_GRID)
    plt.title(title, fontsize=13, fontweight="bold", pad=10, color=DARK_TEXT)
    _add_caption(fig, _CAPTION_BAR)
    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    _save_shap_fig(save_path)


def plot_beeswarm(explanation: shap.Explanation, title: str, save_path: Path,
                  max_display: int = 20):
    """Beeswarm / summary plot — distribusi SHAP per fitur."""
    fig = plt.figure(figsize=(13, max(7, max_display * 0.42)))
    fig.patch.set_facecolor(DARK_BG)
    ax = plt.gca()
    ax.set_facecolor(DARK_AX)

    shap.plots.beeswarm(explanation, max_display=max_display, show=False)

    ax = plt.gca()
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=DARK_TEXT, labelsize=9)
    ax.set_xlabel(
        "Nilai pengaruh  (+ = mendorong ke DEPRESI,  − = mendorong ke NORMAL)",
        fontsize=9, color=DARK_TEXT,
    )
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    for label in ax.get_yticklabels():
        label.set_color(DARK_TEXT)
    for label in ax.get_xticklabels():
        label.set_color(DARK_TEXT)

    # Tambah garis vertikal di x=0
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1, alpha=0.7)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=DARK_GRID)

    plt.title(title, fontsize=13, fontweight="bold", pad=10, color=DARK_TEXT)
    _add_caption(fig, _CAPTION_BEESWARM)
    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    _save_shap_fig(save_path)


def plot_waterfall(explanation_single: shap.Explanation, title: str,
                   save_path: Path, max_display: int = 15):
    """Waterfall plot — kontribusi per fitur untuk SATU sampel."""
    fig = plt.figure(figsize=(13, max(7, max_display * 0.50)))
    fig.patch.set_facecolor(DARK_BG)
    ax = plt.gca()
    ax.set_facecolor(DARK_AX)

    shap.plots.waterfall(explanation_single, max_display=max_display, show=False)

    ax = plt.gca()
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=DARK_TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    for label in ax.get_yticklabels():
        label.set_color(DARK_TEXT)
    for label in ax.get_xticklabels():
        label.set_color(DARK_TEXT)

    ax.set_xlabel(
        "← Mendorong ke NORMAL  |  Nilai SHAP (pengaruh fitur terhadap prediksi)  |  Mendorong ke DEPRESI →",
        fontsize=9, color=DARK_TEXT,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=DARK_GRID)

    plt.title(title, fontsize=12, fontweight="bold", pad=10, color=DARK_TEXT)
    _add_caption(fig, _CAPTION_WATERFALL)
    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    _save_shap_fig(save_path)


# ══════════════════════════════════════════════════════════════════════════════
# SHAP BiLSTM — MFCC Sequences
# ══════════════════════════════════════════════════════════════════════════════

def load_lstm_samples(n_background: int = 30, n_explain: int = 8, device: str = "cpu"):
    import random
    random.seed(42)

    _, _, all_files, _ = lstm_load_data()
    random.shuffle(all_files)

    def load_mfcc(path):
        feat = np.load(path)
        if feat.shape[0] > MAX_T:
            feat = feat[:MAX_T, :]
        elif feat.shape[0] < MAX_T:
            feat = np.pad(feat, ((0, MAX_T - feat.shape[0]), (0, 0)), mode="constant")
        if feat.shape[1] != N_FEATURES:
            feat = feat[:, :N_FEATURES] if feat.shape[1] > N_FEATURES else \
                   np.pad(feat, ((0,0),(0, N_FEATURES - feat.shape[1])), mode="constant")
        return feat.astype(np.float32)

    bg_feats  = [load_mfcc(p) for p, _, _ in all_files[:n_background]]
    bg_tensor = torch.tensor(np.stack(bg_feats), dtype=torch.float32)

    explain_samples = []
    for path, pid, cls_idx in all_files[n_background:n_background + n_explain]:
        feat   = load_mfcc(path)
        tensor = torch.tensor(feat[np.newaxis, :, :], dtype=torch.float32)
        explain_samples.append({
            "tensor"     : tensor,
            "feat"       : feat,
            "true_label" : LSTM_CLASS_NAMES[cls_idx],
            "true_idx"   : cls_idx,
            "name"       : f"{pid}_{path.stem[:20]}",
            "pid"        : str(pid),
        })

    logging.info(f"BiLSTM — Background: {len(bg_feats)} | Explain: {len(explain_samples)}")
    return bg_tensor, explain_samples


def run_shap_bilstm(
    model_path   : Optional[Path] = None,
    n_background : int = 30,
    n_samples    : int = 8,
    device_str   : str = "cpu",
):
    if not LSTM_AVAILABLE:
        logging.error("BiLSTM module tidak tersedia.")
        return

    import torch.backends.cudnn as cudnn
    orig_cudnn   = cudnn.enabled
    cudnn.enabled = False

    device = torch.device(device_str)
    logging.info("\n" + "="*60)
    logging.info("SHAP DeepExplainer — BiLSTM MFCC")
    logging.info("="*60)

    if model_path is None:
        model_path = LSTM_MODEL_DIR / "best_bilstm_mfcc.pt"
    if not model_path.exists():
        logging.error(f"Model BiLSTM tidak ditemukan: {model_path}")
        cudnn.enabled = orig_cudnn
        return

    # ── Load checkpoint & auto-detect hyperparameter ─────────────────────────
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg   = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    _n_features = model_cfg.get("n_features", N_FEATURES)
    _hidden_dim = model_cfg.get("hidden_dim", HIDDEN_DIM)
    _num_layers = model_cfg.get("num_layers", NUM_LAYERS)
    _dropout    = model_cfg.get("dropout",    DROPOUT)
    logging.info(
        f"BiLSTM config dari checkpoint: n_features={_n_features}, "
        f"hidden_dim={_hidden_dim}, num_layers={_num_layers}, dropout={_dropout}"
    )

    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    is_legacy_bilstm = "attn_pool.attn.weight" in state_dict

    if is_legacy_bilstm:
        logging.info("Terdeteksi arsitektur BiLSTM LAMA. Menggunakan _BiLSTMMFCC_Legacy...")
        raw_model = _BiLSTMMFCC_Legacy(
            n_features=_n_features, hidden_dim=_hidden_dim,
            num_layers=_num_layers, dropout=_dropout,
        ).to(device)
        raw_model.load_legacy_state_dict(state_dict)
    else:
        logging.info("Arsitektur BiLSTM BARU terdeteksi.")
        raw_model = BiLSTMMFCC(
            n_features=_n_features, hidden_dim=_hidden_dim,
            num_layers=_num_layers, dropout=_dropout,
        ).to(device)
        raw_model.load_state_dict(state_dict)

    raw_model.eval()
    model = BiLSTMWrapper(raw_model)
    logging.info(f"Model BiLSTM loaded: {model_path}")

    feature_names = get_feature_names(n_features=_n_features)
    bg_tensor, explain_samples = load_lstm_samples(n_background, n_samples, device_str)

    logging.info(f"Inisialisasi SHAP DeepExplainer (background: {bg_tensor.shape[0]} sampel)...")
    explainer = shap.DeepExplainer(model, bg_tensor.to(device))

    # Kumpulkan SHAP values semua sampel untuk Bar & Beeswarm global
    all_sv_depresi  = []   # (T, F) per sampel — kelas DEPRESI (idx=1)
    all_feat_flat   = []   # (T*F,) per sampel — input values untuk beeswarm
    report_lines = [
        "=== SHAP DeepExplainer Report — BiLSTM MFCC ===\n\n",
        f"{'Sample':<30} {'True':>8} {'Pred':>8} {'Conf':>7} {'MeanSHAP':>10} {'TopFeature':>15}\n",
        "-" * 80 + "\n",
    ]

    for s in explain_samples:
        tensor     = s["tensor"].to(device)
        true_label = s["true_label"]
        name       = s["name"]
        pid        = s["pid"]
        logging.info(f"  Computing SHAP for: {name} [{true_label}]")

        with torch.no_grad():
            logits    = model(tensor)
            probs     = torch.softmax(logits, dim=-1)
            pred_idx  = int(torch.argmax(probs, dim=-1).item())
            pred_conf = float(probs[0, pred_idx].item())
            pred_label = LSTM_CLASS_NAMES[pred_idx]

        try:
            raw_sv = explainer.shap_values(tensor, check_additivity=False)
            sv     = _parse_shap_output(raw_sv)   # (2, T, F)

            # ── Waterfall untuk sampel ini ────────────────────────────────
            # Aggregasi SHAP per fitur (mean across time) → (F,) untuk waterfall
            sv_dep_feat  = sv[1].mean(axis=0)   # (F,) — kelas DEPRESI
            feat_mean    = s["feat"].mean(axis=0)  # (F,) — nilai rata-rata input

            expl_single = shap.Explanation(
                values        = sv_dep_feat,
                base_values   = float(explainer.expected_value[1]
                                      if hasattr(explainer.expected_value, '__len__')
                                      else explainer.expected_value),
                data          = feat_mean,
                feature_names = feature_names,
            )
            correct = true_label == pred_label
            wf_title = (
                f"Waterfall — BiLSTM | {name}\n"
                f"True: {true_label}  Pred: {pred_label} ({pred_conf:.1%})  "
                f"{'✓' if correct else '✗'}"
            )
            plot_waterfall(
                expl_single,
                title     = wf_title,
                save_path = RESULTS_XAI_LSTM / f"shap_bilstm_waterfall_{pid}.png",
            )

            # Kumpulkan untuk global plots
            all_sv_depresi.append(sv[1].mean(axis=0))   # (F,)
            all_feat_flat.append(feat_mean)              # (F,)

            fi         = np.abs(sv[pred_idx]).mean(axis=0)
            top_feat   = feature_names[int(np.argmax(fi))]
            mean_shap  = float(np.abs(sv[pred_idx]).mean())
            report_lines.append(
                f"{name[:29]:<30} {true_label:>8} {pred_label:>8} {pred_conf:>7.1%} "
                f"{mean_shap:>10.6f} {top_feat:>15}\n"
            )

        except Exception as e:
            logging.warning(f"  SHAP gagal untuk {name}: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Bar & Beeswarm global ─────────────────────────────────────────────────
    if all_sv_depresi:
        sv_matrix   = np.stack(all_sv_depresi, axis=0)   # (N, F)
        feat_matrix = np.stack(all_feat_flat,  axis=0)   # (N, F)

        base_val = float(explainer.expected_value[1]
                         if hasattr(explainer.expected_value, '__len__')
                         else explainer.expected_value)

        expl_global = shap.Explanation(
            values        = sv_matrix,
            base_values   = np.full(len(all_sv_depresi), base_val),
            data          = feat_matrix,
            feature_names = feature_names,
        )

        plot_bar(
            expl_global,
            title      = "SHAP Bar — BiLSTM MFCC (Kelas DEPRESI)",
            save_path  = RESULTS_XAI_LSTM / "shap_bilstm_bar.png",
            max_display= 20,
        )
        plot_beeswarm(
            expl_global,
            title      = "SHAP Beeswarm — BiLSTM MFCC (Kelas DEPRESI)",
            save_path  = RESULTS_XAI_LSTM / "shap_bilstm_beeswarm.png",
            max_display= 20,
        )

    report_path = RESULTS_XAI_LSTM / "shap_bilstm_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    cudnn.enabled = orig_cudnn
    logging.info(f"BiLSTM SHAP selesai. Report: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SHAP CNN — Mel-Spectrogram
# Karena input adalah pixel 2D, kita aggregasi SHAP per freq-band (20 band)
# sehingga bisa ditampilkan sebagai Bar + Beeswarm + Waterfall yang bermakna.
# ══════════════════════════════════════════════════════════════════════════════

def _make_cnn_band_names(n_bands: int = 20, n_mels: int = 128, sr: int = 16000) -> list:
    """
    Nama freq-band CNN dalam bahasa awam.
    Band rendah (< 500 Hz)  → suara dasar / nada bicara
    Band menengah (500–3kHz) → kejelasan ucapan / vokal
    Band tinggi (> 3kHz)    → kecerahan / tekstur suara
    """
    import librosa
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr // 2)
    step  = n_mels // n_bands
    names = []
    for i in range(n_bands):
        lo_hz = freqs[i * step]
        hi_hz = freqs[min((i + 1) * step - 1, n_mels - 1)]
        lo_k  = lo_hz / 1000
        hi_k  = hi_hz / 1000
        # Label awam berdasarkan rentang frekuensi
        if hi_hz < 300:
            label = "Suara Sangat Rendah"
        elif hi_hz < 600:
            label = "Nada Dasar Bicara"
        elif hi_hz < 1200:
            label = "Vokal Rendah"
        elif hi_hz < 2000:
            label = "Kejelasan Ucapan"
        elif hi_hz < 3500:
            label = "Vokal Tinggi"
        elif hi_hz < 5000:
            label = "Kecerahan Suara"
        else:
            label = "Tekstur Suara Tinggi"
        names.append(f"{label} ({lo_k:.1f}-{hi_k:.1f}kHz)")
    return names


def _aggregate_cnn_shap_to_bands(sv_2d: np.ndarray, n_bands: int = 20) -> np.ndarray:
    """
    Aggregasi SHAP (H, W) → (n_bands,) dengan mean |SHAP| per freq-band.
    sv_2d shape: (n_mels, time_frames)
    """
    n_mels = sv_2d.shape[0]
    step   = n_mels // n_bands
    bands  = []
    for i in range(n_bands):
        band_sv = sv_2d[i * step : (i + 1) * step, :]
        bands.append(float(np.abs(band_sv).mean()))
    return np.array(bands)


def load_cnn_samples(n_background: int = 20, n_explain: int = 8, device: str = "cpu"):
    import random
    random.seed(42)

    _, _, all_files, _ = cnn_load_data()
    AUG_TAGS = ["_noise", "_pitch", "_stretch", "_combo"]
    orig_files = [
        (p, pid, cls_idx) for p, pid, cls_idx in all_files
        if p.stem.endswith("_mel") and not any(t in p.stem for t in AUG_TAGS)
    ]
    random.shuffle(orig_files)

    def load_spec(path):
        spec = np.load(path)
        if spec.shape[1] > CNN_MAX_LEN:
            spec = spec[:, :CNN_MAX_LEN]
        else:
            spec = np.pad(spec, ((0,0),(0, CNN_MAX_LEN - spec.shape[1])), mode="constant")
        return spec

    bg_specs  = [load_spec(p) for p, _, _ in orig_files[:n_background]]
    bg_tensor = torch.tensor(
        np.stack(bg_specs)[:, np.newaxis, :, :], dtype=torch.float32
    ).to(device)

    explain_samples = []
    for path, pid, cls_idx in orig_files[n_background:n_background + n_explain]:
        spec   = load_spec(path)
        tensor = torch.tensor(spec[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
        explain_samples.append({
            "tensor"     : tensor,
            "spec"       : spec,
            "true_label" : CNN_CLASS_NAMES[cls_idx],
            "true_idx"   : cls_idx,
            "name"       : f"{pid}_{path.stem}",
            "pid"        : str(pid),
        })

    logging.info(f"CNN — Background: {len(bg_specs)} | Explain: {len(explain_samples)}")
    return bg_tensor, explain_samples


def run_shap_cnn(
    model_path   : Optional[Path] = None,
    n_background : int = 20,
    n_samples    : int = 8,
    device_str   : str = "cpu",
    n_bands      : int = 20,
):
    if not CNN_AVAILABLE:
        logging.error("CNN module tidak tersedia.")
        return

    device = torch.device(device_str)
    logging.info("\n" + "="*60)
    logging.info("SHAP DeepExplainer — 2D CNN Mel-Spectrogram")
    logging.info("="*60)

    if model_path is None:
        model_path = CNN_MODEL_DIR / "best_model_1;1.pt"
    if not model_path.exists():
        logging.error(f"Model CNN tidak ditemukan: {model_path}")
        return

    raw_model = MelSpectrogram2DCNN(num_classes=2).to(device)
    state     = torch.load(model_path, map_location=device, weights_only=True)
    raw_model.load_state_dict(state)
    raw_model.eval()
    model = CNNWrapper(raw_model)
    logging.info(f"Model CNN loaded: {model_path}")

    band_names = _make_cnn_band_names(n_bands=n_bands)
    bg_tensor, explain_samples = load_cnn_samples(n_background, n_samples, device_str)

    logging.info(f"Inisialisasi SHAP DeepExplainer (background: {bg_tensor.shape[0]} sampel)...")
    explainer = shap.DeepExplainer(model, bg_tensor)

    all_sv_bands  = []   # (n_bands,) per sampel
    all_feat_bands= []   # (n_bands,) nilai input per band
    report_lines  = [
        "=== SHAP DeepExplainer Report — CNN Mel-Spectrogram ===\n\n",
        f"{'Sample':<35} {'True':>8} {'Pred':>8} {'Conf':>7} {'MeanSHAP_D':>12} {'MeanSHAP_N':>12}\n",
        "-" * 85 + "\n",
    ]

    for s in explain_samples:
        tensor     = s["tensor"]
        true_label = s["true_label"]
        name       = s["name"]
        pid        = s["pid"]
        logging.info(f"  Computing SHAP for: {name} [{true_label}]")

        with torch.no_grad():
            logits    = model(tensor)
            probs     = torch.softmax(logits, dim=-1)
            pred_idx  = int(torch.argmax(probs, dim=-1).item())
            pred_conf = float(probs[0, pred_idx].item())
            pred_label = CNN_CLASS_NAMES[pred_idx]

        try:
            raw_sv = explainer.shap_values(tensor, check_additivity=False)
            sv     = _parse_shap_output(raw_sv)   # (2, 1, H, W)

            # Aggregasi ke band frekuensi
            sv_dep_2d   = sv[1, 0]   # (H, W) — kelas DEPRESI
            sv_dep_band = _aggregate_cnn_shap_to_bands(sv_dep_2d, n_bands)   # (n_bands,)
            feat_band   = _aggregate_cnn_shap_to_bands(s["spec"], n_bands)   # (n_bands,) input

            # ── Waterfall per sampel ──────────────────────────────────────
            base_val = float(explainer.expected_value[1]
                             if hasattr(explainer.expected_value, '__len__')
                             else explainer.expected_value)
            expl_single = shap.Explanation(
                values        = sv_dep_band,
                base_values   = base_val,
                data          = feat_band,
                feature_names = band_names,
            )
            correct  = true_label == pred_label
            wf_title = (
                f"Waterfall — CNN | {name}\n"
                f"True: {true_label}  Pred: {pred_label} ({pred_conf:.1%})  "
                f"{'✓' if correct else '✗'}"
            )
            plot_waterfall(
                expl_single,
                title     = wf_title,
                save_path = RESULTS_XAI_CNN / f"shap_cnn_waterfall_{pid}.png",
            )

            all_sv_bands.append(sv_dep_band)
            all_feat_bands.append(feat_band)

            mean_shap_d = float(np.abs(sv[1]).mean())
            mean_shap_n = float(np.abs(sv[0]).mean())
            report_lines.append(
                f"{name[:34]:<35} {true_label:>8} {pred_label:>8} {pred_conf:>7.1%} "
                f"{mean_shap_d:>12.6f} {mean_shap_n:>12.6f}\n"
            )

        except Exception as e:
            logging.warning(f"  SHAP gagal untuk {name}: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Bar & Beeswarm global ─────────────────────────────────────────────────
    if all_sv_bands:
        sv_matrix   = np.stack(all_sv_bands,   axis=0)   # (N, n_bands)
        feat_matrix = np.stack(all_feat_bands, axis=0)   # (N, n_bands)
        base_val    = float(explainer.expected_value[1]
                            if hasattr(explainer.expected_value, '__len__')
                            else explainer.expected_value)

        expl_global = shap.Explanation(
            values        = sv_matrix,
            base_values   = np.full(len(all_sv_bands), base_val),
            data          = feat_matrix,
            feature_names = band_names,
        )
        plot_bar(
            expl_global,
            title      = "SHAP Bar — CNN Mel-Spectrogram (Kelas DEPRESI, per Freq-Band)",
            save_path  = RESULTS_XAI_CNN / "shap_cnn_bar.png",
            max_display= n_bands,
        )
        plot_beeswarm(
            expl_global,
            title      = "SHAP Beeswarm — CNN Mel-Spectrogram (Kelas DEPRESI, per Freq-Band)",
            save_path  = RESULTS_XAI_CNN / "shap_cnn_beeswarm.png",
            max_display= n_bands,
        )

    report_path = RESULTS_XAI_CNN / "shap_cnn_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    logging.info(f"CNN SHAP selesai. Report: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SHAP DeepExplainer — Bar, Beeswarm, Waterfall untuk CNN & BiLSTM"
    )
    parser.add_argument("--model", type=str, default="both",
                        choices=["cnn", "bilstm", "both"])
    parser.add_argument("--n_samples",    type=int, default=4,
                        help="Jumlah sampel untuk XAI (default: 4 = 2 NORMAL + 2 DEPRESI)")
    parser.add_argument("--n_background", type=int, default=20)
    parser.add_argument("--cnn_model_path",   type=str, default=None)
    parser.add_argument("--bilstm_model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    logging.info(f"\n{'='*60}")
    logging.info("SHAP DeepExplainer — Mental Health Audio Classification")
    logging.info(f"{'='*60}")
    logging.info(f"Model    : {args.model}")
    logging.info(f"Samples  : {args.n_samples}")
    logging.info(f"BG Size  : {args.n_background}")
    logging.info(f"Device   : {args.device}")

    if args.model in ("cnn", "both"):
        run_shap_cnn(
            model_path   = Path(args.cnn_model_path) if args.cnn_model_path else None,
            n_background = args.n_background,
            n_samples    = args.n_samples,
            device_str   = args.device,
        )

    if args.model in ("bilstm", "both"):
        run_shap_bilstm(
            model_path   = Path(args.bilstm_model_path) if args.bilstm_model_path else None,
            n_background = args.n_background,
            n_samples    = args.n_samples,
            device_str   = args.device,
        )

    logging.info(f"\n✅ SHAP DeepExplainer selesai!")
    logging.info(f"   Semua hasil di: {RESULTS_XAI}")


if __name__ == "__main__":
    main()
