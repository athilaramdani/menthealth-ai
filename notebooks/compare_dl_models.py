"""
Perbandingan Hasil Terbaik Model Deep Learning
===============================================
Project : Mental Health Audio Classification (NORMAL vs DEPRESI)
Script  : notebooks/compare_dl_models.py

Membandingkan tiga model DL berdasarkan hasil BEST FOLD masing-masing:
  - 2D CNN Mel-Spectrogram
  - BiLSTM MFCC
  - Wav2Vec2 Fine-tuned

Layout plot 2×3:
  [CV Mean F1 ± Std]   [Best Fold Macro F1]   [Best Fold Accuracy]
  [Recall DEPRESI]     [Precision DEPRESI]    [F1 DEPRESI]

Output:
  results/plots/dl_model_comparison.png
  results/metrics/dl_model_comparison.csv

Cara menjalankan:
  python notebooks/compare_dl_models.py
"""

import re
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RESULTS_DIR  = PROJECT_ROOT / "results"
METRICS_DIR  = RESULTS_DIR / "metrics"
PLOTS_DIR    = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI — lokasi file hasil setiap model
# Sesuaikan path jika struktur folder berbeda
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "2D CNN\nMel-Spectrogram": {
        "cv_summary"   : METRICS_DIR / "CNN" / "CNN_v2" / "cv_summary.txt",
        "best_fold_reports": [
            METRICS_DIR / "CNN" / "CNN_v2" / f"report_fold_{i}.txt"
            for i in range(1, 6)
        ],
        "color": "#3498db",
    },
    "BiLSTM\nMFCC": {
        "cv_summary"   : METRICS_DIR / "LSTM" / "cv_summary.txt",
        "best_fold_reports": [
            METRICS_DIR / "LSTM" / f"report_fold_{i}.txt"
            for i in range(1, 6)
        ],
        "color": "#2ecc71",
    },
    "Wav2Vec2\nFine-tuned": {
        "cv_summary"   : METRICS_DIR / "WAV2VEC" / "WAV2VEC_v2" / "wav2vec_cv_summary.txt",
        "best_fold_reports": [
            METRICS_DIR / "WAV2VEC" / "WAV2VEC_v2" / f"wav2vec_report_fold_{i}.txt"
            for i in range(1, 6)
        ],
        "color": "#e74c3c",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PARSER — baca metrik dari file teks
# ══════════════════════════════════════════════════════════════════════════════

def parse_cv_summary(path: Path) -> dict:
    """
    Baca Mean dan Std Macro F1 dari cv_summary.txt.
    Mendukung format CNN/LSTM (Mean Macro F1) dan WAV2VEC (Mean :).
    """
    if not path.exists():
        logging.warning(f"File tidak ditemukan: {path}")
        return {"cv_mean_f1": 0.0, "cv_std_f1": 0.0, "fold_f1s": []}

    text = path.read_text(encoding="utf-8", errors="replace")

    # Ambil semua nilai F1 per fold
    fold_f1s = [float(m) for m in re.findall(r"Fold\s+\d+.*?=\s*([\d.]+)", text)]

    # Mean
    mean_match = re.search(
        r"(?:Mean(?:\s+Macro\s+F1)?\s*[:\|]\s*)([\d.]+)", text, re.IGNORECASE
    )
    cv_mean = float(mean_match.group(1)) if mean_match else (np.mean(fold_f1s) if fold_f1s else 0.0)

    # Std
    std_match = re.search(
        r"(?:Std(?:\s+Macro\s+F1)?\s*[:\|]\s*)([\d.]+)", text, re.IGNORECASE
    )
    cv_std = float(std_match.group(1)) if std_match else (np.std(fold_f1s) if fold_f1s else 0.0)

    return {"cv_mean_f1": cv_mean, "cv_std_f1": cv_std, "fold_f1s": fold_f1s}


def parse_fold_report(path: Path) -> dict:
    """
    Baca metrik dari report_fold_N.txt:
      - Macro F1 (dari header)
      - Accuracy
      - Precision / Recall / F1 per kelas (NORMAL & DEPRESI)
    """
    if not path.exists():
        logging.warning(f"File tidak ditemukan: {path}")
        return {}

    text = path.read_text(encoding="utf-8", errors="replace")

    # Macro F1 dari header
    f1_match = re.search(r"Macro F1[:\s]+([\d.]+)", text, re.IGNORECASE)
    macro_f1 = float(f1_match.group(1)) if f1_match else 0.0

    # Accuracy
    acc_match = re.search(r"accuracy\s+([\d.]+)", text)
    accuracy  = float(acc_match.group(1)) if acc_match else 0.0

    # Per-class metrics — format sklearn classification_report:
    # "  NORMAL       0.72      0.82      0.77        22"
    # "  DEPRESI      0.56      0.42      0.48        12"
    metrics = {}
    for cls in ["NORMAL", "DEPRESI"]:
        pat = rf"{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
        m   = re.search(pat, text)
        if m:
            metrics[f"prec_{cls.lower()}"]   = float(m.group(1))
            metrics[f"recall_{cls.lower()}"] = float(m.group(2))
            metrics[f"f1_{cls.lower()}"]     = float(m.group(3))

    return {
        "macro_f1" : macro_f1,
        "accuracy" : accuracy,
        **metrics,
    }


def find_best_fold(fold_reports: list) -> tuple[int, dict]:
    """
    Cari fold dengan Macro F1 tertinggi.
    Returns: (best_fold_idx_1based, metrics_dict)
    """
    best_idx, best_f1, best_metrics = 0, -1.0, {}
    for i, path in enumerate(fold_reports):
        m = parse_fold_report(path)
        if m.get("macro_f1", 0.0) > best_f1:
            best_f1      = m["macro_f1"]
            best_idx     = i + 1
            best_metrics = m
    return best_idx, best_metrics


# ══════════════════════════════════════════════════════════════════════════════
# KUMPULKAN DATA SEMUA MODEL
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_results() -> pd.DataFrame:
    rows = []
    for model_name, cfg in MODEL_CONFIGS.items():
        cv   = parse_cv_summary(cfg["cv_summary"])
        best_fold_idx, best = find_best_fold(cfg["best_fold_reports"])

        row = {
            "Model"            : model_name,
            "Best Fold"        : best_fold_idx,
            "CV Mean F1"       : round(cv["cv_mean_f1"], 4),
            "CV Std F1"        : round(cv["cv_std_f1"],  4),
            "Best Fold Macro F1" : round(best.get("macro_f1",          0.0), 4),
            "Best Fold Accuracy" : round(best.get("accuracy",          0.0), 4),
            "Recall DEPRESI"   : round(best.get("recall_depresi",      0.0), 4),
            "Precision DEPRESI": round(best.get("prec_depresi",        0.0), 4),
            "F1 DEPRESI"       : round(best.get("f1_depresi",          0.0), 4),
            "Recall NORMAL"    : round(best.get("recall_normal",       0.0), 4),
            "Precision NORMAL" : round(best.get("prec_normal",         0.0), 4),
            "F1 NORMAL"        : round(best.get("f1_normal",           0.0), 4),
        }
        rows.append(row)
        logging.info(
            f"  {model_name.replace(chr(10),' '):<25} | "
            f"Best Fold {best_fold_idx} | "
            f"Macro F1={row['Best Fold Macro F1']:.4f} | "
            f"Acc={row['Best Fold Accuracy']:.4f} | "
            f"Rec-D={row['Recall DEPRESI']:.4f} | "
            f"Prec-D={row['Precision DEPRESI']:.4f}"
        )

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT — 2×3 bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(df: pd.DataFrame, save_path: Path):
    model_labels = [n.replace("\n", "\n") for n in df["Model"].tolist()]
    colors       = [MODEL_CONFIGS[n]["color"] for n in df["Model"].tolist()]
    x            = np.arange(len(model_labels))
    bar_w        = 0.52

    # ── Dark theme constants ──────────────────────────────────────────────────
    BG_DARK  = "#0f0f1a"
    AX_DARK  = "#161625"
    TXT_DARK = "#e0e0e0"
    GRID_CLR = "#2a2a3e"

    # 6 panel: 2 baris × 3 kolom
    panels = [
        {
            "title"  : "CV Mean Macro F1 ± Std",
            "values" : df["CV Mean F1"].tolist(),
            "yerr"   : df["CV Std F1"].tolist(),
            "note"   : "Stabilitas model lintas fold",
            "icon"   : "📊",
        },
        {
            "title"  : "Best Fold Macro F1",
            "values" : df["Best Fold Macro F1"].tolist(),
            "yerr"   : None,
            "note"   : "Puncak performa terbaik",
            "icon"   : "🏆",
        },
        {
            "title"  : "Best Fold Accuracy",
            "values" : df["Best Fold Accuracy"].tolist(),
            "yerr"   : None,
            "note"   : "Akurasi keseluruhan",
            "icon"   : "🎯",
        },
        {
            "title"  : "Recall DEPRESI",
            "values" : df["Recall DEPRESI"].tolist(),
            "yerr"   : None,
            "note"   : "Seberapa banyak kasus depresi terdeteksi\n(↑ lebih baik — false negative berbahaya)",
            "icon"   : "🔍",
        },
        {
            "title"  : "Precision DEPRESI",
            "values" : df["Precision DEPRESI"].tolist(),
            "yerr"   : None,
            "note"   : "Keandalan prediksi depresi",
            "icon"   : "✅",
        },
        {
            "title"  : "F1-Score DEPRESI",
            "values" : df["F1 DEPRESI"].tolist(),
            "yerr"   : None,
            "note"   : "Keseimbangan Recall & Precision\nuntuk kelas DEPRESI",
            "icon"   : "⚖️",
        },
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor(BG_DARK)
    plt.subplots_adjust(top=0.88, bottom=0.14, hspace=0.60, wspace=0.35)

    fig.suptitle(
        "Perbandingan Performa Model Deep Learning — DAIC-WOZ\n"
        "Berdasarkan Hasil Best Fold Masing-masing Model",
        fontsize=15, fontweight="bold", color=TXT_DARK, y=0.96,
    )

    for idx, (ax, panel) in enumerate(zip(axes.flat, panels)):
        vals = panel["values"]
        yerr = panel["yerr"]

        ax.set_facecolor(AX_DARK)

        bars = ax.bar(
            x, vals,
            width     = bar_w,
            color     = colors,
            edgecolor = "#333355",
            linewidth = 0.8,
            alpha     = 0.92,
            yerr      = yerr,
            capsize   = 7 if yerr else 0,
            error_kw  = {"elinewidth": 1.8, "ecolor": "#aaaacc"},
            zorder    = 3,
        )

        # Nilai di atas bar
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(vals) * 0.028),
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TXT_DARK,
            )

        # Highlight bar tertinggi dengan border emas + glow effect
        best_idx_bar = int(np.argmax(vals))
        bars[best_idx_bar].set_edgecolor("#FFD700")
        bars[best_idx_bar].set_linewidth(2.8)

        ax.set_title(
            f"{panel['icon']}  {panel['title']}",
            fontsize=11, fontweight="bold", pad=9, color=TXT_DARK,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=8.5, color=TXT_DARK)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Skor", fontsize=9, color=TXT_DARK)
        ax.tick_params(colors=TXT_DARK, labelsize=8.5)

        # Garis referensi 0.5 dan 0.7
        ax.axhline(0.5, color="#555577", linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)
        ax.axhline(0.7, color="#7777aa", linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)
        ax.text(len(x) - 0.05, 0.502, "0.5", color="#888899", fontsize=7, ha="right", va="bottom")
        ax.text(len(x) - 0.05, 0.702, "0.7", color="#8888aa", fontsize=7, ha="right", va="bottom")

        ax.grid(axis="y", linestyle="--", alpha=0.25, color=GRID_CLR, zorder=1)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        # Catatan kecil di bawah panel
        ax.text(
            0.5, -0.26, panel["note"],
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7.5, color="#9999bb", style="italic",
        )

    # Legend warna model
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=MODEL_CONFIGS[n]["color"], edgecolor="#333355",
              label=n.replace("\n", " "))
        for n in df["Model"].tolist()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True,
        framealpha=0.85,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        labelcolor=TXT_DARK,
        bbox_to_anchor=(0.5, 0.01),
    )

    # Catatan footer
    fig.text(
        0.99, 0.005,
        "★ Border emas = model terbaik pada panel tersebut  |  Garis titik-titik = batas 0.5  |  Garis putus-putus = batas 0.7",
        ha="right", va="bottom", fontsize=7.5, color="#666688", style="italic",
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"Plot disimpan → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TABEL RINGKASAN — cetak ke console
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(df: pd.DataFrame):
    display_cols = [
        "Model", "Best Fold",
        "CV Mean F1", "CV Std F1",
        "Best Fold Macro F1", "Best Fold Accuracy",
        "Recall DEPRESI", "Precision DEPRESI", "F1 DEPRESI",
    ]
    df_disp = df[display_cols].copy()
    df_disp["Model"] = df_disp["Model"].str.replace("\n", " ")

    sep = "=" * 110
    print(f"\n{sep}")
    print(f"{'PERBANDINGAN MODEL DEEP LEARNING — DAIC-WOZ (BEST FOLD)':^110}")
    print(sep)
    print(df_disp.to_string(index=False))
    print(sep)

    # Highlight pemenang per metrik
    metric_cols = [
        "CV Mean F1", "Best Fold Macro F1", "Best Fold Accuracy",
        "Recall DEPRESI", "Precision DEPRESI", "F1 DEPRESI",
    ]
    print("\n🏆  Model terbaik per metrik:")
    for col in metric_cols:
        best_row = df_disp.loc[df[col].idxmax()]
        print(f"   {col:<25} → {best_row['Model']:<30} ({df[col].max():.4f})")
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.info("\n" + "=" * 60)
    logging.info("Perbandingan Model DL — Best Fold")
    logging.info("=" * 60)

    df = collect_all_results()

    # Simpan CSV
    csv_path = METRICS_DIR / "dl_model_comparison.csv"
    df_save  = df.copy()
    df_save["Model"] = df_save["Model"].str.replace("\n", " ")
    df_save.to_csv(csv_path, index=False)
    logging.info(f"Tabel CSV disimpan → {csv_path}")

    # Cetak tabel
    print_summary_table(df)

    # Plot
    plot_path = PLOTS_DIR / "dl_model_comparison.png"
    plot_comparison(df, plot_path)

    logging.info("✅ Selesai.")


if __name__ == "__main__":
    main()
