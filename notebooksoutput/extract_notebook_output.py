"""
extract_notebook_output.py
──────────────────────────
Pilih file Jupyter Notebook (.ipynb) via dialog,
ekstrak semua output teks (stdout, stderr, text/plain),
skip output gambar/image, simpan ke folder notebooksoutput/.

Cara pakai:
    python notebooksoutput/extract_notebook_output.py
"""

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

# Fix encoding Windows PowerShell (cp1252 tidak support karakter Unicode)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ─── Konfigurasi ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))   # folder notebooksoutput/

# Tipe output yang DITAMPILKAN (teks saja)
TEXT_OUTPUT_TYPES = {"stream", "execute_result", "display_data", "error"}

# Mime type teks yang diambil
TEXT_MIME_KEYS = ["text/plain", "text/html", "application/json"]

# Mime type gambar yang di-SKIP
IMAGE_MIME_KEYS = ["image/png", "image/jpeg", "image/svg+xml", "image/gif"]


# ─── Fungsi Utama ─────────────────────────────────────────────────────────────

def pick_notebook() -> str | None:
    """Buka dialog pilih file .ipynb. Return path atau None jika cancel."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Pilih Jupyter Notebook (.ipynb)",
        filetypes=[("Jupyter Notebook", "*.ipynb"), ("All Files", "*.*")],
    )
    root.destroy()
    return path if path else None


def lines_to_str(source) -> str:
    """Konversi list baris atau string menjadi satu string."""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def extract_output_block(output: dict) -> str | None:
    """
    Ekstrak teks dari satu output cell.
    Return string teks, atau None jika output hanya gambar (di-skip).
    """
    otype = output.get("output_type", "")

    # ── stream (print / stderr) ───────────────────────────────────────────────
    if otype == "stream":
        text = lines_to_str(output.get("text", ""))
        return text if text.strip() else None

    # ── error / traceback ─────────────────────────────────────────────────────
    if otype == "error":
        ename  = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        tb     = output.get("traceback", [])
        # Hapus ANSI escape codes dari traceback
        clean_tb = []
        for line in tb:
            import re
            line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            clean_tb.append(line_clean)
        return f"[ERROR] {ename}: {evalue}\n" + "\n".join(clean_tb)

    # ── execute_result / display_data ─────────────────────────────────────────
    if otype in ("execute_result", "display_data"):
        data = output.get("data", {})

        # Cek apakah hanya gambar (tidak ada teks)
        has_image = any(k in data for k in IMAGE_MIME_KEYS)
        has_text  = any(k in data for k in TEXT_MIME_KEYS)

        if has_image and not has_text:
            return None   # skip pure-image output

        # Ambil teks jika ada
        for key in TEXT_MIME_KEYS:
            if key in data:
                text = lines_to_str(data[key])
                if text.strip():
                    return text

        return None

    return None


def extract_notebook_outputs(ipynb_path: str) -> str:
    """
    Baca notebook, ekstrak semua output teks cell per cell.
    Return string lengkap yang siap disimpan.
    """
    with open(ipynb_path, "r", encoding="utf-8", errors="replace") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    notebook_name = os.path.basename(ipynb_path)

    lines = []
    lines.append("=" * 80)
    lines.append(f"  NOTEBOOK OUTPUT EXTRACTOR")
    lines.append(f"  File   : {notebook_name}")
    lines.append(f"  Path   : {ipynb_path}")
    lines.append(f"  Diambil: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total  : {len(cells)} cell")
    lines.append("=" * 80)
    lines.append("")

    cell_num   = 0
    code_cells = 0
    has_output = 0

    for cell in cells:
        ctype = cell.get("cell_type", "")

        # ── Markdown cell: tampilkan sebagai header ───────────────────────────
        if ctype == "markdown":
            src = lines_to_str(cell.get("source", "")).strip()
            if src:
                lines.append(f"{'─' * 60}")
                lines.append(f"[MARKDOWN]")
                lines.append(src)
                lines.append("")
            continue

        # ── Code cell ─────────────────────────────────────────────────────────
        if ctype == "code":
            cell_num  += 1
            code_cells += 1
            exec_count = cell.get("execution_count") or "-"
            outputs    = cell.get("outputs", [])

            if not outputs:
                continue   # tidak ada output, skip

            # Kumpulkan semua teks output
            text_parts = []
            skipped_images = 0

            for out in outputs:
                result = extract_output_block(out)
                if result is None:
                    # Cek apakah di-skip karena gambar
                    otype = out.get("output_type", "")
                    if otype in ("execute_result", "display_data"):
                        data = out.get("data", {})
                        if any(k in data for k in IMAGE_MIME_KEYS):
                            skipped_images += 1
                else:
                    text_parts.append(result)

            if not text_parts and skipped_images == 0:
                continue

            # Tulis header cell
            lines.append(f"{'─' * 80}")
            lines.append(f"In [{exec_count}] (Code Cell #{cell_num})")
            lines.append(f"{'─' * 80}")

            if text_parts:
                has_output += 1
                for part in text_parts:
                    lines.append(part.rstrip())
            
            if skipped_images > 0:
                lines.append(f"  [SKIP] {skipped_images} image output(s) tidak ditampilkan.")

            lines.append("")

    # Ringkasan
    lines.append("=" * 80)
    lines.append(f"  RINGKASAN")
    lines.append(f"  Code cells  : {code_cells}")
    lines.append(f"  Cells dengan output teks: {has_output}")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_output(text: str, ipynb_path: str) -> str:
    """Simpan teks ke file .txt di OUTPUT_DIR. Return path file yang disimpan."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stem      = os.path.splitext(os.path.basename(ipynb_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{stem}_output_{timestamp}.txt"
    out_path  = os.path.join(OUTPUT_DIR, filename)

    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)

    return out_path


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Notebook Output Extractor")
    print("  (teks saja — image di-skip)")
    print("=" * 60)

    # Pilih notebook
    if len(sys.argv) > 1:
        ipynb_path = sys.argv[1]
    else:
        ipynb_path = pick_notebook()
        
    if not ipynb_path:
        print("[INFO] Tidak ada file dipilih. Keluar.")
        return

    print(f"\n[INFO] Memproses: {ipynb_path}")

    # Ekstrak
    try:
        text = extract_notebook_outputs(ipynb_path)
    except json.JSONDecodeError as e:
        print(f"[ERROR] File bukan JSON yang valid: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Gagal membaca notebook: {e}")
        return

    # Simpan
    out_path = save_output(text, ipynb_path)

    # Tampilkan preview 30 baris pertama
    preview_lines = text.split("\n")[:30]
    print("\n-- PREVIEW (30 baris pertama) --------------------------")
    print("\n".join(preview_lines))
    print("...")

    print(f"\n[OK] Output disimpan di:\n     {out_path}")

    # Notifikasi dialog
    if len(sys.argv) <= 1:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Selesai",
            f"Output berhasil disimpan!\n\n{out_path}"
        )
        root.destroy()


if __name__ == "__main__":
    main()
