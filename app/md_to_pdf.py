import os
import tkinter as tk
from tkinter import filedialog, messagebox
import markdown
from xhtml2pdf import pisa
import sys

def convert_md_to_pdf():
    # Setup Tkinter (hidden root window)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # 1. Buka File Explorer untuk pilih file .md
    file_path = filedialog.askopenfilename(
        title="Pilih file Markdown (.md)",
        filetypes=[("Markdown files", "*.md"), ("All files", "*.*")]
    )

    if not file_path:
        print("Batal: Tidak ada file yang dipilih.")
        return

    try:
        # 2. Baca isi file Markdown
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()

        # 3. Konversi MD ke HTML (dengan ekstensi tabel agar rapi)
        # Menambahkan ekstensi untuk tabel dan fenced code
        html_text = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'nl2br'])

        # Ambil path dasar untuk resolving gambar
        base_path = os.path.dirname(os.path.abspath(file_path))

        # Tambahkan styling dasar CSS agar PDF tidak berantakan
        full_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{
                    size: a4 portrait;
                    @frame content_frame {{
                        left: 50pt; width: 495pt; top: 50pt; height: 742pt;
                    }}
                }}
                body {{ font-family: Arial, sans-serif; line-height: 1.5; font-size: 11pt; }}
                h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 5px; }}
                h2 {{ color: #21618c; border-bottom: 1px solid #d4e6f1; padding-top: 10px; }}
                h3 {{ color: #2874a6; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 6px; text-align: left; }}
                th {{ background-color: #ebf5fb; font-weight: bold; }}
                img {{ max-width: 450pt; height: auto; display: block; margin: 10px auto; }}
                pre {{ background: #f8f9f9; padding: 10px; border: 1px solid #e5e8e8; border-radius: 4px; font-family: Courier, monospace; font-size: 9pt; }}
                blockquote {{ background: #fdfefe; border-left: 5px solid #5499c7; padding: 10px; margin: 10px 0; color: #566573; font-style: italic; }}
                .note {{ color: #117a65; font-weight: bold; }}
                .warning {{ color: #a93226; font-weight: bold; }}
            </style>
        </head>
        <body>
            {html_text}
        </body>
        </html>
        """

        # 4. Tentukan nama file output (.pdf)
        pdf_path = os.path.splitext(file_path)[0] + ".pdf"

        # 5. Proses konversi ke PDF
        # link_callback digunakan untuk membantu xhtml2pdf menemukan file lokal (gambar)
        def link_callback(uri, rel):
            if uri.startswith('http'):
                return uri
            # Resolve relative paths
            path = os.path.join(base_path, uri)
            return path

        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(full_html, dest=pdf_file, link_callback=link_callback)

        if not pisa_status.err:
            print(f"Sukses! PDF disimpan di: {pdf_path}")
            messagebox.showinfo("Berhasil", f"PDF berhasil dibuat:\n{pdf_path}")
        else:
            messagebox.showerror("Error", "Gagal melakukan konversi PDF.")

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    convert_md_to_pdf()
