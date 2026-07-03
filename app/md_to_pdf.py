import os
import tkinter as tk
from tkinter import filedialog, messagebox
import markdown
from xhtml2pdf import pisa
import sys
import urllib.request
import base64
import re
import hashlib

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
            
        # Buat folder sementara untuk menyimpan gambar mermaid jika belum ada
        base_path = os.path.dirname(os.path.abspath(file_path))
        mermaid_dir = os.path.join(base_path, "assets")
        if not os.path.exists(mermaid_dir):
            os.makedirs(mermaid_dir)
            
        def process_mermaid(match):
            code = match.group(1).strip()
            # Encode the mermaid code to base64
            encoded = base64.b64encode(code.encode('utf-8')).decode('utf-8')
            url = f"https://mermaid.ink/img/{encoded}"
            
            # Buat nama file unik untuk gambar ini
            img_name = f"mermaid_{hashlib.md5(code.encode('utf-8')).hexdigest()[:8]}.png"
            img_path = os.path.join(mermaid_dir, img_name)
            
            # Download image using urllib dengan User-Agent agar tidak kena 403 Forbidden
            # Cek jika file sudah ada agar tidak perlu download ulang
            if not os.path.exists(img_path):
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                try:
                    with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
                        out_file.write(response.read())
                except Exception as e:
                    print(f"Gagal mendownload diagram mermaid: {e}")
                    return f"*(Mermaid diagram failed to render: {e})*"
            
            # Return markdown image tag
            return f"![Mermaid Diagram](assets/{img_name})"

        # Cari dan ganti semua block mermaid
        md_text = re.sub(r'```mermaid\n(.*?)```', process_mermaid, md_text, flags=re.DOTALL)

        # 3. Konversi MD ke HTML (dengan ekstensi tabel agar rapi)
        html_text = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'nl2br'])

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
            # Resolve relative paths dan pastikan pakai forward slash untuk xhtml2pdf
            path = os.path.normpath(os.path.join(base_path, uri)).replace('\\\\', '/')
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
