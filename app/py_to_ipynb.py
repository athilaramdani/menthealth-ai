"""
py_to_ipynb.py
Skrip praktis untuk mengonversi file Python (.py) biasa menjadi Jupyter Notebook (.ipynb).
Seluruh kode Python akan dimasukkan ke dalam satu block Code Cell.
"""

import json
import sys
import os
import tkinter as tk
from tkinter import filedialog

def convert_to_ipynb(py_file, ipynb_file):
    if not os.path.exists(py_file):
        print(f"Error: File {py_file} tidak ditemukan!")
        return

    with open(py_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cells = []
    current_cell_type = 'code'
    current_source = []

    def add_cell():
        if current_source:
            # Hilangkan newline ekstra di ujung source
            if current_source[-1].endswith('\n'):
                current_source[-1] = current_source[-1].rstrip('\n')
                
            cell = {
                "cell_type": current_cell_type,
                "metadata": {},
                "source": current_source.copy()
            }
            if current_cell_type == 'code':
                cell["execution_count"] = None
                cell["outputs"] = []
            cells.append(cell)

    for line in lines:
        if line.startswith('# %%'):
            add_cell()
            current_source = []
            if '[markdown]' in line.lower():
                current_cell_type = 'markdown'
            else:
                current_cell_type = 'code'
        else:
            if current_cell_type == 'markdown':
                # Hapus '#' di awal baris jika ini cell markdown
                if line.startswith('# '):
                    current_source.append(line[2:])
                elif line.startswith('#'):
                    current_source.append(line[1:])
                else:
                    current_source.append(line)
            else:
                current_source.append(line)
                
    add_cell()

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
        else:
            output_file = os.path.splitext(input_file)[0] + '.ipynb'
    else:
        # Buka dialog File Explorer
        root = tk.Tk()
        root.withdraw() # Sembunyikan window utama tkinter
        print("Membuka File Explorer... Silakan pilih file Python (.py)")
        
        # Membawa window tkinter ke depan
        root.attributes('-topmost', True) 
        
        input_file = filedialog.askopenfilename(
            title="Pilih file Python yang ingin di-convert",
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")]
        )
        if not input_file:
            print("Dibatalkan. Tidak ada file yang dipilih.")
            sys.exit(0)
            
        output_file = os.path.splitext(input_file)[0] + '.ipynb'

    convert_to_ipynb(input_file, output_file)
    print(f"Berhasil mengonversi: {input_file} -> {output_file}")
