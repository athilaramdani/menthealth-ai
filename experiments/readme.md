# Arsip Eksperimen & Versioning

Folder ini digunakan sebagai area **arsip (archiving) dan pencatatan riwayat (versioning)** untuk seluruh kode eksperimen, Jupyter Notebook lama, maupun draf skrip preprocessing/modeling yang sudah tidak digunakan lagi secara aktif di folder utama (`notebooks/` atau `preprocessing/`).

## ⚙️ Aturan Penggunaan Folder
1.  **Menjaga Folder Utama Tetap Bersih**: 
    Folder `notebooks/` dan `preprocessing/` di root direktori hanya boleh berisi kode/pipeline yang **aktif digunakan**, bersih, dan siap dijalankan.
2.  **Arsip & Pemindahan**:
    Jika Anda memiliki Jupyter Notebook (`.ipynb`) atau skrip Python (`.py`) lama yang sudah digantikan oleh versi baru (misalnya model SVM lama, modifikasi parameter yang tidak dipakai, draf EDA awal), **pindahkan** file tersebut ke folder `experiments/` ini.
3.  **Beri Format Penamaan yang Jelas**:
    Agar tidak membingungkan anggota tim lain, sangat disarankan untuk memberikan prefix tanggal atau versi saat memindahkan file ke sini. Contoh:
    *   `2026-05-10_daic_eda_draft.ipynb`
    *   `v1_svm_mfcc_model_old.py`

Dengan cara ini, kita tidak akan kehilangan history eksperimen berharga (versioning tetap terjaga), namun repositori utama kita tetap rapi dan terstruktur dengan baik.
