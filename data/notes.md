# Data Management Notes

> [!IMPORTANT]
> Seluruh folder dan file di dalam direktori `data/` diperbolehkan untuk dimodifikasi, ditambah, atau dihapus **KECUALI** folder `data/raw/`. Folder `raw/` harus tetap murni (immutable) sebagai sumber data asli.

## Persiapan Awal
Karena folder data (`cleaned/`, `features/`, `raw/`, `splits/`) diabaikan oleh Git, Anda perlu menjalankan skrip inisialisasi setelah melakukan clone repository:

1. Masuk ke folder `data/`.
2. Jalankan file `createstruktur.bat`.
3. Folder yang diperlukan akan dibuat secara otomatis jika belum ada.

## Dataset Overview

### 1. DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)
Dataset ini berisi rekaman wawancara klinis yang dirancang untuk membantu diagnosis gangguan psikologis seperti depresi dan kecemasan.

*   **Folder Structure**: Setiap partisipan memiliki folder tersendiri (misal: `300_P`, `301_P`, dll) yang berisi file audio `.wav` dan transkrip wawancara.
*   **Documentation**: 
    *   `DAICWOZDepression_Documentation_AVEC2017.pdf`: Panduan teknis resmi dari kompetisi AVEC 2017.
*   **Metadata & Labels**:
    *   `depresi checker.xlsx`: File rekapitulasi untuk pengecekan label depresi secara cepat.
    *   `train_split_Depression_AVEC2017.csv`, `dev_split_...`, `test_split_...`: File CSV resmi yang menentukan pembagian data untuk pelatihan, validasi, dan pengujian.

### 2. MODMA (Multi-modal Open Dataset for Mental-disorder Analysis)
Dataset multi-modal dari Universitas Lanzhou yang fokus pada analisis depresi melalui berbagai sensor, termasuk audio.

*   **Folder Structure**: Folder partisipan dinamakan berdasarkan ID numerik (misal: `02010001`). Di dalamnya terdapat rekaman audio dari berbagai tugas (seperti membaca atau berbicara bebas).
*   **Documentation**:
    *   `MODMA dataset-a Multi-modal Open Dataset for Mental-disorder Analysis.pdf`: Paper utama yang menjelaskan metodologi pengumpulan data MODMA.
    *   `A Novel Decision Tree for Depression Recognition in Speech.pdf`: Referensi penelitian terkait penggunaan dataset ini untuk klasifikasi depresi.
*   **Metadata**:
    *   `subjects_information_audio_lanzhou_2015.xlsx`: Berisi informasi demografis partisipan (usia, jenis kelamin) serta skor klinis (PHQ-9, SDS, dll) yang digunakan sebagai ground truth.

---

**Alur Kerja Data:**
1.  Ambil data dari `raw/`.
2.  Bersihkan audio dan simpan di `cleaned/`.
3.  Ekstrak fitur dan simpan di `features/` (MFCC, Spectrogram, atau Waveform).
4.  Gunakan metadata dari `raw/` untuk membuat file pembagian data di `splits/`.
