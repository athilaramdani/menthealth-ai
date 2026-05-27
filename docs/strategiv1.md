# Strategi Klasifikasi Biner (2 Kelas) Berbasis PHQ-8

Untuk meningkatkan keandalan model (*accuracy* dan *F1-score*) serta mengatasi ketidakseimbangan kelas (*class imbalance*) akibat sedikitnya jumlah data, proyek ini menggunakan formulasi **Klasifikasi Biner (2 Kelas)**. 

### Logika Pelabelan Biner (Normal vs Depresi)
Pembagian kelas didasarkan pada ambang batas klinis skor PHQ-8 (Patient Health Questionnaire-8) yang standar:

1. **Kelas 0: NORMAL (Non-Depresi)**
   - Kriteria: Skor total PHQ-8/PHQ $\le$ 9.
   - Karakteristik: Mengindikasikan kondisi sehat secara mental atau hanya mengalami distres/stres ringan sehari-hari yang belum masuk kategori klinis depresi.
   - Nilai Ground Truth Biner: `PHQ8_Binary` = 0 atau `PHQ_Binary` = 0.
   - Target Kolom di Kode: `label_depresi` = 0.

2. **Kelas 1: DEPRESI (Depresi Klinis)**
   - Kriteria: Skor total PHQ-8/PHQ $\ge$ 10.
   - Karakteristik: Ambang batas klinis untuk depresi tingkat sedang hingga berat (*moderate-to-severe depression*).
   - Nilai Ground Truth Biner: `PHQ8_Binary` = 1 atau `PHQ_Binary` = 1.
   - Target Kolom di Kode: `label_depresi` = 1.

---

### Keunggulan Strategi Biner:
1. **Keabsahan Data Test (Tanpa Asumsi / Fallback)**:
   - Dataset pengujian resmi (`full_test_split.csv`) hanya memuat kolom `PHQ_Score` dan `PHQ_Binary`. 
   - Dengan pendekatan biner, kita dapat langsung mencocokkan target model dengan ground truth resmi dari dataset tanpa memerlukan data kuisioner individual yang absen pada data uji.
2. **Keseimbangan Kelas yang Sehat**:
   - Menyatukan kelas Normal, Stres, dan Cemas menjadi satu kelas tunggal (Non-Depresi) untuk memperbanyak sampel latihan (Train set: 20 Non-Depresi vs 12 Depresi).
3. **Standar Penelitian Klinis**:
   - Sejalan dengan mayoritas publikasi ilmiah internasional untuk tugas skrining awal (*early screening*) kesehatan mental berbasis audio.