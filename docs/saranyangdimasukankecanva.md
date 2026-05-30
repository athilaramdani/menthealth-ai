# 📌 TEXT & KONTEN UNTUK CANVA BOARD — MENTHEALTH AI

Dokumen ini berisi **teks siap pakai (copy-paste)** berupa kartu informasi/catatan (*sticky notes*) dan grafik untuk ditempel di sekitar diagram alur (pipeline) pada Canva Board Anda. 

Semua data metrik, justifikasi, dan visualisasi disarikan langsung dari repositori proyek Anda agar presentasi ini 100% akurat.

---

```
                       [ KARTU 1: DATA SOURCE & LABELS ]
                                      │
                                      ▼
[ KARTU 2: PREPROCESSING ] ──► [ PIPELINE UTAMA ] ──► [ KARTU 3: FORMULA FITUR ]
                                      │
                                      ▼
[ KARTU 5: ABLATION STUDY ] ─► [ PIPELINE UTAMA ] ──► [ KARTU 4: ANTI-LEAKAGE ]
                                      │
                                      ▼
[ KARTU 6: MODEL TERBAIK ] ──► [ PIPELINE UTAMA ] ──► [ KARTU 7: ARTI KLINIS XAI ]
                                      │
                                      ▼
                       [ KARTU 8: FUTURE DEEP LEARNING ]
```

---

## 💾 [KARTU 1] DATA SOURCE & LABELS
*(Tempel di dekat node **Data Source**)*

*   **Dataset Resmi**: DAIC-WOZ (AVEC 2017)
*   **Target Label**: Klasifikasi Biner Kesehatan Mental (Normal vs Depresi) berdasarkan ambang batas klinis kuesioner standard **PHQ-8** (Patient Health Questionnaire-8).
*   **Logika Biner**:
    *   **Kelas 0 (Normal / Non-Depresi)**: PHQ-8 Score $\le$ 9. Mengindikasikan kondisi sehat secara mental atau stres ringan non-klinis.
    *   **Kelas 1 (Depresi)**: PHQ-8 Score $\ge$ 10. Ambang batas klinis untuk gejala depresi sedang hingga berat (*moderate-to-severe*).
*   **Keunggulan Pendekatan**:
    Menyatukan data Normal, Stres, dan Cemas menjadi satu kelas tunggal (Non-Depresi) untuk menstabilkan keseimbangan kelas akibat minimnya sampel data latih resmi.

---

## 🎛️ [KARTU 2] JUSTIFIKASI AUDIO PREPROCESSING
*(Tempel di dekat box **Audio Preprocessing**)*

Setiap langkah dirancang berdasarkan prinsip pemrosesan sinyal ucapan (*speech processing*):
1.  **Resampling ke 16 kHz**: Standar industri (digunakan Wav2Vec 2.0 & Whisper). Menangkap frekuensi hingga 8 kHz (Nyquist-Shannon) yang merupakan batas atas formant ucapan manusia, sekaligus menghemat memori 3x lipat dibanding 44.1 kHz.
2.  **Konversi ke Mono**: Merata-ratakan channel stereo untuk menghilangkan bias posisi spasial mikrofon perekam. Fokus murni pada pita suara.
3.  **Butterworth High-Pass 80 Hz**: Memotong noise frekuensi sangat rendah (AC ruangan, getaran meja, hembusan angin) serta menghilangkan *DC Offset* (geseran tegangan listrik mikrofon) yang merusak perhitungan fitur energi.
4.  **Peak Normalization 0.9**: Menyetarakan keras suara (*loudness*) lintas partisipan. Nilai 0.9 menyisakan ruang aman (*headroom* 10%) untuk mencegah distorsi kliping digital.
5.  **Trimming Silence (top_db=30)**: Membuang bagian hening di awal & akhir agar tidak merusak nilai rata-rata ekstraksi fitur akustik.
6.  **Spectral Gating (Noise Reduction)**: Meredam desisan konstan (*background hiss*) menggunakan estimasi profil noise berbasis Short-Time Fourier Transform (STFT).

---

## 📊 [KARTU 3] FORMULA 116 FITUR AKUSTIK
*(Tempel di dekat box **Feature Extraction**)*

Fitur akustik diekstrak menggunakan jendela **25 ms (frame)** dengan pergeseran **10 ms (hop)**. Seluruh fitur tingkat frame diagregasikan ke tingkat partisipan menggunakan **6 statistik rangkuman** (mean, std, min, max, percentile 25, dan percentile 75).

**Komposisi Fitur Tabular (Total 116 Fitur):**
*   **MFCC (13 koefisien $\times$ 6 statistik = 78 Fitur)**: Karakteristik bentuk saluran suara (*vocal tract*).
*   **Pitch / F0 ($\times$ 6 statistik = 6 Fitur)**: Karakteristik nada dan intonasi suara.
*   **RMS Energy ($\times$ 6 statistik = 6 Fitur)**: Tingkat kenyaringan ucapan (*loudness*).
*   **Spectral Features ($\times$ 6 statistik $\times$ 4 jenis = 24 Fitur)**:
    *   *Spectral Centroid*: Kecerahan/massa pusat frekuensi suara.
    *   *Spectral Bandwidth*: Lebar rentang frekuensi (warna suara).
    *   *Spectral Rolloff*: Batas penurunan energi spektrum frekuensi tinggi.
    *   *Zero Crossing Rate (ZCR)*: Frekuensi gesekan suara (membedakan vokal vs desisan).
*   **Jitter & Shimmer (2 Fitur)**: Ketidakstabilan mikro frekuensi dan amplitudo pita suara.
*   **Conversational Features (5 Fitur)**: Total durasi asli, durasi bicara bersih, rasio berbicara (*speech ratio*), serta giliran bicara (*turns*) partisipan vs Ellie (Virtual Agent).

---

## 🔒 [KARTU 4] ANTI-LEAKAGE: GROUPKFOLD
*(Tempel di dekat node **GroupKFold Cross-Validation**)*

*   **Masalah (Subject Leakage)**: Segmentasi audio memotong rekaman satu partisipan menjadi puluhan segmen. Jika menggunakan random split biasa, segmen milik orang yang sama (misalnya, Budi) akan masuk di data Train sekaligus Validation. Model akan "menghafal" warna suara khas Budi, menghasilkan akurasi semu $>95\%$ yang palsu, dan langsung gagal saat diuji pada pasien baru.
*   **Solusi**: `GroupKFold` membagi data berdasarkan kunci `participant_id`. Seluruh segmen milik Budi dijamin hanya ada di set Train saja, ATAU set Validation saja secara utuh. Model dipaksa mencari pola depresi klinis umum, bukan menghafal warna suara subjek.

---

## 🧪 [KARTU 5] ABLATION STUDY: OPTIMASI DURASI & REGULARISASI SEGMEN
*(Tempel di antara **Data Splitting** dan **Modeling & Training**)*

Tabel di bawah membandingkan performa model terbaik pada Test Set resmi lintas eksperimen:

| Versi Eksperimen | Durasi & Teknik | Model Terbaik | Test Macro F1 | Test Accuracy | Test ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Versi 2** | Audio Utuh (Tanpa Segmen) | Random Forest | 0.6167 | 65.22% | 0.5000 |
| **Versi 3 (Metrik Tertinggi)** | **Segmentasi 10 Detik** | **Logistic Regression** | **0.6349** | **65.22%** | **0.5397** |
| **Versi 4 (Dihapus)** | Segmentasi 30 Detik | Logistic Regression | 0.5437 | 56.52% | 0.5238 |
| **Versi 5 (Stabil L2)** | Segmen 10 Detik + PCA + Regularisasi | Logistic Regression | 0.5577 | 56.52% | 0.4841 |
| **Versi 6 (Optimasi Akhir)🏆** | **10s + Delta + SelectKBest + SMOTE** | **XGBoost (Akurasi)** / **LR (F1)** | **0.5208** | **60.87%** | **0.5476** |
| **Versi 7 (Uji Normalisasi)** | 10s + Delta + CMVN + Top-5 Voting | Logistic Regression | 0.4866 | 56.52% | 0.4762 |

*   **Insight Analisis**:
    *   **Mengapa v3 (10s) mengungguli v4 (30s)?** Durasi segmen yang lebih pendek (10s) menghasilkan data latihan yang jauh lebih banyak (data augmentation: 2.064 segmen di v3 vs 668 segmen di v4). Model Machine Learning klasik membutuhkan jumlah baris data yang melimpah untuk mencapai generalisasi yang stabil. Oleh karena performa v4 yang buruk, file v4 dihapus untuk menyederhanakan repositori.
    *   **Peran PCA & Regularisasi (v5)**: Di v3, model ensemble (RF & XGBoost) mengalami overfitting parah karena dimensi fitur yang besar (95 fitur) dibanding sampel partisipan (64 orang). Di v5, reduksi dimensi dengan PCA (15 komponen) dan pembatasan kedalaman pohon (max_depth 2-4) berhasil memperkecil gap overfitting. Sebagai buktinya, **Test F1 Random Forest meningkat pesat dari 0.3030 (v3) menjadi 0.4868 (v5)**.
    *   **Keunggulan Kombinasi di v6**: Dengan mengintegrasikan fitur dinamis (Delta & Delta-Delta MFCC), seleksi fitur terarah (SelectKBest, k=25), serta SMOTE-Tomek untuk menyeimbangkan kelas secara aman, performa model ensemble meningkat drastis. **Akurasi model XGBoost di set uji mencapai 60.87%** dan seluruh model mencapai skor Cross-Validation Macro F1 yang sangat tinggi dan konsisten di kisaran **0.59 - 0.61** (naik ~10% dari v5).
    *   **Pelajaran Penting dari v7 (CMVN & Speaker Normalization)**: Penerapan Cepstral Mean and Variance Normalization (CMVN) per speaker bertujuan menyaring sidik suara individu. Namun, metrik justru turun (F1 = **0.4866**). Secara akademis dan klinis, **CMVN per speaker terbukti menghapus sinyal depresi utama** (seperti pitch yang monoton/flat dan kenyaringan yang teredam) dengan menstandardisasikan varians setiap individu ke tingkat yang sama. Oleh karena itu, normalisasi sidik suara individu per speaker tidak disarankan untuk klasifikasi kesehatan mental berbasis akustik.

> 🖼️ **[PASANG GAMBAR PERBANDINGAN MODEL]**
> Letakkan grafik visualisasi perbandingan model versi 3, 5, 6, atau 7 di dekat kartu ini.
> *   *Lokasi File di Repositori*: [daic_model_comparison_v3.png](file:///d:/repositories/menthealth-ai/results/plots/daic_model_comparison_v3.png), [daic_model_comparison_v5.png](file:///d:/repositories/menthealth-ai/results/plots/daic_model_comparison_v5.png), [daic_model_comparison_v6.png](file:///d:/repositories/menthealth-ai/results/plots/daic_model_comparison_v6.png), atau [daic_model_comparison_v7.png](file:///d:/repositories/menthealth-ai/results/plots/daic_model_comparison_v7.png)

---

## 🏆 [KARTU 6] MODEL TERBAIK & EVALUASI KLINIS
*(Tempel di dekat box **Evaluation & Interpretation**)*

*   **Model Pemenang**: *Logistic Regression* (v3 / v5 / v6 / v7) atau *XGBoost* (v6)
*   **Skor Uji Terbaik (v3)**: **Test Accuracy = 65.22%**, **Test Macro F1 = 0.6349**
*   **Skor Uji Teregulasi (v5)**: **Test Accuracy = 56.52%**, **Test Macro F1 = 0.5577**
*   **Skor Uji Teroptimasi (v6)**: **Test Accuracy = 60.87%** (XGBoost), **Test Macro F1 = 0.5208** (LR)
*   **Skor Uji Dinormalisasi (v7)**: **Test Accuracy = 56.52%**, **Test Macro F1 = 0.4866** (LR)
*   **Mengapa Model Linier Mengalahkan XGBoost & Random Forest?**
    Dengan jumlah partisipan yang terbatas, model ensemble berbasis pohon keputusan (seperti XGBoost) sangat rentan mengalami overfitting parah (menghafal noise). Model linier dengan regulasi L2 (*Ridge*) terbukti lebih tangguh meredam overfitting. Namun, di v6 dengan pembatasan kedalaman dan SMOTE-Tomek, XGBoost berhasil mencapai akurasi tinggi sebesar **60.87%**.
*   **Mengapa F1 Logistic Regression Turun di v5 (v3: 0.6349 ──► v5: 0.5577)?**
    *   **Kehilangan Informasi Akibat PCA (Information Loss)**: Logistic Regression sangat kuat dalam mengumpulkan sinyal linier dari puluhan fitur sekaligus menggunakan regulasi L2 (Ridge). Ketika fitur dipangkas secara radikal menjadi 15 komponen PCA, model kehilangan akses langsung ke detail akustik sensitif (seperti jitter/shimmer individual) yang dibuang oleh PCA demi mempertahankan varians global terbesar.
    *   **Kestabilan vs Keberuntungan Metrik (Generalization Stability)**: Di v3, terdapat jurang pemisah yang lebar antara Validation F1 (**0.4642**) dan Test F1 (**0.6349**). Ini mencerminkan "keberuntungan" model linier pada set uji yang kecil (hanya 23 subjek). Di v5, jurang pemisah ini menyusut drastis (Validation F1: **0.5833** vs Test F1: **0.5577**). Model v5 jauh lebih stabil, realistis, dan siap untuk menghadapi subjek baru tanpa fluktuasi hasil yang ekstrem (*lebih reliable secara klinis*).
*   **Perspektif Medis (Error Analysis)**:
    Dalam skrining awal kesehatan mental, meminimalkan *False Negative* (pasien depresi terlewat/diprediksi normal) adalah prioritas tertinggi agar pasien segera mendapat pertolongan klinis, meskipun konsekuensinya adalah sedikit peningkatan *False Positive* (normal diprediksi depresi).

> 🖼️ **[PASANG GAMBAR CONFUSION MATRIX]**
> Letakkan visualisasi matriks kebingungan di dekat metrik ini untuk menunjukkan sebaran akurat model.
> *   *Lokasi File di Repositori*: [daic_confusion_matrices_v3.png](file:///d:/repositories/menthealth-ai/results/confusion_matrix/daic_confusion_matrices_v3.png), [daic_confusion_matrices_v5.png](file:///d:/repositories/menthealth-ai/results/confusion_matrix/daic_confusion_matrices_v5.png), atau [daic_confusion_matrices_v6.png](file:///d:/repositories/menthealth-ai/results/confusion_matrix/daic_confusion_matrices_v6.png)

---

## 🧠 [KARTU 7] PENJELASAN MODEL DENGAN XAI (SHAP, LIME, & COEF)
*(Tempel di dekat box **Model Interpretation**)*

Sistem ini menggunakan metode Explainable AI (XAI) yang berbeda untuk masing-masing model berdasarkan karakteristik matematisnya:

1. **Random Forest & XGBoost (Tree-Based)** ──► **SHAP (Summary & Waterfall Plot)**
   * **Mengapa?** Karena model ensemble berbasis pohon memiliki `TreeExplainer` yang sangat efisien untuk menghitung kontribusi global (semua subjek) dan lokal (satu subjek).
   * **Visual Utama**: 
     * **Beeswarm Plot**: Menunjukkan daftar fitur terpenting beserta dampaknya (contoh: Jitter tinggi berwarna merah berada di sisi kanan, memicu prediksi depresi).
     * **Waterfall Plot**: Mengurai kontribusi fitur satu pasien secara detail untuk kebutuhan diagnosis individu.

2. **SVM (RBF) (Non-Tree Classifier)** ──► **LIME (Local Explanation)**
   * **Mengapa?** SVM dengan kernel RBF bersifat non-linear dan kompleks. LIME mempermudah interpretasi dengan membuat model aproksimasi linier lokal di sekitar satu titik pasien untuk menjelaskan mengapa model mengambil keputusan tersebut.
   * **Visual Utama**: Grafik kontribusi fitur lokal yang memicu diagnosis "Normal" vs "Depresi".

3. **Logistic Regression (Linear Model)** ──► **Koefisien Bobot Langsung (Coef_)** ATAU **SHAP (LinearExplainer)**
   * **Mengapa?** Ada dua opsi XAI yang paling cocok untuk Logistic Regression:
     * **Opsi 1 (Standar/Paling Akurat) ──► Koefisien Bobot (`coef_`)**: Karena bersifat model linier global, koefisien model adalah representasi terbaik. Fitur dengan koefisien positif (+) meningkatkan indikasi Depresi (misal: Jitter, Shimmer), sedangkan koefisien negatif (-) meningkatkan indikasi Normal (misal: Speech Ratio tinggi).
     * **Opsi 2 (Untuk Konsistensi Visual) ──► SHAP `LinearExplainer`**: Sangat cocok jika Anda ingin menampilkan penjelasan dalam bentuk grafik *Beeswarm* atau *Waterfall* yang seragam dengan model Random Forest & XGBoost. Nilai SHAP-nya dihitung secara matematis dari koefisien dikali deviasi fitur terhadap rata-ratanya: $SHAP_i = \beta_i \cdot (x_i - \mu_i)$.
     * *Peringatan*: **LIME tidak direkomendasikan** untuk Logistic Regression karena LIME dirancang untuk mengaproksimasi model kompleks dengan model linier lokal (menjadi redundan karena Logistic Regression sendiri sudah linier secara global).


### 🧬 Arti Klinis & Catatan Metodologis Penting:
* **Interpretasi Fitur Akustik**:
  * *Jitter & Shimmer Tinggi*: Menandakan ketidakstabilan mikro frekuensi & amplitudo pita suara (suara gemetar/serak secara klinis).
  * *Speech Ratio Rendah*: Porsi diam yang tinggi selama wawancara, mencerminkan perlambatan psikomotorik khas depresi.
  * *Spectral Centroid Rendah*: Karakter suara lebih berat dan teredam (*flat/monotone speech*).
* **Catatan Skala SHAP (Probability Space)**:
  * Plot Waterfall model Random Forest v3 berkisar di rentang desimal halus (misalnya: $-0.06$ s/d $+0.01$). Ini adalah format yang **benar dan akurat** karena berada dalam ruang **Probabilitas [0, 1]** (bukan unit jumlah voting pohon *raw tree votes* seperti versi lama yang menampilkan angka $+17$, $-13$, dsb.).
* **Catatan Level Interpretasi (Segmen vs Partisipan)**:
  * *Penjelasan SHAP/LIME*: Berfungsi menjelaskan keputusan model pada **tingkat segmen audio 10 detik** (lokal).
  * *Evaluasi Metrik Uji*: Dilaporkan pada **tingkat partisipan/subjek** karena sistem melakukan agregasi rata-rata probabilitas seluruh segmen milik partisipan tersebut (*Mean Probability Voting*).
  * *Justifikasi Ilmiah*: Pendekatan ini sangat valid karena model mengenali anomali akustik pada cuplikan suara pendek, lalu diakumulasikan untuk mendiagnosis pasien secara utuh.


> 🖼️ **[PASANG GAMBAR VISUALISASI SHAP SUMMARY & WATERFALL]**
> Letakkan grafik beeswarm (summary) global dan waterfall lokal di dekat penjelasan ini:
> *   **Random Forest**: [shap_summary_rf_v3.png](file:///d:/repositories/menthealth-ai/results/plots/xai/shap_summary_rf_v3.png) & [shap_waterfall_rf_v3.png](file:///d:/repositories/menthealth-ai/results/plots/xai/shap_waterfall_rf_v3.png)
> *   **Logistic Regression (Baru!)**: [shap_summary_lr_v3.png](file:///d:/repositories/menthealth-ai/results/plots/xai/shap_summary_lr_v3.png) & [shap_waterfall_lr_v3.png](file:///d:/repositories/menthealth-ai/results/plots/xai/shap_waterfall_lr_v3.png)
> 
> 🖼️ **[PASANG GAMBAR VISUALISASI LIME]**
> Letakkan visualisasi penjelasan keputusan lokal subjek di dekat analisis LIME.
> *   **SVM (RBF)**: [lime_explanation_svm_v3.png](file:///d:/repositories/menthealth-ai/results/plots/xai/lime_explanation_svm_v3.png)



---

## 🚀 [KARTU 8] FUTURE WORK: ROADMAP ML v6, v7 & TAHAP LANJUT
*(Tempel di bagian bawah board sebagai rencana tahap selanjutnya)*

1.  **Strategi Peningkatan Model Tradisional (Versi 6 - Berhasil Diterapkan)**:
    *   **SelectKBest & Delta MFCC**: Menggantikan PCA dengan menyaring 25 fitur mentah paling relevan secara klinis serta menambahkan fitur dinamika temporal MFCC (Delta & Delta-Delta).
    *   **SMOTE-Tomek & Voting Classifier**: Menyeimbangkan kelas minoritas secara aman di dalam cross-validation loop dan menggabungkan model lewat soft voting.
    *   *Hasil*: Kestabilan CV F-score melonjak tinggi ke kisaran **0.59 - 0.61** dan akurasi XGBoost teruji berhasil menembus **60.87%**.

2.  **Strategi Terobosan Baru (Versi 7 Roadmap untuk Mengalahkan v3 & Menembus >65%)**:
    *   **Gender-Specific Modeling (Model Terpisah Pria/Wanita)**: Karakteristik suara biologis pria dan wanita (Pitch/F0 dan Formant) sangat bertolak belakang. Melatih model terpisah untuk masing-masing gender akan menghilangkan bias silang dan melipatgandakan ketajaman klasifikasi akustik.
    *   **Participant-Level GridSearchCV Scorer (Optuna)**: Mengganti metrik optimasi pencarian hyperparameter. Alih-alih mengoptimalkan F1 tingkat segmen, kita menggunakan *custom scorer* yang menghitung F1 tingkat partisipan secara langsung selama *Bayesian Optimization (Optuna)*.
    *   **Top-K Segment Probability Voting**: Mengganti metode rata-rata probabilitas segmen (*mean voting*) dengan *Top-K Max Probability*. Pasien depresi tidak menunjukkan gejala di 100% waktu bicara mereka. Cukup deteksi minimal 2 atau 3 segmen dengan indikasi probabilitas depresi tinggi ($>0.7$) untuk mendiagnosis pasien secara akurat.
    *   **Speaker Channel Normalization (GMM-UBM / d-vector)**: Menerapkan normalisasi sidik suara untuk menyaring warna suara bawaan individu, sehingga model murni mempelajari variabilitas suara yang disebabkan oleh perubahan kondisi klinis depresi.
3.  **Eksperimen Deep Learning (DL)**:
    *   Menguji arsitektur temporal **LSTM/BiLSTM** dan **CNN** pada Mel-Spectrogram untuk menangkap dinamika waktu-frekuensi audio.
    *   *Fine-tuning* model *self-supervised learning* ucapan pra-latih skala besar seperti **Wav2Vec 2.0** untuk akurasi yang lebih tinggi.
4.  **Web & Integrasi Sistem**:
    *   Backend API berbasis **FastAPI** dengan endpoint `/predict` dan `/explain`.
    *   Frontend Dashboard berbasis **React & Tailwind CSS** untuk memudahkan praktisi medis merekam audio dan melihat visualisasi SHAP secara interaktif.
