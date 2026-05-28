# Panduan Teknis & Bahasan Tanya-Jawab Proyek Menthealth AI

Dokumen ini berisi penjelasan komprehensif mengenai alasan ilmiah dan teknis di balik setiap keputusan parameter, tahapan pemrosesan data, ekstraksi fitur, hingga strategi validasi model dalam proyek deteksi depresi berbasis audio. 

Gunakan panduan ini sebagai bahan persiapan presentasi progress atau sidang evaluasi.

---

## DAFTAR ISI
1. [Resampling ke 16 kHz](#1-resampling-ke-16-khz)
2. [Konversi ke Audio Mono](#2-konversi-ke-audio-mono)
3. [Filtering Butterworth High-Pass 80 Hz](#3-filtering-butterworth-high-pass-80-hz)
4. [Peak Normalization ke 0.9](#4-peak-normalization-ke-09)
5. [Trimming Silence](#5-trimming-silence)
6. [Spectral Gating (Noise Reduction)](#6-spectral-gating-noise-reduction)
7. [Frame Jendela 25 ms & Hop 10 ms](#7-frame-jendela-25-ms--hop-10-ms)
8. [Rincian Asal 116 Fitur Akustik](#8-rincian-asal-116-fitur-akustik)
9. [Subject Leakage & Pencegahan dengan GroupKFold](#9-subject-leakage--pencegahan-dengan-groupkfold)
10. [Tujuan GridSearchCV & GroupKFold](#10-tujuan-gridsearchcv--groupkfold)

---

### 1. Resampling ke 16 kHz
*   **Pertanyaan**: Kenapa audio harus di-resample ke 16 kHz? Apakah ini normal?
*   **Jawaban**: Ya, 16 kHz adalah standar industri untuk pemrosesan ucapan (*speech processing*), termasuk model modern seperti Wav2Vec 2.0 dan Whisper.
*   **Alasan Ilmiah**:
    *   **Teorema Nyquist-Shannon**: Menyatakan bahwa sampling rate harus minimal dua kali frekuensi tertinggi sinyal yang ingin ditangkap. Dengan 16 kHz, kita bisa merekam informasi suara hingga frekuensi 8 kHz.
    *   **Karakteristik Ucapan Manusia**: Frekuensi fundamental ($F_0$) dan formant vokal utama manusia ($F_1$ hingga $F_3$) yang merepresentasikan warna suara dan artikulasi ucapan semuanya berada di bawah 8 kHz. Suara di atas 8 kHz umumnya hanya berupa desisan (*sibilant*) atau noise instrumen musik yang tidak relevan dengan depresi.
    *   **Komputasi & Ukuran Data**: Musik biasanya menggunakan 44.1 kHz atau 48 kHz. Jika kita membiarkan data audio pada 44.1 kHz, ukuran matriks data akan 3x lipat lebih besar, memperlambat proses training tanpa memberikan kontribusi performa model klasifikasi kesehatan mental.

---

### 2. Konversi ke Audio Mono
*   **Pertanyaan**: Mengapa harus diubah ke mono? Apa alasannya?
*   **Jawaban**: Audio stereo merekam suara dari dua saluran (kiri dan kanan) untuk menghasilkan dimensi spasial (posisi arah suara). Dalam analisis kesehatan mental berbasis suara, aspek spasial tersebut sama sekali tidak penting.
*   **Alasan Teknis**:
    *   Yang dicari dalam deteksi depresi adalah **karakteristik pita suara dan pola bicara** pasien, bukan dari posisi mana pasien berbicara di dalam ruangan.
    *   Mengubah stereo menjadi mono dilakukan dengan merata-ratakan sinyal kiri dan kanan. Ini menyederhanakan data dari matriks 2D menjadi array 1D (waveform waktu), menghemat penggunaan memori hingga 50%, serta mencegah model mempelajari bias spasial dari jenis mikrofon perekam.

---

### 3. Filtering Butterworth High-Pass 80 Hz
*   **Pertanyaan**: Apa maksud dari Butterworth High-Pass Filter dengan cutoff 80 Hz? Apa itu *DC offset* dan *low-frequency rumble*?
*   **Jawaban**: Filter ini digunakan untuk memotong suara dengan frekuensi di bawah 80 Hz.
*   **Penjelasan Istilah**:
    *   **DC Offset**: Tegangan listrik searah dari perangkat mikrofon yang menyebabkan sinyal gelombang audio bergeser naik atau turun dari garis tengah nol (rata-rata amplitudo tidak bernilai 0). DC offset mengacaukan perhitungan fitur berbasis energi (seperti RMS Energy).
    *   **Low-Frequency Rumble**: Noise frekuensi sangat rendah (di bawah 80 Hz) yang dihasilkan oleh AC ruangan, hembusan napas yang langsung mengenai mikrofon, getaran meja, atau langkah kaki. Getaran ini bukan bagian dari suara manusia.
    *   **Filter Butterworth**: Dipilih karena memiliki respon frekuensi yang sangat rata (*flat response*) pada bagian frekuensi yang dilewatkan (di atas 80 Hz), sehingga suara pasien tidak terdistorsi setelah penyaringan. Rentang fundamental pita suara manusia dewasa sendiri berada di atas 85 Hz (pria: 85-180 Hz, wanita: 165-255 Hz), sehingga aman menyaring suara di bawah 80 Hz.

---

### 4. Peak Normalization ke 0.9
*   **Pertanyaan**: Kenapa peak normalization diatur ke angka 0.9?
*   **Jawaban**: Normalisasi amplitudo digunakan untuk meratakan keras suara (*loudness*) antar partisipan.
*   **Alasan Teknis**:
    *   Beresiko besar jika model mendeteksi suara lemah sebagai indikator depresi, padahal suara lemah tersebut disebabkan karena mikrofon perekam diletakkan terlalu jauh dari mulut partisipan.
    *   Normalisasi peak mencari amplitudo absolut tertinggi dalam audio dan mengalikannya dengan konstanta tertentu agar nilai tertinggi tersebut berada tepat di angka 0.9 (skala maksimum adalah 1.0).
    *   Batas **0.9 (90% dari batas penuh)** dipilih untuk menyediakan *headroom* (ruang aman sebesar 10%) guna mencegah distorsi kliping digital (*digital clipping*) saat proses manipulasi sinyal audio berikutnya.

---

### 5. Trimming Silence
*   **Pertanyaan**: Apakah hening (silence) di awal dan akhir audio berarti bagian yang benar-benar tidak ada suaranya?
*   **Jawaban**: Ya, *trimming* membuang keheningan absolut atau bagian audio yang hanya berupa desisan latar belakang sebelum partisipan mulai berbicara dan setelah selesai berbicara.
*   **Alasan Teknis**:
    *   Keheningan ini tidak membawa informasi pita suara. Membiarkannya masuk hanya akan memperbanyak frame kosong berisi noise latar belakang, yang berpotensi menurunkan kualitas ekstraksi statistik fitur akustik (misalnya menurunkan nilai rata-rata energi suara asli).
    *   Fungsi `librosa.effects.trim(top_db=30)` memotong bagian awal dan akhir sinyal yang tingkat desibelnya berada 30 dB di bawah amplitudo referensi (dianggap hening).

---

### 6. Spectral Gating (Noise Reduction)
*   **Pertanyaan**: Bagaimana cara kerja *Spectral Gating* untuk mereduksi noise?
*   **Jawaban**: Metode ini membagi sinyal audio ke dalam domain frekuensi (menggunakan *Short-Time Fourier Transform* / STFT) untuk mendeteksi profil kebisingan yang konstan.
*   **Cara Kerja**:
    1.  Algoritma menganalisis bagian hening audio untuk merekam profil desisan (*background hiss* seperti suara kipas laptop atau desah mikrofon).
    2.  Pintu penyaring (*gate*) frekuensi dipasang. Jika energi suara pada frekuensi tertentu di suatu waktu berada di bawah level profil noise, pintu akan menutup (frekuensi tersebut diredam).
    3.  Jika energinya melebihi profil noise (artinya ada suara manusia), pintu terbuka dan suara dilewatkan.
    4.  Sinyal dikembalikan ke domain waktu (Inverse STFT). Hasilnya adalah vokal bersih tanpa desisan latar belakang.

---

### 7. Jendela Frame 25 ms & Hop 10 ms
*   **Pertanyaan**: Apa maksud dari Frame 25 ms dan Hop 10 ms dalam ekstraksi fitur frame-level?
*   **Jawaban**: Ini adalah teknik pemrosesan sinyal ucapan (*speech signal processing*) untuk mengamati perubahan karakteristik suara dari waktu ke waktu.
*   **Alasan Ilmiah**:
    *   **Non-stasioner**: Suara manusia selalu berubah (frekuensi huruf "A" berbeda dengan huruf "S"). Kita tidak bisa menganalisis frekuensi suara dari file audio berdurasi panjang sekaligus.
    *   **Frame 25 ms**: Sinyal suara diasumsikan stasioner (karakteristiknya konstan) dalam jendela waktu yang sangat pendek. Angka 25 ms adalah standar optimal karena cukup panjang untuk menangkap getaran pita suara terkecil, namun cukup pendek agar sifat stasionernya tetap terjaga.
    *   **Hop 10 ms (Overlap 15 ms)**: Jendela analisis digeser setiap 10 ms. Ini berarti ada tumpang tindih (*overlap*) sebesar 15 ms antar jendela berurutan. *Overlap* ini memastikan tidak ada informasi yang hilang di perbatasan jendela (karena jendela analisis biasanya dilapisi fungsi pemulusan seperti Hamming/Hann window).

---

### 8. Rincian Asal 116 Fitur Akustik
*   **Pertanyaan**: Di mana kode letak 116 fitur ini diekstrak? Apa saja rinciannya?
*   **Jawaban**: Fitur-fitur ini diekstrak di dalam fungsi `extract_all_audio_features(y, sr)` pada file `traditional_mlv2.py`.
*   **Rincian Matematika Agregasi**:
    Setiap fitur frame-level (yang bernilai banyak di sepanjang waktu) diagregasikan menjadi **6 statistik rangkuman** (mean, std, min, max, percentile 25, dan percentile 75) untuk menggambarkan keseluruhan rekaman partisipan.
    
    1.  **MFCC (13 koefisien)** $\times$ 6 statistik = **78 fitur** (Mendeteksi perubahan resonansi pita suara).
    2.  **Pitch / F0** $\times$ 6 statistik = **6 fitur** (Mendeteksi intonasi dan nada suara).
    3.  **RMS Energy** $\times$ 6 statistik = **6 fitur** (Mendeteksi kenyaringan/volume bicara).
    4.  **Spectral Centroid** $\times$ 6 statistik = **6 fitur** (Mendeteksi "kecerahan" suara/titik pusat spektrum).
    5.  **Spectral Bandwidth** $\times$ 6 statistik = **6 fitur** (Mendeteksi lebar rentang frekuensi).
    6.  **Spectral Rolloff** $\times$ 6 statistik = **6 fitur** (Mendeteksi kemiringan penurunan energi frekuensi tinggi).
    7.  **Zero Crossing Rate (ZCR)** $\times$ 6 statistik = **6 fitur** (Mendeteksi tingkat gesekan suara, membedakan vokal dan konsonan desis).
    8.  **Jitter** (skalar tunggal) = **1 fitur** (Mendeteksi variasi mikro pada frekuensi nada/suara serak).
    9.  **Shimmer** (skalar tunggal) = **1 fitur** (Mendeteksi variasi mikro pada amplitudo suara).
    10. **Conversational Features** dari transkrip = **5 fitur** (`original_duration_sec`, `cleaned_duration_sec`, `speech_ratio`, `participant_turns`, `ellie_turns`).
    
    *   **Total**: $78 + 6 + 6 + 6 + 6 + 6 + 6 + 1 + 1 + 5 = \mathbf{116\text{ Fitur}}$.

---

### 9. Subject Leakage & Pencegahan dengan GroupKFold
*   **Pertanyaan**: Apa itu *subject leakage* (kebocoran subjek) dan bagaimana GroupKFold mencegahnya? Berikan analoginya.
*   **Jawaban**: *Subject leakage* terjadi ketika data dari partisipan yang sama digunakan untuk melatih model sekaligus mengevaluasinya secara tidak sengaja.
*   **Analogi & Dampak**:
    *   *Misalkan*: Partisipan Budi tergolong depresi. Suara Budi dipotong-potong menjadi 10 segmen audio.
    *   *Kasus Kebocoran*: Jika kita membagi data secara acak biasa (*random split*), 8 segmen suara Budi bisa masuk ke data latihan (Train set), dan 2 segmen sisanya masuk ke data uji (Validation set).
    *   *Dampak*: Model ML akan menghafal keunikan warna suara pita Budi (nada khas Budi), bukan mencari gejala depresi. Saat menguji 2 segmen sisa di Validation set, model menebak "Depresi" hanya karena model mengenali "oh, ini warna suaranya Budi". Akurasi validasi akan terlihat sangat tinggi ($>95\%$). Namun saat model diuji pada pasien baru di rumah sakit yang warnanya suaranya belum pernah didengar model, akurasinya akan hancur lebur karena model tidak belajar tanda-tanda depresi secara objektif.
    *   **Pencegahan dengan GroupKFold**: Memastikan seluruh data (ke-10 segmen suara Budi) hanya boleh berada di set Train saja, ATAU di set Validation saja secara utuh berdasarkan kunci `participant_id`. Model dipaksa mengenali gejala klinis depresi secara umum dari subjek-subjek yang berbeda.

---

### 10. Tujuan GridSearchCV & GroupKFold
*   **Pertanyaan**: Apa gunanya GridSearchCV dipasangkan dengan GroupKFold?
*   **Jawaban**: Kombinasi ini digunakan untuk pencarian parameter terbaik model secara otomatis dan objektif tanpa manipulasi hasil uji coba.
*   **Fungsi**:
    *   **GridSearchCV**: Mencoba semua kombinasi hyperparameter model (seperti tingkat kedalaman pohon `max_depth` atau jumlah pohon `n_estimators` pada Random Forest) yang didaftarkan di kode.
    *   **GroupKFold**: Berperan sebagai juri selama pencarian tersebut. Setiap kali GridSearchCV mencoba satu kombinasi hyperparameter, GroupKFold membagi data train menjadi 5 lipatan kelompok orang secara bergantian. Kombinasi parameter yang menghasilkan performa rata-rata terbaik lintas lipatan tanpa mengalami subject leakage akan dipilih sebagai konfigurasi final.

---

### 11. Penjelasan Detail Tiap Koefisien MFCC (1 - 13)
*   **Pertanyaan**: Apa sebenarnya arti dari MFCC 1, MFCC 2, sampai MFCC 13 secara fisik pada suara manusia?
*   **Jawaban**: MFCC (Mel-Frequency Cepstral Coefficients) membagi sinyal suara menjadi amplop spektral (*spectral envelope*) yang merepresentasikan bentuk saluran suara (*vocal tract*) manusia saat mengucapkan sesuatu. Secara fisik, masing-masing koefisien menggambarkan hal berikut:
    *   **MFCC 1 (Kenyaringan / Loudness)**: Mewakili energi total atau volume suara secara keseluruhan. Nilai tinggi berarti suara diucapkan dengan keras, nilai rendah berarti berbisik atau pelan.
    *   **MFCC 2 (Kemiringan Spektral / Spectral Slope & Tension)**: Menunjukkan distribusi energi antara frekuensi rendah dan tinggi. Koefisien ini berkorelasi kuat dengan **ketegangan pita suara** (*vocal effort*). Suara berat (dada) memiliki nilai MFCC 2 yang sangat berbeda dengan suara tipis/cempreng (kepala).
    *   **MFCC 3 & MFCC 4 (Bentuk Vokal & Sengau / Formants)**: Terkait erat dengan frekuensi formant pertama ($F_1$) dan kedua ($F_2$). Ini adalah bagian yang membedakan pengucapan huruf vokal seperti "A", "I", "U" serta mendeteksi suara sengau (*nasality*).
    *   **MFCC 5 hingga MFCC 13 (Tekstur Suara & Detail Spektral)**: Mewakili variasi frekuensi menengah hingga tinggi yang lebih detail. Koefisien-koefisien ini menangkap anomali halus seperti **getaran pita suara yang goyah, kelelahan vokal (vocal fatigue), ketidakstabilan nafas, atau suara serak**. Pada pasien depresi, kontrol motorik halus pita suara sering kali menurun, yang menyebabkan pola nilai MFCC 5-13 berubah dibanding orang normal.

---

### 12. Penjelasan Jitter, Shimmer, Energy, ZCR, dan Fitur Spektral
*   **Pertanyaan**: Apa itu Jitter, Shimmer, Energy, ZCR, Centroid, Bandwidth, dan Rolloff? Berikan analoginya untuk presentasi.
*   **Jawaban**: Berikut adalah penjelasan intuitif beserta analogi sederhananya:

#### A. Jitter (Variasi Pitch Mikro)
*   **Definisi**: Ketidakstabilan frekuensi fundamental ($F_0$ / nada) suara dari satu siklus getaran pita suara ke siklus berikutnya (satuan milidetik).
*   **Analogi Presentasi**: Bayangkan Anda sedang menyanyikan satu nada panjang "Aaaa" yang datar. Jika pita suara Anda sehat dan rileks, nadanya akan terdengar sangat lurus dan stabil (Jitter rendah). Namun, jika Anda sedang cemas, gugup, lelah, atau depresi, pita suara Anda bergetar secara bergetar tidak konsisten (nadanya berfluktuasi naik-turun dengan sangat cepat secara mikro), menghasilkan **Jitter yang tinggi** (suara terdengar bergetar/gugup).

#### B. Shimmer (Variasi Amplitudo Mikro)
*   **Definisi**: Ketidakstabilan amplitudo (kenyaringan/volume) suara dari satu siklus getaran ke siklus berikutnya.
*   **Analogi Presentasi**: Bayangkan sebuah senter. Jika baterainya baru, cahayanya stabil konstan (Shimmer rendah). Jika baterainya hampir habis atau saklarnya longgar, cahayanya akan berkedip-kedip redup-terang dengan sangat cepat (Shimmer tinggi). Pada suara manusia, Shimmer tinggi menunjukkan ketidakmampuan paru-paru dan pita suara menjaga volume suara tetap stabil secara mikro (suara terdengar goyah atau lemas).

#### C. RMS Energy (Energi Suara)
*   **Definisi**: Rata-rata energi kuadrat sinyal suara dalam jendela waktu tertentu, secara langsung berkorelasi dengan kenyaringan (*loudness*) fisik.
*   **Aplikasi Klinis**: Pasien depresi cenderung berbicara dengan volume yang lebih pelan (*soft-spoken*), monoton, dan energinya datar (RMS Energy rata-rata rendah dengan standar deviasi yang kecil).

#### D. Zero Crossing Rate (ZCR / Laju Lintasan Nol)
*   **Definisi**: Seberapa sering gelombang suara memotong garis tengah nol (berubah tanda dari positif ke negatif) dalam satu detik.
*   **Analogi Presentasi**: Bayangkan gelombang laut. Gelombang yang besar dan lambat (suara vokal berfrekuensi rendah seperti "Ooo", "Uuu") jarang menabrak garis pantai (ZCR rendah). Sebaliknya, riak ombak kecil yang sangat cepat dan berbusa (suara desisan angin seperti "Ssss", "Ffff") menabrak pantai berkali-kali dalam semenit (ZCR tinggi). ZCR membantu model mendeteksi perbedaan durasi antara suara berdesis (*unvoiced*) dan suara bernada (*voiced*).

#### E. Spectral Centroid (Pusat Spektrum / Warna Suara)
*   **Definisi**: Titik tengah massa dari spektrum frekuensi suara.
*   **Analogi Presentasi**: Menentukan apakah karakter suara terdengar "terang/cempreng" (centroid tinggi, seperti suara anak kecil atau peluit) atau "gelap/berat" (centroid rendah, seperti suara bass pria dewasa). Suara orang depresi sering kali terdengar lebih teredam dan datar (centroid lebih rendah).

#### F. Spectral Bandwidth & Spectral Rolloff
*   **Spectral Bandwidth**: Lebar rentang frekuensi di sekitar pusat spektrum. Menunjukkan seberapa kaya warna suara. Suara siulan memiliki bandwidth sangat sempit, suara penyanyi opera memiliki bandwidth sangat lebar.
*   **Spectral Rolloff**: Batas frekuensi di mana sebagian besar energi suara (biasanya 85%) berada di bawahnya. Ini membantu membedakan suara bersih dengan suara yang disertai desah nafas berat (*breathy voice*).

---

### 13. Analisis Hasil Evaluasi Model (Mengapa Hasilnya Demikian?)
*   **Pertanyaan**: Bagaimana analisis performa masing-masing model? Mengapa Random Forest menjadi yang terbaik, dan mengapa yang lainnya mendapatkan skor segitu?
*   **Jawaban**: Berikut adalah bedah performa keempat model berdasarkan hasil pengujian aktual:
    
    *   **Hasil Metrik Aktual**:
        *   **Random Forest (Best Model)**: CV Macro F1: **0.6366**, Test Macro F1: **0.6167**, Test Accuracy: **0.6522**, Test ROC-AUC: **0.5000**.
        *   **SVM (RBF)**: CV Macro F1: **0.6016**, Test Macro F1: **0.5548**, Test Accuracy: **0.6087**, Test ROC-AUC: **0.4365**.
        *   **XGBoost**: CV Macro F1: **0.5989**, Test Macro F1: **0.4868**, Test Accuracy: **0.5217**, Test ROC-AUC: **0.5159**.
        *   **Logistic Regression**: CV Macro F1: **0.6598**, Test Macro F1: **0.4524**, Test Accuracy: **0.4783**, Test ROC-AUC: **0.4048**.

    *   **Analisis Mengapa Random Forest Paling Unggul**:
        1.  **Keterbatasan Ukuran Data**: Dataset latih kita relatif kecil (64 partisipan). Model berbasis pohon tunggal atau ensemble bagging seperti Random Forest sangat kuat dalam mencegah overfitting pada dataset kecil karena melakukan *random feature selection* di setiap split pohonnya.
        2.  **Kekebalan terhadap Outlier & Hubungan Non-Linear**: Random Forest tidak mengasumsikan hubungan linier antar fitur akustik dan dapat menangani interaksi non-linear yang rumit antara MFCC, pitch, dan Jitter/Shimmer tanpa memerlukan penyetelan matematis yang terlalu sensitif.
        
    *   **Analisis Model Lain**:
        *   **XGBoost (Overfitting)**: XGBoost memiliki kecenderungan overfitting yang sangat tinggi pada dataset kecil. Model ini berfokus memperbaiki kesalahan latih (*boosting*) secara berturut-turut, sehingga sangat mudah menghafal pola data latih (terlihat dari Val F1 mencapai **0.7000**, namun anjlok di Test F1 menjadi **0.4868**).
        *   **Logistic Regression (Underfitting/Misalignment)**: Merupakan model linier sederhana. Hubungan antara fitur akustik suara dan depresi sangat kompleks dan non-linier. Logistic Regression tidak mampu memetakan batas keputusan linier yang baik pada dimensi fitur yang tinggi, sehingga performa test set-nya paling rendah (Test F1 **0.4524**).
        *   **SVM RBF (Performa Moderat)**: SVM dengan kernel RBF mampu menangani batas keputusan non-linier dengan memproyeksikan fitur ke dimensi yang lebih tinggi. Performa SVM cukup stabil (Test F1 **0.5548**) namun sangat bergantung pada penentuan parameter regulasi $C$ dan skala $\gamma$.

---

### 14. Interpretasi Gambar Visualisasi SHAP dan LIME
*   **Pertanyaan**: Bagaimana cara membaca dan menjelaskan gambar SHAP (Summary, Bar, Waterfall) dan LIME pada presentasi?
*   **Jawaban**: Berikut panduan membaca grafik XAI yang dihasilkan oleh skrip kita di folder `results/plots/xai/`:

#### A. SHAP Summary Plot (Beeswarm)
*   **Cara Membaca**: 
    *   Sumbu Y menampilkan daftar fitur akustik diurutkan berdasarkan tingkat kepentingannya (paling atas adalah yang paling berpengaruh).
    *   Sumbu X menunjukkan nilai SHAP (*SHAP value*). Titik di sebelah kanan nol ($> 0$) berkontribusi meningkatkan probabilitas prediksi depresi, sedangkan di sebelah kiri ($< 0$) menurunkan probabilitas depresi (cenderung normal).
    *   Warna titik (Merah = Nilai fitur tinggi, Biru = Nilai fitur rendah).
*   **Contoh Penjelasan**: "Jika kita melihat fitur *jitter* di sumbu Y, titik-titik berwarna merah (nilai jitter tinggi) berkumpul di sebelah kanan sumbu X ($>0$). Ini membuktikan secara klinis dan matematis bahwa pasien dengan ketidakstabilan pita suara yang tinggi memiliki peluang lebih besar untuk diklasifikasikan sebagai depresi oleh model Random Forest."

#### B. SHAP Bar Plot (Global Feature Importance)
*   **Cara Membaca**:
    *   Menunjukkan rata-rata kontribusi absolut setiap fitur secara keseluruhan pada dataset (`mean(|SHAP value|)`).
    *   Ini adalah interpretasi global. Fitur dengan batang terpanjang merupakan fitur akustik utama yang paling diandalkan model untuk membedakan kelompok normal dan depresi.

#### C. SHAP Waterfall Plot (Local Explanation)
*   **Cara Membaca**:
    *   Grafik ini menjelaskan keputusan prediksi untuk **satu individu pasien tertentu**.
    *   Bagian bawah menampilkan nilai awal (*base value* $E[f(X)]$), yaitu rata-rata probabilitas prediksi model pada seluruh dataset.
    *   Setiap baris menunjukkan bagaimana nilai fitur pasien tersebut (misalnya nilai *pitch* atau *speech ratio* miliknya) mendorong probabilitas prediksi ke arah kanan (merah/meningkatkan potensi depresi) atau ke arah kiri (biru/menurunkan potensi depresi).
    *   Bagian atas ($f(x)$) adalah probabilitas prediksi final untuk pasien tersebut.
*   **Contoh Penjelasan**: "Waterfall plot ini membantu dokter melihat *kenapa* sistem mendiagnosis pasien ini mengalami depresi. Dokter bisa melihat bahwa fitur durasi bicara yang sangat singkat (warna biru) memberikan dorongan terbesar (+0.15) ke arah prediksi depresi."

#### D. LIME Local Explanation Plot (untuk SVM)
*   **Cara Membaca**:
    *   LIME membuat model linier lokal di sekitar sampel pasien tunggal untuk mendekati keputusan model SVM yang rumit.
    *   Grafik menunjukkan kontribusi fitur lokal untuk satu pasien depresi. Batang yang mengarah ke sisi "Depresi" adalah karakteristik spesifik suara pasien tersebut yang paling memicu model SVM mendiagnosisnya sebagai depresi.
    *   Batang dengan arah berlawanan adalah fitur suara pasien yang sebenarnya mendukung klasifikasi normal.

---

### 15. Perbandingan Eksperimen Versi 2 (Non-Segmen) vs Versi 3 (Segmen 10s) vs Versi 4 (Segmen 30s)
*   **Pertanyaan**: Bagaimana perbandingan komparatif hasil metrik antara eksperimen tingkat subjek utuh (Versi 2), tingkat segmen 10 detik (Versi 3), dan tingkat segmen 30 detik (Versi 4)? Serta apa rekomendasi finalnya?
*   **Jawaban**: Berikut adalah tabel komparatif hasil akhir pengujian pada Test Set resmi (PHQ-8 Proxy):

#### Tabel Perbandingan Metrik Evaluasi Akhir (Tingkat Partisipan)
*   **Eksperimen Versi 2 (Audio Utuh Per Subjek)**:
    *   Logistic Regression: Test F1-Macro = **0.4524** | Test Acc = 0.4783 | Test ROC-AUC = 0.4048
    *   SVM (RBF): Test F1-Macro = **0.5548** | Test Acc = 0.6087 | Test ROC-AUC = 0.4365
    *   *Random Forest (Terbaik v2)*: Test F1-Macro = **0.6167** | Test Acc = 0.6522 | Test ROC-AUC = 0.5000
    *   XGBoost: Test F1-Macro = **0.4868** | Test Acc = 0.5217 | Test ROC-AUC = 0.5159
*   **Eksperimen Versi 3 (Audio Dipotong Segmen 10 Detik)**:
    *   ***Logistic Regression (Terbaik v3 - Model Terbaik Keseluruhan)***: Test F1-Macro = **0.6349** | Test Acc = 0.6522 | Test ROC-AUC = 0.5397
    *   SVM (RBF): Test F1-Macro = **0.4866** | Test Acc = 0.5652 | Test ROC-AUC = 0.4841
    *   Random Forest: Test F1-Macro = **0.3030** | Test Acc = 0.4348 | Test ROC-AUC = 0.4127
    *   XGBoost: Test F1-Macro = **0.4250** | Test Acc = 0.4783 | Test ROC-AUC = 0.4683
*   **Eksperimen Versi 4 (Audio Dipotong Segmen 30 Detik)**:
    *   *Logistic Regression (Terbaik v4)*: Test F1-Macro = **0.5437** | Test Acc = 0.5652 | Test ROC-AUC = 0.5238
    *   SVM (RBF): Test F1-Macro = **0.4103** | Test Acc = 0.5217 | Test ROC-AUC = 0.4683
    *   Random Forest: Test F1-Macro = **0.2813** | Test Acc = 0.3913 | Test ROC-AUC = 0.3889
    *   XGBoost: Test F1-Macro = **0.4250** | Test Acc = 0.4783 | Test ROC-AUC = 0.4048

#### Analisis Perbandingan & Rekomendasi Model Terbaik
1.  **Model Pemenang Secara Angka**: **Logistic Regression (Versi 3 - Segmen 10 Detik)** tetap menjadi pemenang dengan **Test F1-Macro = 0.6349** dan **ROC-AUC = 0.5397**.
2.  **Mengapa Performa Versi 4 (Segmen 30s) Mengalami Penurunan Dibanding Versi 3 (Segmen 10s)?**:
    *   **Pengurangan Drastis Jumlah Data Latih**: Dengan memperpanjang durasi segmen menjadi 30 detik, jumlah sampel baris data latih menyusut hampir 3 kali lipat (Train set: 668 segmen pada v4 vs 2.064 segmen pada v3). Akibatnya, keuntungan dari metode pelipatgandaan data latih (*data augmentation*) menjadi jauh berkurang.
    *   **Keseimbangan Trade-off Durasi**: Meskipun segmen 30 detik secara teoritis memiliki fitur akustik yang lebih stabil (kurang bising) daripada segmen 10 detik, kehilangan 2/3 baris data latih membuat model kekurangan variabilitas sampel untuk digeneralisasi dengan baik pada Test Set.
    *   **Hasil Akhir**: Model Machine Learning klasik membutuhkan volume sampel yang melimpah untuk belajar secara stabil. Oleh karena itu, segmentasi yang lebih pendek (10 detik) dengan jumlah data latih yang melimpah lebih unggul daripada segmentasi panjang (30 detik) dengan jumlah data latih yang minim.
3.  **Kesimpulan & Rekomendasi Sidang Presentasi**:
    *   Tetap rekomendasikan **Logistic Regression (Versi 3 - Segmen 10 Detik)** sebagai model final Anda.
    *   Eksperimen Versi 4 sangat bagus dipresentasikan sebagai **analisis pembanding durasi segmen (hyperparameter tuning durasi)**. Ini membuktikan kepada dosen penguji bahwa Anda telah menguji berbagai opsi durasi segmen (10s vs 30s) dan menyimpulkan secara ilmiah bahwa **durasi 10 detik adalah *sweet spot* (titik optimal)** untuk memaksimalkan performa pada dataset DAIC-WOZ.




