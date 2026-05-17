# Detail Tugas Proyek Menthealth

Dokumen ini merinci pembagian tugas dan timeline pengerjaan proyek klasifikasi kesehatan mental berbasis audio selama 8 minggu.

## Peran & Tanggung Jawab
- **A — ML & Data Engineer**: Fokus pada pemrosesan data audio, ekstraksi fitur tradisional, dan model Machine Learning klasik.
- **B — DL & XAI Engineer**: Fokus pada arsitektur Deep Learning dan implementasi metode Explainable AI.
- **C — Web & Integration**: Fokus pada pengembangan antarmuka web, API backend, dan integrasi sistem secara keseluruhan.

---

## Fase 1: Riset, Literatur & Setup (W1-W2)

### **A — ML & Data Engineer**
- [ ] Request akses dataset DAIC-WOZ / MODMA
- [ ] Studi literatur *audio depression detection*
- [x] Setup Python environment (`librosa`, `sklearn`, `SHAP`)
- [ ] Eksplorasi awal data audio (EDA)

### **B — DL & XAI Engineer**
- [ ] Studi arsitektur Deep Learning (`CNN`, `LSTM`, `wav2vec`)
- [ ] Setup PyTorch / TensorFlow environment
- [ ] Riset metode XAI (`LIME`, `SHAP`, `Grad-CAM`)

### **C — Web & Integration**
- [ ] Desain wireframe UI/UX (Figma)
- [ ] Perancangan arsitektur sistem & *API contract*
- [ ] Setup repository & struktur proyek

---

## Fase 2: Data & Feature Engineering (W2-W3)

### **A — ML & Data Engineer**
- [ ] Preprocessing audio (*resampling*, *trim*, *VAD*)
- [ ] Ekstraksi MFCC (13-40 koefisien)
- [ ] Ekstraksi *pitch*, *energy*, *jitter*, *shimmer*
- [ ] *Feature normalization* & *selection*

### **B — DL & XAI Engineer**
- [ ] Generate *mel-spectrograms* dari audio
- [ ] *Data augmentation* (*time stretch*, *noise*, *pitch shift*)
- [ ] Setup DataLoader & pipeline DL

### **C — Web & Integration**
- [ ] Setup proyek React + Tailwind CSS
- [ ] Pembuatan halaman upload audio + UI skeleton
- [ ] Setup FastAPI + basic `/predict` endpoint

---

## Fase 3: Training ML + DL Models (W3-W5)

### **A — ML Engineer (W3-W4)**
- [ ] Train model SVM, Random Forest, XGBoost, Logistic Regression
- [ ] Cross-validation (5-fold / LOSO)
- [ ] Hyperparameter tuning (GridSearch/Optuna)
- [ ] Evaluasi: *Confusion matrix* & *classification report*
- [ ] Export model terbaik (`.pkl`)

### **B — DL Engineer (W3-W5)**
- [ ] Train 1D-CNN pada fitur MFCC
- [ ] Train 2D-CNN pada mel-spectrograms
- [ ] Train LSTM / BiLSTM
- [ ] Fine-tune wav2vec 2.0 (opsional)
- [ ] *Experiment tracking* (W&B / MLflow)

---

## Fase 4: XAI Implementation (W5-W6)

### **A — XAI untuk ML Models (W5)**
- [ ] Implementasi SHAP values untuk RF & XGBoost
- [ ] Implementasi LIME explanations untuk SVM
- [ ] Visualisasi: *Feature importance summary plot*, *waterfall*, & *beeswarm plots*

### **B — XAI untuk DL Models (W5-W6)**
- [ ] Implementasi Grad-CAM heatmaps pada CNN spectrograms
- [ ] Implementasi SHAP DeepExplainer untuk model DL
- [ ] Visualisasi *Attention* (jika menggunakan wav2vec)

### **C — Backend XAI API (W5-W6)**
- [ ] API `/explain` endpoint (integrasi LIME + SHAP)
- [ ] API `/compare` endpoint (perbandingan ML vs DL)
- [ ] Serialisasi plot XAI ke format JSON atau gambar

---

## Fase 5: Website Dev & Integrasi (W4-W7)

### **C — Frontend Development (W4-W5)**
- [ ] Halaman upload audio dengan progress bar
- [ ] Halaman tampilan hasil klasifikasi + confidence chart
- [ ] Dashboard XAI (integrasi plot SHAP/LIME)
- [ ] Halaman perbandingan model + grafik interaktif

### **C — Integrasi & Polish (W6-W7)**
- [ ] Integrasi semua model ML & DL ke API
- [ ] Visualisasi XAI interaktif di sisi frontend
- [ ] *Responsive design* & poles UI/UX
- [ ] Implementasi *error handling*, *loading states*, & *edge cases*

> *Catatan: Anggota A dan B memberikan dukungan penuh untuk integrasi pada W6-W7.*

---

## Fase 6: Testing, Deploy & Docs (W7-W8)

### **Semua Anggota — Testing (W7)**
- [ ] *End-to-end testing* (Upload → Prediksi → XAI)
- [ ] *Unit test* untuk prediksi model
- [ ] UAT & *performance testing*
- [ ] Perbaikan bug & optimasi

### **Semua Anggota — Deploy & Docs (W8)**
- [ ] Deploy backend ke Railway / Render
- [ ] Deploy frontend ke Vercel / Netlify
- [ ] Dokumentasi teknis lengkap (README)
- [ ] Laporan perbandingan model final (*Final comparison report*)
- [ ] Laporan akhir & persiapan presentasi demo
- [ ] Pembuatan video rekaman demo
