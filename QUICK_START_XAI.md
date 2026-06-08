# Quick Start Guide - XAI Pipeline

## 🚀 Cara Cepat Menjalankan XAI

### Default (4 Sampel: 2 NORMAL + 2 DEPRESI)

```bash
python run_xai_pipeline.py
```

**Output:**
- ✅ Grad-CAM untuk CNN (4 visualisasi)
- ✅ SHAP untuk CNN dan BiLSTM (4 visualisasi per model)
- ✅ Attention untuk Wav2Vec2 (4 visualisasi)

**Durasi:** ~2-5 menit (tergantung hardware)

---

## 📊 Opsi Lainnya

### Lebih Banyak Sampel (Lebih Akurat)

```bash
# 6 sampel (3 NORMAL + 3 DEPRESI)
python run_xai_pipeline.py --n_samples 6

# 8 sampel (4 NORMAL + 4 DEPRESI) - Rekomendasi untuk presentasi
python run_xai_pipeline.py --n_samples 8

# 10 sampel (5 NORMAL + 5 DEPRESI)
python run_xai_pipeline.py --n_samples 10
```

### Hanya Model Tertentu

```bash
# Hanya CNN
python run_xai_pipeline.py --model cnn

# Hanya BiLSTM
python run_xai_pipeline.py --model bilstm

# Hanya Wav2Vec2
python run_xai_pipeline.py --model wav2vec
```

### Dengan GPU (Lebih Cepat)

```bash
python run_xai_pipeline.py --device cuda --n_samples 8
```

---

## 📁 Hasil Output

Setelah selesai, cek folder berikut:

```
results/xai/
├── gradcam/              ← Heatmap CNN (area penting di spectrogram)
│   ├── gradcam_300_NORMAL_predNORMAL.png
│   ├── gradcam_301_DEPRESI_predDEPRESI.png
│   ├── gradcam_summary_grid.png
│   └── gradcam_report.txt
│
├── shap/                 ← Feature importance CNN & BiLSTM
│   ├── shap_cnn_bar.png
│   ├── shap_cnn_beeswarm.png
│   ├── shap_cnn_waterfall_300.png
│   ├── shap_bilstm_bar.png
│   ├── shap_bilstm_beeswarm.png
│   └── shap_bilstm_waterfall_300.png
│
└── attention/            ← Attention weights Wav2Vec2
    ├── attn_300_head_heatmap.png
    ├── attn_300_temporal_profile.png
    ├── attn_300_layer_evolution.png
    └── attn_summary_grid.png
```

---

## 🎯 Cara Membaca Hasil

### Grad-CAM (CNN)
- **Warna MERAH/KUNING** = Area yang paling mempengaruhi keputusan model
- **Warna BIRU** = Area yang kurang penting
- Lihat dimana "panas" nya → itu yang dilihat model

### SHAP Values
- **Bar Chart** = Fitur mana yang paling penting secara keseluruhan
- **Beeswarm** = Distribusi pengaruh fitur (merah = nilai tinggi, biru = nilai rendah)
- **Waterfall** = Penjelasan untuk 1 sampel spesifik

### Attention (Wav2Vec2)
- **Temporal Profile** = Momen mana dalam audio yang paling diperhatikan
- **Puncak kuning** = Bagian suara yang paling menentukan prediksi
- **Layer Evolution** = Bagaimana model memproses audio dari lapisan dasar ke lanjutan

---

## ⚠️ Troubleshooting

### Error: "Model tidak ditemukan"
**Solusi:** Pastikan model sudah ditraining terlebih dahulu
```bash
# Training CNN
python notebooks/CNN/train_2d_cnn.py

# Training BiLSTM
python notebooks/LSTM/train_bilstm_mfcc.py

# Training Wav2Vec2
python notebooks/WAV2VEC/train_wav2vec.py
```

### Error: "Jumlah sampel harus minimal 2"
**Solusi:** Gunakan minimal 2 sampel
```bash
python run_xai_pipeline.py --n_samples 4  # ✅ OK
```

### Warning: "Jumlah sampel ganjil"
**Solusi:** Gunakan jumlah genap untuk distribusi seimbang
```bash
python run_xai_pipeline.py --n_samples 6  # ✅ Seimbang (3+3)
```

### Terlalu Lama/Out of Memory
**Solusi:** Kurangi jumlah sampel atau gunakan CPU
```bash
python run_xai_pipeline.py --n_samples 2 --device cpu
```

---

## 💡 Tips

1. **Untuk Testing Cepat:** Gunakan 2-4 sampel
2. **Untuk Presentasi/Paper:** Gunakan 8-10 sampel
3. **Untuk Analisis Mendalam:** Gunakan 16+ sampel (tapi butuh waktu lebih lama)
4. **Gunakan GPU:** Tambahkan `--device cuda` jika tersedia
5. **Fokus ke Model Tertentu:** Gunakan `--model cnn` untuk testing lebih cepat

---

## 📝 Contoh Command Lengkap

```bash
# Test cepat (2 menit)
python run_xai_pipeline.py --n_samples 4 --model cnn

# Untuk presentasi (5-8 menit)
python run_xai_pipeline.py --n_samples 8

# Full analysis dengan GPU (10-15 menit)
python run_xai_pipeline.py --n_samples 16 --device cuda

# BiLSTM only dengan 6 sampel
python run_xai_pipeline.py --model bilstm --n_samples 6
```

---

## 📚 Dokumentasi Lengkap

Untuk penjelasan detail tentang setiap metode XAI, baca:
- `docs/xai_workflow_expectations.md`
- `XAI_PIPELINE_FIX_SUMMARY.md`

Untuk penjelasan cara membaca grafik secara awam, lihat caption di setiap visualisasi yang dihasilkan.
