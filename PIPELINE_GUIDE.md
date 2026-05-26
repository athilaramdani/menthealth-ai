# 🚀 GUIDE: Run Full Pipeline dengan Fixes

## ✅ JAWABAN SINGKAT

**Ya, sama aja!** Semua fixes sudah otomatis diterapkan karena pipeline memanggil `train_2d_cnn.py` yang sudah di-fix.

---

## 📋 PERBANDINGAN: Manual vs Pipeline

### **MANUAL (Step-by-Step)**
```bash
# 1. Generate spectrogram
cd preprocessing
python generate_spectrogram.py

# 2. Augmentasi
python data_augmentation.py

# 3. Training
cd ..\models\dl\cnn
python train_2d_cnn.py

# 4. Test
python quick_test.py
```

**Kelebihan:**
- ✅ Kontrol penuh setiap step
- ✅ Bisa skip step yang sudah jalan
- ✅ Mudah debug jika ada error

**Kekurangan:**
- ❌ Harus manual jalankan 4 command
- ❌ Harus ingat urutan

---

### **PIPELINE (One Command)**
```bash
python run_full_pipeline.py
```

**Kelebihan:**
- ✅ Satu command, semua jalan otomatis
- ✅ Log lengkap di `run_full_pipeline.log`
- ✅ Auto stop jika ada step yang gagal

**Kekurangan:**
- ❌ Jika augmentasi sudah jalan, tetap di-run ulang (waste time)
- ❌ Tidak ada quick test otomatis (harus manual setelah pipeline selesai)

---

## 🎯 REKOMENDASI: Kapan Pakai Apa?

### **Gunakan PIPELINE jika:**
- ✅ First time run (belum ada data spectrogram & augmentasi)
- ✅ Mau re-run dari awal (clean slate)
- ✅ Mau log lengkap untuk dokumentasi

### **Gunakan MANUAL jika:**
- ✅ Spectrogram & augmentasi sudah ada (skip ke training langsung)
- ✅ Mau coba hyperparameter berbeda (edit `train_2d_cnn.py`, langsung run)
- ✅ Debugging (fokus ke satu step yang error)

---

## 🚀 CARA RUN PIPELINE (RECOMMENDED)

### **STEP 1: Backup Model Lama (Optional)**
```bash
copy models\dl\cnn\best_model.pt models\dl\cnn\best_model_OLD.pt
copy results\metrics\2d_cnn_classification_report.txt results\metrics\2d_cnn_classification_report_OLD.txt
```

### **STEP 2: Run Pipeline**
```bash
python run_full_pipeline.py
```

**Output yang diharapkan:**
```
2026-05-23 10:00:00 - INFO - Pipeline dimulai
============================================================
[1/3 Generate Spectrogram] Memulai: preprocessing\generate_spectrogram.py
============================================================
  [1/3 Generate Spectrogram] Total sampel ditemukan: 142
  [1/3 Generate Spectrogram] Distribusi dataset penuh: {'NORMAL': 63, 'STRES': 25, 'CEMAS': 11, 'DEPRESI': 43}
  ...
[1/3 Generate Spectrogram] SELESAI — 45.2s

============================================================
[2/3 Data Augmentation] Memulai: preprocessing\data_augmentation.py
============================================================
  [2/3 Data Augmentation] Ditemukan 142 file audio
  [2/3 Data Augmentation] Augmentasi 300 -> [NORMAL]
  ...
[2/3 Data Augmentation] SELESAI — 180.5s

============================================================
[3/3 Train 2D CNN] Memulai: models\dl\cnn\train_2d_cnn.py
============================================================
  [3/3 Train 2D CNN] Device: cuda
  [3/3 Train 2D CNN] Distribusi kelas training : [441, 175, 77, 301]
  [3/3 Train 2D CNN] Class weights CE-Loss     : [0.998, 1.002, 1.003, 0.997]
  [3/3 Train 2D CNN] --- Epoch 1/60 ---
  [3/3 Train 2D CNN] Train  → Loss: 1.3856 | Acc: 28.45%
  [3/3 Train 2D CNN] Val    → Loss: 1.3421 | Acc: 31.82%
  [3/3 Train 2D CNN] LR     → 0.001000
  [3/3 Train 2D CNN] Dist prediksi val: {'NORMAL': 8, 'STRES': 3, 'CEMAS': 2, 'DEPRESI': 9}
  ...
[3/3 Train 2D CNN] SELESAI — 1850.3s

============================================================
PIPELINE SELESAI TANPA ERROR — 2076.0s
Output ada di: results/metrics/, results/plots/, results/confusion_matrix/
============================================================
```

### **STEP 3: Quick Test (Manual)**
```bash
cd models\dl\cnn
python quick_test.py
```

### **STEP 4: Compare Hasil (Manual)**
```bash
python compare_results.py
```

---

## 📊 MONITORING PIPELINE

### **Real-time Monitoring (Terminal Baru)**
```bash
# Windows PowerShell
Get-Content run_full_pipeline.log -Wait -Tail 20

# Windows CMD
powershell Get-Content run_full_pipeline.log -Wait -Tail 20
```

### **Cek Progress Training**
```bash
# Lihat epoch terakhir
findstr /C:"Epoch" run_full_pipeline.log | more +100

# Lihat distribusi prediksi val
findstr /C:"Dist prediksi val" run_full_pipeline.log
```

---

## ⚠️ TROUBLESHOOTING PIPELINE

### **Error: "Script tidak ditemukan"**
```
[1/3 Generate Spectrogram] Script tidak ditemukan: preprocessing\generate_spectrogram.py
```

**Fix:**
```bash
# Pastikan Anda di root project
cd C:\Users\raiha\OneDrive\Documents\GitHub\menthealth-ai
python run_full_pipeline.py
```

### **Error: "GAGAL (exit code 1)"**
```
[2/3 Data Augmentation] GAGAL (exit code 1) — 5.2s
```

**Fix:**
1. Buka `run_full_pipeline.log`
2. Cari baris error (biasanya ada "ERROR" atau "Traceback")
3. Fix error di script yang gagal
4. Re-run pipeline

### **Pipeline Stuck di Training**
```
[3/3 Train 2D CNN] --- Epoch 1/60 ---
(tidak ada output lagi selama >5 menit)
```

**Kemungkinan:**
- Training sedang jalan (normal, epoch pertama bisa 2-3 menit)
- Cek GPU usage: `nvidia-smi` (jika pakai CUDA)
- Cek CPU usage: Task Manager

**Fix jika benar-benar stuck:**
- Ctrl+C untuk stop
- Cek `models\dl\cnn\train_2dcnn.log` untuk error
- Run manual: `cd models\dl\cnn && python train_2d_cnn.py`

---

## 🎛️ CUSTOMIZATION PIPELINE

### **Skip Step yang Sudah Jalan**

Edit `run_full_pipeline.py`:
```python
STEPS = [
    # Comment out step yang mau di-skip
    # ("1/3 Generate Spectrogram", GENERATE_SPEC, True),  # ← Skip
    # ("2/3 Data Augmentation",    AUGMENT,       True),  # ← Skip
    ("3/3 Train 2D CNN",         TRAIN_CNN,     True),    # ← Run ini aja
]
```

### **Tambah Quick Test Otomatis**

Edit `run_full_pipeline.py`:
```python
STEPS = [
    ("1/3 Generate Spectrogram", GENERATE_SPEC, True),
    ("2/3 Data Augmentation",    AUGMENT,       True),
    ("3/3 Train 2D CNN",         TRAIN_CNN,     True),
    ("4/4 Quick Test",           CNN_DIR / "quick_test.py", False),  # ← Uncomment
]
```

### **Tambah Compare Otomatis**

Edit `run_full_pipeline.py`:
```python
STEPS = [
    ("1/3 Generate Spectrogram", GENERATE_SPEC, True),
    ("2/3 Data Augmentation",    AUGMENT,       True),
    ("3/3 Train 2D CNN",         TRAIN_CNN,     True),
    ("4/5 Quick Test",           CNN_DIR / "quick_test.py", False),
    ("5/5 Compare Results",      CNN_DIR / "compare_results.py", False),
]
```

---

## 📝 CHECKLIST: Run Pipeline

- [ ] **Pre-run**
  - [ ] Backup model lama
  - [ ] Backup classification report lama
  - [ ] Pastikan di root project

- [ ] **Run**
  - [ ] `python run_full_pipeline.py`
  - [ ] Monitor log real-time (terminal baru)
  - [ ] Tunggu hingga "PIPELINE SELESAI"

- [ ] **Post-run**
  - [ ] `cd models\dl\cnn`
  - [ ] `python quick_test.py`
  - [ ] `python compare_results.py`
  - [ ] Cek confusion matrix & learning curves

- [ ] **Verify**
  - [ ] Model tidak collapse (quick_test.py)
  - [ ] CEMAS recall > 0% (compare_results.py)
  - [ ] Overall accuracy 35-50%

---

## 🎯 KESIMPULAN

| Aspek | Manual | Pipeline |
|-------|--------|----------|
| **Fixes diterapkan?** | ✅ Ya | ✅ Ya (sama persis) |
| **Jumlah command** | 4 command | 1 command |
| **Waktu total** | ~35 menit | ~35 menit (sama) |
| **Log lengkap** | Terpisah per step | Satu file lengkap |
| **Flexibility** | ✅ Tinggi (bisa skip step) | ⚠️ Sedang (harus edit script) |
| **Recommended untuk** | Development & debugging | Production & dokumentasi |

**REKOMENDASI:**
- **First run:** Gunakan **PIPELINE** (clean & lengkap)
- **Iterasi cepat:** Gunakan **MANUAL** (skip preprocessing, langsung training)

---

**Last Updated:** 2026-05-23  
**Version:** 1.0
