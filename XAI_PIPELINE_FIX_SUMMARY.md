# Ringkasan Perbaikan XAI Pipeline

## Tanggal: 2 Juni 2026

## Masalah yang Diperbaiki

### 1. **Error saat menjalankan dengan 3 sampel**

**Masalah:**
- Ketika menjalankan `python run_xai_pipeline.py --n_samples 3`, script mengalami error
- Pembagian sampel per kelas menggunakan `n_each = max(1, n_samples // 2)` menghasilkan:
  - 3 // 2 = 1 sampel per kelas
  - Total hanya 2 sampel (bukan 3 seperti yang diminta)
- Jumlah sampel tidak mencukupi untuk analisis yang bermakna

### 2. **Argumen `--model both` tidak valid di xai_shap_dl.py**

**Masalah:**
- `run_xai_pipeline.py` memanggil `xai_shap_dl.py` dengan `--model both`
- Script `xai_shap_dl.py` tidak menangani opsi "both" dengan baik
- Menyebabkan error atau behavior yang tidak konsisten

## Solusi yang Diimplementasikan

### 1. **Perbaikan Pembagian Sampel di `xai_gradcam_cnn.py`**

```python
# SEBELUM (SALAH):
n_each = max(1, n_samples // 2)
selected = []
for cls_idx in [0, 1]:
    cls_samples = by_class[cls_idx]
    random.shuffle(cls_samples)
    selected.extend(cls_samples[:n_each])
selected = selected[:n_samples]  # Truncate bisa kehilangan sampel

# SESUDAH (BENAR):
n_each = n_samples // 2
n_extra = n_samples % 2  # Sisa untuk class pertama jika ganjil

selected = []
for i, cls_idx in enumerate([0, 1]):
    cls_samples = by_class[cls_idx]
    random.shuffle(cls_samples)
    n_take = n_each + (n_extra if i == 0 else 0)
    n_take = min(n_take, len(cls_samples))
    selected.extend(cls_samples[:n_take])

# Jika masih kurang, ambil sampel tambahan
if len(selected) < n_samples:
    remaining_needed = n_samples - len(selected)
    for cls_idx in [0, 1]:
        if remaining_needed <= 0:
            break
        cls_samples = by_class[cls_idx]
        already_taken = sum(1 for s in selected if s[2] == cls_idx)
        available = cls_samples[already_taken:]
        take_more = min(remaining_needed, len(available))
        selected.extend(available[:take_more])
        remaining_needed -= take_more
```

**Keuntungan:**
- ✅ Jumlah sampel yang dihasilkan sesuai dengan yang diminta
- ✅ Distribusi seimbang antar kelas (atau mendekati seimbang jika ganjil)
- ✅ Tidak ada sampel yang hilang karena truncation

### 2. **Perbaikan Handling Model di `run_xai_pipeline.py`**

```python
# SEBELUM (SALAH):
if args.model in ("all", "cnn", "bilstm"):
    shap_args = ["--n_samples", str(args.n_samples), "--device", args.device]
    if args.model == "cnn":
        shap_args += ["--model", "cnn"]
    elif args.model == "bilstm":
        shap_args += ["--model", "bilstm"]
    else:
        shap_args += ["--model", "both"]  # ❌ Tidak valid
    tasks.append((shap_dl_path, shap_args))

# SESUDAH (BENAR):
if args.model in ("all", "cnn", "bilstm"):
    shap_args_base = ["--n_samples", str(args.n_samples), "--device", args.device]
    if args.model == "cnn":
        tasks.append((shap_dl_path, shap_args_base + ["--model", "cnn"]))
    elif args.model == "bilstm":
        tasks.append((shap_dl_path, shap_args_base + ["--model", "bilstm"]))
    else:
        # ✅ Untuk "all", jalankan keduanya secara terpisah
        tasks.append((shap_dl_path, shap_args_base + ["--model", "cnn"]))
        tasks.append((shap_dl_path, shap_args_base + ["--model", "bilstm"]))
```

**Keuntungan:**
- ✅ Tidak ada argumen invalid yang dilewatkan
- ✅ CNN dan BiLSTM dijalankan secara terpisah untuk isolasi yang lebih baik
- ✅ Error handling lebih jelas

### 3. **Default Sampel Diubah Menjadi 4 (2 NORMAL + 2 DEPRESI)**

**File yang Diubah:**
- `run_xai_pipeline.py`: `default=4`
- `xai_gradcam_cnn.py`: `default=4`
- `xai_shap_dl.py`: `default=4`
- `xai_attention_wav2vec.py`: `default=4`

**Alasan:**
- 4 sampel adalah jumlah minimum yang bermakna untuk analisis XAI
- Pembagian seimbang: 2 NORMAL + 2 DEPRESI
- Cukup cepat untuk testing, tapi tetap informatif

### 4. **Validasi Input di `run_xai_pipeline.py`**

```python
# Validasi jumlah sampel minimum
if args.n_samples < 2:
    print(f"[ERROR] Jumlah sampel harus minimal 2")
    sys.exit(1)

# Peringatan untuk jumlah ganjil
if args.n_samples % 2 != 0:
    print(f"[WARNING] Jumlah sampel ganjil ({args.n_samples}). "
          "Sampel akan dibagi tidak merata antar kelas.")
    time.sleep(2)
```

**Keuntungan:**
- ✅ Mencegah error karena jumlah sampel tidak valid
- ✅ Memberikan feedback yang jelas ke user
- ✅ Peringatan untuk kasus edge case (jumlah ganjil)

### 5. **Logging yang Lebih Informatif**

```python
# Di xai_gradcam_cnn.py
logging.info(f"Total sampel untuk Grad-CAM: {len(selected)} (target: {n_samples})")
class_dist = {0: 0, 1: 0}
for _, _, cls_idx in selected:
    class_dist[cls_idx] += 1
logging.info(f"  - NORMAL  : {class_dist[0]} sampel")
logging.info(f"  - DEPRESI : {class_dist[1]} sampel")
```

**Keuntungan:**
- ✅ User dapat verifikasi distribusi sampel
- ✅ Debugging lebih mudah
- ✅ Transparansi dalam proses sampling

## Cara Menjalankan (Setelah Perbaikan)

### Dengan Default (4 Sampel: 2 NORMAL + 2 DEPRESI)

```bash
python run_xai_pipeline.py
```

### Dengan Jumlah Sampel Kustom

```bash
# 6 sampel (3 NORMAL + 3 DEPRESI)
python run_xai_pipeline.py --n_samples 6

# 10 sampel (5 NORMAL + 5 DEPRESI)
python run_xai_pipeline.py --n_samples 10

# Hanya CNN dengan 4 sampel
python run_xai_pipeline.py --model cnn --n_samples 4
```

### ❌ Yang TIDAK AKAN Bekerja

```bash
# Error: Terlalu sedikit sampel
python run_xai_pipeline.py --n_samples 1

# Warning: Jumlah ganjil (akan jalan tapi tidak seimbang)
python run_xai_pipeline.py --n_samples 5
```

## Testing yang Disarankan

1. **Test dengan 4 sampel (default)**
   ```bash
   python run_xai_pipeline.py
   ```

2. **Test dengan 6 sampel**
   ```bash
   python run_xai_pipeline.py --n_samples 6
   ```

3. **Test per model**
   ```bash
   python run_xai_pipeline.py --model cnn --n_samples 4
   python run_xai_pipeline.py --model bilstm --n_samples 4
   python run_xai_pipeline.py --model wav2vec --n_samples 4
   ```

4. **Verifikasi output**
   - Cek folder `results/xai/gradcam/`, `results/xai/shap/`, `results/xai/attention/`
   - Pastikan jumlah file sesuai dengan jumlah sampel
   - Periksa distribusi kelas di report `.txt`

## Expected Output

```
==================================================================
        MENTAL HEALTH AUDIO CLASSIFICATION — XAI RUNNER
==================================================================
Proyek  : Klasifikasi NORMAL vs DEPRESI Berbasis Audio
Fase 4  : Implementasi Explainable AI (XAI) untuk Deep Learning
==================================================================

[RUNNING] Menjalankan xai_gradcam_cnn.py...
Device: cpu
Model loaded dari: models/dl/cnn/best_model_1;1.pt
Total sampel untuk Grad-CAM: 4 (target: 4)
  - NORMAL  : 2 sampel
  - DEPRESI : 2 sampel
...

[SUCCESS] xai_gradcam_cnn.py selesai dalam XX.XX detik.

[RUNNING] Menjalankan xai_shap_dl.py...
...

==================================================================
PIPELINE XAI SELESAI!
Berhasil menjalankan X dari X modul XAI.
==================================================================
```

## Catatan Penting

1. **Jumlah Sampel Minimum**: Selalu gunakan minimal 2 sampel (1 per kelas)
2. **Jumlah Genap Disarankan**: Untuk distribusi seimbang, gunakan jumlah genap (4, 6, 8, dst)
3. **Performance**: 4 sampel = ~30-60 detik, 8 sampel = ~60-120 detik (tergantung hardware)
4. **Model Requirements**: Pastikan model sudah ditraining sebelum menjalankan XAI pipeline

## File yang Dimodifikasi

1. ✅ `run_xai_pipeline.py`
2. ✅ `notebooks/CNN/xai_gradcam_cnn.py`
3. ✅ `notebooks/xai_shap_dl.py`
4. ✅ `notebooks/WAV2VEC/xai_attention_wav2vec.py`

## Versi

- **Before**: Error dengan 3 sampel, default 8 atau 16 sampel
- **After**: Berfungsi dengan 2+ sampel, default 4 sampel (2 NORMAL + 2 DEPRESI)
