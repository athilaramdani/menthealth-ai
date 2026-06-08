# BiLSTM MFCC — Panduan Penggunaan

Folder ini berisi semua script untuk melatih model **Bidirectional LSTM** menggunakan fitur **MFCC** (Mel-Frequency Cepstral Coefficients) dengan strategi **segmentasi 10 detik** untuk memperbanyak data.

---

## 📁 Struktur File

```
notebooks/LSTM/
├── extract_mfcc_segments.py   ← STEP 1: Ekstraksi MFCC + segmentasi 10 detik
├── quick_test_lstm.py         ← STEP 2: Sanity check sebelum training
├── train_bilstm_mfcc.py       ← STEP 3: Training local (5-Fold CV)
├── train_bilstm_colab.py      ← STEP 3 (alt): Training di Google Colab
└── README.md                  ← Dokumen ini
```

---

## 🚀 Cara Menjalankan (Lokal)

### Prasyarat
Pastikan environment Python aktif dan dependensi sudah terinstall:
```bash
.venv\Scripts\activate   # Windows
# atau
conda activate menthealth

pip install torch torchvision torchaudio librosa soundfile pandas scikit-learn tqdm seaborn matplotlib
```

### Step 1 — Ekstraksi MFCC (segmentasi 10 detik)
```bash
cd notebooks/LSTM
python extract_mfcc_segments.py
```
**Yang dilakukan:**
- Membaca semua file WAV dari `data/cleaned/` (189 file).
- Memotong setiap file menjadi segmen **non-overlapping 10 detik**.
- Mengekstrak **MFCC + Delta + Delta-Delta** (120 fitur total) per segmen.
- Menyimpan hasil sebagai `.npy` di `data/features/mfcc/NORMAL/` dan `DEPRESI/`.
- Estimasi output: **~2.000–6.000 segmen** (tergantung durasi audio).

### Step 2 — Sanity Check
```bash
python quick_test_lstm.py
```
Verifikasi bahwa file `.npy`, DataLoader, dan model forward pass semuanya OK.

### Step 3 — Training
```bash
python train_bilstm_mfcc.py
```
**Yang dilakukan:**
- 5-Fold Stratified K-Fold pada level **patient** (bukan segmen).
- WeightedRandomSampler untuk handle class imbalance.
- Simpan model terbaik ke `models/dl/lstm/best_bilstm_mfcc.pt`.
- Simpan metrics, plots, dan confusion matrix ke `results/lstm/`.

---

## 🧠 Kenapa Segmentasi 10 Detik?

| | Tanpa segmentasi | Dengan segmentasi 10s |
|---|---|---|
| **Total sampel** | ~189 (1 per pasien) | ~3.000–6.000 segmen |
| **Pola yang dipelajari** | Statistik level-pasien | Dinamika temporal 10 detik |
| **Generalisasi** | Rendah (overfitting tinggi) | Lebih baik |
| **Cocok untuk** | ML klasik | LSTM / Deep Learning |

**Penting:** Split dilakukan di level PATIENT agar tidak ada data leakage.

---

## 🏗️ Arsitektur BiLSTM

```
Input: (batch, 313 timesteps, 120 features)
  ↓
BatchNorm1d(120)
  ↓
BiLSTM(120→128, 2 layers, bidirectional=True)
  → Output: (batch, 313, 256)
  ↓
Attention Pool (weighted mean over timesteps)
  → (batch, 256)
  ↓
LayerNorm(256)
  ↓
Linear(256→128) → GELU → Dropout(0.35)
  ↓
Linear(128→2)
  ↓
Output: (batch, 2)  — NORMAL / DEPRESI
```

**Attention Pooling** digunakan agar model dapat belajar bagian mana dari percakapan (timestep mana) yang paling informatif untuk prediksi depresi — ini juga berguna untuk **XAI (Explainability)**.

---

## 📊 Output Hasil

```
results/lstm/
├── metrics/
│   ├── report_fold_1.txt ... report_fold_5.txt
│   ├── cv_summary.txt
│   └── report_test_set.txt
├── plots/
│   ├── fold_1_curves.png ... fold_5_curves.png
│   └── cv_summary.png
└── confusion_matrix/
    ├── cm_fold_1.png ... cm_fold_5.png
    └── cm_test_set.png

models/dl/lstm/
└── best_bilstm_mfcc.pt   ← model terbaik (termasuk config & metrics)
```

---

## 🔬 XAI — Explainability

Model ini mendukung XAI melalui **Attention Weights**:

```python
import torch
from train_bilstm_mfcc import BiLSTMMFCC

# Load model
ckpt  = torch.load("models/dl/lstm/best_bilstm_mfcc.pt")
model = BiLSTMMFCC(**ckpt["model_config"])
model.load_state_dict({k: v for k, v in ckpt["model_state_dict"].items()})
model.eval()

# Dapatkan attention weights (shape: [batch, timesteps])
with torch.no_grad():
    attn = model.get_attention_weights(sample_tensor)
    # Plot attn[0] → visualisasi mana detik yang paling penting
```

Untuk XAI yang lebih mendalam, gunakan **SHAP DeepExplainer** atau **Integrated Gradients** (captum).

---

## 📓 Google Colab

Gunakan `train_bilstm_colab.py` untuk training di Google Colab dengan GPU gratis. Sesuaikan `PROJECT_ROOT` dengan path di Google Drive Anda:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/menthealth-ai")
```
