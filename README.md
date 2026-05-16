# Menthealth AI Project

Mental Health Analysis using Machine Learning and Deep Learning from Audio Data.

## Project Structure

```
mentalhealth-ai/
├── data/
│   ├── raw/                 # Original raw dataset
│   ├── cleaned/             # Cleaned/resampled audio
│   ├── features/
│   │   ├── mfcc/            # MFCC features for ML models
│   │   ├── spectrogram/     # Spectrogram data for CNN/LSTM
│   │   └── waveform/        # Raw waveform/tensor data for Wav2Vec2
│   └── splits/              # Train/validation/test split metadata
│
├── notebooks/
│   ├── traditional_ml.ipynb # SVM, Random Forest, XGBoost
│   ├── cnn.ipynb            # 1D-CNN implementation
│   ├── lstm.ipynb           # LSTM implementation
│   └── wav2vec2.ipynb       # Wav2Vec2 implementation
│
├── preprocessing/
│   ├── audio_cleaning.py        # Noise reduction, trimming, normalization
│   ├── extract_mfcc.py          # MFCC feature extraction
│   ├── generate_spectrogram.py  # Spectrogram generation
│   └── prepare_waveform.py      # Waveform preprocessing for Wav2Vec2
│
├── models/
│   ├── ml/
│   │   ├── svm/
│   │   ├── random_forest/
│   │   └── xgboost/
│   │
│   └── dl/
│       ├── cnn/
│       ├── lstm/
│       └── wav2vec2/
│           └── processor.py
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── confusion_matrix/
│
├── requirements.txt
└── README.md
```

## Setup

Sangat disarankan (wajib) untuk menggunakan virtual environment (`.venv`) agar tidak terjadi konflik library:

### 1. Persiapan Virtual Environment
**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalasi Dependensi
Setelah virtual environment aktif, instal library yang diperlukan:
```bash
pip install -r requirements.txt
```

### 3. Persiapan Folder Data
Jalankan skrip inisialisasi untuk membuat folder yang diperlukan (karena folder data di-ignore oleh Git):
```bash
cd data
createstruktur.bat
cd ..
```

### 4. Menjalankan Proyek
1. Letakkan dataset di `data/raw/`.
2. Gunakan `notebooks/traditional_ml.ipynb` atau jalankan `notebooks/traditional_ml.py` untuk menjalankan full pipeline (Part 1-7).
3. Hasil analisis dan model akan tersimpan di folder `results/` dan `models/`.
