
# %% [markdown]
# # Part 2 — Preprocessing Audio: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ## Analisis File Metadata Pendukung
# | File | Digunakan? | Fungsi |
# |------|-----------|--------|
# | `TRANSCRIPT.csv` | ✅ Ya | Speaker Diarization — filter segmen suara `Participant` saja |
# | `COVAREP.csv` | ✅ Ya (kolom index-1) | VAD Primer — kolom VUV (1=voiced, 0=unvoiced) |
# | `FORMANT.csv` | ❌ Tidak | Hanya relevan untuk Feature Extraction, bukan preprocessing |
#
# ## Tahapan Pipeline per Partisipan
# 1. **Load Audio** — resample ke 16kHz, convert mono
# 2. **Speaker Diarization** — potong segmen suara Participant via TRANSCRIPT.csv
# 3. **VAD** — filter frame unvoiced via COVAREP VUV (fallback: librosa energy-VAD)
# 4. **Noise Reduction** — reduksi noise background via noisereduce
# 5. **Amplitude Normalization** — normalisasi peak ke [-1, 1]
# 6. **Simpan** — output ke `data/cleaned/{pid}.wav` + log CSV

# %% [markdown]
# ## 1. Import Library

# %%
import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import noisereduce as nr
from tqdm import tqdm

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("Library berhasil diimport.")

# %% [markdown]
# ## 2. Konfigurasi Path
# Menentukan lokasi folder dataset, output, dan log.
# Script ini ada di `experiments/`, sehingga naik **1 tingkat** untuk sampai ke root project.

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

# Script ada di experiments/ → naik 1 level ke root project menthealth-ai/
BASE_DIR    = os.path.dirname(current_dir)
DAIC_DIR    = os.path.join(BASE_DIR, 'data', 'raw', 'DAIC-WOZ')
CLEANED_DIR = os.path.join(BASE_DIR, 'data', 'cleaned')
LOG_DIR     = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print(f"BASE_DIR    : {BASE_DIR}")
print(f"DAIC_DIR    : {DAIC_DIR}")
print(f"CLEANED_DIR : {CLEANED_DIR}")

# %% [markdown]
# ## 3. Toggle Fitur Preprocessing
#
# Atur `True` / `False` untuk mengaktifkan atau menonaktifkan setiap tahap preprocessing.
# Berguna saat eksperimen untuk membandingkan hasil dengan/tanpa tahap tertentu.
#
# | Flag | Default | Keterangan |
# |------|---------|-----------|
# | `USE_DIARIZATION` | True | Filter segmen Participant via TRANSCRIPT.csv |
# | `USE_VAD_COVAREP` | True | VAD primer dari kolom VUV COVAREP.csv |
# | `USE_VAD_FALLBACK` | True | Fallback ke librosa energy-VAD jika COVAREP gagal |
# | `USE_NOISE_REDUCE` | True | Noise reduction via noisereduce |
# | `USE_NORMALIZE` | True | Peak amplitude normalization ke [-1, 1] |
# | `USE_MIN_SEG_FILTER` | True | Buang segmen terlalu pendek (< MIN_SEG_DUR detik) |

# %%
# ─── Toggle Fitur ────────────────────────────────────────────
USE_DIARIZATION    = True   # Speaker Diarization via TRANSCRIPT.csv
USE_VAD_COVAREP    = True   # VAD menggunakan VUV dari COVAREP.csv
USE_VAD_FALLBACK   = True   # Fallback ke librosa energy VAD
USE_NOISE_REDUCE   = True   # Noise reduction
USE_NORMALIZE      = True   # Amplitude normalization

USE_MIN_SEG_FILTER = True   # Filter segmen terlalu pendek

# ─── Parameter ───────────────────────────────────────────────
TARGET_SR       = 16000  # Hz — sampling rate standar speech processing
MIN_SEG_DUR     = 0.3    # detik — durasi minimum segmen (dipakai jika USE_MIN_SEG_FILTER=True)
TOP_DB_VAD      = 25     # dB threshold untuk librosa VAD fallback
COVAREP_VUV_COL = 1      # index kolom VUV di COVAREP.csv (0-indexed, kolom ke-2)
NOISE_PROP_DEC  = 0.8    # agresivitas noise reduction (0.0 - 1.0)
NOISE_EST_SEC   = 0.5    # durasi estimasi noise dari awal audio (detik)

print("Konfigurasi toggle preprocessing:")
print(f"  USE_DIARIZATION    = {USE_DIARIZATION}")
print(f"  USE_VAD_COVAREP    = {USE_VAD_COVAREP}")
print(f"  USE_VAD_FALLBACK   = {USE_VAD_FALLBACK}")
print(f"  USE_NOISE_REDUCE   = {USE_NOISE_REDUCE}")
print(f"  USE_NORMALIZE      = {USE_NORMALIZE}")
print(f"  USE_MIN_SEG_FILTER = {USE_MIN_SEG_FILTER}")
print(f"\n  TARGET_SR     = {TARGET_SR} Hz")
print(f"  MIN_SEG_DUR   = {MIN_SEG_DUR} detik  (aktif jika USE_MIN_SEG_FILTER=True)")
print(f"  TOP_DB_VAD    = {TOP_DB_VAD} dB       (aktif jika USE_VAD_FALLBACK=True)")
print(f"  NOISE_PROP_DEC= {NOISE_PROP_DEC}          (aktif jika USE_NOISE_REDUCE=True)")

# %% [markdown]
# ## 4. Setup Logging
# Log ditulis ke terminal dan ke file `data/processed/preprocessing.log`.

# %%
log = logging.getLogger('preprocessing')
log.setLevel(logging.INFO)
log.handlers.clear()

fh = logging.FileHandler(os.path.join(LOG_DIR, 'preprocessing.log'), encoding='utf-8')
sh = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
fh.setFormatter(fmt)
sh.setFormatter(fmt)
log.addHandler(fh)
log.addHandler(sh)

log.info("Logger siap.")

# %% [markdown]
# ## 5. Fungsi: Speaker Diarization
#
# Membaca `TRANSCRIPT.csv` untuk mendapatkan timestamp segmen suara Participant.
# Format file: `start_time [TAB] stop_time [TAB] speaker [TAB] value`
#
# Hanya baris dengan `speaker == 'Participant'` yang diambil.
# Suara Ellie (virtual interviewer) dibuang sepenuhnya untuk mencegah data leakage.

# %%
def get_participant_segments(transcript_path):
    """
    Membaca TRANSCRIPT.csv dan mengembalikan list tuple (start_sec, stop_sec)
    hanya untuk segmen suara Participant (bukan Ellie).
    """
    try:
        df = pd.read_csv(transcript_path, sep='\t')
        df.columns = [c.strip().lower() for c in df.columns]
        df_p = df[df['speaker'].str.strip().str.lower() == 'participant'].copy()
        return list(zip(df_p['start_time'].astype(float),
                        df_p['stop_time'].astype(float)))
    except Exception as e:
        log.warning(f"Gagal baca TRANSCRIPT: {e}")
        return []


def extract_segments_audio(y, sr, segments):
    """
    Concat hanya potongan audio yang termasuk segmen Participant.
    Segmen terlalu pendek dibuang jika USE_MIN_SEG_FILTER aktif.
    """
    chunks = []
    min_dur = MIN_SEG_DUR if USE_MIN_SEG_FILTER else 0.0
    for (start, stop) in segments:
        if (stop - start) < min_dur:
            continue
        s = int(start * sr)
        e = min(int(stop * sr), len(y))
        if s < e:
            chunks.append(y[s:e])
    return np.concatenate(chunks) if chunks else np.array([])

print("Fungsi Speaker Diarization siap.")

# %% [markdown]
# ## 6. Fungsi: VAD via COVAREP VUV
#
# COVAREP.csv tidak memiliki header. Kolom ke-2 (index 1) berisi flag VUV:
# - `1` = Voiced (ada suara/speech)
# - `0` = Unvoiced atau silence
#
# COVAREP di-ekstrak pada **100 fps** (frame setiap 10ms).
# Mask frame-level di-expand ke sample-level menggunakan `np.repeat`.

# %%
def get_vuv_mask_from_covarep(covarep_path, n_samples, sr=TARGET_SR, fps=100):
    """
    Membaca kolom VUV dari COVAREP.csv dan konversi ke sample-level boolean mask.
    Returns None jika file tidak valid atau fitur dinonaktifkan.
    """
    if not USE_VAD_COVAREP:
        return None
    try:
        covarep = np.genfromtxt(covarep_path, delimiter=',', filling_values=0)
        if covarep.ndim < 2 or covarep.shape[1] <= COVAREP_VUV_COL:
            return None

        vuv_frames   = covarep[:, COVAREP_VUV_COL]   # shape: (n_frames,)
        hop_samples  = int(sr / fps)                  # 160 samples @ 16kHz, 100fps
        mask         = np.repeat(vuv_frames, hop_samples)

        # Sesuaikan panjang mask dengan jumlah sampel audio
        if len(mask) > n_samples:
            mask = mask[:n_samples]
        else:
            mask = np.pad(mask, (0, n_samples - len(mask)), constant_values=0)

        return mask.astype(bool)
    except Exception as e:
        log.warning(f"Gagal baca COVAREP VUV: {e}")
        return None


def apply_vad_covarep(y, vuv_mask):
    """Terapkan VUV mask — ambil hanya sample dengan VUV=1 (voiced)."""
    if vuv_mask is None or len(vuv_mask) != len(y):
        return None
    voiced = y[vuv_mask]
    return voiced if len(voiced) > 0 else None

print("Fungsi VAD COVAREP siap.")

# %% [markdown]
# ## 7. Fungsi: VAD Fallback (Librosa Energy-Based)
#
# Digunakan jika COVAREP tidak tersedia atau kolom VUV tidak valid.
# `librosa.effects.split` memotong bagian audio yang berada di bawah
# threshold `TOP_DB_VAD` dB dari nilai maksimum sinyal.

# %%
def apply_vad_librosa(y):
    """
    Fallback VAD berbasis energy menggunakan librosa.
    Aktif hanya jika USE_VAD_FALLBACK = True.
    """
    if not USE_VAD_FALLBACK:
        return y
    if len(y) == 0:
        return y
    intervals = librosa.effects.split(y, top_db=TOP_DB_VAD,
                                       frame_length=512, hop_length=128)
    if len(intervals) == 0:
        return y
    return np.concatenate([y[s:e] for s, e in intervals])

print("Fungsi VAD Fallback siap.")

# %% [markdown]
# ## 8. Fungsi: Noise Reduction
#
# Menggunakan library `noisereduce` dengan metode spectral gating.
# Estimasi profil noise diambil dari `NOISE_EST_SEC` detik pertama audio
# (biasanya masih silence sebelum interview dimulai).
#
# `NOISE_PROP_DEC` mengontrol agresivitas: 1.0 = reduksi penuh, 0.5 = sedang.

# %%
def reduce_noise_audio(y, sr):
    """
    Noise reduction via noisereduce spectral gating.
    Dilewati jika USE_NOISE_REDUCE = False.
    """
    if not USE_NOISE_REDUCE:
        return y
    try:
        n_noise = min(int(sr * NOISE_EST_SEC), len(y) // 4)
        if n_noise < 100:
            return y
        noise_clip = y[:n_noise]
        return nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip,
                               prop_decrease=NOISE_PROP_DEC, stationary=False)
    except Exception as e:
        log.warning(f"Noise reduction gagal, skip: {e}")
        return y

print("Fungsi Noise Reduction siap.")

# %% [markdown]
# ## 9. Fungsi: Amplitude Normalization
#
# Peak normalization — memastikan semua audio berada dalam range [-1, 1].
# Dinonaktifkan dengan `USE_NORMALIZE = False`.

# %%
def normalize_audio(y):
    """Peak amplitude normalization ke [-1, 1]. Skip jika USE_NORMALIZE = False."""
    if not USE_NORMALIZE:
        return y
    max_amp = np.max(np.abs(y))
    return y / max_amp if max_amp > 1e-8 else y

print("Fungsi Normalization siap.")

# %% [markdown]
# ## 10. Pipeline Utama: Preprocessing Satu Partisipan
#
# Menggabungkan seluruh fungsi di atas dalam urutan:
# **Load → Diarization → VAD → Noise Reduction → Normalization → Simpan**
#
# Setiap langkah dikontrol oleh flag Toggle di Bagian 3.
# Jika sebuah langkah dinonaktifkan, audio langsung diteruskan ke langkah berikutnya.

# %%
def preprocess_participant(pid, folder_path):
    """
    Full preprocessing pipeline untuk satu partisipan DAIC-WOZ.
    Returns dict info hasil preprocessing (status, durasi, metode, dll).
    """
    audio_path      = os.path.join(folder_path, f"{pid}_AUDIO.wav")
    transcript_path = os.path.join(folder_path, f"{pid}_TRANSCRIPT.csv")
    covarep_path    = os.path.join(folder_path, f"{pid}_COVAREP.csv")
    output_path     = os.path.join(CLEANED_DIR, f"{pid}.wav")

    info = {
        'participant_id'   : pid,
        'status'           : 'ok',
        'dur_original_s'   : 0,
        'dur_after_diar_s' : 0,
        'dur_after_vad_s'  : 0,
        'dur_final_s'      : 0,
        'vad_method'       : 'none',
        'n_segments'       : 0,
        'output_path'      : output_path,
        'error_msg'        : '',
    }

    # Step 1: Load audio
    if not os.path.exists(audio_path):
        info['status'] = 'error'
        info['error_msg'] = "File audio tidak ditemukan"
        return info
    try:
        y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        info['dur_original_s'] = round(len(y) / TARGET_SR, 2)
    except Exception as e:
        info['status'] = 'error'
        info['error_msg'] = f"Gagal load audio: {e}"
        return info

    # Step 2: Speaker Diarization
    if USE_DIARIZATION and os.path.exists(transcript_path):
        segments = get_participant_segments(transcript_path)
        info['n_segments'] = len(segments)
        if segments:
            y_diar = extract_segments_audio(y, TARGET_SR, segments)
            y = y_diar if len(y_diar) > 0 else y
    info['dur_after_diar_s'] = round(len(y) / TARGET_SR, 2)

    # Step 3: VAD — COVAREP VUV (primer)
    if USE_VAD_COVAREP and os.path.exists(covarep_path):
        vuv_mask = get_vuv_mask_from_covarep(covarep_path, len(y))
        y_vad    = apply_vad_covarep(y, vuv_mask)
        if y_vad is not None and len(y_vad) > TARGET_SR:
            y = y_vad
            info['vad_method'] = 'covarep_vuv'

    # Step 3b: VAD — Librosa fallback
    if info['vad_method'] == 'none' and USE_VAD_FALLBACK:
        y = apply_vad_librosa(y)
        info['vad_method'] = 'librosa_energy'

    info['dur_after_vad_s'] = round(len(y) / TARGET_SR, 2)

    # Step 4: Noise Reduction
    y = reduce_noise_audio(y, TARGET_SR)

    # Step 5: Normalization
    y = normalize_audio(y)
    info['dur_final_s'] = round(len(y) / TARGET_SR, 2)

    # Step 6: Validasi minimum durasi output
    if len(y) < TARGET_SR * 2:
        info['status'] = 'warning_too_short'
        info['error_msg'] = f"Audio final terlalu pendek: {info['dur_final_s']}s"

    # Simpan ke cleaned/
    try:
        sf.write(output_path, y, TARGET_SR, subtype='PCM_16')
    except Exception as e:
        info['status'] = 'error'
        info['error_msg'] = f"Gagal simpan: {e}"

    log.info(
        f"PID {pid:4d} | orig {info['dur_original_s']:7.1f}s → "
        f"diar {info['dur_after_diar_s']:7.1f}s → "
        f"vad({info['vad_method']}) {info['dur_after_vad_s']:7.1f}s → "
        f"final {info['dur_final_s']:7.1f}s | {info['status']}"
    )
    return info

print("Pipeline preprocess_participant siap.")

# %% [markdown]
# ## 11. Scan Folder Partisipan DAIC-WOZ
#
# Mencari semua folder dengan format `{ID}_P` di dalam DAIC-WOZ directory.

# %%
def scan_daic_folders(daic_dir):
    """Scan semua folder partisipan format {ID}_P."""
    folders = sorted([
        f for f in os.listdir(daic_dir)
        if os.path.isdir(os.path.join(daic_dir, f)) and f.endswith('_P')
    ])
    result = []
    for folder in folders:
        try:
            pid = int(folder.replace('_P', ''))
            result.append((pid, os.path.join(daic_dir, folder)))
        except ValueError:
            continue
    return result

participants = scan_daic_folders(DAIC_DIR)
print(f"Total partisipan ditemukan: {len(participants)}")

# %% [markdown]
# ## 12. Jalankan Batch Preprocessing
#
# Memproses semua partisipan secara berurutan dengan progress bar.
# Partisipan yang gagal di-skip tanpa menghentikan proses global.
# Hasil log disimpan ke `data/processed/preprocessing_log.csv`.

# %%
results    = []
failed_pids = []

for pid, folder_path in tqdm(participants, desc="Preprocessing", unit="partisipan"):
    info = preprocess_participant(pid, folder_path)
    results.append(info)
    if 'error' in info['status']:
        failed_pids.append(pid)

# %% [markdown]
# ## 13. Simpan Log & Tampilkan Summary
#
# Menyimpan seluruh informasi hasil preprocessing ke CSV dan menampilkan statistik ringkas.

# %%
df_log = pd.DataFrame(results)
log_path = os.path.join(LOG_DIR, 'preprocessing_log.csv')
df_log.to_csv(log_path, index=False)

df_ok   = df_log[df_log['status'] == 'ok']
df_warn = df_log[df_log['status'].str.startswith('warning', na=False)]
df_err  = df_log[df_log['status'].str.startswith('error', na=False)]

print("\n" + "="*60)
print(" SUMMARY PREPROCESSING")
print("="*60)
print(f"  Total diproses   : {len(results)}")
print(f"  Berhasil (ok)    : {len(df_ok)}")
print(f"  Warning          : {len(df_warn)}")
print(f"  Gagal (error)    : {len(df_err)}")
if failed_pids:
    print(f"  PID gagal        : {failed_pids}")

if len(df_ok) > 0:
    print(f"\n  Statistik Durasi (detik):")
    stats = df_ok[['dur_original_s','dur_after_diar_s','dur_after_vad_s','dur_final_s']].describe().round(1)
    print(stats.to_string())

    reduction = ((df_ok['dur_original_s'] - df_ok['dur_final_s']) / df_ok['dur_original_s'] * 100).mean()
    print(f"\n  Rata-rata pengurangan durasi : {reduction:.1f}%")

    print(f"\n  VAD method yang dipakai:")
    for method, count in df_ok['vad_method'].value_counts().items():
        print(f"    {method:25s}: {count} partisipan")

print(f"\n  Log CSV  : {log_path}")
print(f"  Audio    : {CLEANED_DIR}")
print("="*60)
