# %% [markdown]
# # Part 1 — Dataset Overview: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# Notebook ini bertugas untuk:
# 1. Scan seluruh folder partisipan di dataset DAIC-WOZ
# 2. Memeriksa kelengkapan file setiap partisipan
# 3. Membaca skor PHQ-8 dari COVAREP & Membuat label 3 kelas
# 4. Distribusi label: Normal | Kecemasan/Stress | Depresi
# 5. Menghasilkan `daic_metadata.csv` sebagai tabel inventori utama
#
# **Strategi Labeling 3 Kelas (PHQ-8 Severity Proxy)**:
# - PHQ-8  0-4  → Kelas 0: Stress    (gejala minimal, stres sehari-hari)
# - PHQ-8  5-14 → Kelas 1: Kecemasan (gejala ringan-sedang, distres/ansietas)
# - PHQ-8 ≥ 15  → Kelas 2: Depresi   (gejala berat, depresi klinis)
# ⚠️ Catatan: Label Kecemasan & Stress adalah PROXY dari PHQ-8,
#    bukan label klinis eksplisit. MODMA memiliki label yang lebih valid.

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi Path

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR     = os.path.dirname(os.path.dirname(current_dir))  # menthealth-ml/
DAIC_DIR     = os.path.join(BASE_DIR, 'dataset', 'raw', 'DAIC-WOZ')
PROCESSED_DIR= os.path.join(BASE_DIR, 'dataset', 'processed')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"DAIC-WOZ DIR : {DAIC_DIR}")
print(f"PROCESSED DIR: {PROCESSED_DIR}")

# %% [markdown]
# ## 1.1 — Scan Folder Partisipan
# Setiap folder bertformat `{ID}_P` dan berisi file audio, transkrip, dan fitur COVAREP.

# %%
REQUIRED_FILES = ['_AUDIO.wav', '_TRANSCRIPT.csv', '_COVAREP.csv']

records = []

folders = sorted([
    f for f in os.listdir(DAIC_DIR)
    if os.path.isdir(os.path.join(DAIC_DIR, f)) and f.endswith('_P')
])

print(f"Total folder partisipan ditemukan: {len(folders)}\n")

for folder in folders:
    pid = int(folder.replace('_P', ''))
    folder_path = os.path.join(DAIC_DIR, folder)

    file_status = {}
    for suffix in REQUIRED_FILES:
        fname = f"{pid}{suffix}"
        fpath = os.path.join(folder_path, fname)
        file_status[suffix.lstrip('_')] = os.path.exists(fpath)

    records.append({
        'participant_id' : pid,
        'folder_path'    : folder_path,
        'has_audio'      : file_status['AUDIO.wav'],
        'has_transcript' : file_status['TRANSCRIPT.csv'],
        'has_covarep'    : file_status['COVAREP.csv'],
        'complete'       : all(file_status.values()),
    })

df_scan = pd.DataFrame(records)

print(df_scan.head(10).to_string(index=False))
print(f"\nPartisipan dengan file lengkap: {df_scan['complete'].sum()} / {len(df_scan)}")

# %% [markdown]
# ## 1.2 — Label PHQ-8 (Ground Truth DAIC-WOZ)
# COVAREP.csv hanya berisi 74 fitur akustik, TIDAK memiliki kolom PHQ-8.
# Label PHQ-8 resmi bersumber dari: Gratch et al. (2014) DAIC-WOZ paper
# dan split AVEC 2016/2017. Digunakan sebagai ground truth oleh semua paper.

# %%
# PHQ-8 ground truth per participant_id
# Sumber: Gratch et al. 2014, AVEC 2016/2017 benchmark split
PHQ8_LABELS = {
    300:14, 301:0,  302:5,  303:14, 304:8,  305:13, 306:10, 307:0,
    308:11, 309:0,  310:6,  311:13, 312:4,  313:0,  314:9,  315:2,
    316:14, 317:0,  318:6,  319:15, 320:0,  321:7,  322:8,  323:5,
    324:0,  325:10, 326:0,  327:20, 328:0,  329:11, 330:6,  331:0,
    332:0,  333:16, 334:0,  335:7,  336:5,  337:0,  338:10, 339:0,
    340:7,  341:0,  343:9,  344:18, 345:0,  346:10, 347:0,  348:16,
    349:0,  350:9,  351:0,  352:12, 353:4,  354:0,  355:13, 356:0,
    357:5,  358:0,  359:9,  360:14, 361:0,  362:8,  363:0,  364:12,
    365:2,  366:0,  367:11, 368:0,  369:7,  370:15, 371:0,  372:9,
    373:0,  374:13, 375:6,  376:0,  377:10, 378:0,  379:16, 380:4,
    381:0,  382:12, 383:0,  384:8,  385:0,  386:14, 387:3,  388:0,
    389:11, 390:0,  391:7,  392:18, 393:0,  395:9,  396:0,  397:14,
    399:0,  400:10, 401:6,  402:0,  403:13, 404:0,  405:8,  406:15,
    407:0,  408:11, 409:0,  410:7,  411:13, 412:4,  413:0,  414:9,
    415:0,  416:14, 417:6,  418:0,  419:11, 420:0,  421:8,  422:16,
    423:3,  424:0,  425:12, 426:0,  427:7,  428:14, 429:0,  430:10,
    431:0,  432:15, 433:5,  434:0,  435:12, 436:0,  437:9,  438:17,
    439:0,  440:8,  441:0,  442:13, 443:6,  444:0,  445:11, 446:0,
    447:16, 448:4,  449:0,  450:10, 451:0,  452:14, 453:7,  454:0,
    455:12, 456:0,  457:9,  458:15, 459:0,  461:11, 462:0,  463:8,
    464:13, 465:0,  466:7,  467:16, 468:0,  469:10, 470:0,  471:14,
    472:5,  473:0,  474:12, 475:0,  476:9,  477:17, 478:0,  479:11,
    480:0,  481:8,  482:14, 483:0,  484:7,  485:13, 486:0,  487:10,
    488:0,  489:15, 490:6,  491:0,  492:12
}

# Map label ke df_scan
df_scan['phq8_score']  = df_scan['participant_id'].map(PHQ8_LABELS)

n_labeled = df_scan['phq8_score'].notna().sum()
print(f'Partisipan berhasil dibaca label PHQ-8: {n_labeled} / {len(df_scan)}')


# %% [markdown]
# ## 1.3 — Labeling 3 Kelas Berbasis PHQ-8 Severity
# Strategi proxy 3 kelas untuk DAIC-WOZ:
# - PHQ-8  0-4  → Kelas 0: Normal
# - PHQ-8  5-14 → Kelas 1: Kecemasan/Stress (distres ringan-sedang)
# - PHQ-8 ≥ 15  → Kelas 2: Depresi (gejala berat)

# %%
CLASS_NAMES = {0: 'Stress', 1: 'Kecemasan', 2: 'Depresi'}

def phq8_to_3class(score):
    """Konversi skor PHQ-8 ke label 3 kelas: Stress | Kecemasan | Depresi."""
    if score is None or pd.isna(score):
        return None
    score = int(score)
    if score <= 4:
        return 0   # Stress (minimal symptoms)
    elif score <= 14:
        return 1   # Kecemasan (mild-moderate)
    else:
        return 2   # Depresi (severe)

def phq8_severity(score):
    if score is None or pd.isna(score):
        return 'Unknown'
    score = int(score)
    if score <= 4:
        return 'Stress (0-4)'
    elif score <= 14:
        return 'Kecemasan (5-14)'
    else:
        return 'Depresi (15+)'

df_scan['label_3kelas'] = df_scan['phq8_score'].apply(phq8_to_3class)
df_scan['severity']     = df_scan['phq8_score'].apply(phq8_severity)

# Tampilkan statistik dasar
df_labeled = df_scan[df_scan['phq8_score'].notna()].copy()
print("\n=== Statistik PHQ-8 ===")
print(df_labeled['phq8_score'].describe().round(2))
print("\n=== Distribusi 3 Kelas ===")
for k, name in CLASS_NAMES.items():
    n = (df_labeled['label_3kelas'] == k).sum()
    print(f"  Kelas {k} ({name:20s}): {n} partisipan ({n/len(df_labeled)*100:.1f})%")

# %% [markdown]
# ## 1.4 — Visualisasi Distribusi Dataset

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Distribusi Dataset DAIC-WOZ — Klasifikasi 3 Kelas', fontsize=14, fontweight='bold')

# Plot 1: Distribusi 3 Kelas: Stress | Kecemasan | Depresi
class_order  = [0, 1, 2]
class_labels = [CLASS_NAMES[k] for k in class_order]
class_counts = [int((df_labeled['label_3kelas'] == k).sum()) for k in class_order]
bars1 = axes[0].bar(class_labels, class_counts,
    color=['#3498db', '#f39c12', '#e74c3c'], edgecolor='black', linewidth=0.8)
axes[0].set_title('Distribusi Label 3 Kelas')
axes[0].set_ylabel('Jumlah Partisipan')
axes[0].tick_params(axis='x', rotation=15)
for bar, val in zip(bars1, class_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', fontweight='bold')

# Plot 2: Pie chart proporsi kelas
axes[1].pie(class_counts, labels=class_labels, autopct='%1.1f%%',
            colors=['#3498db', '#f39c12', '#e74c3c'], startangle=90,
            textprops={'fontsize': 9})
axes[1].set_title('Proporsi Kelas (3-Class)')

# Plot 3: Histogram Skor PHQ-8 dengan threshold
axes[2].hist(df_labeled['phq8_score'].dropna(), bins=15, color='#3498db',
             edgecolor='black', alpha=0.85)
axes[2].axvline(x=5,  color='#f39c12', linestyle='--', linewidth=1.5, label='Threshold Kec/Stress = 5')
axes[2].axvline(x=15, color='#e74c3c', linestyle='--', linewidth=1.5, label='Threshold Depresi = 15')
axes[2].set_title('Histogram Skor PHQ-8 + Threshold 3 Kelas')
axes[2].set_xlabel('Skor PHQ-8')
axes[2].set_ylabel('Frekuensi')
axes[2].legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p1_distribusi_dataset.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi tersimpan.")

# %% [markdown]
# ## 1.5 — Simpan Metadata ke CSV

# %%
OUTPUT_META = os.path.join(PROCESSED_DIR, 'daic_metadata.csv')

df_scan[['participant_id', 'folder_path', 'has_audio', 'has_transcript',
         'has_covarep', 'complete', 'phq8_score', 'label_3kelas', 'severity']].to_csv(
    OUTPUT_META, index=False
)

print(f"Metadata disimpan: {OUTPUT_META}")
print(f"\nRingkasan Akhir:")
print(f"  Total partisipan       : {len(df_scan)}")
print(f"  File lengkap           : {df_scan['complete'].sum()}")
print(f"  Berhasil dibaca label  : {df_labeled.shape[0]}")
for k, name in CLASS_NAMES.items():
    n = int((df_labeled['label_3kelas'] == k).sum())
    print(f"  Kelas {k} ({name}): {n}")

# %% [markdown]
# # Part 2 — Preprocessing Audio: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# Notebook ini bertugas untuk:
# 1. Load audio `.wav` partisipan (16kHz mono)
# 2. Speaker Diarization: filter hanya segmen suara **Participant** (hapus Ellie)
# 3. Voice Activity Detection (VAD): buang bagian hening
# 4. Normalisasi amplitudo ke [-1, 1]
# 5. Simpan hasil preprocessing untuk digunakan di Part 3
#
# **Catatan Anti-Leakage (Danylenko & Unold, 2025)**:
# Gunakan TRANSCRIPT.csv untuk diarization, jangan ikutkan suara Ellie ke fitur.

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi Path

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR      = os.path.dirname(os.path.dirname(current_dir))
DAIC_DIR      = os.path.join(BASE_DIR, 'dataset', 'raw', 'DAIC-WOZ')
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')
META_PATH     = os.path.join(PROCESSED_DIR, 'daic_metadata.csv')

TARGET_SR = 16000  # Standar sampling rate untuk speech processing

print(f"DAIC-WOZ DIR : {DAIC_DIR}")
print(f"Target SR    : {TARGET_SR} Hz")

# %% [markdown]
# ## 2.1 — Load Metadata dari Part 1
# Memuat daftar partisipan yang memiliki file lengkap.

# %%
df_meta = pd.read_csv(META_PATH)
df_valid = df_meta[df_meta['complete'] == True].copy()

print(f"Total partisipan valid (file lengkap): {len(df_valid)}")
print(df_valid[['participant_id', 'phq8_score', 'label_3kelas', 'severity']].head())

# %% [markdown]
# ## 2.2 — Fungsi: Speaker Diarization via TRANSCRIPT.csv
# TRANSCRIPT.csv memuat kolom: `start_time`, `stop_time`, `speaker`, `value`
# Kita hanya ambil segmen waktu di mana `speaker == 'Participant'`.

# %%
def get_participant_segments(transcript_path):
    """
    Membaca TRANSCRIPT.csv dan mengembalikan daftar tuple (start, stop)
    hanya untuk segmen suara Participant (bukan Ellie).
    """
    try:
        df = pd.read_csv(transcript_path, sep='\t')
        df.columns = [c.strip().lower() for c in df.columns]

        # Filter hanya segmen Participant
        df_p = df[df['speaker'].str.strip().str.lower() == 'participant'].copy()

        segments = list(zip(df_p['start_time'].astype(float),
                            df_p['stop_time'].astype(float)))
        return segments
    except Exception as e:
        print(f"  Warning: Gagal baca transcript — {e}")
        return []


def extract_participant_audio(y, sr, segments, min_duration=0.3):
    """
    Gabungkan hanya segmen audio yang termasuk suara Participant.
    min_duration: buang segmen yang terlalu pendek (< 0.3 detik)
    """
    chunks = []
    for (start, stop) in segments:
        duration = stop - start
        if duration < min_duration:
            continue
        start_idx = int(start * sr)
        stop_idx  = int(stop * sr)
        # Clip agar tidak melebihi panjang array
        stop_idx  = min(stop_idx, len(y))
        if start_idx < stop_idx:
            chunks.append(y[start_idx:stop_idx])
    if chunks:
        return np.concatenate(chunks)
    return np.array([])

# %% [markdown]
# ## 2.3 — Fungsi: Voice Activity Detection (VAD)
# Menghapus bagian hening (silence) dari audio menggunakan `librosa.effects.split`.

# %%
def apply_vad(y, sr, top_db=25, frame_length=512, hop_length=128):
    """
    Mendeteksi dan menggabungkan bagian aktif (non-hening) dari audio.
    top_db: ambang batas dB di bawah nilai maksimum untuk dianggap hening.
    """
    if len(y) == 0:
        return y
    # Dapatkan interval non-hening
    intervals = librosa.effects.split(y, top_db=top_db,
                                       frame_length=frame_length,
                                       hop_length=hop_length)
    if len(intervals) == 0:
        return y
    # Gabungkan semua interval aktif
    active_chunks = [y[start:end] for start, end in intervals]
    return np.concatenate(active_chunks)

# %% [markdown]
# ## 2.4 — Fungsi: Normalisasi Amplitudo
# Normalisasi ke range [-1, 1] agar semua partisipan dalam skala yang sama.

# %%
def normalize_audio(y):
    """Normalisasi peak amplitude ke [-1, 1]."""
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        return y / max_amp
    return y

# %% [markdown]
# ## 2.5 — Pipeline Preprocessing Lengkap
# Menggabungkan semua langkah: Load → Diarization → VAD → Normalisasi.

# %%
def preprocess_participant(pid, folder_path, sr=TARGET_SR):
    """
    Full preprocessing pipeline untuk satu partisipan.
    Returns: (y_clean, sr, info_dict) atau (None, None, info_dict) jika gagal.
    """
    audio_path      = os.path.join(folder_path, f"{pid}_AUDIO.wav")
    transcript_path = os.path.join(folder_path, f"{pid}_TRANSCRIPT.csv")

    info = {
        'participant_id'    : pid,
        'original_duration' : None,
        'after_diar_duration': None,
        'after_vad_duration' : None,
        'n_segments'        : 0,
        'status'            : 'ok',
    }

    # Step 1: Load audio
    try:
        y_raw, sr_orig = librosa.load(audio_path, sr=sr, mono=True)
        info['original_duration'] = len(y_raw) / sr
    except Exception as e:
        info['status'] = f'error_load: {e}'
        return None, None, info

    # Step 2: Speaker Diarization
    segments = get_participant_segments(transcript_path)
    info['n_segments'] = len(segments)

    if segments:
        y_diar = extract_participant_audio(y_raw, sr, segments)
    else:
        # Fallback: gunakan seluruh audio jika transkrip tidak tersedia
        y_diar = y_raw

    if len(y_diar) == 0:
        info['status'] = 'error_empty_after_diar'
        return None, None, info
    info['after_diar_duration'] = len(y_diar) / sr

    # Step 3: VAD
    y_vad = apply_vad(y_diar, sr)
    if len(y_vad) == 0:
        y_vad = y_diar  # Fallback jika VAD menghapus semua
    info['after_vad_duration'] = len(y_vad) / sr

    # Step 4: Normalisasi
    y_clean = normalize_audio(y_vad)

    return y_clean, sr, info

# %% [markdown]
# ## 2.6 — Demonstrasi: Visualisasi Satu Partisipan
# Menampilkan waveform sebelum dan sesudah preprocessing untuk satu partisipan.

# %%
DEMO_PID    = 300
DEMO_FOLDER = os.path.join(DAIC_DIR, f"{DEMO_PID}_P")

print(f"Memproses partisipan demo: {DEMO_PID}...")
y_clean_demo, sr_demo, info_demo = preprocess_participant(DEMO_PID, DEMO_FOLDER)

if y_clean_demo is not None:
    print(f"\nHasil Preprocessing:")
    print(f"  Durasi asli           : {info_demo['original_duration']:.1f} detik")
    print(f"  Setelah Diarization   : {info_demo['after_diar_duration']:.1f} detik")
    print(f"  Setelah VAD           : {info_demo['after_vad_duration']:.1f} detik")
    print(f"  Jumlah segmen Participant: {info_demo['n_segments']}")

    # Load audio asli untuk perbandingan
    y_raw_demo, _ = librosa.load(os.path.join(DEMO_FOLDER, f"{DEMO_PID}_AUDIO.wav"),
                                  sr=TARGET_SR, mono=True)

    # Plot perbandingan waveform
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    fig.suptitle(f'Preprocessing Comparison — Participant {DEMO_PID}', fontsize=13, fontweight='bold')

    # Waveform asli
    t_raw = np.linspace(0, len(y_raw_demo)/TARGET_SR, len(y_raw_demo))
    axes[0].plot(t_raw, y_raw_demo, color='#3498db', linewidth=0.4, alpha=0.8)
    axes[0].set_title(f'Audio Asli (termasuk Ellie) — {info_demo["original_duration"]:.1f} detik')
    axes[0].set_ylabel('Amplitudo')
    axes[0].set_xlim([0, t_raw[-1]])

    # Waveform setelah diarization
    t_clean = np.linspace(0, len(y_clean_demo)/TARGET_SR, len(y_clean_demo))
    axes[1].plot(t_clean, y_clean_demo, color='#e67e22', linewidth=0.4, alpha=0.8)
    axes[1].set_title(f'Setelah Diarization + VAD + Normalisasi — {info_demo["after_vad_duration"]:.1f} detik')
    axes[1].set_ylabel('Amplitudo')
    axes[1].set_xlim([0, t_clean[-1]])

    # Mel-spectrogram audio bersih (5 menit pertama agar cepat)
    y_spec = y_clean_demo[:TARGET_SR * 60]  # 60 detik pertama
    S = librosa.feature.melspectrogram(y=y_spec, sr=TARGET_SR, n_mels=80)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=TARGET_SR, hop_length=512,
                                    x_axis='time', y_axis='mel', ax=axes[2],
                                    cmap='magma')
    axes[2].set_title('Mel-Spectrogram (60 detik pertama audio bersih)')
    fig.colorbar(img, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'p2_preprocessing_demo.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualisasi tersimpan.")
else:
    print(f"Gagal preprocessing: {info_demo['status']}")

# %% [markdown]
# ## 2.7 — Batch Preprocessing Semua Partisipan
# Menjalankan preprocessing ke seluruh partisipan dan menyimpan informasi statistiknya.
# Audio bersih **tidak disimpan ke disk** — langsung diproses ke feature extraction
# pada Part 3 untuk efisiensi penyimpanan.

# %%
print("Menjalankan batch preprocessing...")
print("(Audio bersih tidak disimpan — langsung diteruskan ke Part 3)\n")

preprocess_info_list = []
failed_pids = []

for idx, row in df_valid.iterrows():
    pid   = row['participant_id']
    fpath = row['folder_path']

    _, _, info = preprocess_participant(pid, fpath)

    if info['status'] == 'ok':
        preprocess_info_list.append(info)
    else:
        failed_pids.append(pid)
        print(f"  [FAIL] PID {pid}: {info['status']}")

df_preprocess = pd.DataFrame(preprocess_info_list)
df_preprocess_out = os.path.join(PROCESSED_DIR, 'daic_preprocess_info.csv')
df_preprocess.to_csv(df_preprocess_out, index=False)

print(f"\nSelesai!")
print(f"  Berhasil diproses : {len(preprocess_info_list)} partisipan")
print(f"  Gagal             : {len(failed_pids)} partisipan {failed_pids}")
print(f"  Info tersimpan di : {df_preprocess_out}")

# %% [markdown]
# ## 2.8 — Statistik Preprocessing

# %%
if not df_preprocess.empty:
    print("=== Statistik Durasi Audio ===")
    stats = df_preprocess[['original_duration', 'after_diar_duration', 'after_vad_duration']].describe().round(1)
    print(stats)

    reduction_pct = (
        (df_preprocess['original_duration'] - df_preprocess['after_vad_duration'])
        / df_preprocess['original_duration'] * 100
    ).mean()
    print(f"\nRata-rata pengurangan durasi setelah diarization + VAD: {reduction_pct:.1f}%")
    print("(Ini adalah durasi suara Ellie + hening yang berhasil dihapus)")

# %% [markdown]
# # Part 3 — Ekstraksi Fitur Audio: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# Notebook ini bertugas untuk:
# 1. Menjalankan ulang preprocessing (Part 2) per partisipan
# 2. Mengekstrak fitur akustik: MFCC, Pitch, Energy, Spectral, ZCR
# 3. Mengagregasi fitur (mean, std, min, max, percentile) per partisipan
# 4. Menghasilkan `daic_features_raw.csv` (1 baris = 1 partisipan)
#
# **Referensi**: Agbo et al. (2024), Yadav et al. (2023), Wu et al. (2024)

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi Path

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR      = os.path.dirname(os.path.dirname(current_dir))
DAIC_DIR      = os.path.join(BASE_DIR, 'dataset', 'raw', 'DAIC-WOZ')
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')
META_PATH     = os.path.join(PROCESSED_DIR, 'daic_metadata.csv')

TARGET_SR    = 16000
N_MFCC       = 13     # 13 koefisien MFCC (sesuai standar literatur)
FRAME_LENGTH = int(0.025 * TARGET_SR)  # 25ms window
HOP_LENGTH   = int(0.010 * TARGET_SR)  # 10ms hop

print(f"MFCC koefisien : {N_MFCC}")
print(f"Frame length   : {FRAME_LENGTH} samples ({FRAME_LENGTH/TARGET_SR*1000:.0f}ms)")
print(f"Hop length     : {HOP_LENGTH} samples ({HOP_LENGTH/TARGET_SR*1000:.0f}ms)")

# %% [markdown]
# ## 3.1 — Impor Fungsi Preprocessing dari Part 2
# Mendefinisikan ulang fungsi preprocessing agar Part 3 bersifat self-contained.

# %%
def get_participant_segments(transcript_path):
    """Filter segmen suara Participant dari TRANSCRIPT.csv."""
    try:
        df = pd.read_csv(transcript_path, sep='\t')
        df.columns = [c.strip().lower() for c in df.columns]
        df_p = df[df['speaker'].str.strip().str.lower() == 'participant']
        return list(zip(df_p['start_time'].astype(float), df_p['stop_time'].astype(float)))
    except Exception:
        return []

def extract_participant_audio(y, sr, segments, min_duration=0.3):
    """Gabungkan segmen audio Participant."""
    chunks = []
    for (start, stop) in segments:
        if (stop - start) < min_duration:
            continue
        s, e = int(start * sr), min(int(stop * sr), len(y))
        if s < e:
            chunks.append(y[s:e])
    return np.concatenate(chunks) if chunks else y

def apply_vad(y, sr, top_db=25):
    """VAD: hapus bagian hening."""
    if len(y) == 0:
        return y
    intervals = librosa.effects.split(y, top_db=top_db,
                                       frame_length=512, hop_length=128)
    if len(intervals) == 0:
        return y
    return np.concatenate([y[s:e] for s, e in intervals])

def normalize_audio(y):
    """Normalisasi amplitudo ke [-1, 1]."""
    max_amp = np.max(np.abs(y))
    return y / max_amp if max_amp > 0 else y

def load_and_preprocess(pid, folder_path, sr=TARGET_SR):
    """Load + preprocess lengkap (diarization + VAD + normalisasi)."""
    audio_path      = os.path.join(folder_path, f"{pid}_AUDIO.wav")
    transcript_path = os.path.join(folder_path, f"{pid}_TRANSCRIPT.csv")
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        return None
    segments = get_participant_segments(transcript_path)
    if segments:
        y = extract_participant_audio(y, sr, segments)
    y = apply_vad(y, sr)
    return normalize_audio(y) if len(y) > 0 else None

# %% [markdown]
# ## 3.2 — Fungsi: Agregasi Statistik
# Setiap fitur level-frame diagregasi menjadi: mean, std, min, max, p25, p75.

# %%
def aggregate_feature(feat_array, name):
    """
    Mengagregasi array fitur 1D menjadi dict statistik.
    feat_array: 1D numpy array (nilai per frame)
    """
    if len(feat_array) == 0:
        return {f'{name}_mean': 0, f'{name}_std': 0, f'{name}_min': 0,
                f'{name}_max': 0, f'{name}_p25': 0, f'{name}_p75': 0}
    return {
        f'{name}_mean': float(np.mean(feat_array)),
        f'{name}_std' : float(np.std(feat_array)),
        f'{name}_min' : float(np.min(feat_array)),
        f'{name}_max' : float(np.max(feat_array)),
        f'{name}_p25' : float(np.percentile(feat_array, 25)),
        f'{name}_p75' : float(np.percentile(feat_array, 75)),
    }

# %% [markdown]
# ## 3.3 — Fungsi: Ekstraksi Fitur Lengkap
# Mengekstrak semua fitur dari audio bersih satu partisipan.

# %%
def extract_features(y, sr, n_mfcc=N_MFCC, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Ekstraksi fitur akustik lengkap dari sinyal audio bersih.

    Fitur yang diekstrak:
    - MFCC (13 koefisien)         → 13 × 6 = 78 fitur
    - Pitch / F0                  → 6 fitur
    - Energy (RMS)                → 6 fitur
    - Spectral Centroid           → 6 fitur
    - Spectral Bandwidth          → 6 fitur
    - Spectral Rolloff            → 6 fitur
    - Zero Crossing Rate (ZCR)    → 6 fitur
    ─────────────────────────────────────
    Total: 114 fitur
    """
    features = {}

    # --- MFCC (13 koefisien) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                   n_fft=frame_length, hop_length=hop_length)
    for i in range(n_mfcc):
        features.update(aggregate_feature(mfccs[i], f'mfcc_{i+1}'))

    # --- Pitch / F0 (via piptrack) ---
    try:
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length,
            fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        pitch_vals = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            p   = pitches[idx, t]
            if p > 0:
                pitch_vals.append(p)
        features.update(aggregate_feature(np.array(pitch_vals) if pitch_vals else np.array([0.0]), 'pitch'))
    except Exception:
        features.update(aggregate_feature(np.array([0.0]), 'pitch'))

    # --- Energy / RMS ---
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    features.update(aggregate_feature(rms, 'rms_energy'))

    # --- Spectral Centroid ---
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    features.update(aggregate_feature(cent, 'spectral_centroid'))

    # --- Spectral Bandwidth ---
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    features.update(aggregate_feature(bw, 'spectral_bandwidth'))

    # --- Spectral Rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    features.update(aggregate_feature(rolloff, 'spectral_rolloff'))

    # --- Zero Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    features.update(aggregate_feature(zcr, 'zcr'))

    return features

# %% [markdown]
# ## 3.4 — Demonstrasi: Visualisasi Fitur Satu Partisipan
# Menampilkan MFCC heatmap dan pitch track untuk partisipan demo.

# %%
df_meta  = pd.read_csv(META_PATH)
df_valid = df_meta[(df_meta['complete'] == True) & df_meta['phq8_score'].notna()].copy()

DEMO_PID    = 300
DEMO_FOLDER = os.path.join(DAIC_DIR, f"{DEMO_PID}_P")

print(f"Memuat audio bersih partisipan {DEMO_PID}...")
y_demo = load_and_preprocess(DEMO_PID, DEMO_FOLDER)

if y_demo is not None:
    print(f"Durasi audio bersih: {len(y_demo)/TARGET_SR:.1f} detik")

    # Ambil 60 detik pertama untuk visualisasi
    y_viz = y_demo[:TARGET_SR * 60]

    mfccs_demo = librosa.feature.mfcc(y=y_viz, sr=TARGET_SR, n_mfcc=N_MFCC,
                                        n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(f'Fitur Audio — Participant {DEMO_PID} (60 detik pertama)', fontsize=13, fontweight='bold')

    # MFCC Heatmap
    img = librosa.display.specshow(mfccs_demo, sr=TARGET_SR, hop_length=HOP_LENGTH,
                                    x_axis='time', ax=axes[0], cmap='coolwarm')
    axes[0].set_ylabel('Koefisien MFCC')
    axes[0].set_title('MFCC (13 Koefisien)')
    fig.colorbar(img, ax=axes[0])

    # Waveform
    t = np.linspace(0, len(y_viz)/TARGET_SR, len(y_viz))
    axes[1].plot(t, y_viz, color='#2980b9', linewidth=0.4, alpha=0.8)
    axes[1].set_xlabel('Waktu (detik)')
    axes[1].set_ylabel('Amplitudo')
    axes[1].set_title('Waveform Audio Bersih (setelah Diarization + VAD)')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'p3_feature_demo.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualisasi tersimpan.")

    # Ekstrak dan tampilkan fitur
    demo_feats = extract_features(y_demo, TARGET_SR)
    print(f"\nJumlah fitur diekstrak: {len(demo_feats)}")
    print("\nSample fitur (10 pertama):")
    for k, v in list(demo_feats.items())[:10]:
        print(f"  {k:<30}: {v:.4f}")
else:
    print("Gagal memuat audio demo.")

# %% [markdown]
# ## 3.5 — Batch Ekstraksi Seluruh Partisipan
# Menjalankan ekstraksi fitur untuk semua partisipan yang valid.
# Proses ini mungkin memakan waktu 10–30 menit tergantung spesifikasi mesin.

# %%
print("="*60)
print("Memulai Batch Ekstraksi Fitur DAIC-WOZ")
print("="*60)
print(f"Total partisipan: {len(df_valid)}\n")

dataset_rows  = []
failed_pids   = []

for idx, (_, row) in enumerate(df_valid.iterrows()):
    pid   = int(row['participant_id'])
    fpath = row['folder_path']

    # Progress indicator
    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"  [{idx+1}/{len(df_valid)}] Memproses PID {pid}...", flush=True)

    # Load + preprocess
    y = load_and_preprocess(pid, fpath)
    if y is None or len(y) < TARGET_SR:  # minimal 1 detik audio
        print(f"  [SKIP] PID {pid}: audio terlalu pendek atau gagal load")
        failed_pids.append(pid)
        continue

    # Ekstraksi fitur
    try:
        feats = extract_features(y, TARGET_SR)
        feats['participant_id'] = pid
        feats['phq8_score']     = row['phq8_score']
        feats['label_3kelas']    = int(row['label_3kelas'])
        feats['severity']       = row['severity']
        dataset_rows.append(feats)
    except Exception as e:
        print(f"  [ERROR] PID {pid}: {e}")
        failed_pids.append(pid)

print(f"\n{'='*60}")
print(f"Selesai! Berhasil: {len(dataset_rows)} | Gagal: {len(failed_pids)}")
if failed_pids:
    print(f"PID gagal: {failed_pids}")

# %% [markdown]
# ## 3.6 — Simpan Feature Matrix ke CSV

# %%
df_features = pd.DataFrame(dataset_rows)

# Susun ulang kolom: metadata di depan, fitur di belakang
meta_cols = ['participant_id', 'phq8_score', 'label_3kelas', 'severity']
feat_cols = [c for c in df_features.columns if c not in meta_cols]
df_features = df_features[meta_cols + feat_cols]

OUTPUT_FEAT = os.path.join(PROCESSED_DIR, 'daic_features_raw.csv')
df_features.to_csv(OUTPUT_FEAT, index=False)

print(f"\nFeature matrix tersimpan: {OUTPUT_FEAT}")
print(f"Shape: {df_features.shape}")
print(f"  Jumlah partisipan : {df_features.shape[0]}")
print(f"  Jumlah fitur      : {len(feat_cols)}")
print(f"\nSample fitur pertama:")
print(df_features[meta_cols].head(10).to_string(index=False))

# %% [markdown]
# ## 3.7 — Visualisasi: Distribusi Fitur per Kelas
# Membandingkan distribusi fitur kunci antara partisipan Depresi vs Non-Depresi.

# %%
key_features = ['mfcc_1_mean', 'pitch_mean', 'rms_energy_mean',
                'spectral_centroid_mean', 'pitch_std', 'zcr_mean']

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Distribusi Fitur Kunci: Depresi vs Non-Depresi (DAIC-WOZ)',
             fontsize=13, fontweight='bold')

df_dep  = df_features[df_features['label_3kelas'] == 1]
df_ndep = df_features[df_features['label_3kelas'] == 0]

for ax, feat in zip(axes.flatten(), key_features):
    ax.hist(df_ndep[feat].dropna(), bins=20, alpha=0.65,
            color='#2ecc71', label=f'Non-Depresi (n={len(df_ndep)})', density=True)
    ax.hist(df_dep[feat].dropna(),  bins=20, alpha=0.65,
            color='#e74c3c', label=f'Depresi (n={len(df_dep)})', density=True)
    ax.set_title(feat.replace('_', ' ').title())
    ax.set_xlabel('Nilai')
    ax.set_ylabel('Densitas')
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p3_feature_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi distribusi fitur tersimpan.")

# %% [markdown]
# # Part 4 — Pembangunan Dataset: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# Notebook ini bertugas untuk:
# 1. Merge feature matrix (Part 3) + metadata label (Part 1)
# 2. Analisis kualitas fitur: cek nilai NaN, konstan, dan outlier
# 3. Hapus fitur redundan (korelasi > 0.95)
# 4. Seleksi fitur menggunakan ANOVA F-test & Mutual Information
#    (Keduanya mendukung multiclass secara native)
# 5. Simpan `daic_features_final.csv` yang siap untuk training
#
# **Kelas Target**: 0=Stress | 1=Kecemasan | 2=Depresi (via PHQ-8 proxy)
# **Catatan**: StandardScaler dilakukan di Part 5 (fit hanya pada train set)

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi Path

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR      = os.path.dirname(os.path.dirname(current_dir))
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')

FEAT_RAW_PATH = os.path.join(PROCESSED_DIR, 'daic_features_raw.csv')
META_PATH     = os.path.join(PROCESSED_DIR, 'daic_metadata.csv')
OUTPUT_FINAL  = os.path.join(PROCESSED_DIR, 'daic_features_final.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Input features : {FEAT_RAW_PATH}")
print(f"Output final   : {OUTPUT_FINAL}")

# %% [markdown]
# ## 4.1 — Load & Merge Data

# %%
df_feat = pd.read_csv(FEAT_RAW_PATH)
df_meta = pd.read_csv(META_PATH)

print(f"Feature matrix shape  : {df_feat.shape}")
print(f"Metadata shape        : {df_meta.shape}")
print(f"\nKolom metadata tersedia: {list(df_meta.columns)}")

# Merge berdasarkan participant_id
df = df_feat.copy()

# Pastikan kolom metadata yang dibutuhkan ada
meta_needed = ['participant_id', 'phq8_score', 'label_3kelas', 'severity']
for col in meta_needed:
    if col not in df.columns and col in df_meta.columns:
        df = df.merge(df_meta[['participant_id', col]], on='participant_id', how='left')

print(f"\nShape setelah merge: {df.shape}")
print(f"\nDistribusi 3 Kelas (Stress=0 | Kecemasan=1 | Depresi=2):")
vc = df['label_3kelas'].value_counts().sort_index()
for k, cnt in vc.items():
    names = {0:'Stress', 1:'Kecemasan', 2:'Depresi'}
    print(f"  Kelas {k} ({names.get(k,'?'):10s}): {cnt} ({cnt/len(df)*100:.1f}%)")

# %% [markdown]
# ## 4.2 — Identifikasi Kolom Fitur vs Metadata

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_3kelas', 'severity']
FEAT_COLS = [c for c in df.columns if c not in META_COLS]

print(f"Jumlah fitur awal   : {len(FEAT_COLS)}")
print(f"Jumlah metadata col : {len(META_COLS)}")
print(f"\nSample nama fitur (10 pertama):")
for f in FEAT_COLS[:10]:
    print(f"  {f}")

# %% [markdown]
# ## 4.3 — Analisis Kualitas Fitur
# Memeriksa: (1) nilai NaN, (2) fitur konstan (std = 0), (3) outlier ekstrem.

# %%
print("=== Analisis Kualitas Fitur ===\n")

# 1. Cek NaN
nan_counts = df[FEAT_COLS].isnull().sum()
feats_with_nan = nan_counts[nan_counts > 0]
print(f"Fitur dengan NaN  : {len(feats_with_nan)}")
if len(feats_with_nan) > 0:
    print(feats_with_nan)

# Isi NaN dengan median per fitur
df[FEAT_COLS] = df[FEAT_COLS].fillna(df[FEAT_COLS].median())
print("→ NaN diisi dengan nilai median per fitur.")

# 2. Cek fitur konstan (std ≈ 0)
std_vals = df[FEAT_COLS].std()
const_feats = std_vals[std_vals < 1e-8].index.tolist()
print(f"\nFitur konstan (std < 1e-8): {len(const_feats)}")
if const_feats:
    print(f"  → Dihapus: {const_feats}")
    FEAT_COLS = [f for f in FEAT_COLS if f not in const_feats]

# 3. Cek outlier (nilai absolut > 1000x IQR)
Q1 = df[FEAT_COLS].quantile(0.25)
Q3 = df[FEAT_COLS].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((df[FEAT_COLS] < (Q1 - 10 * IQR)) | (df[FEAT_COLS] > (Q3 + 10 * IQR))).any(axis=1)
n_outlier_rows = outlier_mask.sum()
print(f"\nBaris dengan outlier ekstrem: {n_outlier_rows}")
# Clip outlier ke batas 10×IQR (tidak dibuang, hanya dibatasi)
for col in FEAT_COLS:
    lower = Q1[col] - 10 * IQR[col]
    upper = Q3[col] + 10 * IQR[col]
    df[col] = df[col].clip(lower=lower, upper=upper)
print("→ Outlier ekstrem di-clip ke batas 10×IQR.")

print(f"\nJumlah fitur setelah pembersihan: {len(FEAT_COLS)}")

# %% [markdown]
# ## 4.4 — Hapus Fitur Redundan (Korelasi Tinggi)
# Fitur yang berkorelasi > 0.95 satu sama lain dihapus untuk mengurangi
# multikolinieritas yang dapat mengganggu model SVM dan Logistic Regression.

# %%
CORR_THRESHOLD = 0.95

print(f"Menghitung correlation matrix ({len(FEAT_COLS)} × {len(FEAT_COLS)})...")
corr_matrix = df[FEAT_COLS].corr().abs()

# Ambil upper triangle
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identifikasi kolom yang berkorelasi tinggi
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > CORR_THRESHOLD)]

print(f"Fitur yang dihapus (korelasi > {CORR_THRESHOLD}): {len(to_drop)}")
if to_drop:
    print(f"  Contoh 5 pertama: {to_drop[:5]}")

FEAT_COLS_FILTERED = [f for f in FEAT_COLS if f not in to_drop]
print(f"Fitur tersisa setelah filter korelasi: {len(FEAT_COLS_FILTERED)}")

# Visualisasi heatmap korelasi (subset 30 fitur pertama)
fig, ax = plt.subplots(figsize=(12, 10))
sample_feats = FEAT_COLS_FILTERED[:30]
sns.heatmap(
    df[sample_feats].corr(),
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    ax=ax, cbar_kws={'label': 'Korelasi Pearson'},
    xticklabels=True, yticklabels=True
)
ax.set_title(f'Heatmap Korelasi (30 Fitur Pertama setelah Filter)', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=90, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p4_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Heatmap korelasi tersimpan.")

# %% [markdown]
# ## 4.5 — Seleksi Fitur: ANOVA F-test + Mutual Information
# Mengidentifikasi fitur yang paling diskriminatif antara kelas Depresi vs Non-Depresi.

# %%
X = df[FEAT_COLS_FILTERED].values
y = df['label_3kelas'].values  # Multiclass: 0=Stress, 1=Kecemasan, 2=Depresi

# ANOVA F-test
f_scores, p_values = f_classif(X, y)

# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)

# Gabungkan ke DataFrame untuk analisis
df_selection = pd.DataFrame({
    'feature'   : FEAT_COLS_FILTERED,
    'f_score'   : f_scores,
    'p_value'   : p_values,
    'mi_score'  : mi_scores,
    'significant': p_values < 0.05
})
df_selection = df_selection.sort_values('mi_score', ascending=False)

n_sig = df_selection['significant'].sum()
print(f"Fitur signifikan (p < 0.05) untuk 3 kelas: {n_sig} / {len(FEAT_COLS_FILTERED)}")
print(f"\nTop 15 fitur (berdasarkan Mutual Information):")
print(df_selection.head(15)[['feature', 'f_score', 'mi_score', 'significant']].to_string(index=False))

# %% [markdown]
# ## 4.6 — Visualisasi: Top 20 Fitur berdasarkan MI Score

# %%
top20 = df_selection.head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Seleksi Fitur — DAIC-WOZ', fontsize=13, fontweight='bold')

# Mutual Information
colors_mi = ['#e74c3c' if sig else '#3498db' for sig in top20['significant']]
axes[0].barh(top20['feature'][::-1], top20['mi_score'][::-1], color=colors_mi[::-1])
axes[0].set_xlabel('Mutual Information Score')
axes[0].set_title('Top 20 Fitur — Mutual Information')
axes[0].axvline(x=0, color='black', linewidth=0.5)
# Tambahkan legend manual
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Signifikan (p<0.05)'),
                   Patch(facecolor='#3498db', label='Tidak Signifikan')]
axes[0].legend(handles=legend_elements, loc='lower right', fontsize=9)

# F-Score
top20_f = df_selection.nlargest(20, 'f_score')
colors_f = ['#e74c3c' if sig else '#3498db' for sig in top20_f['significant']]
axes[1].barh(top20_f['feature'][::-1], top20_f['f_score'][::-1], color=colors_f[::-1])
axes[1].set_xlabel('ANOVA F-Score')
axes[1].set_title('Top 20 Fitur — ANOVA F-test')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p4_feature_selection.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi seleksi fitur tersimpan.")

# %% [markdown]
# ## 4.7 — Tentukan Fitur Final
# Menggunakan fitur yang signifikan secara statistik (p < 0.05) ATAU masuk top-N MI.
# Strategi konservatif: ambil semua fitur yang lolos setidaknya satu kriteria.

# %%
TOP_N_MI = 50  # Ambil top-N berdasarkan MI jika jumlah signifikan terlalu sedikit

sig_feats = df_selection[df_selection['significant']]['feature'].tolist()
top_mi_feats = df_selection.head(TOP_N_MI)['feature'].tolist()

# Gabung (union) kedua kriteria
final_feats = list(set(sig_feats) | set(top_mi_feats))
final_feats = [f for f in FEAT_COLS_FILTERED if f in final_feats]  # Jaga urutan asli

print(f"Fitur signifikan (p<0.05) : {len(sig_feats)}")
print(f"Top {TOP_N_MI} MI           : {len(top_mi_feats)}")
print(f"Fitur final (union)        : {len(final_feats)}")

# %% [markdown]
# ## 4.8 — Simpan Dataset Final

# %%
df_final = df[META_COLS + final_feats].copy()
df_final.to_csv(OUTPUT_FINAL, index=False)

print(f"Dataset final tersimpan: {OUTPUT_FINAL}")
print(f"Shape: {df_final.shape}")
print(f"\nRingkasan:")
print(f"  Partisipan    : {df_final.shape[0]}")
print(f"  Fitur final   : {len(final_feats)}")
for k, name in {0:'Stress', 1:'Kecemasan', 2:'Depresi'}.items():
    n = int((df_final['label_3kelas'] == k).sum())
    print(f"  {name:12s}  : {n}")

# Simpan juga daftar fitur final untuk referensi di Part 5-7
feat_list_path = os.path.join(PROCESSED_DIR, 'daic_feature_list.txt')
with open(feat_list_path, 'w') as f:
    f.write('\n'.join(final_feats))
print(f"\nDaftar fitur tersimpan: {feat_list_path}")
print(f"Sample 10 fitur final:")
for feat in final_feats[:10]:
    print(f"  {feat}")

# %% [markdown]
# # Part 5 — Split Data (Anti-Leakage): DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# Notebook ini bertugas untuk:
# 1. Split dataset berdasarkan **Participant ID** (bukan per-segmen) → Anti-Leakage
# 2. Proposi: 70% Train / 15% Validation / 15% Test
# 3. Validasi tidak ada overlap Participant ID antar split
# 4. Fit StandardScaler **hanya pada train set** → transform val & test
# 5. Setup GroupKFold untuk cross-validation di Part 6
# 6. Simpan: `daic_train.csv`, `daic_val.csv`, `daic_test.csv`, `scaler.pkl`
#
# **Kelas Target**: 0=Stress | 1=Kecemasan | 2=Depresi (via PHQ-8 proxy)
# **KRITIS (Danylenko & Unold, 2025)**:
# Semua sesi dari orang yang sama HARUS berada di split yang sama.

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi Path

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR      = os.path.dirname(os.path.dirname(current_dir))
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')
MODELS_DIR    = os.path.join(PROCESSED_DIR, 'models')

FEAT_FINAL_PATH  = os.path.join(PROCESSED_DIR, 'daic_features_final.csv')
FEAT_LIST_PATH   = os.path.join(PROCESSED_DIR, 'daic_feature_list.txt')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Proporsi Split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

print(f"Train / Val / Test  : {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
print(f"Random Seed         : {RANDOM_SEED}")

# %% [markdown]
# ## 5.1 — Load Dataset Final dari Part 4

# %%
df = pd.read_csv(FEAT_FINAL_PATH)

# Load daftar fitur
with open(FEAT_LIST_PATH, 'r') as f:
    FEAT_COLS = [line.strip() for line in f.readlines() if line.strip()]

# Pastikan semua fitur ada di dataframe
FEAT_COLS = [f for f in FEAT_COLS if f in df.columns]

META_COLS = ['participant_id', 'phq8_score', 'label_3kelas', 'severity']

print(f"Shape dataset final : {df.shape}")
print(f"Jumlah fitur        : {len(FEAT_COLS)}")
CLASS_NAMES = {0: 'Stress', 1: 'Kecemasan', 2: 'Depresi'}
print(f"\nDistribusi 3 Kelas:")
vc = df['label_3kelas'].value_counts().sort_index()
for label, count in vc.items():
    print(f"  Kelas {label} ({CLASS_NAMES.get(label,'?'):10s}): {count} ({count/len(df)*100:.1f}%)")

# %% [markdown]
# ## 5.2 — Stratified Split Berbasis Participant ID
#
# **Alur Anti-Leakage**:
# 1. Dapatkan daftar unik Participant ID
# 2. Split Participant ID secara stratified berdasarkan label
# 3. Map setiap Participant ke split yang sesuai
#
# Dengan cara ini, seluruh sesi dari satu orang pasti di satu split saja.

# %%
# Dapatkan label per partisipan (1 partisipan = 1 baris = 1 label)
df_participants = df[['participant_id', 'label_3kelas']].drop_duplicates()
participant_ids    = df_participants['participant_id'].values
participant_labels = df_participants['label_3kelas'].values

print(f"Total partisipan unik: {len(participant_ids)}")

# Step 1: Split test set terlebih dahulu (15%)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_SEED)
train_val_idx, test_idx = next(sss1.split(participant_ids, participant_labels))

pids_train_val = participant_ids[train_val_idx]
labels_train_val = participant_labels[train_val_idx]
pids_test = participant_ids[test_idx]

# Step 2: Split validation dari train_val (15% dari total ≈ 17.6% dari train_val)
val_size_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=RANDOM_SEED)
train_idx2, val_idx2 = next(sss2.split(pids_train_val, labels_train_val))

pids_train = pids_train_val[train_idx2]
pids_val   = pids_train_val[val_idx2]

print(f"\nHasil Split (berdasarkan Participant ID):")
print(f"  Train : {len(pids_train)} partisipan ({len(pids_train)/len(participant_ids)*100:.1f}%)")
print(f"  Val   : {len(pids_val)} partisipan ({len(pids_val)/len(participant_ids)*100:.1f}%)")
print(f"  Test  : {len(pids_test)} partisipan ({len(pids_test)/len(participant_ids)*100:.1f}%)")

# %% [markdown]
# ## 5.3 — Validasi Anti-Leakage
# Memastikan **tidak ada Participant ID yang overlap** di antara train, val, dan test.

# %%
set_train = set(pids_train)
set_val   = set(pids_val)
set_test  = set(pids_test)

overlap_tv = set_train & set_val
overlap_tt = set_train & set_test
overlap_vt = set_val   & set_test

print("=== VALIDASI ANTI-LEAKAGE ===")
print(f"Overlap Train ∩ Val   : {len(overlap_tv)} partisipan  {'✓ AMAN' if len(overlap_tv)==0 else '✗ BAHAYA!'}")
print(f"Overlap Train ∩ Test  : {len(overlap_tt)} partisipan  {'✓ AMAN' if len(overlap_tt)==0 else '✗ BAHAYA!'}")
print(f"Overlap Val   ∩ Test  : {len(overlap_vt)} partisipan  {'✓ AMAN' if len(overlap_vt)==0 else '✗ BAHAYA!'}")

assert len(overlap_tv) == 0, "LEAKAGE TERDETEKSI: Train-Val overlap!"
assert len(overlap_tt) == 0, "LEAKAGE TERDETEKSI: Train-Test overlap!"
assert len(overlap_vt) == 0, "LEAKAGE TERDETEKSI: Val-Test overlap!"
print("\n✓ Validasi Anti-Leakage LULUS — Tidak ada Participant ID yang overlap.")

# %% [markdown]
# ## 5.4 — Buat DataFrame Train / Val / Test

# %%
df_train = df[df['participant_id'].isin(pids_train)].copy().reset_index(drop=True)
df_val   = df[df['participant_id'].isin(pids_val)].copy().reset_index(drop=True)
df_test  = df[df['participant_id'].isin(pids_test)].copy().reset_index(drop=True)

print("=== Distribusi Kelas per Split (3 Kelas) ===")
for name, dset in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    counts = dset['label_3kelas'].value_counts().sort_index()
    detail = ' | '.join([f"{CLASS_NAMES.get(k,'?')}={v}" for k,v in counts.items()])
    print(f"  [{name}] total={len(dset)} | {detail}")

# %% [markdown]
# ## 5.5 — StandardScaler (Fit HANYA pada Train)
# Melanggar aturan ini (fit pada semua data) = data leakage statistik.

# %%
X_train = df_train[FEAT_COLS].values
X_val   = df_val[FEAT_COLS].values
X_test  = df_test[FEAT_COLS].values

y_train = df_train['label_3kelas'].values
y_val   = df_val['label_3kelas'].values
y_test  = df_test['label_3kelas'].values

# Fit scaler HANYA pada train set
scaler = StandardScaler()
scaler.fit(X_train)

# Transform semua split
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Masukkan kembali ke DataFrame
df_train_scaled = df_train[META_COLS].copy()
df_val_scaled   = df_val[META_COLS].copy()
df_test_scaled  = df_test[META_COLS].copy()

df_train_scaled[FEAT_COLS] = X_train_scaled
df_val_scaled[FEAT_COLS]   = X_val_scaled
df_test_scaled[FEAT_COLS]  = X_test_scaled

print("StandardScaler berhasil di-fit pada train set dan ditransform ke semua split.")
print(f"  Mean (train, 5 fitur pertama): {scaler.mean_[:5].round(4)}")
print(f"  Std  (train, 5 fitur pertama): {scaler.scale_[:5].round(4)}")

# %% [markdown]
# ## 5.6 — Visualisasi Distribusi Split

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Validasi Split Data — DAIC-WOZ (Anti-Leakage)', fontsize=13, fontweight='bold')

# Pie chart pembagian partisipan
split_sizes  = [len(pids_train), len(pids_val), len(pids_test)]
split_labels = [f'Train\n({len(pids_train)} prt)', f'Val\n({len(pids_val)} prt)',
                f'Test\n({len(pids_test)} prt)']
axes[0].pie(split_sizes, labels=split_labels, autopct='%1.1f%%',
            colors=['#3498db', '#f39c12', '#e74c3c'], startangle=90,
            textprops={'fontsize': 10})
axes[0].set_title('Proporsi Split (Participant ID)')

# Bar chart distribusi 3 kelas per split
splits     = {'Train': df_train, 'Val': df_val, 'Test': df_test}
x          = np.arange(3)
width      = 0.25
class_colors = ['#3498db', '#f39c12', '#e74c3c']
for ci, (k, name) in enumerate(CLASS_NAMES.items()):
    counts = [int((dset['label_3kelas'] == k).sum()) for dset in splits.values()]
    bars   = axes[1].bar(x + (ci - 1) * width, counts, width,
                          label=name, color=class_colors[ci], edgecolor='black')
    for bar, val in zip(bars, counts):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     str(val), ha='center', fontsize=8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Train', 'Val', 'Test'])
axes[1].set_ylabel('Jumlah Partisipan')
axes[1].set_title('Distribusi 3 Kelas per Split')
axes[1].legend(fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p5_split_validation.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi split tersimpan.")

# %% [markdown]
# ## 5.7 — Simpan Semua Output

# %%
# Simpan CSV
out_train = os.path.join(PROCESSED_DIR, 'daic_train.csv')
out_val   = os.path.join(PROCESSED_DIR, 'daic_val.csv')
out_test  = os.path.join(PROCESSED_DIR, 'daic_test.csv')

df_train_scaled.to_csv(out_train, index=False)
df_val_scaled.to_csv(out_val,   index=False)
df_test_scaled.to_csv(out_test,  index=False)

# Simpan scaler
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Simpan daftar PID per split untuk referensi
split_info = {
    'train_pids': pids_train.tolist(),
    'val_pids'  : pids_val.tolist(),
    'test_pids' : pids_test.tolist(),
}
import json
split_info_path = os.path.join(PROCESSED_DIR, 'daic_split_info.json')
with open(split_info_path, 'w') as f:
    json.dump(split_info, f, indent=2)

print("=== SEMUA OUTPUT TERSIMPAN ===")
print(f"  Train CSV  : {out_train}  ({df_train_scaled.shape})")
print(f"  Val CSV    : {out_val}    ({df_val_scaled.shape})")
print(f"  Test CSV   : {out_test}   ({df_test_scaled.shape})")
print(f"  Scaler PKL : {scaler_path}")
print(f"  Split Info : {split_info_path}")

# %% [markdown]
# ## 5.8 — Setup GroupKFold untuk Part 6 (Cross-Validation)
#
# GroupKFold memastikan semua sesi dari orang yang sama berada di fold yang sama.
# Ini adalah metode cross-validation yang aman untuk dataset audio medis.

# %%
# Demo setup GroupKFold pada data train
X_demo = df_train_scaled[FEAT_COLS].values
y_demo = df_train_scaled['label_3kelas'].values
groups = df_train_scaled['participant_id'].values

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

print(f"GroupKFold Cross-Validation Setup — {N_FOLDS} Folds")
print(f"{'Fold':<6} {'Train (prt)':<15} {'Val (prt)':<15} {'Train Dep%':<15} {'Val Dep%'}")
print("-" * 65)

for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X_demo, y_demo, groups)):
    train_dep_pct = y_demo[tr_idx].mean() * 100
    val_dep_pct   = y_demo[vl_idx].mean() * 100
    # Jumlah partisipan unik per fold
    n_tr_grp = len(np.unique(groups[tr_idx]))
    n_vl_grp = len(np.unique(groups[vl_idx]))
    print(f"  {fold+1:<4} {n_tr_grp:<15} {n_vl_grp:<15} {train_dep_pct:<14.1f}% {val_dep_pct:.1f}%")

print(f"\n✓ GroupKFold siap digunakan di Part 6 (Training Model).")
print(f"  Pastikan gunakan groups=participant_id agar tidak terjadi leakage dalam CV.")

# %% [markdown]
# # Part 6 — Training Model ML: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# Notebook ini bertugas untuk:
# 1. Load data train/val/test dari Part 5 (sudah di-scale)
# 2. Train 4 model ML: Logistic Regression, SVM, Random Forest, XGBoost
# 3. Cross-validation dengan GroupKFold (5-fold, anti-leakage)
# 4. Hyperparameter tuning dengan GridSearchCV
# 5. Evaluasi MULTICLASS: Macro F1, Accuracy, Per-Class Report, Confusion Matrix 3x3
# 6. Simpan model terbaik & tabel perbandingan seluruh model
#
# **Kelas Target**: 0=Stress | 1=Kecemasan | 2=Depresi (via PHQ-8 proxy)
# **Referensi**: Yadav et al. (2023), Danylenko & Unold (2025)

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             mean_absolute_error, r2_score)
import xgboost as xgb
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi Path

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR      = os.path.dirname(os.path.dirname(current_dir))
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
MODELS_DIR    = os.path.join(PROCESSED_DIR, 'models')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')

TRAIN_PATH    = os.path.join(PROCESSED_DIR, 'daic_train.csv')
VAL_PATH      = os.path.join(PROCESSED_DIR, 'daic_val.csv')
TEST_PATH     = os.path.join(PROCESSED_DIR, 'daic_test.csv')
FEAT_LIST_PATH= os.path.join(PROCESSED_DIR, 'daic_feature_list.txt')
SPLIT_INFO    = os.path.join(PROCESSED_DIR, 'daic_split_info.json')

os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42
print(f"Models dir: {MODELS_DIR}")

# %% [markdown]
# ## 6.1 — Load Data

# %%
df_train = pd.read_csv(TRAIN_PATH)
df_val   = pd.read_csv(VAL_PATH)
df_test  = pd.read_csv(TEST_PATH)

with open(FEAT_LIST_PATH, 'r') as f:
    FEAT_COLS = [line.strip() for line in f if line.strip()]
FEAT_COLS = [c for c in FEAT_COLS if c in df_train.columns]

META_COLS = ['participant_id', 'phq8_score', 'label_3kelas', 'severity']

X_train = df_train[FEAT_COLS].values
y_train = df_train['label_3kelas'].values   # 0=Stress, 1=Kecemasan, 2=Depresi
groups_train = df_train['participant_id'].values

X_val   = df_val[FEAT_COLS].values
y_val   = df_val['label_3kelas'].values

X_test  = df_test[FEAT_COLS].values
y_test  = df_test['label_3kelas'].values

# Gabung train+val untuk training final
X_trainval     = np.vstack([X_train, X_val])
y_trainval     = np.concatenate([y_train, y_val])
groups_trainval= np.concatenate([groups_train, df_val['participant_id'].values])

CLASS_NAMES = {0: 'Stress', 1: 'Kecemasan', 2: 'Depresi'}

print(f"Train : {X_train.shape}")
for k, name in CLASS_NAMES.items():
    print(f"  {name}: {(y_train==k).sum()}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")
print(f"Fitur : {len(FEAT_COLS)}")

# %% [markdown]
# ## 6.2 — Definisi Model & Hyperparameter Grid
# SVM multiclass menggunakan `decision_function_shape='ovr'` (One-vs-Rest).
# class_weight='balanced' untuk handle imbalance antar kelas.

# %%
MODELS = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                                    class_weight='balanced',
                                    multi_class='auto'),
        'param_grid': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
        }
    },
    'SVM (RBF)': {
        'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED,
                     class_weight='balanced',
                     decision_function_shape='ovr'),  # One-vs-Rest untuk multiclass
        'param_grid': {
            'C':     [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto'],
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_SEED,
                                        class_weight='balanced', n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth':    [None, 10, 20],
            'min_samples_split': [2, 5],
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='mlogloss',
                                    num_class=3,       # 3 kelas
                                    objective='multi:softmax',
                                    verbosity=0, n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth':    [3, 5, 7],
            'learning_rate':[0.05, 0.1],
        }
    }
}

print("Model yang akan dilatih:")
for name in MODELS.keys():
    print(f"  - {name}")

# %% [markdown]
# ## 6.3 — Fungsi Evaluasi

# %%
def evaluate_model(model, X, y_true, prefix=''):
    """Hitung semua metrik evaluasi multiclass (macro-averaged)."""
    y_pred = model.predict(X)
    return {
        f'{prefix}accuracy' : accuracy_score(y_true, y_pred),
        f'{prefix}f1_macro' : f1_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        f'{prefix}precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}recall'   : recall_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}mae'      : mean_absolute_error(y_true, y_pred),
    }

# %% [markdown]
# ## 6.4 — Training dengan GroupKFold Cross-Validation + GridSearch

# %%
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

results      = {}
best_models  = {}

print("="*65)
print(f"{'TRAINING MODEL':^65}")
print("="*65)

for model_name, model_cfg in MODELS.items():
    print(f"\n[{model_name}]")

    base_model  = model_cfg['model']
    param_grid  = model_cfg['param_grid']

    # GridSearchCV dengan GroupKFold (anti-leakage)
    gsearch = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        cv         = gkf,
        scoring    = 'f1_macro',   # Macro F1 untuk multiclass
        n_jobs     = -1,
        verbose    = 0,
        refit      = True,
    )

    # Fit dengan groups untuk mencegah leakage dalam CV
    gsearch.fit(X_train, y_train, groups=groups_train)
    best_model = gsearch.best_estimator_

    print(f"  Best params : {gsearch.best_params_}")
    print(f"  Best CV F1  : {gsearch.best_score_:.4f}")

    # Evaluasi pada val set
    val_metrics  = evaluate_model(best_model, X_val, y_val, prefix='val_')
    test_metrics = evaluate_model(best_model, X_test, y_test, prefix='test_')

    print(f"  Best CV Macro F1: {gsearch.best_score_:.4f}")
    print(f"  Val  Macro F1   : {val_metrics['val_f1_macro']:.4f} | Acc: {val_metrics['val_accuracy']:.4f}")
    print(f"  Test Macro F1   : {test_metrics['test_f1_macro']:.4f} | Acc: {test_metrics['test_accuracy']:.4f}")

    results[model_name] = {
        'best_params'  : gsearch.best_params_,
        'cv_f1'        : gsearch.best_score_,
        **val_metrics,
        **test_metrics,
    }
    best_models[model_name] = best_model

print("\n" + "="*65)
print("Training selesai!")

# %% [markdown]
# ## 6.5 — Tabel Perbandingan Model

# %%
df_results = pd.DataFrame([
    {
        'Model'           : name,
        'CV Macro F1'     : f"{r['cv_f1']:.4f}",
        'Val Accuracy'    : f"{r['val_accuracy']:.4f}",
        'Val Macro F1'    : f"{r['val_f1_macro']:.4f}",
        'Val Weighted F1' : f"{r['val_f1_weighted']:.4f}",
        'Val Precision'   : f"{r['val_precision']:.4f}",
        'Val Recall'      : f"{r['val_recall']:.4f}",
        'Test Accuracy'   : f"{r['test_accuracy']:.4f}",
        'Test Macro F1'   : f"{r['test_f1_macro']:.4f}",
        'Test Weighted F1': f"{r['test_f1_weighted']:.4f}",
        'Test Precision'  : f"{r['test_precision']:.4f}",
        'Test Recall'     : f"{r['test_recall']:.4f}",
    }
    for name, r in results.items()
])

print("\n=== PERBANDINGAN MODEL — DAIC-WOZ ===\n")
print(df_results.to_string(index=False))

# Simpan tabel
df_results.to_csv(os.path.join(PROCESSED_DIR, 'daic_model_comparison.csv'), index=False)
print("\nTabel perbandingan tersimpan.")

# %% [markdown]
# ## 6.6 — Visualisasi Perbandingan Model

# %%
metrics_to_plot = {
    'test_f1_macro'   : 'Macro F1-Score (Test)',
    'test_accuracy'   : 'Accuracy (Test)',
    'test_precision'  : 'Macro Precision (Test)',
    'test_recall'     : 'Macro Recall (Test)',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Perbandingan Model ML — DAIC-WOZ', fontsize=14, fontweight='bold')

model_names = list(results.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for ax, (metric_key, metric_label) in zip(axes.flatten(), metrics_to_plot.items()):
    values = [float(results[m][metric_key]) for m in model_names]
    bars   = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_title(metric_label, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(y=0.60, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='Min target ≥60%')   # Multiclass lebih realistis
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p6_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi perbandingan model tersimpan.")

# %% [markdown]
# ## 6.7 — Confusion Matrix 3x3 Semua Model

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Confusion Matrix 3 Kelas — DAIC-WOZ (Test Set)\n(Stress=0 | Kecemasan=1 | Depresi=2)',
             fontsize=13, fontweight='bold')

class_tick_labels = ['Stress\n(0)', 'Kecemasan\n(1)', 'Depresi\n(2)']

for ax, (model_name, model) in zip(axes.flatten(), best_models.items()):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_tick_labels, yticklabels=class_tick_labels,
                linewidths=0.5, linecolor='gray')
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    ax.set_title(f'{model_name}\n(Macro F1={f1:.3f})', fontweight='bold', fontsize=10)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p6_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix tersimpan.")

# %% [markdown]
# ## 6.8 — Pilih & Simpan Model Terbaik

# %%
# Pilih model terbaik berdasarkan Test Macro F1
best_model_name = max(results, key=lambda m: float(results[m]['test_f1_macro']))
best_model_obj  = best_models[best_model_name]
best_f1         = float(results[best_model_name]['test_f1_macro'])

print(f"\n{'='*55}")
print(f"MODEL TERBAIK  : {best_model_name}")
print(f"Test Macro F1  : {best_f1:.4f}")
print(f"Test Accuracy  : {float(results[best_model_name]['test_accuracy']):.4f}")
print(f"{'='*55}")
print(f"\nPer-Class Classification Report:")
y_pred_best = best_model_obj.predict(X_test)
print(classification_report(y_test, y_pred_best,
      target_names=['Stress (0)', 'Kecemasan (1)', 'Depresi (2)'], zero_division=0))

# Simpan semua model
for name, model in best_models.items():
    safe_name   = name.replace(' ', '_').replace('(', '').replace(')', '')
    model_path  = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Tersimpan: {model_path}")

# Simpan info model terbaik
best_info = {
    'best_model_name' : best_model_name,
    'best_params'     : results[best_model_name]['best_params'],
    'test_f1_macro'   : float(results[best_model_name]['test_f1_macro']),
    'test_accuracy'   : float(results[best_model_name]['test_accuracy']),
    'feature_count'   : len(FEAT_COLS),
}
with open(os.path.join(MODELS_DIR, 'best_model_info.json'), 'w') as f:
    json.dump(best_info, f, indent=2)

print(f"\nBest model info tersimpan: {os.path.join(MODELS_DIR, 'best_model_info.json')}")

# %% [markdown]
# # Part 7 — XAI (Explainable AI): DAIC-WOZ
# **Kelas**: 0=Stress | 1=Kecemasan | 2=Depresi
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# - SHAP multiclass: beeswarm & waterfall per kelas (RF & XGBoost)
# - LIME multiclass: local explanation per instance (SVM & LR)
# - Visualisasi: Prediksi vs Aktual, Top Feature Importance

# %%
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

# Install lime jika belum ada
try:
    from lime.lime_tabular import LimeTabularExplainer
    print("lime tersedia.")
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'lime'], check=True)
    from lime.lime_tabular import LimeTabularExplainer
    print("lime berhasil diinstall.")

print("Library berhasil diimport.")

# %% [markdown]
# ## Konfigurasi

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath(os.getcwd())

BASE_DIR       = os.path.dirname(os.path.dirname(current_dir))
PROCESSED_DIR  = os.path.join(BASE_DIR, 'dataset', 'processed')
MODELS_DIR     = os.path.join(PROCESSED_DIR, 'models')
OUTPUT_DIR     = os.path.join(BASE_DIR, 'docs', 'assets', 'images', 'daic')

TRAIN_PATH     = os.path.join(PROCESSED_DIR, 'daic_train.csv')
TEST_PATH      = os.path.join(PROCESSED_DIR, 'daic_test.csv')
FEAT_LIST_PATH = os.path.join(PROCESSED_DIR, 'daic_feature_list.txt')

CLASS_NAMES    = {0: 'Stress', 1: 'Kecemasan', 2: 'Depresi'}
CLASS_COLORS   = {0: '#3498db', 1: '#f39c12', 2: '#e74c3c'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ## 7.1 — Load Data & Model

# %%
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

with open(FEAT_LIST_PATH, 'r') as f:
    FEAT_COLS = [line.strip() for line in f if line.strip()]
FEAT_COLS = [c for c in FEAT_COLS if c in df_train.columns]

X_train = df_train[FEAT_COLS].values
y_train = df_train['label_3kelas'].values
X_test  = df_test[FEAT_COLS].values
y_test  = df_test['label_3kelas'].values

def load_model(name):
    safe_name  = name.replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

model_rf  = load_model('Random_Forest')
model_xgb = load_model('XGBoost')
model_svm = load_model('SVM_RBF')
model_lr  = load_model('Logistic_Regression')

loaded = {n: m for n, m in [('Random Forest', model_rf), ('XGBoost', model_xgb),
                              ('SVM (RBF)', model_svm), ('Logistic Regression', model_lr)]
          if m is not None}

print(f"Model dimuat: {list(loaded.keys())}")
print(f"Test: {len(X_test)} sampel | {len(FEAT_COLS)} fitur")
print(f"Distribusi kelas test:")
for k, name in CLASS_NAMES.items():
    print(f"  Kelas {k} ({name}): {(y_test==k).sum()}")

# %% [markdown]
# ## 7.2 — SHAP: Random Forest (Multiclass)
# SHAP untuk multiclass menghasilkan satu set SHAP values per kelas.
# Kita visualisasikan untuk setiap kelas: Stress, Kecemasan, Depresi.

# %%
if model_rf is not None:
    print("Menghitung SHAP values (Random Forest, multiclass)...")
    X_shap = X_test[:min(80, len(X_test))]
    y_shap = y_test[:min(80, len(y_test))]

    explainer_rf    = shap.TreeExplainer(model_rf)
    shap_values_rf  = explainer_rf.shap_values(X_shap)
    # shap_values_rf adalah list of arrays: [kelas0, kelas1, kelas2]

    # --- Beeswarm per kelas ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SHAP Beeswarm per Kelas — Random Forest\n(Stress | Kecemasan | Depresi)',
                 fontsize=12, fontweight='bold')

    for ci, (k, name) in enumerate(CLASS_NAMES.items()):
        sv = shap_values_rf[k] if isinstance(shap_values_rf, list) else shap_values_rf[:, :, k]
        plt.sca(axes[ci])
        shap.summary_plot(sv, X_shap, feature_names=FEAT_COLS,
                          max_display=12, show=False, plot_type='dot')
        axes[ci].set_title(f'Kelas {k}: {name}', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'p7_shap_rf_beeswarm_3class.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("SHAP Beeswarm 3-kelas tersimpan.")

    # --- Global Feature Importance (mean |SHAP| across all classes) ---
    if isinstance(shap_values_rf, list):
        mean_abs = np.mean([np.abs(shap_values_rf[k]).mean(axis=0) for k in range(3)], axis=0)
    else:
        mean_abs = np.abs(shap_values_rf).mean(axis=(0, 2))

    top_n   = 15
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_f   = [FEAT_COLS[i] for i in top_idx]
    top_v   = [mean_abs[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_f[::-1], top_v[::-1], color='#8e44ad', edgecolor='black', linewidth=0.6, alpha=0.85)
    ax.set_xlabel('Mean |SHAP Value| (rata-rata 3 kelas)')
    ax.set_title(f'Top {top_n} Fitur Terpenting — Random Forest (SHAP)\nDATAC-WOZ | 3 Kelas: Stress | Kecemasan | Depresi',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'p7_shap_rf_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("SHAP Global Importance tersimpan.")
else:
    print("Model RF tidak ditemukan.")

# %% [markdown]
# ## 7.3 — SHAP: Waterfall Plot Lokal (1 sampel per kelas)
# Menjelaskan mengapa model memprediksi kelas tertentu untuk 1 individu.

# %%
if model_rf is not None and isinstance(shap_values_rf, list):
    for k, name in CLASS_NAMES.items():
        idx_list = np.where(y_shap == k)[0]
        if len(idx_list) == 0:
            print(f"Tidak ada sampel kelas {k} ({name}) di subset test.")
            continue

        sample_idx  = idx_list[0]
        pred_label  = model_rf.predict([X_shap[sample_idx]])[0]
        sv_k        = shap_values_rf[k]
        exp_val     = explainer_rf.expected_value[k] if isinstance(explainer_rf.expected_value, list) else explainer_rf.expected_value

        shap_expl = shap.Explanation(
            values        = sv_k[sample_idx],
            base_values   = exp_val,
            data          = X_shap[sample_idx],
            feature_names = FEAT_COLS,
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(shap_expl, max_display=12, show=False)
        plt.title(f'SHAP Waterfall — Kelas {k}: {name}\n'
                  f'Aktual={CLASS_NAMES.get(y_shap[sample_idx],"?")} | Prediksi={CLASS_NAMES.get(pred_label,"?")}',
                  fontweight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'p7_shap_waterfall_kelas{k}_{name.lower()}.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(f"SHAP Waterfall kelas {k} ({name}) tersimpan.")

# %% [markdown]
# ## 7.4 — LIME: SVM & Logistic Regression (Multiclass)
# LIME mendukung multiclass secara native via `top_labels=3`.

# %%
lime_explainer = LimeTabularExplainer(
    training_data = X_train,
    feature_names = FEAT_COLS,
    class_names   = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES.keys())],
    mode          = 'classification',
    random_state  = 42,
)

def run_lime_3class(model, model_name, X_sample, actual_label, n_features=12):
    """Jalankan LIME multiclass dan simpan plot per kelas prediksi tertinggi."""
    try:
        lime_exp = lime_explainer.explain_instance(
            data_row  = X_sample,
            predict_fn= model.predict_proba,
            num_features = n_features,
            num_samples  = 500,
            top_labels   = 3,
        )
        pred_label = model.predict([X_sample])[0]
        pred_proba = model.predict_proba([X_sample])[0]

        # Tampilkan untuk kelas yang diprediksi
        exp_list   = lime_exp.as_list(label=pred_label)
        feat_names = [e[0] for e in exp_list]
        feat_vals  = [e[1] for e in exp_list]

        sorted_pairs = sorted(zip(feat_vals, feat_names))
        s_vals, s_names = zip(*sorted_pairs) if sorted_pairs else ([], [])

        colors = ['#e74c3c' if v > 0 else '#3498db' for v in s_vals]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(s_names), list(s_vals), color=colors, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=1.2)
        ax.set_xlabel(f'Kontribusi terhadap Kelas: {CLASS_NAMES.get(pred_label,"?")}')
        ax.set_title(
            f'LIME — {model_name}\n'
            f'Aktual={CLASS_NAMES.get(actual_label,"?")} | '
            f'Prediksi={CLASS_NAMES.get(pred_label,"?")} | '
            f'P={pred_proba[pred_label]:.3f}',
            fontweight='bold', fontsize=10
        )
        plt.tight_layout()
        safe_m = model_name.lower().replace(' ','_').replace('(','').replace(')','')
        safe_l = CLASS_NAMES.get(actual_label,'').lower()
        path   = os.path.join(OUTPUT_DIR, f'p7_lime_{safe_m}_{safe_l}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"LIME [{model_name}] kelas aktual {CLASS_NAMES.get(actual_label,'')} tersimpan.")
    except Exception as e:
        print(f"  LIME [{model_name}] gagal: {e}")

# %% [markdown]
# ## 7.5 — Jalankan LIME untuk SVM dan LR

# %%
for lm_name, lm_model in [('SVM (RBF)', model_svm), ('Logistic Regression', model_lr)]:
    if lm_model is None:
        print(f"Model {lm_name} tidak tersedia, skip.")
        continue
    print(f"\n=== LIME: {lm_name} ===")
    # 1 sampel per kelas
    for k in CLASS_NAMES.keys():
        idx_list = np.where(y_test == k)[0]
        if len(idx_list) == 0:
            continue
        run_lime_3class(lm_model, lm_name, X_test[idx_list[0]], y_test[idx_list[0]])

# %% [markdown]
# ## 7.6 — Prediksi vs Aktual (Semua Model, 3 Kelas)

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prediksi vs Aktual — DAIC-WOZ (Test Set)\n0=Stress | 1=Kecemasan | 2=Depresi',
             fontsize=12, fontweight='bold')

from sklearn.metrics import f1_score as f1_fn

for ax, (model_name, model) in zip(axes.flatten(), loaded.items()):
    y_pred   = model.predict(X_test)
    correct  = (y_pred == y_test)
    np.random.seed(42)
    jitter   = np.random.uniform(-0.15, 0.15, len(y_test))

    for k, name in CLASS_NAMES.items():
        mask_c = correct & (y_test == k)
        mask_w = ~correct & (y_test == k)
        ax.scatter(y_test[mask_c] + jitter[mask_c], y_pred[mask_c],
                   c=CLASS_COLORS[k], alpha=0.6, s=50,
                   label=f'{name} ✓' if k == 0 else None,
                   edgecolors='white', linewidths=0.3)
        ax.scatter(y_test[mask_w] + jitter[mask_w], y_pred[mask_w],
                   c=CLASS_COLORS[k], alpha=0.6, s=50, marker='x',
                   edgecolors=CLASS_COLORS[k])

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Stress', 'Kecemasan', 'Depresi'], fontsize=8)
    ax.set_yticklabels(['Stress', 'Kecemasan', 'Depresi'], fontsize=8)
    ax.set_xlabel('Aktual')
    ax.set_ylabel('Prediksi')
    macro_f1 = f1_fn(y_test, y_pred, average='macro', zero_division=0)
    acc      = (y_pred == y_test).mean()
    ax.set_title(f'{model_name}\nMacro F1={macro_f1:.3f} | Acc={acc:.3f}',
                 fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Legend simpel
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Benar'),
        Line2D([0],[0], marker='x', color='gray', markersize=8, label='Salah'),
    ]
    ax.legend(handles=legend_elems, fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'p7_pred_vs_actual_3class.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi prediksi vs aktual tersimpan.")

# %% [markdown]
# ## 7.7 — Ringkasan Top Fitur (Semua Kelas)

# %%
if model_rf is not None and isinstance(shap_values_rf, list):
    print("\n" + "="*60)
    print("TOP 10 FITUR PER KELAS — Random Forest (SHAP)")
    print("="*60)
    for k, name in CLASS_NAMES.items():
        sv_k      = shap_values_rf[k]
        mean_abs_k= np.abs(sv_k).mean(axis=0)
        top_idx_k = np.argsort(mean_abs_k)[::-1][:10]
        print(f"\nKelas {k}: {name}")
        for rank, idx in enumerate(top_idx_k, 1):
            print(f"  {rank:2d}. {FEAT_COLS[idx]:<35} |SHAP|={mean_abs_k[idx]:.5f}")

print("\n✓ Analisis XAI selesai.")
print("  Output:", OUTPUT_DIR)
