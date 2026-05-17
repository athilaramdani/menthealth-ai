# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
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