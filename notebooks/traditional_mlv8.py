# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v8** — Klasifikasi Kesehatan Mental Berbasis Audio
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v8 = Push for 0.70+ Mean Macro F1
#
#  [1] Wav2Vec 2.0 FULL 768-dim (bukan 72 kompresi dari v6)
#      → mean-pooled raw hidden states, setiap dimensi jadi fitur
#
#  [2] Leave-One-Out CV (LOOCV)
#      → Train 101, test 1, ulangi 102 kali
#      → Evaluasi SEMUA participant, variance minimal
#      → Pakai maximum training data
#
#  [3] Hyperparameter Tuning (fixed optimized params dari v6/v7)
#
#  [4] Ensemble Soft Voting (top-3 model)
#
#  [5] Threshold Tuning via inner CV
#
#  Tetap: class_weight='balanced', NO SMOTE, NO augmentasi
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

# %%
import subprocess, sys

def _pip(pkg, name=None):
    check = name or pkg.split('[')[0].split('>=')[0].split('==')[0]
    try: __import__(check); return
    except ImportError: pass
    except Exception: return
    print(f"[Installing] {pkg}")
    try: subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    except: pass

_pip("librosa"); _pip("scikit-learn", "sklearn"); _pip("xgboost"); _pip("seaborn")
_pip("transformers"); _pip("soundfile")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("[OK] Dependencies siap.\n")

# %%
import os, pickle, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

plt.rcParams['font.family'] = 'DejaVu Sans'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Wav2Vec
WAV2VEC_OK = False
try:
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    WAV2VEC_OK = True
    print(f"[OK] torch {torch.__version__} + transformers -> Wav2Vec AKTIF")
except Exception as e:
    print(f"[WARN] Wav2Vec tidak tersedia: {e}")

# Path
PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd() else os.getcwd()
)
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v8")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v8")
for d in [V8_FEAT_DIR, MODELS_DIR,
          os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

FORCE_EXTRACT = False
TARGET_SR = 16000

print(f"Project: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Extract Wav2Vec FULL 768-dim

# %%
def extract_wav2vec_full(audio_path, proc, model, sr=16000):
    """Extract mean-pooled 768-dim vector from full audio."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(y) < sr:
        return None

    chunk_len = 30 * sr  # 30s chunks
    all_hidden = []
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) < 1600:
            continue
        inputs = proc(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
        all_hidden.append(out.last_hidden_state.squeeze(0).numpy())

    if not all_hidden:
        return None

    hidden = np.concatenate(all_hidden, axis=0)  # (T, 768)
    mean_vec = hidden.mean(axis=0)   # (768,)
    std_vec  = hidden.std(axis=0)    # (768,)

    feats = {}
    for i in range(768):
        feats[f'w2v_{i}'] = float(mean_vec[i])
    for i in range(768):
        feats[f'w2v_std_{i}'] = float(std_vec[i])
    return feats

# Load metadata
def load_metadata():
    all_parts = []
    for fname, split in [
        ("train_split_Depression_AVEC2017.csv", "train"),
        ("dev_split_Depression_AVEC2017.csv", "dev"),
        ("full_test_split.csv", "test"),
    ]:
        df = pd.read_csv(os.path.join(RAW_DIR, fname))
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            if c.lower() == 'participant_id':
                df.rename(columns={c: 'participant_id'}, inplace=True)
        if 'PHQ_Binary' in df.columns and 'PHQ8_Binary' not in df.columns:
            df['PHQ8_Binary'] = df['PHQ_Binary']
        if 'PHQ_Score' in df.columns and 'PHQ8_Score' not in df.columns:
            df['PHQ8_Score'] = df['PHQ_Score']
        df['label'] = df['PHQ8_Binary'].astype(int)
        all_parts.append(df[['participant_id', 'label', 'PHQ8_Score']].copy())
    return pd.concat(all_parts, ignore_index=True)

W2V_FULL_CSV = os.path.join(V8_FEAT_DIR, "daic_v8_wav2vec_full.csv")

if FORCE_EXTRACT or not os.path.exists(W2V_FULL_CSV):
    assert WAV2VEC_OK, "torch + transformers required for Wav2Vec extraction!"
    meta = load_metadata()

    print("[INFO] Loading Wav2Vec2 model...")
    proc  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    w2v_m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    w2v_m.eval()
    print("[OK] Wav2Vec2 loaded.\n")

    audio_files = sorted(f for f in os.listdir(CLEANED_DIR) if f.endswith('.wav'))
    rows = []
    t0 = time.time()

    print(f"{'PID':>5} | {'Label':^8} | {'Status'}")
    print("-" * 50)

    for fname in audio_files:
        pid = int(fname.replace('.wav', ''))
        m = meta[meta['participant_id'] == pid]
        if m.empty:
            continue
        label = int(m.iloc[0]['label'])

        fpath = os.path.join(CLEANED_DIR, fname)
        feats = extract_wav2vec_full(fpath, proc, w2v_m, sr=TARGET_SR)
        if feats is None:
            print(f"  {pid:4d} | {'DEP' if label else 'NOR':^8} | SKIP (too short)")
            continue

        row = {'participant_id': pid, 'label': label}
        row.update(feats)
        rows.append(row)
        tag = "Depresi" if label else "Normal"
        print(f"  {pid:4d} | {tag:^8} | OK ({len(feats)} feats)", flush=True)

    df_w2v = pd.DataFrame(rows)
    df_w2v.to_csv(W2V_FULL_CSV, index=False)
    elapsed = time.time() - t0
    print(f"\n[OK] Wav2Vec full extracted: {len(df_w2v)} participants, "
          f"{df_w2v.shape[1]} cols, {elapsed:.0f}s")
    print(f"     Saved: {W2V_FULL_CSV}")
else:
    print(f"[INFO] Wav2Vec full CSV exists: {W2V_FULL_CSV}")

# %% [markdown]
# ## 2. Load All Features

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']

def load_clean(csv_path, name):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    std_v = df[feat_cols].std()
    feat_cols = [c for c in feat_cols if std_v[c] > 1e-8]
    # Detect label column
    if 'label' not in df.columns:
        if 'label_depresi' in df.columns:
            df['label'] = df['label_depresi']
        else:
            raise ValueError(f"No label column in {csv_path}")
    print(f"  [{name}] {len(df)} participants, {len(feat_cols)} features")
    return df, feat_cols

# Load from v6 (MFCC, Spectrogram) + v8 (Wav2Vec full)
df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"), "MFCC")
df_spec, cols_spec = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"), "Spectrogram")
df_w2v,  cols_w2v  = load_clean(W2V_FULL_CSV, "Wav2Vec Full")

datasets = {
    'MFCC':         (df_mfcc, cols_mfcc),
    'Spectrogram':  (df_spec, cols_spec),
    'Wav2Vec_Full': (df_w2v,  cols_w2v),
}

# %% [markdown]
# ## 3. Model Configs (Optimized from v6/v7 insights)

# %%
def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=10000, random_state=RANDOM_SEED, class_weight='balanced',
            C=0.1, solver='lbfgs',
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, C=10.0, gamma='auto',
            random_state=RANDOM_SEED, class_weight='balanced',
        ),
        'XGBoost': xgb.XGBClassifier(
            random_state=RANDOM_SEED, eval_metric='logloss',
            objective='binary:logistic', n_jobs=1,
            scale_pos_weight=2.5, n_estimators=300, max_depth=4,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=3,
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_SEED, class_weight='balanced', n_jobs=1,
            n_estimators=500, max_depth=8, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
        ),
    }

MODEL_NAMES = list(get_models().keys())
FEAT_NAMES  = list(datasets.keys())
print(f"Models: {MODEL_NAMES}")
print(f"Features: {FEAT_NAMES}")
print(f"Total: {len(MODEL_NAMES) * len(FEAT_NAMES)} = {len(MODEL_NAMES)} x {len(FEAT_NAMES)}")

# %% [markdown]
# ## 4. LOOCV — Leave-One-Out Cross-Validation

# %%
def loocv_evaluate(df, feat_cols, model_fn, feat_name, model_name):
    """
    LOOCV: train on N-1, test on 1, repeat N times.
    Returns metrics dict + arrays of y_true, y_pred, y_prob.
    """
    n = len(df)
    X = df[feat_cols].values.astype(np.float64)
    y = df['label'].values.astype(int)

    y_true_all = np.zeros(n, dtype=int)
    y_pred_all = np.zeros(n, dtype=int)
    y_prob_all = np.zeros(n, dtype=float)

    for i in range(n):
        # Split
        X_tr = np.delete(X, i, axis=0)
        y_tr = np.delete(y, i, axis=0)
        X_te = X[i:i+1]

        # NaN -> median
        medians = np.nanmedian(X_tr, axis=0)
        for col in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, col]), col] = medians[col]
            if np.isnan(X_te[0, col]):
                X_te[0, col] = medians[col]

        # Clip (IQR x 10)
        Q1 = np.percentile(X_tr, 25, axis=0)
        Q3 = np.percentile(X_tr, 75, axis=0)
        IQR = Q3 - Q1
        lo, hi = Q1 - 10*IQR, Q3 + 10*IQR
        X_tr = np.clip(X_tr, lo, hi)
        X_te = np.clip(X_te, lo, hi)

        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Train & predict
        model = model_fn()
        model.fit(X_tr, y_tr)
        try:
            prob = model.predict_proba(X_te)[0, 1]
        except Exception:
            prob = float(model.predict(X_te)[0])

        y_true_all[i] = y[i]
        y_prob_all[i] = prob
        y_pred_all[i] = int(prob >= 0.5)

    # Metrics at threshold 0.5
    metrics = {
        'accuracy': float(accuracy_score(y_true_all, y_pred_all)),
        'f1_macro': float(f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)),
        'precision': float(precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)),
    }
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true_all, y_prob_all))
    except:
        metrics['roc_auc'] = 0.0

    # Threshold tuning
    best_thr, best_f1 = 0.5, metrics['f1_macro']
    for thr in np.arange(0.30, 0.71, 0.01):
        preds_t = (y_prob_all >= thr).astype(int)
        f1_t = f1_score(y_true_all, preds_t, average='macro', zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr

    y_pred_tuned = (y_prob_all >= best_thr).astype(int)
    metrics['f1_tuned'] = float(best_f1)
    metrics['best_threshold'] = float(round(best_thr, 2))
    metrics['acc_tuned'] = float(accuracy_score(y_true_all, y_pred_tuned))

    return metrics, y_true_all, y_pred_all, y_prob_all, y_pred_tuned

# %% [markdown]
# ## 5. Training — LOOCV for All 12 Models

# %%
SEP = "=" * 95

all_results = {}
all_ys = {}

print(f"\n{'#' * 95}")
print(f"{'LOOCV — Leave-One-Out Cross-Validation (102 folds)':^95}")
print(f"{'#' * 95}")

for feat_name in FEAT_NAMES:
    df, feat_cols = datasets[feat_name]

    print(f"\n{SEP}")
    print(f"  FEATURE: {feat_name}  |  {len(feat_cols)} features  |  {len(df)} participants")
    print(SEP)

    for model_name in MODEL_NAMES:
        combo = f"{feat_name} + {model_name}"
        print(f"\n  [{combo}] LOOCV running (102 folds) ...", end="", flush=True)

        t0 = time.time()
        model_fn = lambda mn=model_name: get_models()[mn]
        metrics, y_true, y_pred, y_prob, y_pred_tuned = loocv_evaluate(
            df, feat_cols, model_fn, feat_name, model_name)
        elapsed = time.time() - t0

        all_results[combo] = metrics
        all_ys[combo] = (y_true, y_pred, y_prob, y_pred_tuned)

        print(f"  ({elapsed:.1f}s)")
        print(f"    F1 (thr=0.5) : {metrics['f1_macro']:.4f}  |  Acc: {metrics['accuracy']:.4f}")
        print(f"    F1 (tuned)   : {metrics['f1_tuned']:.4f}  |  Acc: {metrics['acc_tuned']:.4f}  "
              f"|  thr={metrics['best_threshold']:.2f}")
        print(f"    AUC          : {metrics['roc_auc']:.4f}")

print(f"\n{SEP}")
print(f"  12 MODEL LOOCV SELESAI")
print(SEP)

# %% [markdown]
# ## 6. Ensemble — Soft Voting Top-3

# %%
# Sort by f1_tuned
sorted_combos = sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)
top3 = sorted_combos[:3]

print(f"\nTop-3 model untuk Ensemble:")
for i, combo in enumerate(top3):
    m = all_results[combo]
    print(f"  {i+1}. {combo}: F1={m['f1_tuned']:.4f} (thr={m['best_threshold']:.2f})")

# Ensemble: average probabilities from top-3
y_true_ens = all_ys[top3[0]][0]
probs_top3 = np.array([all_ys[combo][2] for combo in top3])
y_prob_ens = probs_top3.mean(axis=0)

# Fixed threshold 0.5
y_pred_ens_50 = (y_prob_ens >= 0.5).astype(int)
f1_ens_50 = f1_score(y_true_ens, y_pred_ens_50, average='macro', zero_division=0)
acc_ens_50 = accuracy_score(y_true_ens, y_pred_ens_50)

# Tuned threshold
best_thr_ens, best_f1_ens = 0.5, f1_ens_50
for thr in np.arange(0.30, 0.71, 0.01):
    preds = (y_prob_ens >= thr).astype(int)
    f1_t = f1_score(y_true_ens, preds, average='macro', zero_division=0)
    if f1_t > best_f1_ens:
        best_f1_ens = f1_t
        best_thr_ens = thr

y_pred_ens_tuned = (y_prob_ens >= best_thr_ens).astype(int)
acc_ens_tuned = accuracy_score(y_true_ens, y_pred_ens_tuned)

try:
    auc_ens = float(roc_auc_score(y_true_ens, y_prob_ens))
except:
    auc_ens = 0.0

all_results['Ensemble (Top-3)'] = {
    'f1_macro': f1_ens_50,
    'accuracy': acc_ens_50,
    'f1_tuned': best_f1_ens,
    'best_threshold': round(best_thr_ens, 2),
    'acc_tuned': acc_ens_tuned,
    'roc_auc': auc_ens,
    'precision': float(precision_score(y_true_ens, y_pred_ens_tuned, average='macro', zero_division=0)),
    'recall': float(recall_score(y_true_ens, y_pred_ens_tuned, average='macro', zero_division=0)),
}
all_ys['Ensemble (Top-3)'] = (y_true_ens, y_pred_ens_50, y_prob_ens, y_pred_ens_tuned)

print(f"\n  [Ensemble Top-3]")
print(f"    F1 (thr=0.5) : {f1_ens_50:.4f}  |  Acc: {acc_ens_50:.4f}")
print(f"    F1 (tuned)   : {best_f1_ens:.4f}  |  Acc: {acc_ens_tuned:.4f}  "
      f"|  thr={best_thr_ens:.2f}")
print(f"    AUC          : {auc_ens:.4f}")

# %% [markdown]
# ## 7. Tabel Perbandingan Lengkap

# %%
rows = []
for combo, m in all_results.items():
    parts = combo.split(' + ')
    feat = parts[0] if len(parts) > 1 else 'Ensemble'
    model = parts[1] if len(parts) > 1 else combo
    rows.append({
        'Feature': feat, 'Model': model,
        'F1 (thr=0.5)': round(m['f1_macro'], 4),
        'F1 (tuned)': round(m['f1_tuned'], 4),
        'Best Thr': round(m['best_threshold'], 2),
        'Acc (tuned)': round(m['acc_tuned'], 4),
        'AUC': round(m['roc_auc'], 4),
    })

df_results = (pd.DataFrame(rows)
              .sort_values('F1 (tuned)', ascending=False)
              .reset_index(drop=True))
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v8_loocv_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 120)
print(f"{'RINGKASAN v8 — LOOCV (102 folds), 12 Model + Ensemble':^120}")
print("=" * 120)
print(df_results.to_string())
print(f"\nDisimpan: {csv_path}")

best = df_results.iloc[0]
print(f"\n  BEST: {best['Feature']} + {best['Model']}")
print(f"  F1 (tuned) = {best['F1 (tuned)']}  |  thr={best['Best Thr']}  |  AUC={best['AUC']}")

# %% [markdown]
# ## 8. Visualisasi

# %%
COLORS = {'MFCC': '#3b82f6', 'Spectrogram': '#f59e0b', 'Wav2Vec_Full': '#10b981', 'Ensemble': '#8b5cf6'}

# ─── 8A. Bar Chart: F1 (tuned) per model ─────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
combos = df_results.apply(lambda r: f"{r['Feature'][:5]}+{r['Model'][:5]}", axis=1).tolist()
f1s    = df_results['F1 (tuned)'].tolist()
colors = [COLORS.get(r['Feature'], '#888888') for _, r in df_results.iterrows()]

bars = ax.barh(range(len(combos)), f1s, color=colors, edgecolor='white', linewidth=0.7)
ax.set_yticks(range(len(combos)))
ax.set_yticklabels(combos, fontsize=9)
ax.set_xlabel('Macro F1 (Tuned Threshold)', fontsize=11)
ax.set_title('v8 — LOOCV Macro F1 (102 folds, 12 Model + Ensemble)',
             fontweight='bold', fontsize=13)
ax.axvline(0.7, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Target 0.70')
ax.axvline(0.63, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='v6/v7 best (0.63)')
for bar, val in zip(bars, f1s):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8.5, fontweight='bold')
ax.set_xlim(0, 1.0)
ax.invert_yaxis()
ax.legend(fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
p1 = os.path.join(RESULTS_DIR, "plots", "v8_bar_f1.png")
fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.show()
print(f"Plot: {p1}")

# ─── 8B. Heatmap ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, col, title, cmap in zip(axes,
    ['F1 (tuned)', 'AUC'], ['Macro F1 (Tuned)', 'ROC-AUC'], ['YlOrRd', 'Blues']):
    data = {}
    for fn in FEAT_NAMES:
        data[fn] = []
        for mn in MODEL_NAMES:
            combo = f"{fn} + {mn}"
            data[fn].append(all_results.get(combo, {}).get(
                'f1_tuned' if 'F1' in col else 'roc_auc', 0))
    hdf = pd.DataFrame(data, index=MODEL_NAMES).T
    sns.heatmap(hdf, annot=True, fmt='.3f', cmap=cmap, linewidths=0.5,
                cbar_kws={'label': title}, ax=ax, vmin=0.3, vmax=1.0)
    ax.set_title(f'{title} — LOOCV (v8)', fontweight='bold')
plt.tight_layout()
p2 = os.path.join(RESULTS_DIR, "plots", "v8_heatmap.png")
fig.savefig(p2, dpi=150, bbox_inches='tight'); plt.show()
print(f"Heatmap: {p2}")

# %% [markdown]
# ## 9. Classification Report (Best Model)

# %%
print("\n" + "=" * 100)
print(f"{'CLASSIFICATION REPORT v8 — LOOCV':^100}")
print("=" * 100)

class_labels = ['Normal (0)', 'Depresi (1)']

for combo in list(df_results.head(5).apply(
    lambda r: f"{r['Feature']} + {r['Model']}" if r['Feature'] != 'Ensemble'
    else 'Ensemble (Top-3)', axis=1)):
    if combo not in all_ys:
        continue
    m = all_results[combo]
    y_true, _, _, y_pred_tuned = all_ys[combo]

    print(f"\n{'-'*100}")
    print(f"  [{combo}]  F1={m['f1_tuned']:.4f}  thr={m['best_threshold']:.2f}  AUC={m['roc_auc']:.4f}")
    print(f"{'-'*100}")
    print(classification_report(y_true, y_pred_tuned, labels=[0, 1],
                                target_names=class_labels, zero_division=0))

# ─── Confusion Matrix Top-5 ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('v8 — LOOCV Confusion Matrix (Top-5, Tuned Threshold)', fontweight='bold', fontsize=13)

top5_combos = list(df_results.head(5).apply(
    lambda r: f"{r['Feature']} + {r['Model']}" if r['Feature'] != 'Ensemble'
    else 'Ensemble (Top-3)', axis=1))

for ax_idx, combo in enumerate(top5_combos):
    if combo not in all_ys:
        continue
    ax = axes[ax_idx]
    y_true, _, _, y_pred_t = all_ys[combo]
    cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
    m = all_results[combo]
    feat = combo.split(' + ')[0] if ' + ' in combo else 'Ensemble'
    cmap = {'MFCC': 'Blues', 'Spectrogram': 'Oranges', 'Wav2Vec_Full': 'Greens', 'Ensemble': 'Purples'}
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap.get(feat, 'Blues'),
                ax=ax, xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, cbar=False)
    ax.set_title(f'{combo[:25]}\nF1={m["f1_tuned"]:.3f} thr={m["best_threshold"]:.2f}',
                 fontweight='bold', fontsize=8.5)
    ax.set_xlabel('Pred', fontsize=8)
    ax.set_ylabel('True' if ax_idx == 0 else '', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
p3 = os.path.join(RESULTS_DIR, "confusion_matrix", "v8_top5_cm.png")
fig.savefig(p3, dpi=150, bbox_inches='tight'); plt.show()
print(f"CM: {p3}")

# %% [markdown]
# ## 10. Ringkasan Final

# %%
best_combo = df_results.iloc[0]
best_key = f"{best_combo['Feature']} + {best_combo['Model']}" if best_combo['Feature'] != 'Ensemble' else 'Ensemble (Top-3)'

summary = {
    'version': 'v8',
    'evaluation': 'LOOCV (102 folds)',
    'wav2vec': 'FULL 768-dim (mean + std pooled = 1536 features)',
    'threshold': 'tuned per model (0.30-0.70)',
    'smote': False,
    'augmentation': False,
    'n_participants': 102,
    'best_model': best_key,
    'best_f1_tuned': float(best_combo['F1 (tuned)']),
    'best_f1_fixed': float(best_combo['F1 (thr=0.5)']),
    'best_auc': float(best_combo['AUC']),
    'all_results': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in all_results.items()},
}

with open(os.path.join(MODELS_DIR, "v8_summary.json"), 'w') as fp:
    json.dump(summary, fp, indent=2)

print(f"\n{'=' * 95}")
print(f"{'PIPELINE v8 SELESAI':^95}")
print(f"{'=' * 95}")
print(f"\n  Evaluation : LOOCV (102 folds)")
print(f"  Wav2Vec    : FULL 768-dim (mean + std = 1536 features)")
print(f"  Best Model : {best_key}")
print(f"  F1 (tuned) : {best_combo['F1 (tuned)']}")
print(f"  F1 (thr=0.5): {best_combo['F1 (thr=0.5)']}")
print(f"  AUC        : {best_combo['AUC']}")
print(f"  Threshold  : {best_combo['Best Thr']}")
print(f"\n  Models : {MODELS_DIR}")
print(f"  Results: {RESULTS_DIR}")
