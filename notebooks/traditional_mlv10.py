# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v10** — Multimodal (Text + Audio) Klasifikasi Depresi
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v10 = BREAKING THE 0.70 CEILING via NLP & Multimodal Fusion
#
#  [1] Text NLP Pipeline: Parse `_TRANSCRIPT.csv`, ekstrak semua 
#      perkataan Participant.
#  [2] TF-IDF Vectorization dalam CV loop (hindari data leakage)
#  [3] Multimodal Early Fusion: 
#      Audio Features (MFCC / Wav2Vec) + Text Features (TF-IDF)
#  [4] Nested CV (LOOCV + 5-Fold Inner CV) dengan Feature Selection
#  [5] Models: Random Forest & XGBoost
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

_pip("scikit-learn", "sklearn"); _pip("xgboost"); _pip("seaborn"); _pip("pandas"); _pip("numpy")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("[OK] Dependencies siap.\n")

# %%
import os, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
import xgboost as xgb

plt.rcParams['font.family'] = 'DejaVu Sans'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd() else os.getcwd()
)
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v8")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v10")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v10")

for d in [MODELS_DIR,
          os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix")]:
    os.makedirs(d, exist_ok=True)

print(f"Project: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Load Audio Features

# %%
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender', 'label']

def load_clean(path, name):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].fillna(0)
    std_v = df[feat_cols].std()
    feat_cols = [c for c in feat_cols if std_v[c] > 1e-8]
    if 'label' not in df.columns:
        if 'label_depresi' in df.columns:
            df['label'] = df['label_depresi']
    print(f"  [Audio: {name}] {len(df)} participants, {len(feat_cols)} features")
    return df, feat_cols

df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"), "MFCC")
df_w2v,  cols_w2v  = load_clean(os.path.join(V8_FEAT_DIR, "daic_v8_wav2vec_full.csv"), "Wav2Vec")

# %% [markdown]
# ## 2. Extract Text Features (Transcripts)

# %%
def extract_transcripts(raw_dir):
    rows = []
    for d in os.listdir(raw_dir):
        if not d.endswith('_P'): continue
        pid = d.split('_')[0]
        tpath = os.path.join(raw_dir, d, f"{pid}_TRANSCRIPT.csv")
        if not os.path.exists(tpath): continue
        
        try:
            df = pd.read_csv(tpath, sep='\t')
        except:
            df = pd.read_csv(tpath)
            if len(df.columns) < 4:
                df = pd.read_csv(tpath, sep=None, engine='python')
                
        df.columns = [c.lower().strip() for c in df.columns]
        if 'speaker' not in df.columns or 'value' not in df.columns:
            continue
            
        # Get only participant's text
        p_df = df[df['speaker'].str.lower().str.strip() == 'participant']
        words = p_df['value'].dropna().astype(str).tolist()
        
        # Simple cleaning: remove common transcriber notes if any like [laughter], [sigh]
        # Or we can keep them! Laughter might be inversely correlated with depression.
        # But TF-IDF handles this well.
        text = " ".join(words)
        rows.append({'participant_id': int(pid), 'text': text})
    
    return pd.DataFrame(rows)

df_text = extract_transcripts(RAW_DIR)
print(f"  [Text] Ekstraksi selesai. {len(df_text)} participants memiliki transkrip.")

# Merge to ensure exact alignment with Audio Data
df_mfcc_fused = pd.merge(df_mfcc, df_text, on='participant_id', how='inner')
df_w2v_fused  = pd.merge(df_w2v, df_text, on='participant_id', how='inner')

print(f"  [Merged MFCC] {len(df_mfcc_fused)} participants ready.")
print(f"  [Merged W2V] {len(df_w2v_fused)} participants ready.")

datasets = {
    'TextOnly': (df_mfcc_fused, [], True),          # Audio cols=[], UseText=True
    'MFCC_AudioOnly': (df_mfcc_fused, cols_mfcc, False),
    'W2V_AudioOnly': (df_w2v_fused, cols_w2v, False),
    'Multimodal_MFCC': (df_mfcc_fused, cols_mfcc, True),
    'Multimodal_W2V': (df_w2v_fused, cols_w2v, True)
}

# %% [markdown]
# ## 3. Multimodal LOOCV Pipeline

# %%
def get_param_grid(model_name):
    if model_name == 'Random Forest':
        return {
            'n_estimators': [200, 400],
            'max_depth': [5, 10, None],
            'max_features': ['sqrt', 'log2'],
        }
    elif model_name == 'XGBoost':
        return {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8],
        }
    return {}

def make_model(model_name):
    if model_name == 'Random Forest':
        return RandomForestClassifier(
            random_state=RANDOM_SEED, class_weight='balanced', n_jobs=1)
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(
            random_state=RANDOM_SEED, eval_metric='logloss',
            objective='binary:logistic', n_jobs=1,
            scale_pos_weight=2.5)
    raise ValueError(f"Unknown model: {model_name}")

def loocv_multimodal(df, audio_cols, use_text, model_name):
    """
    LOOCV for multimodal fusion.
    Outer: Leave-One-Out (102 folds)
    Inner: TF-IDF vectorization -> SelectKBest -> GridSearchCV
    """
    n = len(df)
    y_all = df['label'].values.astype(int)
    
    # Audio Matrix (optional)
    if audio_cols:
        X_audio_all = df[audio_cols].values.astype(np.float64)
    else:
        X_audio_all = None
        
    # Text Array (optional)
    if use_text:
        text_all = df['text'].values.astype(str)
    else:
        text_all = None

    y_true = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_grid = get_param_grid(model_name)
    f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)
    
    # Fixed K to speed up
    AUDIO_K = 50
    TEXT_K = 100

    for i in range(n):
        # ─── Split ──────────────────────────────────────────
        y_tr = np.delete(y_all, i, axis=0)
        
        X_tr_fused = []
        X_te_fused = []
        
        # ─── 1. AUDIO PROCESS ───────────────────────────────
        if audio_cols:
            X_a_tr = np.delete(X_audio_all, i, axis=0)
            X_a_te = X_audio_all[i:i+1]
            
            # Clean
            X_a_tr = np.nan_to_num(X_a_tr, nan=0, posinf=0, neginf=0)
            X_a_te = np.nan_to_num(X_a_te, nan=0, posinf=0, neginf=0)
            
            # Scale
            scaler = StandardScaler()
            X_a_tr_sc = scaler.fit_transform(X_a_tr)
            X_a_te_sc = scaler.transform(X_a_te)
            
            # Feature Selection
            k_act = min(AUDIO_K, X_a_tr_sc.shape[1])
            sel_a = SelectKBest(f_classif, k=k_act)
            X_a_tr_sel = sel_a.fit_transform(X_a_tr_sc, y_tr)
            X_a_te_sel = sel_a.transform(X_a_te_sc)
            
            X_tr_fused.append(X_a_tr_sel)
            X_te_fused.append(X_a_te_sel)
            
        # ─── 2. TEXT PROCESS ────────────────────────────────
        if use_text:
            text_tr = np.delete(text_all, i, axis=0)
            text_te = text_all[i:i+1]
            
            # TF-IDF
            # Using bigrams helps capture phrases like "i am", "not sure"
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
            X_t_tr = tfidf.fit_transform(text_tr).toarray()
            X_t_te = tfidf.transform(text_te).toarray()
            
            # Text Feature Selection
            sel_t = SelectKBest(f_classif, k=TEXT_K)
            X_t_tr_sel = sel_t.fit_transform(X_t_tr, y_tr)
            X_t_te_sel = sel_t.transform(X_t_te)
            
            X_tr_fused.append(X_t_tr_sel)
            X_te_fused.append(X_t_te_sel)

        # ─── 3. FUSION ──────────────────────────────────────
        X_tr = np.hstack(X_tr_fused)
        X_te = np.hstack(X_te_fused)

        # ─── 4. MODEL TUNING ────────────────────────────────
        gs = GridSearchCV(
            make_model(model_name), param_grid,
            cv=inner_cv, scoring=f1_scorer,
            n_jobs=1, refit=True
        )
        gs.fit(X_tr, y_tr)
        model_final = gs.best_estimator_

        # ─── 5. PREDICT ─────────────────────────────────────
        try:
            prob = model_final.predict_proba(X_te)[0, 1]
        except:
            prob = float(model_final.predict(X_te)[0])

        y_true[i] = y_all[i]
        y_prob[i] = prob

    # ─── EVALUATION ─────────────────────────────────────────
    y_pred_50 = (y_prob >= 0.5).astype(int)
    metrics = {
        'f1_050': float(f1_score(y_true, y_pred_50, average='macro', zero_division=0)),
        'acc_050': float(accuracy_score(y_true, y_pred_50)),
    }
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
    except:
        metrics['auc'] = 0.0

    # Threshold tuning
    best_thr, best_f1 = 0.5, metrics['f1_050']
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (y_prob >= thr).astype(int)
        f1_t = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr
            
    y_pred_tuned = (y_prob >= best_thr).astype(int)
    metrics['f1_tuned'] = float(best_f1)
    metrics['thr'] = float(round(best_thr, 2))
    metrics['acc_tuned'] = float(accuracy_score(y_true, y_pred_tuned))

    return metrics, y_true, y_pred_50, y_prob, y_pred_tuned

# %% [markdown]
# ## 4. Run Pipeline

# %%
MODEL_NAMES = ['Random Forest', 'XGBoost']
SEP = "=" * 100

all_results = {}
all_ys = {}

print(f"\n{'#' * 100}")
print(f"{'v10: MULTIMODAL (TEXT + AUDIO) LOOCV NESTED CV':^100}")
print(f"{'#' * 100}")

for d_name, (df, acols, u_text) in datasets.items():
    print(f"\n{SEP}")
    print(f"  DATASET: {d_name}  |  Audio Feats: {len(acols)}  |  Use Text: {u_text}")
    print(SEP)

    for m_name in MODEL_NAMES:
        combo = f"{d_name} + {m_name}"
        print(f"\n  [{combo}] Running ...", flush=True)

        t0 = time.time()
        metrics, y_true, y_pred_50, y_prob, y_pred_tuned = loocv_multimodal(
            df, acols, u_text, m_name)
        elapsed = time.time() - t0

        all_results[combo] = metrics
        all_ys[combo] = (y_true, y_pred_50, y_prob, y_pred_tuned)

        print(f"    Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"    F1 (0.5)   : {metrics['f1_050']:.4f}  |  Acc: {metrics['acc_050']:.4f}")
        print(f"    F1 (tuned) : {metrics['f1_tuned']:.4f}  |  Acc: {metrics['acc_tuned']:.4f}  |  thr={metrics['thr']:.2f}")
        print(f"    AUC        : {metrics['auc']:.4f}")

# %% [markdown]
# ## 5. Tabel Perbandingan Final

# %%
rows = []
for combo, m in all_results.items():
    d_name, model = combo.split(' + ')
    rows.append({
        'Dataset': d_name, 'Model': model,
        'F1 (0.5)': round(m['f1_050'], 4),
        'F1 (tuned)': round(m['f1_tuned'], 4),
        'Thr': round(m['thr'], 2),
        'Acc': round(m['acc_tuned'], 4),
        'AUC': round(m['auc'], 4),
    })

df_results = (pd.DataFrame(rows)
              .sort_values('F1 (tuned)', ascending=False)
              .reset_index(drop=True))
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v10_multimodal_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(f"{'RINGKASAN v10 — MULTIMODAL TEXT + AUDIO':^110}")
print("=" * 110)
print(df_results.to_string())
print(f"\nDisimpan: {csv_path}")

best = df_results.iloc[0]
print(f"\n  BEST: {best['Dataset']} {best['Model']}")
print(f"  F1 (tuned) = {best['F1 (tuned)']}  |  thr={best['Thr']}  |  AUC={best['AUC']}")

# %% [markdown]
# ## 6. Visualisasi

# %%
COLORS = {
    'TextOnly': '#f59e0b',
    'MFCC_AudioOnly': '#3b82f6', 'W2V_AudioOnly': '#10b981',
    'Multimodal_MFCC': '#8b5cf6', 'Multimodal_W2V': '#ef4444'
}

# ─── Bar Chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
labels = df_results.apply(lambda r: f"{r['Dataset']} + {r['Model']}", axis=1).tolist()
f1s = df_results['F1 (tuned)'].tolist()
colors = [COLORS.get(r['Dataset'], '#888') for _, r in df_results.iterrows()]

bars = ax.barh(range(len(labels)), f1s, color=colors, edgecolor='white', linewidth=0.7)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Macro F1 (Tuned)', fontsize=11)
ax.set_title('v10 — Multimodal Text + Audio (LOOCV)', fontweight='bold', fontsize=12)
ax.axvline(0.7, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Target 0.70')
ax.axvline(0.647, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='v9 audio-only best (0.647)')

for bar, val in zip(bars, f1s):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlim(0, 1.0)
ax.invert_yaxis()
ax.legend(fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()

p1 = os.path.join(RESULTS_DIR, "plots", "v10_bar_f1.png")
fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.show()
print(f"Plot: {p1}")

# %% [markdown]
# ## 7. Classification Report

# %%
class_labels = ['Normal (0)', 'Depresi (1)']

print("\n" + "=" * 100)
print(f"{'CLASSIFICATION REPORT v10':^100}")
print("=" * 100)

for combo in sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)[:3]:
    m = all_results[combo]
    y_true, _, _, y_pred_t = all_ys[combo]

    print(f"\n{'-'*100}")
    print(f"  [{combo}]  F1={m['f1_tuned']:.4f}  thr={m['thr']:.2f}  AUC={m['auc']:.4f}")
    print(f"{'-'*100}")
    print(classification_report(y_true, y_pred_t, labels=[0, 1],
                                target_names=class_labels, zero_division=0))

# ─── Confusion Matrix ───────────────────────────────────────────────────
top_combos = sorted(all_results.keys(), key=lambda k: all_results[k]['f1_tuned'], reverse=True)[:4]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('v10 — Top 4 Confusion Matrix (Tuned Threshold)', fontweight='bold', fontsize=13)

for idx, combo in enumerate(top_combos):
    y_true, _, _, y_pred_t = all_ys[combo]
    cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
    m = all_results[combo]
    dataset = combo.split(' + ')[0]
    cmap_d = COLORS.get(dataset, '#3b82f6')
    
    # seaborn requires a colormap name, so we use general palettes
    cmap_choice = 'Blues'
    if 'Text' in dataset: cmap_choice = 'Oranges'
    elif 'Multimodal' in dataset: cmap_choice = 'Purples'
        
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_choice,
                ax=axes[idx], xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, cbar=False)
    axes[idx].set_title(f'{combo[:25]}\nF1={m["f1_tuned"]:.3f} thr={m["thr"]:.2f}',
                 fontweight='bold', fontsize=9)
    axes[idx].set_xlabel('Pred', fontsize=8)
    if idx == 0: axes[idx].set_ylabel('True', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
p3 = os.path.join(RESULTS_DIR, "confusion_matrix", "v10_cm.png")
fig.savefig(p3, dpi=150, bbox_inches='tight'); plt.show()
print(f"CM: {p3}")

# %% [markdown]
# ## 8. Ringkasan Final

# %%
best_key = top_combos[0]
bm = all_results[best_key]

summary = {
    'version': 'v10',
    'evaluation': 'LOOCV (102 folds) + Nested 5-fold GridSearchCV',
    'approach': 'Multimodal (Audio + Text Early Fusion)',
    'best_model': best_key,
    'best_f1_tuned': round(bm['f1_tuned'], 4),
    'best_f1_050': round(bm['f1_050'], 4),
    'best_auc': round(bm['auc'], 4),
    'best_threshold': bm['thr'],
}
with open(os.path.join(MODELS_DIR, "v10_summary.json"), 'w') as fp:
    json.dump(summary, fp, indent=2)

print(f"\n{'=' * 100}")
print(f"{'PIPELINE v10 SELESAI':^100}")
print(f"{'=' * 100}")
print(f"\n  Method   : Multimodal LOOCV + Text TF-IDF + Nested CV")
print(f"  Best     : {best_key}")
print(f"  F1 tuned : {bm['f1_tuned']:.4f}  (thr={bm['thr']:.2f})")
print(f"  F1 (0.5) : {bm['f1_050']:.4f}")
print(f"  AUC      : {bm['auc']:.4f}")

print(f"\n  Cross-Version Comparison:")
print(f"  {'Version':<8} {'Eval':<15} {'Best Model':<25} {'F1':>6}")
print(f"  {'-'*60}")
print(f"  {'v4':<8} {'AVEC test':15} {'Spec+SVM':<25} {'0.544':>6}")
print(f"  {'v6':<8} {'80/10/10':15} {'W2V+XGB':<25} {'0.629':>6}")
print(f"  {'v8':<8} {'LOOCV':15} {'MFCC+RF':<25} {'0.624':>6}")
print(f"  {'v9':<8} {'LOOCV+Nest':15} {'MFCC+XGB':<25} {'0.647':>6}")
print(f"  {'v10':<8} {'Multimodal':15} {best_key:<25} {bm['f1_tuned']:>6.3f}")

print(f"\n  Models : {MODELS_DIR}")
print(f"  Results: {RESULTS_DIR}")
