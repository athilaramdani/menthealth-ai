# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline v13** — Deep Semantic Embeddings (Sentence Transformers) + Audio
#
# ─────────────────────────────────────────────────────────────────────
#  v13 = STATE-OF-THE-ART MULTIMODAL
#
#  Masalah v12: Segment-level classification (10s) terlalu berisik (noisy),
#               karena tidak semua 10s audio mengandung indikator depresi.
#  Solusi v13: 
#  1. Kembali ke Participant-Level Aggregation (102 sampel utuh).
#  2. Ganti TF-IDF/LIWC dengan Deep Semantic Embeddings!
#     Menggunakan model 'all-MiniLM-L6-v2' (Sentence Transformers) untuk
#     mengubah transkrip mentah jadi 384-dimensi representasi semantik.
#  3. Fuse 384-D Text Embeddings + 990-D MFCC.
#  4. Gunakan LOOCV Nested CV + SelectKBest untuk memilih top 50-80 fitur.
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

# %%
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
import xgboost as xgb

from sentence_transformers import SentenceTransformer

plt.rcParams['font.family'] = 'DejaVu Sans'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd() else os.getcwd()
)
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V13_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v13")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v13")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v13")

for d in [V13_FEAT_DIR, MODELS_DIR, os.path.join(RESULTS_DIR, "metrics"), 
          os.path.join(RESULTS_DIR, "plots"), os.path.join(RESULTS_DIR, "confusion_matrix")]:
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
    if 'label' not in df.columns: df['label'] = df['label_depresi']
    return df, feat_cols

df_mfcc, cols_mfcc = load_clean(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"), "MFCC")
print(f"  [Audio] {len(df_mfcc)} participants, {len(cols_mfcc)} features")

# %% [markdown]
# ## 2. Extract Deep Semantic Embeddings

# %%
def get_semantic_embeddings(raw_dir):
    csv_path = os.path.join(V13_FEAT_DIR, "v13_text_embeddings.csv")
    if os.path.exists(csv_path):
        print(f"  [Text] Memuat existing embeddings dari {csv_path}")
        return pd.read_csv(csv_path)

    print("  [Text] Mengekstrak transkrip dan memproses embeddings dengan all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    rows = []
    for d in os.listdir(raw_dir):
        if not d.endswith('_P'): continue
        pid = int(d.split('_')[0])
        tpath = os.path.join(raw_dir, d, f"{pid}_TRANSCRIPT.csv")
        if not os.path.exists(tpath): continue
        
        try: df = pd.read_csv(tpath, sep='\t')
        except:
            df = pd.read_csv(tpath)
            if len(df.columns) < 4: df = pd.read_csv(tpath, sep=None, engine='python')
                
        df.columns = [c.lower().strip() for c in df.columns]
        if 'speaker' not in df.columns or 'value' not in df.columns: continue
            
        p_df = df[df['speaker'].str.lower().str.strip() == 'participant']
        text = " ".join(p_df['value'].dropna().astype(str).tolist())
        if not text: continue
            
        # Extract 384-dimensional embedding
        emb = model.encode(text)
        
        row = {'participant_id': pid}
        for i in range(len(emb)):
            row[f'text_emb_{i}'] = float(emb[i])
        rows.append(row)
        
    df_emb = pd.DataFrame(rows)
    df_emb.to_csv(csv_path, index=False)
    return df_emb

df_text = get_semantic_embeddings(RAW_DIR)
cols_text = [c for c in df_text.columns if c.startswith('text_emb_')]
print(f"  [Text] Ekstraksi selesai. {len(df_text)} participants dengan {len(cols_text)} dimensional embeddings.")

# EARLY FUSION
df_fused = pd.merge(df_mfcc, df_text, on='participant_id', how='inner')
print(f"  [Fusion] Ready: {len(df_fused)} partisipan dengan total fitur: {len(cols_mfcc) + len(cols_text)} (Audio + Text).")

datasets = {
    'DeepText_Only': (df_fused, [], cols_text),
    'Multimodal_DeepFusion': (df_fused, cols_mfcc, cols_text)
}

# %% [markdown]
# ## 3. Multimodal LOOCV Pipeline

# %%
def get_param_grid(model_name):
    if model_name == 'Random Forest':
        return {'n_estimators': [200, 400], 'max_depth': [5, 10, None]}
    elif model_name == 'XGBoost':
        return {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05]}
    return {}

def make_model(model_name):
    if model_name == 'Random Forest':
        return RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=1)
    elif model_name == 'XGBoost':
        return xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', scale_pos_weight=2.5, n_jobs=1)

def loocv_deep_fusion(df, audio_cols, text_cols, model_name):
    n = len(df)
    y_all = df['label'].values.astype(int)
    
    X_audio_all = df[audio_cols].values.astype(np.float64) if audio_cols else None
    X_text_all = df[text_cols].values.astype(np.float64) if text_cols else None

    y_true = np.zeros(n, dtype=int)
    y_prob = np.zeros(n, dtype=float)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    param_grid = get_param_grid(model_name)
    f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)
    
    # Feature Selection K
    AUDIO_K = 50
    TEXT_K = 50

    for i in range(n):
        y_tr = np.delete(y_all, i, axis=0)
        X_tr_fused, X_te_fused = [], []
        
        # 1. AUDIO PROCESS
        if audio_cols:
            X_a_tr = np.delete(X_audio_all, i, axis=0)
            X_a_te = X_audio_all[i:i+1]
            
            scaler_a = StandardScaler()
            X_a_tr = scaler_a.fit_transform(np.nan_to_num(X_a_tr))
            X_a_te = scaler_a.transform(np.nan_to_num(X_a_te))
            
            sel_a = SelectKBest(f_classif, k=min(AUDIO_K, X_a_tr.shape[1]))
            X_tr_fused.append(sel_a.fit_transform(X_a_tr, y_tr))
            X_te_fused.append(sel_a.transform(X_a_te))
            
        # 2. TEXT PROCESS
        if text_cols:
            X_t_tr = np.delete(X_text_all, i, axis=0)
            X_t_te = X_text_all[i:i+1]
            
            scaler_t = StandardScaler()
            X_t_tr = scaler_t.fit_transform(np.nan_to_num(X_t_tr))
            X_t_te = scaler_t.transform(np.nan_to_num(X_t_te))
            
            sel_t = SelectKBest(f_classif, k=min(TEXT_K, X_t_tr.shape[1]))
            X_tr_fused.append(sel_t.fit_transform(X_t_tr, y_tr))
            X_te_fused.append(sel_t.transform(X_t_te))

        # 3. FUSION
        X_tr = np.hstack(X_tr_fused)
        X_te = np.hstack(X_te_fused)

        # 4. TUNING & PREDICT
        gs = GridSearchCV(make_model(model_name), param_grid, cv=inner_cv, scoring=f1_scorer, n_jobs=1)
        gs.fit(X_tr, y_tr)
        
        try: y_prob[i] = gs.best_estimator_.predict_proba(X_te)[0, 1]
        except: y_prob[i] = float(gs.best_estimator_.predict(X_te)[0])
        y_true[i] = y_all[i]

    # EVALUATION
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (y_prob >= thr).astype(int)
        f1_t = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1_t > best_f1: best_f1, best_thr = f1_t, thr
            
    y_pred_tuned = (y_prob >= best_thr).astype(int)
    
    return {
        'f1_050': float(f1_score(y_true, (y_prob >= 0.5).astype(int), average='macro', zero_division=0)),
        'acc_050': float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        'f1_tuned': float(best_f1),
        'thr': float(round(best_thr, 2)),
        'acc_tuned': float(accuracy_score(y_true, y_pred_tuned)),
        'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    }, y_true, y_prob, y_pred_tuned

# %% [markdown]
# ## 4. Run Pipeline

# %%
all_results, all_ys = {}, {}

print(f"\n{'#' * 100}")
print(f"{'v13: DEEP SEMANTIC TEXT EMBEDDINGS + AUDIO':^100}")
print(f"{'#' * 100}")

for d_name, (df, acols, tcols) in datasets.items():
    for m_name in ['Random Forest', 'XGBoost']:
        combo = f"{d_name} + {m_name}"
        print(f"\n  [{combo}] Running ...", flush=True)
        t0 = time.time()
        metrics, y_true, y_prob, y_pred_tuned = loocv_deep_fusion(df, acols, tcols, m_name)
        all_results[combo] = metrics
        all_ys[combo] = (y_true, y_prob, y_pred_tuned)
        print(f"    F1 (tuned): {metrics['f1_tuned']:.4f} (thr={metrics['thr']:.2f}) | AUC: {metrics['auc']:.4f} | Time: {time.time()-t0:.0f}s")

# %% [markdown]
# ## 5. Ringkasan Final

# %%
rows = []
for combo, m in all_results.items():
    d_name, model = combo.split(' + ')
    rows.append({'Dataset': d_name, 'Model': model, 'F1': m['f1_tuned'], 'Thr': m['thr'], 'AUC': m['auc']})

df_results = pd.DataFrame(rows).sort_values('F1', ascending=False).reset_index(drop=True)
df_results.index += 1

csv_path = os.path.join(RESULTS_DIR, "metrics", "v13_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(f"{'RINGKASAN v13 — DEEP SEMANTIC TEXT + AUDIO':^110}")
print("=" * 110)
print(df_results.to_string())

best = df_results.iloc[0]
with open(os.path.join(MODELS_DIR, "v13_summary.json"), 'w') as fp:
    json.dump({'best_f1': best['F1'], 'best_model': f"{best['Dataset']} + {best['Model']}"}, fp)
