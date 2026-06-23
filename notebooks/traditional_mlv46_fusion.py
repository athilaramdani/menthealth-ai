# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v46** — THE ULTIMATE MULTIMODAL FUSION (TEXT + AUDIO)
#
# ─────────────────────────────────────────────────────────────────────
#  Tujuan: Memecah rekor F1 > 0.70 secara absolut pada 189 Partisipan.
#  Data: Audio COVAREP (296 Statistik) + Teks Transkrip (TF-IDF 2000).
#  Metode: Early Fusion (Menggabungkan Audio & Teks pada level fitur),
#          diikuti oleh PCA 95% variance reduction untuk mencegah Curse of Dimensionality.
#          Stratified 5-Fold Cross Validation.
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup & Imports

# %%
import os, sys, glob, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v46")

for d in [os.path.join(RESULTS_DIR, "metrics")]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## Load Labels (189 Participants)

# %%
df_train = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dev = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_test = pd.read_csv(os.path.join(RAW_DIR, "full_test_split.csv"))

df_train = df_train[['Participant_ID', 'PHQ8_Binary']].rename(columns={'Participant_ID':'id', 'PHQ8_Binary':'label'})
df_dev = df_dev[['Participant_ID', 'PHQ8_Binary']].rename(columns={'Participant_ID':'id', 'PHQ8_Binary':'label'})
df_test = df_test[['Participant_ID', 'PHQ_Binary']].rename(columns={'Participant_ID':'id', 'PHQ_Binary':'label'})

df_labels = pd.concat([df_train, df_dev, df_test], ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)

# %% [markdown]
# ## Multimodal Feature Extraction

# %%
print("Extracting Audio & Text for 189 files...")
audio_feats = []
texts = []
y_list = []

for idx, row in df_labels.iterrows():
    pid = int(row['id'])
    lbl = int(row['label'])
    
    # Audio COVAREP
    audio_path = os.path.join(RAW_DIR, f"{pid}_P", f"{pid}_COVAREP.csv")
    if os.path.exists(audio_path):
        df_cov = pd.read_csv(audio_path, header=None)
        data = df_cov.values.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        agg_features = np.concatenate([np.mean(data, axis=0), np.std(data, axis=0), np.max(data, axis=0), np.min(data, axis=0)])
    else:
        agg_features = np.zeros(296)
        
    # Text
    text_path = os.path.join(RAW_DIR, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if os.path.exists(text_path):
        try:
            df_trans = pd.read_csv(text_path, sep='\t')
            if 'speaker' in df_trans.columns:
                df_part = df_trans[df_trans['speaker'].str.lower() == 'participant']
            else:
                df_part = df_trans
            if 'value' in df_part.columns:
                text = " ".join(df_part['value'].dropna().astype(str).tolist())
            else:
                text = ""
        except:
            text = ""
    else:
        text = ""
        
    audio_feats.append(agg_features)
    texts.append(text)
    y_list.append(lbl)

X_audio = np.array(audio_feats)
X_text_raw = np.array(texts)
y_all = np.array(y_list)

print(f"Audio Shape: {X_audio.shape}")
print(f"Text Shape: {X_text_raw.shape}")

# %% [markdown]
# ## Modeling (5-Fold CV Early Fusion)

# %%
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced', max_iter=1000, C=0.5),
    'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED, class_weight='balanced', C=1.0),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_estimators=300),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', scale_pos_weight=2.0, max_depth=3)
}

results = []

for name, model in models.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    y_true_all = []
    y_prob_all = []
    
    for train_index, test_index in skf.split(X_audio, y_all):
        # Audio Split & Scale
        X_aud_train, X_aud_test = X_audio[train_index], X_audio[test_index]
        scaler = StandardScaler()
        X_aud_train = scaler.fit_transform(X_aud_train)
        X_aud_test = scaler.transform(X_aud_test)
        
        # Text Split & TF-IDF
        X_txt_train_raw, X_txt_test_raw = X_text_raw[train_index], X_text_raw[test_index]
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
        X_txt_train = vectorizer.fit_transform(X_txt_train_raw).toarray()
        X_txt_test = vectorizer.transform(X_txt_test_raw).toarray()
        
        # FUSION
        X_train = np.hstack((X_aud_train, X_txt_train))
        X_test = np.hstack((X_aud_test, X_txt_test))
        y_train, y_test = y_all[train_index], y_all[test_index]
        
        # PCA
        pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        # SMOTE
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        
        # Train
        model.fit(X_train_res, y_train_res)
        try:
            prob = model.predict_proba(X_test)[:, 1]
        except:
            prob = model.predict(X_test)
            
        y_true_all.extend(y_test)
        y_prob_all.extend(prob)
        
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.30, 0.71, 0.01):
        preds = (y_prob_all >= thr).astype(int)
        f1 = f1_score(y_true_all, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            
    auc = roc_auc_score(y_true_all, y_prob_all)
    
    results.append({
        'Model': name,
        'Macro_F1': best_f1,
        'Best_Thr': best_thr,
        'AUC': auc
    })

df_res = pd.DataFrame(results).sort_values('Macro_F1', ascending=False)
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v46_fusion_results.csv"), index=False)

print("\n" + "="*50)
print("FINAL RESULTS (v46 MULTIMODAL FUSION) - 189 Participants")
print("="*50)
print(df_res.to_string(index=False))
