# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v47** — ADVANCED NLP WITH BERT EMBEDDINGS (Sentence Transformers)
#
# ─────────────────────────────────────────────────────────────────────
#  Tujuan: Menghancurkan batas F1 > 0.70 pada 189 Partisipan.
#  Data: Transkrip percakapan 189 Partisipan (hanya teks partisipan).
#  Fitur: Sentence-Transformers (BERT embeddings). Model: `all-MiniLM-L6-v2`.
#  Metode: Stratified 5-Fold Cross Validation & Train/Test Eval.
#  Model: LR, SVM, Random Forest, XGBoost.
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v47")

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

df_train['split'] = 'train'
df_dev['split'] = 'dev'
df_test['split'] = 'test'

df_labels = pd.concat([df_train, df_dev, df_test], ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)

# %% [markdown]
# ## Process Transcripts and Encode via BERT

# %%
print("Extracting participant texts and encoding via BERT...")
texts = []
labels = []
splits = []

for idx, row in df_labels.iterrows():
    pid = int(row['id'])
    lbl = int(row['label'])
    split = row['split']
    
    filepath = os.path.join(RAW_DIR, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(filepath):
        texts.append("")
        labels.append(lbl)
        splits.append(split)
        continue
        
    try:
        df_trans = pd.read_csv(filepath, sep='\t')
        if 'speaker' in df_trans.columns:
            df_part = df_trans[df_trans['speaker'].str.lower() == 'participant']
        else:
            df_part = df_trans
        if 'value' in df_part.columns:
            text = " ".join(df_part['value'].dropna().astype(str).tolist())
        else:
            text = ""
    except Exception as e:
        text = ""
        
    texts.append(text)
    labels.append(lbl)
    splits.append(split)

df_data = pd.DataFrame({'text': texts, 'label': labels, 'split': splits})

# ENCODING WITH BERT
print("Downloading/Loading SentenceTransformer Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_all = embedder.encode(df_data['text'].values, show_progress_bar=True)
y_all = df_data['label'].values

print(f"BERT Embeddings Shape: {X_all.shape}")

# %% [markdown]
# ## Modeling (5-Fold CV across ALL 189)

# %%
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced', max_iter=1000, C=1.0),
    'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED, class_weight='balanced', C=1.0),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_estimators=300),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', scale_pos_weight=2.0)
}

results = []

for name, model in models.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    y_true_all = []
    y_prob_all = []
    
    for train_index, test_index in skf.split(X_all, y_all):
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        
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
    
    # Tune threshold
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
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v47_cv_results.csv"), index=False)

print("\n" + "="*50)
print("FINAL RESULTS (v47 BERT NLP) - 189 Participants (5-Fold CV)")
print("="*50)
print(df_res.to_string(index=False))

# %% [markdown]
# ## Evaluate Official Test Set

# %%
train_mask = (df_data['split'] == 'train')
dev_mask = (df_data['split'] == 'dev')
test_mask = (df_data['split'] == 'test')

X_train = X_all[train_mask]
y_train = y_all[train_mask]
X_dev = X_all[dev_mask]
y_dev = y_all[dev_mask]
X_test = X_all[test_mask]
y_test = y_all[test_mask]

sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED, class_weight='balanced', C=1.0)
svm.fit(X_train_res, y_train_res)

dev_probs = svm.predict_proba(X_dev)[:, 1]
best_f1, best_thr = 0.0, 0.5
for thr in np.arange(0.30, 0.71, 0.01):
    preds = (dev_probs >= thr).astype(int)
    f1 = f1_score(y_dev, preds, average='macro', zero_division=0)
    if f1 > best_f1: best_f1, best_thr = f1, thr

test_probs = svm.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_thr).astype(int)
test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)

print("\n" + "="*50)
print("OFFICIAL TEST SPLIT EVALUATION (SVM BERT)")
print("="*50)
print(f"Dev Macro F1 : {best_f1:.4f} (Thr: {best_thr:.2f})")
print(f"Test Macro F1: {test_f1:.4f}")

df_test_res = pd.DataFrame([{'Split': 'Test', 'Model': 'SVM (BERT)', 'Macro_F1': test_f1}])
df_test_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v47_test_results.csv"), index=False)
