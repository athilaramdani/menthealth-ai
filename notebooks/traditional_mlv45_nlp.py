# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v45** — NATURAL LANGUAGE PROCESSING (TF-IDF on Transcripts)
#
# ─────────────────────────────────────────────────────────────────────
#  Tujuan: Menembus F1 > 0.70 secara absolut dengan 189 Partisipan utuh.
#  Data: Transkrip percakapan 189 Partisipan (hanya teks partisipan).
#  Fitur: TF-IDF (Term Frequency - Inverse Document Frequency) uni-bigrams.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v45")

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
# ## Process Transcripts

# %%
print("Extracting participant texts from 189 transcript files...")
texts = []
labels = []
splits = []

for idx, row in df_labels.iterrows():
    pid = int(row['id'])
    lbl = int(row['label'])
    split = row['split']
    
    filepath = os.path.join(RAW_DIR, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(filepath):
        print(f"Missing {filepath}")
        texts.append("")
        labels.append(lbl)
        splits.append(split)
        continue
        
    try:
        # Transcripts are TSV
        df_trans = pd.read_csv(filepath, sep='\t')
        
        # Only keep 'Participant' speech, not 'Ellie'
        if 'speaker' in df_trans.columns:
            df_part = df_trans[df_trans['speaker'].str.lower() == 'participant']
        else:
            df_part = df_trans
            
        # Join all words
        if 'value' in df_part.columns:
            text = " ".join(df_part['value'].dropna().astype(str).tolist())
        else:
            text = ""
            
    except Exception as e:
        print(f"Error reading {pid}: {e}")
        text = ""
        
    texts.append(text)
    labels.append(lbl)
    splits.append(split)

df_data = pd.DataFrame({
    'text': texts,
    'label': labels,
    'split': splits
})

print(f"Extracted shape: {df_data.shape}")

# %% [markdown]
# ## Modeling (5-Fold CV across ALL 189)

# %%
X_all = df_data['text'].values
y_all = df_data['label'].values

models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced', max_iter=1000, C=1.0),
    'SVM': SVC(kernel='linear', probability=True, random_state=RANDOM_SEED, class_weight='balanced', C=0.5),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_estimators=300),
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', scale_pos_weight=2.0)
}

results = []

for name, model in models.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    y_true_all = []
    y_prob_all = []
    
    for train_index, test_index in skf.split(X_all, y_all):
        X_train_text, X_test_text = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        
        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
        X_train = vectorizer.fit_transform(X_train_text).toarray()
        X_test = vectorizer.transform(X_test_text).toarray()
        
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
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v45_cv_results.csv"), index=False)

print("\n" + "="*50)
print("FINAL RESULTS (v45 NLP) - 189 Participants (5-Fold CV)")
print("="*50)
print(df_res.to_string(index=False))

# %% [markdown]
# ## Modeling (Train/Dev on Official Splits)
# Untuk menyamai evaluasi SOTA jurnal, kita latih di Train, validasi di Dev.

# %%
train_mask = (df_data['split'] == 'train')
dev_mask = (df_data['split'] == 'dev')
test_mask = (df_data['split'] == 'test')

X_train_text = df_data.loc[train_mask, 'text'].values
y_train = df_data.loc[train_mask, 'label'].values
X_dev_text = df_data.loc[dev_mask, 'text'].values
y_dev = df_data.loc[dev_mask, 'label'].values
X_test_text = df_data.loc[test_mask, 'text'].values
y_test = df_data.loc[test_mask, 'label'].values

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_dev = vectorizer.transform(X_dev_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

svm = SVC(kernel='linear', probability=True, random_state=RANDOM_SEED, class_weight='balanced', C=0.5)
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
print("OFFICIAL TEST SPLIT EVALUATION (SVM TF-IDF)")
print("="*50)
print(f"Dev Macro F1 : {best_f1:.4f} (Thr: {best_thr:.2f})")
print(f"Test Macro F1: {test_f1:.4f}")

df_test_res = pd.DataFrame([{'Split': 'Test', 'Model': 'SVM (TF-IDF)', 'Macro_F1': test_f1}])
df_test_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v45_test_results.csv"), index=False)
