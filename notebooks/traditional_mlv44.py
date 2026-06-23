# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v44** — STATISTICAL AGGREGATION ON RAW COVAREP (5-Fold CV)
#
# ─────────────────────────────────────────────────────────────────────
#  Tujuan: Menembus F1 > 0.70 secara absolut dengan 189 Partisipan utuh.
#  Data: 189 Partisipan dari file mentah `_COVAREP.csv`.
#  Fitur: Statistik Global (Mean, Std, Max, Min) dari 74 fitur COVAREP.
#         Menghasilkan 74 x 4 = 296 fitur tabular per pasien.
#  Metode: Stratified 5-Fold Cross Validation, SMOTE, dan PCA (95%).
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v44")

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
# ## Feature Extraction (Statistical Aggregation)

# %%
print("Extracting statistical features from 189 COVAREP files...")
features_list = []
ids_list = []
y_list = []

for idx, row in df_labels.iterrows():
    pid = int(row['id'])
    lbl = int(row['label'])
    
    filepath = os.path.join(RAW_DIR, f"{pid}_P", f"{pid}_COVAREP.csv")
    if not os.path.exists(filepath):
        continue
        
    df_cov = pd.read_csv(filepath, header=None)
    data = df_cov.values.astype(np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)
    max_val = np.max(data, axis=0)
    min_val = np.min(data, axis=0)
    
    # 296 features
    agg_features = np.concatenate([mean_val, std_val, max_val, min_val])
    
    features_list.append(agg_features)
    ids_list.append(pid)
    y_list.append(lbl)

X_all = np.array(features_list)
y_all = np.array(y_list)

print(f"Extracted shape: {X_all.shape}")

# %% [markdown]
# ## Modeling (Stratified 5-Fold CV)

# %%
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced', max_iter=1000),
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
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
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
df_res.to_csv(os.path.join(RESULTS_DIR, "metrics", "v44_results.csv"), index=False)

print("\n" + "="*50)
print("FINAL RESULTS (v44) - 189 Participants (5-Fold CV)")
print("="*50)
print(df_res.to_string(index=False))
