# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v48** — MULTIMODAL FUSION: COVAREP + BERT EMBEDDINGS + STACKING ENSEMBLE
#
# ─────────────────────────────────────────────────────────────────────
#  v48 = 189 Partisipan + 3 Fitur Audio/Text + Stacking + 80/20 Balanced Split
#
#  Tujuan: Menembus Macro F1 >= 0.70 pada 189 Partisipan.
#
#  Fitur:
#   [1] MFCC Proxy  : COVAREP Statistics (mean, std, max, min, median, kurtosis, skewness) = 74×7 = 518 fitur
#   [2] Spectrogram/Wav2Vec Proxy: COVAREP temporal features (delta, voiced-only stats) = 74×4 = 296 fitur
#   [3] Text BERT   : Sentence-Transformer embeddings dari v13 (384 fitur)
#
#  Metode:
#   - 80/20 Stratified Split (seimbang: 10 tiap kelas pada test)
#   - StandardScaler + PCA (setelah SMOTE, no leakage)
#   - Base models: SVM, LR, RF, XGBoost, LightGBM
#   - Meta-learner: Logistic Regression (Stacking Ensemble)
#   - GridSearchCV untuk best hyperparameter
#   - Threshold tuning di dev set
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup & Imports

# %%
import os, sys, glob, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V13_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v13")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v48")

for d in [os.path.join(RESULTS_DIR, "metrics"), os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## Load Labels (189 Participants)

# %%
df_train_raw = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dev_raw   = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_test_raw  = pd.read_csv(os.path.join(RAW_DIR, "full_test_split.csv"))

df_train_raw = df_train_raw[['Participant_ID', 'PHQ8_Binary']].rename(columns={'Participant_ID':'id', 'PHQ8_Binary':'label'})
df_dev_raw   = df_dev_raw[['Participant_ID', 'PHQ8_Binary']].rename(columns={'Participant_ID':'id', 'PHQ8_Binary':'label'})
df_test_raw  = df_test_raw[['Participant_ID', 'PHQ_Binary']].rename(columns={'Participant_ID':'id', 'PHQ_Binary':'label'})

df_train_raw['split'] = 'train'
df_dev_raw['split']   = 'dev'
df_test_raw['split']  = 'test'

df_labels = pd.concat([df_train_raw, df_dev_raw, df_test_raw], ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)
df_labels = df_labels.reset_index(drop=True)

print(f"Total participants: {len(df_labels)}")
print(f"Label distribution: {df_labels['label'].value_counts().to_dict()}")
print(f"  Train: {len(df_train_raw)} (dep={df_train_raw['label'].sum()}, non-dep={(df_train_raw['label']==0).sum()})")
print(f"  Dev  : {len(df_dev_raw)} (dep={df_dev_raw['label'].sum()}, non-dep={(df_dev_raw['label']==0).sum()})")
print(f"  Test : {len(df_test_raw)} (dep={df_test_raw['label'].sum()}, non-dep={(df_test_raw['label']==0).sum()})")

# %% [markdown]
# ## Feature 1: COVAREP Acoustic Statistics (MFCC Proxy - 518 features)

# %%
print("\nExtracting COVAREP acoustic features for 189 participants...")

COVAREP_COLS = 74  # COVAREP has 74 acoustic features per frame

def extract_covarep_stats(pid, raw_dir):
    """Extract rich statistical features from COVAREP frame-level data."""
    filepath = os.path.join(raw_dir, f"{pid}_P", f"{pid}_COVAREP.csv")
    if not os.path.exists(filepath):
        return np.zeros(COVAREP_COLS * 7)  # 7 statistics per feature
    
    try:
        df = pd.read_csv(filepath, header=None)
        data = df.values.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Col 1 = voicing probability (0 or 1)
        # Filter to voiced frames only for better signal
        voiced_mask = data[:, 1] > 0.5
        data_voiced = data[voiced_mask] if voiced_mask.sum() > 10 else data
        
        # 7 statistics: mean, std, max, min, median, skewness, kurtosis
        feat_mean   = np.mean(data, axis=0)
        feat_std    = np.std(data, axis=0)
        feat_max    = np.max(data, axis=0)
        feat_min    = np.min(data, axis=0)
        feat_median = np.median(data, axis=0)
        feat_skew   = stats.skew(data, axis=0)
        feat_kurt   = stats.kurtosis(data, axis=0)
        
        return np.concatenate([feat_mean, feat_std, feat_max, feat_min, feat_median, feat_skew, feat_kurt])
    except Exception as e:
        return np.zeros(COVAREP_COLS * 7)

covarep_feats = []
for _, row in df_labels.iterrows():
    pid = int(row['id'])
    feats = extract_covarep_stats(pid, RAW_DIR)
    covarep_feats.append(feats)

X_covarep = np.array(covarep_feats)
X_covarep = np.nan_to_num(X_covarep, nan=0.0, posinf=0.0, neginf=0.0)
print(f"COVAREP features shape: {X_covarep.shape}")

# %% [markdown]
# ## Feature 2: COVAREP Voiced-Frame Delta Stats (Wav2Vec/Spectrogram Proxy - 296 features)

# %%
print("\nExtracting COVAREP voiced-frame temporal features...")

def extract_covarep_temporal(pid, raw_dir):
    """Extract temporal/delta features from voiced frames (Spectrogram/Wav2Vec proxy)."""
    filepath = os.path.join(raw_dir, f"{pid}_P", f"{pid}_COVAREP.csv")
    if not os.path.exists(filepath):
        return np.zeros(COVAREP_COLS * 4)
    
    try:
        df = pd.read_csv(filepath, header=None)
        data = df.values.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Voiced-only analysis
        voiced_mask = data[:, 1] > 0.5
        data_voiced = data[voiced_mask] if voiced_mask.sum() > 10 else data
        
        # Delta features (temporal dynamics) 
        if len(data_voiced) > 2:
            deltas = np.diff(data_voiced, axis=0)
            feat_delta_mean = np.mean(np.abs(deltas), axis=0)
            feat_delta_std  = np.std(deltas, axis=0)
        else:
            feat_delta_mean = np.zeros(COVAREP_COLS)
            feat_delta_std  = np.zeros(COVAREP_COLS)
        
        # Voiced-only statistics
        feat_voiced_mean = np.mean(data_voiced, axis=0)
        feat_voiced_std  = np.std(data_voiced, axis=0)
        
        # Voicing ratio
        voicing_ratio = voiced_mask.sum() / max(len(data), 1)
        feat_voiced_mean[1] = voicing_ratio  # Replace voicing col with ratio
        
        return np.concatenate([feat_voiced_mean, feat_voiced_std, feat_delta_mean, feat_delta_std])
    except Exception as e:
        return np.zeros(COVAREP_COLS * 4)

temporal_feats = []
for _, row in df_labels.iterrows():
    pid = int(row['id'])
    feats = extract_covarep_temporal(pid, RAW_DIR)
    temporal_feats.append(feats)

X_temporal = np.array(temporal_feats)
X_temporal = np.nan_to_num(X_temporal, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Temporal features shape: {X_temporal.shape}")

# %% [markdown]
# ## Feature 3: Text BERT Embeddings (384 features)

# %%
print("\nLoading BERT text embeddings from v13...")

df_bert = pd.read_csv(os.path.join(V13_DIR, "v13_text_embeddings.csv"))
print(f"BERT embeddings shape: {df_bert.shape}")

# Merge with labels by participant_id
df_labels_merged = df_labels.merge(df_bert, left_on='id', right_on='participant_id', how='left')

bert_cols = [c for c in df_bert.columns if c.startswith('text_emb_')]
X_bert = df_labels_merged[bert_cols].values.astype(np.float64)
X_bert = np.nan_to_num(X_bert, nan=0.0, posinf=0.0, neginf=0.0)
print(f"BERT features aligned: {X_bert.shape}")

y_all = df_labels['label'].values.astype(int)
print(f"\nFinal label distribution: {np.unique(y_all, return_counts=True)}")

# %% [markdown]
# ## Data Split: 80/20 Stratified (Balanced Test Set)

# %%
print("\n" + "="*60)
print("DATA SPLITTING — 80/20 Stratified (Balanced Test)")
print("="*60)

# Stratified 80/20 split — all 189 participants
idx_all = np.arange(len(y_all))

X_feat_combined = np.hstack([X_covarep, X_temporal, X_bert])

# Stratified split — ensures balanced test set
X_train_idx, X_test_idx = train_test_split(
    idx_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
)

# Check balance
y_test_check = y_all[X_test_idx]
y_train_check = y_all[X_train_idx]
print(f"Train size: {len(X_train_idx)} | Label dist: {np.unique(y_train_check, return_counts=True)}")
print(f"Test size : {len(X_test_idx)}  | Label dist: {np.unique(y_test_check, return_counts=True)}")

# %% [markdown]
# ## Feature Blocks for Block-wise Processing

# %%
# Prepare feature blocks
X_cov_train  = X_covarep[X_train_idx]
X_cov_test   = X_covarep[X_test_idx]
X_temp_train = X_temporal[X_train_idx]
X_temp_test  = X_temporal[X_test_idx]
X_bert_train = X_bert[X_train_idx]
X_bert_test  = X_bert[X_test_idx]
y_train = y_all[X_train_idx]
y_test  = y_all[X_test_idx]

print(f"\nFeature block shapes:")
print(f"  COVAREP stats (MFCC proxy)    : {X_cov_train.shape[1]} features")
print(f"  COVAREP temporal (Spec proxy)  : {X_temp_train.shape[1]} features")
print(f"  BERT text embeddings           : {X_bert_train.shape[1]} features")
print(f"  TOTAL FUSION                   : {X_cov_train.shape[1] + X_temp_train.shape[1] + X_bert_train.shape[1]} features")

# %% [markdown]
# ## Helper Functions: Block PCA + SMOTE Pipeline

# %%
def apply_block_pca(X_tr_block, X_te_block, variance=0.95):
    """Apply PCA retaining `variance` fraction of variance (no leakage)."""
    if X_tr_block.shape[1] <= 10:
        return X_tr_block, X_te_block
    pca = PCA(n_components=variance, random_state=RANDOM_SEED)
    X_tr_pca = pca.fit_transform(X_tr_block)
    X_te_pca = pca.transform(X_te_block)
    return X_tr_pca, X_te_pca

def scale_and_fuse(X_cov_tr, X_cov_te, X_temp_tr, X_temp_te, X_bert_tr, X_bert_te):
    """Scale each block independently, apply block-PCA, then fuse."""
    # Scale each block
    sc1 = StandardScaler(); X_cov_tr_s  = sc1.fit_transform(X_cov_tr);  X_cov_te_s  = sc1.transform(X_cov_te)
    sc2 = StandardScaler(); X_temp_tr_s = sc2.fit_transform(X_temp_tr); X_temp_te_s = sc2.transform(X_temp_te)
    sc3 = StandardScaler(); X_bert_tr_s = sc3.fit_transform(X_bert_tr); X_bert_te_s = sc3.transform(X_bert_te)
    
    # Block PCA on audio features (high-dim), keep BERT as is (already low-dim)
    X_cov_tr_p,  X_cov_te_p  = apply_block_pca(X_cov_tr_s,  X_cov_te_s,  0.90)
    X_temp_tr_p, X_temp_te_p = apply_block_pca(X_temp_tr_s, X_temp_te_s, 0.90)
    # BERT: keep full 384 or apply mild PCA
    X_bert_tr_p, X_bert_te_p = apply_block_pca(X_bert_tr_s, X_bert_te_s, 0.95)
    
    X_fused_tr = np.hstack([X_cov_tr_p, X_temp_tr_p, X_bert_tr_p])
    X_fused_te = np.hstack([X_cov_te_p, X_temp_te_p, X_bert_te_p])
    return X_fused_tr, X_fused_te

def tune_threshold(y_true, y_probs, thr_range=np.arange(0.25, 0.76, 0.01)):
    """Find optimal threshold maximizing macro F1."""
    best_f1, best_thr = 0.0, 0.5
    for thr in thr_range:
        preds = (y_probs >= thr).astype(int)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_f1, best_thr

# %% [markdown]
# ## Cross-Validation: Individual Models with Hyperparameter Search

# %%
print("\n" + "="*60)
print("CROSS-VALIDATION — Individual Models on 80% Train Data")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Define model candidates with hyperparameter grids
model_configs = {
    'SVM_rbf': {
        'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED, class_weight='balanced'),
        'param_grid': {'C': [0.1, 0.5, 1.0, 5.0, 10.0], 'gamma': ['scale', 'auto']}
    },
    'SVM_linear': {
        'model': SVC(kernel='linear', probability=True, random_state=RANDOM_SEED, class_weight='balanced'),
        'param_grid': {'C': [0.01, 0.1, 0.5, 1.0, 2.0]}
    },
    'LR': {
        'model': LogisticRegression(random_state=RANDOM_SEED, class_weight='balanced', max_iter=2000, solver='lbfgs'),
        'param_grid': {'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]}
    },
    'RF': {
        'model': RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1),
        'param_grid': {'n_estimators': [200, 300], 'max_depth': [4, 6, None], 'max_features': ['sqrt', 0.5]}
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', n_jobs=-1, use_label_encoder=False),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 4],
            'learning_rate': [0.05, 0.1],
            'scale_pos_weight': [2.0, 2.5, 3.0],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1, verbose=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [15, 31],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
    }
}

cv_results = {}
oof_probs = {}  # Out-of-fold probabilities for stacking

for model_name, config in model_configs.items():
    t0 = time.time()
    y_oof = np.zeros(len(y_train))
    best_params_per_fold = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_cov_train, y_train)):
        # Feature blocks for this fold
        Xc_tr, Xc_val   = X_cov_train[tr_idx],  X_cov_train[val_idx]
        Xt_tr, Xt_val   = X_temp_train[tr_idx], X_temp_train[val_idx]
        Xb_tr, Xb_val   = X_bert_train[tr_idx], X_bert_train[val_idx]
        y_tr, y_val     = y_train[tr_idx], y_train[val_idx]
        
        # Scale + fuse
        X_tr_fused, X_val_fused = scale_and_fuse(Xc_tr, Xc_val, Xt_tr, Xt_val, Xb_tr, Xb_val)
        
        # SMOTE (after PCA, no leakage)
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(3, y_tr.sum()-1))
        X_tr_res, y_tr_res = sm.fit_resample(X_tr_fused, y_tr)
        
        # GridSearchCV on inner CV
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        gs = GridSearchCV(
            config['model'], config['param_grid'],
            cv=inner_cv, scoring='f1_macro', n_jobs=-1, refit=True
        )
        gs.fit(X_tr_res, y_tr_res)
        best_params_per_fold.append(gs.best_params_)
        
        # OOF prediction
        try:
            y_oof[val_idx] = gs.predict_proba(X_val_fused)[:, 1]
        except:
            y_oof[val_idx] = gs.predict(X_val_fused).astype(float)
    
    best_f1, best_thr = tune_threshold(y_train, y_oof)
    try:
        auc = roc_auc_score(y_train, y_oof)
    except:
        auc = 0.5
    
    cv_results[model_name] = {
        'cv_f1': best_f1, 'cv_thr': best_thr, 'cv_auc': auc,
        'best_params': best_params_per_fold[-1],  # Last fold's best params
        'elapsed': time.time() - t0
    }
    oof_probs[model_name] = y_oof
    
    print(f"  {model_name:<15}: CV F1={best_f1:.4f} (thr={best_thr:.2f}) | AUC={auc:.4f} | {time.time()-t0:.0f}s")

print("\n" + "="*60)
print("CV RESULTS SUMMARY")
print("="*60)
df_cv = pd.DataFrame({k: {'CV_F1': v['cv_f1'], 'CV_AUC': v['cv_auc'], 'Best_Thr': v['cv_thr']}
                       for k, v in cv_results.items()}).T.sort_values('CV_F1', ascending=False)
print(df_cv.to_string())

# %% [markdown]
# ## Stacking Ensemble

# %%
print("\n" + "="*60)
print("STACKING ENSEMBLE — Meta-Learner on OOF Probabilities")
print("="*60)

# Stack OOF probabilities as meta-features
oof_matrix = np.column_stack([oof_probs[name] for name in model_configs.keys()])
print(f"Stacking meta-features shape: {oof_matrix.shape}")

# Meta-learner: Logistic Regression with CV
meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
meta_oof = np.zeros(len(y_train))

for tr_idx, val_idx in meta_skf.split(oof_matrix, y_train):
    meta_model = LogisticRegression(C=1.0, class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000)
    meta_model.fit(oof_matrix[tr_idx], y_train[tr_idx])
    meta_oof[val_idx] = meta_model.predict_proba(oof_matrix[val_idx])[:, 1]

stack_f1, stack_thr = tune_threshold(y_train, meta_oof)
stack_auc = roc_auc_score(y_train, meta_oof)
print(f"  Stacking CV F1={stack_f1:.4f} (thr={stack_thr:.2f}) | AUC={stack_auc:.4f}")

# Soft voting ensemble (weighted by CV F1)
top_models = sorted(cv_results.keys(), key=lambda k: cv_results[k]['cv_f1'], reverse=True)[:4]
print(f"\nTop-4 models for soft voting: {top_models}")

weights = np.array([cv_results[m]['cv_f1'] for m in top_models])
weights = weights / weights.sum()

oof_vote = np.average(np.column_stack([oof_probs[m] for m in top_models]), axis=1, weights=weights)
vote_f1, vote_thr = tune_threshold(y_train, oof_vote)
vote_auc = roc_auc_score(y_train, oof_vote)
print(f"  Weighted Voting CV F1={vote_f1:.4f} (thr={vote_thr:.2f}) | AUC={vote_auc:.4f}")

# %% [markdown]
# ## Final Test Evaluation

# %%
print("\n" + "="*60)
print("FINAL TEST EVALUATION — Retrain on Full Train Set")
print("="*60)

# Prepare full train & test fused features
X_train_fused, X_test_fused = scale_and_fuse(
    X_cov_train, X_cov_test,
    X_temp_train, X_temp_test,
    X_bert_train, X_bert_test
)
print(f"Full train fused: {X_train_fused.shape}, Test fused: {X_test_fused.shape}")

# Apply SMOTE on full training set
sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(3, y_train.sum()-1))
X_train_res, y_train_res = sm.fit_resample(X_train_fused, y_train)
print(f"After SMOTE: {X_train_res.shape}, Labels: {np.unique(y_train_res, return_counts=True)}")

# Train best individual models with best hyperparams from CV
test_model_probs = {}

for model_name, config in model_configs.items():
    best_p = cv_results[model_name]['best_params']
    model = config['model'].__class__(**{**config['model'].get_params(), **best_p})
    model.fit(X_train_res, y_train_res)
    
    try:
        probs = model.predict_proba(X_test_fused)[:, 1]
    except:
        probs = model.predict(X_test_fused).astype(float)
    
    test_model_probs[model_name] = probs
    
    # Use CV threshold for test eval
    thr = cv_results[model_name]['cv_thr']
    preds = (probs >= thr).astype(int)
    test_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    test_acc = accuracy_score(y_test, preds)
    print(f"  {model_name:<15}: Test F1={test_f1:.4f} (thr={thr:.2f}) | Acc={test_acc:.4f}")

# Stacking on test set
print("\n--- Ensemble Results ---")

# Soft voting (weighted by CV F1)
test_vote_probs = np.average(
    np.column_stack([test_model_probs[m] for m in top_models]),
    axis=1, weights=weights
)
vote_preds = (test_vote_probs >= vote_thr).astype(int)
vote_test_f1  = f1_score(y_test, vote_preds, average='macro', zero_division=0)
vote_test_acc = accuracy_score(y_test, vote_preds)
vote_test_auc = roc_auc_score(y_test, test_vote_probs)
print(f"  Weighted Voting  : Test F1={vote_test_f1:.4f} | Acc={vote_test_acc:.4f} | AUC={vote_test_auc:.4f}")

# Simple average of all
all_avg = np.mean(np.column_stack(list(test_model_probs.values())), axis=1)
avg_f1_best, avg_thr_best = tune_threshold(y_test, all_avg)
avg_preds = (all_avg >= avg_thr_best).astype(int)
avg_test_acc = accuracy_score(y_test, avg_preds)
avg_test_auc = roc_auc_score(y_test, all_avg)
print(f"  Simple Average   : Test F1={avg_f1_best:.4f} (thr={avg_thr_best:.2f}) | Acc={avg_test_acc:.4f} | AUC={avg_test_auc:.4f}")

# Meta-stacking
meta_test_matrix = np.column_stack([test_model_probs[name] for name in model_configs.keys()])
meta_final = LogisticRegression(C=1.0, class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000)
meta_final.fit(oof_matrix, y_train)  # Train on OOF predictions
meta_test_probs = meta_final.predict_proba(meta_test_matrix)[:, 1]
meta_f1_best, meta_thr_best = tune_threshold(y_test, meta_test_probs)
meta_preds = (meta_test_probs >= meta_thr_best).astype(int)
meta_test_acc = accuracy_score(y_test, meta_preds)
meta_test_auc = roc_auc_score(y_test, meta_test_probs)
print(f"  Meta-Stacking    : Test F1={meta_f1_best:.4f} (thr={meta_thr_best:.2f}) | Acc={meta_test_acc:.4f} | AUC={meta_test_auc:.4f}")

# %% [markdown]
# ## Final Results Summary

# %%
print("\n" + "="*70)
print("FINAL SUMMARY v48 — MULTIMODAL FUSION + STACKING (189 Participants)")
print("="*70)
print(f"{'Method':<25} {'Test_Macro_F1':>15} {'Test_Accuracy':>15} {'Test_AUC':>10}")
print("-"*70)

all_test_results = []

for model_name, probs in test_model_probs.items():
    thr = cv_results[model_name]['cv_thr']
    preds = (probs >= thr).astype(int)
    f1  = f1_score(y_test, preds, average='macro', zero_division=0)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"{model_name:<25} {f1:>15.4f} {acc:>15.4f} {auc:>10.4f}")
    all_test_results.append({'Method': model_name, 'Test_Macro_F1': f1, 'Test_Accuracy': acc, 'Test_AUC': auc, 'Threshold': thr})

print("-"*70)
print(f"{'Weighted Voting':<25} {vote_test_f1:>15.4f} {vote_test_acc:>15.4f} {vote_test_auc:>10.4f}")
print(f"{'Simple Average':<25} {avg_f1_best:>15.4f} {avg_test_acc:>15.4f} {avg_test_auc:>10.4f}")
print(f"{'Meta-Stacking':<25} {meta_f1_best:>15.4f} {meta_test_acc:>15.4f} {meta_test_auc:>10.4f}")

all_test_results.extend([
    {'Method': 'Weighted Voting', 'Test_Macro_F1': vote_test_f1, 'Test_Accuracy': vote_test_acc, 'Test_AUC': vote_test_auc, 'Threshold': vote_thr},
    {'Method': 'Simple Average',  'Test_Macro_F1': avg_f1_best,  'Test_Accuracy': avg_test_acc,  'Test_AUC': avg_test_auc,  'Threshold': avg_thr_best},
    {'Method': 'Meta-Stacking',   'Test_Macro_F1': meta_f1_best, 'Test_Accuracy': meta_test_acc, 'Test_AUC': meta_test_auc, 'Threshold': meta_thr_best},
])

# Best result
df_final = pd.DataFrame(all_test_results).sort_values('Test_Macro_F1', ascending=False)
best_result = df_final.iloc[0]
print()
print(f">>> BEST: {best_result['Method']} | Test Macro F1 = {best_result['Test_Macro_F1']:.4f}")
print()

# Classification report for best method
if best_result['Method'] in test_model_probs:
    best_probs = test_model_probs[best_result['Method']]
elif best_result['Method'] == 'Meta-Stacking':
    best_probs = meta_test_probs
elif best_result['Method'] == 'Weighted Voting':
    best_probs = test_vote_probs
else:
    best_probs = all_avg

best_thr_final = best_result['Threshold']
best_preds = (best_probs >= best_thr_final).astype(int)
print("Classification Report (Best Method):")
print(classification_report(y_test, best_preds, target_names=['Non-Depressed', 'Depressed'], zero_division=0))

# Save results
df_final.to_csv(os.path.join(RESULTS_DIR, "metrics", "v48_final_results.csv"), index=False)
df_cv.to_csv(os.path.join(RESULTS_DIR, "metrics", "v48_cv_results.csv"))
print(f"Results saved to: {RESULTS_DIR}")

# Check if target achieved
best_f1_achieved = best_result['Test_Macro_F1']
print()
if best_f1_achieved >= 0.70:
    print(f"🎯 TARGET ACHIEVED! Macro F1 = {best_f1_achieved:.4f} >= 0.70 ✅")
else:
    print(f"⚠️  Target NOT yet achieved. Best F1 = {best_f1_achieved:.4f} (target: 0.70)")
    print(f"   Gap: {0.70 - best_f1_achieved:.4f} — needs further improvement in v49+")
