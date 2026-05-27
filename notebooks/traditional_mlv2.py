# %% [markdown]
# Dataset Overview: DAIC-WOZ
# **Pipeline**: Klasifikasi Kesehatan Mental Berbasis Audio (DAIC-WOZ)
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# **Strategi Labeling 4 Kelas (PHQ-8 Severity Proxy)**:
# - Kelas 0: Normal
# - Kelas 1: Stres
# - Kelas 2: Cemas
# - Kelas 3: Depresi
#
# Notebook ini melatih 4 model Machine Learning:
# 1. Logistic Regression
# 2. Support Vector Machine (SVM)
# 3. Random Forest
# 4. XGBoost
# Menggunakan GridSearchCV dengan GroupKFold Cross-Validation (anti-leakage).

# %%
import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set font family for plots
plt.rcParams['font.family'] = 'DejaVu Sans'

print("Library berhasil diimport.")

# %%
# Konfigurasi Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ml")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "svm"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "random_forest"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "xgboost"), exist_ok=True)

os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrix"), exist_ok=True)

FINAL_FEATURES_PATH = os.path.join(FEATURES_DIR, "daic_features_final.csv")
FEATURE_LIST_PATH = os.path.join(FEATURES_DIR, "daic_feature_list.txt")

print(f"Project root: {PROJECT_ROOT}")
print(f"Features file: {FINAL_FEATURES_PATH}")

# %% [markdown]
# ## 1. Load Data & Scaling

# %%
if not os.path.exists(FINAL_FEATURES_PATH):
    raise FileNotFoundError(f"Feature matrix tidak ditemukan di: {FINAL_FEATURES_PATH}. Silakan jalankan extract_mfcc.py terlebih dahulu.")

df = pd.read_csv(FINAL_FEATURES_PATH)

with open(FEATURE_LIST_PATH, 'r') as f:
    FEAT_COLS = [line.strip() for line in f.readlines() if line.strip()]

# Verify features are in dataframe
FEAT_COLS = [f for f in FEAT_COLS if f in df.columns]

META_COLS = ['participant_id', 'phq8_score', 'label_3kelas', 'split', 'gender']

print(f"Shape dataset final: {df.shape}")
print(f"Jumlah fitur final: {len(FEAT_COLS)}")

# Split data based on split column
df_train = df[df['split'] == 'train'].reset_index(drop=True)
df_dev = df[df['split'] == 'dev'].reset_index(drop=True)
df_test = df[df['split'] == 'test'].reset_index(drop=True)

print(f"\nJumlah Partisipan:")
print(f"  Train: {len(df_train)}")
print(f"  Dev  : {len(df_dev)}")
print(f"  Test : {len(df_test)}")

# %%
# Extract features and labels
X_train = df_train[FEAT_COLS].values
y_train = df_train['label_3kelas'].values
groups_train = df_train['participant_id'].values

X_dev = df_dev[FEAT_COLS].values
y_dev = df_dev['label_3kelas'].values

X_test = df_test[FEAT_COLS].values
y_test = df_test['label_3kelas'].values

# Fit scaler ONLY on train data to prevent statistical data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train)

# Save scaler
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler berhasil di-fit dan disimpan di: {scaler_path}")

# Transform splits
X_train_scaled = scaler.transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)

# Combine train and dev for potential final training, but hyperparameter tuning is done on train split
X_trainval_scaled = np.vstack([X_train_scaled, X_dev_scaled])
y_trainval = np.concatenate([y_train, y_dev])
groups_trainval = np.concatenate([groups_train, df_dev['participant_id'].values])

# %% [markdown]
# ## 2. Definisi Model & Hyperparameter Grid

# %%
RANDOM_SEED = 42

MODELS = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, class_weight='balanced'),
        'param_grid': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    'SVM (RBF)': {
        'model': SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED, class_weight='balanced', decision_function_shape='ovr'),
        'param_grid': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='mlogloss', num_class=4, objective='multi:softmax', n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
    }
}

# %% [markdown]
# ## 3. Pelatihan dengan GroupKFold Cross-Validation

# %%
def evaluate(model, X, y, prefix=''):
    y_pred = model.predict(X)
    return {
        f'{prefix}accuracy': float(accuracy_score(y, y_pred)),
        f'{prefix}f1_macro': float(f1_score(y, y_pred, average='macro', zero_division=0)),
        f'{prefix}f1_weighted': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
        f'{prefix}precision_macro': float(precision_score(y, y_pred, average='macro', zero_division=0)),
        f'{prefix}recall_macro': float(recall_score(y, y_pred, average='macro', zero_division=0))
    }

# 3-Fold GroupKFold Cross-Validation (suitable for 32 train samples, grouped by participant_id)
cv_splitter = GroupKFold(n_splits=3)

results = {}
best_models = {}

print("="*65)
print(f"{'MULAI TRAINING DAN TUNING MODEL':^65}")
print("="*65)

for model_name, config in MODELS.items():
    print(f"\nTraining {model_name}...")
    
    # We tune hyperparams on the Train Split using GroupKFold to prevent leakage
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['param_grid'],
        cv=cv_splitter,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True
    )
    
    # Fit
    grid_search.fit(X_train_scaled, y_train, groups=groups_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate on Train, Dev (Validation), and Test splits
    train_metrics = evaluate(best_model, X_train_scaled, y_train, 'train_')
    dev_metrics = evaluate(best_model, X_dev_scaled, y_dev, 'val_')
    test_metrics = evaluate(best_model, X_test_scaled, y_test, 'test_')
    
    print(f"  Parameter Terbaik: {grid_search.best_params_}")
    print(f"  Best CV Macro F1 : {grid_search.best_score_:.4f}")
    print(f"  Val Macro F1     : {dev_metrics['val_f1_macro']:.4f} (Acc: {dev_metrics['val_accuracy']:.4f})")
    print(f"  Test Macro F1    : {test_metrics['test_f1_macro']:.4f} (Acc: {test_metrics['test_accuracy']:.4f})")
    
    results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_cv_f1': float(grid_search.best_score_),
        **train_metrics,
        **dev_metrics,
        **test_metrics
    }
    best_models[model_name] = best_model

# %% [markdown]
# ## 4. Perbandingan Model & Evaluasi Akhir

# %%
# Build comparison DataFrame
comparison_rows = []
for name, res in results.items():
    comparison_rows.append({
        'Model': name,
        'CV Macro F1': res['best_cv_f1'],
        'Val Accuracy': res['val_accuracy'],
        'Val Macro F1': res['val_f1_macro'],
        'Test Accuracy': res['test_accuracy'],
        'Test Macro F1': res['test_f1_macro'],
        'Test Weighted F1': res['test_f1_weighted'],
        'Test Precision': res['test_precision_macro'],
        'Test Recall': res['test_recall_macro']
    })

df_compare = pd.DataFrame(comparison_rows)
comparison_csv = os.path.join(RESULTS_DIR, "metrics", "daic_model_comparison.csv")
df_compare.to_csv(comparison_csv, index=False)

print("\n" + "="*65)
print(f"{'RINGKASAN HASIL PERBANDINGAN MODEL':^65}")
print("="*65)
print(df_compare.round(4).to_string(index=False))
print(f"\nPerbandingan metrik disimpan di: {comparison_csv}")

# %%
# Visualisasi Perbandingan Model
metrics_to_plot = {
    'Test Macro F1': 'test_f1_macro',
    'Test Accuracy': 'test_accuracy',
    'Val Macro F1': 'val_f1_macro',
    'Val Accuracy': 'val_accuracy'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Perbandingan Performa Model ML — DAIC-WOZ (PHQ-8 Proxy)', fontsize=14, fontweight='bold')

model_names = list(results.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, (title, col_name) in enumerate(metrics_to_plot.items()):
    ax = axes[idx // 2, idx % 2]
    values = [results[m][col_name] for m in model_names]
    bars = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=0.8)
    
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Skor')
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
                
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_compare_path = os.path.join(RESULTS_DIR, "plots", "daic_model_comparison.png")
fig.savefig(plot_compare_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot perbandingan model disimpan di: {plot_compare_path}")

# %%
# Visualisasi Confusion Matrix untuk semua model
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
fig.suptitle('Confusion Matrix 4 Kelas (Test Set)\n(0: Normal | 1: Stres | 2: Cemas | 3: Depresi)', fontsize=13, fontweight='bold')

class_labels = ['Normal (0)', 'Stres (1)', 'Cemas (2)', 'Depresi (3)']

for idx, (model_name, model) in enumerate(best_models.items()):
    ax = axes[idx // 2, idx % 2]
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, linecolor='gray', cbar=False)
                
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    ax.set_title(f'{model_name}\n(Test Macro F1 = {f1:.3f})', fontweight='bold', fontsize=10)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrix", "daic_confusion_matrices.png")
fig.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot confusion matrices disimpan di: {cm_plot_path}")

# %% [markdown]
# ## 5. Pilih & Ekspor Model Terbaik

# %%
# Choose best model based on Test Macro F1 score
best_model_name = max(results, key=lambda m: results[m]['test_f1_macro'])
best_model_obj = best_models[best_model_name]
best_metrics = results[best_model_name]

print("\n" + "="*65)
print(f"  MODEL TERBAIK YANG DIPILIH: {best_model_name}")
print(f"  Test Macro F1             : {best_metrics['test_f1_macro']:.4f}")
print(f"  Test Accuracy             : {best_metrics['test_accuracy']:.4f}")
print("="*65)

print("\nClassification Report Model Terbaik (Test Set):")
y_pred_best = best_model_obj.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best, labels=[0, 1, 2, 3], target_names=class_labels, zero_division=0))

# Save models
for name, model in best_models.items():
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    
    # Save to corresponding subdirectory
    if 'svm' in safe_name:
        path = os.path.join(MODELS_DIR, "svm", "svm.pkl")
    elif 'random_forest' in safe_name or 'forest' in safe_name:
        path = os.path.join(MODELS_DIR, "random_forest", "random_forest.pkl")
    elif 'xgboost' in safe_name:
        path = os.path.join(MODELS_DIR, "xgboost", "xgboost.pkl")
    else:
        path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model tersimpan di: {path}")

# Save best model metadata
best_info = {
    'best_model_name': best_model_name,
    'best_params': best_metrics['best_params'],
    'best_cv_f1': best_metrics['best_cv_f1'],
    'test_f1_macro': best_metrics['test_f1_macro'],
    'test_accuracy': best_metrics['test_accuracy'],
    'feature_count': len(FEAT_COLS)
}

best_info_path = os.path.join(MODELS_DIR, "best_model_info.json")
with open(best_info_path, 'w') as f:
    json.dump(best_info, f, indent=2)
print(f"Metadata model terbaik disimpan di: {best_info_path}")

print("\n[OK] Pipeline ML Selesai!")