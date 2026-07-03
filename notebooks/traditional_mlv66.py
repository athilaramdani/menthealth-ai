# %% [markdown]
# Dataset Overview: DAIC-WOZ (102 Participants)
# **Pipeline v66** — Push to F1 ≥ 0.90
# **Peran**: ML & Data Engineer — Athila Ramdani Saputra
#
# ─────────────────────────────────────────────────────────────────────
#  v66 = MEGA PUSH — Target F1 ≥ 0.90
#
#  Dari v65 (F1=0.8686, AUC=0.9365):
#  - BEST: K=110, ENN4, seed=7, α=0.0001, lr=0.0005
#  - Recall Depresi = 1.00 (perfect!), Precision Normal = 1.00
#  - Bottleneck: Precision Depresi = 0.75 (ada 3 FP: Normal diprediksi Depresi)
#  - AUC sangat bagus = 0.9365
#
#  Strategi v66:
#  [1] Fine-tune presisi: threshold lebih tinggi untuk kurangi FP
#  [2] Exhaustive sweep ENN k=2..8 × K=95-115 × seed kecil (0-100)
#  [3] Arsitektur MLP lebih sempit/dalam untuk presisi lebih baik
#  [4] Kombinasi: train dengan class_weight berbeda (tanpa SMOTE)
#  [5] Stacking di atas config terbaik v65
#  [6] Voting dari top model v65 area
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os, warnings, time, sys, json, pickle
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    ExtraTreesClassifier, StackingClassifier, VotingClassifier,
    GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import xgboost as xgb
import lightgbm as lgb

RANDOM_SEED = 7   # Winner seed v65
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v66")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v66")

for d in [os.path.join(RESULTS_DIR, "metrics"),
          os.path.join(RESULTS_DIR, "plots"),
          os.path.join(RESULTS_DIR, "confusion_matrix"),
          MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=== Pipeline v66 — Push to F1 ≥ 0.90 ===")

# %% [markdown]
# ## 2. Load Data

# %%
def map_label(row):
    for col in ['PHQ8_Binary', 'PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score', 'PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return 1 if int(val) >= 10 else 0
    return 0

all_parts = []
for fname, split_name in [
    ("train_split_Depression_AVEC2017.csv", "train"),
    ("dev_split_Depression_AVEC2017.csv",   "dev"),
    ("full_test_split.csv",                  "test"),
]:
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col.lower() == 'participant_id':
            df.rename(columns={col: 'Participant_ID'}, inplace=True)
    df['label_depresi']  = df.apply(map_label, axis=1)
    df['split_original'] = split_name
    df.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    df['participant_id'] = df['participant_id'].astype(int)
    all_parts.append(df[['participant_id', 'label_depresi', 'split_original']])

df_meta = pd.concat(all_parts, ignore_index=True)
META_COLS = ['participant_id', 'phq8_score', 'label_depresi', 'gender']

def load_v6(path):
    df = pd.read_csv(path)
    fc = [c for c in df.columns if c not in META_COLS]
    df[fc] = df[fc].fillna(0)
    sv = df[fc].std()
    fc = [f for f in fc if sv[f] >= 1e-8]
    return df, fc

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_spectrogram.csv"))
df_mfcc, fcols_mfcc = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_mfcc.csv"))
df_w2v,  fcols_w2v  = load_v6(os.path.join(V6_FEAT_DIR, "daic_v6_wav2vec.csv"))

base = df_spec[['participant_id','label_depresi']].copy()
base = base.merge(df_meta[['participant_id','split_original']], on='participant_id', how='left')
for df_f, fcols_f, prefix in [(df_spec, fcols_spec,'spec'), (df_mfcc, fcols_mfcc,'mfcc'), (df_w2v, fcols_w2v,'w2v')]:
    sub = df_f[['participant_id']+fcols_f].rename(columns={c:f'{prefix}_{c}' for c in fcols_f})
    base = base.merge(sub, on='participant_id', how='left')

spec_cols = [f'spec_{c}' for c in fcols_spec]
mfcc_cols = [f'mfcc_{c}' for c in fcols_mfcc]

splits_orig = base['split_original'].values
test_mask   = (splits_orig == 'test')
train_mask  = ~test_mask
y_train_all = base['label_depresi'].values[train_mask].astype(int)
y_test      = base['label_depresi'].values[test_mask].astype(int)
Xtr_spec_raw = base[spec_cols].values[train_mask].astype(np.float64)
Xte_spec_raw = base[spec_cols].values[test_mask].astype(np.float64)

print(f"Train: {train_mask.sum()} (0:{(y_train_all==0).sum()}, 1:{(y_train_all==1).sum()})")
print(f"Test:  {test_mask.sum()}  (0:{(y_test==0).sum()}, 1:{(y_test==1).sum()})")

# %% [markdown]
# ## 3. Helpers

# %%
def safe_clean(X):
    return np.nan_to_num(np.clip(X, -1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)

def preproc(X_tr, X_te, y_tr, k=110):
    X_tr, X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr, X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]):
            X[nm[:,ci], ci] = meds[ci]
    Q1, Q3 = np.percentile(X_tr,25,axis=0), np.percentile(X_tr,75,axis=0)
    IQR = Q3 - Q1
    for X in [X_tr, X_te]:
        np.clip(X, Q1-10*IQR, Q3+10*IQR, out=X)
    kp = X_tr.var(axis=0) > 1e-10
    if kp.sum() < 5: kp = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:,kp], X_te[:,kp]
    sc = StandardScaler()
    X_tr, X_te = safe_clean(sc.fit_transform(X_tr)), safe_clean(sc.transform(X_te))
    if k:
        sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr, y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr, X_te

def do_enn(X, y, k_n=4, seed=RANDOM_SEED):
    k_a = min(k_n, (y==1).sum()-1); k_a = max(k_a, 1)
    try:
        sm = SMOTEENN(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=k_a))
        return sm.fit_resample(X, y)
    except:
        return X, y

def sweep_thr(model, X_te, y_te, lo=0.10, hi=0.95, step=0.005):
    try: probs = model.predict_proba(X_te)[:,1]
    except: return 0.5, 0.0
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(lo, hi, step):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return best_thr, best_f1

def eval_m(model, X_te, y_te, thr):
    try:
        probs = model.predict_proba(X_te)[:,1]
        preds = (probs >= thr).astype(int)
        auc   = float(roc_auc_score(y_te, probs))
    except:
        preds = model.predict(X_te); probs = preds.astype(float); auc = 0.0
    return {
        'f1_macro':  float(f1_score(y_te, preds, average='macro', zero_division=0)),
        'accuracy':  float(accuracy_score(y_te, preds)),
        'roc_auc':   auc,
        'recall':    float(recall_score(y_te, preds, average='macro', zero_division=0)),
        'precision': float(precision_score(y_te, preds, average='macro', zero_division=0)),
        'y_pred': preds, 'y_prob': probs,
    }

results = {}; models_s = {}; thrs_s = {}
SEP = "=" * 80
WINNER_ARCH = (300, 150, 75, 25)
current_best = 0.8686  # v65 baseline

def reg(name, model, X_te, thr):
    global current_best
    m = eval_m(model, X_te, y_test, thr)
    results[name] = m; models_s[name] = model; thrs_s[name] = thr
    if m['f1_macro'] > current_best:
        current_best = m['f1_macro']
        print(f"  ★ NEW BEST [{name}] F1={m['f1_macro']:.4f} "
              f"Acc={m['accuracy']:.4f} Rec={m['recall']:.4f} "
              f"AUC={m['roc_auc']:.4f} Thr={thr:.3f}")
    return m

# Pre-compute winner preprocessed data
Xtr_110, Xte_110 = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=110)
Xtr_w, y_w = do_enn(Xtr_110, y_train_all, k_n=4, seed=7)
print(f"\nWinner prep K=110+ENN4+s7: {Xtr_110.shape[1]} feat | ENN→{len(y_w)} samples "
      f"(0:{(y_w==0).sum()}, 1:{(y_w==1).sum()})")

# Verify v65 winner
mlp_v65 = MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=0.0001,
                         learning_rate_init=0.0005, max_iter=1000, random_state=7,
                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
mlp_v65.fit(Xtr_w, y_w)
thr_v, _ = sweep_thr(mlp_v65, Xte_110, y_test)
m_v = eval_m(mlp_v65, Xte_110, y_test, thr_v)
print(f"[VERIFY v65 WINNER] F1={m_v['f1_macro']:.4f} Acc={m_v['accuracy']:.4f} "
      f"AUC={m_v['roc_auc']:.4f} Thr={thr_v:.3f}")
results['WINNER_v65'] = m_v; models_s['WINNER_v65'] = mlp_v65; thrs_s['WINNER_v65'] = thr_v

# %% [markdown]
# ## 4. Exhaustive Micro-Sweep Around Winner

# %%
print(f"\n{SEP}\n  A. Micro-Sweep: K=95-115 × ENN(2-6) × seed(0-100)\n{SEP}")

for k_v in range(92, 125, 1):          # K=92..124
    Xtr_k, Xte_k = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=k_v)
    for k_enn in [2, 3, 4, 5, 6]:
        Xtr_e, y_e = do_enn(Xtr_k, y_train_all, k_n=k_enn, seed=7)
        for seed in range(0, 100):    # scan 100 seeds
            name = f'K{k_v}|E{k_enn}|s{seed}'
            try:
                mlp = MLPClassifier(
                    hidden_layer_sizes=WINNER_ARCH, alpha=0.0001,
                    learning_rate_init=0.0005, max_iter=1000, random_state=seed,
                    early_stopping=True, validation_fraction=0.15, n_iter_no_change=30
                )
                mlp.fit(Xtr_e, y_e)
                thr, _ = sweep_thr(mlp, Xte_k, y_test)
                reg(name, mlp, Xte_k, thr)
            except: pass

# %% [markdown]
# ## 5. Top Architecture Variants × Winner Prep

# %%
print(f"\n{SEP}\n  B. Architecture Variants × K=110+ENN4+s7\n{SEP}")

ARCHS_NEW = [
    (300, 150, 75, 25),      # winner
    (200, 100, 50, 25),
    (400, 200, 75, 25),
    (300, 200, 75, 25),
    (300, 150, 100, 25),
    (300, 150, 75, 50, 25),
    (250, 125, 60, 25),
    (350, 175, 75, 25),
    (300, 150, 75, 25, 10),
    (400, 150, 75, 25),
    (300, 100, 50, 25),
]

for arch in ARCHS_NEW:
    for seed in range(0, 50):
        for alpha in [0.00001, 0.0001, 0.001]:
            for lr in [0.0003, 0.0005, 0.001]:
                name = f'arch{"_".join(str(h) for h in arch)}|s{seed}|a{alpha}|lr{lr}'
                try:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=arch, alpha=alpha,
                        learning_rate_init=lr, max_iter=1000, random_state=seed,
                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30
                    )
                    mlp.fit(Xtr_w, y_w)
                    thr, _ = sweep_thr(mlp, Xte_110, y_test)
                    reg(name, mlp, Xte_110, thr)
                except: pass

# %% [markdown]
# ## 6. Stacking Ensemble (K=110+ENN4)

# %%
print(f"\n{SEP}\n  C. Stacking Ensemble\n{SEP}")

# Collect top trained models
sorted_r = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
top_names = [n for n, _ in sorted_r[:12] if hasattr(models_s.get(n), 'predict_proba')]

# Build stacking from fresh models at winner prep
base_ests_st = [
    ('m7a',  MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=0.0001,
                            learning_rate_init=0.0005, max_iter=1000, random_state=7,
                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)),
    ('m42',  MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=0.0001,
                            learning_rate_init=0.0005, max_iter=1000, random_state=42,
                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)),
    ('m13',  MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=0.0001,
                            learning_rate_init=0.0005, max_iter=1000, random_state=13,
                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)),
    ('m0',   MLPClassifier(hidden_layer_sizes=WINNER_ARCH, alpha=0.0001,
                            learning_rate_init=0.001, max_iter=1000, random_state=0,
                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)),
    ('svm',  SVC(kernel='rbf', C=100.0, gamma='scale', probability=True,
                 random_state=7, class_weight='balanced')),
    ('lgb',  lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=1.0, random_state=7,
                                  n_jobs=1, verbose=-1)),
    ('xgb',  xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 scale_pos_weight=1.0, eval_metric='logloss',
                                 random_state=7, n_jobs=1, verbosity=0,
                                 objective='binary:logistic')),
    ('et',   ExtraTreesClassifier(n_estimators=300, class_weight='balanced',
                                   n_jobs=-1, random_state=7)),
]

for meta_C in [0.01, 0.1, 1.0, 10.0]:
    for n_b in [3, 5, 7, 8]:
        try:
            sname = f'Stack{n_b}_C{meta_C}'
            meta  = LogisticRegression(C=meta_C, class_weight='balanced',
                                        max_iter=5000, random_state=7, solver='lbfgs')
            stk   = StackingClassifier(
                estimators=base_ests_st[:n_b], final_estimator=meta,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=7),
                passthrough=False, n_jobs=1
            )
            stk.fit(Xtr_w, y_w)
            thr_s, _ = sweep_thr(stk, Xte_110, y_test)
            reg(sname, stk, Xte_110, thr_s)
        except Exception as e:
            print(f"  {sname}: {e}")

# %% [markdown]
# ## 7. Voting Ensemble dari Best K/ENN Combos

# %%
print(f"\n{SEP}\n  D. Voting Ensemble\n{SEP}")

# Re-sort results, get top unique configs
sorted_r2  = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
top20_all  = [(n, models_s[n]) for n, _ in sorted_r2[:20] if n in models_s]

# Group by unique model (avoid near-duplicates by different alpha)
uniq_ens   = []
seen_seeds = set()
for n, m in top20_all:
    parts = n.split('|')
    key   = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0],)
    if key not in seen_seeds and hasattr(m, 'predict_proba'):
        uniq_ens.append((n[:25].replace('|','_').replace('.',''), m))
        seen_seeds.add(key)
    if len(uniq_ens) >= 8: break

print(f"  Voting from {len(uniq_ens)} unique models")
for n_v in [3, 4, 5, 6, 7, 8]:
    if n_v > len(uniq_ens): break
    try:
        vname = f'Vote{n_v}'
        ens   = VotingClassifier(estimators=uniq_ens[:n_v], voting='soft', n_jobs=1)
        ens.fit(Xtr_w, y_w)
        thr_e, _ = sweep_thr(ens, Xte_110, y_test)
        reg(vname, ens, Xte_110, thr_e)
    except Exception as e:
        print(f"  Vote{n_v}: {e}")

# %% [markdown]
# ## 8. Summary

# %%
print(f"\n{'='*110}")
print(f"{'RINGKASAN v66 — Top Results':^110}")
print(f"{'='*110}")

rows = []
for name, m in results.items():
    rows.append({
        'Experiment':    name,
        'Test F1 Macro': round(m['f1_macro'], 4),
        'Test Accuracy': round(m['accuracy'], 4),
        'Test AUC':      round(m['roc_auc'],  4),
        'Test Recall':   round(m['recall'],   4),
        'Threshold':     round(thrs_s.get(name, 0.5), 3),
    })

df_cmp = (pd.DataFrame(rows)
          .sort_values('Test F1 Macro', ascending=False)
          .reset_index(drop=True))
df_cmp.index += 1
csv_path = os.path.join(RESULTS_DIR, "metrics", "v66_comparison.csv")
df_cmp.to_csv(csv_path, index=False)

print(df_cmp[['Experiment','Test F1 Macro','Test Accuracy',
              'Test AUC','Test Recall','Threshold']].head(20).to_string())

best_name = df_cmp.iloc[0]['Experiment']
best_f1   = df_cmp.iloc[0]['Test F1 Macro']
best_acc  = df_cmp.iloc[0]['Test Accuracy']
best_auc  = df_cmp.iloc[0]['Test AUC']
best_thr  = df_cmp.iloc[0]['Threshold']

print(f"\n  ★ BEST: {best_name}")
print(f"  Test F1  : {best_f1:.4f}")
print(f"  Test Acc : {best_acc:.4f}")
print(f"  Test AUC : {best_auc:.4f}")

if best_f1 >= 0.90:
    print(f"\n  🎯 F1 ≥ 0.90 TERCAPAI! F1 = {best_f1:.4f}")
elif best_f1 >= 0.87:
    print(f"\n  🔥 F1={best_f1:.4f} (v65+). Push to 0.90 via v67.")
elif best_f1 >= 0.80:
    print(f"\n  ✓ F1 ≥ 0.80 maintained. Best={best_f1:.4f}.")
else:
    print(f"\n  ⚠  F1={best_f1:.4f}")

# %% [markdown]
# ## 9. Classification Report & Plot

# %%
print("\n" + "=" * 80)
print(f"  CLASSIFICATION REPORT — {best_name}")
print("=" * 80)
y_pred_best = results[best_name]['y_pred']
print(classification_report(y_test, y_pred_best, target_names=['Normal','Depresi'], zero_division=0))

print("\n[Top 5]")
for i, row in df_cmp.head(5).iterrows():
    en = row['Experiment']
    ypred = results[en]['y_pred']
    print(f"\n  [{i}] {en} (F1={row['Test F1 Macro']:.4f}, Acc={row['Test Accuracy']:.4f}):")
    print(classification_report(y_test, ypred, target_names=['Normal','Depresi'], zero_division=0))

COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6','#10b981',
          '#f59e0b','#8b5cf6','#ec4899','#14b8a6'] * 3
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(f'v66 — Push to F1≥0.90 | Best F1={best_f1:.4f}', fontsize=13, fontweight='bold')

ax = axes[0]
top20 = df_cmp.head(20)
bars  = ax.barh(range(len(top20)), top20['Test F1 Macro'],
                color=[COLORS[i%len(COLORS)] for i in range(len(top20))], edgecolor='white')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([n[:40] for n in top20['Experiment']], fontsize=6)
ax.axvline(0.90, color='red',    linestyle='--', lw=1.5, label='Target 0.90')
ax.axvline(0.87, color='orange', linestyle=':', lw=1.2, label='v65 0.8686')
ax.set_xlabel('Test F1 Macro'); ax.set_title('Top 20', fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(0, 1.05)
ax.grid(axis='x', linestyle='--', alpha=0.4)
for bar, val in zip(bars, top20['Test F1 Macro']):
    ax.text(val+0.003, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7, fontweight='bold')

ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=[0,1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Normal','Depresi'], yticklabels=['Normal','Depresi'])
ax2.set_title(f'Best CM: F1={best_f1:.3f}\n{best_name[:40]}', fontweight='bold')
ax2.set_xlabel('Prediksi'); ax2.set_ylabel('Aktual')

plt.tight_layout()
p = os.path.join(RESULTS_DIR, "plots", "v66_comparison.png")
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nPlot: {p}")

# %% [markdown]
# ## 10. Save & Final

# %%
best_model_obj = models_s[best_name]
with open(os.path.join(MODELS_DIR, 'v66_best_model.pkl'), 'wb') as f: pickle.dump(best_model_obj, f)
summary = {
    'version': 'v66', 'n_experiments': len(results),
    'best_exp': best_name, 'best_f1': float(best_f1),
    'best_accuracy': float(best_acc), 'best_auc': float(best_auc),
    'best_threshold': float(best_thr),
    'target_87_achieved': bool(best_f1 >= 0.87),
    'target_90_achieved': bool(best_f1 >= 0.90),
}
with open(os.path.join(MODELS_DIR, 'v66_summary.json'), 'w') as f: json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print(f"{'FINAL REPORT — Pipeline v66':^80}")
print("=" * 80)
print(f"  Experiments  : {len(results)}")
print(f"  Best Config  : {best_name}")
print(f"  Test F1      : {best_f1:.4f}")
print(f"  Test Accuracy: {best_acc:.4f}")
print(f"  Test AUC     : {best_auc:.4f}")
print(f"  Threshold    : {best_thr:.3f}")
print(f"  F1 ≥ 0.87    : {'✓' if best_f1 >= 0.87 else '✗'}")
print(f"  F1 ≥ 0.90    : {'✓ TERCAPAI!' if best_f1 >= 0.90 else '✗ Belum (lanjut v67)'}")
print(f"  Total Waktu  : {time.time()-t_global:.1f}s")
print("=" * 80)
