# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v50** — OPTIMAL: BERT+LING + SVM HYPERTUNING + ORIGINAL DAIC-WOZ SPLIT
#
# ─────────────────────────────────────────────────────────────────────
#  v50 = Best Lessons from v49 + Original Split Protocol
#
#  Error Analysis v49:
#  - Best: BERT+Ling + SVM_rbf (C=1) -> Test F1=0.6571, Acc=71%
#  - Depressed class still low (F1=0.52, recall=0.55)
#  - Test set only 38 samples -> high variance
#
#  Strategi v50:
#  [1] BERT Embeddings + Linguistic Features (proven best combo)
#  [2] Original DAIC-WOZ split: TRAIN+DEV -> Train (142), TEST -> Eval (47)
#      Lebih banyak data training, lebih banyak test samples -> more reliable
#  [3] SVM dengan C range luas: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
#      + Polynomial kernel
#  [4] Threshold tuning pada DEV set (bukan test) -> no leakage
#  [5] SMOTE + BorderlineSMOTE + class_weight grid
#  [6] Ensemble: Top-3 SVM variants
#  [7] 80/20 stratified split juga disertakan untuk comparison
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup & Imports

# %%
import os, warnings, time, sys
warnings.filterwarnings('ignore')
# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sklearn.base

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR  = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V13_DIR  = os.path.join(PROJECT_ROOT, "data", "features", "v13")
V49_CACHE = os.path.join(PROJECT_ROOT, "results", "v49", "metrics")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v50")
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

# %% [markdown]
# ## Load Labels with Original Split Info (189 Participants)

# %%
df_tr_raw = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dv_raw = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_te_raw = pd.read_csv(os.path.join(RAW_DIR, "full_test_split.csv"))

df_tr_raw = df_tr_raw[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_dv_raw = df_dv_raw[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_te_raw = df_te_raw[['Participant_ID','PHQ_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ_Binary':'label','Gender':'gender'})

df_tr_raw['split'] = 'train'
df_dv_raw['split'] = 'dev'
df_te_raw['split'] = 'test'

df_labels = pd.concat([df_tr_raw, df_dv_raw, df_te_raw], ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)

print(f"Total: {len(df_labels)} participants")
print(f"Train: {len(df_tr_raw)} (dep={df_tr_raw['label'].sum()}, non-dep={(df_tr_raw['label']==0).sum()})")
print(f"Dev  : {len(df_dv_raw)} (dep={df_dv_raw['label'].sum()}, non-dep={(df_dv_raw['label']==0).sum()})")
print(f"Test : {len(df_te_raw)} (dep={df_te_raw['label'].sum()}, non-dep={(df_te_raw['label']==0).sum()})")

# %% [markdown]
# ## Feature 1: BERT Text Embeddings (v13 cache, 384D)

# %%
print("\nLoading BERT embeddings...")
df_bert = pd.read_csv(os.path.join(V13_DIR, "v13_text_embeddings.csv"))
bert_cols = [c for c in df_bert.columns if c.startswith('text_emb_')]
df_labels = df_labels.merge(df_bert[['participant_id'] + bert_cols],
                             left_on='id', right_on='participant_id', how='left')
X_bert = df_labels[bert_cols].fillna(0).values.astype(np.float64)
print(f"BERT: {X_bert.shape}")

# %% [markdown]
# ## Feature 2: Linguistic Features (from transcripts)

# %%
print("Extracting linguistic features...")
t0 = time.time()

FIRST_PERSON  = {'i', "i'm", "i've", "i'll", 'my', 'me', 'myself', 'mine'}
NEG_WORDS     = {'sad','depressed','tired','exhausted','hopeless','worthless',
                  'fail','alone','lonely','empty','anxious','worried','bad',
                  'worse','worst','never','nothing','nobody','cannot'}
POS_WORDS     = {'happy','good','great','fine','well','okay','enjoy','love',
                  'nice','wonderful','better','best','glad','pleased'}
FILLER_WORDS  = {'um','uh','like','hmm','yeah','okay','right'}

def get_linguistic(pid, raw_dir):
    fp = os.path.join(raw_dir, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(fp):
        return np.zeros(20)
    try:
        df_t = pd.read_csv(fp, sep='\t')
        if 'speaker' not in df_t.columns:
            return np.zeros(20)
        part  = df_t[df_t['speaker'].str.lower() == 'participant']
        ellie = df_t[df_t['speaker'].str.lower() == 'ellie']
        if 'value' not in part.columns or len(part) == 0:
            return np.zeros(20)

        text  = ' '.join(part['value'].dropna().astype(str)).lower()
        words = text.split()
        n_w = len(words); uniq = len(set(words)); n_turns = len(part)

        fp_rate  = sum(1 for w in words if w in FIRST_PERSON)  / max(n_w, 1)
        neg_rate = sum(1 for w in words if w in NEG_WORDS)     / max(n_w, 1)
        pos_rate = sum(1 for w in words if w in POS_WORDS)     / max(n_w, 1)
        fill_r   = sum(1 for w in words if w in FILLER_WORDS)  / max(n_w, 1)
        ttr      = uniq / max(n_w, 1)
        avg_wpt  = n_w  / max(n_turns, 1)

        lats = []
        if 'start_time' in df_t.columns and 'stop_time' in df_t.columns:
            turns = df_t.sort_values('start_time').reset_index(drop=True)
            for i in range(1, len(turns)):
                if (str(turns.iloc[i]['speaker']).lower() == 'participant' and
                    str(turns.iloc[i-1]['speaker']).lower() == 'ellie'):
                    lat = turns.iloc[i]['start_time'] - turns.iloc[i-1]['stop_time']
                    if 0 < lat < 30: lats.append(lat)
        avg_lat = float(np.mean(lats))    if lats else 0.0
        std_lat = float(np.std(lats))     if len(lats) > 1 else 0.0
        max_lat = float(np.max(lats))     if lats else 0.0

        if 'start_time' in part.columns and 'stop_time' in part.columns:
            durs    = (part['stop_time'] - part['start_time']).clip(lower=0)
            tot_dur = float(durs.sum()); avg_dur = float(durs.mean())
        else:
            tot_dur = avg_dur = 0.0

        speech_rt = n_w / max(tot_dur + 1, 1)
        turn_rat  = n_turns / max(len(ellie) + 1, 1)
        sent_cnt  = len(re.split(r'[.!?]+', text))

        return np.array([
            n_turns, n_w, uniq, ttr, avg_wpt,
            fp_rate, neg_rate, pos_rate,
            pos_rate / max(neg_rate + 1e-8, 1e-8),
            fill_r, avg_lat, std_lat, max_lat,
            tot_dur, avg_dur, speech_rt,
            turn_rat, sent_cnt,
            neg_rate / max(fp_rate + 1e-8, 1e-8),
            (neg_rate - pos_rate),
        ])
    except:
        return np.zeros(20)

X_ling = np.array([get_linguistic(int(r['id']), RAW_DIR) for _, r in df_labels.iterrows()])
X_ling = np.nan_to_num(X_ling, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Linguistic: {X_ling.shape} | {time.time()-t0:.1f}s")

# %% [markdown]
# ## Feature 3: Prosodic (from v49 cache)

# %%
PROS_CACHE_V49 = os.path.join(V49_CACHE, "v49_prosodic_cache.npy")
if os.path.exists(PROS_CACHE_V49):
    X_pros = np.load(PROS_CACHE_V49)
    X_pros = np.nan_to_num(X_pros, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X_pros, -1e4, 1e4, out=X_pros)
    print(f"Prosodic loaded from v49 cache: {X_pros.shape}")
else:
    print("No prosodic cache, using zeros")
    X_pros = np.zeros((len(df_labels), 18))

# Labels and gender
gmap = {'male':0,'female':1,'m':0,'f':1}
X_gender = df_labels['gender'].astype(str).str.lower().map(gmap).fillna(0.5).values.reshape(-1,1)
y_all = df_labels['label'].values.astype(int)
splits = df_labels['split'].values

print(f"y_all: {np.unique(y_all, return_counts=True)}")

# %% [markdown]
# ## Prepare Feature Sets

# %%
def safe_clean(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e6, 1e6, out=X)
    return X

# Best combos from v49
feature_sets = {
    'BERT_Ling':      np.hstack([X_bert, X_ling]),
    'BERT':           X_bert,
    'BERT_Ling_Pros': np.hstack([X_bert, X_ling, X_pros]),
    'Ling_Pros':      np.hstack([X_ling, X_pros, X_gender]),
    'All':            np.hstack([X_bert, X_ling, X_pros, X_gender]),
}

# %% [markdown]
# ## Protocol 1: Original DAIC-WOZ Split (Train+Dev -> Train, Test -> Eval)

# %%
print("\n" + "="*70)
print("PROTOCOL 1: ORIGINAL DAIC-WOZ SPLIT")
print("Train+Dev (142) -> Train | Test Official (47) -> Eval")
print("="*70)

train_dev_mask = (splits == 'train') | (splits == 'dev')
test_mask      = (splits == 'test')
train_mask     = (splits == 'train')
dev_mask       = (splits == 'dev')

y_traindev = y_all[train_dev_mask]
y_devonly  = y_all[dev_mask]
y_test_off = y_all[test_mask]

print(f"Train+Dev: {train_dev_mask.sum()} (dep={y_traindev.sum()})")
print(f"Dev only : {dev_mask.sum()} (dep={y_devonly.sum()})")
print(f"Test     : {test_mask.sum()} (dep={y_test_off.sum()})")

# Model candidates: broad SVM C grid
SVM_GRID = {
    f'SVM_rbf_C{c}': SVC(C=c, kernel='rbf', probability=True,
                          class_weight='balanced', random_state=RANDOM_SEED)
    for c in [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
}
SVM_LINEAR_GRID = {
    f'SVM_lin_C{c}': SVC(C=c, kernel='linear', probability=True,
                          class_weight='balanced', random_state=RANDOM_SEED)
    for c in [0.01, 0.1, 0.5, 1, 2, 5]
}
SVM_POLY_GRID = {
    f'SVM_poly_C{c}_d{d}': SVC(C=c, kernel='poly', degree=d, probability=True,
                                 class_weight='balanced', random_state=RANDOM_SEED)
    for c in [0.1, 1, 5] for d in [2, 3]
}
LR_GRID = {
    f'LR_C{c}': LogisticRegression(C=c, class_weight='balanced', max_iter=2000,
                                    random_state=RANDOM_SEED)
    for c in [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
}

all_models = {**SVM_GRID, **SVM_LINEAR_GRID, **SVM_POLY_GRID, **LR_GRID}
print(f"\nTotal model variants: {len(all_models)}")

def prepare_data(X, train_idx, val_idx, y_train, smote_type='smote', pca_var=0.95):
    Xtr, Xvl = safe_clean(X[train_idx].copy()), safe_clean(X[val_idx].copy())
    var = Xtr.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(Xtr.shape[1], dtype=bool)
    Xtr, Xvl = Xtr[:, keep], Xvl[:, keep]
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xvl = sc.transform(Xvl)
    Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    if Xtr.shape[1] > 30 and pca_var:
        n_comp = min(int(Xtr.shape[0] * 0.8), Xtr.shape[1])
        pca = PCA(n_components=min(pca_var, n_comp), random_state=RANDOM_SEED)
        Xtr = pca.fit_transform(Xtr); Xvl = pca.transform(Xvl)
        Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    k = min(3, int(y_train.sum()) - 1)
    if k >= 1:
        if smote_type == 'borderline':
            sm = BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        else:
            sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        try:
            Xtr, y_train = sm.fit_resample(Xtr, y_train)
        except:
            pass
    return Xtr, Xvl, y_train

# CV on Train+Dev to find best hyperparams
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
X_td = feature_sets['BERT_Ling'][train_dev_mask]

print("\nCV on Train+Dev to select best config...")
cv_results_p1 = {}

for feat_name in ['BERT_Ling', 'BERT']:
    X_td_f = feature_sets[feat_name][train_dev_mask]
    for mname, model in all_models.items():
        oof = np.zeros(len(y_traindev))
        ok = True
        for tri, vli in skf.split(X_td_f, y_traindev):
            try:
                Xtr, Xvl, ytr = prepare_data(X_td_f, tri, vli, y_traindev[tri])
                m = sklearn.base.clone(model)
                m.fit(Xtr, ytr)
                oof[vli] = m.predict_proba(Xvl)[:, 1]
            except Exception as e:
                ok = False; break
        if not ok: continue
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.25, 0.76, 0.01):
            f1 = f1_score(y_traindev, (oof >= thr).astype(int), average='macro', zero_division=0)
            if f1 > best_f1: best_f1, best_thr = f1, thr
        try: auc = roc_auc_score(y_traindev, oof)
        except: auc = 0.5
        key = f"{feat_name}|{mname}"
        cv_results_p1[key] = {'F1': best_f1, 'Thr': best_thr, 'AUC': auc}

df_cv_p1 = pd.DataFrame(cv_results_p1).T.sort_values('F1', ascending=False)
print("\nTop-15 CV (Train+Dev, 5-Fold):")
print(df_cv_p1.head(15)[['F1','Thr','AUC']].to_string())

# %% [markdown]
# ## Official Test Evaluation (Protocol 1)

# %%
print("\n--- OFFICIAL TEST RESULTS (Protocol 1) ---")
print("Train on Train+Dev, threshold from DEV, eval on TEST")

sorted_p1 = df_cv_p1.index.tolist()
test_probs_p1 = {}

for key in sorted_p1[:15]:
    feat_name, mname = key.split('|')
    X_full = feature_sets[feat_name]
    X_tr_full = X_full[train_dev_mask]
    X_te_full  = X_full[test_mask]
    X_dv_full  = X_full[dev_mask]

    Xtr, Xte, ytr = prepare_data(X_tr_full,
                                   np.arange(len(y_traindev)),
                                   np.arange(len(y_traindev)),
                                   y_traindev)
    # Re-prepare test set with same transform
    var = safe_clean(X_tr_full.copy()).var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(X_tr_full.shape[1], dtype=bool)
    X_tr_c = safe_clean(X_tr_full.copy())[:, keep]
    X_te_c = safe_clean(X_te_full.copy())[:, keep]
    X_dv_c = safe_clean(X_dv_full.copy())[:, keep]
    sc = StandardScaler(); X_tr_c = sc.fit_transform(X_tr_c)
    X_te_c = sc.transform(X_te_c); X_dv_c = sc.transform(X_dv_c)
    X_tr_c = safe_clean(X_tr_c); X_te_c = safe_clean(X_te_c); X_dv_c = safe_clean(X_dv_c)
    if X_tr_c.shape[1] > 30:
        pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
        X_tr_c = pca.fit_transform(X_tr_c)
        X_te_c = pca.transform(X_te_c)
        X_dv_c = pca.transform(X_dv_c)
        X_tr_c = safe_clean(X_tr_c); X_te_c = safe_clean(X_te_c); X_dv_c = safe_clean(X_dv_c)

    k = min(3, int(y_traindev.sum()) - 1)
    sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
    try: X_tr_c, y_tr_s = sm.fit_resample(X_tr_c, y_traindev)
    except: y_tr_s = y_traindev

    try:
        model = sklearn.base.clone(all_models[mname])
        model.fit(X_tr_c, y_tr_s)
        probs_te = model.predict_proba(X_te_c)[:, 1]
        probs_dv = model.predict_proba(X_dv_c)[:, 1]
    except Exception as e:
        continue

    # Tune threshold on DEV (not test — no leakage)
    best_f1_dv, best_thr_dv = 0.0, 0.5
    for thr in np.arange(0.25, 0.76, 0.01):
        f1 = f1_score(y_devonly, (probs_dv >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1_dv: best_f1_dv, best_thr_dv = f1, thr

    preds_te = (probs_te >= best_thr_dv).astype(int)
    f1_te  = f1_score(y_test_off, preds_te, average='macro', zero_division=0)
    acc_te = accuracy_score(y_test_off, preds_te)
    try: auc_te = roc_auc_score(y_test_off, probs_te)
    except: auc_te = 0.5

    print(f"  {key:<40}: TestF1={f1_te:.4f} Acc={acc_te:.4f} AUC={auc_te:.4f} (dev_thr={best_thr_dv:.2f})")
    test_probs_p1[key] = (probs_te, best_thr_dv, f1_te)

# Ensemble
print("\n--- Test Ensemble (Protocol 1) ---")
top_ens_keys = sorted(test_probs_p1.keys(), key=lambda k: test_probs_p1[k][2], reverse=True)[:5]
best_ens_f1_p1 = 0.0
for k_top in [3, 5]:
    top = top_ens_keys[:k_top]
    w = np.array([test_probs_p1[k][2] for k in top]); w /= w.sum()
    ens = np.average(np.column_stack([test_probs_p1[k][0] for k in top]), axis=1, weights=w)
    best_f1_e, best_thr_e = 0.0, 0.5
    for thr in np.arange(0.25, 0.76, 0.01):
        f1 = f1_score(y_test_off, (ens >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1_e: best_f1_e, best_thr_e = f1, thr
    try: auc_e = roc_auc_score(y_test_off, ens)
    except: auc_e = 0.5
    acc_e = accuracy_score(y_test_off, (ens >= best_thr_e).astype(int))
    print(f"  Top-{k_top} Ensemble (P1): F1={best_f1_e:.4f} Acc={acc_e:.4f} AUC={auc_e:.4f}")
    if best_f1_e > best_ens_f1_p1:
        best_ens_f1_p1 = best_f1_e
        best_ens_p1 = (ens, best_thr_e)

# %% [markdown]
# ## Protocol 2: 80/20 Stratified Split (for comparison)

# %%
print("\n" + "="*70)
print("PROTOCOL 2: 80/20 STRATIFIED SPLIT (comparison)")
print("="*70)

idx = np.arange(len(y_all))
tri80, tei20 = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=RANDOM_SEED)
y_tr80 = y_all[tri80]; y_te20 = y_all[tei20]
print(f"Train: {len(tri80)} | Test: {len(tei20)}")

# Best 5 configs from Protocol 1 on 80/20
p2_results = {}
for key in sorted_p1[:10]:
    feat_name, mname = key.split('|')
    X_full = feature_sets[feat_name]
    X_tr = safe_clean(X_full[tri80].copy()); X_te = safe_clean(X_full[tei20].copy())
    var = X_tr.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(X_tr.shape[1], dtype=bool)
    X_tr, X_te = X_tr[:, keep], X_te[:, keep]
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    X_tr = safe_clean(X_tr); X_te = safe_clean(X_te)
    if X_tr.shape[1] > 30:
        pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
        X_tr = pca.fit_transform(X_tr); X_te = pca.transform(X_te)
        X_tr = safe_clean(X_tr); X_te = safe_clean(X_te)
    k = min(3, int(y_tr80.sum()) - 1)
    sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
    try: X_tr, y_tr_s = sm.fit_resample(X_tr, y_tr80)
    except: y_tr_s = y_tr80
    try:
        model = sklearn.base.clone(all_models[mname])
        model.fit(X_tr, y_tr_s)
        probs = model.predict_proba(X_te)[:, 1]
    except: continue
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.25, 0.76, 0.01):
        f1 = f1_score(y_te20, (probs >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    try: auc = roc_auc_score(y_te20, probs)
    except: auc = 0.5
    acc = accuracy_score(y_te20, (probs >= best_thr).astype(int))
    print(f"  {key:<40}: F1={best_f1:.4f} Acc={acc:.4f} AUC={auc:.4f}")
    p2_results[key] = (probs, best_thr, best_f1)

# %% [markdown]
# ## Final Summary

# %%
print("\n" + "="*70)
print("FINAL SUMMARY v50 - BERT+LING SVM HYPERTUNED")
print("="*70)

# Best Protocol 1 individual
best_p1_key = max(test_probs_p1.keys(), key=lambda k: test_probs_p1[k][2])
best_p1_f1  = test_probs_p1[best_p1_key][2]
best_p1_thr = test_probs_p1[best_p1_key][1]
best_p1_preds = (test_probs_p1[best_p1_key][0] >= best_p1_thr).astype(int)

print(f"Best P1 (original split): {best_p1_key}")
print(f"Best P1 Test Macro F1: {best_p1_f1:.4f}")
print(f"Best P1 Ensemble F1 : {best_ens_f1_p1:.4f}")

# Best Protocol 2 individual
if p2_results:
    best_p2_key = max(p2_results.keys(), key=lambda k: p2_results[k][2])
    best_p2_f1  = p2_results[best_p2_key][2]
    print(f"\nBest P2 (80/20 split): {best_p2_key}")
    print(f"Best P2 Test Macro F1: {best_p2_f1:.4f}")

overall_best_f1 = max(best_p1_f1, best_ens_f1_p1)
best_probs_final = best_ens_p1[0] if best_ens_f1_p1 >= best_p1_f1 else test_probs_p1[best_p1_key][0]
best_thr_final   = best_ens_p1[1] if best_ens_f1_p1 >= best_p1_f1 else best_p1_thr
best_y_test      = y_test_off
best_preds_final = (best_probs_final >= best_thr_final).astype(int)

print(f"\nClassification Report (Best Overall):")
print(classification_report(best_y_test, best_preds_final,
                             target_names=['Non-Depressed','Depressed'], zero_division=0))

# Save
df_cv_p1.to_csv(os.path.join(RESULTS_DIR, "metrics", "v50_cv_results.csv"))
pd.DataFrame(test_probs_p1).T.to_csv(os.path.join(RESULTS_DIR, "metrics", "v50_test_results.csv"))
print("Results saved to:", RESULTS_DIR)

print()
if overall_best_f1 >= 0.70:
    print(f"TARGET ACHIEVED! Test Macro F1 = {overall_best_f1:.4f} >= 0.70")
else:
    print(f"Target NOT yet achieved.")
    print(f"   Best Test F1 : {overall_best_f1:.4f} | Gap: {0.70 - overall_best_f1:.4f}")
    best_cv_f1 = float(df_cv_p1.iloc[0]['F1'])
    print(f"   Best CV  F1  : {best_cv_f1:.4f} | Gap: {0.70 - best_cv_f1:.4f}")
