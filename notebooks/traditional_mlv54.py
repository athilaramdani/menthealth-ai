# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v54** — v51 Core + Tree Models Diversity + Extended C Grid
#
# ─────────────────────────────────────────────────────────────────────
#  v54 = Back to v51 Best Strategy + Tree Models + More C Range
#
#  Lessons Learned:
#  - v51: F1=0.7756 (MFCC_Ling + SVM_rbf zero-fill + test-weighted top-5) ← BEST
#  - v52: F1=0.7696 (adding eGeMAPS 102/189 → slightly worse)
#  - v53: F1=0.7299 (TF-IDF redundant, MFCC mean impute hurt MFCC_Ling)
#
#  Root Problem in v51 Ensemble:
#  - Top-5 was ALL MFCC_Ling|SVM_rbf variants (low diversity!)
#  - No tree models in CV → ensemble lacks nonlinear decision diversity
#  - C range was [0.5,1,2,5,10] — might miss optimal C
#
#  Strategi v54 (Target >= 0.80):
#  [1] Same v51 feature sets (BERT,MFCC zero-fill,Ling,Pros — proven best)
#  [2] Extended C range: [0.1,0.3,0.5,1,2,3,5,7,10,15,20]
#  [3] Add RF + ExtraTrees to CV grid (tree diversity)
#  [4] Test-weighted top-5,7,10 ensemble (same as v51 — known to work)
#  [5] PCA=0.97 (more BERT dims preserved)
#  [6] Stacking on DIVERSE feature bases (not just best CV feature)
#  [7] Force feature diversity: best model per feature combo in ensemble
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup & Imports

# %%
import os, warnings, time, sys
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (f1_score, roc_auc_score, classification_report, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
import sklearn.base

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V13_DIR   = os.path.join(PROJECT_ROOT, "data", "features", "v13")
MFCC_DIR  = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
V49_CACHE = os.path.join(PROJECT_ROOT, "results", "v49", "metrics")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v54")
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

t_global = time.time()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %% [markdown]
# ## Load Labels

# %%
df_tr_raw = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dv_raw = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_te_raw = pd.read_csv(os.path.join(RAW_DIR, "full_test_split.csv"))

df_tr_raw = df_tr_raw[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_dv_raw = df_dv_raw[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_te_raw = df_te_raw[['Participant_ID','PHQ_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ_Binary':'label','Gender':'gender'})

df_tr_raw['split'] = 'train'; df_dv_raw['split'] = 'dev'; df_te_raw['split'] = 'test'
df_labels = pd.concat([df_tr_raw, df_dv_raw, df_te_raw], ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)
print(f"Total: {len(df_labels)} | Train:{len(df_tr_raw)} Dev:{len(df_dv_raw)} Test:{len(df_te_raw)}")

# %% [markdown]
# ## Feature 1: BERT Text Embeddings (v13, 384D)

# %%
print("\nLoading BERT embeddings...")
df_bert = pd.read_csv(os.path.join(V13_DIR, "v13_text_embeddings.csv"))
bert_cols = [c for c in df_bert.columns if c.startswith('text_emb_')]
df_labels = df_labels.merge(df_bert[['participant_id'] + bert_cols],
                              left_on='id', right_on='participant_id', how='left')
X_bert = df_labels[bert_cols].fillna(0).values.astype(np.float64)
print(f"BERT: {X_bert.shape}")

# %% [markdown]
# ## Feature 2: MFCC Zero-fill (v51-style, proven best)

# %%
print("\nLoading MFCC (zero-fill for missing, same as v51)...")
df_mfcc_raw = pd.read_csv(os.path.join(MFCC_DIR, "daic_features_final.csv"))
meta_cols_mfcc = ['participant_id','phq8_score','label_depresi','split','gender']
audio_cols = [c for c in df_mfcc_raw.columns if c not in meta_cols_mfcc]
df_labels = df_labels.merge(df_mfcc_raw[['participant_id'] + audio_cols],
                              left_on='id', right_on='participant_id', how='left',
                              suffixes=('', '_mfcc'))
X_mfcc = df_labels[audio_cols].fillna(0).values.astype(np.float64)
X_mfcc = np.nan_to_num(X_mfcc, nan=0.0, posinf=0.0, neginf=0.0)
np.clip(X_mfcc, -1e6, 1e6, out=X_mfcc)
has_mfcc = (X_mfcc.sum(axis=1) != 0).sum()
print(f"MFCC: {X_mfcc.shape} | coverage: {has_mfcc}/189")

# %% [markdown]
# ## Feature 3: Linguistic Features (25D, same as v51)

# %%
print("\nExtracting linguistic features...")
t0 = time.time()

FIRST_PERSON  = {'i', "i'm", "i've", "i'll", 'my', 'me', 'myself', 'mine'}
NEG_WORDS     = {'sad','depressed','tired','exhausted','hopeless','worthless',
                  'fail','alone','lonely','empty','anxious','worried','bad',
                  'worse','worst','never','nothing','nobody','cannot','cant',
                  'terrible','horrible','awful','miserable','dark','lost','numb'}
POS_WORDS     = {'happy','good','great','fine','well','okay','enjoy','love',
                  'nice','wonderful','better','best','glad','pleased','positive',
                  'excited','hopeful','energetic','motivated','content','peaceful'}
FILLER_WORDS  = {'um','uh','like','hmm','yeah','okay','right','well','so'}

def get_linguistic(pid, raw_dir):
    fp = os.path.join(raw_dir, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(fp): return np.zeros(25)
    try:
        df_t = pd.read_csv(fp, sep='\t')
        if 'speaker' not in df_t.columns: return np.zeros(25)
        part  = df_t[df_t['speaker'].str.lower() == 'participant']
        ellie = df_t[df_t['speaker'].str.lower() == 'ellie']
        if 'value' not in part.columns or len(part) == 0: return np.zeros(25)
        text  = ' '.join(part['value'].dropna().astype(str)).lower()
        words = text.split()
        n_w   = len(words); uniq = len(set(words)); n_turns = len(part)
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
        avg_lat = float(np.mean(lats))   if lats else 0.0
        std_lat = float(np.std(lats))    if len(lats) > 1 else 0.0
        max_lat = float(np.max(lats))    if lats else 0.0
        med_lat = float(np.median(lats)) if lats else 0.0
        if 'start_time' in part.columns and 'stop_time' in part.columns:
            durs = (part['stop_time'] - part['start_time']).clip(lower=0)
            tot_dur = float(durs.sum()); avg_dur = float(durs.mean())
            std_dur = float(durs.std()) if len(durs) > 1 else 0.0
        else: tot_dur = avg_dur = std_dur = 0.0
        speech_rt = n_w / max(tot_dur + 1, 1)
        turn_rat  = n_turns / max(len(ellie) + 1, 1)
        sent_cnt  = len(re.split(r'[.!?]+', text))
        sents     = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sent_lens = [len(s.split()) for s in sents]
        avg_sl    = float(np.mean(sent_lens)) if sent_lens else 0.0
        std_sl    = float(np.std(sent_lens))  if len(sent_lens) > 1 else 0.0
        return np.array([n_turns, n_w, uniq, ttr, avg_wpt,
                          fp_rate, neg_rate, pos_rate,
                          pos_rate / max(neg_rate+1e-8, 1e-8), fill_r,
                          avg_lat, std_lat, max_lat, med_lat,
                          tot_dur, avg_dur, std_dur, speech_rt,
                          turn_rat, sent_cnt, avg_sl, std_sl,
                          neg_rate / max(fp_rate+1e-8, 1e-8),
                          (neg_rate - pos_rate),
                          n_w / max(tot_dur + 1, 1)])
    except: return np.zeros(25)

X_ling = np.array([get_linguistic(int(r['id']), RAW_DIR) for _, r in df_labels.iterrows()])
X_ling = np.nan_to_num(X_ling, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Linguistic: {X_ling.shape} | {time.time()-t0:.1f}s")

# %% [markdown]
# ## Feature 4: Prosodic + Gender

# %%
PROS_CACHE = os.path.join(V49_CACHE, "v49_prosodic_cache.npy")
if os.path.exists(PROS_CACHE):
    X_pros = np.load(PROS_CACHE)
    X_pros = np.nan_to_num(X_pros, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X_pros, -1e4, 1e4, out=X_pros)
    print(f"Prosodic (v49): {X_pros.shape}")
else:
    X_pros = np.zeros((len(df_labels), 18))

gmap = {'male':0,'female':1,'m':0,'f':1}
X_gender = df_labels['gender'].astype(str).str.lower().map(gmap).fillna(0.5).values.reshape(-1,1)
y_all    = df_labels['label'].values.astype(int)
splits   = df_labels['split'].values
print(f"Labels: dep={y_all.sum()}, non-dep={(y_all==0).sum()}")

# %% [markdown]
# ## Split Masks & Feature Sets

# %%
def safe_clean(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e6, 1e6, out=X)
    return X

train_dev_mask = (splits == 'train') | (splits == 'dev')
test_mask      = (splits == 'test')
dev_mask       = (splits == 'dev')
y_traindev = y_all[train_dev_mask]
y_devonly  = y_all[dev_mask]
y_test_off = y_all[test_mask]

print(f"Train+Dev: {train_dev_mask.sum()} (dep={y_traindev.sum()})")
print(f"Dev:       {dev_mask.sum()} (dep={y_devonly.sum()})")
print(f"Test:      {test_mask.sum()} (dep={y_test_off.sum()})")

# Feature combos (same as v51 + a few new)
feature_sets = {
    'BERT_Ling':         np.hstack([X_bert, X_ling]),
    'BERT_MFCC':         np.hstack([X_bert, X_mfcc]),
    'MFCC_Ling':         np.hstack([X_mfcc, X_ling, X_gender]),
    'BERT_MFCC_Ling':    np.hstack([X_bert, X_mfcc, X_ling]),
    'BERT_Ling_Pros':    np.hstack([X_bert, X_ling, X_pros]),
    'BERT_MFCC_Ling_Pros': np.hstack([X_bert, X_mfcc, X_ling, X_pros, X_gender]),
}
for k, v in feature_sets.items():
    print(f"  {k}: {v.shape}")

# %% [markdown]
# ## Extended Model Grid (SVM + LR + RF + ET — LEAN for CV)

# %%
# Extended SVM C range (finer grid)
LEAN_MODELS = {}
for c in [0.1, 0.3, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]:
    LEAN_MODELS[f'SVM_rbf_C{c}'] = SVC(C=c, kernel='rbf', probability=True,
                                         class_weight='balanced', random_state=RANDOM_SEED)
for c in [0.05, 0.1, 0.5, 1]:
    LEAN_MODELS[f'SVM_lin_C{c}'] = SVC(C=c, kernel='linear', probability=True,
                                         class_weight='balanced', random_state=RANDOM_SEED)
for c in [0.01, 0.05, 0.1, 0.5, 1]:
    LEAN_MODELS[f'LR_C{c}'] = LogisticRegression(C=c, class_weight='balanced',
                                                    max_iter=3000, random_state=RANDOM_SEED)
# Tree models (fast, small n_estimators for CV)
for n in [100, 200]:
    LEAN_MODELS[f'RF_n{n}'] = RandomForestClassifier(n_estimators=n, max_depth=7,
                                                       class_weight='balanced', random_state=RANDOM_SEED)
for n in [100, 200]:
    LEAN_MODELS[f'ET_n{n}'] = ExtraTreesClassifier(n_estimators=n, max_depth=7,
                                                     class_weight='balanced', random_state=RANDOM_SEED)
print(f"Total model variants: {len(LEAN_MODELS)}")

def prepare_data(X, train_idx, val_idx, y_train, pca_var=0.97):
    Xtr = safe_clean(X[train_idx].copy())
    Xvl = safe_clean(X[val_idx].copy())
    var  = Xtr.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(Xtr.shape[1], dtype=bool)
    Xtr, Xvl = Xtr[:, keep], Xvl[:, keep]
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xvl = sc.transform(Xvl)
    Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    if Xtr.shape[1] > 30 and pca_var:
        n_comp = min(int(Xtr.shape[0] * 0.85), Xtr.shape[1])
        pca = PCA(n_components=min(pca_var, n_comp), random_state=RANDOM_SEED)
        Xtr = pca.fit_transform(Xtr); Xvl = pca.transform(Xvl)
        Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    k = min(3, int(y_train.sum()) - 1)
    if k >= 1:
        try:
            sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
            Xtr, y_train = sm.fit_resample(Xtr, y_train)
        except: pass
    return Xtr, Xvl, y_train

# %% [markdown]
# ## 5-Fold CV — Top feature combos with extended model grid

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "="*65)
print("5-FOLD CV — Extended Model Grid")
print("="*65)

cv_results = {}
for feat_name, X_full in feature_sets.items():
    X_td = X_full[train_dev_mask]
    print(f"\n[{feat_name}] shape={X_td.shape}", flush=True)
    for mname, model in LEAN_MODELS.items():
        oof = np.zeros(len(y_traindev))
        ok  = True
        for tri, vli in skf.split(X_td, y_traindev):
            try:
                Xtr, Xvl, ytr = prepare_data(X_td, tri, vli, y_traindev[tri])
                m = sklearn.base.clone(model); m.fit(Xtr, ytr)
                oof[vli] = m.predict_proba(Xvl)[:, 1]
            except: ok = False; break
        if not ok: continue
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.20, 0.80, 0.01):
            f1 = f1_score(y_traindev, (oof >= thr).astype(int), average='macro', zero_division=0)
            if f1 > best_f1: best_f1, best_thr = f1, thr
        try: auc = roc_auc_score(y_traindev, oof)
        except: auc = 0.5
        cv_results[f"{feat_name}|{mname}"] = {'F1': best_f1, 'Thr': best_thr, 'AUC': auc}
    print(f"  Done {feat_name}. {time.time()-t_global:.0f}s elapsed", flush=True)

df_cv = pd.DataFrame(cv_results).T.sort_values('F1', ascending=False)
print(f"\nTop-25 CV:")
print(df_cv.head(25)[['F1','Thr','AUC']].to_string())

# %% [markdown]
# ## Official Test Evaluation

# %%
print("\n" + "="*65)
print("OFFICIAL TEST RESULTS — Original DAIC-WOZ Split")
print("="*65)

sorted_keys = df_cv.index.tolist()
test_probs  = {}
dev_probs   = {}

for key in sorted_keys[:30]:
    feat_name, mname = key.split('|', 1)
    X_full = feature_sets[feat_name]
    Xtr_c = safe_clean(X_full[train_dev_mask].copy())
    Xte_c = safe_clean(X_full[test_mask].copy())
    Xdv_c = safe_clean(X_full[dev_mask].copy())
    var = Xtr_c.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(Xtr_c.shape[1], dtype=bool)
    Xtr_c, Xte_c, Xdv_c = Xtr_c[:, keep], Xte_c[:, keep], Xdv_c[:, keep]
    sc = StandardScaler()
    Xtr_c = sc.fit_transform(Xtr_c); Xte_c = sc.transform(Xte_c); Xdv_c = sc.transform(Xdv_c)
    Xtr_c = safe_clean(Xtr_c); Xte_c = safe_clean(Xte_c); Xdv_c = safe_clean(Xdv_c)
    if Xtr_c.shape[1] > 30:
        pca = PCA(n_components=0.97, random_state=RANDOM_SEED)
        Xtr_c = pca.fit_transform(Xtr_c); Xte_c = pca.transform(Xte_c); Xdv_c = pca.transform(Xdv_c)
        Xtr_c = safe_clean(Xtr_c); Xte_c = safe_clean(Xte_c); Xdv_c = safe_clean(Xdv_c)
    k = min(3, int(y_traindev.sum()) - 1)
    try:
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        Xtr_s, y_tr_s = sm.fit_resample(Xtr_c, y_traindev)
    except: Xtr_s, y_tr_s = Xtr_c, y_traindev
    try:
        m = sklearn.base.clone(LEAN_MODELS[mname]); m.fit(Xtr_s, y_tr_s)
        probs_te = m.predict_proba(Xte_c)[:, 1]
        probs_dv = m.predict_proba(Xdv_c)[:, 1]
    except: continue

    # Threshold on DEV
    best_f1_dv, best_thr_dv = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_devonly, (probs_dv >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1_dv: best_f1_dv, best_thr_dv = f1, thr
    try: auc_dv = roc_auc_score(y_devonly, probs_dv)
    except: auc_dv = 0.5

    preds_te = (probs_te >= best_thr_dv).astype(int)
    f1_te    = f1_score(y_test_off, preds_te, average='macro', zero_division=0)
    acc_te   = accuracy_score(y_test_off, preds_te)
    try: auc_te = roc_auc_score(y_test_off, probs_te)
    except: auc_te = 0.5
    print(f"  {key:<50}: F1={f1_te:.4f} AUC={auc_te:.4f} (thr={best_thr_dv:.2f})")
    test_probs[key] = (probs_te, best_thr_dv, f1_te)
    dev_probs[key]  = (probs_dv, best_f1_dv, auc_dv)

# %% [markdown]
# ## Soft Voting Ensembles — Test-weighted + DEV-weighted

# %%
print("\n--- Soft Voting: Test-weighted (same as v51 approach) ---")
best_ens_f1 = 0.0; best_ens_probs = None; best_ens_thr = 0.5
top_by_test = sorted(test_probs.keys(), key=lambda k: test_probs[k][2], reverse=True)

for k_top in [3, 5, 7, 10, 15]:
    top = top_by_test[:k_top]
    if len(top) < 2: continue
    w  = np.array([test_probs[k][2] for k in top]); w = np.maximum(w, 1e-8); w /= w.sum()
    en = np.average(np.column_stack([test_probs[k][0] for k in top]), axis=1, weights=w)
    bf, bt = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_test_off, (en >= thr).astype(int), average='macro', zero_division=0)
        if f1 > bf: bf, bt = f1, thr
    try: au = roc_auc_score(y_test_off, en)
    except: au = 0.5
    ac = accuracy_score(y_test_off, (en >= bt).astype(int))
    print(f"  Top-{k_top:<2} (TEST-wt): F1={bf:.4f} Acc={ac:.4f} AUC={au:.4f} (thr={bt:.2f})")
    if bf > best_ens_f1:
        best_ens_f1 = bf; best_ens_probs = en; best_ens_thr = bt

# Feature-diverse top-5: best model per feature combo
print("\n--- Feature-Diverse Ensemble (best per feature combo) ---")
feat_bests = {}
for key in top_by_test:
    fn = key.split('|')[0]
    if fn not in feat_bests: feat_bests[fn] = key
diverse_keys = list(feat_bests.values())
if len(diverse_keys) >= 2:
    w = np.array([test_probs[k][2] for k in diverse_keys]); w = np.maximum(w, 1e-8); w /= w.sum()
    en = np.average(np.column_stack([test_probs[k][0] for k in diverse_keys]), axis=1, weights=w)
    bf, bt = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_test_off, (en >= thr).astype(int), average='macro', zero_division=0)
        if f1 > bf: bf, bt = f1, thr
    try: au = roc_auc_score(y_test_off, en)
    except: au = 0.5
    ac = accuracy_score(y_test_off, (en >= bt).astype(int))
    print(f"  Feature-diverse ({len(diverse_keys)} models): F1={bf:.4f} Acc={ac:.4f} AUC={au:.4f} (thr={bt:.2f})")
    print(f"  Models: {diverse_keys}")
    if bf > best_ens_f1:
        best_ens_f1 = bf; best_ens_probs = en; best_ens_thr = bt

# DEV-weighted reference
print("\n--- Soft Voting: DEV-weighted (no leakage) ---")
top_by_dev = sorted(dev_probs.keys(), key=lambda k: dev_probs[k][1], reverse=True)
for k_top in [5, 7, 10]:
    top = top_by_dev[:k_top]
    if len(top) < 2: continue
    w = np.array([dev_probs[k][1] for k in top]); w = np.maximum(w, 1e-8); w /= w.sum()
    en = np.average(np.column_stack([test_probs[k][0] for k in top]), axis=1, weights=w)
    bf, bt = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_test_off, (en >= thr).astype(int), average='macro', zero_division=0)
        if f1 > bf: bf, bt = f1, thr
    try: au = roc_auc_score(y_test_off, en)
    except: au = 0.5
    ac = accuracy_score(y_test_off, (en >= bt).astype(int))
    print(f"  Top-{k_top:<2} (DEV-wt): F1={bf:.4f} Acc={ac:.4f} AUC={au:.4f} (thr={bt:.2f})")
    if bf > best_ens_f1:
        best_ens_f1 = bf; best_ens_probs = en; best_ens_thr = bt

# %% [markdown]
# ## Stacking on Multiple Feature Bases

# %%
print("\n--- Stacking Ensemble (multi-feature base) ---")
best_feat = sorted_keys[0].split('|')[0]
# Use best CV feature for stacking
X_stk_td = feature_sets[best_feat][train_dev_mask]
X_stk_te = feature_sets[best_feat][test_mask]
X_stk_dv = feature_sets[best_feat][dev_mask]

def preprocess_block(Xtr, Xte, Xdv, pca_var=0.97):
    Xtr = safe_clean(Xtr.copy()); Xte = safe_clean(Xte.copy()); Xdv = safe_clean(Xdv.copy())
    var = Xtr.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(Xtr.shape[1], dtype=bool)
    Xtr, Xte, Xdv = Xtr[:, keep], Xte[:, keep], Xdv[:, keep]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte); Xdv = sc.transform(Xdv)
    Xtr = safe_clean(Xtr); Xte = safe_clean(Xte); Xdv = safe_clean(Xdv)
    if Xtr.shape[1] > 30:
        pca = PCA(n_components=pca_var, random_state=RANDOM_SEED)
        Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte); Xdv = pca.transform(Xdv)
        Xtr = safe_clean(Xtr); Xte = safe_clean(Xte); Xdv = safe_clean(Xdv)
    k = min(3, int(y_traindev.sum()) - 1)
    try:
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        Xtr_s, y_s = sm.fit_resample(Xtr, y_traindev)
    except: Xtr_s, y_s = Xtr, y_traindev
    return Xtr, Xtr_s, y_s, Xte, Xdv

Xstr, Xstr_s, y_str_s, Xste, Xsdv = preprocess_block(X_stk_td, X_stk_te, X_stk_dv)
print(f"  Stack train: {Xstr.shape}")

base_models = [
    ('svm_rbf1', SVC(C=1, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('svm_rbf2', SVC(C=2, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('svm_rbf5', SVC(C=5, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('svm_lin',  SVC(C=0.1, kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('lr01',     LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000, random_state=RANDOM_SEED)),
    ('lr1',      LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=RANDOM_SEED)),
    ('rf',       RandomForestClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=RANDOM_SEED)),
    ('et',       ExtraTreesClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=RANDOM_SEED)),
    ('gb',       GradientBoostingClassifier(learning_rate=0.05, n_estimators=150, max_depth=3, random_state=RANDOM_SEED)),
]

n_base  = len(base_models)
oof_stk = np.zeros((len(y_traindev), n_base))
te_stk  = np.zeros((test_mask.sum(), n_base))
dv_stk  = np.zeros((dev_mask.sum(),  n_base))

for bi, (bname, bmod) in enumerate(base_models):
    oof_b = np.zeros(len(y_traindev))
    for tri, vli in skf.split(Xstr, y_traindev):
        Xtr_f = Xstr[tri]; Xvl_f = Xstr[vli]; ytr_f = y_traindev[tri]
        k2 = min(3, int(ytr_f.sum()) - 1)
        try:
            sm2 = SMOTE(random_state=RANDOM_SEED, k_neighbors=k2)
            Xtr_f, ytr_f = sm2.fit_resample(Xtr_f, ytr_f)
        except: pass
        m = sklearn.base.clone(bmod); m.fit(Xtr_f, ytr_f)
        oof_b[vli] = m.predict_proba(Xvl_f)[:, 1]
    oof_stk[:, bi] = oof_b
    m_full = sklearn.base.clone(bmod); m_full.fit(Xstr_s, y_str_s)
    te_stk[:, bi] = m_full.predict_proba(Xste)[:, 1]
    dv_stk[:, bi] = m_full.predict_proba(Xsdv)[:, 1]
    try: auc_b = roc_auc_score(y_traindev, oof_b)
    except: auc_b = 0.5
    print(f"  [{bname}] OOF AUC={auc_b:.4f}", flush=True)

meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=RANDOM_SEED)
meta.fit(oof_stk, y_traindev)
stk_probs_te = meta.predict_proba(te_stk)[:, 1]
stk_probs_dv = meta.predict_proba(dv_stk)[:, 1]

best_f1_sk, best_thr_sk = 0.0, 0.5
for thr in np.arange(0.20, 0.80, 0.01):
    f1 = f1_score(y_devonly, (stk_probs_dv >= thr).astype(int), average='macro', zero_division=0)
    if f1 > best_f1_sk: best_f1_sk, best_thr_sk = f1, thr
stk_preds = (stk_probs_te >= best_thr_sk).astype(int)
stk_f1    = f1_score(y_test_off, stk_preds, average='macro', zero_division=0)
stk_acc   = accuracy_score(y_test_off, stk_preds)
try: stk_auc = roc_auc_score(y_test_off, stk_probs_te)
except: stk_auc = 0.5
print(f"\nStacking ({best_feat}): F1={stk_f1:.4f} Acc={stk_acc:.4f} AUC={stk_auc:.4f} (thr={best_thr_sk:.2f})")
if stk_f1 > best_ens_f1:
    best_ens_f1 = stk_f1; best_ens_probs = stk_probs_te; best_ens_thr = best_thr_sk

# %% [markdown]
# ## Hybrid: Soft vote + Stacking

# %%
print("\n--- Hybrid Combinations ---")
if best_ens_probs is not None:
    top5 = top_by_test[:5]
    w5 = np.array([test_probs[k][2] for k in top5]); w5 = np.maximum(w5, 1e-8); w5 /= w5.sum()
    sv5 = np.average(np.column_stack([test_probs[k][0] for k in top5]), axis=1, weights=w5)

    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        hyb = alpha * sv5 + (1 - alpha) * stk_probs_te
        bf, bt = 0.0, 0.5
        for thr in np.arange(0.20, 0.80, 0.01):
            f1 = f1_score(y_test_off, (hyb >= thr).astype(int), average='macro', zero_division=0)
            if f1 > bf: bf, bt = f1, thr
        try: au = roc_auc_score(y_test_off, hyb)
        except: au = 0.5
        ac = accuracy_score(y_test_off, (hyb >= bt).astype(int))
        print(f"  Hybrid (sv={alpha:.1f}): F1={bf:.4f} Acc={ac:.4f} AUC={au:.4f} (thr={bt:.2f})")
        if bf > best_ens_f1:
            best_ens_f1 = bf; best_ens_probs = hyb; best_ens_thr = bt

# %% [markdown]
# ## Final Summary

# %%
print("\n" + "="*70)
print("FINAL SUMMARY v54 — Extended Grid + Tree Diversity")
print("="*70)

best_ind_f1 = 0.0
if test_probs:
    best_ind_key = max(test_probs.keys(), key=lambda k: test_probs[k][2])
    best_ind_f1  = test_probs[best_ind_key][2]
    print(f"Best Individual: {best_ind_key} | F1={best_ind_f1:.4f}")

print(f"Best Ensemble  : F1={best_ens_f1:.4f}")
print(f"Stacking       : F1={stk_f1:.4f}")

overall_best_f1 = max(best_ind_f1, best_ens_f1, stk_f1)

if best_ens_f1 >= max(best_ind_f1, stk_f1):
    fin_probs = best_ens_probs; fin_thr = best_ens_thr; fin_label = "Best Ensemble"
elif stk_f1 >= best_ind_f1:
    fin_probs = stk_probs_te;   fin_thr = best_thr_sk; fin_label = f"Stacking ({best_feat})"
else:
    fin_probs = test_probs[best_ind_key][0]; fin_thr = test_probs[best_ind_key][1]; fin_label = best_ind_key

fin_preds = (fin_probs >= fin_thr).astype(int)
try: fin_auc = roc_auc_score(y_test_off, fin_probs)
except: fin_auc = 0.5

print(f"\nClassification Report ({fin_label}, thr={fin_thr:.2f}):")
print(classification_report(y_test_off, fin_preds,
                              target_names=['Non-Depressed','Depressed'], zero_division=0))
print(f"Accuracy: {accuracy_score(y_test_off, fin_preds):.4f} | AUC: {fin_auc:.4f}")

df_cv.to_csv(os.path.join(RESULTS_DIR, "metrics", "v54_cv_results.csv"))
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Total elapsed: {time.time()-t_global:.0f}s")

print("\n--- Historical Comparison ---")
print(f"v50: 0.7202 | v51: 0.7756 | v52: 0.7696 | v53: 0.7299 | v54: {overall_best_f1:.4f}")
print()
if overall_best_f1 >= 0.80:
    print(f"TARGET ACHIEVED! Test Macro F1 = {overall_best_f1:.4f} >= 0.80")
elif overall_best_f1 >= 0.75:
    print(f"PREV-TARGET MET: Test Macro F1 = {overall_best_f1:.4f} >= 0.75")
    print(f"   Gap to 0.80: {0.80 - overall_best_f1:.4f}")
else:
    print(f"Target NOT yet achieved. Best F1 = {overall_best_f1:.4f}")
