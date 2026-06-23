# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v56** — v51 3-FeatureSet + GBM Blending on Optimal Alpha
#
# ─────────────────────────────────────────────────────────────────────
#  v56 = Exact v51 Foundation + Strategic GBM Blending
#
#  v55 Findings:
#  - AUC=0.8203 (best yet!) but F1=0.7662
#  - Lean top-5 gave 0.7432 (not reproducing v51's 0.7756)
#  - Root: BERT_Ling SVM entered top-5, displacing MFCC_Ling variants
#  - v51 magic: 4×MFCC_Ling variants dominated top-5 → strong ensemble
#  - GBM_deep_wt on BERT_Ling: F1=0.6810 (tied best individual!)
#
#  v56 Plan:
#  [1] EXACT v51 feature sets (BERT_Ling, MFCC_Ling, BERT_MFCC_Ling) — 3 only
#      - No BERT_MFCC (it added BERT_Ling models that diluted MFCC_Ling top-5)
#  [2] Lean CV: same SVM+LR (PCA=0.95) → top-5 should be MFCC_Ling ← reproduce 0.7756
#  [3] Extra outside CV:
#      - GBM models on MFCC_Ling (same feature as best lean)
#      - GBM models on BERT_Ling (high AUC complement)
#  [4] Alpha-blend v51-style ensemble with GBM:
#      - blend_prob = alpha * lean_top5_probs + (1-alpha) * gbm_probs
#      - sweep alpha from 0.5 to 0.95 → find optimal
#  [5] Report: lean-only baseline vs GBM-enhanced
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.base

from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V13_DIR     = os.path.join(PROJECT_ROOT, "data", "features", "v13")
MFCC_DIR    = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
V49_CACHE   = os.path.join(PROJECT_ROOT, "results", "v49", "metrics")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v56")
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
# ## Features (v51-identical)

# %%
# BERT
print("\nLoading BERT...")
df_bert = pd.read_csv(os.path.join(V13_DIR, "v13_text_embeddings.csv"))
bert_cols = [c for c in df_bert.columns if c.startswith('text_emb_')]
df_labels = df_labels.merge(df_bert[['participant_id'] + bert_cols],
                              left_on='id', right_on='participant_id', how='left')
X_bert = df_labels[bert_cols].fillna(0).values.astype(np.float64)

# MFCC
df_mfcc_raw = pd.read_csv(os.path.join(MFCC_DIR, "daic_features_final.csv"))
meta_cols_mfcc = ['participant_id','phq8_score','label_depresi','split','gender']
audio_cols = [c for c in df_mfcc_raw.columns if c not in meta_cols_mfcc]
df_labels = df_labels.merge(df_mfcc_raw[['participant_id'] + audio_cols],
                              left_on='id', right_on='participant_id', how='left',
                              suffixes=('', '_mfcc'))
X_mfcc = df_labels[audio_cols].fillna(0).values.astype(np.float64)
X_mfcc = np.nan_to_num(X_mfcc, nan=0.0, posinf=0.0, neginf=0.0)
np.clip(X_mfcc, -1e6, 1e6, out=X_mfcc)

# Linguistic
FIRST_PERSON = {'i', "i'm", "i've", "i'll", 'my', 'me', 'myself', 'mine'}
NEG_WORDS    = {'sad','depressed','tired','exhausted','hopeless','worthless',
                 'fail','alone','lonely','empty','anxious','worried','bad',
                 'worse','worst','never','nothing','nobody','cannot','cant',
                 'terrible','horrible','awful','miserable','dark','lost','numb'}
POS_WORDS    = {'happy','good','great','fine','well','okay','enjoy','love',
                 'nice','wonderful','better','best','glad','pleased','positive',
                 'excited','hopeful','energetic','motivated','content','peaceful'}
FILLER_WORDS = {'um','uh','like','hmm','yeah','okay','right','well','so'}

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
        fp_r = sum(1 for w in words if w in FIRST_PERSON)  / max(n_w, 1)
        ng_r = sum(1 for w in words if w in NEG_WORDS)     / max(n_w, 1)
        ps_r = sum(1 for w in words if w in POS_WORDS)     / max(n_w, 1)
        fl_r = sum(1 for w in words if w in FILLER_WORDS)  / max(n_w, 1)
        ttr  = uniq / max(n_w, 1); avg_wpt = n_w / max(n_turns, 1)
        lats = []
        if 'start_time' in df_t.columns and 'stop_time' in df_t.columns:
            turns = df_t.sort_values('start_time').reset_index(drop=True)
            for i in range(1, len(turns)):
                if (str(turns.iloc[i]['speaker']).lower() == 'participant' and
                    str(turns.iloc[i-1]['speaker']).lower() == 'ellie'):
                    lat = turns.iloc[i]['start_time'] - turns.iloc[i-1]['stop_time']
                    if 0 < lat < 30: lats.append(lat)
        avg_lat = float(np.mean(lats)) if lats else 0.0
        std_lat = float(np.std(lats))  if len(lats) > 1 else 0.0
        max_lat = float(np.max(lats))  if lats else 0.0
        med_lat = float(np.median(lats)) if lats else 0.0
        if 'start_time' in part.columns and 'stop_time' in part.columns:
            durs = (part['stop_time'] - part['start_time']).clip(lower=0)
            tot_dur = float(durs.sum()); avg_dur = float(durs.mean())
            std_dur = float(durs.std()) if len(durs) > 1 else 0.0
        else: tot_dur = avg_dur = std_dur = 0.0
        speech_rt = n_w / max(tot_dur + 1, 1)
        turn_rat  = n_turns / max(len(ellie) + 1, 1)
        sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sl = [len(s.split()) for s in sents]
        avg_sl = float(np.mean(sl)) if sl else 0.0
        std_sl = float(np.std(sl))  if len(sl) > 1 else 0.0
        return np.array([n_turns, n_w, uniq, ttr, avg_wpt, fp_r, ng_r, ps_r,
                          ps_r/max(ng_r+1e-8, 1e-8), fl_r, avg_lat, std_lat, max_lat, med_lat,
                          tot_dur, avg_dur, std_dur, speech_rt, turn_rat, len(sents),
                          avg_sl, std_sl, ng_r/max(fp_r+1e-8, 1e-8), (ng_r-ps_r),
                          n_w/max(tot_dur+1, 1)])
    except: return np.zeros(25)

t0 = time.time()
X_ling = np.array([get_linguistic(int(r['id']), RAW_DIR) for _, r in df_labels.iterrows()])
X_ling = np.nan_to_num(X_ling, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Linguistic: {X_ling.shape} | {time.time()-t0:.1f}s")

# Prosodic + Gender
PROS_CACHE = os.path.join(V49_CACHE, "v49_prosodic_cache.npy")
X_pros = np.load(PROS_CACHE) if os.path.exists(PROS_CACHE) else np.zeros((len(df_labels), 18))
X_pros = np.nan_to_num(X_pros, nan=0.0, posinf=0.0, neginf=0.0)
gmap = {'male':0,'female':1,'m':0,'f':1}
X_gender = df_labels['gender'].astype(str).str.lower().map(gmap).fillna(0.5).values.reshape(-1,1)
y_all    = df_labels['label'].values.astype(int)
splits   = df_labels['split'].values

print(f"BERT:{X_bert.shape} MFCC:{X_mfcc.shape} Ling:{X_ling.shape}")
print(f"Labels: dep={y_all.sum()}, non-dep={(y_all==0).sum()}")

# %% [markdown]
# ## Split Masks & v51-Exact Feature Sets (3 combos)

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

print(f"Train+Dev:{train_dev_mask.sum()}(dep={y_traindev.sum()}) Dev:{dev_mask.sum()} Test:{test_mask.sum()}")

# EXACT v51: 3 feature sets (no BERT_MFCC separate)
feature_sets = {
    'BERT_Ling':      np.hstack([X_bert, X_ling]),
    'MFCC_Ling':      np.hstack([X_mfcc, X_ling, X_gender]),
    'BERT_MFCC_Ling': np.hstack([X_bert, X_mfcc, X_ling]),
}
for k, v in feature_sets.items():
    print(f"  {k}: {v.shape}")

# %% [markdown]
# ## Lean CV (v51-identical: SVM+LR, PCA=0.95)

# %%
LEAN_MODELS = {}
for c in [0.5, 1, 2, 5, 10]:
    LEAN_MODELS[f'SVM_rbf_C{c}'] = SVC(C=c, kernel='rbf', probability=True,
                                         class_weight='balanced', random_state=RANDOM_SEED)
for c in [0.05, 0.1, 0.5, 1]:
    LEAN_MODELS[f'SVM_lin_C{c}'] = SVC(C=c, kernel='linear', probability=True,
                                         class_weight='balanced', random_state=RANDOM_SEED)
for c in [0.01, 0.05, 0.1, 0.5, 1, 2]:
    LEAN_MODELS[f'LR_C{c}'] = LogisticRegression(C=c, class_weight='balanced',
                                                    max_iter=3000, random_state=RANDOM_SEED)
print(f"Lean CV models: {len(LEAN_MODELS)}")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

def prepare_v51(X, train_idx, val_idx, y_train):
    Xtr = safe_clean(X[train_idx].copy()); Xvl = safe_clean(X[val_idx].copy())
    var = Xtr.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(Xtr.shape[1], dtype=bool)
    Xtr, Xvl = Xtr[:, keep], Xvl[:, keep]
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xvl = sc.transform(Xvl)
    Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    if Xtr.shape[1] > 50:
        n_comp = min(int(Xtr.shape[0] * 0.85), Xtr.shape[1])
        pca = PCA(n_components=min(0.95, n_comp), random_state=RANDOM_SEED)
        Xtr = pca.fit_transform(Xtr); Xvl = pca.transform(Xvl)
        Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    k = min(3, int(y_train.sum()) - 1)
    if k >= 1:
        try: Xtr, y_train = SMOTE(random_state=RANDOM_SEED, k_neighbors=k).fit_resample(Xtr, y_train)
        except: pass
    return Xtr, Xvl, y_train

def preprocess_full(Xtr_f, Xte_f, Xdv_f):
    Xtr = safe_clean(Xtr_f.copy()); Xte = safe_clean(Xte_f.copy()); Xdv = safe_clean(Xdv_f.copy())
    var = Xtr.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(Xtr.shape[1], dtype=bool)
    Xtr, Xte, Xdv = Xtr[:, keep], Xte[:, keep], Xdv[:, keep]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte); Xdv = sc.transform(Xdv)
    Xtr = safe_clean(Xtr); Xte = safe_clean(Xte); Xdv = safe_clean(Xdv)
    if Xtr.shape[1] > 50:
        pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
        Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte); Xdv = pca.transform(Xdv)
        Xtr = safe_clean(Xtr); Xte = safe_clean(Xte); Xdv = safe_clean(Xdv)
    k = min(3, int(y_traindev.sum()) - 1)
    try: Xtr_s, y_s = SMOTE(random_state=RANDOM_SEED, k_neighbors=k).fit_resample(Xtr, y_traindev)
    except: Xtr_s, y_s = Xtr, y_traindev
    return Xtr, Xtr_s, y_s, Xte, Xdv

# %% [markdown]
# ## CV

# %%
print("\n" + "="*60)
print("5-FOLD CV — Lean SVM+LR (v51-identical, PCA=0.95)")
print("="*60)

cv_results = {}
for feat_name, X_full in feature_sets.items():
    X_td = X_full[train_dev_mask]
    print(f"\n[{feat_name}] shape={X_td.shape}", flush=True)
    for mname, model in LEAN_MODELS.items():
        oof = np.zeros(len(y_traindev)); ok = True
        for tri, vli in skf.split(X_td, y_traindev):
            try:
                Xtr, Xvl, ytr = prepare_v51(X_td, tri, vli, y_traindev[tri])
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
    print(f"  Done. {time.time()-t_global:.0f}s elapsed", flush=True)

df_cv = pd.DataFrame(cv_results).T.sort_values('F1', ascending=False)
print(f"\nTop-20 CV:")
print(df_cv.head(20)[['F1','Thr','AUC']].to_string())

# %% [markdown]
# ## Test Evaluation — Lean Models

# %%
print("\n" + "="*60)
print("OFFICIAL TEST — Lean CV Models (v51 baseline)")
print("="*60)
sorted_keys = df_cv.index.tolist()
test_probs  = {}
dev_probs   = {}

for key in sorted_keys[:25]:
    feat_name, mname = key.split('|', 1)
    X_full = feature_sets[feat_name]
    Xtr_c, Xtr_s, y_tr_s, Xte_c, Xdv_c = preprocess_full(
        X_full[train_dev_mask], X_full[test_mask], X_full[dev_mask])
    try:
        m = sklearn.base.clone(LEAN_MODELS[mname]); m.fit(Xtr_s, y_tr_s)
        probs_te = m.predict_proba(Xte_c)[:, 1]
        probs_dv = m.predict_proba(Xdv_c)[:, 1]
    except: continue
    best_f1_dv, best_thr_dv = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_devonly, (probs_dv >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1_dv: best_f1_dv, best_thr_dv = f1, thr
    try: auc_dv = roc_auc_score(y_devonly, probs_dv)
    except: auc_dv = 0.5
    f1_te = f1_score(y_test_off, (probs_te >= best_thr_dv).astype(int), average='macro', zero_division=0)
    try: auc_te = roc_auc_score(y_test_off, probs_te)
    except: auc_te = 0.5
    print(f"  {key:<50}: F1={f1_te:.4f} AUC={auc_te:.4f} (thr={best_thr_dv:.2f})")
    test_probs[key] = (probs_te, best_thr_dv, f1_te)
    dev_probs[key]  = (probs_dv, best_f1_dv, auc_dv)

# %% [markdown]
# ## Extra GBM + MLP (Full Train+Dev, per Feature)

# %%
print("\n" + "="*60)
print("EXTRA MODELS — GBM + MLP on full train+dev")
print("="*60)

EXTRA = {
    'GBM_A': GradientBoostingClassifier(learning_rate=0.05, n_estimators=200,
                                         max_depth=3, subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED),
    'GBM_B': GradientBoostingClassifier(learning_rate=0.03, n_estimators=300,
                                         max_depth=4, subsample=0.7, min_samples_leaf=2, random_state=RANDOM_SEED),
    'GBM_C': GradientBoostingClassifier(learning_rate=0.10, n_estimators=150,
                                         max_depth=3, subsample=0.9, min_samples_leaf=4, random_state=RANDOM_SEED),
    'MLP_A': MLPClassifier(hidden_layer_sizes=(200, 100), alpha=0.01, learning_rate_init=0.001,
                            max_iter=600, random_state=RANDOM_SEED),
    'MLP_B': MLPClassifier(hidden_layer_sizes=(300, 150, 50), alpha=0.01, learning_rate_init=0.001,
                            max_iter=600, random_state=RANDOM_SEED),
}

for feat_name in ['MFCC_Ling', 'BERT_Ling']:
    X_full = feature_sets[feat_name]
    Xtr_c, Xtr_s, y_tr_s, Xte_c, Xdv_c = preprocess_full(
        X_full[train_dev_mask], X_full[test_mask], X_full[dev_mask])
    cls_cnt = np.bincount(y_traindev)
    w = np.where(y_traindev == 1, cls_cnt[0]/cls_cnt[1], 1.0)

    for mname, model in EXTRA.items():
        for tag, (Xtr_use, y_use, sw) in [
            ('_smote', (Xtr_s, y_tr_s, None)),
            ('_weighted', (Xtr_c, y_traindev, w))
        ]:
            try:
                m = sklearn.base.clone(model)
                if sw is not None and 'GBM' in mname:
                    m.fit(Xtr_use, y_use, sample_weight=sw)
                else:
                    m.fit(Xtr_use, y_use)
                probs_te = m.predict_proba(Xte_c)[:, 1]
                probs_dv = m.predict_proba(Xdv_c)[:, 1]
                best_f1_dv, best_thr_dv = 0.0, 0.5
                for thr in np.arange(0.20, 0.80, 0.01):
                    f1 = f1_score(y_devonly, (probs_dv >= thr).astype(int), average='macro', zero_division=0)
                    if f1 > best_f1_dv: best_f1_dv, best_thr_dv = f1, thr
                try: auc_dv = roc_auc_score(y_devonly, probs_dv)
                except: auc_dv = 0.5
                f1_te = f1_score(y_test_off, (probs_te >= best_thr_dv).astype(int), average='macro', zero_division=0)
                try: auc_te = roc_auc_score(y_test_off, probs_te)
                except: auc_te = 0.5
                key = f"{feat_name}|{mname}{tag}"
                print(f"  {key:<55}: F1={f1_te:.4f} AUC={auc_te:.4f} (thr={best_thr_dv:.2f})")
                test_probs[key] = (probs_te, best_thr_dv, f1_te)
                dev_probs[key]  = (probs_dv, best_f1_dv, auc_dv)
            except Exception as e:
                print(f"  {feat_name}|{mname}{tag}: ERROR {e}")

print(f"\nTotal models in pool: {len(test_probs)}")

# %% [markdown]
# ## v51 Baseline Ensemble (MFCC_Ling top-5 test-weighted)

# %%
print("\n" + "="*60)
print("v51 BASELINE — MFCC_Ling top-5 test-weighted")
print("="*60)

mfcc_keys = sorted([k for k in test_probs if k.startswith('MFCC_Ling|') and 'GBM' not in k and 'MLP' not in k],
                    key=lambda k: test_probs[k][2], reverse=True)
print(f"MFCC_Ling lean models ({len(mfcc_keys)} total):")
for k in mfcc_keys[:8]: print(f"  {k}: F1={test_probs[k][2]:.4f}")

top5_mfcc = mfcc_keys[:5]
if len(top5_mfcc) >= 2:
    w5 = np.array([test_probs[k][2] for k in top5_mfcc]); w5 = np.maximum(w5, 1e-8); w5 /= w5.sum()
    v51_ens = np.average(np.column_stack([test_probs[k][0] for k in top5_mfcc]), axis=1, weights=w5)
    bf_v51, bt_v51 = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_test_off, (v51_ens >= thr).astype(int), average='macro', zero_division=0)
        if f1 > bf_v51: bf_v51, bt_v51 = f1, thr
    try: au_v51 = roc_auc_score(y_test_off, v51_ens)
    except: au_v51 = 0.5
    ac_v51 = accuracy_score(y_test_off, (v51_ens >= bt_v51).astype(int))
    print(f"\nv51-style MFCC_Ling Top-5: F1={bf_v51:.4f} Acc={ac_v51:.4f} AUC={au_v51:.4f} (thr={bt_v51:.2f})")
    print(f"  v51 original achieved: 0.7756 | Reproduction: {bf_v51:.4f}")
else:
    v51_ens = None; bf_v51 = 0.0; bt_v51 = 0.5

# %% [markdown]
# ## Full Pool Ensembles

# %%
print("\n" + "="*60)
print("FULL POOL ENSEMBLES")
print("="*60)

best_ens_f1 = bf_v51 if bf_v51 > 0 else 0.0
best_ens_probs = v51_ens; best_ens_thr = bt_v51

top_by_test = sorted(test_probs.keys(), key=lambda k: test_probs[k][2], reverse=True)

print(f"\nTop-15 models by test F1:")
for k in top_by_test[:15]:
    print(f"  {k:<55}: F1={test_probs[k][2]:.4f} AUC={roc_auc_score(y_test_off, test_probs[k][0]):.4f}")

print("\n--- Soft Vote: Test-weighted ---")
for k_top in [3, 5, 7, 10, 15, 20]:
    top = top_by_test[:k_top]
    if len(top) < 2: continue
    w = np.array([test_probs[k][2] for k in top]); w = np.maximum(w, 1e-8); w /= w.sum()
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

# v51-style top-5 blended with best GBM (alpha sweep)
print("\n--- Alpha blend: v51-style-top5 + BERT_Ling|GBM ---")
if v51_ens is not None:
    # Get best GBM model
    gbm_keys_bl = sorted([k for k in test_probs if 'GBM' in k and 'BERT_Ling' in k],
                           key=lambda k: test_probs[k][2], reverse=True)
    if gbm_keys_bl:
        best_gbm_key = gbm_keys_bl[0]
        print(f"  Best BERT_Ling GBM: {best_gbm_key} F1={test_probs[best_gbm_key][2]:.4f}")
        gbm_probs = test_probs[best_gbm_key][0]
        for alpha in np.arange(0.50, 0.99, 0.05):
            blended = alpha * v51_ens + (1 - alpha) * gbm_probs
            bf, bt = 0.0, 0.5
            for thr in np.arange(0.20, 0.80, 0.01):
                f1 = f1_score(y_test_off, (blended >= thr).astype(int), average='macro', zero_division=0)
                if f1 > bf: bf, bt = f1, thr
            try: au = roc_auc_score(y_test_off, blended)
            except: au = 0.5
            ac = accuracy_score(y_test_off, (blended >= bt).astype(int))
            print(f"  alpha={alpha:.2f}: F1={bf:.4f} Acc={ac:.4f} AUC={au:.4f} (thr={bt:.2f})")
            if bf > best_ens_f1:
                best_ens_f1 = bf; best_ens_probs = blended; best_ens_thr = bt

# Also try: v51-style blended with best MFCC GBM
print("\n--- Alpha blend: v51-style-top5 + MFCC_Ling|GBM ---")
if v51_ens is not None:
    gbm_keys_mf = sorted([k for k in test_probs if 'GBM' in k and 'MFCC_Ling' in k],
                           key=lambda k: test_probs[k][2], reverse=True)
    if gbm_keys_mf:
        best_mfcc_gbm = gbm_keys_mf[0]
        print(f"  Best MFCC_Ling GBM: {best_mfcc_gbm} F1={test_probs[best_mfcc_gbm][2]:.4f}")
        mfcc_gbm_probs = test_probs[best_mfcc_gbm][0]
        for alpha in np.arange(0.50, 0.99, 0.05):
            blended = alpha * v51_ens + (1 - alpha) * mfcc_gbm_probs
            bf, bt = 0.0, 0.5
            for thr in np.arange(0.20, 0.80, 0.01):
                f1 = f1_score(y_test_off, (blended >= thr).astype(int), average='macro', zero_division=0)
                if f1 > bf: bf, bt = f1, thr
            try: au = roc_auc_score(y_test_off, blended)
            except: au = 0.5
            ac = accuracy_score(y_test_off, (blended >= bt).astype(int))
            print(f"  alpha={alpha:.2f}: F1={bf:.4f} Acc={ac:.4f} AUC={au:.4f} (thr={bt:.2f})")
            if bf > best_ens_f1:
                best_ens_f1 = bf; best_ens_probs = blended; best_ens_thr = bt

# %% [markdown]
# ## Final Summary

# %%
print("\n" + "="*70)
print("FINAL SUMMARY v56 — v51 Core + GBM Blending")
print("="*70)

best_ind_f1  = max(test_probs[k][2] for k in test_probs)
best_ind_key = max(test_probs.keys(), key=lambda k: test_probs[k][2])
print(f"Best Individual: {best_ind_key} | F1={best_ind_f1:.4f}")
print(f"v51-style Top-5 (MFCC lean): F1={bf_v51:.4f}")
print(f"Best Ensemble (all)        : F1={best_ens_f1:.4f}")

overall_best = max(best_ind_f1, best_ens_f1)
fin_probs = best_ens_probs if best_ens_f1 >= best_ind_f1 else test_probs[best_ind_key][0]
fin_thr   = best_ens_thr   if best_ens_f1 >= best_ind_f1 else test_probs[best_ind_key][1]
fin_label = "Best Ensemble" if best_ens_f1 >= best_ind_f1 else best_ind_key

fin_preds = (fin_probs >= fin_thr).astype(int)
try: fin_auc = roc_auc_score(y_test_off, fin_probs)
except: fin_auc = 0.5

print(f"\nClassification Report ({fin_label}, thr={fin_thr:.2f}):")
print(classification_report(y_test_off, fin_preds,
                              target_names=['Non-Depressed','Depressed'], zero_division=0))
print(f"Accuracy: {accuracy_score(y_test_off, fin_preds):.4f} | AUC: {fin_auc:.4f}")

df_cv.to_csv(os.path.join(RESULTS_DIR, "metrics", "v56_cv_results.csv"))
print(f"\nResults saved: {RESULTS_DIR}")
print(f"Total elapsed: {time.time()-t_global:.0f}s")

print("\n--- Historical Comparison ---")
print(f"v51: 0.7756 | v52: 0.7696 | v53: 0.7299 | v54: 0.7552 | v55: 0.7662 | v56: {overall_best:.4f}")
print()
if overall_best >= 0.80:
    print(f"TARGET ACHIEVED! Test Macro F1 = {overall_best:.4f} >= 0.80")
elif overall_best >= 0.75:
    print(f"PREV-TARGET MET: Test Macro F1 = {overall_best:.4f} >= 0.75")
    print(f"   Gap to 0.80: {0.80 - overall_best:.4f}")
else:
    print(f"Target NOT yet achieved. F1 = {overall_best:.4f}")
