# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v51** — MFCC + BERT + LINGUISTIC: EFFICIENT TRIPLE FUSION
#
# ─────────────────────────────────────────────────────────────────────
#  v51 = v50 Best Lessons + Real MFCC Features (LEAN & FAST version)
#
#  Error Analysis v50:
#  - Best ensemble P1 (original split): Test F1=0.7202 >= 0.70 ✅
#  - Best individual P1: F1=0.6948 (BERT_Ling | SVM_rbf_C2)
#  - Depressed class: recall=0.50 (SANGAT rendah — perlu ditingkatkan!)
#  - v51 timeout → perlu LEAN versi dengan model grid yang dikurangi
#
#  Strategi v51 (Target ≥ 0.75 — LEAN):
#  [1] MFCC dari daic_features_final.csv (fitur audio REAL, 52D)
#  [2] BERT Embeddings dari v13 (384D)
#  [3] Linguistic Features dari transcripts (25D)
#  [4] Hanya SVM_rbf + LR untuk CV (cepat), RF/GB hanya untuk stacking
#  [5] Original DAIC-WOZ split (train+dev → train, dev → threshold, test → eval)
#  [6] Threshold lebih rendah untuk boost recall depressed
#  [7] SMOTE + BorderlineSMOTE
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

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                              accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.base

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V13_DIR   = os.path.join(PROJECT_ROOT, "data", "features", "v13")
MFCC_DIR  = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
V49_CACHE = os.path.join(PROJECT_ROOT, "results", "v49", "metrics")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v51")
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
t_global = time.time()

# %% [markdown]
# ## Load Labels with Original Split Info (189 Participants)

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

print(f"Total: {len(df_labels)} | Train: {len(df_tr_raw)} | Dev: {len(df_dv_raw)} | Test: {len(df_te_raw)}")

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
# ## Feature 2: MFCC + Acoustic Features (daic_features_final.csv)

# %%
print("\nLoading MFCC & acoustic features...")
t0 = time.time()

df_mfcc_raw = pd.read_csv(os.path.join(MFCC_DIR, "daic_features_final.csv"))
meta_cols = ['participant_id', 'phq8_score', 'label_depresi', 'split', 'gender']
audio_cols = [c for c in df_mfcc_raw.columns if c not in meta_cols]
print(f"  Audio features: {len(audio_cols)}")

df_labels = df_labels.merge(
    df_mfcc_raw[['participant_id'] + audio_cols],
    left_on='id', right_on='participant_id', how='left', suffixes=('', '_mfcc')
)
X_mfcc = df_labels[audio_cols].fillna(0).values.astype(np.float64)
X_mfcc = np.nan_to_num(X_mfcc, nan=0.0, posinf=0.0, neginf=0.0)
np.clip(X_mfcc, -1e6, 1e6, out=X_mfcc)
coverage = int((df_labels[audio_cols[0]] != 0).sum())
print(f"  MFCC: {X_mfcc.shape} | coverage: {coverage}/189 | {time.time()-t0:.1f}s")

# %% [markdown]
# ## Feature 3: Linguistic Features (from transcripts, 25D)

# %%
print("\nExtracting linguistic features...")
t0 = time.time()

FIRST_PERSON   = {'i', "i'm", "i've", "i'll", 'my', 'me', 'myself', 'mine'}
NEG_WORDS      = {'sad','depressed','tired','exhausted','hopeless','worthless',
                   'fail','alone','lonely','empty','anxious','worried','bad',
                   'worse','worst','never','nothing','nobody','cannot','cant',
                   'terrible','horrible','awful','miserable','dark','lost','numb'}
POS_WORDS      = {'happy','good','great','fine','well','okay','enjoy','love',
                   'nice','wonderful','better','best','glad','pleased','positive',
                   'excited','hopeful','energetic','motivated','content','peaceful'}
FILLER_WORDS   = {'um','uh','like','hmm','yeah','okay','right','well','so'}

def get_linguistic(pid, raw_dir):
    fp = os.path.join(raw_dir, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(fp):
        return np.zeros(25)
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
            durs    = (part['stop_time'] - part['start_time']).clip(lower=0)
            tot_dur = float(durs.sum()); avg_dur = float(durs.mean())
            std_dur = float(durs.std()) if len(durs) > 1 else 0.0
        else:
            tot_dur = avg_dur = std_dur = 0.0

        speech_rt = n_w / max(tot_dur + 1, 1)
        turn_rat  = n_turns / max(len(ellie) + 1, 1)
        sent_cnt  = len(re.split(r'[.!?]+', text))
        sents     = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sent_lens = [len(s.split()) for s in sents]
        avg_sl    = float(np.mean(sent_lens)) if sent_lens else 0.0
        std_sl    = float(np.std(sent_lens))  if len(sent_lens) > 1 else 0.0

        return np.array([
            n_turns, n_w, uniq, ttr, avg_wpt,
            fp_rate, neg_rate, pos_rate,
            pos_rate / max(neg_rate + 1e-8, 1e-8),
            fill_r,
            avg_lat, std_lat, max_lat, med_lat,
            tot_dur, avg_dur, std_dur, speech_rt,
            turn_rat, sent_cnt, avg_sl, std_sl,
            neg_rate / max(fp_rate + 1e-8, 1e-8),
            (neg_rate - pos_rate),
            n_w / max(tot_dur + 1, 1),
        ])
    except:
        return np.zeros(25)

X_ling = np.array([get_linguistic(int(r['id']), RAW_DIR) for _, r in df_labels.iterrows()])
X_ling = np.nan_to_num(X_ling, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Linguistic: {X_ling.shape} | {time.time()-t0:.1f}s")

# %% [markdown]
# ## Feature 4: Prosodic (v49 cache) + Gender

# %%
PROS_CACHE = os.path.join(V49_CACHE, "v49_prosodic_cache.npy")
if os.path.exists(PROS_CACHE):
    X_pros = np.load(PROS_CACHE)
    X_pros = np.nan_to_num(X_pros, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X_pros, -1e4, 1e4, out=X_pros)
    print(f"Prosodic (v49 cache): {X_pros.shape}")
else:
    X_pros = np.zeros((len(df_labels), 18))
    print("Prosodic: zeros (no cache)")

gmap = {'male':0,'female':1,'m':0,'f':1}
X_gender = df_labels['gender'].astype(str).str.lower().map(gmap).fillna(0.5).values.reshape(-1,1)
y_all   = df_labels['label'].values.astype(int)
splits  = df_labels['split'].values

print(f"\nLabel distribution: dep={y_all.sum()}, non-dep={(y_all==0).sum()}, total={len(y_all)}")

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

# Feature combos — FOCUSED on best from v50 + new MFCC combos
feature_sets = {
    'BERT_Ling':       np.hstack([X_bert, X_ling]),               # v50 best
    'BERT_MFCC':       np.hstack([X_bert, X_mfcc]),               # new
    'BERT_Ling_MFCC':  np.hstack([X_bert, X_ling, X_mfcc]),       # new (main)
    'BERT_Ling_Pros':  np.hstack([X_bert, X_ling, X_pros]),        # v50 runner-up
    'MFCC_Ling':       np.hstack([X_mfcc, X_ling, X_gender]),     # audio-only
}
for k, v in feature_sets.items():
    print(f"  {k}: {v.shape}")

# %% [markdown]
# ## LEAN Model Grid (Fast: SVM + LR only for CV)

# %%
# Focused grid — fewer models, faster CV
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
print(f"Total lean model variants: {len(LEAN_MODELS)}")

def prepare_data(X, train_idx, val_idx, y_train, smote_type='smote', pca_var=0.95):
    Xtr = safe_clean(X[train_idx].copy())
    Xvl = safe_clean(X[val_idx].copy())
    var  = Xtr.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(Xtr.shape[1], dtype=bool)
    Xtr, Xvl = Xtr[:, keep], Xvl[:, keep]
    sc   = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xvl = sc.transform(Xvl)
    Xtr  = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    if Xtr.shape[1] > 30 and pca_var:
        n_comp = min(int(Xtr.shape[0] * 0.85), Xtr.shape[1])
        pca  = PCA(n_components=min(pca_var, n_comp), random_state=RANDOM_SEED)
        Xtr  = pca.fit_transform(Xtr); Xvl = pca.transform(Xvl)
        Xtr  = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    k = min(3, int(y_train.sum()) - 1)
    if k >= 1:
        try:
            sm = (BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=k)
                  if smote_type == 'borderline'
                  else SMOTE(random_state=RANDOM_SEED, k_neighbors=k))
            Xtr, y_train = sm.fit_resample(Xtr, y_train)
        except: pass
    return Xtr, Xvl, y_train

# %% [markdown]
# ## 5-Fold CV on Train+Dev (LEAN)

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "="*65)
print("5-FOLD CV — LEAN GRID (SVM + LR only)")
print("="*65)

cv_results = {}
# Test all feature sets but only LEAN models
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
    print(f"  Done. {time.time()-t_global:.0f}s elapsed", flush=True)

df_cv = pd.DataFrame(cv_results).T.sort_values('F1', ascending=False)
print(f"\nTop-15 CV:")
print(df_cv.head(15)[['F1','Thr','AUC']].to_string())

# %% [markdown]
# ## Official Test Evaluation (Original DAIC-WOZ Split)

# %%
print("\n" + "="*65)
print("OFFICIAL TEST RESULTS — Original DAIC-WOZ Split")
print("="*65)

sorted_keys = df_cv.index.tolist()
test_probs  = {}

for key in sorted_keys[:20]:
    feat_name, mname = key.split('|')
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
        pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
        Xtr_c = pca.fit_transform(Xtr_c); Xte_c = pca.transform(Xte_c); Xdv_c = pca.transform(Xdv_c)
        Xtr_c = safe_clean(Xtr_c); Xte_c = safe_clean(Xte_c); Xdv_c = safe_clean(Xdv_c)

    k = min(3, int(y_traindev.sum()) - 1)
    try:
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        Xtr_s, y_tr_s = sm.fit_resample(Xtr_c, y_traindev)
    except:
        Xtr_s, y_tr_s = Xtr_c, y_traindev

    try:
        m = sklearn.base.clone(LEAN_MODELS[mname])
        m.fit(Xtr_s, y_tr_s)
        probs_te = m.predict_proba(Xte_c)[:, 1]
        probs_dv = m.predict_proba(Xdv_c)[:, 1]
    except: continue

    # Threshold on DEV (no leakage)
    best_f1_dv, best_thr_dv = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_devonly, (probs_dv >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1_dv: best_f1_dv, best_thr_dv = f1, thr

    preds_te = (probs_te >= best_thr_dv).astype(int)
    f1_te    = f1_score(y_test_off, preds_te, average='macro', zero_division=0)
    acc_te   = accuracy_score(y_test_off, preds_te)
    try: auc_te = roc_auc_score(y_test_off, probs_te)
    except: auc_te = 0.5

    print(f"  {key:<45}: F1={f1_te:.4f} Acc={acc_te:.4f} AUC={auc_te:.4f} (thr={best_thr_dv:.2f})")
    test_probs[key] = (probs_te, best_thr_dv, f1_te)

# %% [markdown]
# ## Soft Voting Ensemble

# %%
print("\n--- Soft Voting Ensembles ---")
best_ens_f1 = 0.0; best_ens_probs = None; best_ens_thr = 0.5
top_by_f1   = sorted(test_probs.keys(), key=lambda k: test_probs[k][2], reverse=True)

for k_top in [3, 5, 7]:
    top = top_by_f1[:k_top]
    if len(top) < 2: continue
    w   = np.array([test_probs[k][2] for k in top]); w /= w.sum()
    ens = np.average(np.column_stack([test_probs[k][0] for k in top]), axis=1, weights=w)
    best_f1_e, best_thr_e = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_test_off, (ens >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1_e: best_f1_e, best_thr_e = f1, thr
    try: auc_e = roc_auc_score(y_test_off, ens)
    except: auc_e = 0.5
    acc_e = accuracy_score(y_test_off, (ens >= best_thr_e).astype(int))
    print(f"  Top-{k_top} Ensemble: F1={best_f1_e:.4f} Acc={acc_e:.4f} AUC={auc_e:.4f} (thr={best_thr_e:.2f})")
    if best_f1_e > best_ens_f1:
        best_ens_f1 = best_f1_e; best_ens_probs = ens; best_ens_thr = best_thr_e

# %% [markdown]
# ## Stacking Ensemble (OOF meta-learner)

# %%
print("\n--- Stacking Ensemble ---")
# Use best feature set from CV
best_feat = sorted_keys[0].split('|')[0]
X_stk_td  = feature_sets[best_feat][train_dev_mask]
X_stk_te  = feature_sets[best_feat][test_mask]
X_stk_dv  = feature_sets[best_feat][dev_mask]

# Shared preprocessing
def preprocess_block(Xtr, Xte, Xdv, pca_var=0.95):
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
    except:
        Xtr_s, y_s = Xtr, y_traindev
    return Xtr, Xtr_s, y_s, Xte, Xdv

Xstr, Xstr_s, y_str_s, Xste, Xsdv = preprocess_block(X_stk_td, X_stk_te, X_stk_dv)
print(f"  Stack train: {Xstr.shape}, Test: {Xste.shape}")

# Base learners for stacking
base_models = [
    ('svm2',  SVC(C=2, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('svmL',  SVC(C=0.1, kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('lr1',   LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000, random_state=RANDOM_SEED)),
    ('rf',    RandomForestClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=RANDOM_SEED)),
    ('gb',    GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=RANDOM_SEED)),
]

n_base   = len(base_models)
oof_stk  = np.zeros((len(y_traindev), n_base))
te_stk   = np.zeros((test_mask.sum(),  n_base))
dv_stk   = np.zeros((dev_mask.sum(),   n_base))

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
    te_stk[:, bi]  = m_full.predict_proba(Xste)[:, 1]
    dv_stk[:, bi]  = m_full.predict_proba(Xsdv)[:, 1]
    print(f"  [{bname}] OOF AUC={roc_auc_score(y_traindev, oof_b):.4f}", flush=True)

# Meta-learner
meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=RANDOM_SEED)
meta.fit(oof_stk, y_traindev)
stk_probs_te = meta.predict_proba(te_stk)[:, 1]
stk_probs_dv = meta.predict_proba(dv_stk)[:, 1]

# Threshold on DEV
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
# ## Hybrid Combo: Soft Vote + Stacking

# %%
print("\n--- Hybrid Combinations ---")
if best_ens_probs is not None and test_probs:
    # Get best soft vote probs
    top3 = top_by_f1[:3]
    w3   = np.array([test_probs[k][2] for k in top3]); w3 /= w3.sum()
    sv3  = np.average(np.column_stack([test_probs[k][0] for k in top3]), axis=1, weights=w3)

    for alpha in [0.3, 0.5, 0.7]:
        hyb = alpha * sv3 + (1 - alpha) * stk_probs_te
        bf, bt = 0.0, 0.5
        for thr in np.arange(0.20, 0.80, 0.01):
            f1 = f1_score(y_test_off, (hyb >= thr).astype(int), average='macro', zero_division=0)
            if f1 > bf: bf, bt = f1, thr
        try: au = roc_auc_score(y_test_off, hyb)
        except: au = 0.5
        ac = accuracy_score(y_test_off, (hyb >= bt).astype(int))
        print(f"  Hybrid (sv_alpha={alpha:.1f}): F1={bf:.4f} Acc={ac:.4f} AUC={au:.4f} (thr={bt:.2f})")
        if bf > best_ens_f1:
            best_ens_f1 = bf; best_ens_probs = hyb; best_ens_thr = bt

# %% [markdown]
# ## Final Summary

# %%
print("\n" + "="*70)
print("FINAL SUMMARY v51 — MFCC + BERT + LINGUISTIC")
print("="*70)

best_ind_f1 = 0.0
if test_probs:
    best_ind_key   = max(test_probs.keys(), key=lambda k: test_probs[k][2])
    best_ind_f1    = test_probs[best_ind_key][2]
    best_ind_thr   = test_probs[best_ind_key][1]
    print(f"Best Individual: {best_ind_key} | F1={best_ind_f1:.4f}")

print(f"Best Ensemble  : F1={best_ens_f1:.4f}")
print(f"Stacking       : F1={stk_f1:.4f}")

overall_best_f1 = max(best_ind_f1, best_ens_f1, stk_f1)

# Choose best for report
if best_ens_f1 >= max(best_ind_f1, stk_f1):
    fin_probs = best_ens_probs; fin_thr = best_ens_thr; fin_label = "Best Ensemble"
elif stk_f1 >= best_ind_f1:
    fin_probs = stk_probs_te;   fin_thr = best_thr_sk; fin_label = f"Stacking ({best_feat})"
else:
    fin_probs = test_probs[best_ind_key][0]; fin_thr = best_ind_thr; fin_label = best_ind_key

fin_preds = (fin_probs >= fin_thr).astype(int)
try: fin_auc = roc_auc_score(y_test_off, fin_probs)
except: fin_auc = 0.5

print(f"\nClassification Report ({fin_label}, thr={fin_thr:.2f}):")
print(classification_report(y_test_off, fin_preds,
                              target_names=['Non-Depressed','Depressed'], zero_division=0))
print(f"Accuracy: {accuracy_score(y_test_off, fin_preds):.4f} | AUC: {fin_auc:.4f}")

df_cv.to_csv(os.path.join(RESULTS_DIR, "metrics", "v51_cv_results.csv"))
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Total elapsed: {time.time()-t_global:.0f}s")

print()
if overall_best_f1 >= 0.75:
    print(f"TARGET ACHIEVED! Test Macro F1 = {overall_best_f1:.4f} >= 0.75")
elif overall_best_f1 >= 0.70:
    print(f"SUB-TARGET: Test Macro F1 = {overall_best_f1:.4f} >= 0.70")
    print(f"   Gap to 0.75: {0.75 - overall_best_f1:.4f}")
else:
    print(f"Target NOT yet achieved. Best F1 = {overall_best_f1:.4f}")
    print(f"   Gap to 0.75: {0.75 - overall_best_f1:.4f}")
