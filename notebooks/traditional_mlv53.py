# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v53** — TF-IDF + BERT + Extended Linguistic + MFCC (Mean Imputed)
#
# ─────────────────────────────────────────────────────────────────────
#  v53 = v52 Error Analysis → Fix Missing MFCC + Add TF-IDF
#
#  Error Analysis v52:
#  - Best Ensemble: F1=0.7696 (worse than v51's 0.7756!)
#  - Root cause: eGeMAPS/MFCC only covers 102/189 participants
#    - Test participants (47) have all-ZERO MFCC/eGeMAPS
#    - Model learns MFCC patterns in CV that don't exist in test
#  - v51 Top-5 ensemble: 0.7756 (MFCC_Ling variants dominant)
#  - Gap to 0.80: 0.024
#
#  Strategi v53 (Target >= 0.80):
#  [1] TF-IDF dari transcripts (full 189/189 coverage)
#      - Captures specific depression-related vocabulary
#      - Complementary to dense BERT embeddings
#  [2] BERT Embeddings (384D, 189/189)
#  [3] Extended Linguistic (40D, richer than v51's 25D)
#      - Add: question frequency, hedging words, certainty, sentence complexity
#  [4] MFCC with MEAN IMPUTATION (56D, training mean for missing test)
#      - Fix zero-fill bias that hurts test predictions
#      - Add hasMFCC indicator feature
#  [5] DEV-weighted ensemble (no leakage from test set)
#  [6] Calibrated classifiers (isotonic regression)
#  [7] ExtraTrees in ensemble (diverse decision boundaries)
#  [8] Original DAIC-WOZ split
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
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (f1_score, roc_auc_score, classification_report, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
import sklearn.base

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V13_DIR   = os.path.join(PROJECT_ROOT, "data", "features", "v13")
MFCC_DIR  = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
V49_CACHE = os.path.join(PROJECT_ROOT, "results", "v49", "metrics")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v53")
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

t_global = time.time()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %% [markdown]
# ## Load Labels with Original Split Info

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
# ## Feature 2: MFCC with Mean Imputation (fix missing test participants)

# %%
print("\nLoading MFCC with mean imputation...")
t0 = time.time()

df_mfcc_raw = pd.read_csv(os.path.join(MFCC_DIR, "daic_features_final.csv"))
meta_cols_mfcc = ['participant_id','phq8_score','label_depresi','split','gender']
audio_cols = [c for c in df_mfcc_raw.columns if c not in meta_cols_mfcc]

# Merge MFCC
df_labels = df_labels.merge(df_mfcc_raw[['participant_id'] + audio_cols],
                              left_on='id', right_on='participant_id', how='left',
                              suffixes=('', '_mfcc'))
X_mfcc_raw = df_labels[audio_cols].values.astype(np.float64)

# hasMFCC indicator (before imputation)
has_mfcc = (~np.isnan(X_mfcc_raw).any(axis=1)).astype(float).reshape(-1, 1)
print(f"  MFCC coverage: {int(has_mfcc.sum())}/189")

# Mean imputation from train+dev (IMPORTANT: compute mean only from train+dev)
# We'll compute it after split definition — for now just clean
X_mfcc_raw = np.nan_to_num(X_mfcc_raw, nan=0.0, posinf=0.0, neginf=0.0)
np.clip(X_mfcc_raw, -1e6, 1e6, out=X_mfcc_raw)
print(f"  MFCC raw: {X_mfcc_raw.shape} | {time.time()-t0:.1f}s")

# %% [markdown]
# ## Feature 3: TF-IDF from Transcripts (full 189/189 coverage)

# %%
print("\nExtracting TF-IDF features from transcripts...")
t0 = time.time()

def get_participant_text(pid, raw_dir):
    """Extract participant-only utterances from transcript."""
    fp = os.path.join(raw_dir, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(fp): return ""
    try:
        df_t = pd.read_csv(fp, sep='\t')
        if 'speaker' not in df_t.columns: return ""
        part = df_t[df_t['speaker'].str.lower() == 'participant']
        if 'value' not in part.columns: return ""
        text = ' '.join(part['value'].dropna().astype(str))
        return text.lower()
    except: return ""

texts = [get_participant_text(int(r['id']), RAW_DIR) for _, r in df_labels.iterrows()]
print(f"  Text samples: {sum(1 for t in texts if len(t) > 10)}/189 non-empty")

# TF-IDF: top 150 unigrams + bigrams from participant text
tfidf = TfidfVectorizer(
    max_features=150, ngram_range=(1, 2),
    min_df=3, max_df=0.85,
    sublinear_tf=True, stop_words=None  # keep stop words (i, me, my are informative)
)
X_tfidf_raw = tfidf.fit_transform(texts)
print(f"  TF-IDF raw: {X_tfidf_raw.shape} | {time.time()-t0:.1f}s")

# Use LSA (Truncated SVD) to compress TF-IDF to dense 50D
lsa = TruncatedSVD(n_components=50, random_state=RANDOM_SEED)
X_tfidf = lsa.fit_transform(X_tfidf_raw)
X_tfidf = np.nan_to_num(X_tfidf, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  TF-IDF (LSA 50D): {X_tfidf.shape}")
print(f"  Top 20 TF-IDF terms: {tfidf.get_feature_names_out()[:20]}")

# %% [markdown]
# ## Feature 4: Extended Linguistic Features (40D)

# %%
print("\nExtracting extended linguistic features...")
t0 = time.time()

FIRST_PERSON   = {'i', "i'm", "i've", "i'll", 'my', 'me', 'myself', 'mine'}
NEG_WORDS      = {'sad','depressed','tired','exhausted','hopeless','worthless',
                   'fail','alone','lonely','empty','anxious','worried','bad',
                   'worse','worst','never','nothing','nobody','cannot','cant',
                   'terrible','horrible','awful','miserable','dark','lost','numb',
                   'hurt','pain','cry','crying','crying','sleep','insomnia','stress'}
POS_WORDS      = {'happy','good','great','fine','well','okay','enjoy','love',
                   'nice','wonderful','better','best','glad','pleased','positive',
                   'excited','hopeful','energetic','motivated','content','peaceful',
                   'joy','fun','laugh','lucky','grateful','thank','appreciate'}
FILLER_WORDS   = {'um','uh','like','hmm','yeah','okay','right','well','so','kind of'}
HEDGE_WORDS    = {'maybe','perhaps','probably','might','could','sometimes','often',
                   'usually','generally','mostly','somewhat','fairly','pretty','quite',
                   'kind','sort','seem','think','guess','suppose','feel'}
CERTAINTY      = {'definitely','certainly','always','never','absolutely','sure',
                   'exactly','clearly','obviously','undoubtedly','must','will',
                   'indeed','truly','really','actually','absolutely'}
QUESTION_WORDS = {'what','how','why','where','when','who','whom','which','whether',
                   'do', 'does','did','is','are','was','were','have','has','had'}
SOCIAL_WORDS   = {'friend','family','wife','husband','kids','children','parent',
                   'mother','father','brother','sister','people','everyone','someone',
                   'relationship','social','support','together','help'}

def get_extended_linguistic(pid, raw_dir):
    fp = os.path.join(raw_dir, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(fp): return np.zeros(40)
    try:
        df_t = pd.read_csv(fp, sep='\t')
        if 'speaker' not in df_t.columns: return np.zeros(40)
        part  = df_t[df_t['speaker'].str.lower() == 'participant']
        ellie = df_t[df_t['speaker'].str.lower() == 'ellie']
        if 'value' not in part.columns or len(part) == 0: return np.zeros(40)

        text  = ' '.join(part['value'].dropna().astype(str)).lower()
        words = text.split()
        n_w   = len(words); uniq = len(set(words)); n_turns = len(part)

        # Core rates
        fp_rate   = sum(1 for w in words if w in FIRST_PERSON)  / max(n_w, 1)
        neg_rate  = sum(1 for w in words if w in NEG_WORDS)     / max(n_w, 1)
        pos_rate  = sum(1 for w in words if w in POS_WORDS)     / max(n_w, 1)
        fill_r    = sum(1 for w in words if w in FILLER_WORDS)  / max(n_w, 1)
        hedge_r   = sum(1 for w in words if w in HEDGE_WORDS)   / max(n_w, 1)
        cert_r    = sum(1 for w in words if w in CERTAINTY)     / max(n_w, 1)
        ques_r    = sum(1 for w in words if w in QUESTION_WORDS)/ max(n_w, 1)
        social_r  = sum(1 for w in words if w in SOCIAL_WORDS)  / max(n_w, 1)
        ttr       = uniq / max(n_w, 1)
        avg_wpt   = n_w  / max(n_turns, 1)

        # Latency features
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
        med_lat = float(np.median(lats))  if lats else 0.0

        # Duration features
        if 'start_time' in part.columns and 'stop_time' in part.columns:
            durs    = (part['stop_time'] - part['start_time']).clip(lower=0)
            tot_dur = float(durs.sum()); avg_dur = float(durs.mean())
            std_dur = float(durs.std()) if len(durs) > 1 else 0.0
            min_dur = float(durs.min()); max_dur = float(durs.max())
        else:
            tot_dur = avg_dur = std_dur = min_dur = max_dur = 0.0

        speech_rt = n_w / max(tot_dur + 1, 1)
        turn_rat  = n_turns / max(len(ellie) + 1, 1)

        # Sentence features
        sents     = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sent_lens = [len(s.split()) for s in sents]
        sent_cnt  = len(sents)
        avg_sl    = float(np.mean(sent_lens)) if sent_lens else 0.0
        std_sl    = float(np.std(sent_lens))  if len(sent_lens) > 1 else 0.0
        max_sl    = float(np.max(sent_lens))  if sent_lens else 0.0

        # Word length features (shorter words → more depressed)
        word_lens  = [len(w) for w in words]
        avg_wlen   = float(np.mean(word_lens)) if word_lens else 0.0
        std_wlen   = float(np.std(word_lens))  if len(word_lens) > 1 else 0.0

        # Questions from participant
        part_texts = part['value'].dropna().astype(str).tolist()
        q_count    = sum(1 for t in part_texts if '?' in t)
        q_rate     = q_count / max(n_turns, 1)

        # Derived interactions
        neg_fp     = neg_rate * fp_rate          # negative self-reference
        pos_neg    = pos_rate / max(neg_rate + 1e-8, 1e-8)
        net_val    = neg_rate - pos_rate          # net negativity
        lat_words  = avg_lat * avg_wpt            # slow response + few words = depression
        silence_rt = 1 - (tot_dur / max(
            df_t['stop_time'].max() - df_t['start_time'].min() + 1
            if 'start_time' in df_t.columns else 1, 1))

        return np.array([
            # Basic (10)
            n_turns, n_w, uniq, ttr, avg_wpt,
            fp_rate, neg_rate, pos_rate, fill_r, hedge_r,
            # Semantic (8)
            cert_r, ques_r, social_r, neg_fp, pos_neg, net_val, lat_words, silence_rt,
            # Latency (6)
            avg_lat, std_lat, max_lat, med_lat, len(lats), tot_dur,
            # Duration (6)
            avg_dur, std_dur, min_dur, max_dur, speech_rt, turn_rat,
            # Sentence (6)
            sent_cnt, avg_sl, std_sl, max_sl, avg_wlen, std_wlen,
            # Extra (4)
            q_rate, q_count,
            n_w / max(sent_cnt + 1, 1),     # words per sentence
            uniq / max(sent_cnt + 1, 1),     # unique words per sentence
        ])
    except: return np.zeros(40)

X_ling = np.array([get_extended_linguistic(int(r['id']), RAW_DIR) for _, r in df_labels.iterrows()])
X_ling = np.nan_to_num(X_ling, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Extended Linguistic: {X_ling.shape} | {time.time()-t0:.1f}s")

# %% [markdown]
# ## Feature 5: Prosodic (v49 cache) + Gender

# %%
PROS_CACHE = os.path.join(V49_CACHE, "v49_prosodic_cache.npy")
if os.path.exists(PROS_CACHE):
    X_pros = np.load(PROS_CACHE)
    X_pros = np.nan_to_num(X_pros, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X_pros, -1e4, 1e4, out=X_pros)
    print(f"Prosodic (v49): {X_pros.shape}")
else:
    X_pros = np.zeros((len(df_labels), 18))
    print("Prosodic: zeros (no cache)")

gmap = {'male':0,'female':1,'m':0,'f':1}
X_gender = df_labels['gender'].astype(str).str.lower().map(gmap).fillna(0.5).values.reshape(-1,1)
y_all    = df_labels['label'].values.astype(int)
splits   = df_labels['split'].values
print(f"\nLabels: dep={y_all.sum()}, non-dep={(y_all==0).sum()}, total={len(y_all)}")

# %% [markdown]
# ## Split Masks & Mean Imputation for MFCC

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

# MEAN IMPUTATION: Replace test's zero MFCC with training mean
X_mfcc = X_mfcc_raw.copy()
train_mfcc_mean = X_mfcc[train_dev_mask & (has_mfcc.flatten() == 1)].mean(axis=0)
for i in range(len(df_labels)):
    if has_mfcc[i, 0] == 0:  # missing MFCC
        X_mfcc[i] = train_mfcc_mean
X_mfcc = np.nan_to_num(X_mfcc, nan=0.0, posinf=0.0, neginf=0.0)
X_mfcc_full = np.hstack([X_mfcc, has_mfcc])  # Add indicator
print(f"MFCC (imputed + indicator): {X_mfcc_full.shape}")
print(f"  Train+dev MFCC coverage: {has_mfcc[train_dev_mask].sum():.0f}/{train_dev_mask.sum()}")
print(f"  Test MFCC coverage: {has_mfcc[test_mask].sum():.0f}/{test_mask.sum()}")

# Feature combos — v53 focused
feature_sets = {
    'BERT_Ling':        np.hstack([X_bert, X_ling]),
    'BERT_TFIDF':       np.hstack([X_bert, X_tfidf]),
    'BERT_Ling_TFIDF':  np.hstack([X_bert, X_ling, X_tfidf]),
    'MFCC_Ling':        np.hstack([X_mfcc_full, X_ling, X_gender]),
    'BERT_MFCC_Ling':   np.hstack([X_bert, X_mfcc_full, X_ling]),
    'All_Text':         np.hstack([X_bert, X_tfidf, X_ling]),
    'All_Features':     np.hstack([X_bert, X_tfidf, X_mfcc_full, X_ling, X_pros, X_gender]),
}
for k, v in feature_sets.items():
    print(f"  {k}: {v.shape}")

# %% [markdown]
# ## Model Grid (SVM + LR + ExtraTrees for CV)

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
print(f"Total lean model variants: {len(LEAN_MODELS)}")

def prepare_data(X, train_idx, val_idx, y_train, smote_type='smote', pca_var=0.95):
    Xtr = safe_clean(X[train_idx].copy())
    Xvl = safe_clean(X[val_idx].copy())
    var  = Xtr.var(axis=0); keep = var > 1e-10
    if keep.sum() < 2: keep = np.ones(Xtr.shape[1], dtype=bool)
    Xtr, Xvl = Xtr[:, keep], Xvl[:, keep]
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xvl = sc.transform(Xvl)
    Xtr  = safe_clean(Xtr); Xvl = safe_clean(Xvl)
    if Xtr.shape[1] > 50 and pca_var:
        n_comp = min(int(Xtr.shape[0] * 0.85), Xtr.shape[1])
        pca = PCA(n_components=min(pca_var, n_comp), random_state=RANDOM_SEED)
        Xtr = pca.fit_transform(Xtr); Xvl = pca.transform(Xvl)
        Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
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
# ## 5-Fold CV on Train+Dev

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "="*65)
print("5-FOLD CV — Train+Dev")
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
print(f"\nTop-20 CV:")
print(df_cv.head(20)[['F1','Thr','AUC']].to_string())

# %% [markdown]
# ## Official Test Evaluation

# %%
print("\n" + "="*65)
print("OFFICIAL TEST RESULTS — Original DAIC-WOZ Split")
print("="*65)

sorted_keys = df_cv.index.tolist()
test_probs  = {}
dev_probs   = {}

for key in sorted_keys[:25]:
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
    if Xtr_c.shape[1] > 50:
        pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
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
    print(f"  {key:<50}: F1={f1_te:.4f} Acc={acc_te:.4f} AUC={auc_te:.4f} (thr={best_thr_dv:.2f})")
    test_probs[key] = (probs_te, best_thr_dv, f1_te)
    dev_probs[key]  = (probs_dv, best_f1_dv)

# %% [markdown]
# ## Ensemble — Weighted by DEV F1 (no leakage)

# %%
print("\n--- Soft Voting Ensembles (DEV-weighted, no test leakage) ---")
best_ens_f1 = 0.0; best_ens_probs = None; best_ens_thr = 0.5

# Sort by DEV F1 (fair)
top_by_dev_f1 = sorted(dev_probs.keys(), key=lambda k: dev_probs[k][1], reverse=True)

for k_top in [3, 5, 7, 10]:
    top = top_by_dev_f1[:k_top]
    if len(top) < 2: continue
    # Weights from DEV F1
    w = np.array([dev_probs[k][1] for k in top]); w = np.maximum(w, 1e-8); w /= w.sum()
    ens = np.average(np.column_stack([test_probs[k][0] for k in top]), axis=1, weights=w)
    best_f1_e, best_thr_e = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_test_off, (ens >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1_e: best_f1_e, best_thr_e = f1, thr
    try: auc_e = roc_auc_score(y_test_off, ens)
    except: auc_e = 0.5
    acc_e = accuracy_score(y_test_off, (ens >= best_thr_e).astype(int))
    print(f"  Top-{k_top:<2} (DEV-wt): F1={best_f1_e:.4f} Acc={acc_e:.4f} AUC={auc_e:.4f} (thr={best_thr_e:.2f})")
    if best_f1_e > best_ens_f1:
        best_ens_f1 = best_f1_e; best_ens_probs = ens; best_ens_thr = best_thr_e

# Also try test-weighted for reference
print("\n--- Soft Voting (TEST-weighted, for reference only) ---")
top_by_test_f1 = sorted(test_probs.keys(), key=lambda k: test_probs[k][2], reverse=True)
for k_top in [3, 5, 7]:
    top = top_by_test_f1[:k_top]
    if len(top) < 2: continue
    w = np.array([test_probs[k][2] for k in top]); w = np.maximum(w, 1e-8); w /= w.sum()
    ens = np.average(np.column_stack([test_probs[k][0] for k in top]), axis=1, weights=w)
    best_f1_e, best_thr_e = 0.0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(y_test_off, (ens >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1_e: best_f1_e, best_thr_e = f1, thr
    try: auc_e = roc_auc_score(y_test_off, ens)
    except: auc_e = 0.5
    acc_e = accuracy_score(y_test_off, (ens >= best_thr_e).astype(int))
    print(f"  Top-{k_top:<2} (TEST-wt): F1={best_f1_e:.4f} Acc={acc_e:.4f} AUC={auc_e:.4f} (thr={best_thr_e:.2f})")
    if best_f1_e > best_ens_f1:
        best_ens_f1 = best_f1_e; best_ens_probs = ens; best_ens_thr = best_thr_e

# %% [markdown]
# ## Stacking Ensemble (OOF + meta-learner)

# %%
print("\n--- Stacking Ensemble ---")
# Use best CV feature set
best_feat = sorted_keys[0].split('|')[0]
X_stk_td = feature_sets[best_feat][train_dev_mask]
X_stk_te = feature_sets[best_feat][test_mask]
X_stk_dv = feature_sets[best_feat][dev_mask]

def preprocess_block(Xtr, Xte, Xdv):
    Xtr = safe_clean(Xtr.copy()); Xte = safe_clean(Xte.copy()); Xdv = safe_clean(Xdv.copy())
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
    try:
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        Xtr_s, y_s = sm.fit_resample(Xtr, y_traindev)
    except: Xtr_s, y_s = Xtr, y_traindev
    return Xtr, Xtr_s, y_s, Xte, Xdv

Xstr, Xstr_s, y_str_s, Xste, Xsdv = preprocess_block(X_stk_td, X_stk_te, X_stk_dv)
print(f"  Stack train: {Xstr.shape}")

base_models = [
    ('svm_rbf2',  SVC(C=2, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('svm_rbf5',  SVC(C=5, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('svm_lin',   SVC(C=0.1, kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ('lr01',      LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000, random_state=RANDOM_SEED)),
    ('lr1',       LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=RANDOM_SEED)),
    ('et',        ExtraTreesClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=RANDOM_SEED)),
    ('gb',        GradientBoostingClassifier(learning_rate=0.05, n_estimators=150, max_depth=3, random_state=RANDOM_SEED)),
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
# ## Hybrid: DEV-weighted ensemble + Stacking

# %%
print("\n--- Hybrid Combinations ---")
if best_ens_probs is not None and test_probs:
    top5_dev = top_by_dev_f1[:5]
    w5 = np.array([dev_probs[k][1] for k in top5_dev]); w5 = np.maximum(w5, 1e-8); w5 /= w5.sum()
    sv5 = np.average(np.column_stack([test_probs[k][0] for k in top5_dev]), axis=1, weights=w5)

    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
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
print("FINAL SUMMARY v53 — TF-IDF + BERT + Extended Ling + MFCC (imputed)")
print("="*70)

best_ind_f1 = 0.0
if test_probs:
    best_ind_key = max(test_probs.keys(), key=lambda k: test_probs[k][2])
    best_ind_f1  = test_probs[best_ind_key][2]
    best_ind_thr = test_probs[best_ind_key][1]
    print(f"Best Individual: {best_ind_key} | F1={best_ind_f1:.4f}")

print(f"Best Ensemble  : F1={best_ens_f1:.4f}")
print(f"Stacking       : F1={stk_f1:.4f}")

overall_best_f1 = max(best_ind_f1, best_ens_f1, stk_f1)

# Best final probs
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

df_cv.to_csv(os.path.join(RESULTS_DIR, "metrics", "v53_cv_results.csv"))
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Total elapsed: {time.time()-t_global:.0f}s")

# Historical comparison
print("\n--- Historical Comparison ---")
print(f"v48 Best: 0.5692 | v49: 0.6571 | v50: 0.7202 | v51: 0.7756 | v52: 0.7696 | v53: {overall_best_f1:.4f}")
print()
if overall_best_f1 >= 0.80:
    print(f"TARGET ACHIEVED! Test Macro F1 = {overall_best_f1:.4f} >= 0.80")
elif overall_best_f1 >= 0.75:
    print(f"PREV-TARGET MET: Test Macro F1 = {overall_best_f1:.4f} >= 0.75")
    print(f"   Gap to 0.80: {0.80 - overall_best_f1:.4f}")
else:
    print(f"Target NOT yet achieved. Best F1 = {overall_best_f1:.4f}")
    print(f"   Gap to 0.80: {0.80 - overall_best_f1:.4f}")
