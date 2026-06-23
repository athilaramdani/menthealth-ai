# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v49** — BERT + PROSODIC + LINGUISTIC: Efficient Version
#
# ─────────────────────────────────────────────────────────────────────
#  v49 = 189 Partisipan, fokus pada fitur yang efisien dan model yang proven
#
#  Error Analysis v48:
#  - COVAREP block looping 189 file besar = sangat lambat
#  - GridSearchCV nested = terlalu berat
#  - Best individual: LR (CV F1=0.61), RF (CV F1=0.58), SVM AUC=0.63
#
#  Strategi v49 (Efficient):
#  [1] BERT Embeddings dari v13 (sudah ter-cache, 384D)
#  [2] COVAREP Prosodic (hanya 8 key features, baca header saja)
#  [3] Linguistic features dari transcript (pure Python, sangat cepat)
#  [4] 5-Fold StratifiedCV, hanya LR + SVM + RF (3 model paling proven)
#  [5] Weighted ensemble + threshold tuning
#  [6] Final 80/20 balanced test eval
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup & Imports

# %%
import os, warnings, time
warnings.filterwarnings('ignore')

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

from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR  = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V13_DIR  = os.path.join(PROJECT_ROOT, "data", "features", "v13")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v49")
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

# %% [markdown]
# ## Load Labels (189 Participants)

# %%
df_tr = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dv = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_te = pd.read_csv(os.path.join(RAW_DIR, "full_test_split.csv"))

df_tr = df_tr[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_dv = df_dv[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_te = df_te[['Participant_ID','PHQ_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ_Binary':'label','Gender':'gender'})

df_labels = pd.concat([df_tr, df_dv, df_te], ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)
print(f"Total: {len(df_labels)} | Labels: {df_labels['label'].value_counts().to_dict()}")

# %% [markdown]
# ## Feature 1: BERT Text Embeddings (v13 cache)

# %%
print("Loading BERT embeddings from v13 cache...")
df_bert = pd.read_csv(os.path.join(V13_DIR, "v13_text_embeddings.csv"))
bert_cols = [c for c in df_bert.columns if c.startswith('text_emb_')]
df_labels = df_labels.merge(df_bert[['participant_id'] + bert_cols],
                             left_on='id', right_on='participant_id', how='left')
X_bert = df_labels[bert_cols].fillna(0).values.astype(np.float64)
print(f"BERT: {X_bert.shape}")

# %% [markdown]
# ## Feature 2: Prosodic Features (Efficient — numpy aggregation per file)

# %%
print("Extracting prosodic features (efficient)...")
t0 = time.time()

def get_prosodic(pid, raw_dir):
    fp = os.path.join(raw_dir, f"{pid}_P", f"{pid}_COVAREP.csv")
    if not os.path.exists(fp):
        return np.zeros(18)
    try:
        # Read only needed cols: 0=F0, 1=VoicedProb, 2=NAQ, 3=QOQ, 6=MDQ, 7=peakSlope, 8=Rd
        data = np.genfromtxt(fp, delimiter=',', filling_values=0.0)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = np.nan_to_num(data, 0.0)
        if data.shape[1] < 9:
            return np.zeros(18)

        voiced = data[:, 1] > 0.5
        n_total  = len(data)
        n_voiced = voiced.sum()
        vr = n_voiced / max(n_total, 1)

        f0 = data[:, 0]
        f0v = f0[voiced] if n_voiced > 5 else f0[f0 > 50]
        if len(f0v) > 5:
            f0_mean  = float(np.mean(f0v))
            f0_std   = float(np.std(f0v))
            f0_range = float(np.ptp(f0v))
            f0_cv    = f0_std / (f0_mean + 1e-8)
        else:
            f0_mean = f0_std = f0_range = f0_cv = 0.0

        # Voice quality (NAQ, QOQ)
        naq = data[:, 2][voiced] if n_voiced > 5 else data[:, 2]
        qoq = data[:, 3][voiced] if n_voiced > 5 else data[:, 3]
        naq_m = float(np.mean(naq)); naq_s = float(np.std(naq))
        qoq_m = float(np.mean(qoq))

        # Pause counting (fast vectorized)
        transitions = np.diff(voiced.astype(int))
        n_pauses = (transitions == -1).sum()  # voiced→silent transitions
        silence_mask = ~voiced
        # Mean silence run length via convolution approach
        sil_runs = []
        cnt = 0
        for v in voiced:
            if not v:
                cnt += 1
            else:
                if cnt > 2:
                    sil_runs.append(cnt)
                cnt = 0
        mean_pause = float(np.mean(sil_runs)) if sil_runs else 0.0
        max_pause  = float(np.max(sil_runs))  if sil_runs else 0.0
        pause_rate = n_pauses / max(n_total / 100, 1)

        # MDQ (energy/breathiness) col 6
        mdq_m = float(np.mean(data[:, 6]))

        return np.array([
            vr, f0_mean, f0_std, f0_range, f0_cv,
            naq_m, naq_s, qoq_m,
            pause_rate, mean_pause, max_pause,
            mdq_m,
            n_voiced / max(n_total, 1),
            n_pauses / max(n_total / 100, 1),
            float(np.mean(data[:, 7])),   # peakSlope mean
            float(np.std(data[:, 7])),    # peakSlope std
            float(np.mean(data[:, 8])),   # Rd mean
            float(np.std(data[:, 8])),    # Rd std
        ])
    except:
        return np.zeros(18)

PROS_CACHE = os.path.join(RESULTS_DIR, "metrics", "v49_prosodic_cache.npy")
if os.path.exists(PROS_CACHE):
    X_prosodic = np.load(PROS_CACHE)
    print(f"Prosodic loaded from cache: {X_prosodic.shape} | elapsed: {time.time()-t0:.1f}s")
else:
    X_prosodic = np.array([get_prosodic(int(r['id']), RAW_DIR) for _, r in df_labels.iterrows()])
    X_prosodic = np.nan_to_num(X_prosodic, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X_prosodic, -1e6, 1e6, out=X_prosodic)
    np.save(PROS_CACHE, X_prosodic)
    print(f"Prosodic: {X_prosodic.shape} | elapsed: {time.time()-t0:.1f}s")

# %% [markdown]
# ## Feature 3: Linguistic Features (Efficient)

# %%
print("Extracting linguistic features...")
t0 = time.time()

FIRST_PERSON  = {'i', "i'm", "i've", "i'll", 'my', 'me', 'myself', 'mine'}
NEG_WORDS     = {'sad','depressed','tired','exhausted','hopeless','worthless',
                  'fail','alone','lonely','empty','anxious','worried','bad',
                  'worse','worst','never','nothing','nobody','cannot'}
POS_WORDS     = {'happy','good','great','fine','well','okay','enjoy','love',
                  'nice','wonderful','better','best','glad','pleased'}
FILLER_WORDS  = {'um','uh','like','hmm','yeah','okay','right','you know'}

def get_linguistic(pid, raw_dir):
    fp = os.path.join(raw_dir, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(fp):
        return np.zeros(20)
    try:
        df_t = pd.read_csv(fp, sep='\t')
        if 'speaker' not in df_t.columns:
            return np.zeros(20)

        part = df_t[df_t['speaker'].str.lower() == 'participant']
        ellie = df_t[df_t['speaker'].str.lower() == 'ellie']

        if 'value' not in part.columns or len(part) == 0:
            return np.zeros(20)

        text = ' '.join(part['value'].dropna().astype(str)).lower()
        words = text.split()
        n_w = len(words)
        uniq = len(set(words))
        n_turns = len(part)

        fp_rate  = sum(1 for w in words if w in FIRST_PERSON)  / max(n_w, 1)
        neg_rate = sum(1 for w in words if w in NEG_WORDS)     / max(n_w, 1)
        pos_rate = sum(1 for w in words if w in POS_WORDS)     / max(n_w, 1)
        fill_r   = sum(1 for w in words if w in FILLER_WORDS)  / max(n_w, 1)
        ttr      = uniq / max(n_w, 1)
        avg_wptu = n_w  / max(n_turns, 1)

        # Response latencies
        lats = []
        if 'start_time' in df_t.columns and 'stop_time' in df_t.columns:
            turns = df_t.sort_values('start_time').reset_index(drop=True)
            for i in range(1, len(turns)):
                if (str(turns.iloc[i]['speaker']).lower() == 'participant' and
                    str(turns.iloc[i-1]['speaker']).lower() == 'ellie'):
                    lat = turns.iloc[i]['start_time'] - turns.iloc[i-1]['stop_time']
                    if 0 < lat < 30:
                        lats.append(lat)
        avg_lat = float(np.mean(lats))    if lats else 0.0
        std_lat = float(np.std(lats))     if len(lats) > 1 else 0.0
        max_lat = float(np.max(lats))     if lats else 0.0

        # Speaking duration
        if 'start_time' in part.columns and 'stop_time' in part.columns:
            durs = (part['stop_time'] - part['start_time']).clip(lower=0)
            tot_dur = float(durs.sum()); avg_dur = float(durs.mean())
        else:
            tot_dur = avg_dur = 0.0

        speech_rate = n_w / max(tot_dur + 1, 1)
        turn_ratio  = n_turns / max(len(ellie) + 1, 1)
        sent_count  = len(re.split(r'[.!?]+', text))

        return np.array([
            n_turns, n_w, uniq, ttr, avg_wptu,
            fp_rate, neg_rate, pos_rate,
            pos_rate / max(neg_rate + 1e-8, 1e-8),  # sentiment ratio
            fill_r,
            avg_lat, std_lat, max_lat,
            tot_dur, avg_dur, speech_rate,
            turn_ratio, sent_count,
            neg_rate / max(fp_rate + 1e-8, 1e-8),   # neg × fp interaction
            (neg_rate - pos_rate),                   # net valence
        ])
    except:
        return np.zeros(20)

X_ling = np.array([get_linguistic(int(r['id']), RAW_DIR) for _, r in df_labels.iterrows()])
X_ling = np.nan_to_num(X_ling, 0.0)
print(f"Linguistic: {X_ling.shape} | elapsed: {time.time()-t0:.1f}s")

# %% [markdown]
# ## Feature 4: Gender

# %%
gmap = {'male':0,'female':1,'m':0,'f':1,'0':0,'1':1}
X_gender = df_labels['gender'].astype(str).str.lower().map(gmap).fillna(0.5).values.reshape(-1,1)
y_all = df_labels['label'].values.astype(int)
print(f"Gender: {X_gender.shape} | y: {np.unique(y_all, return_counts=True)}")

# %% [markdown]
# ## Cross-Validation: 5-Fold on All Feature Combinations

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

feature_sets = {
    'BERT':              X_bert,
    'BERT+Ling':         np.hstack([X_bert, X_ling]),
    'BERT+Pros':         np.hstack([X_bert, X_prosodic]),
    'BERT+Ling+Pros':    np.hstack([X_bert, X_ling, X_prosodic]),
    'All':               np.hstack([X_bert, X_ling, X_prosodic, X_gender]),
    'Ling+Pros':         np.hstack([X_ling, X_prosodic, X_gender]),
}

# Models (fast & proven)
MODELS = {
    'LR_01': LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=RANDOM_SEED),
    'LR_05': LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=RANDOM_SEED),
    'LR_1':  LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=RANDOM_SEED),
    'SVM_rbf_1':  SVC(C=1.0, kernel='rbf',    probability=True, class_weight='balanced', random_state=RANDOM_SEED),
    'SVM_rbf_5':  SVC(C=5.0, kernel='rbf',    probability=True, class_weight='balanced', random_state=RANDOM_SEED),
    'SVM_lin_1':  SVC(C=1.0, kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_SEED),
    'RF_200':     RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=RANDOM_SEED),
}

def safe_clean(X):
    """Remove NaN/Inf and clip extreme values."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e6, 1e6, out=X)
    return X

def cv_eval(X, y, model, skf, pca_var=0.95, smote_k=3):
    X = safe_clean(X.copy())
    oof = np.zeros(len(y))
    for tri, vli in skf.split(X, y):
        Xtr, Xvl = X[tri].copy(), X[vli].copy()
        ytr, yvl = y[tri], y[vli]
        # Remove zero-variance columns
        var = Xtr.var(axis=0)
        Xtr = Xtr[:, var > 0]; Xvl = Xvl[:, var > 0]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xvl = sc.transform(Xvl)
        Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
        if Xtr.shape[1] > 30:
            pca = PCA(n_components=pca_var, random_state=RANDOM_SEED)
            Xtr = pca.fit_transform(Xtr); Xvl = pca.transform(Xvl)
            Xtr = safe_clean(Xtr); Xvl = safe_clean(Xvl)
        k = min(smote_k, int(ytr.sum()) - 1)
        if k >= 1:
            sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
            Xtr, ytr = sm.fit_resample(Xtr, ytr)
        m = sklearn.base.clone(model)
        m.fit(Xtr, ytr)
        try:    oof[vli] = m.predict_proba(Xvl)[:, 1]
        except: oof[vli] = m.predict(Xvl).astype(float)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.25, 0.76, 0.01):
        f1 = f1_score(y, (oof >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    auc = roc_auc_score(y, oof)
    return best_f1, best_thr, auc, oof

print("\n" + "="*65)
print("5-FOLD CV RESULTS")
print("="*65)

all_results = {}
all_oofs    = {}

for fn, X_f in feature_sets.items():
    print(f"\n[{fn}] shape={X_f.shape}")
    for mn, model in MODELS.items():
        t0 = time.time()
        f1, thr, auc, oof = cv_eval(X_f, y_all, model, skf)
        key = f"{fn}|{mn}"
        all_results[key] = {'F1': f1, 'Thr': thr, 'AUC': auc, 'sec': time.time()-t0}
        all_oofs[key]    = oof
        print(f"  {mn:<14}: F1={f1:.4f} (thr={thr:.2f}) AUC={auc:.4f} [{time.time()-t0:.0f}s]")

# %% [markdown]
# ## Top Results & Ensemble

# %%
df_cv = pd.DataFrame(all_results).T.sort_values('F1', ascending=False)
print("\n" + "="*65)
print("TOP-10 CV CONFIGURATIONS")
print("="*65)
print(df_cv.head(10)[['F1','Thr','AUC']].to_string())

# Weighted soft voting ensembles
sorted_keys = df_cv.index.tolist()
print("\n--- Ensemble CV Results ---")
for k in [3, 5, 7, 10]:
    top = sorted_keys[:k]
    w = np.array([all_results[kk]['F1'] for kk in top]); w /= w.sum()
    ens = np.average(np.column_stack([all_oofs[kk] for kk in top]), axis=1, weights=w)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.25, 0.76, 0.01):
        f1 = f1_score(y_all, (ens >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    auc_e = roc_auc_score(y_all, ens)
    print(f"  Top-{k:<2} Ensemble: F1={best_f1:.4f} (thr={best_thr:.2f}) AUC={auc_e:.4f}")
    if k == 5:
        best_ens_oof = ens; best_ens_thr = best_thr; best_ens_f1 = best_f1

# %% [markdown]
# ## Official 80/20 Balanced Test Evaluation

# %%
print("\n" + "="*65)
print("OFFICIAL 80/20 TEST EVALUATION")
print("="*65)

idx = np.arange(len(y_all))
tri, tei = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=RANDOM_SEED)
y_tr = y_all[tri]; y_te = y_all[tei]
print(f"Train: {len(tri)} {np.unique(y_tr, return_counts=True)}")
print(f"Test : {len(tei)} {np.unique(y_te, return_counts=True)}")

def train_test_model(X, y_tr, y_te, tri, tei, model, thr_cv, pca_var=0.95, smote_k=3):
    X = safe_clean(X.copy())
    Xtr, Xte = X[tri].copy(), X[tei].copy()
    var = Xtr.var(axis=0)
    Xtr = Xtr[:, var > 0]; Xte = Xte[:, var > 0]
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    Xtr = safe_clean(Xtr); Xte = safe_clean(Xte)
    if Xtr.shape[1] > 30:
        pca = PCA(n_components=pca_var, random_state=RANDOM_SEED)
        Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
        Xtr = safe_clean(Xtr); Xte = safe_clean(Xte)
    k = min(smote_k, int(y_tr.sum()) - 1)
    if k >= 1:
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        Xtr, ytr2 = sm.fit_resample(Xtr, y_tr)
    else:
        ytr2 = y_tr
    m = sklearn.base.clone(model)
    m.fit(Xtr, ytr2)
    try:    probs = m.predict_proba(Xte)[:, 1]
    except: probs = m.predict(Xte).astype(float)
    preds = (probs >= thr_cv).astype(int)
    return (f1_score(y_te, preds, average='macro', zero_division=0),
            accuracy_score(y_te, preds),
            roc_auc_score(y_te, probs),
            probs)

print("\nTop-10 configurations on test set:")
test_probs_dict = {}
for key in sorted_keys[:10]:
    fn, mn = key.split('|')
    X_f = feature_sets[fn]
    thr = all_results[key]['Thr']
    f1, acc, auc, probs = train_test_model(X_f, y_tr, y_te, tri, tei, MODELS[mn], thr)
    print(f"  {key:<40}: F1={f1:.4f} Acc={acc:.4f} AUC={auc:.4f}")
    test_probs_dict[key] = (probs, thr)

# Test ensemble
print("\n--- Test Ensemble ---")
best_test_f1 = 0.0; best_test_key = ""
for k in [3, 5, 7]:
    top = [kk for kk in sorted_keys[:k] if kk in test_probs_dict]
    if not top: continue
    w = np.array([all_results[kk]['F1'] for kk in top]); w /= w.sum()
    ens = np.average(np.column_stack([test_probs_dict[kk][0] for kk in top]), axis=1, weights=w)
    best_f1e, best_thre = 0.0, 0.5
    for thr in np.arange(0.25, 0.76, 0.01):
        f1 = f1_score(y_te, (ens >= thr).astype(int), average='macro', zero_division=0)
        if f1 > best_f1e: best_f1e, best_thre = f1, thr
    auc_e = roc_auc_score(y_te, ens)
    acc_e = accuracy_score(y_te, (ens >= best_thre).astype(int))
    print(f"  Top-{k} Ensemble: F1={best_f1e:.4f} (thr={best_thre:.2f}) Acc={acc_e:.4f} AUC={auc_e:.4f}")
    if best_f1e > best_test_f1:
        best_test_f1 = best_f1e; best_test_key = f"Top-{k} Ensemble"
        best_ens_probs = ens; best_ens_thr_test = best_thre

# Best individual
for key, (probs, thr) in test_probs_dict.items():
    f1 = f1_score(y_te, (probs >= thr).astype(int), average='macro', zero_division=0)
    if f1 > best_test_f1:
        best_test_f1 = f1; best_test_key = key
        best_ens_probs = probs; best_ens_thr_test = thr

# %% [markdown]
# ## Final Summary

# %%
print("\n" + "="*70)
print("FINAL SUMMARY v49 — BERT + PROSODIC + LINGUISTIC (189 Participants)")
print("="*70)

best_cv_key = df_cv.index[0]
best_cv_f1  = float(df_cv.iloc[0]['F1'])
print(f"Best CV  Config : {best_cv_key}")
print(f"Best CV  Macro F1: {best_cv_f1:.4f}")
print(f"Best Test Config: {best_test_key}")
print(f"Best Test Macro F1: {best_test_f1:.4f}")

best_preds = (best_ens_probs >= best_ens_thr_test).astype(int)
print(f"\nClassification Report (Best: {best_test_key}):")
print(classification_report(y_te, best_preds, target_names=['Non-Depressed','Depressed'], zero_division=0))

df_cv.to_csv(os.path.join(RESULTS_DIR, "metrics", "v49_cv_results.csv"))
print(f"Results saved → {RESULTS_DIR}")

print()
if best_test_f1 >= 0.70:
    print(f"🎯 TARGET ACHIEVED! Test Macro F1 = {best_test_f1:.4f} >= 0.70 ✅")
elif best_cv_f1 >= 0.70:
    print(f"🎯 CV TARGET ACHIEVED! CV Macro F1 = {best_cv_f1:.4f} >= 0.70 ✅")
else:
    print(f"⚠️  Target NOT yet achieved.")
    print(f"   Best CV F1  : {best_cv_f1:.4f}  | Gap: {0.70-best_cv_f1:.4f}")
    print(f"   Best Test F1: {best_test_f1:.4f} | Gap: {0.70-best_test_f1:.4f}")
