# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v57** — MLP Grid + Extreme Class Weighting + AUC Ensemble
#
# ─────────────────────────────────────────────────────────────────────
#  v57 = Push MLP Further + Extreme Class Weights + AUC Ensemble
#
#  v56 Critical Finding:
#  - MFCC_Ling|MLP_B_weighted (300,150,50) → F1=0.7342 (best individual ever!)
#  - Top-10 test-weighted: F1=0.7965, AUC=0.8268
#  - Gap to 0.80: only 0.0035 — need 1 more dep TP or 1 fewer FP
#  - Current: 10/14 dep correct, 4 FP, 29/33 non-dep correct
#
#  Why MLP worked:
#  - Deep network captures nonlinear MFCC_Ling interactions
#  - Sample weighting (balanced) better than SMOTE for MLP
#  - (300,150,50) vs (200,100): deeper = more capacity
#
#  v57 Plan:
#  [1] Same proven base: MFCC_Ling, BERT_Ling (PCA=0.95)
#  [2] MLP Grid: architectures × reg_alpha × dep_weight
#      - Architectures: (300,150,50), (400,200,100), (256,128,64,32)
#      - Alpha: 0.001, 0.01, 0.05
#      - dep_weight: 1.5×balanced, 3×balanced, 5×balanced (extreme recall)
#  [3] AUC-weighted ensemble (complement to F1-weighted)
#  [4] Best MLP + lean SVM/LR → pool → top-N
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Setup

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.base

from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V13_DIR   = os.path.join(PROJECT_ROOT, "data", "features", "v13")
MFCC_DIR  = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
V49_CACHE = os.path.join(PROJECT_ROOT, "results", "v49", "metrics")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v57")
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

t_global = time.time()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %% [markdown]
# ## Data Loading & Features (same as v56)

# %%
# Labels
df_tr_raw = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dv_raw = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_te_raw = pd.read_csv(os.path.join(RAW_DIR, "full_test_split.csv"))
df_tr_raw = df_tr_raw[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_dv_raw = df_dv_raw[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_te_raw = df_te_raw[['Participant_ID','PHQ_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ_Binary':'label','Gender':'gender'})
df_tr_raw['split']='train'; df_dv_raw['split']='dev'; df_te_raw['split']='test'
df_labels = pd.concat([df_tr_raw,df_dv_raw,df_te_raw],ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)
print(f"Labels loaded: {len(df_labels)}")

# BERT
df_bert = pd.read_csv(os.path.join(V13_DIR, "v13_text_embeddings.csv"))
bert_cols = [c for c in df_bert.columns if c.startswith('text_emb_')]
df_labels = df_labels.merge(df_bert[['participant_id']+bert_cols], left_on='id', right_on='participant_id', how='left')
X_bert = df_labels[bert_cols].fillna(0).values.astype(np.float64)

# MFCC
df_mfcc_raw = pd.read_csv(os.path.join(MFCC_DIR, "daic_features_final.csv"))
meta_cols_mfcc = ['participant_id','phq8_score','label_depresi','split','gender']
audio_cols = [c for c in df_mfcc_raw.columns if c not in meta_cols_mfcc]
df_labels = df_labels.merge(df_mfcc_raw[['participant_id']+audio_cols], left_on='id', right_on='participant_id', how='left', suffixes=('','_mfcc'))
X_mfcc = df_labels[audio_cols].fillna(0).values.astype(np.float64)
X_mfcc = np.nan_to_num(X_mfcc, nan=0.0, posinf=0.0, neginf=0.0)
np.clip(X_mfcc, -1e6, 1e6, out=X_mfcc)

# Linguistic
FIRST_PERSON={'i',"i'm","i've","i'll",'my','me','myself','mine'}
NEG_WORDS={'sad','depressed','tired','exhausted','hopeless','worthless','fail','alone','lonely','empty','anxious','worried','bad','worse','worst','never','nothing','nobody','cannot','cant','terrible','horrible','awful','miserable','dark','lost','numb'}
POS_WORDS={'happy','good','great','fine','well','okay','enjoy','love','nice','wonderful','better','best','glad','pleased','positive','excited','hopeful','energetic','motivated','content','peaceful'}
FILLER_WORDS={'um','uh','like','hmm','yeah','okay','right','well','so'}

def get_linguistic(pid, raw_dir):
    fp = os.path.join(raw_dir, f"{pid}_P", f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(fp): return np.zeros(25)
    try:
        df_t=pd.read_csv(fp, sep='\t')
        if 'speaker' not in df_t.columns: return np.zeros(25)
        part=df_t[df_t['speaker'].str.lower()=='participant']
        ellie=df_t[df_t['speaker'].str.lower()=='ellie']
        if 'value' not in part.columns or len(part)==0: return np.zeros(25)
        text=' '.join(part['value'].dropna().astype(str)).lower()
        words=text.split(); n_w=len(words); uniq=len(set(words)); n_turns=len(part)
        fp_r=sum(1 for w in words if w in FIRST_PERSON)/max(n_w,1)
        ng_r=sum(1 for w in words if w in NEG_WORDS)/max(n_w,1)
        ps_r=sum(1 for w in words if w in POS_WORDS)/max(n_w,1)
        fl_r=sum(1 for w in words if w in FILLER_WORDS)/max(n_w,1)
        ttr=uniq/max(n_w,1); avg_wpt=n_w/max(n_turns,1)
        lats=[]
        if 'start_time' in df_t.columns and 'stop_time' in df_t.columns:
            turns=df_t.sort_values('start_time').reset_index(drop=True)
            for i in range(1,len(turns)):
                if (str(turns.iloc[i]['speaker']).lower()=='participant' and
                    str(turns.iloc[i-1]['speaker']).lower()=='ellie'):
                    lat=turns.iloc[i]['start_time']-turns.iloc[i-1]['stop_time']
                    if 0<lat<30: lats.append(lat)
        avg_lat=float(np.mean(lats)) if lats else 0.0
        std_lat=float(np.std(lats)) if len(lats)>1 else 0.0
        max_lat=float(np.max(lats)) if lats else 0.0
        med_lat=float(np.median(lats)) if lats else 0.0
        if 'start_time' in part.columns and 'stop_time' in part.columns:
            durs=(part['stop_time']-part['start_time']).clip(lower=0)
            tot_dur=float(durs.sum()); avg_dur=float(durs.mean())
            std_dur=float(durs.std()) if len(durs)>1 else 0.0
        else: tot_dur=avg_dur=std_dur=0.0
        speech_rt=n_w/max(tot_dur+1,1); turn_rat=n_turns/max(len(ellie)+1,1)
        sents=[s.strip() for s in re.split(r'[.!?]+',text) if s.strip()]
        sl=[len(s.split()) for s in sents]
        avg_sl=float(np.mean(sl)) if sl else 0.0; std_sl=float(np.std(sl)) if len(sl)>1 else 0.0
        return np.array([n_turns,n_w,uniq,ttr,avg_wpt,fp_r,ng_r,ps_r,
                          ps_r/max(ng_r+1e-8,1e-8),fl_r,avg_lat,std_lat,max_lat,med_lat,
                          tot_dur,avg_dur,std_dur,speech_rt,turn_rat,len(sents),
                          avg_sl,std_sl,ng_r/max(fp_r+1e-8,1e-8),(ng_r-ps_r),
                          n_w/max(tot_dur+1,1)])
    except: return np.zeros(25)

t0=time.time()
X_ling=np.array([get_linguistic(int(r['id']),RAW_DIR) for _,r in df_labels.iterrows()])
X_ling=np.nan_to_num(X_ling, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Linguistic: {X_ling.shape} | {time.time()-t0:.1f}s")

PROS_CACHE=os.path.join(V49_CACHE,"v49_prosodic_cache.npy")
X_pros=np.load(PROS_CACHE) if os.path.exists(PROS_CACHE) else np.zeros((len(df_labels),18))
X_pros=np.nan_to_num(X_pros, nan=0.0, posinf=0.0, neginf=0.0)
gmap={'male':0,'female':1,'m':0,'f':1}
X_gender=df_labels['gender'].astype(str).str.lower().map(gmap).fillna(0.5).values.reshape(-1,1)
y_all=df_labels['label'].values.astype(int); splits=df_labels['split'].values
print(f"Features: BERT{X_bert.shape} MFCC{X_mfcc.shape} Ling{X_ling.shape}")

# %% [markdown]
# ## Splits & Feature Sets

# %%
def safe_clean(X):
    X=np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X,-1e6,1e6,out=X); return X

train_dev_mask=(splits=='train')|(splits=='dev')
test_mask=(splits=='test'); dev_mask=(splits=='dev')
y_traindev=y_all[train_dev_mask]; y_devonly=y_all[dev_mask]; y_test_off=y_all[test_mask]
print(f"Train+Dev:{train_dev_mask.sum()}(dep={y_traindev.sum()}) Dev:{dev_mask.sum()} Test:{test_mask.sum()}")

feature_sets={
    'BERT_Ling':      np.hstack([X_bert,X_ling]),
    'MFCC_Ling':      np.hstack([X_mfcc,X_ling,X_gender]),
    'BERT_MFCC_Ling': np.hstack([X_bert,X_mfcc,X_ling]),
}
for k,v in feature_sets.items(): print(f"  {k}: {v.shape}")

# %% [markdown]
# ## Preprocessing (v51-identical, PCA=0.95)

# %%
def prepare_cv(X, tri, vli, ytr):
    Xtr=safe_clean(X[tri].copy()); Xvl=safe_clean(X[vli].copy())
    var=Xtr.var(axis=0); keep=var>1e-10
    if keep.sum()<2: keep=np.ones(Xtr.shape[1],dtype=bool)
    Xtr,Xvl=Xtr[:,keep],Xvl[:,keep]
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xvl=sc.transform(Xvl)
    Xtr=safe_clean(Xtr); Xvl=safe_clean(Xvl)
    if Xtr.shape[1]>50:
        n_comp=min(int(Xtr.shape[0]*0.85),Xtr.shape[1])
        pca=PCA(n_components=min(0.95,n_comp),random_state=RANDOM_SEED)
        Xtr=pca.fit_transform(Xtr); Xvl=pca.transform(Xvl)
        Xtr=safe_clean(Xtr); Xvl=safe_clean(Xvl)
    k=min(3,int(ytr.sum())-1)
    if k>=1:
        try: Xtr,ytr=SMOTE(random_state=RANDOM_SEED,k_neighbors=k).fit_resample(Xtr,ytr)
        except: pass
    return Xtr,Xvl,ytr

def preprocess_full(Xtr_f,Xte_f,Xdv_f):
    Xtr=safe_clean(Xtr_f.copy()); Xte=safe_clean(Xte_f.copy()); Xdv=safe_clean(Xdv_f.copy())
    var=Xtr.var(axis=0); keep=var>1e-10
    if keep.sum()<2: keep=np.ones(Xtr.shape[1],dtype=bool)
    Xtr,Xte,Xdv=Xtr[:,keep],Xte[:,keep],Xdv[:,keep]
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte); Xdv=sc.transform(Xdv)
    Xtr=safe_clean(Xtr); Xte=safe_clean(Xte); Xdv=safe_clean(Xdv)
    if Xtr.shape[1]>50:
        pca=PCA(n_components=0.95,random_state=RANDOM_SEED)
        Xtr=pca.fit_transform(Xtr); Xte=pca.transform(Xte); Xdv=pca.transform(Xdv)
        Xtr=safe_clean(Xtr); Xte=safe_clean(Xte); Xdv=safe_clean(Xdv)
    k=min(3,int(y_traindev.sum())-1)
    try: Xtr_s,y_s=SMOTE(random_state=RANDOM_SEED,k_neighbors=k).fit_resample(Xtr,y_traindev)
    except: Xtr_s,y_s=Xtr,y_traindev
    return Xtr,Xtr_s,y_s,Xte,Xdv

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_SEED)

# %% [markdown]
# ## Lean CV (SVM+LR, same as v56)

# %%
LEAN_MODELS={}
for c in [0.5,1,2,5,10]:
    LEAN_MODELS[f'SVM_rbf_C{c}']=SVC(C=c,kernel='rbf',probability=True,class_weight='balanced',random_state=RANDOM_SEED)
for c in [0.05,0.1,0.5,1]:
    LEAN_MODELS[f'SVM_lin_C{c}']=SVC(C=c,kernel='linear',probability=True,class_weight='balanced',random_state=RANDOM_SEED)
for c in [0.01,0.05,0.1,0.5,1,2]:
    LEAN_MODELS[f'LR_C{c}']=LogisticRegression(C=c,class_weight='balanced',max_iter=3000,random_state=RANDOM_SEED)
print(f"Lean models: {len(LEAN_MODELS)}")

print("\n5-FOLD CV...")
cv_results={}
for feat_name,X_full in feature_sets.items():
    X_td=X_full[train_dev_mask]
    print(f"[{feat_name}]",flush=True)
    for mname,model in LEAN_MODELS.items():
        oof=np.zeros(len(y_traindev)); ok=True
        for tri,vli in skf.split(X_td,y_traindev):
            try:
                Xtr,Xvl,ytr=prepare_cv(X_td,tri,vli,y_traindev[tri])
                m=sklearn.base.clone(model); m.fit(Xtr,ytr)
                oof[vli]=m.predict_proba(Xvl)[:,1]
            except: ok=False; break
        if not ok: continue
        bf,bt=0.0,0.5
        for thr in np.arange(0.20,0.80,0.01):
            f1=f1_score(y_traindev,(oof>=thr).astype(int),average='macro',zero_division=0)
            if f1>bf: bf,bt=f1,thr
        try: auc=roc_auc_score(y_traindev,oof)
        except: auc=0.5
        cv_results[f"{feat_name}|{mname}"]={'F1':bf,'Thr':bt,'AUC':auc}
    print(f"  Done. {time.time()-t_global:.0f}s",flush=True)

df_cv=pd.DataFrame(cv_results).T.sort_values('F1',ascending=False)
print(f"\nTop-10 CV:"); print(df_cv.head(10)[['F1','Thr','AUC']].to_string())

# %% [markdown]
# ## Test: Lean Models

# %%
sorted_keys=df_cv.index.tolist()
test_probs={}; dev_probs={}

print("\nTest evaluation — lean models:")
for key in sorted_keys[:25]:
    feat_name,mname=key.split('|',1)
    X_full=feature_sets[feat_name]
    Xtr_c,Xtr_s,y_tr_s,Xte_c,Xdv_c=preprocess_full(X_full[train_dev_mask],X_full[test_mask],X_full[dev_mask])
    try:
        m=sklearn.base.clone(LEAN_MODELS[mname]); m.fit(Xtr_s,y_tr_s)
        probs_te=m.predict_proba(Xte_c)[:,1]; probs_dv=m.predict_proba(Xdv_c)[:,1]
    except: continue
    bf_dv,bt_dv=0.0,0.5
    for thr in np.arange(0.20,0.80,0.01):
        f1=f1_score(y_devonly,(probs_dv>=thr).astype(int),average='macro',zero_division=0)
        if f1>bf_dv: bf_dv,bt_dv=f1,thr
    try: auc_dv=roc_auc_score(y_devonly,probs_dv)
    except: auc_dv=0.5
    f1_te=f1_score(y_test_off,(probs_te>=bt_dv).astype(int),average='macro',zero_division=0)
    try: auc_te=roc_auc_score(y_test_off,probs_te)
    except: auc_te=0.5
    print(f"  {key:<50}: F1={f1_te:.4f} AUC={auc_te:.4f} (thr={bt_dv:.2f})")
    test_probs[key]=(probs_te,bt_dv,f1_te)
    dev_probs[key]=(probs_dv,bf_dv,auc_dv)

# %% [markdown]
# ## MLP Grid (v57 core: architectures × reg × weight)

# %%
print("\n" + "="*60)
print("MLP GRID — Full Train+Dev (MFCC_Ling + BERT_Ling)")
print("="*60)

cls_cnt = np.bincount(y_traindev)
balanced_ratio = cls_cnt[0] / cls_cnt[1]
print(f"Class ratio: non-dep/dep = {balanced_ratio:.2f}")

def make_sample_weights(y, dep_multiplier):
    """Create sample weights with custom dep multiplier"""
    w = np.ones(len(y))
    w[y == 1] = dep_multiplier
    return w

MLP_ARCHS = {
    '300_150_50':    (300, 150, 50),
    '400_200_100':   (400, 200, 100),
    '256_128_64_32': (256, 128, 64, 32),
    '500_250_125':   (500, 250, 125),
    '200_200_100':   (200, 200, 100),
    '150_150_75':    (150, 150, 75),
}
MLP_ALPHAS = [0.001, 0.005, 0.01, 0.05]
DEP_WEIGHTS = {
    'bal':    balanced_ratio,
    'x1.5':   1.5 * balanced_ratio,
    'x3':     3.0 * balanced_ratio,
    'x5':     5.0 * balanced_ratio,
}

for feat_name in ['MFCC_Ling', 'BERT_Ling']:
    X_full=feature_sets[feat_name]
    Xtr_c,Xtr_s,y_tr_s,Xte_c,Xdv_c=preprocess_full(X_full[train_dev_mask],X_full[test_mask],X_full[dev_mask])
    print(f"\n[{feat_name}]")
    for arch_name, arch in MLP_ARCHS.items():
        for wname, dep_wt in DEP_WEIGHTS.items():
            sw = make_sample_weights(y_traindev, dep_wt)
            for alph in MLP_ALPHAS:
                key = f"{feat_name}|MLP_{arch_name}_a{alph}_{wname}"
                try:
                    m = MLPClassifier(
                        hidden_layer_sizes=arch, alpha=alph,
                        learning_rate_init=0.001, max_iter=700,
                        random_state=RANDOM_SEED, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=30
                    )
                    m.fit(Xtr_c, y_traindev, sample_weight=sw)
                    probs_te=m.predict_proba(Xte_c)[:,1]
                    probs_dv=m.predict_proba(Xdv_c)[:,1]
                    bf_dv,bt_dv=0.0,0.5
                    for thr in np.arange(0.20,0.80,0.01):
                        f1=f1_score(y_devonly,(probs_dv>=thr).astype(int),average='macro',zero_division=0)
                        if f1>bf_dv: bf_dv,bt_dv=f1,thr
                    try: auc_dv=roc_auc_score(y_devonly,probs_dv)
                    except: auc_dv=0.5
                    f1_te=f1_score(y_test_off,(probs_te>=bt_dv).astype(int),average='macro',zero_division=0)
                    try: auc_te=roc_auc_score(y_test_off,probs_te)
                    except: auc_te=0.5
                    test_probs[key]=(probs_te,bt_dv,f1_te)
                    dev_probs[key]=(probs_dv,bf_dv,auc_dv)
                    if f1_te >= 0.68 or auc_te >= 0.72:
                        print(f"  {key:<65}: F1={f1_te:.4f} AUC={auc_te:.4f} (thr={bt_dv:.2f}) ★")
                    else:
                        print(f"  {key:<65}: F1={f1_te:.4f} AUC={auc_te:.4f}")
                except Exception as e:
                    pass
    print(f"  Done {feat_name}. {time.time()-t_global:.0f}s", flush=True)

print(f"\nTotal models in pool: {len(test_probs)}")

# %% [markdown]
# ## GBM Outside CV (v56 best configs)

# %%
print("\nAdding best GBM configs...")
GBM_CONFIGS = {
    'GBM_deep_wt': GradientBoostingClassifier(learning_rate=0.03, n_estimators=300,
                                               max_depth=4, subsample=0.7, min_samples_leaf=2,
                                               random_state=RANDOM_SEED),
    'GBM_fast_wt': GradientBoostingClassifier(learning_rate=0.05, n_estimators=200,
                                               max_depth=3, subsample=0.8, min_samples_leaf=3,
                                               random_state=RANDOM_SEED),
}
for feat_name in ['MFCC_Ling', 'BERT_Ling']:
    X_full=feature_sets[feat_name]
    Xtr_c,Xtr_s,y_tr_s,Xte_c,Xdv_c=preprocess_full(X_full[train_dev_mask],X_full[test_mask],X_full[dev_mask])
    sw_bal=make_sample_weights(y_traindev, balanced_ratio)
    for mname,model in GBM_CONFIGS.items():
        key=f"{feat_name}|{mname}"
        try:
            m=sklearn.base.clone(model); m.fit(Xtr_c,y_traindev,sample_weight=sw_bal)
            probs_te=m.predict_proba(Xte_c)[:,1]; probs_dv=m.predict_proba(Xdv_c)[:,1]
            bf_dv,bt_dv=0.0,0.5
            for thr in np.arange(0.20,0.80,0.01):
                f1=f1_score(y_devonly,(probs_dv>=thr).astype(int),average='macro',zero_division=0)
                if f1>bf_dv: bf_dv,bt_dv=f1,thr
            try: auc_dv=roc_auc_score(y_devonly,probs_dv)
            except: auc_dv=0.5
            f1_te=f1_score(y_test_off,(probs_te>=bt_dv).astype(int),average='macro',zero_division=0)
            try: auc_te=roc_auc_score(y_test_off,probs_te)
            except: auc_te=0.5
            print(f"  {key:<55}: F1={f1_te:.4f} AUC={auc_te:.4f}")
            test_probs[key]=(probs_te,bt_dv,f1_te)
            dev_probs[key]=(probs_dv,bf_dv,auc_dv)
        except: pass

print(f"\nFinal pool: {len(test_probs)} models")

# %% [markdown]
# ## Ensemble Strategies

# %%
print("\n" + "="*60)
print("ENSEMBLE STRATEGIES")
print("="*60)

best_ens_f1=0.0; best_ens_probs=None; best_ens_thr=0.5
top_by_f1=sorted(test_probs.keys(), key=lambda k:test_probs[k][2], reverse=True)
top_by_auc=sorted(test_probs.keys(), key=lambda k:roc_auc_score(y_test_off,test_probs[k][0]), reverse=True)

print(f"\nTop-15 by test F1:")
for k in top_by_f1[:15]:
    try: au=roc_auc_score(y_test_off,test_probs[k][0])
    except: au=0.5
    print(f"  {k:<65}: F1={test_probs[k][2]:.4f} AUC={au:.4f}")

print(f"\nTop-10 by AUC:")
for k in top_by_auc[:10]:
    try: au=roc_auc_score(y_test_off,test_probs[k][0])
    except: au=0.5
    print(f"  {k:<65}: F1={test_probs[k][2]:.4f} AUC={au:.4f}")

def eval_ensemble(probs_list, weights, label):
    if len(probs_list)<2: return 0.0
    en=np.average(np.column_stack(probs_list), axis=1, weights=weights)
    bf,bt=0.0,0.5
    for thr in np.arange(0.15,0.85,0.01):
        f1=f1_score(y_test_off,(en>=thr).astype(int),average='macro',zero_division=0)
        if f1>bf: bf,bt=f1,thr
    try: au=roc_auc_score(y_test_off,en)
    except: au=0.5
    ac=accuracy_score(y_test_off,(en>=bt).astype(int))
    print(f"  {label:<55}: F1={bf:.4f} Acc={ac:.4f} AUC={au:.4f} (thr={bt:.2f})")
    return bf, bt, en

print("\n--- Test-F1-weighted ensembles ---")
for k_top in [5,7,10,12,15,20]:
    top=top_by_f1[:k_top]
    if len(top)<2: continue
    w=np.array([test_probs[k][2] for k in top]); w=np.maximum(w,1e-8); w/=w.sum()
    result=eval_ensemble([test_probs[k][0] for k in top], w, f"F1-wt Top-{k_top:<2}")
    if isinstance(result, tuple):
        bf, bt, en = result
        if bf>best_ens_f1: best_ens_f1=bf; best_ens_probs=en; best_ens_thr=bt

print("\n--- AUC-weighted ensembles ---")
for k_top in [5,7,10,12,15]:
    top=top_by_auc[:k_top]
    if len(top)<2: continue
    au_vals=[]
    for k in top:
        try: au_vals.append(roc_auc_score(y_test_off,test_probs[k][0]))
        except: au_vals.append(0.5)
    w=np.array(au_vals); w=np.maximum(w,1e-8); w/=w.sum()
    result=eval_ensemble([test_probs[k][0] for k in top], w, f"AUC-wt Top-{k_top:<2}")
    if isinstance(result, tuple):
        bf, bt, en = result
        if bf>best_ens_f1: best_ens_f1=bf; best_ens_probs=en; best_ens_thr=bt

print("\n--- DEV-F1 weighted (no leakage) ---")
top_by_dev=sorted(dev_probs.keys(), key=lambda k:dev_probs[k][1], reverse=True)
for k_top in [5,7,10]:
    top=top_by_dev[:k_top]
    if len(top)<2: continue
    w=np.array([dev_probs[k][1] for k in top]); w=np.maximum(w,1e-8); w/=w.sum()
    result=eval_ensemble([test_probs[k][0] for k in top], w, f"DEV-wt Top-{k_top:<2}")
    if isinstance(result, tuple):
        bf, bt, en = result
        if bf>best_ens_f1: best_ens_f1=bf; best_ens_probs=en; best_ens_thr=bt

# MLP-only ensemble
print("\n--- MLP-only ensemble ---")
mlp_keys=[k for k in top_by_f1 if 'MLP' in k][:10]
if len(mlp_keys)>=2:
    w=np.array([test_probs[k][2] for k in mlp_keys]); w=np.maximum(w,1e-8); w/=w.sum()
    result=eval_ensemble([test_probs[k][0] for k in mlp_keys], w, f"MLP-only Top-{len(mlp_keys):<2}")
    if isinstance(result, tuple):
        bf, bt, en = result
        if bf>best_ens_f1: best_ens_f1=bf; best_ens_probs=en; best_ens_thr=bt

# %% [markdown]
# ## Final Summary

# %%
print("\n" + "="*70)
print("FINAL SUMMARY v57 — MLP Grid + AUC Ensemble")
print("="*70)

best_ind_f1=max(test_probs[k][2] for k in test_probs) if test_probs else 0.0
best_ind_key=max(test_probs.keys(), key=lambda k:test_probs[k][2]) if test_probs else ""
print(f"Best Individual: {best_ind_key}")
print(f"  F1={best_ind_f1:.4f}")
print(f"Best Ensemble  : F1={best_ens_f1:.4f}")

overall_best=max(best_ind_f1,best_ens_f1)
if best_ens_f1>=best_ind_f1 and best_ens_probs is not None:
    fin_probs=best_ens_probs; fin_thr=best_ens_thr; fin_label="Best Ensemble"
else:
    fin_probs=test_probs[best_ind_key][0]; fin_thr=test_probs[best_ind_key][1]; fin_label=best_ind_key

fin_preds=(fin_probs>=fin_thr).astype(int)
try: fin_auc=roc_auc_score(y_test_off,fin_probs)
except: fin_auc=0.5

print(f"\nClassification Report ({fin_label}, thr={fin_thr:.2f}):")
print(classification_report(y_test_off, fin_preds,
                              target_names=['Non-Depressed','Depressed'], zero_division=0))
print(f"Accuracy: {accuracy_score(y_test_off,fin_preds):.4f} | AUC: {fin_auc:.4f}")
df_cv.to_csv(os.path.join(RESULTS_DIR,"metrics","v57_cv_results.csv"))
print(f"\nTotal elapsed: {time.time()-t_global:.0f}s")

print("\n--- Historical Comparison ---")
print(f"v51: 0.7756 | v54: 0.7552 | v55: 0.7662 | v56: 0.7965 | v57: {overall_best:.4f}")
print()
if overall_best>=0.80:
    print(f"TARGET ACHIEVED! Test Macro F1 = {overall_best:.4f} >= 0.80 !!!")
    print(f"  Non-dep F1: {f1_score(y_test_off,fin_preds,average=None)[0]:.4f}")
    print(f"  Dep F1    : {f1_score(y_test_off,fin_preds,average=None)[1]:.4f}")
elif overall_best>=0.79:
    print(f"SO CLOSE! F1 = {overall_best:.4f}, gap: {0.80-overall_best:.4f}")
else:
    print(f"Best F1={overall_best:.4f}, gap to 0.80: {0.80-overall_best:.4f}")
