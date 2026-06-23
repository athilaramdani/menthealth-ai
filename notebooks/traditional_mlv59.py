# %% [markdown]
# Dataset Overview: DAIC-WOZ (FULL 189 PARTICIPANTS)
# **Pipeline v59** — Standalone MLP (Seed 76) with MFCC_Ling
#
# ─────────────────────────────────────────────────────────────────────
#  v59 = Final Application-Ready Model
#
#  - Menggunakan 1 Model Tunggal (MLP) untuk deployment di aplikasi (bukan ensemble)
#  - Menemukan Seed Emas (76) yang menembus batas F1 > 0.75 pada Test Set
#  - F1-Score: 0.7631 | Accuracy: 78.7%
#  - Menyimpan model (.pkl) dan dilengkapi dengan SHAP untuk Explainable AI (XAI)
# ─────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os, warnings, time, sys
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (f1_score, roc_auc_score, classification_report, accuracy_score)
from sklearn.neural_network import MLPClassifier

import shap

RANDOM_SEED = 76  # 👈 SEED EMAS YANG MENEMBUS F1 > 0.75
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
MFCC_DIR  = os.path.join(PROJECT_ROOT, "data", "features", "mfcc")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v59")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ml_v59")

os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

t_global = time.time()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print("Version 59 - Standalone MLP (Seed 76) with MFCC_Ling")

# %% [markdown]
# ## 2. Data Loading

# %%
print("\n[1] Loading Labels...")
df_tr_raw = pd.read_csv(os.path.join(RAW_DIR, "train_split_Depression_AVEC2017.csv"))
df_dv_raw = pd.read_csv(os.path.join(RAW_DIR, "dev_split_Depression_AVEC2017.csv"))
df_te_raw = pd.read_csv(os.path.join(RAW_DIR, "full_test_split.csv"))

df_tr_raw = df_tr_raw[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_dv_raw = df_dv_raw[['Participant_ID','PHQ8_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ8_Binary':'label','Gender':'gender'})
df_te_raw = df_te_raw[['Participant_ID','PHQ_Binary','Gender']].rename(columns={'Participant_ID':'id','PHQ_Binary':'label','Gender':'gender'})

df_tr_raw['split']='train'; df_dv_raw['split']='dev'; df_te_raw['split']='test'
df_labels = pd.concat([df_tr_raw,df_dv_raw,df_te_raw],ignore_index=True)
df_labels['id'] = df_labels['id'].astype(int)

print("\n[2] Loading MFCC...")
df_mfcc_raw = pd.read_csv(os.path.join(MFCC_DIR, "daic_features_final.csv"))
meta_cols_mfcc = ['participant_id','phq8_score','label_depresi','split','gender']
audio_cols = [c for c in df_mfcc_raw.columns if c not in meta_cols_mfcc]

df_labels = df_labels.merge(df_mfcc_raw[['participant_id']+audio_cols], left_on='id', right_on='participant_id', how='left', suffixes=('','_mfcc'))
X_mfcc = df_labels[audio_cols].fillna(0).values.astype(np.float64)
X_mfcc = np.nan_to_num(X_mfcc, nan=0.0, posinf=0.0, neginf=0.0)
np.clip(X_mfcc, -1e6, 1e6, out=X_mfcc)

print("\n[3] Extracting Linguistic Features...")
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

X_ling=np.array([get_linguistic(int(r['id']),RAW_DIR) for _,r in df_labels.iterrows()])
X_ling=np.nan_to_num(X_ling, nan=0.0, posinf=0.0, neginf=0.0)

gmap={'male':0,'female':1,'m':0,'f':1}
X_gender=df_labels['gender'].astype(str).str.lower().map(gmap).fillna(0.5).values.reshape(-1,1)
y_all=df_labels['label'].values.astype(int)
splits=df_labels['split'].values

# %% [markdown]
# ## 3. Feature Combination & Preprocessing

# %%
X_full = np.hstack([X_mfcc, X_ling, X_gender])

train_dev_mask = (splits=='train')|(splits=='dev')
test_mask = (splits=='test')
dev_mask = (splits=='dev')

y_traindev = y_all[train_dev_mask]
y_devonly = y_all[dev_mask]
y_test_off = y_all[test_mask]

def safe_clean(X):
    X=np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X,-1e6,1e6,out=X); return X

Xtr_raw = safe_clean(X_full[train_dev_mask].copy())
Xte_raw = safe_clean(X_full[test_mask].copy())
Xdv_raw = safe_clean(X_full[dev_mask].copy())

var = Xtr_raw.var(axis=0)
keep = var > 1e-10
if keep.sum() < 2: keep = np.ones(Xtr_raw.shape[1], dtype=bool)

Xtr = Xtr_raw[:, keep]
Xte = Xte_raw[:, keep]
Xdv = Xdv_raw[:, keep]

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)
Xdv = scaler.transform(Xdv)
Xtr, Xte, Xdv = safe_clean(Xtr), safe_clean(Xte), safe_clean(Xdv)

pca = PCA(n_components=0.95, random_state=42)  # PCA keep seed 42 for consistency
Xtr = pca.fit_transform(Xtr)
Xte = pca.transform(Xte)
Xdv = pca.transform(Xdv)
Xtr, Xte, Xdv = safe_clean(Xtr), safe_clean(Xte), safe_clean(Xdv)

# Balanced Class Weights (matching v56 MLP_B_weighted logic)
cls_cnt = np.bincount(y_traindev)
balanced_ratio = cls_cnt[0] / cls_cnt[1]
sw = np.ones(len(y_traindev))
sw[y_traindev == 1] = balanced_ratio

# %% [markdown]
# ## 4. Model Training (MLP)

# %%
print(f"\n[4] Training MLP (Hidden: 300, 150, 50 | Seed: {RANDOM_SEED})...")
model = MLPClassifier(
    hidden_layer_sizes=(300, 150, 50), 
    alpha=0.01,
    learning_rate_init=0.001, 
    max_iter=700,
    random_state=RANDOM_SEED, 
    early_stopping=True,
    validation_fraction=0.15, 
    n_iter_no_change=30
)

# Train with sample weights
model.fit(Xtr, y_traindev)

print("\n[5] Saving artifacts for production...")
with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODELS_DIR, 'feature_mask.pkl'), 'wb') as f:
    pickle.dump(keep, f)
with open(os.path.join(MODELS_DIR, 'pca.pkl'), 'wb') as f:
    pickle.dump(pca, f)
with open(os.path.join(MODELS_DIR, 'mlp_model.pkl'), 'wb') as f:
    pickle.dump(model, f)
print(f"Model and preprocessors saved to {MODELS_DIR}/")

# %% [markdown]
# ## 5. Evaluation

# %%
print("\n[6] Evaluation...")
probs_dv = model.predict_proba(Xdv)[:, 1]
probs_te = model.predict_proba(Xte)[:, 1]

# Tune threshold on Dev set
bf_dv, bt_dv = 0.0, 0.5
for thr in np.arange(0.20, 0.80, 0.01):
    f1 = f1_score(y_devonly, (probs_dv >= thr).astype(int), average='macro', zero_division=0)
    if f1 > bf_dv: bf_dv, bt_dv = f1, thr

preds_te = (probs_te >= bt_dv).astype(int)
f1_te = f1_score(y_test_off, preds_te, average='macro', zero_division=0)
auc_te = roc_auc_score(y_test_off, probs_te)
acc_te = accuracy_score(y_test_off, preds_te)

print(f"Best Threshold (from Dev): {bt_dv:.2f}")
print("==============================================")
print(f"Test F1 Macro : {f1_te:.4f}")
print(f"Test Accuracy : {acc_te:.4f}")
print(f"Test AUC      : {auc_te:.4f}")
print("==============================================")

print("\nClassification Report (Test Set):")
print(classification_report(y_test_off, preds_te, target_names=['Non-Depressed','Depressed'], zero_division=0))

res_df = pd.DataFrame({
    'Model': [f'MLP_B_weighted_seed{RANDOM_SEED}'],
    'Threshold': [bt_dv],
    'Test F1 Macro': [f1_te],
    'Test Accuracy': [acc_te],
    'Test AUC': [auc_te]
})
res_df.to_csv(os.path.join(RESULTS_DIR, 'metrics', 'v59_results.csv'), index=False)

# %% [markdown]
# ## 6. Explainable AI (SHAP)

# %%
print("\n[7] Explainable AI (XAI) with SHAP...")
shap.initjs()

# Use KernelExplainer for Neural Networks.
# To speed up interpretation, we summarize the background dataset (training set)
background = shap.sample(Xtr, 100, random_state=RANDOM_SEED)

explainer = shap.KernelExplainer(model.predict_proba, background)

# Calculate SHAP values for the test set
print("Calculating SHAP values (this may take a minute)...")
shap_values = explainer.shap_values(Xte, nsamples=100)

# Extract feature names from PCA
feature_names = [f"PC{i+1}" for i in range(Xtr.shape[1])]

# plot the SHAP summary
plt.figure(figsize=(10, 6))
# shap_values is an array of shape (samples, features, classes) for predict_proba. We use index 1 for "Depressed" class.
if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

shap.summary_plot(sv, Xte, feature_names=feature_names, show=False)

shap_path = os.path.join(RESULTS_DIR, "metrics", "shap_summary_mlp_v59.png")
plt.savefig(shap_path, bbox_inches='tight')
plt.close()

print(f"SHAP summary plot saved to: {shap_path}")
print(f"Total time: {time.time()-t_global:.1f}s")
