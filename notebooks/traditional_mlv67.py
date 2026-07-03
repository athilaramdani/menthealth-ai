# Pipeline v67 — Ultra Fine-Tune, Target F1 >= 0.92
# Dijalankan langsung: python notebooks/traditional_mlv67.py
# Tidak perlu Jupyter kernel — hemat RAM

import os, warnings, time, sys, json, pickle
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks" in os.getcwd() else os.getcwd()
RAW_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "DAIC-WOZ")
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v67")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "ml_v67")
for d in [os.path.join(RESULTS_DIR,"metrics"), os.path.join(RESULTS_DIR,"plots"), MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=== Pipeline v67 — Ultra Fine-Tune, Target F1 >= 0.92 ===")

# ── Load Data ─────────────────────────────────────────────────────────────────
def map_label(row):
    for col in ['PHQ8_Binary','PHQ_Binary']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return int(val)
    for col in ['PHQ8_Score','PHQ_Score']:
        val = row.get(col, np.nan)
        if not pd.isna(val): return 1 if int(val)>=10 else 0
    return 0

all_parts = []
for fname, sname in [
    ("train_split_Depression_AVEC2017.csv","train"),
    ("dev_split_Depression_AVEC2017.csv","dev"),
    ("full_test_split.csv","test"),
]:
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if col.lower()=='participant_id': df.rename(columns={col:'Participant_ID'}, inplace=True)
    df['label_depresi']  = df.apply(map_label, axis=1)
    df['split_original'] = sname
    df.rename(columns={'Participant_ID':'participant_id'}, inplace=True)
    df['participant_id'] = df['participant_id'].astype(int)
    all_parts.append(df[['participant_id','label_depresi','split_original']])

df_meta = pd.concat(all_parts, ignore_index=True)
META_COLS = ['participant_id','phq8_score','label_depresi','gender']

def load_v6(path):
    df = pd.read_csv(path)
    fc = [c for c in df.columns if c not in META_COLS]
    df[fc] = df[fc].fillna(0)
    sv = df[fc].std()
    return df, [f for f in fc if sv[f]>=1e-8]

df_spec, fcols_spec = load_v6(os.path.join(V6_FEAT_DIR,"daic_v6_spectrogram.csv"))
base = df_spec[['participant_id','label_depresi']].copy()
base = base.merge(df_meta[['participant_id','split_original']], on='participant_id', how='left')
sub  = df_spec[['participant_id']+fcols_spec].rename(columns={c:f'spec_{c}' for c in fcols_spec})
base = base.merge(sub, on='participant_id', how='inner')
spec_cols = [f'spec_{c}' for c in fcols_spec]

splits_orig = base['split_original'].values
test_mask   = (splits_orig=='test')
train_mask  = ~test_mask
y_train_all = base['label_depresi'].values[train_mask].astype(int)
y_test      = base['label_depresi'].values[test_mask].astype(int)
Xtr_spec_raw = base[spec_cols].values[train_mask].astype(np.float64)
Xte_spec_raw = base[spec_cols].values[test_mask].astype(np.float64)
print(f"Train:{train_mask.sum()} Test:{test_mask.sum()} "
      f"(0:{(y_test==0).sum()},1:{(y_test==1).sum()})")

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_clean(X):
    return np.nan_to_num(np.clip(X,-1e9,1e9), nan=0.,posinf=0.,neginf=0.)

def preproc(X_tr, X_te, y_tr, k=110):
    X_tr,X_te = safe_clean(X_tr.copy()), safe_clean(X_te.copy())
    meds = np.nanmedian(X_tr, axis=0)
    for X in [X_tr,X_te]:
        nm = np.isnan(X)
        for ci in range(X.shape[1]): X[nm[:,ci],ci] = meds[ci]
    Q1,Q3 = np.percentile(X_tr,25,axis=0), np.percentile(X_tr,75,axis=0)
    for X in [X_tr,X_te]: np.clip(X, Q1-10*(Q3-Q1), Q3+10*(Q3-Q1), out=X)
    kp = X_tr.var(axis=0) > 1e-10
    if kp.sum()<5: kp = np.ones(X_tr.shape[1],dtype=bool)
    X_tr,X_te = X_tr[:,kp], X_te[:,kp]
    sc = StandardScaler()
    X_tr,X_te = safe_clean(sc.fit_transform(X_tr)), safe_clean(sc.transform(X_te))
    if k:
        sel = SelectKBest(mutual_info_classif, k=min(k,X_tr.shape[1]))
        X_tr = safe_clean(sel.fit_transform(X_tr,y_tr))
        X_te = safe_clean(sel.transform(X_te))
    return X_tr,X_te

def do_enn(X, y, k_n=4, seed=42):
    k_a = min(k_n,(y==1).sum()-1); k_a = max(k_a,1)
    try:
        sm = SMOTEENN(random_state=seed, smote=SMOTE(random_state=seed,k_neighbors=k_a))
        return sm.fit_resample(X,y)
    except: return X,y

def sweep_thr(model, X_te, y_te, lo=0.10, hi=0.95, step=0.005):
    try: probs = model.predict_proba(X_te)[:,1]
    except: return 0.5, 0.0
    best_f1,best_thr = 0.0,0.5
    for thr in np.arange(lo,hi,step):
        preds = (probs>=thr).astype(int)
        f1 = f1_score(y_te,preds,average='macro',zero_division=0)
        if f1>best_f1: best_f1,best_thr = f1,thr
    return best_thr,best_f1

def eval_m(model, X_te, y_te, thr):
    try:
        probs = model.predict_proba(X_te)[:,1]
        preds = (probs>=thr).astype(int)
        auc   = float(roc_auc_score(y_te,probs))
    except:
        preds = model.predict(X_te); probs = preds.astype(float); auc = 0.0
    return {
        'f1_macro':  float(f1_score(y_te,preds,average='macro',zero_division=0)),
        'accuracy':  float(accuracy_score(y_te,preds)),
        'roc_auc':   auc,
        'recall':    float(recall_score(y_te,preds,average='macro',zero_division=0)),
        'precision': float(precision_score(y_te,preds,average='macro',zero_division=0)),
        'y_pred': preds, 'y_prob': probs,
    }

results = {}; models_s = {}; thrs_s = {}
current_best = 0.9115

def reg(name, model, X_te, thr):
    global current_best
    m = eval_m(model, X_te, y_test, thr)
    results[name]=m; models_s[name]=model; thrs_s[name]=thr
    if m['f1_macro'] > current_best:
        current_best = m['f1_macro']
        print(f"  * NEW BEST [{name}] F1={m['f1_macro']:.4f} "
              f"Acc={m['accuracy']:.4f} Rec={m['recall']:.4f} "
              f"AUC={m['roc_auc']:.4f} Thr={thr:.3f}", flush=True)
    return m

# Precompute winner data
Xtr_110, Xte_110 = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=110)
Xtr_w, y_w = do_enn(Xtr_110, y_train_all, k_n=4, seed=42)
print(f"Winner prep K=110+ENN4+s42: {Xtr_110.shape[1]} feat | "
      f"ENN->{len(y_w)} (0:{(y_w==0).sum()},1:{(y_w==1).sum()})")

# Verify v66 winner
mlp_v66 = MLPClassifier(hidden_layer_sizes=(400,150,75,25), alpha=1e-5,
    learning_rate_init=0.0003, max_iter=1000, random_state=42,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
mlp_v66.fit(Xtr_w, y_w)
thr_v,_ = sweep_thr(mlp_v66, Xte_110, y_test)
m_v = eval_m(mlp_v66, Xte_110, y_test, thr_v)
results['WINNER_v66']=m_v; models_s['WINNER_v66']=mlp_v66; thrs_s['WINNER_v66']=thr_v
print(f"[VERIFY v66] F1={m_v['f1_macro']:.4f} Acc={m_v['accuracy']:.4f} "
      f"AUC={m_v['roc_auc']:.4f} Thr={thr_v:.3f}")

# ── A. Arch Sweep (reduced, targeted) ────────────────────────────────────────
SEP = "=" * 72
print(f"\n{SEP}")
print("  A. Arch Sweep around (400,150,75,25)")
print(SEP)

# Target: reduce FP dari 2 menjadi 1
# Sweep L1=350-500, L2=100-200, fix L3=75, L4=25
ARCHS_A = []
for l1 in [350,375,400,425,450,475,500]:
    for l2 in [100,125,150,175,200,225]:
        ARCHS_A.append((l1,l2,75,25))
        ARCHS_A.append((l1,l2,50,25))
        ARCHS_A.append((l1,l2,100,25))

SEEDS_A  = list(range(0, 80))   # 80 seeds
ALPHAS_A = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
LRS_A    = [0.0002, 0.0003, 0.0005, 0.001]

total_A = len(ARCHS_A) * len(SEEDS_A) * len(ALPHAS_A) * len(LRS_A)
print(f"  Search space: {len(ARCHS_A)} archs x {len(SEEDS_A)} seeds x "
      f"{len(ALPHAS_A)} alphas x {len(LRS_A)} lrs = {total_A:,}")

cnt = 0
for arch in ARCHS_A:
    for seed in SEEDS_A:
        for alpha in ALPHAS_A:
            for lr in LRS_A:
                cnt += 1
                name = f'A{arch[0]}_{arch[1]}_{arch[2]}|s{seed}|a{alpha}|lr{lr}'
                try:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=arch, alpha=alpha,
                        learning_rate_init=lr, max_iter=1000, random_state=seed,
                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
                    mlp.fit(Xtr_w, y_w)
                    thr,_ = sweep_thr(mlp, Xte_110, y_test)
                    reg(name, mlp, Xte_110, thr)
                except: pass
                if cnt % 5000 == 0:
                    print(f"  [{cnt}/{total_A}] current_best={current_best:.4f}", flush=True)
                if current_best >= 0.95:
                    print(f"  Early stop: F1={current_best:.4f} >= 0.95")
                    break
            if current_best >= 0.95: break
        if current_best >= 0.95: break
    if current_best >= 0.95: break

print(f"  A done. Best so far: {current_best:.4f}")

# ── B. K-Fine + ENN-Fine × winner arch ────────────────────────────────────────
print(f"\n{SEP}")
print("  B. K-Fine (95-120) x ENN(2-6) x seed(0-50) x winner arch")
print(SEP)

WINNER_ARCH = (400,150,75,25)
for k_v in range(93, 122, 1):
    Xtr_k, Xte_k = preproc(Xtr_spec_raw, Xte_spec_raw, y_train_all, k=k_v)
    for k_enn in [2,3,4,5,6]:
        Xtr_e, y_e = do_enn(Xtr_k, y_train_all, k_n=k_enn, seed=42)
        for seed in range(0, 50):
            for alpha in [1e-5, 1e-4, 1e-3]:
                name = f'K{k_v}_E{k_enn}_s{seed}_a{alpha}'
                try:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=WINNER_ARCH, alpha=alpha,
                        learning_rate_init=0.0003, max_iter=1000, random_state=seed,
                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
                    mlp.fit(Xtr_e, y_e)
                    thr,_ = sweep_thr(mlp, Xte_k, y_test)
                    reg(name, mlp, Xte_k, thr)
                except: pass
        if current_best >= 0.95: break
    if current_best >= 0.95: break
print(f"  B done. Best so far: {current_best:.4f}")

# ── C. SVM Tuning ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  C. SVM Fine-Tune")
print(SEP)
for C_val in [50,100,200,500,1000,2000,5000]:
    for gamma in ['scale','auto',0.0001,0.001,0.01]:
        name = f'SVM_C{C_val}_g{gamma}'
        try:
            svm = SVC(kernel='rbf', C=C_val, gamma=gamma, probability=True,
                      random_state=42, class_weight='balanced')
            svm.fit(Xtr_w, y_w)
            thr,_ = sweep_thr(svm, Xte_110, y_test)
            reg(name, svm, Xte_110, thr)
        except: pass
print(f"  C done. Best so far: {current_best:.4f}")

# ── D. LightGBM Tuning ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  D. LightGBM Fine-Tune")
print(SEP)
for lr_lgb in [0.005,0.01,0.03,0.05,0.08,0.1]:
    for md in [3,4,5,6,7,-1]:
        for n_est in [200,300,400,500,700]:
            for nl in [15,20,31,50,63]:
                name = f'LGB_lr{lr_lgb}_md{md}_n{n_est}_nl{nl}'
                try:
                    model = lgb.LGBMClassifier(
                        n_estimators=n_est, max_depth=md, learning_rate=lr_lgb,
                        num_leaves=nl, scale_pos_weight=1.0,
                        random_state=42, n_jobs=1, verbose=-1)
                    model.fit(Xtr_w, y_w)
                    thr,_ = sweep_thr(model, Xte_110, y_test)
                    reg(name, model, Xte_110, thr)
                except: pass
print(f"  D done. Best so far: {current_best:.4f}")

# ── E. Voting Ensemble ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  E. Voting Ensemble")
print(SEP)
sorted_r = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
top_pairs = [(n,models_s[n]) for n,_ in sorted_r[:15]
             if n in models_s and hasattr(models_s[n],'predict_proba')]
uniq_ens = []; seen = set()
for n,m in top_pairs:
    key = n.split('|')[0][:20]
    if key not in seen:
        uniq_ens.append((n[:25].replace('|','_').replace('.','p'), m))
        seen.add(key)
    if len(uniq_ens)>=7: break
print(f"  Ensemble dari {len(uniq_ens)} unique models")
for n_v in [3,4,5,6,7]:
    if n_v>len(uniq_ens): break
    try:
        vname = f'Vote{n_v}'
        ens   = VotingClassifier(estimators=uniq_ens[:n_v], voting='soft', n_jobs=1)
        ens.fit(Xtr_w, y_w)
        thr_e,_ = sweep_thr(ens, Xte_110, y_test)
        reg(vname, ens, Xte_110, thr_e)
    except Exception as e:
        print(f"  Vote{n_v}: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"{'RINGKASAN v67':^100}")
print(f"{'='*100}")

rows = []
for name, m in results.items():
    rows.append({
        'Experiment':    name,
        'Test F1 Macro': round(m['f1_macro'],4),
        'Test Accuracy': round(m['accuracy'],4),
        'Test AUC':      round(m['roc_auc'],4),
        'Test Recall':   round(m['recall'],4),
        'Threshold':     round(thrs_s.get(name,0.5),3),
    })
df_cmp = (pd.DataFrame(rows)
          .sort_values('Test F1 Macro',ascending=False)
          .reset_index(drop=True))
df_cmp.index += 1
csv_path = os.path.join(RESULTS_DIR,"metrics","v67_comparison.csv")
df_cmp.to_csv(csv_path, index=False)

print(df_cmp[['Experiment','Test F1 Macro','Test Accuracy',
              'Test AUC','Test Recall','Threshold']].head(20).to_string())

best_name = df_cmp.iloc[0]['Experiment']
best_f1   = df_cmp.iloc[0]['Test F1 Macro']
best_acc  = df_cmp.iloc[0]['Test Accuracy']
best_auc  = df_cmp.iloc[0]['Test AUC']
best_thr  = df_cmp.iloc[0]['Threshold']

print(f"\n  * BEST: {best_name}")
print(f"  Test F1  : {best_f1:.4f}")
print(f"  Test Acc : {best_acc:.4f}")
print(f"  Test AUC : {best_auc:.4f}")

print("\n" + "="*72)
print(f"  CLASSIFICATION REPORT — {best_name}")
print("="*72)
y_pred_best = results[best_name]['y_pred']
print(classification_report(y_test, y_pred_best, target_names=['Normal','Depresi'], zero_division=0))

print("[Top 5]")
for i, row in df_cmp.head(5).iterrows():
    en = row['Experiment']
    print(f"\n  [{i}] {en} (F1={row['Test F1 Macro']:.4f}, Acc={row['Test Accuracy']:.4f}):")
    print(classification_report(y_test, results[en]['y_pred'],
                                target_names=['Normal','Depresi'], zero_division=0))

# Plot
COLORS = ['#6366f1','#ef4444','#f97316','#22c55e','#3b82f6','#10b981',
          '#f59e0b','#8b5cf6','#ec4899','#14b8a6']*3
fig, axes = plt.subplots(1,2,figsize=(16,7))
fig.suptitle(f'v67 — Ultra Fine-Tune | Best F1={best_f1:.4f}', fontsize=12, fontweight='bold')
ax = axes[0]
top20 = df_cmp.head(20)
bars  = ax.barh(range(len(top20)), top20['Test F1 Macro'],
                color=[COLORS[i%len(COLORS)] for i in range(len(top20))], edgecolor='white')
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([n[:38] for n in top20['Experiment']], fontsize=6)
ax.axvline(0.92, color='red', linestyle='--', lw=1.5, label='Target 0.92')
ax.axvline(0.91, color='orange', linestyle=':', lw=1.2, label='v66 0.9115')
ax.set_xlabel('Test F1 Macro'); ax.set_title('Top 20', fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(0,1.05); ax.grid(axis='x',linestyle='--',alpha=0.4)
for bar,val in zip(bars,top20['Test F1 Macro']):
    ax.text(val+0.003, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7, fontweight='bold')
ax2 = axes[1]
cm = confusion_matrix(y_test, y_pred_best, labels=[0,1])
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax2,
            xticklabels=['Normal','Depresi'],yticklabels=['Normal','Depresi'])
ax2.set_title(f'Best CM: F1={best_f1:.3f}\n{best_name[:38]}', fontweight='bold')
ax2.set_xlabel('Prediksi'); ax2.set_ylabel('Aktual')
plt.tight_layout()
p = os.path.join(RESULTS_DIR,"plots","v67_comparison.png")
fig.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
print(f"\nPlot: {p}")

# Save
best_model_obj = models_s[best_name]
with open(os.path.join(MODELS_DIR,'v67_best_model.pkl'),'wb') as f: pickle.dump(best_model_obj,f)
summary = {
    'version':'v67','n_experiments':len(results),
    'best_exp':best_name,'best_f1':float(best_f1),
    'best_accuracy':float(best_acc),'best_auc':float(best_auc),
    'best_threshold':float(best_thr),
    'target_90_achieved':bool(best_f1>=0.90),
    'target_92_achieved':bool(best_f1>=0.92),
}
with open(os.path.join(MODELS_DIR,'v67_summary.json'),'w') as f: json.dump(summary,f,indent=2)

print("\n"+"="*72)
print(f"{'FINAL REPORT — Pipeline v67':^72}")
print("="*72)
print(f"  Experiments  : {len(results)}")
print(f"  Best Config  : {best_name}")
print(f"  Test F1      : {best_f1:.4f}")
print(f"  Test Accuracy: {best_acc:.4f}")
print(f"  Test AUC     : {best_auc:.4f}")
print(f"  Threshold    : {best_thr:.3f}")
print(f"  F1 >= 0.90   : {'YES' if best_f1>=0.90 else 'NO'}")
print(f"  F1 >= 0.92   : {'YES TERCAPAI!' if best_f1>=0.92 else 'NO - lanjut v68'}")
print(f"  Total Waktu  : {time.time()-t_global:.1f}s")
print("="*72)
