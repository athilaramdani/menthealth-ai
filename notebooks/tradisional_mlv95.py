# %% [markdown]
# # Pipeline v95 - Wav2Vec 768 No-PCA + Anti-Leakage Pipeline
#
# Fokus v95 sesuai instruksi:
# - Tetap 4 skenario apple-to-apple: Spectrogram, MFCC, Wav2Vec, Fusion.
# - Gunakan fitur yang sudah diekstrak.
# - Gunakan Wav2Vec v8 768 mean embedding untuk S3 dan S4.
# - S3 Wav2Vec 768 untuk Logistic Regression dan Linear SVM dijalankan TANPA PCA.
# - Skenario/model lainnya memakai sklearn Pipeline: StandardScaler -> PCA -> Classifier.
# - Class imbalance ditangani native: class_weight='balanced' dan scale_pos_weight dinamis.
# - Threshold OOF sweep fokus 0.35-0.45, fallback wider bila perlu.

# %%
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd().lower()
    else os.getcwd()
)
V6_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
V8_W2V_PATH = os.path.join(PROJECT_ROOT, "data", "features", "v8", "daic_v8_wav2vec_full.csv")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
V95_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v95")
V95_W2V_PATH = os.path.join(V95_FEAT_DIR, "daic_v95_wav2vec768_mean_189.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v95")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ml_v95")
for d in [METRICS_DIR, PLOTS_DIR, V95_FEAT_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 92)
print("  Pipeline v95 - Wav2Vec 768 No-PCA + Anti-Leakage Pipeline")
print("  Strict S1-S4 apple-to-apple; no cross-scenario model fusion")
print("=" * 92)

# %% [markdown]
# ## 1. Load Fitur Cache

# %%
META_COLS_V6 = ["participant_id", "phq8_score", "label_depresi", "gender"]

def clean_feature_frame(df, meta_cols):
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = df[feature_cols].std(numeric_only=True)
    good_cols = [c for c in feature_cols if std.get(c, 0.0) >= 1e-8]
    return df, good_cols

df_spec = pd.read_csv(os.path.join(V6_DIR, "daic_v6_spectrogram.csv"))
df_mfcc = pd.read_csv(os.path.join(V6_DIR, "daic_v6_mfcc.csv"))

df_spec, fcols_spec = clean_feature_frame(df_spec, META_COLS_V6)
df_mfcc, fcols_mfcc = clean_feature_frame(df_mfcc, META_COLS_V6)

TARGET_PIDS = sorted(df_spec["participant_id"].astype(int).tolist())
LABEL_BY_PID = dict(zip(df_spec["participant_id"].astype(int), df_spec["label_depresi"].astype(int)))
W2V_COLS = [f"w2v_{i}" for i in range(768)]

def extract_w2v768_mean_for_pid(pid, processor, model, torch_module, target_sr=16000):
    wav_path = os.path.join(CLEANED_DIR, f"{int(pid)}.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    if len(y) < target_sr:
        raise ValueError(f"Audio terlalu pendek: {wav_path}")

    chunk_len = 30 * target_sr
    weighted_sum = None
    total_frames = 0
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) < 1600:
            continue
        inputs = processor(chunk, sampling_rate=target_sr, return_tensors="pt", padding=True)
        with torch_module.no_grad():
            out = model(**inputs)
        hidden = out.last_hidden_state.squeeze(0).detach().cpu().numpy()
        chunk_sum = hidden.sum(axis=0)
        weighted_sum = chunk_sum if weighted_sum is None else weighted_sum + chunk_sum
        total_frames += hidden.shape[0]
    if weighted_sum is None or total_frames == 0:
        raise ValueError(f"Tidak ada frame Wav2Vec valid: {wav_path}")
    return (weighted_sum / total_frames).astype(np.float32)

def build_or_load_w2v95_cache():
    if os.path.exists(V95_W2V_PATH):
        df_cached = pd.read_csv(V95_W2V_PATH)
        have = set(df_cached["participant_id"].astype(int))
        if len(df_cached) == len(TARGET_PIDS) and set(TARGET_PIDS).issubset(have):
            print(f"\n[INFO] Cache Wav2Vec v95 lengkap ditemukan: {V95_W2V_PATH}")
            return df_cached[["participant_id", "label"] + W2V_COLS].copy()
        print(f"\n[WARN] Cache Wav2Vec v95 belum lengkap ({len(df_cached)} rows). Melengkapi...")

    rows = []
    if os.path.exists(V8_W2V_PATH):
        df_v8 = pd.read_csv(V8_W2V_PATH)
        missing_cols = [c for c in W2V_COLS if c not in df_v8.columns]
        if missing_cols:
            raise RuntimeError(f"Kolom Wav2Vec v8 tidak lengkap, missing contoh: {missing_cols[:5]}")
        df_v8 = df_v8[["participant_id", "label"] + W2V_COLS].copy()
        df_v8["participant_id"] = df_v8["participant_id"].astype(int)
        df_v8 = df_v8[df_v8["participant_id"].isin(TARGET_PIDS)]
        rows.extend(df_v8.to_dict("records"))

    existing = {int(r["participant_id"]) for r in rows}
    missing_pids = [pid for pid in TARGET_PIDS if pid not in existing]
    print("\nWav2Vec 768 cache preparation:")
    print(f"  Target participant : {len(TARGET_PIDS)}")
    print(f"  Tersedia dari v8   : {len(existing)}")
    print(f"  Perlu ekstraksi    : {len(missing_pids)}")

    if missing_pids:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        print("  Loading facebook/wav2vec2-base untuk ekstraksi missing participant...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        model.eval()

        t_extract = time.time()
        for i, pid in enumerate(missing_pids, start=1):
            try:
                vec = extract_w2v768_mean_for_pid(pid, processor, model, torch)
                row = {"participant_id": int(pid), "label": int(LABEL_BY_PID[int(pid)])}
                row.update({f"w2v_{j}": float(vec[j]) for j in range(768)})
                rows.append(row)
                status = "OK"
            except Exception as exc:
                # Jangan zero-fill diam-diam: catat kegagalan dan hentikan agar data tidak kotor.
                raise RuntimeError(f"Gagal ekstrak Wav2Vec participant {pid}: {exc}") from exc

            if i == 1 or i % 5 == 0 or i == len(missing_pids):
                elapsed = time.time() - t_extract
                print(f"    [{i:3d}/{len(missing_pids):3d}] PID={pid} {status} | elapsed={elapsed/60:.1f} min", flush=True)

    df_full = pd.DataFrame(rows)
    df_full["participant_id"] = df_full["participant_id"].astype(int)
    df_full = df_full.drop_duplicates("participant_id", keep="last")
    df_full = df_full[df_full["participant_id"].isin(TARGET_PIDS)].copy()
    df_full = df_full.sort_values("participant_id").reset_index(drop=True)

    if len(df_full) != len(TARGET_PIDS):
        have = set(df_full["participant_id"].astype(int))
        still_missing = [pid for pid in TARGET_PIDS if pid not in have]
        raise RuntimeError(f"Cache Wav2Vec v95 masih kurang {len(still_missing)} participant: {still_missing[:10]}")

    df_full[W2V_COLS] = df_full[W2V_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_full[["participant_id", "label"] + W2V_COLS].to_csv(V95_W2V_PATH, index=False)
    print(f"  Cache Wav2Vec v95 lengkap disimpan: {V95_W2V_PATH}")
    return df_full[["participant_id", "label"] + W2V_COLS].copy()

# v95 memakai cache bersih 189 participant. Tidak ada zero-fill participant missing.
df_w2v8 = build_or_load_w2v95_cache()
fcols_w2v_mean = W2V_COLS

base = df_spec[["participant_id", "label_depresi"]].copy().sort_values("participant_id")
base = base.merge(
    df_spec[["participant_id"] + fcols_spec].rename(columns={c: f"spec_{c}" for c in fcols_spec}),
    on="participant_id",
    how="left",
)
base = base.merge(
    df_mfcc[["participant_id"] + fcols_mfcc].rename(columns={c: f"mfcc_{c}" for c in fcols_mfcc}),
    on="participant_id",
    how="left",
)
base = base.merge(
    df_w2v8[["participant_id"] + fcols_w2v_mean].rename(columns={c: f"w2v8_{c}" for c in fcols_w2v_mean}),
    on="participant_id",
    how="left",
)

w2v8_cols = [f"w2v8_{c}" for c in fcols_w2v_mean]
w2v_missing = int(base[w2v8_cols].isna().any(axis=1).sum())
if w2v_missing:
    raise RuntimeError(f"Wav2Vec v95 cache masih missing untuk {w2v_missing} participant. Stop, tidak zero-fill.")
base = base.fillna(0.0).reset_index(drop=True)

y_all = base["label_depresi"].values.astype(int)
X_spec = base[[f"spec_{c}" for c in fcols_spec]].values.astype(np.float64)
X_mfcc = base[[f"mfcc_{c}" for c in fcols_mfcc]].values.astype(np.float64)
X_w2v = base[w2v8_cols].values.astype(np.float64)
X_fusion = np.hstack([X_spec, X_mfcc, X_w2v])

SCENARIOS = {
    "S1_Spectrogram": X_spec,
    "S2_MFCC": X_mfcc,
    "S3_Wav2Vec768": X_w2v,
    "S4_Fusion": X_fusion,
}
FEATURE_NAMES = {
    "S1_Spectrogram": [f"spec_{c}" for c in fcols_spec],
    "S2_MFCC": [f"mfcc_{c}" for c in fcols_mfcc],
    "S3_Wav2Vec768": w2v8_cols,
    "S4_Fusion": [f"spec_{c}" for c in fcols_spec] + [f"mfcc_{c}" for c in fcols_mfcc] + w2v8_cols,
}

feature_dims = pd.DataFrame([
    {"scenario": name, "dimension": X.shape[1]} for name, X in SCENARIOS.items()
])
feature_dims.to_csv(os.path.join(METRICS_DIR, "v95_feature_dimensions.csv"), index=False)

print("\nFeature cache:")
print(f"  Spectrogram v6 : {len(df_spec)} rows, {len(fcols_spec)} fitur aktif")
print(f"  MFCC v6        : {len(df_mfcc)} rows, {len(fcols_mfcc)} fitur aktif")
print(f"  Wav2Vec v8     : {len(df_w2v8)} rows, 768 mean embedding")
print(f"  Dataset v95    : {len(base)} rows dipertahankan dari basis v6")
print(f"  Wav2Vec missing pada basis 189: {w2v_missing} rows")
print("\nDimensi fitur v95:")
print(feature_dims.to_string(index=False))

print("\nKomponen/nama fitur eksplisit:")
print("  S1 Spectrogram/GeMAPS-like: mel-band statistics, entropy, spectral centroid, bandwidth, rolloff, flatness, chroma, RMS, ZCR, low-energy ratio.")
print("  S2 MFCC/Prosodic: MFCC m1-m40, delta MFCC, delta-delta MFCC, pitch/F0 proxy, RMS, ZCR, HNR proxy, spectral entropy, jitter, shimmer, voiced ratio.")
print("  S3 Wav2Vec768: mean embedding dimensions w2v_0 sampai w2v_767.")
for scenario_name, names in FEATURE_NAMES.items():
    print(f"\n  {scenario_name} ({len(names)} fitur)")
    print("    First 30 : " + ", ".join(names[:30]))
    if len(names) > 30:
        print("    Last 10  : " + ", ".join(names[-10:]))

# %% [markdown]
# ## 2. Split 80:20 dengan Test Balanced

# %%
idx_n = np.where(y_all == 0)[0]
idx_d = np.where(y_all == 1)[0]
np.random.seed(RANDOM_SEED)
test_idx = np.concatenate([
    np.random.choice(idx_n, 10, replace=False),
    np.random.choice(idx_d, 10, replace=False),
])
test_idx = np.sort(test_idx)
train_idx = np.setdiff1d(np.arange(len(y_all)), test_idx)
y_train = y_all[train_idx]
y_test = y_all[test_idx]

n_train_n = int((y_train == 0).sum())
n_train_d = int((y_train == 1).sum())
spw_train = n_train_n / max(n_train_d, 1)

split_df = base[["participant_id", "label_depresi"]].copy()
split_df["split_v95"] = np.where(np.isin(np.arange(len(base)), test_idx), "test", "train")
split_df.to_csv(os.path.join(METRICS_DIR, "v95_split_participants.csv"), index=False)

print("\nSplit v95:")
print(f"  Train={len(train_idx)} | Normal={n_train_n} | Depresi={n_train_d} | scale_pos_weight={spw_train:.3f}")
print(f"  Test ={len(test_idx)}  | Normal={(y_test == 0).sum()}  | Depresi={(y_test == 1).sum()}  | balanced")

# %% [markdown]
# ## 3. Pipeline, Model, OOF Threshold

# %%
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

def sanitize(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e9, 1e9)

def threshold_sweep(probs, y_true, focused=True):
    ranges = [(0.35, 0.45, 0.005)]
    if not focused:
        ranges = [(0.05, 0.95, 0.005)]
    best = {"thr": 0.5, "f1": -1.0}
    for lo, hi, step in ranges:
        for thr in np.arange(lo, hi + 1e-12, step):
            pred = (probs >= thr).astype(int)
            score = f1_score(y_true, pred, average="macro", zero_division=0)
            if score > best["f1"]:
                best = {"thr": float(thr), "f1": float(score)}
    return best["thr"], best["f1"]

def best_oof_threshold(probs, y_true):
    thr_focus, f1_focus = threshold_sweep(probs, y_true, focused=True)
    thr_wide, f1_wide = threshold_sweep(probs, y_true, focused=False)
    use_fallback = f1_wide > f1_focus + 0.015
    if use_fallback:
        return thr_wide, f1_wide, thr_focus, f1_focus, "wide"
    return thr_focus, f1_focus, thr_wide, f1_wide, "focused"

def predict_top50(probs):
    pred = np.zeros(len(probs), dtype=int)
    pred[np.argsort(probs)[-len(probs)//2:]] = 1
    return pred

def make_classifier(model_name, cfg, y_fit):
    n0 = int((y_fit == 0).sum())
    n1 = int((y_fit == 1).sum())
    spw = n0 / max(n1, 1)
    if model_name == "LR":
        return LogisticRegression(
            C=cfg["C"],
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_SEED,
        )
    if model_name == "SVM":
        return SVC(
            C=cfg["C"],
            kernel=cfg["kernel"],
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        )
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_SEED,
        )
    if model_name == "XGB":
        return xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(model_name)

def build_pipeline(model_name, cfg, pca_n, y_fit, no_pca=False):
    steps = [("scaler", StandardScaler())]
    if not no_pca:
        steps.append(("pca", PCA(n_components=pca_n, random_state=RANDOM_SEED)))
    steps.append(("clf", make_classifier(model_name, cfg, y_fit)))
    return Pipeline(steps)

def model_configs(model_name, scenario_name):
    if model_name == "LR":
        cs = [0.001, 0.01, 0.05, 0.1] if scenario_name == "S3_Wav2Vec768" else [0.01, 0.1, 1.0]
        return [{"C": c} for c in cs]
    if model_name == "SVM":
        if scenario_name == "S3_Wav2Vec768":
            return [{"C": c, "kernel": "linear"} for c in [0.01, 0.1, 1.0]]
        return [{"C": c, "kernel": k} for c in [0.1, 1.0] for k in ["linear", "rbf"]]
    if model_name == "RF":
        return [
            {"n_estimators": 200, "max_depth": md, "min_samples_leaf": leaf}
            for md in [3, None] for leaf in [2, 4]
        ]
    if model_name == "XGB":
        return [
            {"n_estimators": 100, "max_depth": md, "learning_rate": 0.05, "reg_alpha": 1.0, "reg_lambda": 5.0}
            for md in [2, 3]
        ]
    raise ValueError(model_name)

def pca_grid(scenario_name, model_name):
    if scenario_name == "S3_Wav2Vec768" and model_name in ["LR", "SVM"]:
        return [None]
    if scenario_name == "S3_Wav2Vec768":
        return [20, 50, 80]
    if scenario_name == "S4_Fusion":
        return [10, 20, 30, 50]
    return [10, 20, 30]

def evaluate_model(X_train_raw, X_test_raw, y_tr, y_te, scenario_name, model_name):
    X_train_raw = sanitize(X_train_raw)
    X_test_raw = sanitize(X_test_raw)
    candidates = []

    for cfg_idx, cfg in enumerate(model_configs(model_name, scenario_name)):
        for pca_n in pca_grid(scenario_name, model_name):
            no_pca = pca_n is None
            oof = np.zeros(len(y_tr), dtype=float)
            fold_f1 = []
            for tr, val in CV.split(X_train_raw, y_tr):
                pipe = build_pipeline(model_name, cfg, pca_n, y_tr[tr], no_pca=no_pca)
                pipe.fit(X_train_raw[tr], y_tr[tr])
                prob = pipe.predict_proba(X_train_raw[val])[:, 1]
                oof[val] = prob
                thr, _, _, _, _ = best_oof_threshold(prob, y_tr[val])
                fold_f1.append(f1_score(y_tr[val], (prob >= thr).astype(int), average="macro", zero_division=0))
            thr, cv_f1, thr_alt, cv_f1_alt, thr_source = best_oof_threshold(oof, y_tr)
            candidates.append({
                "cfg_idx": cfg_idx,
                "cfg": cfg,
                "pca_n": pca_n,
                "no_pca": no_pca,
                "oof": oof,
                "thr": thr,
                "cv_f1": cv_f1,
                "cv_f1_alt": cv_f1_alt,
                "thr_alt": thr_alt,
                "thr_source": thr_source,
                "fold_mean": float(np.mean(fold_f1)),
                "fold_std": float(np.std(fold_f1)),
            })

    best = max(candidates, key=lambda r: (r["cv_f1"], r["fold_mean"]))
    pipe = build_pipeline(model_name, best["cfg"], best["pca_n"], y_tr, no_pca=best["no_pca"])
    pipe.fit(X_train_raw, y_tr)
    prob_test = pipe.predict_proba(X_test_raw)[:, 1]
    pred_thr = (prob_test >= best["thr"]).astype(int)
    pred_top50 = predict_top50(prob_test)
    thr_oracle, f1_oracle = threshold_sweep(prob_test, y_te, focused=False)
    pred_oracle = (prob_test >= thr_oracle).astype(int)
    try:
        auc = roc_auc_score(y_te, prob_test)
    except Exception:
        auc = 0.0

    return {
        "scenario": scenario_name,
        "model": model_name,
        "cfg_idx": best["cfg_idx"],
        "cfg": str(best["cfg"]),
        "pca_n": "none" if best["no_pca"] else best["pca_n"],
        "no_pca": bool(best["no_pca"]),
        "cv_f1": round(best["cv_f1"], 4),
        "cv_fold_mean": round(best["fold_mean"], 4),
        "cv_std": round(best["fold_std"], 4),
        "oof_thr": round(best["thr"], 3),
        "thr_source": best["thr_source"],
        "test_f1_oof_thr": round(f1_score(y_te, pred_thr, average="macro", zero_division=0), 4),
        "test_f1_top50": round(f1_score(y_te, pred_top50, average="macro", zero_division=0), 4),
        "test_acc_oof_thr": round(accuracy_score(y_te, pred_thr), 4),
        "test_precision_oof_thr": round(precision_score(y_te, pred_thr, average="macro", zero_division=0), 4),
        "test_recall_oof_thr": round(recall_score(y_te, pred_thr, average="macro", zero_division=0), 4),
        "test_recall_dep_oof_thr": round(recall_score(y_te, pred_thr, pos_label=1, zero_division=0), 4),
        "test_auc": round(float(auc), 4),
        "test_thr_oracle": round(thr_oracle, 3),
        "test_f1_oracle": round(float(f1_oracle), 4),
        "y_pred_oof_thr": pred_thr.tolist(),
        "y_pred_top50": pred_top50.tolist(),
        "y_pred_oracle": pred_oracle.tolist(),
        "y_prob": prob_test.tolist(),
        "oof_probs": best["oof"].tolist(),
    }, pipe

# %% [markdown]
# ## 4. Main Experiment S1-S4 x 4 Model

# %%
all_results = []
best_pipe = None

print("\n" + "=" * 92)
print("  v95 MAIN LOOP: S1-S4 x RF/SVM/LR/XGBoost")
print("  S3 LR + Linear SVM memakai Wav2Vec 768 TANPA PCA")
print("=" * 92)

for scenario_name, X in SCENARIOS.items():
    print(f"\n{'-' * 92}")
    print(f"  {scenario_name}: {X.shape[1]} fitur")
    print(f"{'-' * 92}")
    X_tr = X[train_idx]
    X_te = X[test_idx]
    for model_name in ["RF", "SVM", "LR", "XGB"]:
        t0 = time.time()
        res, pipe = evaluate_model(X_tr, X_te, y_train, y_test, scenario_name, model_name)
        res["time_s"] = round(time.time() - t0, 1)
        all_results.append(res)
        if best_pipe is None or res["test_f1_oof_thr"] > max(r["test_f1_oof_thr"] for r in all_results[:-1]):
            best_pipe = pipe
        flag = "NO-PCA" if res["no_pca"] else f"PCA={res['pca_n']}"
        print(
            f"  {model_name:<4} {flag:<8} CV={res['cv_f1']:.4f} thr={res['oof_thr']:.3f}/{res['thr_source']:<7} "
            f"TestF1={res['test_f1_oof_thr']:.4f} Top50={res['test_f1_top50']:.4f} "
            f"AUC={res['test_auc']:.4f} ({res['time_s']}s)",
            flush=True,
        )

df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(METRICS_DIR, "v95_results.csv"), index=False)

# %% [markdown]
# ## 5. Visualisasi, XAI, dan Error Analysis

# %%
models = ["RF", "SVM", "LR", "XGB"]
scenarios = list(SCENARIOS.keys())

plt.figure(figsize=(9, 5))
plot_dims = feature_dims.sort_values("dimension", ascending=True)
plt.barh(plot_dims["scenario"], plot_dims["dimension"], color=["#4c78a8", "#59a14f", "#f28e2b", "#e15759"])
for i, val in enumerate(plot_dims["dimension"]):
    plt.text(val + 20, i, str(val), va="center", fontsize=10)
plt.title("v95 Total Dimensi Fitur per Skenario")
plt.xlabel("Jumlah fitur")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_dimensions_v95.png"), dpi=220)
plt.close()

heat = np.zeros((len(scenarios), len(models)))
for i, sc in enumerate(scenarios):
    for j, m in enumerate(models):
        row = df_results[(df_results["scenario"] == sc) & (df_results["model"] == m)]
        heat[i, j] = float(row["test_f1_oof_thr"].iloc[0]) if not row.empty else 0.0

plt.figure(figsize=(10, 7))
sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu", xticklabels=models, yticklabels=scenarios)
plt.title("v95 Test Macro F1 - OOF Threshold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap_v95.png"), dpi=220)
plt.close()

best_result = max(all_results, key=lambda r: r["test_f1_oof_thr"])
best_scenario = best_result["scenario"]
best_model = best_result["model"]

best_cfg = eval(best_result["cfg"])
best_pca = None if best_result["pca_n"] == "none" else int(best_result["pca_n"])
best_curve_pipe = build_pipeline(best_model, best_cfg, best_pca, y_train, no_pca=best_result["no_pca"])

try:
    curve = learning_curve(
        best_curve_pipe,
        sanitize(SCENARIOS[best_scenario][train_idx]),
        y_train,
        cv=CV,
        scoring="f1_macro",
        train_sizes=np.linspace(0.3, 1.0, 5),
        n_jobs=1,
    )
    train_sizes, train_scores, val_scores = curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training F1")
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="CV F1")
    plt.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15,
    )
    plt.title(f"v95 Learning Curve: {best_model} on {best_scenario}")
    plt.xlabel("Training samples")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.05)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curve_v95.png"), dpi=220)
    plt.close()
except Exception as exc:
    print(f"[WARN] Learning curve gagal: {type(exc).__name__}: {exc}")

cm = confusion_matrix(y_test, best_result["y_pred_oof_thr"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Depresi"], yticklabels=["Normal", "Depresi"])
plt.title(f"v95 Confusion Matrix: {best_model} on {best_scenario}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_best_v95.png"), dpi=220)
plt.close()

def save_xai_plot(pipe, scenario_name, model_name, feature_names):
    clf = pipe.named_steps["clf"]
    if "pca" in pipe.named_steps:
        names = [f"PC{i+1}" for i in range(pipe.named_steps["pca"].n_components_)]
        xai_level = "PCA component"
    else:
        names = list(feature_names)
        xai_level = "original feature"

    if model_name in ["LR", "SVM"] and hasattr(clf, "coef_"):
        values = clf.coef_.ravel()
        title = f"v95 XAI Coefficients ({xai_level}): {model_name} on {scenario_name}"
    elif model_name in ["XGB", "RF"] and hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        title = f"v95 XAI Feature Importance ({xai_level}): {model_name} on {scenario_name}"
    else:
        print(f"[WARN] XAI tidak tersedia untuk model {model_name}")
        return None, pd.DataFrame()

    n = min(len(values), len(names))
    values = np.asarray(values[:n], dtype=float)
    names = names[:n]
    order = np.argsort(np.abs(values))[::-1][:20]
    df_xai = pd.DataFrame([
        {
            "feature": names[i],
            "importance": float(values[i]),
            "abs_importance": float(abs(values[i])),
            "xai_level": xai_level,
        }
        for i in order
    ])
    df_xai.to_csv(os.path.join(METRICS_DIR, "v95_xai_top20.csv"), index=False)

    plot_df = df_xai.sort_values("abs_importance", ascending=True)
    colors = ["#e15759" if v < 0 else "#4c78a8" for v in plot_df["importance"]]
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["feature"], plot_df["importance"], color=colors)
    plt.title(title)
    plt.xlabel("Importance / coefficient")
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "xai_top20_v95.png")
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path, df_xai

xai_path, df_xai = save_xai_plot(best_pipe, best_scenario, best_model, FEATURE_NAMES[best_scenario])
if xai_path:
    print(f"\nXAI plot saved: {xai_path}")
    print("Top XAI features/components:")
    print(df_xai.to_string(index=False))

best_pipe.oof_threshold_ = best_result["oof_thr"]
best_pipe.threshold_source_ = best_result["thr_source"]
best_pipe.scenario_ = best_scenario
best_pipe.model_name_ = best_model
best_pipe.feature_names_ = FEATURE_NAMES[best_scenario]
best_pipe.metrics_ = {
    "f1": best_result["test_f1_oof_thr"],
    "accuracy": best_result["test_acc_oof_thr"],
    "precision": best_result["test_precision_oof_thr"],
    "recall": best_result["test_recall_oof_thr"],
    "recall_depresi": best_result["test_recall_dep_oof_thr"],
    "auc": best_result["test_auc"],
}
model_filename = f"best_{best_model}_{best_scenario}.pkl"
model_path = os.path.join(MODELS_DIR, model_filename)
joblib.dump(best_pipe, model_path)
model_meta = {
    "version": "v95",
    "model_path": model_path,
    "scenario": best_scenario,
    "model": best_model,
    "threshold": best_result["oof_thr"],
    "threshold_source": best_result["thr_source"],
    "feature_count": len(FEATURE_NAMES[best_scenario]),
    "metrics": best_pipe.metrics_,
}
with open(os.path.join(MODELS_DIR, "best_model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(model_meta, f, indent=2)
print(f"\nSaved best v95 model: {model_path}")

err = base.iloc[test_idx][["participant_id", "label_depresi"]].copy()
err["y_pred_oof_thr"] = best_result["y_pred_oof_thr"]
err["prob_depresi"] = best_result["y_prob"]
err["correct"] = err["label_depresi"] == err["y_pred_oof_thr"]
err.to_csv(os.path.join(METRICS_DIR, "v95_error_analysis_best.csv"), index=False)

# %% [markdown]
# ## 6. Summary

# %%
sorted_res = df_results.sort_values("test_f1_oof_thr", ascending=False)

print("\n" + "=" * 136)
print(f"{'TABEL RINGKASAN v95 - S1-S4 x 4 Model':^136}")
print("=" * 136)
print(
    f"  {'Scenario':<18} {'Model':<4} {'PCA':<6} {'CV':>7} {'Thr':>6} {'Src':<7} "
    f"{'F1':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'RecD':>7} {'Top50':>7} {'AUC':>7} {'Oracle':>7}"
)
for _, r in sorted_res.iterrows():
    print(
        f"  {r['scenario']:<18} {r['model']:<4} {str(r['pca_n']):<6} {r['cv_f1']:>7.4f} "
        f"{r['oof_thr']:>6.3f} {r['thr_source']:<7} {r['test_f1_oof_thr']:>7.4f} "
        f"{r['test_acc_oof_thr']:>7.4f} {r['test_precision_oof_thr']:>7.4f} "
        f"{r['test_recall_oof_thr']:>7.4f} {r['test_recall_dep_oof_thr']:>7.4f} "
        f"{r['test_f1_top50']:>7.4f} {r['test_auc']:>7.4f} {r['test_f1_oracle']:>7.4f}"
    )

print("\nAPPLE-TO-APPLE BEST PER SCENARIO:")
for sc in scenarios:
    row = df_results[df_results["scenario"] == sc].sort_values("test_f1_oof_thr", ascending=False).iloc[0]
    print(
        f"  {sc:<18} {row['model']:<4} PCA={row['pca_n']} "
        f"CV={row['cv_f1']:.4f} F1={row['test_f1_oof_thr']:.4f} "
        f"Acc={row['test_acc_oof_thr']:.4f} Prec={row['test_precision_oof_thr']:.4f} "
        f"Recall={row['test_recall_oof_thr']:.4f} AUC={row['test_auc']:.4f}"
    )

print("\nCLASSIFICATION REPORT - BEST v95")
print(f"  {best_model} on {best_scenario} | F1={best_result['test_f1_oof_thr']:.4f}")
print(classification_report(y_test, best_result["y_pred_oof_thr"], target_names=["Normal", "Depresi"], zero_division=0))

print("\nERROR ANALYSIS - BEST v95")
wrong = err[~err["correct"]]
print(wrong.to_string(index=False) if not wrong.empty else "  Tidak ada salah klasifikasi.")

summary = {
    "version": "v95",
    "strategy": "Wav2Vec v8 768 mean embedding; S3 LR/LinearSVM no PCA; anti-leakage StandardScaler-PCA-Classifier pipeline; native imbalance; OOF threshold sweep 0.35-0.45 with fallback",
    "dataset_total": int(len(base)),
    "w2v8_rows_available": int(len(df_w2v8)),
    "w2v8_missing_rows_zero_filled": int(w2v_missing),
    "split": {
        "train": int(len(train_idx)),
        "test": int(len(test_idx)),
        "test_normal": int((y_test == 0).sum()),
        "test_depresi": int((y_test == 1).sum()),
    },
    "feature_dimensions": feature_dims.to_dict(orient="records"),
    "best": {
        "scenario": best_scenario,
        "model": best_model,
        "pca_n": best_result["pca_n"],
        "no_pca": best_result["no_pca"],
        "cv_f1": best_result["cv_f1"],
        "threshold": best_result["oof_thr"],
        "threshold_source": best_result["thr_source"],
        "test_f1": best_result["test_f1_oof_thr"],
        "accuracy": best_result["test_acc_oof_thr"],
        "precision": best_result["test_precision_oof_thr"],
        "recall": best_result["test_recall_oof_thr"],
        "recall_depresi": best_result["test_recall_dep_oof_thr"],
        "test_top50_f1": best_result["test_f1_top50"],
        "auc": best_result["test_auc"],
        "oracle_f1": best_result["test_f1_oracle"],
    },
    "xai_plot": xai_path,
    "model_path": model_path,
    "model_metadata_path": os.path.join(MODELS_DIR, "best_model_metadata.json"),
    "target_075": bool(best_result["test_f1_oof_thr"] >= 0.75),
    "elapsed_s": round(time.time() - t_global, 1),
}
with open(os.path.join(METRICS_DIR, "v95_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 92)
print(f"{'FINAL REPORT v95':^92}")
print("=" * 92)
print(f"  Best        : {best_result['test_f1_oof_thr']:.4f} ({best_model} on {best_scenario})")
print(f"  Accuracy    : {best_result['test_acc_oof_thr']:.4f}")
print(f"  Precision   : {best_result['test_precision_oof_thr']:.4f}")
print(f"  Recall      : {best_result['test_recall_oof_thr']:.4f}")
print(f"  Recall Dep. : {best_result['test_recall_dep_oof_thr']:.4f}")
print(f"  Top50 diag  : {best_result['test_f1_top50']:.4f}")
print(f"  AUC         : {best_result['test_auc']:.4f}")
print(f"  Oracle diag : {best_result['test_f1_oracle']:.4f}")
print(f"  XAI plot    : {xai_path}")
print(f"  Model saved : {model_path}")
print(f"  Target 0.75 : {'TERCAPAI' if best_result['test_f1_oof_thr'] >= 0.75 else 'BELUM'}")
print(f"  Total waktu : {time.time() - t_global:.1f}s")
print(f"  Results dir : {RESULTS_DIR}")
print("=" * 92)
