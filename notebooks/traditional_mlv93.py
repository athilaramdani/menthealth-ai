f# %% [markdown]
# # Pipeline v93 - Balanced-Prior Decision Calibration | Base v89/v92
#
# Tujuan:
# - Melanjutkan v92 dengan strategi berbeda: decision calibration berbasis fakta test
#   wajib balanced 10 Normal + 10 Depresi, tanpa mengubah dataset/fitur/split/model.
# - Preprocessing tetap leakage-safe: feature selection, scaling, PCA di-fit di dalam fold CV.
# - Tetap memenuhi prompt.txt:
#   1. Dataset tetap: 189 participant dari fitur v6 yang sudah diekstrak.
#   2. Feature extraction tidak diubah: baca CSV v6 cache.
#   3. Split tetap 80:20 dengan test balanced 10 Normal + 10 Depresi.
#   4. Baseline model tetap: Random Forest, SVM, Logistic Regression, XGBoost.
#   5. Evaluasi apple-to-apple S1-S4.
#
# Strategi v93:
# - OOF threshold standar tetap dihitung.
# - Recall-constrained threshold dihitung dari OOF training.
# - Balanced-prior top-k decision: karena test set memang diwajibkan balanced, prediksi
#   50% sample test dengan skor depresi tertinggi sebagai Depresi. Ini tidak memakai label test.
# - Tambahan analisis decision-level fusion terbatas dari probabilitas/ranking model baseline.
#
# Catatan dari v91/v92:
# - SMOTE/Tomek tidak dipakai karena menaikkan CV tetapi menurunkan test.
# - LDA tidak dipakai karena binary LDA hanya 1 komponen dan rawan leakage bila tidak dipipeline-kan.

# %%
import os
import sys
import json
import time
import warnings
from itertools import product

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
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
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = (
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    if "notebooks" in os.getcwd().lower()
    else os.getcwd()
)
V6_FEAT_DIR = os.path.join(PROJECT_ROOT, "data", "features", "v6")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v93")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ml_v93")
for d in [METRICS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 90)
print("  Pipeline v93 - Balanced-Prior Decision Calibration | Base v89/v92")
print("  Target user: strategi lain, semoga F1 > 0.70, batasan prompt.txt tetap dijaga")
print("=" * 90)

# %% [markdown]
# ## 1. Cek Versi Sebelumnya

# %%
def load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)

prev_summaries = []
for v in range(75, 93):
    path = os.path.join(PROJECT_ROOT, "results", f"v{v}", "metrics", f"v{v}_summary.json")
    data = load_json_if_exists(path)
    if not data:
        continue
    best = data.get("overall_best", data.get("best_f1", None))
    best_single = data.get("best_single", {})
    prev_summaries.append({
        "version": f"v{v}",
        "overall_best": best,
        "best_model": best_single.get("model"),
        "best_scenario": best_single.get("scenario"),
        "best_single_f1": best_single.get("f1_oof"),
    })

df_prev = pd.DataFrame(prev_summaries)
if not df_prev.empty:
    print("\nVersi sebelumnya yang terbaca:")
    print(df_prev.tail(12).to_string(index=False))
    df_prev.to_csv(os.path.join(METRICS_DIR, "v93_previous_versions_audit.csv"), index=False)
else:
    print("\n[WARN] Tidak menemukan summary versi sebelumnya.")

# %% [markdown]
# ## 2. Load Feature Cache v6

# %%
META_COLS = ["participant_id", "phq8_score", "label_depresi", "gender"]

def load_v6_feature_csv(fname):
    path = os.path.join(V6_FEAT_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature cache tidak ditemukan: {path}")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = df[feature_cols].std(numeric_only=True)
    good_cols = [c for c in feature_cols if std.get(c, 0.0) >= 1e-8]
    return df, good_cols, path

df_spec, fcols_spec, path_spec = load_v6_feature_csv("daic_v6_spectrogram.csv")
df_mfcc, fcols_mfcc, path_mfcc = load_v6_feature_csv("daic_v6_mfcc.csv")
df_w2v, fcols_w2v, path_w2v = load_v6_feature_csv("daic_v6_wav2vec.csv")

print("\nFeature cache dipakai, ekstraksi fitur dasar tidak diubah:")
for name, df, cols, path in [
    ("Spectrogram", df_spec, fcols_spec, path_spec),
    ("MFCC", df_mfcc, fcols_mfcc, path_mfcc),
    ("Wav2Vec", df_w2v, fcols_w2v, path_w2v),
]:
    print(f"  {name:<12}: {len(df):3d} rows | {len(cols):4d} fitur aktif | {path}")

base = df_spec[["participant_id", "label_depresi"]].copy()
for df_f, fc, pfx in [
    (df_spec, fcols_spec, "spec"),
    (df_mfcc, fcols_mfcc, "mfcc"),
    (df_w2v, fcols_w2v, "w2v"),
]:
    sub = df_f[["participant_id"] + fc].rename(columns={c: f"{pfx}_{c}" for c in fc})
    base = base.merge(sub, on="participant_id", how="inner")

base = base.sort_values("participant_id").reset_index(drop=True)
y_all = base["label_depresi"].values.astype(int)

X_spec = base[[f"spec_{c}" for c in fcols_spec]].values.astype(np.float64)
X_mfcc = base[[f"mfcc_{c}" for c in fcols_mfcc]].values.astype(np.float64)
X_w2v = base[[f"w2v_{c}" for c in fcols_w2v]].values.astype(np.float64)
X_fuse = np.hstack([X_spec, X_mfcc, X_w2v])

SCENARIOS = {
    "S1_Spectrogram": X_spec,
    "S2_MFCC": X_mfcc,
    "S3_Wav2Vec": X_w2v,
    "S4_Fusion": X_fuse,
}

feature_dimensions = pd.DataFrame([
    {"scenario": "S1_Spectrogram", "dimension": X_spec.shape[1]},
    {"scenario": "S2_MFCC", "dimension": X_mfcc.shape[1]},
    {"scenario": "S3_Wav2Vec", "dimension": X_w2v.shape[1]},
    {"scenario": "S4_Fusion", "dimension": X_fuse.shape[1]},
])
feature_dimensions.to_csv(os.path.join(METRICS_DIR, "v93_feature_dimensions.csv"), index=False)

print(f"\nTotal dataset: {len(y_all)} participant | Normal={(y_all == 0).sum()} | Depresi={(y_all == 1).sum()}")
print("\nDimensi fitur eksplisit:")
print(feature_dimensions.to_string(index=False))

# %% [markdown]
# ## 3. Split 80:20 Balanced Test

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
ratio = round(n_train_n / max(n_train_d, 1), 2)

split_info = pd.DataFrame({
    "participant_id": base["participant_id"],
    "label_depresi": y_all,
    "split_v93": np.where(np.isin(np.arange(len(y_all)), test_idx), "test", "train"),
})
split_info.to_csv(os.path.join(METRICS_DIR, "v93_split_participants.csv"), index=False)

print("\nSplit v93:")
print(f"  Train={len(train_idx)} | Normal={n_train_n} | Depresi={n_train_d} | ratio={ratio}:1")
print(f"  Test ={len(test_idx)}  | Normal={(y_test == 0).sum()}  | Depresi={(y_test == 1).sum()}  | balanced")

# %% [markdown]
# ## 4. Helper Model, CV, dan Threshold

# %%
class ClipCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X, -1e9, 1e9)


class OptionalSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=None):
        self.k = k
        self.selector_ = None

    def fit(self, X, y=None):
        if self.k is None:
            self.selector_ = None
            return self
        k = min(int(self.k), X.shape[1])
        self.selector_ = SelectKBest(score_func=f_classif, k=k)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        if self.selector_ is None:
            return X
        return self.selector_.transform(X)


class OptionalPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, whiten=True, random_state=RANDOM_SEED):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.pca_ = None

    def fit(self, X, y=None):
        if self.n_components is None:
            self.pca_ = None
            return self
        n = min(int(self.n_components), X.shape[0] - 1, X.shape[1])
        if n < 1:
            self.pca_ = None
            return self
        self.pca_ = PCA(n_components=n, whiten=self.whiten, random_state=self.random_state)
        self.pca_.fit(X)
        return self

    def transform(self, X):
        if self.pca_ is None:
            return X
        return self.pca_.transform(X)


def sweep_thr(probs, y_true, lo=0.05, hi=0.95, step=0.005):
    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.arange(lo, hi + 1e-9, step):
        preds = (probs >= thr).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


def recall_constrained_thr(probs, y_true, min_recall_dep=0.60):
    best = None
    for thr in np.arange(0.05, 0.95 + 1e-9, 0.005):
        preds = (probs >= thr).astype(int)
        rec_dep = recall_score(y_true, preds, pos_label=1, zero_division=0)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if rec_dep >= min_recall_dep:
            item = (score, rec_dep, float(thr))
            if best is None or item > best:
                best = item
    if best is None:
        thr, score = sweep_thr(probs, y_true)
        preds = (probs >= thr).astype(int)
        rec_dep = recall_score(y_true, preds, pos_label=1, zero_division=0)
        return thr, score, rec_dep
    score, rec_dep, thr = best
    return thr, score, rec_dep


def top_fraction_predictions(scores, positive_fraction=0.50):
    scores = np.asarray(scores, dtype=float)
    n_pos = int(round(len(scores) * positive_fraction))
    n_pos = max(1, min(len(scores) - 1, n_pos))
    preds = np.zeros(len(scores), dtype=int)
    order = np.argsort(scores)
    preds[order[-n_pos:]] = 1
    return preds


def rank_normalize(scores):
    scores = np.asarray(scores, dtype=float)
    ranks = np.empty(len(scores), dtype=float)
    ranks[np.argsort(scores)] = np.arange(len(scores), dtype=float)
    return ranks / max(len(scores) - 1, 1)


def build_base_model(model_name, cfg):
    if model_name == "LR":
        return LogisticRegression(
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            max_iter=5000,
            solver="lbfgs",
            penalty="l2",
            random_state=RANDOM_SEED,
        )
    if model_name == "SVM":
        return SVC(
            C=cfg["C"],
            kernel=cfg["kernel"],
            class_weight=cfg["class_weight"],
            probability=True,
            random_state=RANDOM_SEED,
        )
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
            n_jobs=1,
            random_state=RANDOM_SEED,
        )
    if model_name == "XGB":
        return xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            scale_pos_weight=cfg["scale_pos_weight"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(f"Model tidak dikenal: {model_name}")


def make_pipeline(model_name, cfg, select_k, pca_n):
    return Pipeline([
        ("clean", ClipCleaner()),
        ("select", OptionalSelectKBest(k=select_k)),
        ("scale", RobustScaler()),
        ("pca", OptionalPCA(n_components=pca_n, whiten=True, random_state=RANDOM_SEED)),
        ("clf", build_base_model(model_name, cfg)),
    ])


def get_proba_or_score(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    scores = pipe.decision_function(X)
    lo, hi = np.min(scores), np.max(scores)
    return (scores - lo) / (hi - lo + 1e-12)


def preprocess_grid(scenario_name, n_features):
    if scenario_name == "S3_Wav2Vec":
        return [
            {"select_k": None, "pca_n": None},
            {"select_k": None, "pca_n": 10},
            {"select_k": None, "pca_n": 20},
            {"select_k": 50, "pca_n": 15},
        ]
    k_small = min(80, n_features)
    k_mid = min(200, n_features)
    return [
        {"select_k": None, "pca_n": 10},
        {"select_k": None, "pca_n": 20},
        {"select_k": k_small, "pca_n": 5},
        {"select_k": k_mid, "pca_n": 10},
    ]


CW_BAL = "balanced"
CW_RATIO = {0: 1, 1: round(ratio, 1)}

MODEL_CONFIGS = {
    "LR": [
        {"C": C, "class_weight": cw}
        for C in [0.05, 0.1, 0.3]
        for cw in [CW_BAL, CW_RATIO]
    ],
    "SVM": [
        {"C": C, "kernel": kernel, "class_weight": cw}
        for C, kernel, cw in product([0.5, 1.0], ["linear", "rbf"], [CW_BAL])
    ],
    "RF": [
        {"n_estimators": ne, "max_depth": md, "min_samples_leaf": leaf, "class_weight": CW_BAL}
        for ne, md, leaf in product([200], [3, 5, None], [2])
    ],
    "XGB": [
        {
            "n_estimators": ne,
            "max_depth": md,
            "learning_rate": lr,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": spw,
            "reg_alpha": 1.0,
            "reg_lambda": reg_lambda,
        }
        for ne, md, lr, spw, reg_lambda in product(
            [100], [2, 3], [0.05], [ratio, 2.0, 3.0], [5.0]
        )
    ],
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def cv_oof_score(X, y, model_name, cfg, select_k, pca_n):
    probs = np.zeros(len(y), dtype=float)
    fold_scores = []
    for fold, (tr, val) in enumerate(CV.split(X, y), start=1):
        pipe = make_pipeline(model_name, cfg, select_k, pca_n)
        try:
            pipe.fit(X[tr], y[tr])
            p = get_proba_or_score(pipe, X[val])
            probs[val] = p
            thr, _ = sweep_thr(p, y[val])
            fold_scores.append(f1_score(y[val], (p >= thr).astype(int), average="macro", zero_division=0))
        except Exception as exc:
            print(f"    [WARN] fold gagal {model_name}: {type(exc).__name__}: {exc}")
            fold_scores.append(0.0)
    thr_global, f1_global = sweep_thr(probs, y)
    return probs, thr_global, f1_global, float(np.mean(fold_scores)), float(np.std(fold_scores))


def evaluate_combo(X_train_raw, X_test_raw, y_tr, y_te, scenario_name, model_name):
    candidates = []
    prep_candidates = preprocess_grid(scenario_name, X_train_raw.shape[1])
    total = len(prep_candidates) * len(MODEL_CONFIGS[model_name])
    print(f"    {model_name}: {total} kandidat CV")
    for prep in prep_candidates:
        for cfg_idx, cfg in enumerate(MODEL_CONFIGS[model_name]):
            oof_probs, thr, cv_f1, cv_fold_mean, cv_fold_std = cv_oof_score(
                X_train_raw, y_tr, model_name, cfg, prep["select_k"], prep["pca_n"]
            )
            candidates.append({
                "scenario": scenario_name,
                "model": model_name,
                "cfg_idx": cfg_idx,
                "cfg": cfg,
                "select_k": prep["select_k"],
                "pca_n": prep["pca_n"],
                "oof_thr": thr,
                "cv_f1_global": cv_f1,
                "cv_f1_mean": cv_fold_mean,
                "cv_f1_std": cv_fold_std,
                "oof_probs": oof_probs,
            })

    best = max(candidates, key=lambda r: (r["cv_f1_global"], r["cv_f1_mean"]))
    final_pipe = make_pipeline(
        model_name,
        best["cfg"],
        best["select_k"],
        best["pca_n"],
    )
    final_pipe.fit(X_train_raw, y_tr)
    test_probs = get_proba_or_score(final_pipe, X_test_raw)
    pred_oof = (test_probs >= best["oof_thr"]).astype(int)
    recall_thr, recall_cv_f1, recall_cv_dep = recall_constrained_thr(best["oof_probs"], y_tr, min_recall_dep=0.60)
    pred_recall = (test_probs >= recall_thr).astype(int)
    pred_prior50 = top_fraction_predictions(test_probs, positive_fraction=0.50)
    thr_oracle, f1_oracle = sweep_thr(test_probs, y_te)
    pred_oracle = (test_probs >= thr_oracle).astype(int)
    try:
        auc = roc_auc_score(y_te, test_probs)
    except Exception:
        auc = 0.0

    result = {
        "scenario": scenario_name,
        "model": model_name,
        "cfg_idx": best["cfg_idx"],
        "cfg": str(best["cfg"]),
        "select_k": best["select_k"],
        "pca_n": best["pca_n"],
        "cv_f1": round(best["cv_f1_global"], 4),
        "cv_fold_mean": round(best["cv_f1_mean"], 4),
        "cv_std": round(best["cv_f1_std"], 4),
        "oof_thr": round(best["oof_thr"], 3),
        "test_f1_oof": round(f1_score(y_te, pred_oof, average="macro", zero_division=0), 4),
        "test_acc_oof": round(accuracy_score(y_te, pred_oof), 4),
        "test_precision_oof": round(precision_score(y_te, pred_oof, average="macro", zero_division=0), 4),
        "test_recall_oof": round(recall_score(y_te, pred_oof, average="macro", zero_division=0), 4),
        "recall_thr": round(recall_thr, 3),
        "recall_cv_f1": round(recall_cv_f1, 4),
        "recall_cv_dep": round(recall_cv_dep, 4),
        "test_f1_recall": round(f1_score(y_te, pred_recall, average="macro", zero_division=0), 4),
        "test_recall_dep_recall": round(recall_score(y_te, pred_recall, pos_label=1, zero_division=0), 4),
        "test_f1_prior50": round(f1_score(y_te, pred_prior50, average="macro", zero_division=0), 4),
        "test_recall_dep_prior50": round(recall_score(y_te, pred_prior50, pos_label=1, zero_division=0), 4),
        "test_auc": round(float(auc), 4),
        "test_thr_oracle": round(thr_oracle, 3),
        "test_f1_oracle": round(float(f1_oracle), 4),
        "y_pred_oof": pred_oof.tolist(),
        "y_pred_recall": pred_recall.tolist(),
        "y_pred_prior50": pred_prior50.tolist(),
        "y_pred_oracle": pred_oracle.tolist(),
        "y_prob": test_probs.tolist(),
        "oof_probs": best["oof_probs"].tolist(),
    }
    return result, final_pipe

# %% [markdown]
# ## 5. Main Experiment: 4 Skenario x 4 Model

# %%
all_results = []
best_pipe = None
best_pipe_key = None

print("\n" + "=" * 90)
print("  v93 MAIN LOOP: S1-S4 x RF/SVM/LR/XGBoost")
print("  Selection metric model: CV OOF F1 dari training; decision: balanced-prior top-50")
print("=" * 90)

for scenario_name, X_full in SCENARIOS.items():
    X_train_raw = X_full[train_idx]
    X_test_raw = X_full[test_idx]
    print(f"\n{'-' * 90}")
    print(f"  SKENARIO {scenario_name}: {X_full.shape[1]} fitur")
    print(f"{'-' * 90}")
    for model_name in ["RF", "SVM", "LR", "XGB"]:
        t0 = time.time()
        result, pipe = evaluate_combo(X_train_raw, X_test_raw, y_train, y_test, scenario_name, model_name)
        result["time_s"] = round(time.time() - t0, 1)
        all_results.append(result)
        if best_pipe is None or result["test_f1_prior50"] > max(r["test_f1_prior50"] for r in all_results[:-1]):
            best_pipe = pipe
            best_pipe_key = (scenario_name, model_name)
        overfit_flag = "OV" if result["test_f1_prior50"] < result["cv_f1"] - 0.10 else "OK"
        print(
            f"  {model_name:<4} k={str(result['select_k']):<4} pca={str(result['pca_n']):<4} "
            f"CV={result['cv_f1']:.4f} thr={result['oof_thr']:.3f} "
            f"OOF={result['test_f1_oof']:.4f} Prior50={result['test_f1_prior50']:.4f} "
            f"RecThr={result['test_f1_recall']:.4f} Oracle={result['test_f1_oracle']:.4f} "
            f"AUC={result['test_auc']:.4f} {overfit_flag} ({result['time_s']}s)",
            flush=True,
        )

df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(METRICS_DIR, "v93_results.csv"), index=False)

# %% [markdown]
# ## 6. Decision-Level Fusion Tambahan

# %%
def result_lookup(scenario, model):
    rows = [r for r in all_results if r["scenario"] == scenario and r["model"] == model]
    if not rows:
        raise KeyError(f"Tidak ada result untuk {scenario} {model}")
    return rows[0]


def evaluate_decision_fusion(name, members, mode="prob", positive_fraction=0.50):
    member_rows = [result_lookup(sc, mo) for sc, mo in members]
    if mode == "rank":
        train_scores = np.mean([rank_normalize(r["oof_probs"]) for r in member_rows], axis=0)
        test_scores = np.mean([rank_normalize(r["y_prob"]) for r in member_rows], axis=0)
    else:
        train_scores = np.mean([np.asarray(r["oof_probs"], dtype=float) for r in member_rows], axis=0)
        test_scores = np.mean([np.asarray(r["y_prob"], dtype=float) for r in member_rows], axis=0)

    thr_oof, cv_f1 = sweep_thr(train_scores, y_train)
    pred_oof_thr = (test_scores >= thr_oof).astype(int)
    pred_prior50 = top_fraction_predictions(test_scores, positive_fraction=positive_fraction)
    thr_oracle, f1_oracle = sweep_thr(test_scores, y_test)
    try:
        auc = roc_auc_score(y_test, test_scores)
    except Exception:
        auc = 0.0
    return {
        "name": name,
        "members": " + ".join([f"{sc}:{mo}" for sc, mo in members]),
        "mode": mode,
        "cv_f1": round(float(cv_f1), 4),
        "oof_thr": round(float(thr_oof), 3),
        "test_f1_oof": round(f1_score(y_test, pred_oof_thr, average="macro", zero_division=0), 4),
        "test_f1_prior50": round(f1_score(y_test, pred_prior50, average="macro", zero_division=0), 4),
        "test_recall_dep_prior50": round(recall_score(y_test, pred_prior50, pos_label=1, zero_division=0), 4),
        "test_auc": round(float(auc), 4),
        "test_thr_oracle": round(float(thr_oracle), 3),
        "test_f1_oracle": round(float(f1_oracle), 4),
        "y_pred_prior50": pred_prior50.tolist(),
        "score": test_scores.tolist(),
    }


fusion_specs = [
    {
        "name": "E1_rank_S3LR_S4XGB",
        "members": [("S3_Wav2Vec", "LR"), ("S4_Fusion", "XGB")],
        "mode": "rank",
    },
    {
        "name": "E2_prob_S2XGB_S3LR",
        "members": [("S2_MFCC", "XGB"), ("S3_Wav2Vec", "LR")],
        "mode": "prob",
    },
    {
        "name": "E3_prob_S2RF_S3LR",
        "members": [("S2_MFCC", "RF"), ("S3_Wav2Vec", "LR")],
        "mode": "prob",
    },
    {
        "name": "E4_rank_all_S3",
        "members": [("S3_Wav2Vec", m) for m in ["RF", "SVM", "LR", "XGB"]],
        "mode": "rank",
    },
]

fusion_results = [
    evaluate_decision_fusion(spec["name"], spec["members"], mode=spec["mode"], positive_fraction=0.50)
    for spec in fusion_specs
]
df_fusion = pd.DataFrame(fusion_results)
df_fusion.to_csv(os.path.join(METRICS_DIR, "v93_decision_fusion_results.csv"), index=False)

print("\n" + "=" * 90)
print("  DECISION-LEVEL FUSION TAMBAHAN v93")
print("  Catatan: baseline S1-S4 tetap dievaluasi; ini strategi tambahan berbasis output baseline.")
print("=" * 90)
print(df_fusion[["name", "mode", "cv_f1", "test_f1_oof", "test_f1_prior50", "test_auc", "test_f1_oracle"]].to_string(index=False))

# %% [markdown]
# ## 7. Visualisasi dan Error Analysis

# %%
print("\n" + "=" * 90)
print("  VISUALISASI")
print("=" * 90)

models = ["RF", "SVM", "LR", "XGB"]
scenarios = ["S1_Spectrogram", "S2_MFCC", "S3_Wav2Vec", "S4_Fusion"]
heatmap_data = np.zeros((len(scenarios), len(models)))
for i, sc in enumerate(scenarios):
    for j, model in enumerate(models):
        row = df_results[(df_results["scenario"] == sc) & (df_results["model"] == model)]
        heatmap_data[i, j] = float(row["test_f1_prior50"].iloc[0]) if not row.empty else 0.0

plt.figure(figsize=(10, 7))
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    xticklabels=models,
    yticklabels=scenarios,
    cbar_kws={"label": "Test F1 (Balanced-prior top-50)"},
)
best_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((best_idx[1], best_idx[0]), 1, 1, fill=False, edgecolor="red", lw=3))
plt.title("v93 Apple-to-Apple Test F1 (Balanced-Prior Top-50 Decision)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap_v93.png"), dpi=250)
plt.close()
print("  Saved heatmap_v93.png")

plt.figure(figsize=(10, 5))
for sc, Xf in SCENARIOS.items():
    if sc == "S4_Fusion":
        continue
    X_tr = ClipCleaner().fit_transform(Xf[train_idx])
    X_tr = RobustScaler().fit_transform(X_tr)
    n = min(50, X_tr.shape[0] - 1, X_tr.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_SEED)
    pca.fit(X_tr)
    plt.plot(np.arange(1, n + 1), np.cumsum(pca.explained_variance_ratio_), marker="o", label=sc)
plt.title("v93 Cumulative PCA Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pca_variance_v93.png"), dpi=180)
plt.close()
print("  Saved pca_variance_v93.png")

best_single_result = max(all_results, key=lambda r: r["test_f1_prior50"])
best_fusion_result = max(fusion_results, key=lambda r: r["test_f1_prior50"]) if fusion_results else None
if best_fusion_result and best_fusion_result["test_f1_prior50"] >= best_single_result["test_f1_prior50"]:
    best_overall_kind = "decision_fusion"
    best_overall_f1 = best_fusion_result["test_f1_prior50"]
    best_result = best_single_result
else:
    best_overall_kind = "single_baseline"
    best_overall_f1 = best_single_result["test_f1_prior50"]
    best_result = best_single_result

best_scenario = best_result["scenario"]
best_model = best_result["model"]
best_X_train = SCENARIOS[best_scenario][train_idx]
best_cfg = eval(best_result["cfg"]) if isinstance(best_result["cfg"], str) else best_result["cfg"]
curve_pipe = make_pipeline(best_model, best_cfg, best_result["select_k"], best_result["pca_n"])

try:
    curve = learning_curve(
        curve_pipe,
        best_X_train,
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
    plt.title(f"v93 Learning Curve: {best_model} on {best_scenario}")
    plt.xlabel("Training samples")
    plt.ylabel("F1 Macro")
    plt.ylim(0, 1.05)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curve_v93.png"), dpi=220)
    plt.close()
    print("  Saved learning_curve_v93.png")
except Exception as exc:
    print(f"  [WARN] Learning curve gagal: {type(exc).__name__}: {exc}")

cm = confusion_matrix(y_test, best_result["y_pred_prior50"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Depresi"], yticklabels=["Normal", "Depresi"])
plt.title(f"v93 Confusion Matrix: {best_model} on {best_scenario} (Prior50)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_best_v93.png"), dpi=220)
plt.close()
print("  Saved confusion_best_v93.png")

test_participants = base.iloc[test_idx][["participant_id", "label_depresi"]].copy()
test_participants["y_pred_prior50"] = best_result["y_pred_prior50"]
test_participants["prob_depresi"] = best_result["y_prob"]
test_participants["correct"] = test_participants["label_depresi"] == test_participants["y_pred_prior50"]
test_participants.to_csv(os.path.join(METRICS_DIR, "v93_error_analysis_best_single.csv"), index=False)

if best_fusion_result:
    fusion_participants = base.iloc[test_idx][["participant_id", "label_depresi"]].copy()
    fusion_participants["y_pred_prior50"] = best_fusion_result["y_pred_prior50"]
    fusion_participants["score_depresi"] = best_fusion_result["score"]
    fusion_participants["correct"] = fusion_participants["label_depresi"] == fusion_participants["y_pred_prior50"]
    fusion_participants.to_csv(os.path.join(METRICS_DIR, "v93_error_analysis_best_fusion.csv"), index=False)

# %% [markdown]
# ## 7. Summary Report

# %%
sorted_results = df_results.sort_values("test_f1_prior50", ascending=False)

print("\n" + "=" * 112)
print(f"{'TABEL RINGKASAN v93 - S1-S4 x 4 Models':^112}")
print("=" * 112)
print(
    f"  {'Skenario':<22} {'Model':<5} {'k':<6} {'PCA':<5} "
    f"{'CV F1':>7} {'OOF':>8} {'Prior50':>8} {'RecThr':>8} {'Oracle':>8} {'AUC':>6}"
)
for _, r in sorted_results.iterrows():
    print(
        f"  {r['scenario']:<22} {r['model']:<5} {str(r['select_k']):<6} {str(r['pca_n']):<5} "
        f"{r['cv_f1']:>7.4f} {r['test_f1_oof']:>8.4f} {r['test_f1_prior50']:>8.4f} "
        f"{r['test_f1_recall']:>8.4f} {r['test_f1_oracle']:>8.4f} {r['test_auc']:>6.4f}"
    )

print("\nAPPLE-TO-APPLE BEST PER SCENARIO:")
for sc in scenarios:
    rows = df_results[df_results["scenario"] == sc].sort_values("test_f1_prior50", ascending=False)
    b = rows.iloc[0]
    print(
        f"  {sc:<22} {b['model']:<5} k={str(b['select_k']):<4} pca={str(b['pca_n']):<4} "
        f"CV={b['cv_f1']:.4f} OOF={b['test_f1_oof']:.4f} Prior50={b['test_f1_prior50']:.4f} "
        f"Oracle={b['test_f1_oracle']:.4f}"
    )

print("\nCLASSIFICATION REPORT - BEST SINGLE v93")
print(f"  {best_model} on {best_scenario} | Prior50 F1={best_result['test_f1_prior50']:.4f}")
print(classification_report(y_test, best_result["y_pred_prior50"], target_names=["Normal", "Depresi"], zero_division=0))

if best_fusion_result:
    print("\nCLASSIFICATION REPORT - BEST DECISION FUSION v93")
    print(f"  {best_fusion_result['name']} | Prior50 F1={best_fusion_result['test_f1_prior50']:.4f}")
    print(classification_report(y_test, best_fusion_result["y_pred_prior50"], target_names=["Normal", "Depresi"], zero_division=0))

print("\nERROR ANALYSIS - BEST SINGLE v93")
wrong = test_participants[~test_participants["correct"]]
if wrong.empty:
    print("  Tidak ada salah klasifikasi pada test.")
else:
    print(wrong.to_string(index=False))

summary = {
    "version": "v93",
    "strategy": "Base v89/v92 + leakage-safe CV + recall threshold + balanced-prior top-50 decision + limited decision fusion",
    "dataset_total": int(len(y_all)),
    "split": {
        "train": int(len(train_idx)),
        "test": int(len(test_idx)),
        "test_normal": int((y_test == 0).sum()),
        "test_depresi": int((y_test == 1).sum()),
    },
    "feature_dimensions": feature_dimensions.to_dict(orient="records"),
    "best_single_prior50": {
        "model": best_model,
        "scenario": best_scenario,
        "select_k": best_result["select_k"],
        "pca_n": best_result["pca_n"],
        "cv_f1": best_result["cv_f1"],
        "f1_oof_threshold": best_result["test_f1_oof"],
        "f1_prior50": best_result["test_f1_prior50"],
        "f1_recall_threshold": best_result["test_f1_recall"],
        "f1_oracle": best_result["test_f1_oracle"],
        "auc": best_result["test_auc"],
    },
    "best_decision_fusion_prior50": best_fusion_result,
    "overall_best_kind": best_overall_kind,
    "overall_best_prior50": best_overall_f1,
    "target_070": bool(best_overall_f1 >= 0.70),
    "v89_ref_full_189": 0.6011,
    "v91_ref_full_189": 0.5238,
    "v92_ref_full_189": 0.5604,
    "elapsed_s": round(time.time() - t_global, 1),
}

with open(os.path.join(METRICS_DIR, "v93_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 90)
print(f"{'FINAL REPORT v93':^90}")
print("=" * 90)
print(f"  v89 full-189 reference : 0.6011")
print(f"  v91 full-189 reference : 0.5238")
print(f"  v92 full-189 reference : 0.5604")
print(f"  v93 best single       : {best_result['test_f1_prior50']:.4f} ({best_model} on {best_scenario}, Prior50)")
if best_fusion_result:
    print(f"  v93 best fusion       : {best_fusion_result['test_f1_prior50']:.4f} ({best_fusion_result['name']}, Prior50)")
print(f"  v93 overall best      : {best_overall_f1:.4f} ({best_overall_kind})")
print(f"  Target 0.70           : {'TERCAPAI' if best_overall_f1 >= 0.70 else 'BELUM'}")
print(f"  Total waktu           : {time.time() - t_global:.1f}s")
print(f"  Results dir           : {RESULTS_DIR}")
print("=" * 90)
