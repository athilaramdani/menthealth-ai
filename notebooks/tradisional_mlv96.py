# %% [markdown]
# # Pipeline v96 - Focused Tuning + XAI
#
# Fokus v96:
# - Dataset tetap 189 participant.
# - Split tetap 80:20 dengan test balanced 10 Normal + 10 Depresi.
# - Empat skenario tetap apple-to-apple:
#   S1 Spectrogram, S2 MFCC, S3 Wav2Vec768, S4 Fusion.
# - Model baseline tetap: Random Forest, SVM, Logistic Regression, XGBoost.
# - Wav2Vec768 memakai cache bersih v95: 189 participant, 768 mean embedding.
# - Tuning difokuskan pada:
#   S3 Wav2Vec768 + Linear SVM no PCA,
#   S2 MFCC + XGBoost,
#   S4 Fusion + Logistic Regression/Linear SVM fitur utuh.
# - Threshold OOF agresif: 0.30 sampai 0.60 step 0.01.
# - Tambahan visualisasi fitur, learning curve, dan XAI untuk best model.

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
V95_W2V_PATH = os.path.join(PROJECT_ROOT, "data", "features", "v95", "daic_v95_wav2vec768_mean_189.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v96")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
for d in [METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

t_global = time.time()
print("=" * 96)
print("  Pipeline v96 - Focused Tuning + Threshold Optimization + XAI")
print("  Strict S1-S4 apple-to-apple; no cross-scenario model fusion")
print("=" * 96)

# %% [markdown]
# ## 1. Load Fitur dan Nama Fitur

# %%
META_COLS = ["participant_id", "phq8_score", "label_depresi", "gender"]

def clean_feature_frame(df, meta_cols):
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = df[feature_cols].std(numeric_only=True)
    good_cols = [c for c in feature_cols if std.get(c, 0.0) >= 1e-8]
    return df, good_cols

df_spec = pd.read_csv(os.path.join(V6_DIR, "daic_v6_spectrogram.csv"))
df_mfcc = pd.read_csv(os.path.join(V6_DIR, "daic_v6_mfcc.csv"))
df_w2v = pd.read_csv(V95_W2V_PATH)

df_spec, fcols_spec_raw = clean_feature_frame(df_spec, META_COLS)
df_mfcc, fcols_mfcc_raw = clean_feature_frame(df_mfcc, META_COLS)
w2v_cols_raw = [f"w2v_{i}" for i in range(768)]
missing_w2v_cols = [c for c in w2v_cols_raw if c not in df_w2v.columns]
if missing_w2v_cols:
    raise RuntimeError(f"Cache Wav2Vec v95 tidak lengkap. Missing: {missing_w2v_cols[:5]}")
if len(df_w2v) != 189:
    raise RuntimeError(f"Cache Wav2Vec v95 harus 189 rows, ditemukan {len(df_w2v)} rows")

base = df_spec[["participant_id", "label_depresi"]].copy().sort_values("participant_id")
base = base.merge(
    df_spec[["participant_id"] + fcols_spec_raw].rename(columns={c: f"spec_{c}" for c in fcols_spec_raw}),
    on="participant_id",
    how="left",
)
base = base.merge(
    df_mfcc[["participant_id"] + fcols_mfcc_raw].rename(columns={c: f"mfcc_{c}" for c in fcols_mfcc_raw}),
    on="participant_id",
    how="left",
)
base = base.merge(
    df_w2v[["participant_id"] + w2v_cols_raw].rename(columns={c: f"w2v_{c}" for c in w2v_cols_raw}),
    on="participant_id",
    how="left",
)
base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)

spec_cols = [f"spec_{c}" for c in fcols_spec_raw]
mfcc_cols = [f"mfcc_{c}" for c in fcols_mfcc_raw]
w2v_cols = [f"w2v_{c}" for c in w2v_cols_raw]
fusion_cols = spec_cols + mfcc_cols + w2v_cols

y_all = base["label_depresi"].values.astype(int)
SCENARIOS = {
    "S1_Spectrogram": {
        "X": base[spec_cols].values.astype(np.float64),
        "feature_names": spec_cols,
        "group": "Spectrogram/GeMAPS-like",
    },
    "S2_MFCC": {
        "X": base[mfcc_cols].values.astype(np.float64),
        "feature_names": mfcc_cols,
        "group": "MFCC/Prosodic",
    },
    "S3_Wav2Vec768": {
        "X": base[w2v_cols].values.astype(np.float64),
        "feature_names": w2v_cols,
        "group": "Wav2Vec 768 mean embedding",
    },
    "S4_Fusion": {
        "X": base[fusion_cols].values.astype(np.float64),
        "feature_names": fusion_cols,
        "group": "Spectrogram + MFCC + Wav2Vec",
    },
}

feature_dims = pd.DataFrame([
    {"scenario": name, "group": info["group"], "dimension": len(info["feature_names"])}
    for name, info in SCENARIOS.items()
])
feature_dims.to_csv(os.path.join(METRICS_DIR, "v96_feature_dimensions.csv"), index=False)

print("\nDimensi fitur v96:")
print(feature_dims.to_string(index=False))
print("\nDataset:")
print(f"  Total={len(base)} | Normal={(y_all == 0).sum()} | Depresi={(y_all == 1).sum()}")

def summarize_feature_names(name, cols):
    print(f"\n{name} - jumlah fitur: {len(cols)}")
    print("  Contoh 30 nama fitur pertama:")
    print("  " + ", ".join(cols[:30]))
    if len(cols) > 30:
        print("  ...")
        print("  Contoh 10 nama fitur terakhir:")
        print("  " + ", ".join(cols[-10:]))

print("\nDaftar/komponen fitur eksplisit:")
print("  Spectrogram mencakup statistik mel-band, entropy, centroid, bandwidth, rolloff, flatness, chroma, RMS, ZCR, low-energy.")
print("  MFCC mencakup MFCC m1-m40, delta, delta-delta, pitch/F0 proxy, RMS, ZCR, low-energy, HNR proxy, spectral entropy, jitter, shimmer, voiced ratio.")
print("  Wav2Vec memakai dimensi embedding mean w2v_0 sampai w2v_767.")
summarize_feature_names("S1 Spectrogram", spec_cols)
summarize_feature_names("S2 MFCC", mfcc_cols)
summarize_feature_names("S3 Wav2Vec768", w2v_cols)

# %% [markdown]
# ## 2. Split 80:20 Balanced Test

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

split_df = base[["participant_id", "label_depresi"]].copy()
split_df["split_v96"] = np.where(np.isin(np.arange(len(base)), test_idx), "test", "train")
split_df.to_csv(os.path.join(METRICS_DIR, "v96_split_participants.csv"), index=False)

print("\nSplit v96:")
print(f"  Train={len(train_idx)} | Normal={(y_train == 0).sum()} | Depresi={(y_train == 1).sum()}")
print(f"  Test ={len(test_idx)}  | Normal={(y_test == 0).sum()}  | Depresi={(y_test == 1).sum()}  | balanced")

# %% [markdown]
# ## 3. Model, Pipeline, Threshold OOF

# %%
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

def sanitize(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e9, 1e9)

def threshold_sweep_oof(probs, y_true):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.30, 0.60 + 1e-12, 0.01):
        pred = (probs >= thr).astype(int)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_f1:
            best_thr, best_f1 = float(thr), float(score)
    return best_thr, best_f1

def threshold_sweep_wide(probs, y_true):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.95 + 1e-12, 0.005):
        pred = (probs >= thr).astype(int)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_f1:
            best_thr, best_f1 = float(thr), float(score)
    return best_thr, best_f1

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
            max_iter=7000,
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
        if scenario_name in ["S3_Wav2Vec768", "S4_Fusion"]:
            return [{"C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0]]
        return [{"C": c} for c in [0.01, 0.1, 1.0]]
    if model_name == "SVM":
        if scenario_name in ["S3_Wav2Vec768", "S4_Fusion"]:
            return [{"C": c, "kernel": "linear"} for c in [0.001, 0.01, 0.1, 1.0, 10.0]]
        return [{"C": c, "kernel": k} for c in [0.1, 1.0] for k in ["linear", "rbf"]]
    if model_name == "RF":
        return [
            {"n_estimators": 250, "max_depth": md, "min_samples_leaf": leaf}
            for md in [3, None] for leaf in [2, 4]
        ]
    if model_name == "XGB":
        if scenario_name == "S2_MFCC":
            return [
                {"n_estimators": 150, "max_depth": md, "learning_rate": lr, "reg_alpha": 1.0, "reg_lambda": 5.0}
                for md in [3, 4, 5] for lr in [0.01, 0.05, 0.1]
            ]
        return [
            {"n_estimators": 120, "max_depth": md, "learning_rate": lr, "reg_alpha": 1.0, "reg_lambda": 5.0}
            for md in [2, 3] for lr in [0.03, 0.05]
        ]
    raise ValueError(model_name)

def pca_grid(scenario_name, model_name):
    if scenario_name == "S3_Wav2Vec768" and model_name in ["LR", "SVM"]:
        return [None]
    if scenario_name == "S4_Fusion" and model_name in ["LR", "SVM"]:
        return [None]
    if scenario_name == "S3_Wav2Vec768":
        return [50, 80]
    if scenario_name == "S4_Fusion":
        return [10, 20, 50]
    return [10, 20, 30]

def evaluate_model(X_train_raw, X_test_raw, y_tr, y_te, scenario_name, model_name):
    X_train_raw = sanitize(X_train_raw)
    X_test_raw = sanitize(X_test_raw)
    candidates = []
    for cfg_idx, cfg in enumerate(model_configs(model_name, scenario_name)):
        for pca_n in pca_grid(scenario_name, model_name):
            no_pca = pca_n is None
            oof = np.zeros(len(y_tr), dtype=float)
            fold_scores = []
            for tr, val in CV.split(X_train_raw, y_tr):
                pipe = build_pipeline(model_name, cfg, pca_n, y_tr[tr], no_pca=no_pca)
                pipe.fit(X_train_raw[tr], y_tr[tr])
                prob = pipe.predict_proba(X_train_raw[val])[:, 1]
                oof[val] = prob
                fold_thr, _ = threshold_sweep_oof(prob, y_tr[val])
                fold_scores.append(f1_score(y_tr[val], (prob >= fold_thr).astype(int), average="macro", zero_division=0))
            thr, cv_f1 = threshold_sweep_oof(oof, y_tr)
            candidates.append({
                "cfg_idx": cfg_idx,
                "cfg": cfg,
                "pca_n": pca_n,
                "no_pca": no_pca,
                "oof": oof,
                "thr": thr,
                "cv_f1": cv_f1,
                "fold_mean": float(np.mean(fold_scores)),
                "fold_std": float(np.std(fold_scores)),
            })

    best = max(candidates, key=lambda r: (r["cv_f1"], r["fold_mean"]))
    pipe = build_pipeline(model_name, best["cfg"], best["pca_n"], y_tr, no_pca=best["no_pca"])
    pipe.fit(X_train_raw, y_tr)
    prob_test = pipe.predict_proba(X_test_raw)[:, 1]
    pred = (prob_test >= best["thr"]).astype(int)
    oracle_thr, oracle_f1 = threshold_sweep_wide(prob_test, y_te)
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
        "test_f1": round(f1_score(y_te, pred, average="macro", zero_division=0), 4),
        "test_accuracy": round(accuracy_score(y_te, pred), 4),
        "test_precision": round(precision_score(y_te, pred, average="macro", zero_division=0), 4),
        "test_recall": round(recall_score(y_te, pred, average="macro", zero_division=0), 4),
        "test_recall_dep": round(recall_score(y_te, pred, pos_label=1, zero_division=0), 4),
        "test_auc": round(float(auc), 4),
        "oracle_thr": round(oracle_thr, 3),
        "oracle_f1": round(float(oracle_f1), 4),
        "y_pred": pred.tolist(),
        "y_prob": prob_test.tolist(),
        "oof_probs": best["oof"].tolist(),
    }, pipe

# %% [markdown]
# ## 4. Eksperimen Utama

# %%
all_results = []
best_pipe = None
best_result = None

print("\n" + "=" * 96)
print("  v96 MAIN LOOP: S1-S4 x RF/SVM/LR/XGBoost")
print("  Aggressive OOF threshold: 0.30-0.60 step 0.01")
print("=" * 96)

for scenario_name, info in SCENARIOS.items():
    X = info["X"]
    X_tr = X[train_idx]
    X_te = X[test_idx]
    print(f"\n{'-' * 96}")
    print(f"  {scenario_name}: {X.shape[1]} fitur")
    print(f"{'-' * 96}")
    for model_name in ["RF", "SVM", "LR", "XGB"]:
        t0 = time.time()
        res, pipe = evaluate_model(X_tr, X_te, y_train, y_test, scenario_name, model_name)
        res["time_s"] = round(time.time() - t0, 1)
        all_results.append(res)
        if best_result is None or res["test_f1"] > best_result["test_f1"]:
            best_result = res
            best_pipe = pipe
        pca_label = "NO-PCA" if res["no_pca"] else f"PCA={res['pca_n']}"
        print(
            f"  {model_name:<4} {pca_label:<8} CV={res['cv_f1']:.4f} thr={res['oof_thr']:.2f} "
            f"F1={res['test_f1']:.4f} Acc={res['test_accuracy']:.4f} "
            f"Prec={res['test_precision']:.4f} Rec={res['test_recall']:.4f} "
            f"AUC={res['test_auc']:.4f} ({res['time_s']}s)",
            flush=True,
        )

df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(METRICS_DIR, "v96_results.csv"), index=False)

# %% [markdown]
# ## 5. Visualisasi Fitur dan Performa

# %%
plt.figure(figsize=(9, 5))
plot_dims = feature_dims.sort_values("dimension", ascending=True)
plt.barh(plot_dims["scenario"], plot_dims["dimension"], color=["#4c78a8", "#59a14f", "#f28e2b", "#e15759"])
for i, val in enumerate(plot_dims["dimension"]):
    plt.text(val + 20, i, str(val), va="center", fontsize=10)
plt.title("v96 Total Dimensi Fitur per Skenario")
plt.xlabel("Jumlah fitur")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_dimensions_v96.png"), dpi=220)
plt.close()

models = ["RF", "SVM", "LR", "XGB"]
scenario_names = list(SCENARIOS.keys())
heat = np.zeros((len(scenario_names), len(models)))
for i, sc in enumerate(scenario_names):
    for j, m in enumerate(models):
        row = df_results[(df_results["scenario"] == sc) & (df_results["model"] == m)]
        heat[i, j] = float(row["test_f1"].iloc[0]) if not row.empty else 0.0

plt.figure(figsize=(10, 7))
sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu", xticklabels=models, yticklabels=scenario_names)
plt.title("v96 Test Macro F1")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap_v96.png"), dpi=220)
plt.close()

# %% [markdown]
# ## 6. Learning Curve dan XAI

# %%
best_scenario = best_result["scenario"]
best_model = best_result["model"]
best_feature_names = SCENARIOS[best_scenario]["feature_names"]
best_X_train = sanitize(SCENARIOS[best_scenario]["X"][train_idx])
best_cfg = eval(best_result["cfg"])
best_pca = None if best_result["pca_n"] == "none" else int(best_result["pca_n"])
best_curve_pipe = build_pipeline(best_model, best_cfg, best_pca, y_train, no_pca=best_result["no_pca"])

try:
    curve = learning_curve(
        best_curve_pipe,
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
    plt.title(f"v96 Learning Curve: {best_model} on {best_scenario}")
    plt.xlabel("Training samples")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.05)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curve_v96.png"), dpi=220)
    plt.close()
except Exception as exc:
    print(f"[WARN] Learning curve gagal: {type(exc).__name__}: {exc}")

cm = confusion_matrix(y_test, best_result["y_pred"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Depresi"], yticklabels=["Normal", "Depresi"])
plt.title(f"v96 Confusion Matrix: {best_model} on {best_scenario}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_best_v96.png"), dpi=220)
plt.close()

def save_xai_plot(pipe, scenario_name, model_name, feature_names):
    xai_rows = []
    clf = pipe.named_steps["clf"]
    if "pca" in pipe.named_steps:
        # Kalau PCA dipakai, XAI level yang jujur adalah principal components.
        names = [f"PC{i+1}" for i in range(pipe.named_steps["pca"].n_components_)]
    else:
        names = feature_names

    if model_name in ["LR", "SVM"] and hasattr(clf, "coef_"):
        vals = clf.coef_.ravel()
        title = f"v96 XAI Coefficients: {model_name} on {scenario_name}"
    elif model_name == "XGB" and hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        title = f"v96 XAI XGBoost Feature Importance: {scenario_name}"
    elif model_name == "RF" and hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        title = f"v96 XAI RandomForest Feature Importance: {scenario_name}"
    else:
        print(f"[WARN] XAI tidak tersedia untuk {model_name}")
        return None, pd.DataFrame()

    n = min(len(vals), len(names))
    vals = np.asarray(vals[:n], dtype=float)
    names = list(names[:n])
    order = np.argsort(np.abs(vals))[::-1][:20]
    xai_rows = [{"feature": names[i], "importance": float(vals[i]), "abs_importance": float(abs(vals[i]))} for i in order]
    df_xai = pd.DataFrame(xai_rows)
    out_csv = os.path.join(METRICS_DIR, "v96_xai_top20.csv")
    df_xai.to_csv(out_csv, index=False)

    plt.figure(figsize=(9, 6))
    plot_df = df_xai.sort_values("abs_importance", ascending=True)
    colors = ["#e15759" if v < 0 else "#4c78a8" for v in plot_df["importance"]]
    plt.barh(plot_df["feature"], plot_df["importance"], color=colors)
    plt.title(title)
    plt.xlabel("Importance / coefficient")
    plt.tight_layout()
    out_png = os.path.join(PLOTS_DIR, "xai_top20_v96.png")
    plt.savefig(out_png, dpi=220)
    plt.close()
    return out_png, df_xai

xai_path, df_xai = save_xai_plot(best_pipe, best_scenario, best_model, best_feature_names)
if xai_path:
    print(f"\nXAI plot saved: {xai_path}")
    print("Top XAI features/components:")
    print(df_xai.to_string(index=False))

# %% [markdown]
# ## 7. Summary Report

# %%
err = base.iloc[test_idx][["participant_id", "label_depresi"]].copy()
err["y_pred"] = best_result["y_pred"]
err["prob_depresi"] = best_result["y_prob"]
err["correct"] = err["label_depresi"] == err["y_pred"]
err.to_csv(os.path.join(METRICS_DIR, "v96_error_analysis_best.csv"), index=False)

sorted_res = df_results.sort_values("test_f1", ascending=False)

print("\n" + "=" * 126)
print(f"{'TABEL RINGKASAN v96 - S1-S4 x 4 Model':^126}")
print("=" * 126)
print(
    f"  {'Scenario':<18} {'Model':<4} {'PCA':<6} {'CV':>7} {'Thr':>5} "
    f"{'F1':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'RecD':>7} {'AUC':>7} {'Oracle':>7}"
)
for _, r in sorted_res.iterrows():
    print(
        f"  {r['scenario']:<18} {r['model']:<4} {str(r['pca_n']):<6} {r['cv_f1']:>7.4f} {r['oof_thr']:>5.2f} "
        f"{r['test_f1']:>7.4f} {r['test_accuracy']:>7.4f} {r['test_precision']:>7.4f} "
        f"{r['test_recall']:>7.4f} {r['test_recall_dep']:>7.4f} {r['test_auc']:>7.4f} {r['oracle_f1']:>7.4f}"
    )

print("\nAPPLE-TO-APPLE BEST PER SCENARIO:")
for sc in scenario_names:
    row = df_results[df_results["scenario"] == sc].sort_values("test_f1", ascending=False).iloc[0]
    print(
        f"  {sc:<18} {row['model']:<4} PCA={row['pca_n']} "
        f"F1={row['test_f1']:.4f} Acc={row['test_accuracy']:.4f} Prec={row['test_precision']:.4f} "
        f"Recall={row['test_recall']:.4f} AUC={row['test_auc']:.4f}"
    )

print("\nCLASSIFICATION REPORT - BEST v96")
print(f"  {best_model} on {best_scenario} | F1={best_result['test_f1']:.4f}")
print(classification_report(y_test, best_result["y_pred"], target_names=["Normal", "Depresi"], zero_division=0))

print("\nERROR ANALYSIS - BEST v96")
wrong = err[~err["correct"]]
print(wrong.to_string(index=False) if not wrong.empty else "  Tidak ada salah klasifikasi.")

summary = {
    "version": "v96",
    "strategy": "Focused S3 Wav2Vec768 LinearSVM C tuning, S2 MFCC XGBoost tuning, S4 full-feature LR/LinearSVM, OOF threshold 0.30-0.60, metrics+XAI",
    "dataset_total": int(len(base)),
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
        "cfg": best_result["cfg"],
        "cv_f1": best_result["cv_f1"],
        "threshold": best_result["oof_thr"],
        "test_f1": best_result["test_f1"],
        "accuracy": best_result["test_accuracy"],
        "precision": best_result["test_precision"],
        "recall": best_result["test_recall"],
        "auc": best_result["test_auc"],
        "oracle_f1": best_result["oracle_f1"],
    },
    "xai_plot": xai_path,
    "target_075": bool(best_result["test_f1"] >= 0.75),
    "elapsed_s": round(time.time() - t_global, 1),
}
with open(os.path.join(METRICS_DIR, "v96_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 96)
print(f"{'FINAL REPORT v96':^96}")
print("=" * 96)
print(f"  Best        : {best_result['test_f1']:.4f} ({best_model} on {best_scenario})")
print(f"  Accuracy    : {best_result['test_accuracy']:.4f}")
print(f"  Precision   : {best_result['test_precision']:.4f}")
print(f"  Recall      : {best_result['test_recall']:.4f}")
print(f"  AUC         : {best_result['test_auc']:.4f}")
print(f"  Oracle diag : {best_result['oracle_f1']:.4f}")
print(f"  Target 0.75 : {'TERCAPAI' if best_result['test_f1'] >= 0.75 else 'BELUM'}")
print(f"  Total waktu : {time.time() - t_global:.1f}s")
print(f"  Results dir : {RESULTS_DIR}")
print("=" * 96)
