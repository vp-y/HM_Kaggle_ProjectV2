from pathlib import Path
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)

from config import FAST_MODE, RANDOM_STATE, MAX_MODEL_ROWS_FAST
from kaggle_runner import (
    find_data_dir, load_data, make_cutoffs, build_dataset
)

RESULTS = ROOT / "results" / "phase12"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "models" / "best_model_xgboost.joblib"
DEFAULT_THRESHOLD = 0.62


def load_model_and_metadata():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Frozen model not found: {MODEL_PATH}. Run kaggle_runner.py first."
        )

    model = joblib.load(MODEL_PATH)
    metadata = {}
    meta_path = ROOT / "results" / "best_model.json"

    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            pass

    threshold = float(
        metadata.get("validation_selected_threshold", DEFAULT_THRESHOLD)
    )
    return model, metadata, threshold


def recreate_temporal_sets(tx, customers, articles):
    train_cutoff, val_cutoff, test_cutoff = make_cutoffs(tx)

    print("\nTemporal cutoffs")
    print("Train:", train_cutoff.date())
    print("Validation:", val_cutoff.date())
    print("Test:", test_cutoff.date())

    val = build_dataset(tx, customers, articles, val_cutoff)
    test = build_dataset(tx, customers, articles, test_cutoff)

    y_val = val.pop("target")
    y_test = test.pop("target")
    id_val = val.pop("customer_id")
    id_test = test.pop("customer_id")

    return val, y_val, id_val, test, y_test, id_test


def sample_aligned(X, y, ids, name):
    if not FAST_MODE or len(X) <= MAX_MODEL_ROWS_FAST:
        return X.copy(), y.copy(), ids.copy()

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=MAX_MODEL_ROWS_FAST, replace=False)

    Xs = X.iloc[idx].copy()
    ys = y.iloc[idx].copy()
    ids_s = ids.iloc[idx].copy()

    assert len(Xs) == len(ys) == len(ids_s)
    assert Xs.index.equals(ys.index)
    assert Xs.index.equals(ids_s.index)

    print(f"{name}: sampled {len(Xs):,} / {len(X):,}")
    return Xs, ys, ids_s


def calculate_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "actual_positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def save_diagnostics(y, p, threshold):
    pred = (p >= threshold).astype(int)

    cm = confusion_matrix(y, pred)
    pd.DataFrame(
        cm, index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"]
    ).to_csv(RESULTS / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix — Final Test Set")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, roc_thresholds = roc_curve(y, p)
    pd.DataFrame({
        "fpr": fpr, "tpr": tpr, "threshold": roc_thresholds
    }).to_csv(RESULTS / "roc_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y,p):.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Final Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "roc_curve.png", dpi=160)
    plt.close(fig)

    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": np.r_[pr_thresholds, np.nan]
    }).to_csv(RESULTS / "precision_recall_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision,
            label=f"PR-AUC = {average_precision_score(y,p):.4f}")
    ax.axhline(np.mean(y), linestyle="--",
               label=f"Prevalence = {np.mean(y):.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Final Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p[np.asarray(y) == 0], bins=50, alpha=0.65, label="Actual 0")
    ax.hist(p[np.asarray(y) == 1], bins=50, alpha=0.65, label="Actual 1")
    ax.set_xlabel("Predicted Purchase Probability")
    ax.set_ylabel("Customers")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "probability_distribution.png", dpi=160)
    plt.close(fig)

    rows = []
    for t in np.arange(0.05, 0.951, 0.01):
        rows.append(calculate_metrics(y, p, float(t)))
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(RESULTS / "threshold_analysis.csv", index=False)

    best_idx = threshold_df["f1"].idxmax()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    ax.axvline(
        threshold_df.loc[best_idx, "threshold"],
        linestyle="--", label="Best F1 threshold"
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "threshold_analysis.png", dpi=160)
    plt.close(fig)


def save_error_analysis(X, y, ids, p, threshold):
    pred = (p >= threshold).astype(int)
    actual = np.asarray(y)

    error_type = np.where(
        (actual == 1) & (pred == 0), "false_negative",
        np.where(
            (actual == 0) & (pred == 1), "false_positive",
            np.where(actual == 1, "true_positive", "true_negative")
        )
    )

    df = X.copy()
    df.insert(0, "customer_id", np.asarray(ids))
    df["actual"] = actual
    df["purchase_probability"] = p
    df["prediction"] = pred
    df["error_type"] = error_type
    df.to_csv(RESULTS / "test_error_analysis.csv", index=False)

    counts = (
        pd.Series(error_type)
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )
    counts["share"] = counts["count"] / len(df)
    counts.to_csv(RESULTS / "error_type_counts.csv", index=False)

    mistakes = df[df["error_type"].isin(
        ["false_positive", "false_negative"]
    )].copy()

    mistakes["confidence"] = np.where(
        mistakes["error_type"].eq("false_positive"),
        mistakes["purchase_probability"],
        1.0 - mistakes["purchase_probability"]
    )
    mistakes.sort_values("confidence", ascending=False).head(500).to_csv(
        RESULTS / "top_500_confident_errors.csv", index=False
    )

    return df


def save_segment_analysis(df):
    specs = {}

    if "recency_days" in df:
        specs["recency_segment"] = pd.cut(
            df["recency_days"],
            [-np.inf, 7, 30, 90, 180, np.inf],
            labels=["0-7d", "8-30d", "31-90d", "91-180d", "180d+"]
        )

    if "total_items" in df:
        specs["purchase_volume_segment"] = pd.qcut(
            df["total_items"].rank(method="first"), 4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"]
        )

    if "total_spend" in df:
        specs["spend_segment"] = pd.qcut(
            df["total_spend"].rank(method="first"), 4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"]
        )

    if "customer_tenure_days" in df:
        specs["tenure_segment"] = pd.cut(
            df["customer_tenure_days"],
            [-np.inf, 30, 90, 180, 365, 730, np.inf],
            labels=["0-30d", "31-90d", "91-180d",
                    "181-365d", "366-730d", "730d+"]
        )

    rows = []
    for name, segment in specs.items():
        tmp = df.copy()
        tmp[name] = segment
        for value, group in tmp.groupby(name, observed=False):
            if len(group) == 0:
                continue
            rows.append({
                "segment_type": name,
                "segment": str(value),
                "rows": len(group),
                "actual_positive_rate": group["actual"].mean(),
                "predicted_positive_rate": group["prediction"].mean(),
                "precision": precision_score(
                    group["actual"], group["prediction"], zero_division=0
                ),
                "recall": recall_score(
                    group["actual"], group["prediction"], zero_division=0
                ),
                "f1": f1_score(
                    group["actual"], group["prediction"], zero_division=0
                ),
                "pr_auc": (
                    average_precision_score(
                        group["actual"], group["purchase_probability"]
                    )
                    if group["actual"].nunique() > 1 else np.nan
                )
            })

    pd.DataFrame(rows).to_csv(
        RESULTS / "segment_performance.csv", index=False
    )


def save_feature_importance(model):
    estimator = model.named_steps.get("model")
    preprocessor = model.named_steps.get("preprocess")

    if estimator is None or preprocessor is None:
        return

    if not hasattr(estimator, "feature_importances_"):
        return

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([
            f"feature_{i}" for i in range(len(estimator.feature_importances_))
        ])

    importance = np.asarray(estimator.feature_importances_)
    n = min(len(names), len(importance))

    df = pd.DataFrame({
        "feature": names[:n],
        "importance": importance[:n]
    }).sort_values("importance", ascending=False)

    df.to_csv(RESULTS / "feature_importance.csv", index=False)

    top = df.head(25).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 25 XGBoost Feature Importances")
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_importance_top25.png", dpi=160)
    plt.close(fig)


def save_permutation_importance(model, X, y):
    n = min(len(X), 5000)
    if len(X) > n:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), n, replace=False)
        Xp = X.iloc[idx].copy()
        yp = y.iloc[idx].copy()
    else:
        Xp, yp = X.copy(), y.copy()

    print(f"Permutation importance rows: {len(Xp):,}")
    start = time.time()

    result = permutation_importance(
        model,
        Xp,
        yp,
        scoring="average_precision",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print(f"Permutation importance time: {time.time()-start:.1f}s")

    df = pd.DataFrame({
        "feature": Xp.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    df.to_csv(RESULTS / "permutation_importance.csv", index=False)

    top = df.head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"]
    )
    ax.set_xlabel("Mean PR-AUC decrease")
    ax.set_ylabel("Original feature")
    ax.set_title("Top 20 Permutation Importances")
    fig.tight_layout()
    fig.savefig(RESULTS / "permutation_importance_top20.png", dpi=160)
    plt.close(fig)


def main():
    print("=" * 70)
    print("PHASE 12 — FINAL EVALUATION, ERROR ANALYSIS & EXPLAINABILITY")
    print("=" * 70)

    model, metadata, threshold = load_model_and_metadata()

    data_dir = find_data_dir()
    tx, customers, articles = load_data(data_dir)

    Xv, yv, idv, Xt, yt, idt = recreate_temporal_sets(
        tx, customers, articles
    )

    Xv, yv, idv = sample_aligned(Xv, yv, idv, "Validation")
    Xt, yt, idt = sample_aligned(Xt, yt, idt, "Test")

    print("\nValidation rows:", len(Xv))
    print("Test rows:", len(Xt))
    print("Test positive rate:", float(yt.mean()))
    print("Threshold:", threshold)

    print("\nGenerating frozen-model predictions...")
    p_val = model.predict_proba(Xv)[:, 1]
    p_test = model.predict_proba(Xt)[:, 1]

    assert len(p_val) == len(yv)
    assert len(p_test) == len(yt)
    assert np.isfinite(p_val).all()
    assert np.isfinite(p_test).all()
    assert ((p_test >= 0) & (p_test <= 1)).all()

    val_metrics = calculate_metrics(yv, p_val, threshold)
    test_metrics = calculate_metrics(yt, p_test, threshold)

    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        print(f"{k:25s}: {v:.6f}")

    print("\nFINAL TEST METRICS:")
    for k, v in test_metrics.items():
        print(f"{k:25s}: {v:.6f}")

    pd.DataFrame([val_metrics]).to_csv(
        RESULTS / "validation_metrics.csv", index=False
    )
    pd.DataFrame([test_metrics]).to_csv(
        RESULTS / "test_metrics.csv", index=False
    )

    pred = (p_test >= threshold).astype(int)
    report = classification_report(
        yt, pred,
        target_names=["No purchase", "Purchase"],
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        RESULTS / "classification_report.csv"
    )

    print("\nSaving diagnostics...")
    save_diagnostics(yt, p_test, threshold)

    print("Saving error analysis...")
    error_df = save_error_analysis(
        Xt, yt, idt, p_test, threshold
    )
    save_segment_analysis(error_df)

    print("Saving XGBoost feature importance...")
    save_feature_importance(model)

    print("Saving permutation importance...")
    save_permutation_importance(model, Xt, yt)

    pd.DataFrame({
        "customer_id": np.asarray(idt),
        "actual": np.asarray(yt),
        "purchase_probability": p_test,
        "prediction": pred,
        "error_type": error_df["error_type"].to_numpy()
    }).to_csv(
        RESULTS / "final_test_predictions.csv", index=False
    )

    cutoffs = make_cutoffs(tx)
    train_cutoff, val_cutoff, test_cutoff = cutoffs

    summary = f"""
H&M CUSTOMER PURCHASE PREDICTION — PHASE 12 FINAL ANALYSIS

Frozen model: {metadata.get("best_model", "xgboost")}
Selection metric: {metadata.get("selection_metric", "validation PR-AUC")}

Temporal cutoffs:
Train      : {train_cutoff.date()}
Validation : {val_cutoff.date()}
Test       : {test_cutoff.date()}

FAST_MODE: {FAST_MODE}
Validation-selected threshold: {threshold:.2f}

FINAL TEST METRICS
Accuracy  : {test_metrics["accuracy"]:.6f}
Precision : {test_metrics["precision"]:.6f}
Recall    : {test_metrics["recall"]:.6f}
F1        : {test_metrics["f1"]:.6f}
ROC-AUC   : {test_metrics["roc_auc"]:.6f}
PR-AUC    : {test_metrics["pr_auc"]:.6f}

Actual positive rate:
{test_metrics["actual_positive_rate"]:.6f}

Predicted positive rate:
{test_metrics["predicted_positive_rate"]:.6f}

The threshold was selected using validation data and was not optimized
using the final test labels.

All features were reconstructed through the same leakage-safe
build_dataset/build_features pipeline used by kaggle_runner.py.
"""

    (RESULTS / "PHASE12_SUMMARY.txt").write_text(summary.strip() + "\n")

    print("\n" + summary)
    print("Phase 12 complete.")
    print("Outputs:", RESULTS)


if __name__ == "__main__":
    main()
