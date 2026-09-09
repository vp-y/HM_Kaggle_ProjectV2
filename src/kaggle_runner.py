from pathlib import Path
import sys
import json, time, warnings

warnings.filterwarnings("ignore")

# Add project root to Python path so config.py can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from config import (
    DATA_DIR, FAST_MODE, RANDOM_STATE, HORIZON_DAYS,
    RUN_MODELS, MAX_MODEL_ROWS_FAST, RUN_TUNING
)


RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

def find_data_dir():
    """Locate the H&M competition files, including kagglehub cache locations."""
    import os

    env_dir = os.environ.get("HM_DATA_DIR")
    candidates = []

    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend([
        Path(DATA_DIR),
        Path("/kaggle/input/h-and-m-personalized-fashion-recommendations"),
        Path("/kaggle/input/hm-personalized-fashion-recommendations"),
        Path("/kaggle/working"),
        Path("/root/.cache"),
    ])

    for p in candidates:
        if p.is_file() and p.name == "transactions_train.csv":
            return p.parent
        if p.is_dir() and (p / "transactions_train.csv").exists():
            return p

    seen = set()
    for base in candidates:
        if not base.exists() or not base.is_dir():
            continue
        key = str(base.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            for csv_file in base.rglob("transactions_train.csv"):
                parent = csv_file.parent
                if ((parent / "customers.csv").exists()
                        and (parent / "articles.csv").exists()):
                    print(f"Automatically found H&M dataset at: {parent}")
                    return parent
        except (PermissionError, OSError):
            continue

    raise FileNotFoundError(
        "Could not find transactions_train.csv. "
        "Run the kagglehub download cell first."
    )

def load_data(data_dir):
    print("Loading H&M data from:", data_dir)
    customers = pd.read_csv(data_dir / "customers.csv")
    articles = pd.read_csv(data_dir / "articles.csv")
    transactions = pd.read_csv(
        data_dir / "transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        dtype={
            "customer_id": "string",
            "article_id": "int32",
            "price": "float32",
            "sales_channel_id": "int8",
        }
    )
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["customer_id"] = transactions["customer_id"].astype("string")
    print("transactions:", transactions.shape)
    print("customers:", customers.shape)
    print("articles:", articles.shape)
    return transactions, customers, articles

def make_cutoffs(transactions):
    max_date = transactions["t_dat"].max()
    test_cutoff = max_date - pd.Timedelta(days=HORIZON_DAYS)
    val_cutoff = test_cutoff - pd.Timedelta(days=HORIZON_DAYS)
    train_cutoff = val_cutoff - pd.Timedelta(days=HORIZON_DAYS)
    return train_cutoff, val_cutoff, test_cutoff

def build_features(transactions, customers, articles, cutoff):
    """Build customer features using only transactions <= cutoff."""
    tx = transactions[transactions["t_dat"] <= cutoff].copy()

    # Customer universe is all known customers; customers without history get
    # zero/unknown features and can still receive predictions.
    base = customers[["customer_id", "FN", "Active", "club_member_status",
                      "fashion_news_frequency", "age"]].copy()
    base["customer_id"] = base["customer_id"].astype("string")

    g = tx.groupby("customer_id", sort=False)
    f = pd.DataFrame(index=g.size().index)
    f["total_items"] = g.size().astype("float32")
    f["purchase_days"] = g["t_dat"].nunique().astype("float32")
    f["unique_articles"] = g["article_id"].nunique().astype("float32")
    f["total_spend"] = g["price"].sum().astype("float32")
    f["avg_price"] = g["price"].mean().astype("float32")
    f["median_price"] = g["price"].median().astype("float32")
    f["min_price"] = g["price"].min().astype("float32")
    f["max_price"] = g["price"].max().astype("float32")
    f["price_std"] = g["price"].std().fillna(0).astype("float32")

    last_purchase = g["t_dat"].max()
    first_purchase = g["t_dat"].min()
    f["recency_days"] = (cutoff - last_purchase).dt.days.astype("float32")
    f["customer_tenure_days"] = (cutoff - first_purchase).dt.days.astype("float32")
    f["purchase_rate"] = (
        f["total_items"] / f["customer_tenure_days"].clip(lower=1)
    ).astype("float32")

    # Recent windows.
    for days in (30, 90):
        start = cutoff - pd.Timedelta(days=days)
        r = tx[tx["t_dat"] > start].groupby("customer_id", sort=False)
        f[f"items_{days}d"] = r.size().astype("float32")
        f[f"spend_{days}d"] = r["price"].sum().astype("float32")
        f[f"purchase_days_{days}d"] = r["t_dat"].nunique().astype("float32")
        f[f"unique_articles_{days}d"] = r["article_id"].nunique().astype("float32")
        f[f"avg_price_{days}d"] = r["price"].mean().astype("float32")

    f["recent_spend_ratio"] = (
        f["spend_30d"] / f["total_spend"].replace(0, np.nan)
    ).astype("float32")
    f["recent_items_ratio"] = (
        f["items_30d"] / f["total_items"].replace(0, np.nan)
    ).astype("float32")

    # ============================================================
    # Advanced behavioral / RFM features
    # ============================================================

    for days in (7, 180):
        start = cutoff - pd.Timedelta(days=days)
        r = tx[tx["t_dat"] > start].groupby("customer_id", sort=False)
        f[f"items_{days}d"] = r.size().astype("float32")
        f[f"spend_{days}d"] = r["price"].sum().astype("float32")
        f[f"purchase_days_{days}d"] = r["t_dat"].nunique().astype("float32")
        f[f"unique_articles_{days}d"] = r["article_id"].nunique().astype("float32")

    f["items_per_purchase_day"] = (f["total_items"] / f["purchase_days"].clip(lower=1)).astype("float32")
    f["spend_per_purchase_day"] = (f["total_spend"] / f["purchase_days"].clip(lower=1)).astype("float32")
    f["items_per_article"] = (f["total_items"] / f["unique_articles"].clip(lower=1)).astype("float32")

    f["items_7d_ratio"] = (f["items_7d"] / f["total_items"].replace(0, np.nan)).astype("float32")
    f["items_90d_ratio"] = (f["items_90d"] / f["total_items"].replace(0, np.nan)).astype("float32")
    f["spend_30d_ratio"] = (f["spend_30d"] / f["total_spend"].replace(0, np.nan)).astype("float32")
    f["spend_90d_ratio"] = (f["spend_90d"] / f["total_spend"].replace(0, np.nan)).astype("float32")

    older_60d_items = (f["items_90d"] - f["items_30d"]).clip(lower=0)
    older_60d_spend = (f["spend_90d"] - f["spend_30d"]).clip(lower=0)
    f["recent_vs_older_items"] = (f["items_30d"] / older_60d_items.replace(0, np.nan)).astype("float32")
    f["recent_vs_older_spend"] = (f["spend_30d"] / older_60d_spend.replace(0, np.nan)).astype("float32")

    f["purchase_day_ratio_30d"] = (f["purchase_days_30d"] / 30.0).astype("float32")
    f["purchase_day_ratio_90d"] = (f["purchase_days_90d"] / 90.0).astype("float32")
    f["recent_avg_price_ratio"] = (f["avg_price_30d"] / f["avg_price"].replace(0, np.nan)).astype("float32")
    f["price_range"] = (f["max_price"] - f["min_price"]).astype("float32")

    f["recency_log"] = np.log1p(f["recency_days"].clip(lower=0)).astype("float32")
    f["tenure_log"] = np.log1p(f["customer_tenure_days"].clip(lower=0)).astype("float32")
    f["recency_to_tenure"] = (f["recency_days"] / f["customer_tenure_days"].clip(lower=1)).astype("float32")

    customer_dates = (
        tx[["customer_id", "t_dat"]]
        .drop_duplicates()
        .sort_values(["customer_id", "t_dat"])
    )
    customer_dates["days_since_previous"] = (
        customer_dates.groupby("customer_id")["t_dat"].diff().dt.days
    )
    interval_stats = customer_dates.groupby("customer_id")["days_since_previous"].agg(["mean", "median", "std", "min", "max"])
    interval_stats.columns = [
        "avg_purchase_interval", "median_purchase_interval",
        "std_purchase_interval", "min_purchase_interval", "max_purchase_interval"
    ]
    f = f.join(interval_stats)
    f["recency_vs_avg_interval"] = (f["recency_days"] / f["avg_purchase_interval"].replace(0, np.nan)).astype("float32")

    f["cutoff_month"] = cutoff.month
    f["cutoff_quarter"] = cutoff.quarter
    f["cutoff_dayofweek"] = cutoff.dayofweek

    # Channel behavior.
    channel = pd.crosstab(tx["customer_id"], tx["sales_channel_id"])
    if 1 in channel.columns:
        f["channel_1_items"] = channel[1].astype("float32")
    else:
        f["channel_1_items"] = 0.0
    if 2 in channel.columns:
        f["channel_2_items"] = channel[2].astype("float32")
    else:
        f["channel_2_items"] = 0.0
    denom = (f["channel_1_items"] + f["channel_2_items"]).replace(0, np.nan)
    f["channel_2_ratio"] = (f["channel_2_items"] / denom).astype("float32")

    # Product diversity. Merge only article attributes; images are never loaded.
    article_small = articles[[
        "article_id", "product_type_name", "product_group_name",
        "department_no", "section_no", "index_name", "garment_group_name"
    ]].copy()
    txa = tx[["customer_id", "article_id"]].merge(article_small, on="article_id", how="left")
    ag = txa.groupby("customer_id", sort=False)
    f["unique_product_types"] = ag["product_type_name"].nunique().astype("float32")
    f["unique_product_groups"] = ag["product_group_name"].nunique().astype("float32")
    f["unique_departments"] = ag["department_no"].nunique().astype("float32")
    f["unique_sections"] = ag["section_no"].nunique().astype("float32")
    f["unique_indices"] = ag["index_name"].nunique().astype("float32")
    f["unique_garment_groups"] = ag["garment_group_name"].nunique().astype("float32")

    # Preferred product group as a categorical feature.
    pg = (
        txa.dropna(subset=["product_group_name"])
           .groupby(["customer_id", "product_group_name"])
           .size()
           .reset_index(name="n")
           .sort_values(["customer_id", "n"], ascending=[True, False])
           .drop_duplicates("customer_id")
           .set_index("customer_id")["product_group_name"]
    )
    f["preferred_product_group"] = pg

    f = f.reset_index().rename(columns={"index": "customer_id"})
    out = base.merge(f, on="customer_id", how="left")

    numeric = out.select_dtypes(include=np.number).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    # Log transforms are calculated after aggregation and do not remove originals.
    for col in ["total_items", "total_spend", "unique_articles", "avg_price",
                "purchase_days", "items_30d", "spend_30d", "items_90d", "spend_90d"]:
        if col in out.columns:
            out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))

    return out

def build_target(transactions, cutoff):
    future = transactions[
        (transactions["t_dat"] > cutoff) &
        (transactions["t_dat"] <= cutoff + pd.Timedelta(days=HORIZON_DAYS))
    ]
    purchased = future.groupby("customer_id", sort=False).size()
    return purchased

def build_dataset(transactions, customers, articles, cutoff):
    X = build_features(transactions, customers, articles, cutoff)
    purchasers = build_target(transactions, cutoff)
    X["target"] = X["customer_id"].isin(purchasers.index).astype("int8")
    return X

def temporal_data(transactions, customers, articles):
    train_cutoff, val_cutoff, test_cutoff = make_cutoffs(transactions)
    print("Cutoffs:", train_cutoff.date(), val_cutoff.date(), test_cutoff.date())

    train = build_dataset(transactions, customers, articles, train_cutoff)
    val = build_dataset(transactions, customers, articles, val_cutoff)
    test = build_dataset(transactions, customers, articles, test_cutoff)

    # Keep customer IDs out of the ML matrix.
    y_train = train.pop("target")
    y_val = val.pop("target")
    y_test = test.pop("target")
    train_ids = train.pop("customer_id")
    val_ids = val.pop("customer_id")
    test_ids = test.pop("customer_id")

    print("Target rates:",
          "train", y_train.mean(),
          "val", y_val.mean(),
          "test", y_test.mean())
    return (train, y_train, train_ids), (val, y_val, val_ids), (test, y_test, test_ids)

def make_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

def get_models(y):
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    scale = neg / pos

    models = {
        "logistic": LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear",
            class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=50,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150 if FAST_MODE else 300,
            max_depth=14, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=150 if FAST_MODE else 300,
            max_depth=14, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120 if FAST_MODE else 200,
            learning_rate=0.05, max_depth=3,
            min_samples_leaf=50, subsample=0.8,
            random_state=RANDOM_STATE
        ),
    }
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300 if FAST_MODE else 600,
            max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=10, reg_lambda=2, reg_alpha=0.1,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1,
            scale_pos_weight=scale,
            random_state=RANDOM_STATE
        )
    return {k: v for k, v in models.items() if k in RUN_MODELS}

def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
    }

def best_f1_threshold(y, p):
    thresholds = np.arange(0.05, 0.951, 0.01)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])

def sample_for_speed(X, y, ids):
    if not FAST_MODE or len(X) <= MAX_MODEL_ROWS_FAST:
        return X, y, ids
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=MAX_MODEL_ROWS_FAST, replace=False)
    return X.iloc[idx].copy(), y.iloc[idx].copy(), ids.iloc[idx].copy()

def run():
    data_dir = find_data_dir()
    tx, customers, articles = load_data(data_dir)
    (Xtr, ytr, idtr), (Xv, yv, idv), (Xte, yte, idte) = temporal_data(tx, customers, articles)

    # Speed mode samples customers only after leakage-safe temporal feature/target creation.
    Xtr_fit, ytr_fit, _ = sample_for_speed(Xtr, ytr, idtr)
    Xv_fit, yv_fit, _ = sample_for_speed(Xv, yv, idv)
    print("Model rows:", len(Xtr_fit), "train /", len(Xv_fit), "validation")

    preprocessor = make_preprocessor(Xtr_fit)
    models = get_models(ytr_fit)

    rows = []
    fitted = {}
    validation_probs = {}

    for name, model in models.items():
        print("\nTraining:", name)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        t0 = time.time()
        pipe.fit(Xtr_fit, ytr_fit)
        fit_time = time.time() - t0

        p_train = pipe.predict_proba(Xtr_fit)[:, 1]
        p_val = pipe.predict_proba(Xv_fit)[:, 1]

        train_m = metrics(ytr_fit, p_train, 0.5)
        val_m = metrics(yv_fit, p_val, 0.5)
        threshold, _ = best_f1_threshold(yv_fit, p_val)
        val_opt = metrics(yv_fit, p_val, threshold)

        row = {
            "model": name,
            "threshold": threshold,
            "fit_time_sec": fit_time,
            "train_accuracy": train_m["accuracy"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "train_f1": train_m["f1"],
            "train_roc_auc": train_m["roc_auc"],
            "train_pr_auc": train_m["pr_auc"],
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_roc_auc": val_m["roc_auc"],
            "val_pr_auc": val_m["pr_auc"],
            "val_opt_f1": val_opt["f1"],
            "generalization_gap_pr_auc": train_m["pr_auc"] - val_m["pr_auc"],
        }
        rows.append(row)
        fitted[name] = pipe
        validation_probs[name] = p_val

        joblib.dump(pipe, MODELS / f"{name}_validation_model.joblib")

    comparison = pd.DataFrame(rows).sort_values(
        ["val_pr_auc", "val_opt_f1"], ascending=False
    ).reset_index(drop=True)

    # Baseline is Logistic Regression. Improvement is always relative to baseline
    # on the same validation metric.
    baseline = comparison.loc[comparison["model"] == "logistic"].iloc[0]
    best = comparison.iloc[0]

    improvement_rows = []
    for metric in ["val_pr_auc", "val_roc_auc", "val_f1", "val_opt_f1", "val_precision", "val_recall"]:
        b = float(baseline[metric])
        for_metric = float(best[metric])
        improvement_rows.append({
            "metric": metric,
            "baseline_model": "logistic",
            "best_model": best["model"],
            "baseline_score": b,
            "best_score": for_metric,
            "absolute_improvement": for_metric - b,
            "relative_improvement_percent": ((for_metric - b) / b * 100) if b != 0 else np.nan
        })

    improvement = pd.DataFrame(improvement_rows)

    # Final test: only the selected best model and the threshold chosen on validation.
    best_name = best["model"]
    best_pipe = fitted[best_name]
    # IMPORTANT: sample X, y, and IDs with the exact same positional indices.
    # Never reconstruct X afterward with index filtering, because that can
    # reorder rows and silently misalign features with labels.
    if FAST_MODE and len(Xte) > MAX_MODEL_ROWS_FAST:
        rng = np.random.RandomState(RANDOM_STATE)
        test_idx = rng.choice(
            len(Xte),
            size=MAX_MODEL_ROWS_FAST,
            replace=False
        )
        test_sample = Xte.iloc[test_idx].copy()
        ytest_sample = yte.iloc[test_idx].copy()
        idtest_sample = idte.iloc[test_idx].copy()
    else:
        test_sample = Xte.copy()
        ytest_sample = yte.copy()
        idtest_sample = idte.copy()

    # Hard safety checks: prediction rows must correspond exactly to labels.
    assert len(test_sample) == len(ytest_sample) == len(idtest_sample)
    assert test_sample.index.equals(ytest_sample.index)
    assert test_sample.index.equals(idtest_sample.index)

    threshold = float(best["threshold"])
    print("Test rows:", len(test_sample))
    print("Test positive rate:", float(ytest_sample.mean()))
    p_test = best_pipe.predict_proba(test_sample)[:, 1]

    # Sanity checks for a valid probability evaluation.
    assert len(p_test) == len(ytest_sample)
    assert np.isfinite(p_test).all()
    assert ((p_test >= 0) & (p_test <= 1)).all()

    test_m = metrics(ytest_sample, p_test, threshold)

    test_row = pd.DataFrame([{
        "best_model": best_name,
        "threshold_selected_on_validation": threshold,
        **{f"test_{k}": v for k, v in test_m.items()}
    }])

    # Save everything needed for a clean CV/report table.
    comparison.to_csv(RESULTS / "final_model_comparison.csv", index=False)
    improvement.to_csv(RESULTS / "baseline_improvement.csv", index=False)
    test_row.to_csv(RESULTS / "best_model_test_metrics.csv", index=False)

    best_json = {
        "best_model": best_name,
        "selection_metric": "validation PR-AUC",
        "best_validation_pr_auc": float(best["val_pr_auc"]),
        "baseline_validation_pr_auc": float(baseline["val_pr_auc"]),
        "absolute_pr_auc_improvement": float(best["val_pr_auc"] - baseline["val_pr_auc"]),
        "relative_pr_auc_improvement_percent": float(
            (best["val_pr_auc"] - baseline["val_pr_auc"]) / baseline["val_pr_auc"] * 100
        ) if baseline["val_pr_auc"] else None,
        "validation_selected_threshold": threshold,
        "fast_mode": FAST_MODE,
        "xgboost_available": XGB_AVAILABLE,
    }
    (RESULTS / "best_model.json").write_text(json.dumps(best_json, indent=2))

    # Save the final selected validation-trained model and test predictions.
    joblib.dump(best_pipe, MODELS / f"best_model_{best_name}.joblib")
    pd.DataFrame({
        "customer_id": idtest_sample.values,
        "purchase_probability": p_test,
        "prediction": (p_test >= threshold).astype(int)
    }).to_csv(RESULTS / "best_model_test_predictions.csv", index=False)

    # Human-readable summary.
    summary = f"""
H&M CUSTOMER PURCHASE PREDICTION — FINAL SUMMARY

Baseline: Logistic Regression
Best model: {best_name}

Baseline validation PR-AUC: {baseline["val_pr_auc"]:.6f}
Best validation PR-AUC:     {best["val_pr_auc"]:.6f}
Absolute PR-AUC improvement: {best["val_pr_auc"] - baseline["val_pr_auc"]:.6f}
Relative PR-AUC improvement: {((best["val_pr_auc"] - baseline["val_pr_auc"]) / baseline["val_pr_auc"] * 100) if baseline["val_pr_auc"] else float("nan"):.2f}%

Validation-selected threshold: {threshold:.2f}

Best model test metrics:
Accuracy:  {test_m["accuracy"]:.6f}
Precision: {test_m["precision"]:.6f}
Recall:    {test_m["recall"]:.6f}
F1:        {test_m["f1"]:.6f}
ROC-AUC:   {test_m["roc_auc"]:.6f}
PR-AUC:    {test_m["pr_auc"]:.6f}

Model-wise validation table: results/final_model_comparison.csv
Baseline improvement table:  results/baseline_improvement.csv
Best model test metrics:      results/best_model_test_metrics.csv
"""
    (RESULTS / "FINAL_SUMMARY.txt").write_text(summary.strip() + "\n")
    print(summary)

if __name__ == "__main__":
    run()
