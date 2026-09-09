from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise RuntimeError("XGBoost is required for Phase 11 tuning. Install xgboost first.") from exc

from config import DATA_DIR, FAST_MODE, RANDOM_STATE, HORIZON_DAYS, MAX_MODEL_ROWS_FAST
from kaggle_runner import (
    find_data_dir, load_data, make_cutoffs, build_dataset,
    make_preprocessor
)

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# FAST_MODE intentionally keeps the search small enough for Kaggle iteration.
TUNING_ROWS = min(MAX_MODEL_ROWS_FAST, 150_000) if FAST_MODE else 500_000


def sample(X, y, seed):
    if len(X) <= TUNING_ROWS:
        return X.copy(), y.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=TUNING_ROWS, replace=False)
    return X.iloc[idx].copy(), y.iloc[idx].copy()


def make_xgb(params, y):
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    scale = neg / pos
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale,
        random_state=RANDOM_STATE,
    )


def main():
    data_dir = find_data_dir()
    tx, customers, articles = load_data(data_dir)
    train_cutoff, val_cutoff, _ = make_cutoffs(tx)

    # Two genuine rolling-origin folds. The second fold reproduces the
    # project's final train -> validation period, while the first is earlier.
    fold_specs = [
        (train_cutoff - pd.Timedelta(days=HORIZON_DAYS), train_cutoff),
        (train_cutoff, val_cutoff),
    ]

    print("\nPHASE 11 — TEMPORAL XGBOOST TUNING")
    print("Tuning rows per split:", TUNING_ROWS)

    folds = []
    for i, (tr_cut, va_cut) in enumerate(fold_specs, start=1):
        print(f"\nBuilding fold {i}: train cutoff={tr_cut.date()}, validation cutoff={va_cut.date()}")
        tr = build_dataset(tx, customers, articles, tr_cut)
        va = build_dataset(tx, customers, articles, va_cut)
        ytr = tr.pop("target")
        yva = va.pop("target")
        tr.pop("customer_id")
        va.pop("customer_id")
        Xtr, ytr = sample(tr, ytr, RANDOM_STATE + i)
        Xva, yva = sample(va, yva, RANDOM_STATE + 100 + i)
        folds.append((Xtr, ytr, Xva, yva))
        print("  rows:", len(Xtr), "train /", len(Xva), "validation")
        print("  positive rates:", round(float(ytr.mean()), 5), round(float(yva.mean()), 5))

    search_space = [
        dict(n_estimators=250, max_depth=3, learning_rate=0.05, min_child_weight=10, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=2),
        dict(n_estimators=350, max_depth=3, learning_rate=0.04, min_child_weight=10, subsample=0.90, colsample_bytree=0.90, reg_alpha=0.1, reg_lambda=3),
        dict(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=10, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=3),
        dict(n_estimators=400, max_depth=4, learning_rate=0.03, min_child_weight=15, subsample=0.90, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=4),
        dict(n_estimators=300, max_depth=5, learning_rate=0.05, min_child_weight=15, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=4),
        dict(n_estimators=450, max_depth=5, learning_rate=0.03, min_child_weight=20, subsample=0.90, colsample_bytree=0.80, reg_alpha=0.3, reg_lambda=5),
    ]

    results = []
    for j, params in enumerate(search_space, start=1):
        print(f"\n[{j}/{len(search_space)}] Testing:", params)
        fold_scores = []
        fold_rocs = []
        t0 = time.time()

        for fold_no, (Xtr, ytr, Xva, yva) in enumerate(folds, start=1):
            pre = make_preprocessor(Xtr)
            model = make_xgb(params, ytr)
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            pipe.fit(Xtr, ytr)
            p = pipe.predict_proba(Xva)[:, 1]
            pr = average_precision_score(yva, p)
            roc = roc_auc_score(yva, p)
            fold_scores.append(pr)
            fold_rocs.append(roc)
            print(f"  Fold {fold_no}: PR-AUC={pr:.6f}, ROC-AUC={roc:.6f}")

        mean_pr = float(np.mean(fold_scores))
        std_pr = float(np.std(fold_scores))
        mean_roc = float(np.mean(fold_rocs))
        elapsed = time.time() - t0
        print(f"  Mean PR-AUC={mean_pr:.6f} +/- {std_pr:.6f} | {elapsed:.1f}s")

        results.append({
            "config_id": j,
            **params,
            "fold1_pr_auc": fold_scores[0],
            "fold2_pr_auc": fold_scores[1],
            "mean_pr_auc": mean_pr,
            "std_pr_auc": std_pr,
            "mean_roc_auc": mean_roc,
            "fit_time_sec": elapsed,
        })

    table = pd.DataFrame(results).sort_values(
        ["mean_pr_auc", "mean_roc_auc"], ascending=False
    ).reset_index(drop=True)
    table.to_csv(RESULTS / "temporal_xgboost_tuning.csv", index=False)

    best = table.iloc[0].to_dict()
    best_params = {k: best[k] for k in search_space[0].keys()}
    # Restore integer fields that pandas may represent as numpy scalars.
    for k in ["n_estimators", "max_depth", "min_child_weight"]:
        best_params[k] = int(best_params[k])
    for k in ["learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]:
        best_params[k] = float(best_params[k])

    payload = {
        "selection_metric": "mean temporal validation PR-AUC",
        "best_config_id": int(best["config_id"]),
        "best_mean_pr_auc": float(best["mean_pr_auc"]),
        "best_std_pr_auc": float(best["std_pr_auc"]),
        "best_mean_roc_auc": float(best["mean_roc_auc"]),
        "fast_mode": bool(FAST_MODE),
        "tuning_rows_per_split": int(TUNING_ROWS),
        "best_params": best_params,
    }
    (RESULTS / "best_temporal_xgboost_params.json").write_text(json.dumps(payload, indent=2))

    print("\nBEST TEMPORAL CONFIG")
    print(json.dumps(payload, indent=2))
    print("\nSaved:")
    print("  results/temporal_xgboost_tuning.csv")
    print("  results/best_temporal_xgboost_params.json")


if __name__ == "__main__":
    main()
