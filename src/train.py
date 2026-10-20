"""Train, compare, tune and evaluate yield prediction models.

Usage:
    python -m src.train --data data/raw/crop_yield.csv
    python -m src.train --data data/raw/crop_yield.csv --tune
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV

from .cleaning import clean_data
from .config import RANDOM_STATE, RAW_INPUT_COLUMNS, TARGET, TEST_YEARS, VAL_YEARS
from .data_loader import load_data
from .eda import run_eda
from .evaluate import metrics_by_group, regression_metrics
from .models import MODELS, PARAM_DISTRIBUTIONS, build_pipeline
from .splitting import split_by_year, year_based_cv_splits


def _xy(df):
    return df[RAW_INPUT_COLUMNS], df[TARGET]


def _tune(name, pipe, train_df, n_iter):
    X, y = _xy(train_df)
    cv = year_based_cv_splits(train_df["year"], n_splits=3)
    space = PARAM_DISTRIBUTIONS[name]
    n_combos = int(np.prod([len(v) for v in space.values()]))
    search = RandomizedSearchCV(
        pipe, space, n_iter=min(n_iter, n_combos), cv=cv,
        scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, n_jobs=1,
    )
    search.fit(X, y)
    print(f"   tuned {name}: best CV RMSE={-search.best_score_:.4f}, params={search.best_params_}")
    return search.best_estimator_, search.best_params_


def run(data_path, output_dir="outputs", model_dir="models",
        tune=False, n_iter=10, test_years=TEST_YEARS, val_years=VAL_YEARS):
    out = Path(output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load and clean
    print("1. Loading and cleaning data...")
    df = load_data(data_path)
    df, log = clean_data(df)
    (out / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    print("   ", log)
    run_eda(df, fig_dir)

    # 2. Time-based split
    train, val, test = split_by_year(df, test_years, val_years)
    print(f"2. Split by year: train={len(train)} rows ({train.year.min()}-{train.year.max()}), "
          f"val={len(val)} ({val.year.min()}-{val.year.max()}), "
          f"test={len(test)} ({test.year.min()}-{test.year.max()})")

    # 3. Train every model and compare on VALIDATION years
    print("3. Training models...")
    X_tr, y_tr = _xy(train)
    X_va, y_va = _xy(val)
    fitted, rows = {}, []
    for name in MODELS:
        t0 = time.time()
        pipe = build_pipeline(name)
        if tune:
            pipe, _ = _tune(name, pipe, train, n_iter)
        else:
            pipe.fit(X_tr, y_tr)
        m_tr = regression_metrics(y_tr, pipe.predict(X_tr))
        m_va = regression_metrics(y_va, pipe.predict(X_va))
        rows.append({"model": name, "train_r2": m_tr["r2"],
                     "val_mae": m_va["mae"], "val_rmse": m_va["rmse"], "val_r2": m_va["r2"],
                     "seconds": round(time.time() - t0, 1)})
        fitted[name] = pipe
        print(f"   {name:18s} val RMSE={m_va['rmse']:.4f}  MAE={m_va['mae']:.4f}  R2={m_va['r2']:.4f}")
    comparison = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    comparison.to_csv(out / "model_comparison.csv", index=False)

    # 4. Pick best by validation RMSE, refit on train+val, evaluate on TEST once
    best_name = comparison.loc[0, "model"]
    print(f"4. Best model on validation: {best_name}. Refitting on train+val...")
    trval = pd.concat([train, val], ignore_index=True)
    best = clone(fitted[best_name])
    X_trval, y_trval = _xy(trval)
    best.fit(X_trval, y_trval)

    X_te, y_te = _xy(test)
    pred = best.predict(X_te)
    test_metrics = regression_metrics(y_te, pred)
    test_metrics.update({"best_model": best_name, "mean_test_yield": float(y_te.mean()),
                         "mae_pct_of_mean_yield": float(100 * test_metrics["mae"] / y_te.mean())})
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print("   TEST:", {k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()})

    # Baseline comparison on the test years (Ridge trained the same way)
    ridge = build_pipeline("ridge").fit(X_trval, y_trval)
    ridge_rmse = regression_metrics(y_te, ridge.predict(X_te))["rmse"]
    print(f"   Ridge baseline TEST RMSE={ridge_rmse:.4f}; "
          f"improvement={(1 - test_metrics['rmse'] / ridge_rmse) * 100:.1f}%")

    # 5. Error analysis by group
    preds = test.assign(predicted=pred)
    preds.to_csv(out / "test_predictions.csv", index=False)
    for col in ["crop", "state", "season"]:
        metrics_by_group(test, pred, col).to_csv(out / f"test_metrics_by_{col}.csv", index=False)

    # 6. Figures: actual vs predicted, permutation importance
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, pred, s=6, alpha=0.4)
    lim = [0, max(y_te.max(), pred.max())]
    ax.plot(lim, lim, "r--")
    ax.set_xlabel("Actual yield")
    ax.set_ylabel("Predicted yield")
    ax.set_title(f"{best_name}: actual vs predicted (test years)")
    fig.tight_layout()
    fig.savefig(fig_dir / "actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    imp = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=RANDOM_STATE,
                                 scoring="neg_root_mean_squared_error")
    imp_s = pd.Series(imp.importances_mean, index=RAW_INPUT_COLUMNS).sort_values()
    imp_s.to_csv(out / "permutation_importance.csv", header=["rmse_increase"])
    fig, ax = plt.subplots(figsize=(6, 4))
    imp_s.plot.barh(ax=ax)
    ax.set_xlabel("Increase in RMSE when the column is shuffled")
    ax.set_title("Permutation importance (test years)")
    fig.tight_layout()
    fig.savefig(fig_dir / "permutation_importance.png", dpi=120)
    plt.close(fig)

    joblib.dump(best, Path(model_dir) / "best_model.joblib")
    print(f"5. Saved model to {model_dir}/best_model.joblib and results to {out}/")
    return comparison, test_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to the crop yield CSV file")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--tune", action="store_true", help="Run randomized hyperparameter search")
    p.add_argument("--n-iter", type=int, default=10, help="Search iterations when --tune is used")
    p.add_argument("--test-years", type=int, default=TEST_YEARS)
    p.add_argument("--val-years", type=int, default=VAL_YEARS)
    a = p.parse_args()
    run(a.data, a.output_dir, a.model_dir, a.tune, a.n_iter, a.test_years, a.val_years)


if __name__ == "__main__":
    main()
