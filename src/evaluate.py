"""Evaluation metrics: MAE, RMSE, R2 (overall and by group)."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metrics_by_group(df: pd.DataFrame, y_pred, group_col: str, min_rows: int = 5) -> pd.DataFrame:
    """Error per crop / state / season, to detect where the model is weak."""
    tmp = pd.DataFrame({"group": df[group_col].to_numpy(),
                        "y_true": df["yield"].to_numpy(),
                        "y_pred": np.asarray(y_pred)})
    rows = []
    for name, g in tmp.groupby("group"):
        if len(g) < min_rows:
            continue
        m = regression_metrics(g["y_true"], g["y_pred"])
        rows.append({group_col: name, "n_rows": len(g),
                     "mean_yield": g["y_true"].mean(), **m})
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
