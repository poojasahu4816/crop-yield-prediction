"""Data cleaning. Imputation is NOT done here; it is done inside the model
pipeline so that it is fitted on training data only (avoids leakage)."""
import numpy as np
import pandas as pd

NUMERIC_COLS = ["year", "area", "production", "annual_rainfall",
                "fertilizer", "pesticide", "yield"]
TEXT_COLS = ["crop", "state", "season"]


def clean_data(df: pd.DataFrame):
    """Return (cleaned_df, log). The log records every cleaning decision."""
    df = df.copy()
    log = {"rows_raw": int(len(df))}

    # 1. Standardize text: trim spaces and collapse repeated spaces.
    for col in TEXT_COLS:
        s = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        df[col] = s.astype(object).where(s.notna(), np.nan)

    # 2. Convert numeric columns (bad values become NaN).
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Remove exact duplicates.
    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = int(before - len(df))

    # 4. Drop rows missing a key field.
    key = ["crop", "state", "season", "year", "area", "yield"]
    before = len(df)
    df = df.dropna(subset=key)
    log["rows_dropped_missing_key_fields"] = int(before - len(df))

    # 5. Drop physically invalid rows.
    invalid = (
        (df["area"] <= 0) | (df["yield"] <= 0)
        | (df["annual_rainfall"] < 0) | (df["fertilizer"] < 0) | (df["pesticide"] < 0)
    )
    log["rows_dropped_invalid_values"] = int(invalid.sum())
    df = df[~invalid].copy()
    df["year"] = df["year"].astype(int)

    # 6. Consistency check: yield should be close to production / area.
    has_prod = df["production"].notna() & (df["area"] > 0)
    expected = df.loc[has_prod, "production"] / df.loc[has_prod, "area"]
    rel_err = (expected - df.loc[has_prod, "yield"]).abs() / df.loc[has_prod, "yield"]
    log["yield_vs_production_over_area_mismatch_pct"] = (
        round(float((rel_err > 0.10).mean() * 100), 2) if len(rel_err) else None
    )

    # 7. Report (but do not remove) outliers: values outside 1.5*IQR per crop.
    g = df.groupby("crop")["yield"]
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    outliers = (df["yield"] < q1 - 1.5 * iqr) | (df["yield"] > q3 + 1.5 * iqr)
    log["yield_iqr_outliers_flagged_not_removed"] = int(outliers.sum())

    # 8. Missing values left for the pipeline imputer.
    log["missing_values_left_for_pipeline_imputer"] = {
        c: int(df[c].isna().sum()) for c in ["annual_rainfall", "fertilizer", "pesticide"]
    }
    log["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), log
