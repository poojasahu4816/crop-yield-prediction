"""Time-based splitting. Training uses earlier years, testing uses later years."""
import numpy as np
import pandas as pd


def split_by_year(df: pd.DataFrame, test_years: int = 3, val_years: int = 3):
    years = np.sort(df["year"].unique())
    if len(years) < test_years + val_years + 3:
        raise ValueError(
            f"Only {len(years)} distinct years found; need at least "
            f"{test_years + val_years + 3} for a time-based split."
        )
    test_set = set(years[-test_years:])
    val_set = set(years[-(test_years + val_years):-test_years])
    test = df[df["year"].isin(test_set)]
    val = df[df["year"].isin(val_set)]
    train = df[~df["year"].isin(test_set | val_set)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def year_based_cv_splits(years: pd.Series, n_splits: int = 3):
    """Expanding-window splits: train on earlier years, validate on the next block."""
    years = pd.Series(years).reset_index(drop=True)
    uniq = np.sort(years.unique())
    blocks = np.array_split(uniq, n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_years = np.concatenate(blocks[:i])
        val_years = blocks[i]
        tr = np.where(years.isin(train_years))[0]
        va = np.where(years.isin(val_years))[0]
        splits.append((tr, va))
    return splits
