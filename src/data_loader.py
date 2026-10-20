"""Loading the dataset and standardizing column names."""
from pathlib import Path

import pandas as pd

from .config import COLUMN_ALIASES, REQUIRED_COLUMNS


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, replace spaces with underscores, apply aliases."""
    df = df.copy()
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.columns = [COLUMN_ALIASES.get(c, c) for c in cols]
    return df


def load_data(path: str) -> pd.DataFrame:
    """Read the CSV file and check that all required columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Download the dataset first "
            "(see README, section 'Getting the data')."
        )
    df = standardize_columns(pd.read_csv(path))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns after standardizing names: {missing}. "
            f"Columns found: {list(df.columns)}. "
            "Add the correct names to COLUMN_ALIASES in src/config.py."
        )
    return df


def download_from_huggingface(dataset_id: str, out_csv: str, split: str = "train") -> str:
    """Download a dataset from Hugging Face and save it as CSV.

    Check the exact dataset id on the Hugging Face dataset page before using.
    """
    from datasets import load_dataset  # imported here so it stays optional

    ds = load_dataset(dataset_id, split=split)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(str(out))
    return str(out)
