"""Predict yield for new records using the saved model.

Usage:
    python -m src.predict --input new_records.csv --output predictions.csv
The input CSV needs: crop, year, season, state, area, annual_rainfall, fertilizer, pesticide.
"""
import argparse

import joblib
import pandas as pd

from .config import RAW_INPUT_COLUMNS
from .data_loader import standardize_columns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--model", default="models/best_model.joblib")
    a = p.parse_args()
    df = standardize_columns(pd.read_csv(a.input))
    model = joblib.load(a.model)
    df["predicted_yield"] = model.predict(df[RAW_INPUT_COLUMNS])
    df.to_csv(a.output, index=False)
    print(f"Saved {len(df)} predictions to {a.output}")


if __name__ == "__main__":
    main()
