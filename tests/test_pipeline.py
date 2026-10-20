"""Smoke tests (run with: pytest -q). They use synthetic data only."""
import numpy as np
import pandas as pd

from src.cleaning import clean_data
from src.config import RAW_INPUT_COLUMNS
from src.data_loader import standardize_columns
from src.features import FeatureBuilder
from src.splitting import split_by_year, year_based_cv_splits
from tests.synthetic import make_synthetic


def _clean():
    return clean_data(standardize_columns(make_synthetic()))


def test_production_is_not_a_model_input():
    assert "production" not in RAW_INPUT_COLUMNS
    assert "yield" not in RAW_INPUT_COLUMNS


def test_cleaning_removes_invalid_and_duplicates():
    df, log = _clean()
    assert log["duplicates_removed"] >= 1
    assert log["rows_dropped_invalid_values"] >= 1
    assert (df["area"] > 0).all() and (df["yield"] > 0).all()
    assert not df["crop"].str.endswith(" ").any()


def test_per_hectare_features():
    X = pd.DataFrame({"crop": ["Rice"], "state": ["A"], "season": ["Kharif"], "year": [2000],
                      "area": [50000.0], "annual_rainfall": [900.0],
                      "fertilizer": [6_000_000.0], "pesticide": [30_000.0]})
    out = FeatureBuilder().fit(X).transform(X)
    assert out.loc[0, "fert_per_ha"] == 120.0
    assert out.loc[0, "pest_per_ha"] == 0.6
    assert abs(out.loc[0, "log_area"] - np.log1p(50000)) < 1e-9
    assert "production" not in out.columns and "yield" not in out.columns


def test_rain_deviation_uses_training_state_average_only():
    train = pd.DataFrame({"crop": ["R"] * 2, "state": ["A"] * 2, "season": ["K"] * 2,
                          "year": [2000, 2001], "area": [1.0, 1.0],
                          "annual_rainfall": [1000.0, 1400.0], "fertilizer": [1, 1], "pesticide": [1, 1]})
    fb = FeatureBuilder().fit(train)               # state A mean = 1200
    new = train.iloc[[0]].assign(annual_rainfall=900.0)
    assert fb.transform(new).loc[0, "rain_deviation"] == -300.0


def test_time_split_has_no_overlap_and_is_ordered():
    df, _ = _clean()
    tr, va, te = split_by_year(df)
    assert tr.year.max() < va.year.min() <= va.year.max() < te.year.min()
    splits = year_based_cv_splits(tr["year"], 3)
    for a, b in splits:
        assert tr.loc[a, "year"].max() < tr.loc[b, "year"].min()


def test_end_to_end(tmp_path):
    from src.train import run
    csv = tmp_path / "synthetic.csv"
    make_synthetic().to_csv(csv, index=False)
    comp, test = run(str(csv), output_dir=str(tmp_path / "out"), model_dir=str(tmp_path / "m"))
    assert len(comp) == 3
    assert test["r2"] > 0.3
    assert (tmp_path / "out" / "test_metrics.json").exists()
