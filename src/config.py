"""Central configuration: column names, feature lists and constants."""

RANDOM_STATE = 42

TARGET = "yield"

# Columns that must exist after column names are standardized.
REQUIRED_COLUMNS = [
    "crop", "year", "season", "state", "area",
    "production", "annual_rainfall", "fertilizer", "pesticide", "yield",
]

# Alternative column names found in public versions of this dataset.
COLUMN_ALIASES = {
    "crop_year": "year",
    "rainfall": "annual_rainfall",
    "annual_rain": "annual_rainfall",
}

# Raw columns given to the model pipeline.
# NOTE: "production" is deliberately NOT included. Yield = Production / Area,
# so using Production as an input would leak the answer (data leakage).
RAW_INPUT_COLUMNS = [
    "crop", "year", "season", "state", "area",
    "annual_rainfall", "fertilizer", "pesticide",
]

# Features produced by FeatureBuilder (src/features.py).
CATEGORICAL_FEATURES = ["crop", "state", "season", "rain_category"]
NUMERIC_FEATURES = [
    "log_area", "fert_per_ha", "pest_per_ha", "annual_rainfall",
    "rain_deviation", "rain_deviation_pct", "year_trend",
]

# Default time-based split: last 3 years = test, 3 years before = validation.
TEST_YEARS = 3
VAL_YEARS = 3
