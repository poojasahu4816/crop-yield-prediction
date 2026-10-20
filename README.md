# Crop Yield Prediction for Indian Agriculture

A machine learning project that predicts **crop yield (tonnes per hectare)** from crop, state, season,
year, area, rainfall, fertilizer and pesticide data.
Built as part of a Machine Learning Engineer internship (Agriculture and Agribusiness).

**Author:** Pooja Sahu, Baderia Global Institute of Engineering and Management

## Why this project

Farmers, agribusinesses and policy makers need yield estimates to plan inputs, procurement, storage and
insurance. This project builds an explainable regression pipeline that handles typical problems of
agricultural data: skewed values, missing readings, regional and seasonal differences, and time trends.

## Project structure

```
crop-yield-prediction/
├── data/raw/            # put the CSV here (not committed)
├── notebooks/           # Google Colab notebook that runs the whole pipeline
├── src/
│   ├── config.py        # column names, feature lists, constants
│   ├── data_loader.py   # load CSV, standardize column names
│   ├── cleaning.py      # duplicates, invalid values, consistency checks, cleaning log
│   ├── features.py      # FeatureBuilder (per-hectare, log, rainfall deviation, ...)
│   ├── splitting.py     # time-based train/validation/test split
│   ├── models.py        # Ridge, Random Forest, Gradient Boosting pipelines
│   ├── evaluate.py      # MAE, RMSE, R2 (overall and by crop / state / season)
│   ├── eda.py           # exploratory plots
│   ├── train.py         # main script: train, compare, tune, evaluate, save
│   └── predict.py       # predict for new records with the saved model
├── tests/               # smoke tests (use synthetic data only)
├── outputs/             # metrics, tables and figures created by train.py
├── models/              # saved model (not committed)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/poojasahu4816/crop-yield-prediction.git
cd crop-yield-prediction
pip install -r requirements.txt
```

## Getting the data

The project uses the India Crop Yield dataset (columns: Crop, Crop_Year, Season, State, Area, Production,
Annual_Rainfall, Fertilizer, Pesticide, Yield). Download it from Hugging Face and save it as
`data/raw/crop_yield.csv`, or download with code:

```python
from src.data_loader import download_from_huggingface
download_from_huggingface("<dataset-id-from-the-hugging-face-page>", "data/raw/crop_yield.csv")
```

Column names are standardized automatically (`Crop_Year` becomes `year`). If your file uses other names,
add them to `COLUMN_ALIASES` in `src/config.py`.

## Run

```bash
python -m src.train --data data/raw/crop_yield.csv            # train and compare models
python -m src.train --data data/raw/crop_yield.csv --tune     # with hyperparameter search
python -m src.predict --input new_records.csv --output predictions.csv
pytest -q                                                     # run the smoke tests
```

## Method

1. **Cleaning** (`cleaning.py`): trim text, remove duplicates, drop rows with missing key fields or
   impossible values (area or yield <= 0, negative rainfall), check that Yield is close to Production / Area,
   flag outliers per crop (flagged, not removed). Every step is written to `outputs/cleaning_log.json`.
2. **No data leakage:** `Production` is **not** a model input because Yield = Production / Area.
   Missing values are imputed inside the model pipeline, so imputation values come from training data only.
3. **Feature engineering** (`features.py`): fertilizer and pesticide per hectare, log of area, rainfall
   deviation from the state's training average (absolute and percent), rainfall category (low, normal,
   high from training quartiles), year trend. One-hot encoding groups rare categories automatically.
4. **Time-based split:** the last 3 years are the test set, the 3 years before are validation, the rest is
   training. This tests the model on future years instead of randomly mixed years.
5. **Models:** Ridge Regression (baseline), Random Forest, Gradient Boosting. The best model is chosen on
   validation RMSE, refitted on train + validation and evaluated **once** on the test years.
6. **Tuning (optional):** randomized search with expanding-window cross-validation over years.
7. **Evaluation:** MAE, RMSE and R2 overall, and error tables by crop, state and season to find where the
   model is weak. Permutation importance shows which inputs matter most.

## Outputs (created in `outputs/`)

`cleaning_log.json`, `model_comparison.csv`, `test_metrics.json`, `test_metrics_by_crop.csv`,
`test_metrics_by_state.csv`, `test_metrics_by_season.csv`, `permutation_importance.csv`, and figures in
`outputs/figures/` (EDA plots, actual vs predicted, permutation importance).

## Results

> Fill this table after running the pipeline on the real dataset.

| Model | Validation RMSE | Validation MAE | Validation R2 |
|---|---|---|---|
| Ridge Regression | | | |
| Random Forest | | | |
| Gradient Boosting | | | |

Final test result (best model): RMSE = , MAE = , R2 =

## Limitations

- State-level data hides differences between districts and farms.
- Soil, temperature, irrigation, seed variety and prices are not in the dataset.
- Tree models cannot predict beyond the range of yields seen in training, so the model should be retrained
  every year.
- Predictions are decision support, not a replacement for human judgement, especially for loans or
  insurance decisions.

## Future work

Add IMD weather data (growing degree days, dry spells), soil health card features, SHAP explanations, and a
simple web app for predictions.

## Tests

`tests/` contains smoke tests that use a small **synthetic** dataset to check that the code runs and that
important rules hold (no Production in model inputs, per-hectare calculation, no year overlap between
train and test). They do not say anything about real-world accuracy.
