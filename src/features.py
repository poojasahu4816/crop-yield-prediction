"""Feature engineering (Week 3 plan).

FeatureBuilder learns reference values (state average rainfall, rainfall
quartiles, first year) from TRAINING data only in fit(), and applies them in
transform(). It never uses Production or Yield.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.state_rain_mean_ = X.groupby("state")["annual_rainfall"].mean()
        self.global_rain_mean_ = float(X["annual_rainfall"].mean())
        self.rain_q25_, self.rain_q75_ = (
            float(v) for v in X["annual_rainfall"].quantile([0.25, 0.75])
        )
        self.min_year_ = int(X["year"].min())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)

        # Categorical columns passed through.
        for col in ["crop", "state", "season"]:
            out[col] = X[col]

        # Ratio features: dose per hectare (area <= 0 -> NaN, avoids divide by zero).
        area = X["area"].where(X["area"] > 0)
        out["fert_per_ha"] = X["fertilizer"] / area
        out["pest_per_ha"] = X["pesticide"] / area

        # Log transform of skewed area.
        out["log_area"] = np.log1p(X["area"].clip(lower=0))

        # Rainfall features relative to the state's normal (training average).
        out["annual_rainfall"] = X["annual_rainfall"]
        state_mean = X["state"].map(self.state_rain_mean_).fillna(self.global_rain_mean_)
        out["rain_deviation"] = X["annual_rainfall"] - state_mean
        out["rain_deviation_pct"] = 100 * out["rain_deviation"] / state_mean.replace(0, np.nan)

        rain = X["annual_rainfall"]
        cat = np.select(
            [rain < self.rain_q25_, rain > self.rain_q75_], ["low", "high"], default="normal"
        ).astype(object)
        cat[rain.isna().to_numpy()] = np.nan
        out["rain_category"] = cat

        # Long-term technology trend.
        out["year_trend"] = X["year"] - self.min_year_

        return out.replace([np.inf, -np.inf], np.nan)
