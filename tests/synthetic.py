"""Synthetic data generator used ONLY for smoke tests (not real data)."""
import numpy as np
import pandas as pd


def make_synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    crops = {"Rice": 2.2, "Wheat": 2.8, "Maize": 2.5, "Gram": 1.0, "Groundnut": 1.4, "Jowar": 0.9}
    states = ["Maharashtra", "Punjab", "Bihar", "Gujarat", "Assam", "Karnataka"]
    seasons = ["Kharif", "Rabi", "Whole Year", "Summer"]
    crop = rng.choice(list(crops), n)
    state = rng.choice(states, n)
    season = rng.choice(seasons, n)
    year = rng.integers(1997, 2021, n)
    area = np.exp(rng.normal(9.5, 1.5, n))
    state_rain = {s: 800 + 120 * i for i, s in enumerate(states)}
    rain = np.array([state_rain[s] for s in state]) + rng.normal(0, 150, n)
    fert_ha = rng.uniform(30, 180, n)
    pest_ha = rng.uniform(0.1, 1.2, n)
    base = np.array([crops[c] for c in crop])
    y = (base * (1 + 0.015 * (year - 1997))
         * (1 - ((rain - 1100) / 900) ** 2 * 0.4)
         * (1 + 0.003 * np.sqrt(fert_ha)) + rng.normal(0, 0.15, n))
    y = np.clip(y, 0.1, None)
    df = pd.DataFrame({
        "Crop": crop, "Crop_Year": year, "Season": season, "State": state,
        "Area": area, "Production": y * area, "Annual_Rainfall": rain,
        "Fertilizer": fert_ha * area, "Pesticide": pest_ha * area, "Yield": y,
    })
    # add realistic mess: spaces, missing values, a duplicate, an invalid row
    df.loc[::50, "Crop"] = df.loc[::50, "Crop"] + " "
    df.loc[5::97, "Annual_Rainfall"] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.loc[10, "Area"] = -5
    return df
