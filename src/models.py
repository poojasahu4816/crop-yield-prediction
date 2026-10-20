"""Model pipelines: FeatureBuilder -> preprocessing -> model."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from .features import FeatureBuilder


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:  # scaling is needed for Ridge, not for tree models
        num_steps.append(("scaler", StandardScaler()))
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Rare categories (fewer than 5 rows) are grouped together automatically.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=5, sparse_output=False)),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES),
    ])


MODELS = {
    # name: (estimator factory, needs_scaling)
    "ridge": (lambda: Ridge(alpha=1.0), True),
    "random_forest": (lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE), False),
    "gradient_boosting": (lambda: HistGradientBoostingRegressor(
        learning_rate=0.1, max_depth=6, max_iter=300, random_state=RANDOM_STATE), False),
}

# Small search spaces for the optional --tune step.
PARAM_DISTRIBUTIONS = {
    "ridge": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [4, 6, 8],
        "model__max_iter": [300, 600],
        "model__l2_regularization": [0.0, 1.0, 10.0],
    },
}


def build_pipeline(name: str) -> Pipeline:
    factory, scale = MODELS[name]
    return Pipeline([
        ("features", FeatureBuilder()),
        ("prep", make_preprocessor(scale)),
        ("model", factory()),
    ])
