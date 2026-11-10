"""Checkpoint feature selection and leakage-safe model preprocessing.

The canonical feature allowlist lives in :mod:`src.data_generator`
(``MODEL_NUMERIC_FEATURES`` / ``MODEL_CATEGORICAL_FEATURES``); this module
imports it rather than keeping a copy.

Row-wise, stateless preparation (column selection, casting booleans to 0/1)
happens in :func:`select_features`. Everything that learns from data
(imputation, scaling, one-hot categories) lives inside the ColumnTransformers
returned by :func:`build_gbr_preprocessor` and :func:`build_ridge_preprocessor`,
which must be cloned and fitted on each training fold only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src import data_generator as dg

NUMERIC_FEATURES: list[str] = list(dg.MODEL_NUMERIC_FEATURES)
CATEGORICAL_FEATURES: list[str] = list(dg.MODEL_CATEGORICAL_FEATURES)
FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET: str = dg.MODEL_TARGET

ARRIVAL_COL = "arrival_delay_h"
CONGESTION_COL = "dest_congestion_index"
ARRIVAL_POS_NAME = "arrival_delay_pos_h"
INTERACTION_NAME = "congestion_x_arrival_delay_pos"

#: Columns that must never be model inputs at the port-arrival checkpoint.
EXPLICITLY_FORBIDDEN: frozenset[str] = frozenset(
    set(dg.POST_CHECKPOINT_COLUMNS) | set(dg.TARGET_COLUMNS) | set(dg.LATENT_COLUMNS)
    | {"planned_customs_h", "planned_customs_processing_h", "planned_customs_wait_h",
       "carrier", "departure_delay_h", "ocean_transit_delay_h", "shipment_id", "validation_window"}
)


class ForbiddenFeatureError(ValueError):
    """Raised when a requested feature is not allowed at the checkpoint."""


def assert_allowed(features: Sequence[str], allow_ablation: bool = False) -> None:
    """Reject features outside the allowlist (plus ``carrier`` if ``allow_ablation``)."""
    allowed = set(FEATURES) | (set(dg.ABLATION_FEATURES) if allow_ablation else set())
    bad_forbidden = [f for f in features if f in EXPLICITLY_FORBIDDEN and f not in allowed]
    if bad_forbidden:
        raise ForbiddenFeatureError(f"forbidden columns requested as features: {bad_forbidden}")
    unknown = [f for f in features if f not in allowed]
    if unknown:
        raise ForbiddenFeatureError(f"columns outside the feature allowlist: {unknown}")


def select_features(df: pd.DataFrame, features: Sequence[str] | None = None,
                    allow_ablation: bool = False) -> pd.DataFrame:
    """Return the model input frame ``X`` (stateless, row-wise).

    Booleans become 0/1 floats and categoricals become plain ``object``
    strings. Missing values are preserved for the pipeline imputers.
    """
    features = list(FEATURES if features is None else features)
    assert_allowed(features, allow_ablation=allow_ablation)
    X = df.loc[:, features].copy()
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(float)
        elif col in CATEGORICAL_FEATURES or col in dg.ABLATION_FEATURES:
            X[col] = X[col].astype(object)
    return X


def get_target(df: pd.DataFrame) -> pd.Series:
    """Model target ``downstream_delay_h``."""
    return df[TARGET].astype(float)


class PositiveArrivalInteractions(TransformerMixin, BaseEstimator):
    """Append ``max(A, 0)`` and ``congestion * max(A, 0)`` to an imputed numeric block.

    Stateless: ``fit`` only records input column names. Must run *after*
    imputation so the interaction never sees NaN congestion.
    """

    def __init__(self, arrival_col: str = ARRIVAL_COL, congestion_col: str = CONGESTION_COL):
        self.arrival_col = arrival_col
        self.congestion_col = congestion_col

    def fit(self, X, y=None):
        names = getattr(X, "columns", None)
        if names is None:
            raise ValueError("PositiveArrivalInteractions requires a DataFrame with column names")
        self.feature_names_in_ = np.asarray(list(names), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        if self.arrival_col not in self.feature_names_in_ or self.congestion_col not in self.feature_names_in_:
            raise ValueError("arrival and congestion columns must be present")
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        arr = np.asarray(X, dtype=float)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {arr.shape[1]}")
        names = list(self.feature_names_in_)
        a_pos = np.maximum(arr[:, names.index(self.arrival_col)], 0.0)
        inter = arr[:, names.index(self.congestion_col)] * a_pos
        return np.column_stack([arr, a_pos, inter])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        base = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(base + [ARRIVAL_POS_NAME, INTERACTION_NAME], dtype=object)


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_gbr_preprocessor(numeric: Sequence[str] = NUMERIC_FEATURES,
                           categorical: Sequence[str] = CATEGORICAL_FEATURES) -> ColumnTransformer:
    """Median imputation with missing indicators + one-hot; no scaling."""
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), list(numeric)),
            ("cat", _one_hot(), list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_ridge_preprocessor(numeric: Sequence[str] = NUMERIC_FEATURES,
                             categorical: Sequence[str] = CATEGORICAL_FEATURES) -> ColumnTransformer:
    """Median imputation (+ missing indicators) -> positive-arrival interactions -> StandardScaler.

    The numeric block (originals, missing indicators, interactions) is scaled;
    one-hot categorical columns are not.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True).set_output(transform="pandas")),
        ("interact", PositiveArrivalInteractions()),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [
            ("num", numeric_pipe, list(numeric)),
            ("cat", _one_hot(), list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def feature_sources(fitted: ColumnTransformer) -> dict[str, tuple[str, ...]]:
    """Map each transformed feature name to the raw input column(s) it derives from."""
    categorical = sorted(CATEGORICAL_FEATURES + list(dg.ABLATION_FEATURES), key=len, reverse=True)
    out: dict[str, tuple[str, ...]] = {}
    for name in fitted.get_feature_names_out():
        prefix, _, rest = name.partition("__")
        if rest.startswith("missingindicator_"):
            out[name] = (rest[len("missingindicator_"):],)
        elif rest == ARRIVAL_POS_NAME:
            out[name] = (ARRIVAL_COL,)
        elif rest == INTERACTION_NAME:
            out[name] = (CONGESTION_COL, ARRIVAL_COL)
        elif prefix == "cat":
            match = next((c for c in categorical if rest.startswith(c + "_")), None)
            if match is None:
                raise ValueError(f"cannot trace categorical feature {name!r}")
            out[name] = (match,)
        else:
            out[name] = (rest,)
    return out
