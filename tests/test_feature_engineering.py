"""Tests for src.feature_engineering (allowlist, pipelines, leakage safety)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import data_generator as dg
from src import feature_engineering as fe
from src import preprocessing as pp

APPROVED_FEATURES = {
    "arrival_delay_h", "dest_congestion_index", "dest_weather_severity", "destination_port",
    "planned_port_dwell_h", "projected_nonworking_wait_h", "cargo_type", "documentation_complete",
    "inland_distance_km", "planned_inland_h",
}
BUILDERS = {"gbr": fe.build_gbr_preprocessor, "ridge": fe.build_ridge_preprocessor}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pp.load_dataset()


@pytest.fixture(scope="module")
def X(df) -> pd.DataFrame:
    return fe.select_features(df)


@pytest.fixture(scope="module")
def fold1(df):
    return pp.forward_folds(df)[0]


# Allowlist / forbidden columns ---------------------------------------------

def test_single_canonical_allowlist():
    assert set(fe.FEATURES) == APPROVED_FEATURES
    assert fe.NUMERIC_FEATURES == dg.MODEL_NUMERIC_FEATURES
    assert fe.CATEGORICAL_FEATURES == dg.MODEL_CATEGORICAL_FEATURES
    assert fe.TARGET == "downstream_delay_h"


def test_X_contains_exactly_the_allowlist(X):
    assert list(X.columns) == fe.FEATURES
    assert len(X) == 250


@pytest.mark.parametrize("col", sorted(
    {"planned_customs_h", "planned_customs_processing_h", "carrier", "departure_delay_h", "ocean_transit_delay_h",
     "booked_window_missed", "_latent_truck_wait_h", "_latent_windows_skipped", "_latent_slot_availability",
     "actual_delivery", "actual_port_exit", "customs_inspection", "downstream_delay_h", "final_delivery_delay_h",
     "is_delayed_24h", "port_processing_delay_h", "customs_delay_h", "inland_delivery_delay_h",
     "_latent_dest_congestion_true", "_latent_congestion_dwell", "shipment_id", "validation_window"}))
def test_forbidden_columns_cannot_enter_X(df, col):
    with pytest.raises(fe.ForbiddenFeatureError):
        fe.select_features(df, fe.FEATURES + [col])


def test_every_post_checkpoint_target_and_latent_column_forbidden(df):
    for col in dg.POST_CHECKPOINT_COLUMNS + dg.TARGET_COLUMNS + dg.LATENT_COLUMNS:
        assert col in fe.EXPLICITLY_FORBIDDEN
        with pytest.raises(fe.ForbiddenFeatureError):
            fe.select_features(df, [col])


def test_unlisted_columns_rejected(df):
    with pytest.raises(fe.ForbiddenFeatureError, match="outside the feature allowlist"):
        fe.select_features(df, fe.FEATURES + ["origin_congestion_index"])


def test_carrier_only_with_explicit_ablation_flag(df):
    with pytest.raises(fe.ForbiddenFeatureError):
        fe.select_features(df, fe.FEATURES + ["carrier"])
    X_ab = fe.select_features(df, fe.FEATURES + ["carrier"], allow_ablation=True)
    assert "carrier" in X_ab.columns


def test_select_features_is_stateless_and_preserves_missing(df, X):
    assert X["dest_congestion_index"].isna().sum() == df["dest_congestion_index"].isna().sum() > 0
    assert set(X["documentation_complete"].unique()) <= {0.0, 1.0}
    assert X["destination_port"].dtype == object


# Missing values and unseen categories ---------------------------------------

@pytest.mark.parametrize("name", BUILDERS)
def test_missing_congestion_handled(X, fold1, name):
    pre = BUILDERS[name]().fit(X.iloc[fold1.train_idx])
    Z = pre.transform(X)
    assert np.isfinite(Z).all()
    if name == "gbr":
        names = list(pre.get_feature_names_out())
        ind = Z[:, names.index("num__missingindicator_dest_congestion_index")]
        assert np.array_equal(ind, X["dest_congestion_index"].isna().to_numpy().astype(float))


@pytest.mark.parametrize("name", BUILDERS)
def test_unseen_categories_do_not_break_transform(X, fold1, name):
    pre = BUILDERS[name]().fit(X.iloc[fold1.train_idx])
    test = X.iloc[fold1.test_idx[:3]].copy()
    test["destination_port"] = "Atlantis"
    test["cargo_type"] = "livestock"
    Z = pre.transform(test)
    names = list(pre.get_feature_names_out())
    cat_cols = [i for i, n in enumerate(names) if n.startswith("cat__")]
    assert np.isfinite(Z).all()
    assert (Z[:, cat_cols] == 0).all()


# No fitting on test data ------------------------------------------------------

def test_gbr_fitted_statistics_come_from_training_rows_only(X, fold1):
    train, test = X.iloc[fold1.train_idx], X.iloc[fold1.test_idx]
    pre = fe.build_gbr_preprocessor().fit(train)
    imputer = pre.named_transformers_["num"]
    expected = train[fe.NUMERIC_FEATURES].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected)
    before = imputer.statistics_.copy()
    pre.transform(test)
    assert np.array_equal(imputer.statistics_, before)
    # Medians differ from those of the full dataset, so the test rows were not used.
    assert not np.allclose(imputer.statistics_, X[fe.NUMERIC_FEATURES].median().to_numpy())


def test_ridge_scaler_and_encoder_fitted_on_training_rows_only(X, fold1):
    train = X.iloc[fold1.train_idx]
    pre = fe.build_ridge_preprocessor().fit(train)
    num = pre.named_transformers_["num"]
    imputed = train[fe.NUMERIC_FEATURES].fillna(train[fe.NUMERIC_FEATURES].median())
    a_pos = np.maximum(imputed["arrival_delay_h"], 0)
    indicator = train["dest_congestion_index"].isna().astype(float)
    expected_mean = np.r_[imputed.mean().to_numpy(), indicator.mean(), a_pos.mean(),
                          (imputed["dest_congestion_index"] * a_pos).mean()]
    assert np.allclose(num.named_steps["scale"].mean_, expected_mean)
    enc = pre.named_transformers_["cat"]
    for col, cats in zip(fe.CATEGORICAL_FEATURES, enc.categories_):
        assert set(cats) == set(train[col].unique())


def test_encoder_categories_exclude_values_only_in_test(X):
    train = X[X["destination_port"] != "Dubai"]
    pre = fe.build_gbr_preprocessor().fit(train)
    assert "cat__destination_port_Dubai" not in pre.get_feature_names_out()
    Z = pre.transform(X[X["destination_port"] == "Dubai"])
    assert np.isfinite(Z).all()


def test_fresh_clone_per_fold_is_independent(X, df):
    base = fe.build_gbr_preprocessor()
    stats = []
    for f in pp.forward_folds(df):
        pre = clone(base).fit(X.iloc[f.train_idx])
        stats.append(pre.named_transformers_["num"].statistics_.copy())
    assert not np.allclose(stats[0], stats[2])
    with pytest.raises(Exception):
        base.transform(X)  # the template itself was never fitted


# Feature names and outputs ---------------------------------------------------

@pytest.mark.parametrize("name", BUILDERS)
def test_feature_names_traceable(X, fold1, name):
    pre = BUILDERS[name]().fit(X.iloc[fold1.train_idx])
    names = list(pre.get_feature_names_out())
    sources = fe.feature_sources(pre)
    assert list(sources) == names
    assert all(set(src) <= set(fe.FEATURES) for src in sources.values())
    assert {s for src in sources.values() for s in src} == set(fe.FEATURES)
    assert len(names) == pre.transform(X.iloc[:2]).shape[1]


def test_ridge_interactions_and_scaling(X, fold1):
    train = X.iloc[fold1.train_idx]
    pre = fe.build_ridge_preprocessor().fit(train)
    names = list(pre.get_feature_names_out())
    Z = pre.transform(train)
    num_idx = [i for i, n in enumerate(names) if n.startswith("num__")]
    cat_idx = [i for i, n in enumerate(names) if n.startswith("cat__")]
    assert np.allclose(Z[:, num_idx].mean(axis=0), 0, atol=1e-9)
    assert set(np.unique(Z[:, cat_idx])) <= {0.0, 1.0}  # one-hot not scaled
    sources = fe.feature_sources(pre)
    assert sources["num__congestion_x_arrival_delay_pos"] == ("dest_congestion_index", "arrival_delay_h")
    # Unscale the interaction column and compare with its definition on imputed values.
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    k = names.index("num__congestion_x_arrival_delay_pos")
    raw = Z[:, k] * scaler.scale_[k] + scaler.mean_[k]
    imputed = train[fe.NUMERIC_FEATURES].fillna(train[fe.NUMERIC_FEATURES].median())
    assert np.allclose(raw, imputed["dest_congestion_index"] * np.maximum(imputed["arrival_delay_h"], 0))


EXPECTED_RIDGE_NAMES = (
    [f"num__{c}" for c in fe.NUMERIC_FEATURES]
    + ["num__missingindicator_dest_congestion_index", "num__arrival_delay_pos_h", "num__congestion_x_arrival_delay_pos"]
)


@pytest.mark.parametrize("name", BUILDERS)
def test_missing_indicator_present_in_both_pipelines_on_every_fold(X, df, name):
    """Regression: Ridge previously omitted the congestion missing indicator (20 vs 21 features)."""
    splits = [(f.train_idx, f.test_idx) for f in pp.forward_folds(df)] + pp.repeated_route_cv(df)
    for train_idx, test_idx in splits:
        train = X.iloc[train_idx]
        assert train["dest_congestion_index"].isna().any()  # precondition for add_indicator
        pre = BUILDERS[name]().fit(train)
        names = list(pre.get_feature_names_out())
        k = names.index("num__missingindicator_dest_congestion_index")
        assert fe.feature_sources(pre)[names[k]] == ("dest_congestion_index",)
        Z = pre.transform(X.iloc[test_idx])
        is_missing = X.iloc[test_idx]["dest_congestion_index"].isna().to_numpy()
        if name == "gbr":
            assert np.array_equal(Z[:, k], is_missing.astype(float))
        else:  # scaled indicator: two distinct values ordered by missingness
            if is_missing.any() and (~is_missing).any():
                assert Z[is_missing, k].min() > Z[~is_missing, k].max()


def test_ridge_dimension_and_layout(X, fold1):
    train = X.iloc[fold1.train_idx]
    pre = fe.build_ridge_preprocessor().fit(train)
    names = list(pre.get_feature_names_out())
    assert len(names) == 21
    assert names[:11] == EXPECTED_RIDGE_NAMES
    assert sum(n.startswith("cat__destination_port_") for n in names) == 6
    assert sum(n.startswith("cat__cargo_type_") for n in names) == 4
    # The indicator's fitted scaling statistics come from the training rows only.
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    k = names.index("num__missingindicator_dest_congestion_index")
    assert scaler.mean_[k] == pytest.approx(train["dest_congestion_index"].isna().mean())
    assert len(fe.build_gbr_preprocessor().fit(train).get_feature_names_out()) == 19


def test_gbr_numeric_values_are_not_scaled(X, fold1):
    pre = fe.build_gbr_preprocessor().fit(X.iloc[fold1.train_idx])
    names = list(pre.get_feature_names_out())
    Z = pre.transform(X)
    k = names.index("num__arrival_delay_h")
    assert np.allclose(Z[:, k], X["arrival_delay_h"].to_numpy())


def test_interaction_transformer_requires_named_input():
    with pytest.raises(ValueError):
        fe.PositiveArrivalInteractions().fit(np.zeros((3, 8)))


@pytest.mark.parametrize("name", BUILDERS)
def test_finite_outputs_on_all_folds(X, df, name):
    splits = [(f.train_idx, f.test_idx) for f in pp.forward_folds(df)] + pp.repeated_route_cv(df)
    for train_idx, test_idx in splits:
        pre = BUILDERS[name]().fit(X.iloc[train_idx])
        for idx in (train_idx, test_idx):
            Z = pre.transform(X.iloc[idx])
            assert Z.dtype.kind == "f" and np.isfinite(Z).all()


def test_target(df):
    y = fe.get_target(df)
    assert y.name == "downstream_delay_h" and y.notna().all()
