"""Tests for src.preprocessing (schema, targets, splits)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as pp


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pp.load_dataset()


@pytest.fixture(scope="module")
def folds(df):
    return pp.forward_folds(df)


# 1. Schema validation ------------------------------------------------------

def test_frozen_dataset_validates(df):
    assert df.shape == (250, 62)
    assert df["shipment_id"].is_unique


def test_missing_column_rejected(df):
    with pytest.raises(pp.DatasetValidationError, match="missing required columns"):
        pp.validate_dataset(df.drop(columns=["arrival_delay_h"]))


def test_wrong_dtypes_rejected(df):
    bad = df.copy()
    bad["actual_delivery"] = bad["actual_delivery"].astype(str)
    with pytest.raises(pp.DatasetValidationError, match="UTC datetime"):
        pp.validate_dataset(bad)
    bad = df.copy()
    bad["arrival_delay_h"] = bad["arrival_delay_h"].astype(str)
    with pytest.raises(pp.DatasetValidationError, match="numeric"):
        pp.validate_dataset(bad)


def test_unexpected_missing_values_rejected(df):
    bad = df.copy()
    bad.loc[0, "arrival_delay_h"] = np.nan
    with pytest.raises(pp.DatasetValidationError, match="unexpected missing"):
        pp.validate_dataset(bad)


def test_duplicate_ids_rejected(df):
    bad = df.copy()
    bad.loc[1, "shipment_id"] = bad.loc[0, "shipment_id"]
    with pytest.raises(pp.DatasetValidationError, match="not unique"):
        pp.validate_dataset(bad)


def test_timestamp_order_violation_rejected(df):
    bad = df.copy()
    bad.loc[3, "actual_port_exit"] = bad.loc[3, "actual_port_arrival"] - pd.Timedelta(hours=1)
    with pytest.raises(pp.DatasetValidationError, match="not increasing"):
        pp.validate_timestamp_order(bad)


# 2. Targets ----------------------------------------------------------------

def test_downstream_target_calculation(df):
    y = pp.compute_downstream_delay(df)
    assert np.allclose(y, df["downstream_delay_h"], atol=pp.IDENTITY_TOL_H)
    assert np.allclose(df["final_delivery_delay_h"], df["arrival_delay_h"] + y, atol=pp.IDENTITY_TOL_H)


def test_target_corruption_detected(df):
    bad = df.copy()
    bad.loc[5, "final_delivery_delay_h"] += 1.0
    with pytest.raises(pp.DatasetValidationError):
        pp.verify_targets(bad)


def test_stage_identities(df):
    assert pp.verify_stage_identities(df) <= pp.IDENTITY_TOL_H


# 3-7. Forward folds ---------------------------------------------------------

def test_three_folds_with_expected_windows(folds):
    assert [f.fold for f in folds] == [1, 2, 3]
    assert [(f.test_start.isoformat(), f.test_end.isoformat()) for f in folds] == [
        ("2025-01-01T00:00:00+00:00", "2025-05-01T00:00:00+00:00"),
        ("2025-05-01T00:00:00+00:00", "2025-09-01T00:00:00+00:00"),
        ("2025-09-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    ]


def test_exact_forward_fold_membership(df, folds):
    t1 = df["actual_port_arrival"]
    for f in folds:
        start, end = pd.Timestamp(f.test_start), pd.Timestamp(f.test_end)
        expected_test = np.flatnonzero(((t1 >= start) & (t1 < end)).to_numpy())
        assert np.array_equal(f.test_idx, expected_test)
        # Independent cross-check against the generator's window label.
        assert np.array_equal(f.test_idx, np.flatnonzero((df["validation_window"] == f.fold).to_numpy()))


def test_label_availability_embargo(df, folds):
    t1, delivered = df["actual_port_arrival"], df["actual_delivery"]
    for f in folds:
        start = pd.Timestamp(f.test_start)
        train = df.iloc[f.train_idx]
        assert (train["actual_port_arrival"] < start).all()
        assert (train["actual_delivery"] < start).all()
        expected = np.flatnonzero(((t1 < start) & (delivered < start)).to_numpy())
        assert np.array_equal(f.train_idx, expected)
        embargoed = ((t1 < start) & (delivered >= start)).to_numpy()
        assert f.n_embargoed == int(embargoed.sum())
        assert not set(np.flatnonzero(embargoed)) & set(f.train_idx)
    assert sum(f.n_embargoed for f in folds) > 0  # the embargo is active on this dataset


def test_no_train_test_overlap(df, folds):
    for f in folds:
        assert not set(f.train_idx) & set(f.test_idx)
        assert df.iloc[f.train_idx]["actual_port_arrival"].max() < df.iloc[f.test_idx]["actual_port_arrival"].min()


def test_every_2025_shipment_in_exactly_one_test_fold(df, folds):
    counts = np.zeros(len(df), dtype=int)
    for f in folds:
        counts[f.test_idx] += 1
    is_2025 = (df["actual_port_arrival"].dt.year == 2025).to_numpy()
    assert (counts[is_2025] == 1).all()
    assert (counts[~is_2025] == 0).all()


def test_seven_per_route_per_fold(df, folds):
    for f in folds:
        counts = df.iloc[f.test_idx]["route_id"].value_counts()
        assert len(counts) == 6 and (counts == 7).all()
        assert len(f.test_idx) == 42


def test_training_sets_expand(folds):
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    for a, b in zip(folds[:-1], folds[1:]):
        assert set(a.train_idx) <= set(b.train_idx)


# 8. Secondary CV ------------------------------------------------------------

def test_repeated_route_stratified_cv(df):
    splits = pp.repeated_route_cv(df)
    assert len(splits) == 25
    routes = df["route_id"].to_numpy()
    totals = df["route_id"].value_counts()
    for r in range(5):
        seen = np.zeros(len(df), dtype=int)
        for train, test in splits[r * 5:(r + 1) * 5]:
            assert not set(train) & set(test)
            assert len(train) + len(test) == len(df)
            seen[test] += 1
            per_route = pd.Series(routes[test]).value_counts()
            for route, n in totals.items():
                assert per_route.get(route, 0) in (n // 5, n // 5 + 1)
        assert (seen == 1).all()


def test_cv_is_deterministic(df):
    a, b = pp.repeated_route_cv(df), pp.repeated_route_cv(df)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    c = pp.repeated_route_cv(df, random_state=7)
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(a, c))
