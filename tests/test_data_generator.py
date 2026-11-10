"""Tests for src.data_generator."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import data_generator as dg
from src import delay_propagation as dp

UTC = timezone.utc
TOL = 1e-6


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return dg.generate_shipments()


@pytest.fixture(scope="module")
def config() -> dict:
    return copy.deepcopy(dg.DEFAULT_CONFIG)


def _row_milestones(row: pd.Series, actual_departure_col: str = "_latent_actual_departure"):
    planned = [row[c] for c in ("scheduled_departure", "scheduled_port_arrival", "planned_port_exit",
                                "planned_customs_release", "scheduled_delivery")]
    actual = [row[c] for c in (actual_departure_col, "actual_port_arrival", "actual_port_exit",
                               "actual_customs_release", "actual_delivery")]
    return planned, actual


# 1-4: size and allocation ----------------------------------------------------

def test_exactly_250_unique_shipments(df):
    assert len(df) == 250
    assert df["shipment_id"].is_unique
    assert list(df.columns) == dg.ALL_COLUMNS


def test_exact_route_allocation(df, config):
    expected = {r["id"]: r["records"] for r in config["routes"]}
    assert df["route_id"].value_counts().to_dict() == expected
    assert sorted(expected.values()) == [41, 41, 42, 42, 42, 42]


def test_seven_per_route_per_validation_window(df):
    counts = df[df["validation_window"] > 0].groupby(["route_id", "validation_window"]).size()
    assert len(counts) == 6 * 3
    assert (counts == 7).all()
    assert (df["validation_window"].value_counts().drop(0) == 42).all()


def test_every_2025_shipment_in_exactly_one_window(df, config):
    windows = [(dg._ts(s), dg._ts(e)) for s, e in config["allocation"]["test_windows"]]
    for t1, w in zip(df["actual_port_arrival"], df["validation_window"]):
        memberships = [i for i, (s, e) in enumerate(windows, start=1) if s <= t1 < e]
        if t1.year == 2025:
            assert memberships == [w]
        else:
            assert t1.year == 2024 and memberships == [] and w == 0
    assert (df["actual_port_arrival"].dt.second == 0).all()


# 5-6: reproducibility -------------------------------------------------------

def test_same_seed_reproducible(df):
    pd.testing.assert_frame_equal(df, dg.generate_shipments())


def test_different_seed_differs(df):
    other = dg.generate_shipments(seed=43)
    assert not other["actual_port_arrival"].equals(df["actual_port_arrival"])
    assert not np.allclose(other["final_delivery_delay_h"], df["final_delivery_delay_h"])
    # Allocation counts are fixed by design regardless of seed.
    assert other["route_id"].value_counts().to_dict() == df["route_id"].value_counts().to_dict()


def test_t1_allocation_independent_of_other_parameters(df, config):
    changed = copy.deepcopy(config)
    changed["customs"]["noise_frac"] = 0.5
    changed["ocean"]["sigma_frac"] = 0.1
    changed["missingness"]["actual_departure"] = 0.5
    other = dg.generate_shipments(changed)
    assert other["actual_port_arrival"].equals(df["actual_port_arrival"])
    assert other["route_id"].equals(df["route_id"])


# 7-9: timestamps and identities --------------------------------------------

def test_timestamp_ordering_and_positive_durations(df):
    for _, row in df.iterrows():
        planned, actual = _row_milestones(row)
        for seq in (planned, actual):
            for a, b in zip(seq[:-1], seq[1:]):
                assert b > a
        assert row["actual_customs_processing_start"] >= row["actual_port_exit"]
        assert row["actual_customs_release"] > row["actual_customs_processing_start"]


def test_minimum_stage_durations(df, config):
    for _, row in df.iterrows():
        _, actual = _row_milestones(row)
        a = [dp.hours_between(x, y) for x, y in zip(actual[:-1], actual[1:])]
        km = row["inland_distance_km"]
        assert a[0] >= config["ocean"]["min_frac_of_planned"] * row["planned_transit_h"] - TOL
        assert a[1] >= config["port"]["min_h"] - TOL
        processing = dp.hours_between(row["actual_customs_processing_start"], row["actual_customs_release"])
        assert processing >= config["customs"]["min_processing_h"] - TOL
        assert a[3] >= km / config["inland"]["min_speed_kmh"] + config["inland"]["min_handling_h"] - TOL


def test_final_equals_arrival_plus_downstream(df):
    lhs = df["final_delivery_delay_h"]
    rhs = df["arrival_delay_h"] + df["downstream_delay_h"]
    assert np.allclose(lhs, rhs, atol=TOL)
    assert (df["is_delayed_24h"] == (df["final_delivery_delay_h"] > 24)).all()


def test_stage_delays_reconcile_with_milestones(df):
    for _, row in df.iterrows():
        planned, actual = _row_milestones(row)
        d = dp.stage_delays_from_milestones(planned, actual)
        stored = [row["_latent_departure_delay_h"], row["_latent_ocean_transit_delay_h"],
                  row["port_processing_delay_h"], row["customs_delay_h"], row["inland_delivery_delay_h"]]
        assert np.allclose(d, stored, atol=TOL)
        c = dp.cumulative_delays(d)
        for j in range(5):
            assert c[j] == pytest.approx(dp.hours_between(planned[j], actual[j]), abs=TOL)
        assert row["arrival_delay_h"] == pytest.approx(c[1], abs=TOL)
        assert row["final_delivery_delay_h"] == pytest.approx(c[4], abs=TOL)
        assert row["planned_customs_h"] == pytest.approx(
            row["planned_customs_wait_h"] + row["planned_customs_processing_h"], abs=TOL)


# 10-11: missingness ---------------------------------------------------------

@pytest.mark.parametrize("column, key", [("dest_congestion_index", "dest_congestion_obs"),
                                         ("actual_departure", "actual_departure")])
def test_missingness_rate_within_binomial_tolerance(df, config, column, key):
    p, n = config["missingness"][key], len(df)
    k = int(df[column].isna().sum())
    sd = math.sqrt(n * p * (1 - p))
    assert abs(k - n * p) <= 4 * sd
    assert k > 0


def test_missingness_only_masks_observed_columns(df):
    miss_dep = df["actual_departure"].isna()
    assert (df.loc[miss_dep, ["departure_delay_h", "ocean_transit_delay_h"]].isna()).all().all()
    assert (df.loc[~miss_dep, ["departure_delay_h", "ocean_transit_delay_h"]].notna()).all().all()
    for col in dg.LATENT_COLUMNS + dg.TARGET_COLUMNS + dg.POST_CHECKPOINT_COLUMNS:
        assert df[col].notna().all(), col
    observed_cong = df["dest_congestion_index"].dropna()
    assert np.allclose(observed_cong, df.loc[observed_cong.index, "_latent_dest_congestion_true"])


def test_arrival_delay_observable_when_departure_missing(df):
    miss = df[df["actual_departure"].isna()]
    assert len(miss) > 0
    assert miss["arrival_delay_h"].notna().all()
    for _, row in miss.iterrows():
        a = dp.hours_between(row["scheduled_port_arrival"], row["actual_port_arrival"])
        assert row["arrival_delay_h"] == pytest.approx(a, abs=TOL)
    assert df[["scheduled_departure", "scheduled_port_arrival", "actual_port_arrival"]].notna().all().all()


# 12-14: behaviour -----------------------------------------------------------

def test_early_and_delayed_shipments_present(df):
    final = df["final_delivery_delay_h"]
    assert (final < 0).sum() > 0
    assert (final > 24).sum() > 0


def _shipment_inputs(config, index=0, seed=7):
    rngs = dg.spawn_streams(seed, config["rng_streams"])
    pairs = dg.allocate_checkpoints(config, rngs["t1_allocation"])
    route, t1 = pairs[index]
    ctx = dg.draw_context(rngs["shipment_context"], route, config)
    up = dg.draw_upstream(rngs["upstream"], config)
    down = dg.draw_downstream(rngs["downstream"], config)
    return route, t1, ctx, up, down


@pytest.mark.parametrize("index", range(0, 250, 25))
def test_congestion_intervention_increases_port_delay(config, index):
    route, t1, ctx, up, down = _shipment_inputs(config, index)
    ctx_lo, ctx_hi = dict(ctx, congestion_noise_z=-1.5), dict(ctx, congestion_noise_z=1.5)
    down = dict(down, drift_z=0.0)
    lo = dg.simulate_shipment(route, t1, ctx_lo, up, down, config)
    hi = dg.simulate_shipment(route, t1, ctx_hi, up, down, config)
    assert hi["dest_congestion_index"] > lo["dest_congestion_index"]
    assert hi["port_processing_delay_h"] > lo["port_processing_delay_h"]
    assert hi["arrival_delay_h"] == lo["arrival_delay_h"]  # upstream untouched


@pytest.mark.parametrize("index", range(0, 250, 25))
def test_weather_intervention_increases_port_delay(config, index):
    route, t1, ctx, up, down = _shipment_inputs(config, index)
    lo = dg.simulate_shipment(route, t1, dict(ctx, dest_weather_base=0.1), up, down, config)
    hi = dg.simulate_shipment(route, t1, dict(ctx, dest_weather_base=0.6), up, down, config)
    assert hi["port_processing_delay_h"] > lo["port_processing_delay_h"]


def test_population_relationships_have_expected_sign(df):
    assert df["_latent_dest_congestion_true"].corr(df["port_processing_delay_h"]) > 0
    assert df["arrival_delay_h"].corr(df["port_processing_delay_h"]) > 0
    missed = df["_latent_missed_berth_window"]
    assert df.loc[missed, "_latent_berth_wait_h"].mean() > df.loc[~missed, "_latent_berth_wait_h"].mean()


def test_customs_calendar_and_weekend_propagation(df, config):
    nwd = set(config["calendar"]["non_working_weekdays"])
    for _, row in df.iterrows():
        start_local = dp.to_local(row["actual_customs_processing_start"], row["dest_timezone"])
        assert start_local.weekday() not in nwd
        assert start_local.hour >= config["calendar"]["workday_start_hour"]
        expected_start = dp.next_working_start(row["actual_port_exit"], row["dest_timezone"])
        assert row["actual_customs_processing_start"] == expected_start
    # Lateness pushes some shipments into a weekend the plan did not have, and D4 includes that wait.
    pushed = df[(df["actual_customs_wait_h"] > 0) & (df["planned_customs_wait_h"] == 0)]
    assert len(pushed) > 0
    for _, row in pushed.iterrows():
        processing = dp.hours_between(row["actual_customs_processing_start"], row["actual_customs_release"])
        expected_d4 = row["actual_customs_wait_h"] + processing - row["planned_customs_h"]
        assert row["customs_delay_h"] == pytest.approx(expected_d4, abs=TOL)


def _slot_lateness(df: pd.DataFrame) -> np.ndarray:
    return np.array([dp.hours_between(p, a) for p, a in zip(df["planned_customs_release"], df["actual_customs_release"])])


def test_truck_window_rule_per_shipment(df, config):
    inl = config["inland"]
    window, n_max = inl["slot_window_h"], inl["max_windows_skipped"]
    lateness = _slot_lateness(df)
    for L, (_, row) in zip(lateness, df.iterrows()):
        n = row["_latent_windows_skipped"]
        wait = row["_latent_truck_wait_h"]
        assert 0 <= n <= n_max
        assert row["_latent_slot_availability"] == pytest.approx(
            dg.slot_availability(row["_latent_congestion_dwell"], config))
        assert row["booked_window_missed"] == (L >= window)
        if L < window or n == 0:
            expected = 0.0
        else:
            expected = (math.floor(L / window) + n) * window - L
        assert wait == pytest.approx(expected, abs=TOL)
        assert 0.0 <= wait <= n_max * window + TOL
        if n >= 1 and L >= window:
            assert (n - 1) * window - TOL < wait <= n * window + TOL
    # Early pickup: cargo released before its slot never waits.
    assert (df.loc[lateness < 0, "_latent_truck_wait_h"] == 0).all()
    assert (df.loc[lateness < window, "_latent_truck_wait_h"] == 0).all()


def test_some_late_shipments_get_zero_wait(df, config):
    late = _slot_lateness(df) >= config["inland"]["slot_window_h"]
    assert late.any()
    assert (df.loc[late, "_latent_truck_wait_h"] == 0).any()


def test_slot_availability_decreases_with_congestion(config):
    grid = np.linspace(0.0, 1.0, 101)
    q = np.array([dg.slot_availability(c, config) for c in grid])
    inl = config["inland"]
    assert np.all(np.diff(q) <= 0)
    assert q[0] == pytest.approx(inl["slot_availability_max"])
    assert q.min() >= inl["slot_availability_min"]
    assert dg.slot_availability(0.8, config) < dg.slot_availability(0.2, config)


def test_higher_congestion_cannot_reduce_skipped_windows(config):
    n_max = config["inland"]["max_windows_skipped"]
    us = np.linspace(1e-6, 1.0, 400)
    congestion = np.linspace(0.0, 1.0, 21)
    for u in us:
        ns = [dg.sample_windows_skipped(float(u), dg.slot_availability(c, config), n_max) for c in congestion]
        assert all(b >= a for a, b in zip(ns[:-1], ns[1:]))


@pytest.mark.parametrize("q", [0.9, 0.6, 0.3])
def test_geometric_sampling_matches_theory(q):
    n_max, n = 8, 200_000
    rng = np.random.default_rng(123)
    samples = np.array([dg.sample_windows_skipped(float(1.0 - u), q, n_max) for u in rng.random(n)])
    for k in range(n_max + 1):
        p = (1 - q) ** k * (q if k < n_max else 1.0)
        observed = (samples == k).mean()
        assert abs(observed - p) <= 5 * math.sqrt(p * (1 - p) / n) + 1e-12
    expected_mean = sum((1 - q) ** k for k in range(1, n_max + 1))
    assert samples.mean() == pytest.approx(expected_mean, abs=5 * samples.std() / math.sqrt(n))


def test_sample_windows_skipped_validation():
    assert dg.sample_windows_skipped(1.0, 0.5, 8) == 0
    assert dg.sample_windows_skipped(1e-300, 0.5, 8) == 8
    for u in (0.0, 1.5):
        with pytest.raises(ValueError):
            dg.sample_windows_skipped(u, 0.5, 8)
    with pytest.raises(ValueError):
        dg.sample_windows_skipped(0.5, 1.0, 8)


def test_projected_wait_uses_checkpoint_information_only(df):
    for _, row in df.iterrows():
        expected = dp.projected_customs_wait_h(row["actual_port_arrival"], row["planned_port_dwell_h"], row["dest_timezone"])
        assert row["projected_nonworking_wait_h"] == pytest.approx(expected, abs=TOL)


# 15: leakage / feature schema ----------------------------------------------

def test_feature_allowlist_excludes_future_and_latent(df):
    forbidden = set(dg.POST_CHECKPOINT_COLUMNS) | set(dg.TARGET_COLUMNS) | set(dg.LATENT_COLUMNS)
    features = set(dg.MODEL_FEATURES)
    assert not features & forbidden
    assert not any(c.startswith("_latent_") for c in features)
    assert features <= set(dg.CHECKPOINT_AVAILABLE_COLUMNS)
    assert features <= set(df.columns)
    assert dg.MODEL_TARGET in dg.TARGET_COLUMNS and dg.MODEL_TARGET not in features


def test_feature_allowlist_excludes_redundant_and_ablation_columns():
    features = set(dg.MODEL_FEATURES)
    excluded = {"planned_customs_h", "planned_customs_processing_h", "planned_customs_wait_h", "carrier",
                "departure_delay_h", "ocean_transit_delay_h", "route_id", "shipment_id", "validation_window",
                "actual_departure", "actual_port_arrival", "scheduled_delivery", "booked_window_missed",
                "_latent_truck_wait_h", "_latent_windows_skipped", "_latent_slot_availability"}
    assert {"_latent_truck_wait_h", "_latent_windows_skipped", "_latent_slot_availability"} <= set(dg.LATENT_COLUMNS)
    assert "booked_window_missed" in dg.POST_CHECKPOINT_COLUMNS
    assert not features & excluded
    assert "cargo_type" in features
    assert dg.ABLATION_FEATURES == ["carrier"]


def test_schema_groups_partition_columns():
    groups = dg.column_roles()
    data_groups = ["id_meta", "context", "plan", "observed_at_checkpoint", "post_checkpoint_outcomes",
                   "targets", "latent_audit_only"]
    cols = [c for g in data_groups for c in groups[g]]
    assert len(cols) == len(set(cols)) == len(dg.ALL_COLUMNS)


# Persistence ---------------------------------------------------------------

def test_save_and_reload_round_trip(df, config, tmp_path):
    csv_path, json_path = dg.save_dataset(df, config, tmp_path)
    loaded = dg.load_shipments(csv_path)
    assert len(loaded) == 250
    for col in dg.TIMESTAMP_COLUMNS:
        assert (loaded[col].isna() == df[col].isna()).all()
        diff = (loaded[col] - df[col]).dropna().abs().max()
        assert pd.isna(diff) or diff <= pd.Timedelta(microseconds=1)
    assert np.allclose(loaded["final_delivery_delay_h"], loaded["arrival_delay_h"] + loaded["downstream_delay_h"], atol=TOL)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config"] == json.loads(json.dumps(config))
    assert set(payload) == {"notes", "config", "schema", "output"}


def test_invalid_config_rejected(config):
    bad = copy.deepcopy(config)
    bad["destinations"]["Dubai"]["tz"] = "Asia/Nowhere"
    with pytest.raises(ValueError):
        dg.generate_shipments(bad)
    bad = copy.deepcopy(config)
    bad["carriers"]["Carrier A"]["share"] = 0.9
    with pytest.raises(ValueError):
        dg.generate_shipments(bad)
