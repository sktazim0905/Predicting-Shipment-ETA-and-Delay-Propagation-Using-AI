"""Tests for src.predict (single-shipment prediction interface)."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src import data_generator as dg
from src import delay_propagation as dp
from src import explainability as ex
from src import feature_engineering as fe
from src import predict as pr
from src import preprocessing as pp
from src import train as tr


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pp.load_dataset()


@pytest.fixture(scope="module")
def predictors() -> dict[str, pr.ShipmentPredictor]:
    return {name: pr.ShipmentPredictor(name) for name in tr.LEARNED_MODELS}


@pytest.fixture(scope="module")
def record(df) -> dict:
    return pr.checkpoint_record(df.iloc[200])


@pytest.fixture(scope="module")
def missing_cong_record(df) -> dict:
    return pr.checkpoint_record(df[df["dest_congestion_index"].isna()].iloc[0])


# 5-6. Inputs ----------------------------------------------------------------

@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_predict_without_future_timestamps(predictors, record, name):
    future = {"actual_port_exit", "actual_customs_processing_start", "actual_customs_release", "actual_delivery"}
    assert not set(record) & future
    assert not set(record) & pr.FORBIDDEN_INPUTS
    result = predictors[name].predict(record)
    assert result.model_name == name


@pytest.mark.parametrize("field", ["actual_port_exit", "actual_customs_release", "actual_delivery",
                                   "port_processing_delay_h", "customs_inspection", "booked_window_missed",
                                   "downstream_delay_h", "final_delivery_delay_h", "_latent_truck_wait_h",
                                   "_latent_congestion_dwell", "validation_window"])
def test_forbidden_fields_rejected(predictors, record, df, field):
    bad = dict(record, **{field: df.iloc[200][field]})
    with pytest.raises(pr.InvalidShipmentError, match="not allowed"):
        predictors["gbr"].predict(bad)


def test_unknown_field_rejected(predictors, record):
    with pytest.raises(pr.InvalidShipmentError, match="unknown fields"):
        predictors["gbr"].predict(dict(record, vessel_name="X"))


@pytest.mark.parametrize("field", ["actual_port_arrival", "scheduled_delivery", "cargo_type", "planned_inland_h"])
def test_missing_required_field_rejected(predictors, record, field):
    bad = {k: v for k, v in record.items() if k != field}
    with pytest.raises(pr.InvalidShipmentError, match="missing required"):
        predictors["gbr"].predict(bad)


@pytest.mark.parametrize("update, match", [
    ({"dest_congestion_index": 1.5}, "within"),
    ({"dest_weather_severity": -0.1}, "within"),
    ({"planned_port_dwell_h": 0.0}, "> 0"),
    ({"planned_inland_h": float("nan")}, "missing required"),
    ({"inland_distance_km": "far"}, "number"),
    ({"destination_port": "Atlantis"}, "unknown destination"),
    ({"cargo_type": "livestock"}, "unknown cargo"),
    ({"documentation_complete": "yes"}, "boolean"),
    ({"actual_port_arrival": datetime(2025, 3, 1, 12)}, "timezone-aware"),
    ({"scheduled_delivery": "not a date"}, "ISO-8601"),
    ({"dest_timezone": "Europe/Paris"}, "does not match"),
    ({"arrival_delay_h": 999.0}, "disagrees"),
])
def test_invalid_values_rejected(predictors, record, update, match):
    with pytest.raises(pr.InvalidShipmentError, match=match):
        predictors["gbr"].predict(dict(record, **update))


def test_consistent_derived_fields_and_iso_strings_accepted(predictors, record, df):
    row = df.iloc[200]
    rec = dict(record, arrival_delay_h=row["arrival_delay_h"],
               projected_nonworking_wait_h=row["projected_nonworking_wait_h"],
               scheduled_delivery=row["scheduled_delivery"].isoformat())
    assert predictors["gbr"].predict(rec).predicted_final_delay_h == pytest.approx(
        predictors["gbr"].predict(record).predicted_final_delay_h)


def test_prepare_input_reproduces_dataset_features(df):
    X_ref = fe.select_features(df)
    for i in range(len(df)):
        X, _ = pr.prepare_input(pr.checkpoint_record(df.iloc[i]))
        expected = X_ref.iloc[[i]].reset_index(drop=True)
        pd.testing.assert_frame_equal(X, expected, check_exact=False, atol=1e-6)


# 4. Missing congestion --------------------------------------------------------

@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
@pytest.mark.parametrize("value", [None, float("nan")])
def test_missing_congestion_handled(predictors, missing_cong_record, name, value):
    result = predictors[name].predict(dict(missing_cong_record, dest_congestion_index=value))
    assert math.isfinite(result.predicted_final_delay_h)
    assert "num__missingindicator_dest_congestion_index" in result.feature_contributions_h


# 7-9. ETA, arrival delay counted once, finiteness ---------------------------

@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_eta_equals_schedule_plus_final_delay(predictors, record, name):
    r = predictors[name].predict(record)
    sched = datetime.fromisoformat(r.scheduled_delivery_utc)
    eta = datetime.fromisoformat(r.predicted_eta_utc)
    assert dp.hours_between(sched, eta) == pytest.approx(r.predicted_final_delay_h, abs=1e-6)
    local = datetime.fromisoformat(r.predicted_eta_local)
    assert str(local.tzinfo) != "UTC" or r.destination_timezone == "UTC"
    assert abs(dp.hours_between(eta, local)) < 1 / 60  # local display is truncated to the minute
    assert r.destination_timezone == dg.DEFAULT_CONFIG["destinations"][r.destination_port]["tz"]


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_arrival_delay_counted_once(predictors, record, df, name):
    r = predictors[name].predict(record)
    row = df.iloc[200]
    assert r.arrival_delay_h == pytest.approx(row["arrival_delay_h"], abs=1e-6)
    assert r.predicted_final_delay_h == pytest.approx(r.arrival_delay_h + r.predicted_downstream_delay_h)
    shap_total = r.shap_base_value_h + sum(r.feature_contributions_h.values())
    assert shap_total == pytest.approx(r.predicted_downstream_delay_h, abs=1e-6)  # SHAP explains Y_hat only
    assert r.shap_base_value_h + sum(r.group_contributions_h.values()) == pytest.approx(
        r.predicted_downstream_delay_h, abs=1e-6)
    assert not math.isclose(shap_total + r.arrival_delay_h, r.predicted_downstream_delay_h, abs_tol=1e-3)


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_outputs_finite(predictors, df, name):
    for i in range(0, len(df), 10):
        r = predictors[name].predict(pr.checkpoint_record(df.iloc[i]))
        numbers = [r.arrival_delay_h, r.predicted_downstream_delay_h, r.predicted_final_delay_h, r.shap_base_value_h,
                   *r.feature_contributions_h.values(), *r.group_contributions_h.values()]
        assert all(math.isfinite(v) for v in numbers)


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_predictions_match_batch_pipeline(predictors, df, name):
    X = fe.select_features(df)
    batch = predictors[name].pipeline.predict(X)
    for i in (0, 57, 123, 249):
        r = predictors[name].predict(pr.checkpoint_record(df.iloc[i]))
        assert r.predicted_downstream_delay_h == pytest.approx(batch[i], abs=1e-9)


# 10-11. Reproducibility and explanations --------------------------------------

@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_repeated_predictions_identical(predictors, record, name):
    a = predictors[name].predict(record).to_dict()
    b = predictors[name].predict(record).to_dict()
    c = pr.ShipmentPredictor(name).predict(record).to_dict()
    assert a == b == c


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_explanations_generated(predictors, record, name):
    r = predictors[name].predict(record)
    assert set(r.group_contributions_h) == set(ex.GROUPS)
    assert len(r.feature_contributions_h) == (21 if name == "ridge" else 19)
    text = r.explanation
    assert f"the {name} model" in text
    assert "not physical causes" in text
    assert "model average" not in text
    assert f"Observed arrival delay A = {r.arrival_delay_h:+.1f} h" in text
    assert f"Predicted downstream delay Y_hat = {r.predicted_downstream_delay_h:+.1f} h" in text
    assert f"SHAP baseline = {r.shap_base_value_h:+.1f} h" in text
    total = sum(r.feature_contributions_h.values())
    assert f"relative to that baseline total {total:+.1f} h" in text
    assert f"A + Y_hat = ({r.arrival_delay_h:+.1f} h) + ({r.predicted_downstream_delay_h:+.1f} h) = " \
           f"{r.predicted_final_delay_h:+.1f} h" in text
    # The five quantities appear in order: A, Y_hat, baseline, contributions, final.
    positions = [text.index(s) for s in ("1) Observed", "2) Predicted downstream", "3) SHAP baseline",
                                         "4) Feature contributions", "5) Predicted final")]
    assert positions == sorted(positions)
    d = r.to_dict()
    assert d["shipment_id"] == record["shipment_id"]


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_final_model_results_labelled_in_sample(predictors, record, name):
    r = predictors[name].predict(record)
    assert r.model_scope == pr.SCOPE_IN_SAMPLE and "IN-SAMPLE" in r.explanation
    new = dict(record, shipment_id="NEW-0001")
    assert predictors[name].predict(new).model_scope == pr.SCOPE_NEW_RECORD


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_held_out_explanation_shp_0209(df, name):
    result, check = pr.explain_held_out("SHP-0209", name, df)
    assert check["fold"] == 3 and check["shipment_in_training_rows"] is False
    assert check["latest_training_delivery_utc"] < check["test_window_utc"][0] <= check["shipment_port_arrival_utc"]
    assert check["shap_additivity_error_h"] < 1e-6 and check["group_reconciliation_error_h"] < 1e-9
    assert "HELD-OUT" in result.model_scope and "fold-3" in result.model_scope
    # Reproduces the saved forward evaluation prediction exactly.
    fwd = tr.forward_evaluation(df, models=[name]).predictions.set_index("shipment_id")
    assert result.predicted_downstream_delay_h == pytest.approx(fwd.loc["SHP-0209", f"pred_downstream_{name}"], abs=1e-9)
    # Observed A and predicted Y_hat are separate; final = A + Y_hat.
    assert result.arrival_delay_h == pytest.approx(check["actual_arrival_delay_h"], abs=1e-6)
    assert result.predicted_final_delay_h == pytest.approx(result.arrival_delay_h + result.predicted_downstream_delay_h)
    assert check["downstream_prediction_error_h"] == pytest.approx(
        result.predicted_downstream_delay_h - check["actual_downstream_delay_h"])


def test_held_out_rejects_2024_shipments(df):
    sid = df.loc[df["actual_port_arrival"].dt.year == 2024, "shipment_id"].iloc[0]
    with pytest.raises(ValueError, match="not in a forward test window"):
        pr.explain_held_out(sid, "gbr", df)


# Scenario API (used by the dashboard) ----------------------------------------

@pytest.mark.parametrize("local, tz, expected", [
    (datetime(2026, 3, 2, 21, 30), "Europe/Amsterdam", datetime(2026, 3, 2, 20, 30)),      # CET +1
    (datetime(2025, 7, 1, 12, 0), "Europe/Amsterdam", datetime(2025, 7, 1, 10, 0)),        # CEST +2
    (datetime(2025, 1, 15, 8, 0), "America/New_York", datetime(2025, 1, 15, 13, 0)),       # EST -5
    (datetime(2025, 7, 15, 8, 0), "America/Los_Angeles", datetime(2025, 7, 15, 15, 0)),    # PDT -7
    (datetime(2025, 9, 7, 4, 49), "Asia/Singapore", datetime(2025, 9, 6, 20, 49)),         # +8
    (datetime(2025, 9, 7, 4, 49), "Asia/Dubai", datetime(2025, 9, 7, 0, 49)),              # +4
])
def test_local_to_utc(local, tz, expected):
    out = pr.local_to_utc(local, tz)
    assert out == expected.replace(tzinfo=dp.UTC) and out.utcoffset().total_seconds() == 0


@pytest.mark.parametrize("local, tz, match", [
    (datetime(2025, 3, 30, 2, 30), "Europe/Amsterdam", "does not exist"),
    (datetime(2025, 10, 26, 2, 30), "Europe/Amsterdam", "ambiguous"),
    (datetime(2025, 3, 9, 2, 15), "America/New_York", "does not exist"),
    (datetime(2025, 11, 2, 1, 30), "America/New_York", "ambiguous"),
])
def test_local_to_utc_rejects_dst_gaps_and_overlaps(local, tz, match):
    with pytest.raises(pr.InvalidShipmentError, match=match):
        pr.local_to_utc(local, tz)


def test_local_to_utc_rejects_aware_and_non_datetime():
    with pytest.raises(pr.InvalidShipmentError, match="naive"):
        pr.local_to_utc(datetime(2025, 1, 1, tzinfo=dp.UTC), "Asia/Dubai")
    with pytest.raises(pr.InvalidShipmentError):
        pr.local_to_utc("2025-01-01 10:00", "Asia/Dubai")


def _scenario_inputs(row) -> dict:
    tz = row["dest_timezone"]
    naive_local = lambda ts: dp.to_local(ts.to_pydatetime(), tz).replace(tzinfo=None)
    return dict(destination_port=row["destination_port"], cargo_type=row["cargo_type"],
                documentation_complete=bool(row["documentation_complete"]),
                dest_congestion_index=None if pd.isna(row["dest_congestion_index"]) else float(row["dest_congestion_index"]),
                dest_weather_severity=float(row["dest_weather_severity"]),
                planned_port_dwell_h=float(row["planned_port_dwell_h"]),
                inland_distance_km=float(row["inland_distance_km"]), planned_inland_h=float(row["planned_inland_h"]),
                arrival_delay_h=float(row["arrival_delay_h"]),
                actual_port_arrival_local=naive_local(row["actual_port_arrival"]),
                scheduled_delivery_local=naive_local(row["scheduled_delivery"]))


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_scenario_matches_record_path(predictors, df, name):
    """A scenario built from a dataset row's checkpoint values reproduces the existing prediction path."""
    rows = [df[df["shipment_id"] == "SHP-0209"].iloc[0], df[df["dest_congestion_index"].isna()].iloc[0],
            df[df["destination_port"] == "New York"].iloc[3]]
    for row in rows:
        s = pr.predict_scenario(name, predictor=predictors[name], **_scenario_inputs(row))
        r = predictors[name].predict(pr.checkpoint_record(row))
        assert s.model_scope == pr.SCOPE_NEW_RECORD  # no shipment_id: treated as a new scenario
        assert s.arrival_delay_h == pytest.approx(row["arrival_delay_h"], abs=1e-6)
        assert s.predicted_downstream_delay_h == pytest.approx(r.predicted_downstream_delay_h, abs=1e-6)
        assert s.predicted_final_delay_h == pytest.approx(r.predicted_final_delay_h, abs=1e-6)
        assert s.predicted_eta_utc[:16] == r.predicted_eta_utc[:16]
        assert s.model_inputs["projected_nonworking_wait_h"] == pytest.approx(row["projected_nonworking_wait_h"], abs=1e-6)


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_scenario_definitions_and_missing_congestion(predictors, df, name):
    inputs = _scenario_inputs(df.iloc[10]) | {"dest_congestion_index": None, "arrival_delay_h": 14.5}
    r = pr.predict_scenario(name, predictor=predictors[name], **inputs)
    assert r.arrival_delay_h == pytest.approx(14.5, abs=1e-6)
    assert r.predicted_final_delay_h == pytest.approx(r.arrival_delay_h + r.predicted_downstream_delay_h)
    sched, eta = datetime.fromisoformat(r.scheduled_delivery_utc), datetime.fromisoformat(r.predicted_eta_utc)
    assert dp.hours_between(sched, eta) == pytest.approx(r.predicted_final_delay_h, abs=1e-6)
    assert r.model_inputs["dest_congestion_index"] is None
    assert "num__missingindicator_dest_congestion_index" in r.feature_contributions_h
    assert r.shap_base_value_h + sum(r.feature_contributions_h.values()) == pytest.approx(
        r.predicted_downstream_delay_h, abs=1e-6)
    assert all(math.isfinite(v) for v in r.feature_contributions_h.values())


def test_scenario_validation(predictors, df):
    base = _scenario_inputs(df.iloc[10])
    with pytest.raises(pr.InvalidShipmentError, match="unknown destination"):
        pr.scenario_record(**(base | {"destination_port": "Atlantis"}))
    with pytest.raises(pr.InvalidShipmentError, match="within"):
        pr.predict_scenario("gbr", predictor=predictors["gbr"], **(base | {"dest_congestion_index": 1.4}))
    with pytest.raises(pr.InvalidShipmentError, match="finite"):
        pr.scenario_record(**(base | {"arrival_delay_h": float("inf")}))
    with pytest.raises(ValueError, match="holds"):
        pr.predict_scenario("ridge", predictor=predictors["gbr"], **base)


def test_predictor_rejects_mismatched_artifacts(tmp_path):
    with pytest.raises(ValueError, match="holds"):
        pr.ShipmentPredictor("ridge", model_path=tr.MODELS_DIR / "gbr_model.joblib")
    other = tmp_path / "other.csv"
    other.write_bytes(pp.DATA_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        pr.ShipmentPredictor("gbr", background_path=other)
    with pytest.raises(ValueError):
        pr.ShipmentPredictor("carry_forward")
