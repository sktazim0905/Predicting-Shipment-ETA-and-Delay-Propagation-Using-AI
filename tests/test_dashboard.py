"""Tests for the dashboard data layer and headless runs of the Streamlit app."""

from __future__ import annotations

import hashlib
import json
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import dashboard_data as dd
from src import data_generator as dg
from src import explainability as ex
from src import train as tr

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "dashboard" / "app.py")
TIMEOUT = 180


@pytest.fixture(scope="module")
def artifacts():
    return {"mc": dd.load_model_comparison(), "metrics": dd.load_training_metrics(),
            "fwd": dd.load_forward_predictions(), "shap": dd.load_forward_shap_values(), "df": dd.load_dataset()}


# Data layer -------------------------------------------------------------------

def test_overview_values_come_from_artifacts(artifacts):
    s = dd.dataset_summary(artifacts["df"], artifacts["metrics"])
    assert (s["n_shipments"], s["n_routes"], s["n_folds"], s["n_held_out"]) == (250, 6, 3, 126)
    table = dd.comparison_table(artifacts["mc"])
    mc = artifacts["mc"].set_index("model")
    assert list(table["Model"]) == ["Schedule-only", "Carry-forward", "Ridge", "Gradient Boosting"]
    for label, model in zip(table["Model"], dd.MODEL_LABELS):
        row = table[table["Model"] == label].iloc[0]
        assert row["MAE (h)"] == mc.loc[model, "mae_h"] and row["RMSE (h)"] == mc.loc[model, "rmse_h"]
        assert row["P90 abs. error (h)"] == mc.loc[model, "p90_ae_h"]
        assert row["Within ±24 h (%)"] == mc.loc[model, "within_24h_pct"]


def test_fold_metrics_match_training_metrics(artifacts):
    f = dd.fold_metrics(artifacts["metrics"])
    assert len(f) == 12 and set(f["fold"]) == {1, 2, 3}
    for _, row in f.iterrows():
        src = artifacts["metrics"]["forward"]["per_fold"][str(row["fold"])][row["model"]]
        assert row["mae_h"] == src["mae_h"] and row["skill_vs_carry_forward"] == src["skill_vs_carry_forward"]


def test_key_findings_are_data_driven(artifacts):
    folds = dd.fold_metrics(artifacts["metrics"])
    text = " ".join(dd.key_findings(artifacts["mc"], folds))
    for fold, g in folds.groupby("fold"):
        worse = (g.set_index("model").loc[dd.LEARNED_MODELS, "skill_vs_carry_forward"] < 0).any()
        assert (f"Fold {fold}:" in text) == bool(worse)
    assert "Fold 1:" in text  # the known fold-1 underperformance is reported
    assert "No decisive winner" in text
    assert "definitively" not in text.lower() and "best model" not in text.lower()


def test_held_out_view_matches_saved_example(artifacts):
    for model in dd.LEARNED_MODELS:
        v = dd.held_out_view("SHP-0209", model, artifacts["fwd"], artifacts["shap"])
        saved = json.loads((tr.REPORTS_DIR / f"heldout_example_SHP-0209_{model}.json").read_text(encoding="utf-8"))
        r, check = saved["result"], saved["verification"]
        assert v.fold == check["fold"] == 3
        assert v.predicted_downstream_h == pytest.approx(r["predicted_downstream_delay_h"], abs=1e-9)
        assert v.predicted_final_h == pytest.approx(r["predicted_final_delay_h"], abs=1e-9)
        assert v.arrival_delay_h == pytest.approx(r["arrival_delay_h"], abs=1e-6)
        assert v.actual_downstream_h == pytest.approx(check["actual_downstream_delay_h"], abs=1e-9)
        assert v.prediction_error_h == pytest.approx(v.predicted_final_h - check["actual_final_delay_h"], abs=1e-9)
        assert v.shap_base_value_h == pytest.approx(r["shap_base_value_h"], abs=1e-9)
        for g, val in r["group_contributions_h"].items():
            assert v.groups.set_index("group").loc[g, "contribution_h"] == pytest.approx(val, abs=1e-9)
        assert v.additivity_error_h < 1e-6


def test_every_held_out_view_is_consistent(artifacts):
    for sid in artifacts["fwd"]["shipment_id"]:
        for model in dd.LEARNED_MODELS:
            v = dd.held_out_view(sid, model, artifacts["fwd"], artifacts["shap"])
            assert v.predicted_final_h == pytest.approx(v.arrival_delay_h + v.predicted_downstream_h, abs=1e-9)
            assert np.isfinite([v.predicted_downstream_h, v.prediction_error_h, v.shap_base_value_h]).all()


def test_held_out_view_rejects_inconsistent_or_unknown(artifacts):
    bad = artifacts["shap"].copy()
    mask = (bad["shipment_id"] == "SHP-0209") & (bad["model"] == "gbr") & (bad["level"] == "feature")
    bad.loc[bad.index[mask][0], "contribution_h"] += 1.0
    with pytest.raises(dd.ArtifactError, match="reconcile"):
        dd.held_out_view("SHP-0209", "gbr", artifacts["fwd"], bad)
    with pytest.raises(dd.ArtifactError):
        dd.held_out_view("SHP-0001", "gbr", artifacts["fwd"], artifacts["shap"])  # 2024: not held out
    with pytest.raises(ValueError):
        dd.held_out_view("SHP-0209", "carry_forward", artifacts["fwd"], artifacts["shap"])


def test_delay_status_and_formatting():
    assert dd.delay_status(24.0)[0] is False and dd.delay_status(24.01)[0] is True
    assert dd.delay_status(-5)[0] is False
    assert dd.fmt_hours(3.14159) == "+3.1 h" and dd.fmt_hours(-2.0) == "-2.0 h" and dd.fmt_hours(None) == "n/a"
    assert dd.fmt_local("2026-01-15T12:00:00+00:00", "Europe/Amsterdam") == "2026-01-15 13:00 (UTC+01:00)"
    assert dd.fmt_local("2026-07-15T12:00:00+00:00", "Europe/Amsterdam") == "2026-07-15 14:00 (UTC+02:00)"
    assert dd.fmt_local(pd.Timestamp("2025-09-06T20:49:30Z"), "Asia/Singapore") == "2025-09-07 04:49 (UTC+08:00)"
    assert dd.group_frame({g: 1.0 for g in ex.GROUPS})["group"].tolist() == list(ex.GROUPS)


def test_group_breakdown_reconciles_with_existing_shap_groups(artifacts):
    from src import predict as pr

    # Held-out explanations: net per group equals the saved group rows; parts add up to the net.
    for sid in ["SHP-0209", "SHP-0150", "SHP-0250"]:
        for model in dd.LEARNED_MODELS:
            v = dd.held_out_view(sid, model, artifacts["fwd"], artifacts["shap"])
            b = dd.group_breakdown(v.features).set_index("group")
            assert list(b.index) == list(ex.GROUPS)
            assert np.allclose(b["contribution_h"], v.groups.set_index("group")["contribution_h"], atol=1e-9)
            assert np.allclose(b["positive_h"] + b["negative_h"], b["contribution_h"], atol=1e-12)
            assert (b["positive_h"] >= 0).all() and (b["negative_h"] <= 0).all()
            assert b["n_features"].sum() == len(v.features)
    # Scenario predictions: the same holds against PredictionResult.group_contributions_h.
    predictor = pr.ShipmentPredictor("ridge")
    row = artifacts["df"].iloc[40]
    r = predictor.predict(pr.checkpoint_record(row))
    feats = dd.feature_frame(r.feature_contributions_h, predictor.explainer.feature_groups)
    b = dd.group_breakdown(feats).set_index("group")
    for g, val in r.group_contributions_h.items():
        assert b.loc[g, "contribution_h"] == pytest.approx(val, abs=1e-9)
    assert feats["contribution_h"].abs().is_monotonic_decreasing


@pytest.mark.parametrize("today", [date(2026, 11, 11), date(2026, 3, 14), date(2026, 10, 12), date(2027, 2, 28)])
def test_default_scenario_times_are_future_and_valid(today):
    from src import predict as pr

    arrival, delivery = dd.default_scenario_times(today)
    assert arrival.date() >= today + pd.Timedelta(days=14)
    assert arrival.weekday() == 0 and arrival.time() == time(21, 30)
    assert delivery.weekday() == 4 and delivery.time() == time(11, 0) and delivery > arrival
    for dest in dg.DEFAULT_CONFIG["destinations"].values():  # valid local times in every destination zone
        pr.local_to_utc(arrival, dest["tz"])
        pr.local_to_utc(delivery, dest["tz"])
    assert "Demonstration inputs" in dd.SCENARIO_DATE_NOTE


def test_stage_chart_frames_use_source_counts(artifacts):
    from src import stage_analysis as sa

    tables = sa.run_all(artifacts["df"])
    frames = dd.stage_chart_frames(tables)
    by_dest = tables["stage_delays_by_destination"]
    for _, row in frames["destination"].iterrows():
        assert row["n"] == by_dest.loc[row["destination"], ("n", "")]
        assert row["label"] == f"{row['destination']}\n(n={row['n']})"
    for key, table in [("arrival", "port_delay_by_arrival_status"), ("congestion", "port_delay_by_congestion")]:
        f, src = frames[key], tables[table]
        assert list(f["n"]) == list(src["count"]) and np.allclose(f["mean"], src["mean"])
        assert all(lbl.endswith(f"(n={n})") for lbl, n in zip(f["label"], f["n"]))
    docs = frames["documentation"].set_index("band")
    src = tables["customs_delay_by_documentation"]
    assert docs.loc["Complete", "n"] == src.loc[True, "count"] and docs.loc["Incomplete", "n"] == src.loc[False, "count"]
    assert list(frames["documentation"]["band"]) == ["Complete", "Incomplete"]
    assert frames["congestion"]["n"].sum() == len(artifacts["df"])


def test_truck_wait_note_describes_table_population(artifacts):
    from src import stage_analysis as sa

    note = dd.truck_wait_note(artifacts["df"])
    table = sa.truck_wait_by_congestion(artifacts["df"])
    assert note["n_missed"] == int(table["count"].sum())  # the table covers exactly the missed-window shipments
    assert note["n_missed"] + note["n_made"] == len(artifacts["df"])
    zero_from_table = float((table["count"] * table["zero_wait_share"]).sum())
    assert note["n_missed_zero_wait"] == pytest.approx(zero_from_table, abs=0.5)  # shares are rounded to 3 dp
    assert note["window_h"] == dg.DEFAULT_CONFIG["inland"]["slot_window_h"]


# Streamlit app (headless) -------------------------------------------------------

def _chart_text(at: AppTest) -> str:
    """Vega-Lite specs (titles, tooltips) plus serialized chart data (category labels) as one string."""
    charts = at.get("vega_lite_chart")
    assert charts, "no charts rendered"
    return " ".join(e.proto.spec + e.proto.SerializeToString().decode("utf-8", "ignore") for e in charts)

def _hashes() -> dict[str, str]:
    paths = [*ROOT.glob("data/*"), *ROOT.glob("models/*"), *ROOT.glob("reports/**/*")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def _no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio[0].set_value(page).run(timeout=TIMEOUT)
    _no_errors(at)
    return at


def _widget(items, label):
    return next(w for w in items if w.label == label)


@pytest.fixture
def no_training(monkeypatch):
    """Any attempt to fit a model or regenerate data during an app run fails loudly."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    def boom(*_a, **_k):
        raise AssertionError("dashboard attempted to fit a model or regenerate data")

    for target, name in [(Pipeline, "fit"), (GradientBoostingRegressor, "fit"), (Ridge, "fit"),
                         (dg, "generate_shipments"), (dg, "save_dataset"), (tr, "run"), (tr, "fit_final_models"),
                         (ex, "forward_explanations"), (ex, "run_forward_report")]:
        monkeypatch.setattr(target, name, boom)


def test_no_training_guard_is_effective_inside_apptest(no_training, tmp_path):
    """Control: an app that does fit a model must fail under the guard."""
    script = tmp_path / "fits.py"
    script.write_text("from sklearn.linear_model import Ridge\nRidge().fit([[0.0], [1.0]], [0.0, 1.0])\n",
                      encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=TIMEOUT).run()
    assert at.exception and "attempted to fit" in at.exception[0].value


def test_app_pages_render_without_training_or_writes(no_training):
    before = _hashes()
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    _no_errors(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Synthetic shipments"] == "250" and metrics["Routes"] == "6"
    assert metrics["Forward validation folds"] == "3"
    for page in ["Predict ETA", "Explain prediction", "Stage analysis", "Overview"]:
        _goto(at, page)
    _goto(at, "Predict ETA")
    at.button[0].click().run(timeout=TIMEOUT)
    _no_errors(at)
    assert "Scenario forecast using the full-data model. This is not an out-of-sample evaluation result." in \
        [i.value for i in at.info]
    assert _hashes() == before  # nothing in data/, models/ or reports/ changed


def test_predict_page_values_follow_definitions(no_training):
    at = _goto(AppTest.from_file(APP, default_timeout=TIMEOUT).run(), "Predict ETA")
    _widget(at.number_input, "Departure delay D1 (h)").set_value(6.0)
    _widget(at.number_input, "Ocean transit delay D2 (h)").set_value(-2.5)
    _widget(at.checkbox, "Congestion reading unavailable").check()
    _widget(at.radio, "Model").set_value("Ridge")
    at.button[0].click().run(timeout=TIMEOUT)
    _no_errors(at)
    result = at.session_state["scenario_result"]["result"]
    m = {x.label: x.value for x in at.metric}
    assert result.model_name == "ridge"
    assert result.arrival_delay_h == pytest.approx(3.5, abs=1e-6)  # A = D1 + D2
    assert m["Observed arrival delay A"] == "+3.5 h"
    assert m["Predicted downstream delay"] == dd.fmt_hours(result.predicted_downstream_delay_h)
    assert m["Predicted final delay (A + predicted downstream)"] == dd.fmt_hours(result.arrival_delay_h + result.predicted_downstream_delay_h)
    assert result.model_inputs["dest_congestion_index"] is None
    assert np.isfinite(result.predicted_final_delay_h)


def test_predict_page_reports_invalid_local_time(no_training):
    at = _goto(AppTest.from_file(APP, default_timeout=TIMEOUT).run(), "Predict ETA")
    _widget(at.date_input, "Actual port arrival date").set_value(date(2025, 3, 30))
    _widget(at.time_input, "Actual port arrival time").set_value(time(2, 30))  # DST gap in Rotterdam
    at.button[0].click().run(timeout=TIMEOUT)
    _no_errors(at)
    assert any("does not exist" in e.value for e in at.error)
    assert "scenario_result" not in at.session_state


def test_explain_page_shows_held_out_values(no_training, artifacts):
    at = _goto(AppTest.from_file(APP, default_timeout=TIMEOUT).run(), "Explain prediction")
    assert _widget(at.selectbox, "Held-out shipment (2025 forward test windows)").value == "SHP-0209"
    v = dd.held_out_view("SHP-0209", "gbr", artifacts["fwd"], artifacts["shap"])
    m = {x.label: x.value for x in at.metric}
    assert m["Predicted downstream delay"] == dd.fmt_hours(v.predicted_downstream_h)
    assert m["Prediction error (predicted - actual)"] == dd.fmt_hours(v.prediction_error_h)
    assert any("fold-3 evaluation model" in s.value and "not in its training data" in s.value for s in at.success)
    _widget(at.selectbox, "Held-out shipment (2025 forward test windows)").set_value("SHP-0150").run(timeout=TIMEOUT)
    _widget(at.radio, "Model").set_value("Ridge").run(timeout=TIMEOUT)
    _no_errors(at)
    v2 = dd.held_out_view("SHP-0150", "ridge", artifacts["fwd"], artifacts["shap"])
    m = {x.label: x.value for x in at.metric}
    assert m["Predicted downstream delay"] == dd.fmt_hours(v2.predicted_downstream_h)


def test_stage_page_labels_synthetic(no_training, artifacts):
    at = _goto(AppTest.from_file(APP, default_timeout=TIMEOUT).run(), "Stage analysis")
    assert any("synthetic simulation" in w.value and "not empirical" in w.value for w in at.warning)
    note = dd.truck_wait_note(artifacts["df"])
    headings = [m.value for m in at.markdown]
    assert any("Truck waiting after missing the booked 6-hour pickup window" in h and f"({note['n_missed']} shipments)" in h
               for h in headings)
    assert not any("(shipments that missed their booked window)" in h for h in headings)
    captions = " ".join(c.value for c in at.caption)
    assert f"Covers the {note['n_missed']} shipments" in captions
    assert f"({note['n_missed_zero_wait']} of {note['n_missed']})" in captions
    specs = _chart_text(at)
    assert "(n=" in specs and "Shipments" in specs  # counts in labels and tooltips


def test_shap_chart_title_and_tooltips(no_training):
    at = _goto(AppTest.from_file(APP, default_timeout=TIMEOUT).run(), "Explain prediction")
    specs = _chart_text(at)
    assert "Net SHAP contribution per driver group (h)" in specs
    assert "Net contribution to predicted downstream delay" not in specs
    assert "Sum of positive feature contributions (h)" in specs and "Sum of negative feature contributions (h)" in specs
    assert "Raises delay" in specs and "Lowers delay" in specs  # legend domain`n    assert "Net effect" not in specs
    assert len(at.dataframe) >= 1  # feature table rendered (below the chart)


def _dataframes_inside_columns(node, inside=False) -> list[bool]:
    """For each dataframe in the tree, whether it sits inside a st.columns column."""
    found = []
    children = getattr(node, "children", None)
    for child in (children.values() if isinstance(children, dict) else []):
        kind = type(child).__name__
        if kind == "Dataframe":
            found.append(inside)
        found += _dataframes_inside_columns(child, inside or kind == "Column")
    return found


def test_feature_tables_use_full_width_below_chart(no_training):
    at = _goto(AppTest.from_file(APP, default_timeout=TIMEOUT).run(), "Explain prediction")
    placement = _dataframes_inside_columns(at.main)
    assert placement and not any(placement)  # not squeezed into a side column
    _goto(at, "Predict ETA")
    at.button[0].click().run(timeout=TIMEOUT)
    _no_errors(at)
    placement = _dataframes_inside_columns(at.main)
    assert placement and not any(placement)


def test_predict_defaults_are_future_demo_inputs(no_training):
    at = _goto(AppTest.from_file(APP, default_timeout=TIMEOUT).run(), "Predict ETA")
    arrival = _widget(at.date_input, "Actual port arrival date").value
    delivery = _widget(at.date_input, "Scheduled delivery date").value
    assert arrival > date.today() and delivery > arrival
    assert any("Demonstration inputs" in c.value for c in at.caption)
