"""Tests for src.train (baselines, forward evaluation, metrics, ablation, artifacts)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import data_generator as dg
from src import delay_propagation as dp
from src import feature_engineering as fe
from src import preprocessing as pp
from src import train as tr


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pp.load_dataset()


@pytest.fixture(scope="module")
def fwd(df) -> tr.ForwardResult:
    return tr.forward_evaluation(df)


@pytest.fixture(scope="module")
def folds(df):
    return pp.forward_folds(df)


# 1-3. Baselines, final delay, ETA ---------------------------------------------

def test_baselines(df):
    a = df["arrival_delay_h"].to_numpy()
    so = tr.baseline_downstream(df, "schedule_only")
    cf = tr.baseline_downstream(df, "carry_forward")
    assert np.array_equal(so, -a)
    assert np.array_equal(cf, np.zeros_like(a))
    assert np.allclose(tr.predicted_final_delay(a, so), 0.0)
    assert np.array_equal(tr.predicted_final_delay(a, cf), a)
    with pytest.raises(ValueError):
        tr.baseline_downstream(df, "oracle")


@pytest.mark.parametrize("model", tr.MODEL_NAMES)
def test_final_delay_is_arrival_plus_predicted_downstream(fwd, model):
    p = fwd.predictions
    assert np.allclose(p[f"pred_final_{model}"], p["arrival_delay_h"] + p[f"pred_downstream_{model}"], atol=1e-12)


@pytest.mark.parametrize("model", tr.MODEL_NAMES)
def test_eta_consistent(fwd, model):
    p = fwd.predictions
    hours = (p[f"pred_eta_{model}"] - p["scheduled_delivery"]).dt.total_seconds() / 3600
    assert np.allclose(hours, p[f"pred_final_{model}"], atol=1e-6)
    assert str(p[f"pred_eta_{model}"].dtype) == str(p["scheduled_delivery"].dtype)
    if model == "schedule_only":
        assert np.allclose((p["pred_eta_schedule_only"] - p["scheduled_delivery"]).dt.total_seconds(), 0, atol=1e-3)
    # The ETA error equals the final-delay error (A is observed exactly).
    eta_err = (p[f"pred_eta_{model}"] - p["actual_delivery"]).dt.total_seconds() / 3600
    assert np.allclose(eta_err, p[f"pred_final_{model}"] - p["actual_final_delay_h"], atol=1e-6)


# 4-8, 15. Forward evaluation integrity ----------------------------------------

def test_every_2025_shipment_predicted_exactly_once(df, fwd):
    p = fwd.predictions
    ids_2025 = set(df.loc[df["actual_port_arrival"].dt.year == 2025, "shipment_id"])
    assert p["shipment_id"].is_unique
    assert set(p["shipment_id"]) == ids_2025
    assert len(p) == 126
    assert p.groupby("fold").size().to_dict() == {1: 42, 2: 42, 3: 42}


def test_no_training_row_violates_embargo(df, folds, fwd):
    for f in folds:
        start = pd.Timestamp(f.test_start)
        train = df.iloc[f.train_idx]
        assert (train["actual_delivery"] < start).all()
        assert (train["actual_port_arrival"] < start).all()
        assert (fwd.predictions.loc[fwd.predictions["fold"] == f.fold, "n_train"] == len(f.train_idx)).all()


def test_no_train_test_overlap(df, folds, fwd):
    for f in folds:
        train_ids = set(df.iloc[f.train_idx]["shipment_id"])
        test_ids = set(fwd.predictions.loc[fwd.predictions["fold"] == f.fold, "shipment_id"])
        assert not train_ids & test_ids
        assert test_ids == set(df.iloc[f.test_idx]["shipment_id"])


def test_each_fold_fits_fresh_preprocessing_on_its_own_training_rows(df, folds, fwd):
    X = fe.select_features(df)
    pipes = fwd.fold_pipelines
    assert set(pipes) == {(f.fold, m) for f in folds for m in tr.LEARNED_MODELS}
    assert len({id(p) for p in pipes.values()}) == len(pipes)
    for f in folds:
        train = X.iloc[f.train_idx]
        medians = train[fe.NUMERIC_FEATURES].median().to_numpy()
        gbr_imp = pipes[(f.fold, "gbr")].named_steps["preprocess"].named_transformers_["num"]
        ridge_imp = pipes[(f.fold, "ridge")].named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
        assert np.allclose(gbr_imp.statistics_, medians)
        assert np.allclose(ridge_imp.statistics_, medians)
    s1 = pipes[(1, "gbr")].named_steps["preprocess"].named_transformers_["num"].statistics_
    s3 = pipes[(3, "gbr")].named_steps["preprocess"].named_transformers_["num"].statistics_
    assert not np.allclose(s1, s3)


@pytest.mark.parametrize("model", tr.MODEL_NAMES)
def test_predictions_finite(fwd, model):
    p = fwd.predictions
    assert np.isfinite(p[f"pred_downstream_{model}"]).all()
    assert np.isfinite(p[f"pred_final_{model}"]).all()
    assert p[f"pred_eta_{model}"].notna().all()


def test_predictions_come_from_held_out_fold_models(df, folds, fwd):
    X, y = fe.select_features(df), fe.get_target(df)
    p = fwd.predictions
    for f in folds:
        test_mask = p["fold"] == f.fold
        for model in tr.LEARNED_MODELS:
            fresh = tr.PIPELINE_FACTORIES[model]().fit(X.iloc[f.train_idx], y.iloc[f.train_idx])
            assert np.allclose(fresh.predict(X.iloc[f.test_idx]), p.loc[test_mask, f"pred_downstream_{model}"])
    # The full-data final models give different predictions on the same rows.
    final = tr.fit_final_models(df)
    idx_2025 = np.concatenate([f.test_idx for f in folds])
    for model in tr.LEARNED_MODELS:
        assert not np.allclose(final[model].predict(X.iloc[idx_2025]), p[f"pred_downstream_{model}"])


# 9-11. Metrics ------------------------------------------------------------------

def test_regression_metrics_known_example():
    actual = np.array([0.0, 10.0, 30.0, -5.0, 50.0])
    pred = np.array([2.0, 10.0, 20.0, -5.0, 20.0])  # abs errors 2, 0, 10, 0, 30
    m = tr.regression_metrics(actual, pred)
    assert m["n"] == 5
    assert m["mae_h"] == pytest.approx(8.4)
    assert m["rmse_h"] == pytest.approx(np.sqrt((4 + 0 + 100 + 0 + 900) / 5))
    assert m["median_ae_h"] == pytest.approx(2.0)
    assert m["p90_ae_h"] == pytest.approx(22.0)  # linear interpolation between 10 and 30
    assert m["within_12h_pct"] == pytest.approx(80.0)
    assert m["within_24h_pct"] == pytest.approx(80.0)


def test_flag_metrics_known_example():
    actual = np.array([30.0, 30.0, 10.0, 10.0, 50.0, 0.0])
    pred = np.array([25.0, 20.0, 30.0, 5.0, 60.0, 1.0])
    m = tr.flag_metrics(actual, pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    # Ranking: positives scores {25, 20, 60}, negatives {30, 5, 1}: 7 of 9 pairs ordered correctly.
    assert m["roc_auc_ranking"] == pytest.approx(7 / 9)
    assert 0 < m["pr_auc_ranking"] <= 1


def test_flag_metrics_degenerate_cases():
    m = tr.flag_metrics(np.array([0.0, 30.0]), np.array([0.0, 0.0]))
    assert m["precision"] is None and m["recall"] == 0.0
    m = tr.flag_metrics(np.array([0.0, 1.0]), np.array([0.0, 30.0]))
    assert m["roc_auc_ranking"] is None  # only one actual class


def test_skill_score():
    assert tr.skill_score(20.0, 25.0) == pytest.approx(0.2)
    assert tr.skill_score(30.0, 25.0) == pytest.approx(-0.2)
    assert tr.skill_score(25.0, 25.0) == 0.0
    with pytest.raises(ValueError):
        tr.skill_score(1.0, 0.0)


def test_pooled_skill_matches_definition(fwd):
    s = tr.summarise_forward(fwd.predictions, n_boot=50)
    cf = s["pooled"]["carry_forward"]["mae_h"]
    for model in tr.MODEL_NAMES:
        assert s["pooled"][model]["skill_vs_carry_forward"] == pytest.approx(1 - s["pooled"][model]["mae_h"] / cf)
    assert s["pooled"]["carry_forward"]["skill_vs_carry_forward"] == 0.0
    a = fwd.predictions["actual_downstream_delay_h"].abs().mean()
    assert cf == pytest.approx(a)  # carry-forward error is |Y|


def test_bootstrap_reproducible_and_fold_preserving():
    rng = np.random.default_rng(0)
    err, ref = rng.exponential(10, 120), rng.exponential(12, 120)
    groups = np.repeat([1, 2, 3], 40)
    a = tr.bootstrap_ci(err, groups, n_boot=300, seed=11)
    b = tr.bootstrap_ci(err, groups, n_boot=300, seed=11)
    c = tr.bootstrap_ci(err, groups, n_boot=300, seed=12)
    assert a == b and a != c
    assert a["ci_low"] <= a["point"] <= a["ci_high"]
    paired = tr.bootstrap_ci(err, groups, ref_abs_err=ref, n_boot=300, seed=11)
    assert paired["point"] == pytest.approx(ref.mean() - err.mean())
    # Errors constant within equal-sized groups: within-group resampling cannot change the mean.
    const = np.repeat([5.0, 10.0, 20.0], 40)
    ci = tr.bootstrap_ci(const, groups, n_boot=200, seed=1)
    assert ci["ci_low"] == pytest.approx(ci["ci_high"]) == pytest.approx(const.mean())


# 12-13. CV and ablation ---------------------------------------------------------

def test_cv_evaluation_uses_route_stratified_splits(df):
    splits = pp.repeated_route_cv(df)
    totals = df["route_id"].value_counts()
    for _, test in splits:
        counts = df.iloc[test]["route_id"].value_counts()
        for route, n in totals.items():
            assert counts.get(route, 0) in (n // 5, n // 5 + 1)
    cv = tr.cv_evaluation(df, splits[:5])
    assert len(cv) == 5 * len(tr.MODEL_NAMES)
    assert (cv["n_test"] == 50).all() and (cv["n_train"] == 200).all()
    assert np.isfinite(cv["mae_h"]).all()
    summary = tr.summarise_cv(cv)
    assert summary["carry_forward"]["paired_diff_vs_carry_forward_mean_h"] == 0.0
    assert summary["gbr"]["n_splits"] == 5


def test_carrier_ablation_does_not_change_allowlist(df):
    before = (list(fe.FEATURES), list(dg.MODEL_FEATURES), list(fe.CATEGORICAL_FEATURES))
    abl = tr.carrier_ablation(df, pp.repeated_route_cv(df)[:2])
    after = (list(fe.FEATURES), list(dg.MODEL_FEATURES), list(fe.CATEGORICAL_FEATURES))
    assert before == after
    assert "carrier" not in fe.FEATURES
    assert len(abl) == 2 and np.allclose(abl["delta_mae_h"], abl["mae_with_carrier_h"] - abl["mae_primary_h"])
    names = tr.make_gbr_pipeline(extra_categorical=["carrier"]).fit(
        fe.select_features(df, fe.FEATURES + ["carrier"], allow_ablation=True), fe.get_target(df)
    ).named_steps["preprocess"].get_feature_names_out()
    assert any(n.startswith("cat__carrier_") for n in names)
    s = tr.summarise_ablation(abl)
    assert s["n_splits"] == 2 and "Diagnostic only" in s["interpretation"]


# 14. Serialization + artifact export ------------------------------------------

def test_model_serialization_round_trip(df, tmp_path):
    X = fe.select_features(df)
    final = tr.fit_final_models(df)
    for name, pipe in final.items():
        path = tmp_path / f"{name}_model.joblib"
        tr.save_model(pipe, name, path, dataset_sha256="abc")
        loaded = tr.load_model(path)
        assert loaded["artifact_type"] == "final_full_data_fit"
        assert loaded["features"] == fe.FEATURES
        assert np.array_equal(loaded["pipeline"].predict(X), pipe.predict(X))


def test_run_writes_all_artifacts(df, tmp_path, monkeypatch):
    original = pp.repeated_route_cv
    monkeypatch.setattr(pp, "repeated_route_cv", lambda d, **k: original(d, **k)[:2])
    metrics = tr.run(models_dir=tmp_path / "models", reports_dir=tmp_path / "reports", n_boot=50)
    for f in ("forward_predictions.csv", "model_comparison.csv", "cv_metrics.csv", "training_metrics.json"):
        assert (tmp_path / "reports" / f).exists()
    for m in ("gbr", "ridge"):
        assert (tmp_path / "models" / f"{m}_model.joblib").exists()
    saved = json.loads((tmp_path / "reports" / "training_metrics.json").read_text(encoding="utf-8"))
    assert saved["dataset"]["sha256"] == metrics["dataset"]["sha256"]
    assert set(saved["forward"]["pooled"]) == set(tr.MODEL_NAMES)
    comparison = pd.read_csv(tmp_path / "reports" / "model_comparison.csv")
    assert list(comparison["model"]) == tr.MODEL_NAMES
