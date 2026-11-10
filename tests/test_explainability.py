"""Tests for src.explainability and src.stage_analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import explainability as ex
from src import feature_engineering as fe
from src import preprocessing as pp
from src import stage_analysis as sa
from src import train as tr


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pp.load_dataset()


@pytest.fixture(scope="module")
def X(df) -> pd.DataFrame:
    return fe.select_features(df)


@pytest.fixture(scope="module")
def explainers(X) -> dict[str, ex.PipelineExplainer]:
    out = {}
    for name in tr.LEARNED_MODELS:
        pipe = tr.load_model(tr.MODELS_DIR / f"{name}_model.joblib")["pipeline"]
        out[name] = ex.PipelineExplainer(pipe, name, X if name == "ridge" else None)
    return out


@pytest.fixture(scope="module")
def explanations(explainers, X) -> dict[str, ex.Explanation]:
    return {name: e.explain(X) for name, e in explainers.items()}


# 1-2. Additivity and group reconciliation -----------------------------------

@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_shap_additivity_final_models(explanations, explainers, X, name):
    e = explanations[name]
    assert e.additivity_error().max() < 1e-6
    assert np.allclose(e.predictions, explainers[name].pipeline.predict(X))


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_shap_additivity_fold_model_on_held_out_rows(df, X, name):
    fold = pp.forward_folds(df)[2]
    y = fe.get_target(df)
    pipe = tr.PIPELINE_FACTORIES[name]().fit(X.iloc[fold.train_idx], y.iloc[fold.train_idx])
    explainer = ex.PipelineExplainer(pipe, name, X.iloc[fold.train_idx] if name == "ridge" else None)
    e = explainer.explain(X.iloc[fold.test_idx])
    assert e.additivity_error().max() < 1e-6


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_group_sums_reconcile(explanations, name):
    e = explanations[name]
    grouped = e.grouped()
    assert list(grouped.columns) == list(ex.GROUPS)
    assert e.group_reconciliation_error().max() < 1e-9
    assert np.allclose(e.base_value + grouped.sum(axis=1), e.predictions, atol=1e-6)


def test_ridge_shap_matches_linear_formula(explainers, explanations, X):
    e, explainer = explanations["ridge"], explainers["ridge"]
    Z = explainer.preprocessor.transform(X)
    coef = explainer.model.coef_
    assert np.allclose(e.contributions.to_numpy(), coef * (Z - Z.mean(axis=0)), atol=1e-8)
    assert e.base_value == pytest.approx(explainer.model.intercept_ + coef @ Z.mean(axis=0))


# 3-4. Feature names and missing congestion ----------------------------------

def test_feature_names_map_through_fitted_preprocessors(explainers):
    for name, explainer in explainers.items():
        names = list(explainer.preprocessor.get_feature_names_out())
        assert explainer.feature_names == names
        assert set(explainer.feature_groups) == set(names)
        assert set(explainer.feature_groups.values()) == set(ex.GROUPS)
        groups = explainer.feature_groups
        assert groups["num__arrival_delay_h"] == "upstream_propagation"
        assert groups["num__missingindicator_dest_congestion_index"] == "port_conditions"
        assert groups["num__projected_nonworking_wait_h"] == "port_conditions"
        assert groups["num__documentation_complete"] == "customs_risk"
        assert groups["num__planned_inland_h"] == "inland_delivery"
        assert all(groups[n] == "destination_context" for n in names if n.startswith("cat__destination_port_"))
        assert all(groups[n] == "customs_risk" for n in names if n.startswith("cat__cargo_type_"))
    ridge = explainers["ridge"].feature_groups
    assert len(ridge) == 21 and len(explainers["gbr"].feature_groups) == 19
    assert ridge["num__congestion_x_arrival_delay_pos"] == "upstream_propagation"
    assert ridge["num__arrival_delay_pos_h"] == "upstream_propagation"


def test_group_for_sources_rules():
    assert ex.group_for_sources(("dest_congestion_index", "arrival_delay_h")) == "upstream_propagation"
    assert ex.group_for_sources(("cargo_type",)) == "customs_risk"
    with pytest.raises(ValueError):
        ex.group_for_sources(("cargo_type", "inland_distance_km"))


@pytest.mark.parametrize("name", tr.LEARNED_MODELS)
def test_missing_congestion_explained(explanations, X, name):
    missing = X["dest_congestion_index"].isna()
    assert missing.sum() > 0
    e = explanations[name]
    assert np.isfinite(e.contributions.loc[missing].to_numpy()).all()
    indicator = e.transformed_values["num__missingindicator_dest_congestion_index"]
    assert indicator.loc[missing].nunique() == 1 and indicator.loc[~missing].nunique() == 1
    assert indicator.loc[missing].iloc[0] > indicator.loc[~missing].iloc[0]  # raw 1 vs 0 (scaled for Ridge)
    assert e.contributions["num__missingindicator_dest_congestion_index"].loc[missing].abs().sum() >= 0


def test_global_importance(explanations):
    for e in explanations.values():
        per_feature, per_group = ex.global_importance(e)
        assert per_feature["mean_abs_shap_h"].is_monotonic_decreasing
        assert set(per_group.index) == set(ex.GROUPS)
        assert (per_group[ex.GROUP_IMPORTANCE] >= 0).all()
        assert np.allclose(per_group[ex.GROUP_MEAN].sum(), (e.predictions - e.base_value).mean())


def test_unsupported_model_and_missing_background(explainers):
    pipe = explainers["ridge"].pipeline
    with pytest.raises(ValueError):
        ex.PipelineExplainer(pipe, "ridge", None)
    with pytest.raises(ValueError):
        ex.PipelineExplainer(pipe, "lasso", None)


# Forward out-of-sample explanations -----------------------------------------

@pytest.fixture(scope="module")
def forward_items(df):
    return ex.forward_explanations(df)


def test_forward_explanations_cover_each_2025_shipment_once(df, forward_items):
    ids_2025 = set(df.loc[df["actual_port_arrival"].dt.year == 2025, "shipment_id"])
    for name in tr.LEARNED_MODELS:
        ids = [s for it in forward_items if it.model_name == name for s in it.shipment_ids]
        assert len(ids) == 126 and len(set(ids)) == 126 and set(ids) == ids_2025
        assert sorted(it.fold for it in forward_items if it.model_name == name) == [1, 2, 3]


def test_forward_explanations_additive_and_match_evaluation_predictions(df, forward_items):
    fwd = tr.forward_evaluation(df, models=tr.LEARNED_MODELS).predictions.set_index("shipment_id")
    for it in forward_items:
        e = it.explanation
        assert e.additivity_error().max() < 1e-6
        assert e.group_reconciliation_error().max() < 1e-9
        expected = fwd.loc[it.shipment_ids, f"pred_downstream_{it.model_name}"].to_numpy()
        assert np.allclose(e.predictions, expected, atol=1e-9)  # exact rebuild of the evaluation models


def test_forward_importance_tables(forward_items):
    feat, group = ex.forward_importance_report(forward_items)
    assert set(feat["scope"]) == {"pooled_forward", "fold_1", "fold_2", "fold_3"}
    for name in tr.LEARNED_MODELS:
        g = group[(group.model == name) & (group.scope == "pooled_forward")]
        assert set(g["group"]) == set(ex.GROUPS) and (g["n_rows"] == 126).all()
        assert g[ex.GROUP_SHARE].sum() == pytest.approx(1.0)
        pooled, groups = ex.pooled_contributions(forward_items, name)
        assert pooled.shape[0] == 126
        assert set(groups.values()) == set(ex.GROUPS)
        f = feat[(feat.model == name) & (feat.scope == "pooled_forward")]
        assert f["mean_abs_shap_h"].is_monotonic_decreasing
        assert list(f["rank"]) == list(range(1, len(f) + 1))
        for k in (1, 2, 3):
            assert (group[(group.model == name) & (group.scope == f"fold_{k}")]["n_rows"] == 42).all()


def test_importance_tables_group_uses_row_level_sums():
    contrib = pd.DataFrame({"a": [1.0, -1.0], "b": [-1.0, 1.0]})
    groups = {"a": "port_conditions", "b": "port_conditions"}
    _, per_group = ex.importance_tables(contrib, groups)
    assert per_group.loc["port_conditions", ex.GROUP_IMPORTANCE] == 0.0  # contributions cancel within group


def test_group_importance_is_net_and_bounded_by_feature_sum(forward_items):
    """Group value = mean(|sum SHAP in group|) <= sum of the features' mean(|SHAP|) (triangle inequality)."""
    feat, group = ex.forward_importance_report(forward_items)
    for (model, scope), g in group.groupby(["model", "scope"]):
        f = feat[(feat.model == model) & (feat.scope == scope)]
        feature_sum = f.groupby("group")[ex.FEATURE_IMPORTANCE].sum()
        for _, row in g.iterrows():
            assert row[ex.GROUP_IMPORTANCE] <= feature_sum.get(row["group"], 0.0) + 1e-9
    assert ex.GROUP_IMPORTANCE == "mean_abs_net_group_contribution_h"
    assert "mean_abs_shap_h" not in group.columns


def test_comparison_table_labels_importance_definitions(forward_items, df):
    feat, group = ex.forward_importance_report(forward_items)
    in_feat, in_group = ex.in_sample_importance_report(df)
    comp = ex.compare_in_sample_forward(in_feat, feat, in_group, group)
    assert set(comp["level"]) == {"feature", "group"}
    assert (comp.loc[comp.level == "group", "importance_definition"] == ex.GROUP_IMPORTANCE_DEFINITION).all()
    assert (comp.loc[comp.level == "feature", "importance_definition"] == ex.FEATURE_IMPORTANCE_DEFINITION).all()
    assert (in_group["scope"] == "in_sample_full_data").all() and (in_group["n_rows"] == 250).all()


# 12. Historical analysis never changes model inputs ------------------------

def test_historical_analysis_does_not_change_inputs(df, X, tmp_path):
    df_before, X_before = df.copy(deep=True), X.copy(deep=True)
    allowlist_before = list(fe.FEATURES)
    tables = sa.run_all(df)
    paths = sa.save_figures(df, tmp_path)
    pd.testing.assert_frame_equal(df, df_before)
    pd.testing.assert_frame_equal(fe.select_features(df), X_before)
    assert fe.FEATURES == allowlist_before
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert len(tables) == 7


def test_stage_analysis_tables(df):
    by_dest = sa.stage_delays_by_destination(df)
    assert by_dest[("n", "")].sum() == len(df) and len(by_dest) == 6
    by_cong = sa.port_delay_by_congestion(df)
    assert by_cong["count"].sum() == len(df) and "missing" in by_cong.index
    by_arr = sa.port_delay_by_arrival_status(df)
    assert by_arr["count"].sum() == len(df)
    docs = sa.customs_delay_by_documentation(df)
    assert docs["count"].sum() == len(df)
    truck = sa.truck_wait_by_congestion(df)
    assert truck["count"].sum() == int(df["booked_window_missed"].sum())
    means, corr = sa.upstream_vs_downstream(df)
    assert means["n"].sum() == len(df)
    assert set(corr.index) == {"port_processing_delay_h", "customs_delay_h", "inland_delivery_delay_h", "downstream_delay_h"}
