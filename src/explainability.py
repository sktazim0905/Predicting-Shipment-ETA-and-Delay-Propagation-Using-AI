"""SHAP explanations of the predicted downstream delay ``Y_hat``.

What is explained
-----------------
Only the model output ``Y_hat`` (predicted downstream delay, hours). The
observed arrival delay ``A`` is a fact, reported separately; the predicted
final delay is ``A + Y_hat``. A SHAP contribution of ``arrival_delay_h`` is the
model's *downstream* response to arriving late, not ``A`` itself, so ``A`` is
never added twice.

Explainers
----------
- GBR: ``shap.TreeExplainer`` with the default path-dependent perturbation
  (no background data; base value = model's expected output over the training
  tree covers).
- Ridge: ``shap.LinearExplainer`` with an independent masker over the full
  transformed background (training rows). For a linear model this equals
  ``coef_j * (z_j - mean(z_j))``.

Both are computed on the *transformed* features of the fitted pipeline and
traced back to raw columns with :func:`feature_engineering.feature_sources`.
Positions are never hard-coded.

Interpretation caveats
----------------------
SHAP values describe how the fitted model's prediction depends on its inputs.
They are not physical or causal effects. Correlated inputs (e.g. planned inland
duration and inland distance, or arrival delay and its derived interaction
terms) can share or trade attribution, so individual feature values should be
read together with their group totals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src import feature_engineering as fe

GROUPS: dict[str, str] = {
    "upstream_propagation": "Upstream propagation",
    "port_conditions": "Port conditions",
    "customs_risk": "Customs risk",
    "inland_delivery": "Inland delivery",
    "destination_context": "Destination context",
}
SOURCE_GROUP: dict[str, str] = {
    "arrival_delay_h": "upstream_propagation",
    "dest_congestion_index": "port_conditions",
    "dest_weather_severity": "port_conditions",
    "planned_port_dwell_h": "port_conditions",
    "projected_nonworking_wait_h": "port_conditions",
    "cargo_type": "customs_risk",
    "documentation_complete": "customs_risk",
    "inland_distance_km": "inland_delivery",
    "planned_inland_h": "inland_delivery",
    "destination_port": "destination_context",
}
CAVEAT = ("Contributions describe the model's prediction, not physical causes; correlated inputs "
          "can share attribution.")
ADDITIVITY_TOL = 1e-6


def group_for_sources(sources: tuple[str, ...]) -> str:
    """Driver group of a transformed feature.

    Any feature derived from ``arrival_delay_h`` (including the Ridge
    congestion x positive-arrival interaction) belongs to upstream propagation.
    """
    if fe.ARRIVAL_COL in sources:
        return "upstream_propagation"
    groups = {SOURCE_GROUP[s] for s in sources}
    if len(groups) != 1:
        raise ValueError(f"ambiguous group for sources {sources}")
    return groups.pop()


def feature_groups(preprocessor) -> dict[str, str]:
    """Map each transformed feature name of a *fitted* preprocessor to its driver group."""
    return {name: group_for_sources(src) for name, src in fe.feature_sources(preprocessor).items()}


@dataclass
class Explanation:
    """SHAP explanation of ``Y_hat`` for one or more rows."""

    model_name: str
    base_value: float
    predictions: np.ndarray            # Y_hat, shape (n,)
    contributions: pd.DataFrame        # transformed feature -> contribution, shape (n, k)
    feature_groups: dict[str, str]     # transformed feature -> group key
    transformed_values: pd.DataFrame   # model inputs after preprocessing

    def grouped(self) -> pd.DataFrame:
        """Contributions summed by driver group (columns in GROUPS order)."""
        g = self.contributions.T.groupby(pd.Series(self.feature_groups)).sum().T
        return g.reindex(columns=list(GROUPS), fill_value=0.0)

    def additivity_error(self) -> np.ndarray:
        """``|base + sum(contributions) - Y_hat|`` per row."""
        return np.abs(self.base_value + self.contributions.sum(axis=1).to_numpy() - self.predictions)

    def group_reconciliation_error(self) -> np.ndarray:
        """``|sum(groups) - sum(features)|`` per row."""
        return np.abs(self.grouped().sum(axis=1).to_numpy() - self.contributions.sum(axis=1).to_numpy())


class PipelineExplainer:
    """SHAP explainer for a fitted Ridge or GBR pipeline from :mod:`src.train`."""

    def __init__(self, pipeline: Pipeline, model_name: str, X_background: pd.DataFrame | None = None):
        self.pipeline = pipeline
        self.model_name = model_name
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self.feature_groups = feature_groups(self.preprocessor)
        if model_name == "gbr":
            self._explainer = shap.TreeExplainer(self.model)
        elif model_name == "ridge":
            if X_background is None:
                raise ValueError("Ridge explanations need the training rows as background data")
            Z_bg = self.preprocessor.transform(X_background)
            masker = shap.maskers.Independent(Z_bg, max_samples=len(Z_bg))
            self._explainer = shap.LinearExplainer(self.model, masker)
        else:
            raise ValueError(f"unsupported model {model_name!r}")
        self.base_value = float(np.ravel(self._explainer.expected_value)[0])

    def explain(self, X: pd.DataFrame) -> Explanation:
        Z = self.preprocessor.transform(X)
        values = np.asarray(self._explainer.shap_values(Z), dtype=float)
        preds = self.model.predict(Z)
        exp = Explanation(
            model_name=self.model_name,
            base_value=self.base_value,
            predictions=np.asarray(preds, dtype=float),
            contributions=pd.DataFrame(values, columns=self.feature_names, index=X.index),
            feature_groups=self.feature_groups,
            transformed_values=pd.DataFrame(Z, columns=self.feature_names, index=X.index),
        )
        worst = float(exp.additivity_error().max()) if len(X) else 0.0
        if worst > ADDITIVITY_TOL:
            raise RuntimeError(f"SHAP additivity violated for {self.model_name}: max error {worst:.3g} h")
        return exp


#: Feature-level importance column: mean over rows of |SHAP_j|.
FEATURE_IMPORTANCE = "mean_abs_shap_h"
#: Group-level importance column: mean over rows of |sum of SHAP_j within the group|.
GROUP_IMPORTANCE = "mean_abs_net_group_contribution_h"
GROUP_MEAN = "mean_net_group_contribution_h"
GROUP_SHARE = "share_of_total_abs_net"
FEATURE_IMPORTANCE_DEFINITION = "mean(abs(SHAP_feature))"
GROUP_IMPORTANCE_DEFINITION = ("mean(abs(sum(feature SHAP values within group))); "
                               "not the sum of the features' mean |SHAP|")


def importance_tables(contributions: pd.DataFrame, groups: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature- and group-level importance (hours) with ranks.

    Feature level: ``mean(|SHAP_j|)`` and ``mean(SHAP_j)``.

    Group level: the *mean absolute net group contribution*
    ``mean(|sum_{j in group} SHAP_j|)``. Contributions of opposite sign inside
    a group cancel row by row, so this can be smaller than the sum of the
    group's per-feature ``mean(|SHAP_j|)``.
    """
    per_feature = pd.DataFrame({
        "group": pd.Series(groups).reindex(contributions.columns),
        FEATURE_IMPORTANCE: contributions.abs().mean(),
        "mean_shap_h": contributions.mean(),
    }).sort_values(FEATURE_IMPORTANCE, ascending=False)
    per_feature["rank"] = np.arange(1, len(per_feature) + 1)
    grouped = contributions.T.groupby(pd.Series(groups).reindex(contributions.columns)).sum().T
    grouped = grouped.reindex(columns=list(GROUPS), fill_value=0.0)
    per_group = pd.DataFrame({"label": pd.Series(GROUPS), GROUP_IMPORTANCE: grouped.abs().mean(),
                              GROUP_MEAN: grouped.mean()}).sort_values(GROUP_IMPORTANCE, ascending=False)
    per_group["rank"] = np.arange(1, len(per_group) + 1)
    per_group[GROUP_SHARE] = per_group[GROUP_IMPORTANCE] / per_group[GROUP_IMPORTANCE].sum()
    return per_feature, per_group


def global_importance(exp: Explanation) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean |SHAP| per transformed feature and per group (hours), sorted descending."""
    return importance_tables(exp.contributions, exp.feature_groups)


# ---------------------------------------------------------------------------
# Forward out-of-sample explanations
# ---------------------------------------------------------------------------

@dataclass
class ForwardExplanation:
    """Explanation of one fold model on that fold's held-out test rows."""

    fold: int
    model_name: str
    shipment_ids: list[str]
    explanation: Explanation


def forward_explanations(df: pd.DataFrame, models: tuple[str, ...] = ("ridge", "gbr")) -> list[ForwardExplanation]:
    """Rebuild each forward-fold evaluation model and explain its held-out test rows.

    The fold pipelines are rebuilt exactly as in :func:`src.train.forward_evaluation`
    (same factories, training rows and random_state). Nothing is saved.
    """
    from src import preprocessing as pp
    from src import train as tr

    X, y = fe.select_features(df), fe.get_target(df)
    out = []
    for fold in pp.forward_folds(df):
        X_train, y_train = X.iloc[fold.train_idx], y.iloc[fold.train_idx]
        for name in models:
            pipe = tr.PIPELINE_FACTORIES[name]().fit(X_train, y_train)
            explainer = PipelineExplainer(pipe, name, X_train if name == "ridge" else None)
            exp = explainer.explain(X.iloc[fold.test_idx])
            out.append(ForwardExplanation(fold.fold, name, df.iloc[fold.test_idx]["shipment_id"].tolist(), exp))
    return out


def pooled_contributions(items: list[ForwardExplanation], model_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Concatenate one model's held-out contributions across folds (absent columns -> 0)."""
    parts = [it.explanation.contributions for it in items if it.model_name == model_name]
    groups: dict[str, str] = {}
    for it in items:
        if it.model_name == model_name:
            groups.update(it.explanation.feature_groups)
    return pd.concat(parts).fillna(0.0), groups


def forward_importance_report(items: list[ForwardExplanation]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long tables of feature and group importance: pooled (126 rows) and per fold."""
    feat_rows, group_rows = [], []
    for name in sorted({it.model_name for it in items}):
        scopes = [("pooled_forward", *pooled_contributions(items, name))]
        scopes += [(f"fold_{it.fold}", it.explanation.contributions, it.explanation.feature_groups)
                   for it in items if it.model_name == name]
        for scope, contrib, groups in scopes:
            pf, pg = importance_tables(contrib, groups)
            feat_rows.append(pf.rename_axis("feature").reset_index().assign(model=name, scope=scope, n_rows=len(contrib)))
            group_rows.append(pg.rename_axis("group").reset_index().assign(model=name, scope=scope, n_rows=len(contrib)))
    return pd.concat(feat_rows)[FEATURE_COLUMNS], pd.concat(group_rows)[GROUP_COLUMNS]


FEATURE_COLUMNS = ["model", "scope", "n_rows", "rank", "feature", "group", FEATURE_IMPORTANCE, "mean_shap_h"]
GROUP_COLUMNS = ["model", "scope", "n_rows", "rank", "group", "label", GROUP_IMPORTANCE, GROUP_MEAN, GROUP_SHARE]


def in_sample_importance_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive importance of the saved full-data models on all rows (in-sample)."""
    from src import train as tr

    X = fe.select_features(df)
    feat_rows, group_rows = [], []
    for name in ("ridge", "gbr"):
        pipe = tr.load_model(tr.MODELS_DIR / f"{name}_model.joblib")["pipeline"]
        exp = PipelineExplainer(pipe, name, X if name == "ridge" else None).explain(X)
        pf, pg = importance_tables(exp.contributions, exp.feature_groups)
        feat_rows.append(pf.rename_axis("feature").reset_index().assign(model=name, scope="in_sample_full_data", n_rows=len(X)))
        group_rows.append(pg.rename_axis("group").reset_index().assign(model=name, scope="in_sample_full_data", n_rows=len(X)))
    return pd.concat(feat_rows)[FEATURE_COLUMNS], pd.concat(group_rows)[GROUP_COLUMNS]


def compare_in_sample_forward(in_feat: pd.DataFrame, fwd_feat: pd.DataFrame,
                              in_group: pd.DataFrame, fwd_group: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side ranks and importance (pooled forward vs in-sample) for features and groups.

    ``importance_h`` is ``mean(|SHAP|)`` for features and the mean absolute net
    group contribution for groups; the ``importance_definition`` column says which.
    """
    rows = []
    for level, a, b, key, col, definition in (
        ("feature", in_feat, fwd_feat, "feature", FEATURE_IMPORTANCE, FEATURE_IMPORTANCE_DEFINITION),
        ("group", in_group, fwd_group, "group", GROUP_IMPORTANCE, GROUP_IMPORTANCE_DEFINITION),
    ):
        b = b[b["scope"] == "pooled_forward"]
        m = a.merge(b, on=["model", key], suffixes=("_in_sample", "_forward"))
        m = m.rename(columns={key: "name", f"{col}_in_sample": "importance_h_in_sample",
                              f"{col}_forward": "importance_h_forward"})
        m = m.assign(level=level, importance_definition=definition)
        rows.append(m[["level", "model", "name", "rank_in_sample", "rank_forward",
                       "importance_h_in_sample", "importance_h_forward", "importance_definition"]])
    out = pd.concat(rows, ignore_index=True)
    out["rank_change"] = out["rank_forward"] - out["rank_in_sample"]
    out["importance_ratio_forward_to_in_sample"] = out["importance_h_forward"] / out["importance_h_in_sample"]
    return out


def save_forward_figure(fwd_feat: pd.DataFrame, fwd_group: pd.DataFrame, path, top_k: int = 10):
    """Grouped importance (pooled, with fold min-max) and top features per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["ridge", "gbr"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    pooled = fwd_group[fwd_group.scope == "pooled_forward"].pivot(index="label", columns="model", values=GROUP_IMPORTANCE)
    folds = fwd_group[fwd_group.scope != "pooled_forward"]
    order = pooled.mean(axis=1).sort_values().index
    ypos = np.arange(len(order))
    for k, m in enumerate(models):
        vals = pooled.loc[order, m].to_numpy()
        f = folds[folds.model == m].groupby("label")[GROUP_IMPORTANCE]
        lo, hi = f.min().reindex(order).to_numpy(), f.max().reindex(order).to_numpy()
        axes[0].barh(ypos + (k - 0.5) * 0.38, vals, height=0.38, label=m,
                     xerr=[vals - lo, hi - vals], capsize=3)
    axes[0].set_yticks(ypos, order)
    axes[0].set_xlabel("mean |net group contribution| to held-out Y_hat (h)")
    axes[0].set_title("Driver groups, 126 held-out shipments\n(bars: pooled; whiskers: fold min-max)")
    axes[0].legend()
    for ax, m in zip(axes[1:], models):
        top = fwd_feat[(fwd_feat.scope == "pooled_forward") & (fwd_feat.model == m)].nsmallest(top_k, "rank")[::-1]
        ax.barh(top["feature"].str.replace("num__", "").str.replace("cat__", ""), top[FEATURE_IMPORTANCE])
        ax.set_xlabel("mean |SHAP| per feature (h)")
        ax.set_title(f"{m.upper()}: top {top_k} features (held-out)")
    fig.suptitle("Forward out-of-sample SHAP importance - model attribution, not causal effects (synthetic data)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_in_sample_group_figure(in_group: pd.DataFrame, path):
    """Group importance of the full-data models, explicitly labelled as in-sample."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = in_group.pivot(index="label", columns="model", values=GROUP_IMPORTANCE)
    table = table.loc[table.mean(axis=1).sort_values().index, ["ridge", "gbr"]]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    table.plot.barh(ax=ax)
    ax.set_xlabel("mean |net group contribution| to Y_hat (h)")
    ax.set_ylabel("")
    ax.set_title("IN-SAMPLE (full-data models, all 250 shipments): driver groups\n"
                 "descriptive only - see forward_shap_importance.png for held-out results")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_forward_report(df: pd.DataFrame | None = None, reports_dir=None) -> dict:
    """Compute and save forward and in-sample SHAP importance outputs."""
    from pathlib import Path

    from src import preprocessing as pp
    from src import train as tr

    df = pp.load_dataset() if df is None else df
    reports_dir = Path(reports_dir or tr.REPORTS_DIR)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    items = forward_explanations(df)
    fwd_feat, fwd_group = forward_importance_report(items)
    in_feat, in_group = in_sample_importance_report(df)
    comparison = compare_in_sample_forward(in_feat, fwd_feat, in_group, fwd_group)
    fwd_feat.to_csv(reports_dir / "forward_shap_importance.csv", index=False)
    fwd_group.to_csv(reports_dir / "forward_shap_group_importance.csv", index=False)
    in_feat.to_csv(reports_dir / "insample_shap_importance.csv", index=False)
    in_group.to_csv(reports_dir / "insample_shap_group_importance.csv", index=False)
    comparison.to_csv(reports_dir / "shap_insample_vs_forward.csv", index=False)
    save_forward_figure(fwd_feat, fwd_group, reports_dir / "figures" / "forward_shap_importance.png")
    save_in_sample_group_figure(in_group, reports_dir / "figures" / "insample_shap_group_importance.png")
    return {"items": items, "forward_feature": fwd_feat, "forward_group": fwd_group,
            "in_sample_feature": in_feat, "in_sample_group": in_group, "comparison": comparison}


if __name__ == "__main__":
    result = run_forward_report()
    worst = max(float(it.explanation.additivity_error().max()) for it in result["items"])
    n = sum(len(it.shipment_ids) for it in result["items"] if it.model_name == "gbr")
    print(f"explained {n} held-out shipments per model; max additivity error {worst:.2e} h")
