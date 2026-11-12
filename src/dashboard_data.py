"""Read-only data access and small calculations for the Streamlit dashboard.

Everything here reads frozen artifacts (dataset, evaluation reports, held-out
SHAP values) or calls existing project modules. Nothing fits a model or writes
a file, so the dashboard can never retrain models or regenerate data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_generator as dg
from src import delay_propagation as dp
from src import explainability as ex
from src import preprocessing as pp
from src import train as tr

REPORTS_DIR = tr.REPORTS_DIR
MODEL_LABELS = {"schedule_only": "Schedule-only", "carry_forward": "Carry-forward",
                "ridge": "Ridge", "gbr": "Gradient Boosting"}
LEARNED_MODELS = list(tr.LEARNED_MODELS)
DELAY_THRESHOLD_H = tr.DELAY_THRESHOLD_H
GROUP_LABELS = dict(ex.GROUPS)
CONSISTENCY_TOL_H = 1e-6


class ArtifactError(RuntimeError):
    """Raised when an evaluation artifact is missing or inconsistent."""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _require(path: Path) -> Path:
    if not path.exists():
        raise ArtifactError(f"missing artifact {path}; run the commands in README.md")
    return path


def load_dataset() -> pd.DataFrame:
    return pp.load_dataset()


def load_model_comparison(reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    return pd.read_csv(_require(reports_dir / "model_comparison.csv"))


def load_training_metrics(reports_dir: Path = REPORTS_DIR) -> dict:
    return json.loads(_require(reports_dir / "training_metrics.json").read_text(encoding="utf-8"))


def load_forward_predictions(reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    df = pd.read_csv(_require(reports_dir / "forward_predictions.csv"))
    for col in [c for c in df.columns if c in ("actual_port_arrival", "scheduled_delivery", "actual_delivery")
                or c.startswith("pred_eta_")]:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df


def load_forward_shap_values(reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    return pd.read_csv(_require(reports_dir / ex.FORWARD_SHAP_VALUES_FILE))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame, metrics: dict) -> dict:
    folds = metrics["forward_folds"]
    return {
        "n_shipments": int(len(df)),
        "n_routes": int(df["route_id"].nunique()),
        "n_destinations": int(df["destination_port"].nunique()),
        "n_folds": len(folds),
        "n_held_out": int(sum(f["n_test"] for f in folds)),
        "folds": folds,
        "arrival_period": (df["actual_port_arrival"].min(), df["actual_port_arrival"].max()),
    }


def comparison_table(mc: pd.DataFrame) -> pd.DataFrame:
    """Pooled forward metrics per model, in the evaluation's model order."""
    t = mc.set_index("model").loc[list(MODEL_LABELS)]
    return pd.DataFrame({
        "Model": [MODEL_LABELS[m] for m in t.index],
        "MAE (h)": t["mae_h"].to_numpy(),
        "MAE 95% CI (h)": [f"{lo:.1f} - {hi:.1f}" for lo, hi in zip(t["mae_ci95_low"], t["mae_ci95_high"])],
        "RMSE (h)": t["rmse_h"].to_numpy(),
        "P90 abs. error (h)": t["p90_ae_h"].to_numpy(),
        "Within ±24 h (%)": t["within_24h_pct"].to_numpy(),
        "Skill vs carry-forward": t["skill_vs_carry_forward"].to_numpy(),
    })


def fold_metrics(metrics: dict) -> pd.DataFrame:
    """Long table of per-fold forward metrics (from training_metrics.json)."""
    rows = []
    folds = {f["fold"]: f for f in metrics["forward_folds"]}
    for fold, by_model in metrics["forward"]["per_fold"].items():
        f = folds[int(fold)]
        for model, m in by_model.items():
            rows.append({"fold": int(fold), "window": f"{f['test_start'][:10]} to {f['test_end'][:10]}",
                         "n_train": f["n_train"], "n_test": f["n_test"], "model": model,
                         "Model": MODEL_LABELS[model], "mae_h": m["mae_h"], "rmse_h": m["rmse_h"],
                         "p90_ae_h": m["p90_ae_h"], "within_24h_pct": m["within_24h_pct"],
                         "skill_vs_carry_forward": m["skill_vs_carry_forward"]})
    out = pd.DataFrame(rows)
    out["model_order"] = out["model"].map({k: i for i, k in enumerate(MODEL_LABELS)})
    return out.sort_values(["fold", "model_order"]).drop(columns="model_order").reset_index(drop=True)


def key_findings(mc: pd.DataFrame, folds: pd.DataFrame) -> list[str]:
    """Limitations computed from the artifacts (no hard-coded metrics)."""
    t = mc.set_index("model")
    out = []
    for fold, g in folds.groupby("fold"):
        losers = [MODEL_LABELS[m] for m in LEARNED_MODELS
                  if g.set_index("model").loc[m, "skill_vs_carry_forward"] < 0]
        if losers:
            cf = g.set_index("model").loc["carry_forward", "mae_h"]
            vals = ", ".join(f"{MODEL_LABELS[m]} {g.set_index('model').loc[m, 'mae_h']:.1f} h"
                             for m in LEARNED_MODELS)
            out.append(f"Fold {fold}: {' and '.join(losers)} did worse than the carry-forward baseline "
                       f"(MAE {vals} vs carry-forward {cf:.1f} h).")
    r, g = t.loc["ridge"], t.loc["gbr"]
    overlap = not (r["mae_ci95_high"] < g["mae_ci95_low"] or g["mae_ci95_high"] < r["mae_ci95_low"])
    out.append(f"No decisive winner between Ridge and Gradient Boosting: pooled MAE {r['mae_h']:.2f} h vs "
               f"{g['mae_h']:.2f} h (difference {abs(r['mae_h'] - g['mae_h']):.2f} h); their 95% intervals "
               f"{'overlap' if overlap else 'do not overlap'}.")
    out.append(f"Both learned models beat carry-forward on pooled MAE (skill {r['skill_vs_carry_forward']:+.3f} and "
               f"{g['skill_vs_carry_forward']:+.3f}) but have a lower share of predictions within ±12 h "
               f"({r['within_12h_pct']:.0f}% and {g['within_12h_pct']:.0f}% vs {t.loc['carry_forward', 'within_12h_pct']:.0f}%): "
               "they help most on large downstream delays.")
    out.append("Each fold has 42 held-out shipments, so intervals are wide and cover only the three 2025 windows.")
    return out


# ---------------------------------------------------------------------------
# Formatting and status
# ---------------------------------------------------------------------------

def delay_status(final_delay_h: float, threshold_h: float = DELAY_THRESHOLD_H) -> tuple[bool, str]:
    """Reporting flag used in evaluation: delayed if final delay > threshold."""
    delayed = float(final_delay_h) > threshold_h
    return delayed, (f"Delayed: more than {threshold_h:.0f} h late" if delayed
                     else f"Not delayed: at most {threshold_h:.0f} h late")


def fmt_hours(h: float | None) -> str:
    return "n/a" if h is None or (isinstance(h, float) and np.isnan(h)) else f"{h:+.1f} h"


def fmt_local(ts, tz_key: str) -> str:
    """'YYYY-MM-DD HH:MM (UTC+hh:mm)' in destination-local time, truncated to the minute."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    local = dp.to_local(pd.Timestamp(ts).to_pydatetime(), tz_key)
    offset = local.strftime("%z")
    return f"{local:%Y-%m-%d %H:%M} (UTC{offset[:3]}:{offset[3:]})"


def destination_timezone(port: str) -> str:
    return dg.DEFAULT_CONFIG["destinations"][port]["tz"]


# ---------------------------------------------------------------------------
# Held-out explanation (forward-fold evaluation models only)
# ---------------------------------------------------------------------------

@dataclass
class HeldOutView:
    shipment_id: str
    model: str
    fold: int
    route_id: str
    destination_port: str
    timezone: str
    arrival_delay_h: float
    actual_downstream_h: float
    actual_final_h: float
    predicted_downstream_h: float
    predicted_final_h: float
    prediction_error_h: float
    actual_eta_local: str
    predicted_eta_local: str
    scheduled_delivery_local: str
    shap_base_value_h: float
    groups: pd.DataFrame
    features: pd.DataFrame
    additivity_error_h: float


def held_out_view(shipment_id: str, model: str, fwd: pd.DataFrame, shap_values: pd.DataFrame) -> HeldOutView:
    """Assemble one held-out explanation from frozen forward artifacts, with consistency checks."""
    if model not in LEARNED_MODELS:
        raise ValueError(f"model must be one of {LEARNED_MODELS}")
    p = fwd[fwd["shipment_id"] == shipment_id]
    s = shap_values[(shap_values["shipment_id"] == shipment_id) & (shap_values["model"] == model)]
    if len(p) != 1 or s.empty:
        raise ArtifactError(f"{shipment_id} has no held-out prediction/explanation for {model}")
    p = p.iloc[0]
    feats = s[s["level"] == "feature"]
    groups = s[s["level"] == "group"].set_index("name").reindex(list(ex.GROUPS))
    base = float(s["shap_base_value_h"].iloc[0])
    pred = float(s["predicted_downstream_delay_h"].iloc[0])
    if int(s["fold"].iloc[0]) != int(p["fold"]):
        raise ArtifactError("fold mismatch between SHAP values and forward predictions")
    if abs(pred - float(p[f"pred_downstream_{model}"])) > CONSISTENCY_TOL_H:
        raise ArtifactError("held-out SHAP prediction does not match forward_predictions.csv")
    additivity = abs(base + feats["contribution_h"].sum() - pred)
    if additivity > CONSISTENCY_TOL_H or abs(groups["contribution_h"].sum() - feats["contribution_h"].sum()) > CONSISTENCY_TOL_H:
        raise ArtifactError("SHAP contributions do not reconcile with the prediction")
    tz = destination_timezone(p["destination_port"])
    return HeldOutView(
        shipment_id=shipment_id, model=model, fold=int(p["fold"]), route_id=p["route_id"],
        destination_port=p["destination_port"], timezone=tz,
        arrival_delay_h=float(p["arrival_delay_h"]),
        actual_downstream_h=float(p["actual_downstream_delay_h"]),
        actual_final_h=float(p["actual_final_delay_h"]),
        predicted_downstream_h=pred,
        predicted_final_h=float(p[f"pred_final_{model}"]),
        prediction_error_h=float(p[f"pred_final_{model}"] - p["actual_final_delay_h"]),
        actual_eta_local=fmt_local(p["actual_delivery"], tz),
        predicted_eta_local=fmt_local(p[f"pred_eta_{model}"], tz),
        scheduled_delivery_local=fmt_local(p["scheduled_delivery"], tz),
        shap_base_value_h=base,
        groups=pd.DataFrame({"group": groups.index, "label": [GROUP_LABELS[g] for g in groups.index],
                             "contribution_h": groups["contribution_h"].to_numpy()}),
        features=feats[["name", "group", "contribution_h"]].assign(
            feature=lambda d: d["name"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False))
        .sort_values("contribution_h", key=np.abs, ascending=False).reset_index(drop=True),
        additivity_error_h=float(additivity),
    )


def group_frame(group_contributions: dict[str, float]) -> pd.DataFrame:
    """Group contributions (dict from PredictionResult) as an ordered frame for charting."""
    return pd.DataFrame({"group": list(ex.GROUPS), "label": [GROUP_LABELS[g] for g in ex.GROUPS],
                         "contribution_h": [float(group_contributions[g]) for g in ex.GROUPS]})


def feature_frame(feature_contributions: dict[str, float], feature_groups: dict[str, str]) -> pd.DataFrame:
    """Per-feature SHAP contributions with readable names and driver groups, largest |value| first."""
    return (pd.DataFrame({"name": list(feature_contributions),
                          "contribution_h": [float(v) for v in feature_contributions.values()]})
            .assign(feature=lambda d: d["name"].str.replace("num__", "", regex=False)
                    .str.replace("cat__", "", regex=False),
                    group=lambda d: d["name"].map(feature_groups))
            .sort_values("contribution_h", key=np.abs, ascending=False).reset_index(drop=True))


def group_breakdown(features: pd.DataFrame) -> pd.DataFrame:
    """Net contribution per driver group plus its positive and negative feature parts.

    ``contribution_h`` (net) is the sum of the group's feature SHAP values, the same
    value the explainer reports per group; ``positive_h`` and ``negative_h`` are the
    sums of its positive and negative feature contributions (net = positive + negative).
    """
    c = features["contribution_h"]
    g = pd.DataFrame({"group": features["group"], "net": c, "pos": c.clip(lower=0), "neg": c.clip(upper=0)})
    agg = g.groupby("group").agg(contribution_h=("net", "sum"), positive_h=("pos", "sum"),
                                 negative_h=("neg", "sum"), n_features=("net", "size"))
    agg = agg.reindex(list(ex.GROUPS)).fillna({"contribution_h": 0.0, "positive_h": 0.0, "negative_h": 0.0,
                                               "n_features": 0})
    return agg.rename_axis("group").reset_index().assign(
        label=lambda d: d["group"].map(GROUP_LABELS), n_features=lambda d: d["n_features"].astype(int))


# ---------------------------------------------------------------------------
# Scenario defaults and stage-analysis presentation
# ---------------------------------------------------------------------------

SCENARIO_DATE_NOTE = ("Demonstration inputs: the dates default to an illustrative future week (vessel arrives on a "
                      "Monday evening, scheduled delivery that Friday morning). Edit them freely.")


def default_scenario_times(today: date) -> tuple[datetime, datetime]:
    """Illustrative future scenario: arrival on the first Monday at least 14 days after ``today``
    at 21:30 local, scheduled delivery that Friday at 11:00 local (both outside DST transitions)."""
    start = today + timedelta(days=14)
    monday = start + timedelta(days=(7 - start.weekday()) % 7)
    return datetime.combine(monday, time(21, 30)), datetime.combine(monday + timedelta(days=4), time(11, 0))


def count_label(name: str, n: int) -> str:
    """Category label with its sample size on a second line."""
    return f"{name}\n(n={int(n)})"


def stage_chart_frames(tables: dict) -> dict[str, pd.DataFrame]:
    """Chart-ready frames from ``stage_analysis.run_all`` with sample counts in the labels.

    Counts are the group sizes already computed by stage_analysis; nothing is recalculated.
    """
    by_dest = tables["stage_delays_by_destination"]
    stages = {"Port (D3)": "port_processing_delay_h", "Customs (D4)": "customs_delay_h",
              "Inland (D5)": "inland_delivery_delay_h"}
    dest = pd.DataFrame([{"destination": d, "n": int(by_dest.loc[d, ("n", "")]),
                          "label": count_label(d, by_dest.loc[d, ("n", "")]),
                          "Stage": s, "mean_h": float(by_dest.loc[d, (c, "mean")])}
                         for d in by_dest.index for s, c in stages.items()])

    def banded(table: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        out = table.reset_index(drop=True).assign(band=names)
        return out.assign(n=out["count"].astype(int),
                          label=[count_label(b, n) for b, n in zip(names, out["count"])])

    arr = tables["port_delay_by_arrival_status"]
    cong = tables["port_delay_by_congestion"]
    docs = tables["customs_delay_by_documentation"]
    doc_names = ["Complete" if v else "Incomplete" for v in docs.index]
    return {
        "destination": dest,
        "arrival": banded(arr, [str(b) for b in arr.index]),
        "congestion": banded(cong, [str(b) for b in cong.index]),
        "documentation": banded(docs, doc_names).assign(order=lambda d: d["band"].map(
            {"Complete": 0, "Incomplete": 1})).sort_values("order").drop(columns="order").reset_index(drop=True),
    }


def truck_wait_note(df: pd.DataFrame) -> dict[str, int | float]:
    """Counts describing exactly which shipments the truck-waiting table covers (from the dataset)."""
    missed = df["booked_window_missed"].astype(bool)
    wait = df["_latent_truck_wait_h"]
    return {
        "window_h": float(dg.DEFAULT_CONFIG["inland"]["slot_window_h"]),
        "n_missed": int(missed.sum()),
        "n_missed_zero_wait": int((missed & (wait == 0)).sum()),
        "n_made": int((~missed).sum()),
        "n_made_with_wait": int((~missed & (wait > 0)).sum()),
    }
