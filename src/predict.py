"""Single-shipment ETA prediction with explanations at the port-arrival checkpoint.

Inputs are checkpoint-available fields only. Post-checkpoint outcomes (port
exit, customs release, delivery), targets and latent generator variables are
rejected. ``arrival_delay_h`` and ``projected_nonworking_wait_h`` are derived
here from timestamps with :mod:`src.delay_propagation`; if supplied, they must
agree with the derived values.

Output::

    observed arrival delay A            (fact)
    predicted downstream delay Y_hat    (model)
    predicted final delay = A + Y_hat
    predicted ETA = scheduled_delivery + predicted final delay (UTC and destination-local)
    SHAP contributions of Y_hat (per transformed feature and per driver group)

The saved model artifacts are final full-data fits; predictions for shipments
in the frozen dataset are therefore in-sample, and results say so in
``model_scope``. :func:`explain_held_out` explains a 2025 shipment with the
forward-fold evaluation model that did not see it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src import data_generator as dg
from src import delay_propagation as dp
from src import explainability as ex
from src import feature_engineering as fe
from src import preprocessing as pp
from src import train as tr

MODEL_PATHS = {name: tr.MODELS_DIR / f"{name}_model.joblib" for name in tr.LEARNED_MODELS}

REQUIRED_FIELDS = [
    "destination_port", "cargo_type", "documentation_complete", "dest_congestion_index",
    "dest_weather_severity", "planned_port_dwell_h", "inland_distance_km", "planned_inland_h",
    "scheduled_port_arrival", "actual_port_arrival", "scheduled_delivery",
]
NULLABLE_FIELDS = {"dest_congestion_index"}
DERIVED_FIELDS = {"arrival_delay_h", "projected_nonworking_wait_h"}
TIMESTAMP_FIELDS = ["scheduled_port_arrival", "actual_port_arrival", "scheduled_delivery"]
FORBIDDEN_INPUTS = (set(dg.POST_CHECKPOINT_COLUMNS) | set(dg.TARGET_COLUMNS) | set(dg.LATENT_COLUMNS)
                    | {"validation_window"})
ALLOWED_INPUTS = set(dg.CHECKPOINT_AVAILABLE_COLUMNS) | {"shipment_id", "route_id"}
DERIVED_TOL_H = 1e-6

SCOPE_NEW_RECORD = "final full-data model"
SCOPE_IN_SAMPLE = "final full-data model; IN-SAMPLE for this dataset shipment"


class InvalidShipmentError(ValueError):
    """Raised when a prediction input is missing, invalid or not checkpoint-safe."""


@dataclass
class PredictionResult:
    shipment_id: str | None
    model_name: str
    destination_port: str
    destination_timezone: str
    arrival_delay_h: float
    predicted_downstream_delay_h: float
    predicted_final_delay_h: float
    scheduled_delivery_utc: str
    predicted_eta_utc: str
    predicted_eta_local: str
    shap_base_value_h: float
    feature_contributions_h: dict[str, float] = field(default_factory=dict)
    group_contributions_h: dict[str, float] = field(default_factory=dict)
    model_scope: str = SCOPE_NEW_RECORD
    explanation: str = ""
    caveat: str = ex.CAVEAT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _parse_ts(value: Any, name: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise InvalidShipmentError(f"{name} is not an ISO-8601 timestamp: {value!r}") from exc
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise InvalidShipmentError(f"{name} must be a timestamp, got {type(value).__name__}")
    try:
        return dp.ensure_utc(value, name)
    except ValueError as exc:
        raise InvalidShipmentError(str(exc)) from exc


def _finite(value: Any, name: str, lo: float | None = None, hi: float | None = None,
            positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidShipmentError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise InvalidShipmentError(f"{name} must be finite, got {value!r}")
    if positive and v <= 0:
        raise InvalidShipmentError(f"{name} must be > 0, got {v}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise InvalidShipmentError(f"{name} must be within [{lo}, {hi}], got {v}")
    return v


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA


def prepare_input(record: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a checkpoint record; return the model input row ``X`` and context values."""
    keys = set(record)
    forbidden = sorted(keys & FORBIDDEN_INPUTS)
    if forbidden:
        raise InvalidShipmentError(f"post-checkpoint, target or latent fields are not allowed: {forbidden}")
    unknown = sorted(keys - ALLOWED_INPUTS - DERIVED_FIELDS)
    if unknown:
        raise InvalidShipmentError(f"unknown fields: {unknown}")
    missing = [f for f in REQUIRED_FIELDS if f not in record or
               (_is_missing(record[f]) and f not in NULLABLE_FIELDS)]
    if missing:
        raise InvalidShipmentError(f"missing required fields: {missing}")

    dest = record["destination_port"]
    destinations = dg.DEFAULT_CONFIG["destinations"]
    if dest not in destinations:
        raise InvalidShipmentError(f"unknown destination_port {dest!r}; expected one of {sorted(destinations)}")
    tz = destinations[dest]["tz"]
    if "dest_timezone" in record and record["dest_timezone"] != tz:
        raise InvalidShipmentError(f"dest_timezone {record['dest_timezone']!r} does not match {dest} ({tz})")
    cargo = record["cargo_type"]
    if cargo not in dg.DEFAULT_CONFIG["cargo"]["mix"]:
        raise InvalidShipmentError(f"unknown cargo_type {cargo!r}")
    docs = record["documentation_complete"]
    if not isinstance(docs, (bool, np.bool_)):
        raise InvalidShipmentError("documentation_complete must be a boolean")

    congestion = record["dest_congestion_index"]
    congestion = np.nan if _is_missing(congestion) else _finite(congestion, "dest_congestion_index", 0.0, 1.0)
    weather = _finite(record["dest_weather_severity"], "dest_weather_severity", 0.0, 1.0)
    dwell = _finite(record["planned_port_dwell_h"], "planned_port_dwell_h", positive=True)
    inland_km = _finite(record["inland_distance_km"], "inland_distance_km", lo=0.0)
    inland_h = _finite(record["planned_inland_h"], "planned_inland_h", positive=True)

    ts = {name: _parse_ts(record[name], name) for name in TIMESTAMP_FIELDS}
    if ts["scheduled_delivery"] <= ts["scheduled_port_arrival"]:
        raise InvalidShipmentError("scheduled_delivery must be after scheduled_port_arrival")

    arrival_delay = dp.hours_between(ts["scheduled_port_arrival"], ts["actual_port_arrival"])
    cal = dg.DEFAULT_CONFIG["calendar"]
    projected_wait = dp.projected_customs_wait_h(
        ts["actual_port_arrival"], dwell, tz, frozenset(cal["non_working_weekdays"]), cal["workday_start_hour"])
    for name, derived in (("arrival_delay_h", arrival_delay), ("projected_nonworking_wait_h", projected_wait)):
        if name in record and not _is_missing(record[name]):
            given = _finite(record[name], name)
            if abs(given - derived) > DERIVED_TOL_H:
                raise InvalidShipmentError(f"{name}={given} disagrees with timestamps ({derived})")

    row = {
        "arrival_delay_h": arrival_delay, "dest_congestion_index": congestion, "dest_weather_severity": weather,
        "planned_port_dwell_h": dwell, "projected_nonworking_wait_h": projected_wait,
        "inland_distance_km": inland_km, "planned_inland_h": inland_h, "documentation_complete": bool(docs),
        "destination_port": dest, "cargo_type": cargo,
    }
    X = fe.select_features(pd.DataFrame([row]))
    context = {"shipment_id": record.get("shipment_id"), "destination_port": dest, "tz": tz,
               "arrival_delay_h": arrival_delay, "scheduled_delivery": ts["scheduled_delivery"]}
    return X, context


def checkpoint_record(row: pd.Series) -> dict[str, Any]:
    """Build a prediction input from a dataset row using checkpoint fields only."""
    return {"shipment_id": row["shipment_id"], **{f: row[f] for f in REQUIRED_FIELDS}}


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signed_hours(h: float) -> str:
    return f"{abs(h):.1f} h {'late' if h > 0 else 'early' if h < 0 else 'on schedule'}"


def explanation_text(result: PredictionResult, top_k: int = 3) -> str:
    """Plain-language summary built only from the computed result."""
    groups = sorted(result.group_contributions_h.items(), key=lambda kv: abs(kv[1]), reverse=True)
    drivers = ", ".join(f"{ex.GROUPS[g].lower()} {v:+.1f} h" for g, v in groups[:top_k])
    a = result.arrival_delay_h
    y = result.predicted_downstream_delay_h
    base = result.shap_base_value_h
    contrib_total = y - base
    return (
        f"1) Observed arrival delay A = {a:+.1f} h: the shipment reached {result.destination_port} "
        f"{_signed_hours(a)} (a measured fact, not a prediction). "
        f"2) Predicted downstream delay Y_hat = {y:+.1f} h: the {result.model_name} model "
        f"({result.model_scope}) prediction of how much "
        f"the remaining port, customs and inland stages will {'add to' if y >= 0 else 'recover from'} the delay "
        f"against plan. "
        f"3) SHAP baseline = {base:+.1f} h: the explainer's expected value of Y_hat over the model's training "
        f"data; it is a reference point, not a forecast for this shipment. "
        f"4) Feature contributions relative to that baseline total {contrib_total:+.1f} h "
        f"(baseline {base:+.1f} h + contributions {contrib_total:+.1f} h = Y_hat {y:+.1f} h); "
        f"largest driver groups: {drivers}. "
        f"5) Predicted final delay = A + Y_hat = ({a:+.1f} h) + ({y:+.1f} h) = {result.predicted_final_delay_h:+.1f} h; "
        f"predicted delivery {result.predicted_eta_local[:16].replace('T', ' ')} local time "
        f"({result.destination_timezone}). {ex.CAVEAT}"
    )


class ShipmentPredictor:
    """Loads a saved final model artifact and explains single-shipment predictions."""

    def __init__(self, model_name: str = "gbr", model_path: str | Path | None = None,
                 background_path: str | Path = pp.DATA_PATH):
        if model_name not in tr.LEARNED_MODELS:
            raise ValueError(f"model_name must be one of {tr.LEARNED_MODELS}")
        artifact = tr.load_model(model_path or MODEL_PATHS[model_name])
        if artifact.get("model_name") != model_name:
            raise ValueError(f"artifact holds {artifact.get('model_name')!r}, not {model_name!r}")
        if artifact.get("features") != fe.FEATURES:
            raise ValueError("artifact feature list differs from the canonical allowlist")
        if artifact.get("dataset_sha256") != _sha256(background_path):
            raise ValueError("background dataset hash differs from the dataset the model was trained on")
        self.model_name = model_name
        self.artifact = artifact
        self.pipeline = artifact["pipeline"]
        training_df = pp.load_dataset(background_path)
        self.training_ids = frozenset(training_df["shipment_id"])
        X_bg = fe.select_features(training_df) if model_name == "ridge" else None
        self.explainer = ex.PipelineExplainer(self.pipeline, model_name, X_bg)

    def predict(self, record: Mapping[str, Any]) -> PredictionResult:
        X, ctx = prepare_input(record)
        scope = SCOPE_IN_SAMPLE if ctx["shipment_id"] in self.training_ids else SCOPE_NEW_RECORD
        return _build_result(self.model_name, scope, ctx, self.explainer.explain(X))


def _build_result(model_name: str, scope: str, ctx: dict[str, Any], exp: ex.Explanation) -> PredictionResult:
    y_hat = float(exp.predictions[0])
    final = tr.predicted_final_delay(np.array([ctx["arrival_delay_h"]]), np.array([y_hat]))[0]
    eta_utc = dp.add_hours(ctx["scheduled_delivery"], float(final))
    values = [ctx["arrival_delay_h"], y_hat, float(final), exp.base_value]
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("non-finite prediction")
    result = PredictionResult(
        shipment_id=ctx["shipment_id"],
        model_name=model_name,
        destination_port=ctx["destination_port"],
        destination_timezone=ctx["tz"],
        arrival_delay_h=float(ctx["arrival_delay_h"]),
        predicted_downstream_delay_h=y_hat,
        predicted_final_delay_h=float(final),
        scheduled_delivery_utc=ctx["scheduled_delivery"].isoformat(),
        predicted_eta_utc=eta_utc.isoformat(),
        predicted_eta_local=dp.to_local(eta_utc, ctx["tz"]).isoformat(timespec="minutes"),
        shap_base_value_h=exp.base_value,
        feature_contributions_h={k: float(v) for k, v in exp.contributions.iloc[0].items()},
        group_contributions_h={k: float(v) for k, v in exp.grouped().iloc[0].items()},
        model_scope=scope,
    )
    result.explanation = explanation_text(result)
    return result


# ---------------------------------------------------------------------------
# Held-out explanation of a dataset shipment (forward-fold evaluation model)
# ---------------------------------------------------------------------------

def explain_held_out(shipment_id: str, model_name: str = "gbr",
                     df: pd.DataFrame | None = None) -> tuple[PredictionResult, dict[str, Any]]:
    """Explain a 2025 shipment with the forward-fold evaluation model that never saw it.

    The fold pipeline is rebuilt in memory exactly as in :func:`src.train.forward_evaluation`
    (same training rows, same random_state); nothing is saved. Returns the result
    and a verification record (fold, training-set exclusion, additivity, actual outcome).
    """
    if model_name not in tr.LEARNED_MODELS:
        raise ValueError(f"model_name must be one of {tr.LEARNED_MODELS}")
    df = pp.load_dataset() if df is None else df
    matches = np.flatnonzero((df["shipment_id"] == shipment_id).to_numpy())
    if matches.size != 1:
        raise ValueError(f"shipment {shipment_id!r} not found exactly once")
    pos = int(matches[0])
    fold = next((f for f in pp.forward_folds(df) if pos in set(f.test_idx)), None)
    if fold is None:
        raise ValueError(f"{shipment_id} is not in a forward test window; it has no held-out model")
    if pos in set(fold.train_idx):
        raise RuntimeError("held-out shipment found in its fold's training rows")

    X, y = fe.select_features(df), fe.get_target(df)
    X_train = X.iloc[fold.train_idx]
    pipe = tr.PIPELINE_FACTORIES[model_name]().fit(X_train, y.iloc[fold.train_idx])
    explainer = ex.PipelineExplainer(pipe, model_name, X_train if model_name == "ridge" else None)
    row = df.iloc[pos]
    X_one, ctx = prepare_input(checkpoint_record(row))
    exp = explainer.explain(X_one)
    scope = f"fold-{fold.fold} evaluation model, HELD-OUT: this shipment is not in its training rows"
    result = _build_result(model_name, scope, ctx, exp)
    train = df.iloc[fold.train_idx]
    verification = {
        "fold": fold.fold,
        "test_window_utc": [fold.test_start.isoformat(), fold.test_end.isoformat()],
        "n_train": int(len(fold.train_idx)),
        "shipment_in_training_rows": bool(shipment_id in set(train["shipment_id"])),
        "latest_training_delivery_utc": train["actual_delivery"].max().isoformat(),
        "shipment_port_arrival_utc": row["actual_port_arrival"].isoformat(),
        "shap_additivity_error_h": float(exp.additivity_error()[0]),
        "group_reconciliation_error_h": float(exp.group_reconciliation_error()[0]),
        "actual_arrival_delay_h": float(row["arrival_delay_h"]),
        "actual_downstream_delay_h": float(row["downstream_delay_h"]),
        "actual_final_delay_h": float(row["final_delivery_delay_h"]),
        "actual_stage_delays_h": {"D3_port": float(row["port_processing_delay_h"]),
                                  "D4_customs": float(row["customs_delay_h"]),
                                  "D5_inland": float(row["inland_delivery_delay_h"])},
        "downstream_prediction_error_h": float(result.predicted_downstream_delay_h - row["downstream_delay_h"]),
    }
    return result, verification


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict and explain one shipment at the port-arrival checkpoint.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="JSON file with checkpoint fields of a new shipment")
    src.add_argument("--shipment-id", help="shipment from the frozen dataset (checkpoint fields only are used)")
    parser.add_argument("--model", choices=tr.LEARNED_MODELS, default="gbr")
    parser.add_argument("--held-out", action="store_true",
                        help="with --shipment-id: explain using the forward-fold model that did not see it")
    parser.add_argument("--out", help="optional path for the JSON result")
    args = parser.parse_args(argv)

    verification = None
    if args.held_out:
        if not args.shipment_id:
            parser.error("--held-out requires --shipment-id")
        result, verification = explain_held_out(args.shipment_id, args.model)
    else:
        if args.input:
            record = json.loads(Path(args.input).read_text(encoding="utf-8"))
        else:
            df = pp.load_dataset()
            rows = df[df["shipment_id"] == args.shipment_id]
            if rows.empty:
                parser.error(f"unknown shipment {args.shipment_id}")
            record = checkpoint_record(rows.iloc[0])
        result = ShipmentPredictor(args.model).predict(record)
    payload = {"result": result.to_dict(), "verification": verification}
    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(result.explanation)
    if verification:
        print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
