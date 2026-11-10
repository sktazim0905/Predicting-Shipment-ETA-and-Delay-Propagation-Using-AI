"""Loading, validation, targets and leakage-safe splits for the frozen dataset.

Nothing here fits a statistic on the data: imputers, scalers and encoders live
in the sklearn pipelines of :mod:`src.feature_engineering` and are fitted per
training fold.

Splits
------
Primary: three chronological forward folds. Test rows are shipments whose
actual port arrival ``T1`` lies in the fold's half-open UTC window. Training
rows satisfy ``T1 < window_start`` **and** ``actual_delivery < window_start``
(label-availability embargo: a label is only usable once the shipment has
been delivered).

Secondary: repeated route-stratified 5-fold CV (5 repeats, random_state=42).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src import data_generator as dg
from src import delay_propagation as dp

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "shipments.csv"
IDENTITY_TOL_H = 1e-6

PLANNED_MILESTONES = ["scheduled_departure", "scheduled_port_arrival", "planned_port_exit",
                      "planned_customs_release", "scheduled_delivery"]
ACTUAL_MILESTONES = ["_latent_actual_departure", "actual_port_arrival", "actual_port_exit",
                     "actual_customs_release", "actual_delivery"]
STAGE_DELAY_COLUMNS = ["_latent_departure_delay_h", "_latent_ocean_transit_delay_h",
                       "port_processing_delay_h", "customs_delay_h", "inland_delivery_delay_h"]
BOOL_COLUMNS = ["documentation_complete", "has_chokepoint", "chokepoint_disruption", "customs_inspection",
                "booked_window_missed", "is_delayed_24h", "_latent_missed_berth_window"]
STRING_COLUMNS = ["shipment_id", "route_id", "origin_port", "destination_port", "dest_timezone",
                  "carrier", "cargo_type"]
#: Columns allowed to contain missing values (MCAR masks applied by the generator).
NULLABLE_COLUMNS = {"actual_departure", "departure_delay_h", "ocean_transit_delay_h", "dest_congestion_index"}


class DatasetValidationError(ValueError):
    """Raised when the dataset violates the expected schema or identities."""


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path = DATA_PATH, validate: bool = True) -> pd.DataFrame:
    """Load the frozen CSV with UTC timestamps and (optionally) validate it."""
    df = dg.load_shipments(path)
    if validate:
        validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Check columns, dtypes, missingness, timestamp ordering and delay identities."""
    missing = [c for c in dg.ALL_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"missing required columns: {missing}")
    if not df["shipment_id"].is_unique:
        raise DatasetValidationError("shipment_id is not unique")

    for col in dg.TIMESTAMP_COLUMNS:
        dtype = df[col].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != "UTC":
            raise DatasetValidationError(f"{col} must be a UTC datetime column, got {dtype}")
    for col in BOOL_COLUMNS:
        if not pd.api.types.is_bool_dtype(df[col]):
            raise DatasetValidationError(f"{col} must be boolean, got {df[col].dtype}")
    for col in STRING_COLUMNS:
        if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
            raise DatasetValidationError(f"{col} must be string-like, got {df[col].dtype}")
    numeric_cols = [c for c in dg.ALL_COLUMNS
                    if c not in dg.TIMESTAMP_COLUMNS + BOOL_COLUMNS + STRING_COLUMNS]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            raise DatasetValidationError(f"{col} must be numeric, got {df[col].dtype}")

    unexpected_na = [c for c in dg.ALL_COLUMNS if c not in NULLABLE_COLUMNS and df[c].isna().any()]
    if unexpected_na:
        raise DatasetValidationError(f"unexpected missing values in: {unexpected_na}")

    validate_timestamp_order(df)
    verify_targets(df)


def validate_timestamp_order(df: pd.DataFrame) -> None:
    """Planned and actual milestones must be strictly increasing for every shipment."""
    for name, cols in (("planned", PLANNED_MILESTONES), ("actual", ACTUAL_MILESTONES)):
        values = df[cols]
        for a, b in zip(cols[:-1], cols[1:]):
            bad = ~(values[b] > values[a])
            if bad.any():
                ids = df.loc[bad, "shipment_id"].tolist()[:5]
                raise DatasetValidationError(f"{name} milestones not increasing ({a} -> {b}) for {ids}")
    if (df["actual_customs_processing_start"] < df["actual_port_exit"]).any():
        raise DatasetValidationError("customs processing starts before port exit")


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def compute_downstream_delay(df: pd.DataFrame) -> pd.Series:
    """``downstream_delay_h = final_delivery_delay_h - arrival_delay_h``."""
    return (df["final_delivery_delay_h"] - df["arrival_delay_h"]).rename("downstream_delay_h")


def verify_targets(df: pd.DataFrame, tol: float = IDENTITY_TOL_H) -> None:
    """Verify target identities against stored values and milestone timestamps."""
    downstream = compute_downstream_delay(df)
    if not np.allclose(downstream, df["downstream_delay_h"], atol=tol):
        raise DatasetValidationError("downstream_delay_h != final_delivery_delay_h - arrival_delay_h")
    if not np.allclose(df["final_delivery_delay_h"], df["arrival_delay_h"] + df["downstream_delay_h"], atol=tol):
        raise DatasetValidationError("final_delivery_delay_h != arrival_delay_h + downstream_delay_h")
    stage_sum = df[["port_processing_delay_h", "customs_delay_h", "inland_delivery_delay_h"]].sum(axis=1)
    if not np.allclose(stage_sum, df["downstream_delay_h"], atol=tol):
        raise DatasetValidationError("downstream_delay_h != D3 + D4 + D5")
    final_from_ts = (df["actual_delivery"] - df["scheduled_delivery"]).dt.total_seconds() / 3600.0
    if not np.allclose(final_from_ts, df["final_delivery_delay_h"], atol=tol):
        raise DatasetValidationError("final_delivery_delay_h != actual_delivery - scheduled_delivery")
    arrival_from_ts = (df["actual_port_arrival"] - df["scheduled_port_arrival"]).dt.total_seconds() / 3600.0
    if not np.allclose(arrival_from_ts, df["arrival_delay_h"], atol=tol):
        raise DatasetValidationError("arrival_delay_h != actual_port_arrival - scheduled_port_arrival")
    threshold = dg.DEFAULT_CONFIG["reporting"]["delayed_threshold_h"]
    if not (df["is_delayed_24h"] == (df["final_delivery_delay_h"] > threshold)).all():
        raise DatasetValidationError("is_delayed_24h inconsistent with final_delivery_delay_h")


def verify_stage_identities(df: pd.DataFrame, tol: float = IDENTITY_TOL_H) -> float:
    """Recompute stage delays from milestones for every row; return the max abs error."""
    worst = 0.0
    for _, row in df.iterrows():
        d = dp.stage_delays_from_milestones([row[c] for c in PLANNED_MILESTONES],
                                            [row[c] for c in ACTUAL_MILESTONES])
        worst = max(worst, max(abs(a - row[c]) for a, c in zip(d, STAGE_DELAY_COLUMNS)))
    if worst > tol:
        raise DatasetValidationError(f"stage delays do not reconcile with milestones (max error {worst:.3g} h)")
    return worst


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForwardFold:
    """One chronological fold. Indices are positional (``df.iloc``)."""

    fold: int
    test_start: datetime
    test_end: datetime
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_embargoed: int  # rows with T1 < start excluded only because delivery >= start


def forward_windows() -> list[tuple[datetime, datetime]]:
    """Validation windows from the generator configuration (single source of truth)."""
    return [(dg._ts(s), dg._ts(e)) for s, e in dg.DEFAULT_CONFIG["allocation"]["test_windows"]]


def forward_folds(df: pd.DataFrame) -> list[ForwardFold]:
    """Three expanding-window forward folds with a label-availability embargo."""
    t1 = df["actual_port_arrival"]
    delivered = df["actual_delivery"]
    folds = []
    for k, (start, end) in enumerate(forward_windows(), start=1):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        test = ((t1 >= start_ts) & (t1 < end_ts)).to_numpy()
        arrived_before = (t1 < start_ts).to_numpy()
        train = arrived_before & (delivered < start_ts).to_numpy()
        folds.append(ForwardFold(
            fold=k, test_start=start, test_end=end,
            train_idx=np.flatnonzero(train), test_idx=np.flatnonzero(test),
            n_embargoed=int((arrived_before & ~train).sum()),
        ))
    return folds


def repeated_route_cv(
    df: pd.DataFrame, n_splits: int = 5, n_repeats: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated route-stratified K-fold splits as positional ``(train_idx, test_idx)`` pairs."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    routes = df["route_id"].to_numpy()
    return [(train, test) for train, test in cv.split(np.zeros(len(df)), routes)]
