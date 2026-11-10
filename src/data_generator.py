"""Synthetic maritime shipment generator.

Every parameter in :data:`DEFAULT_CONFIG` is a **synthetic assumption**. Port
names are labels only; distances, durations, congestion levels, customs rules
and carrier behaviour are not verified operational figures.

Generation outline (per shipment)
---------------------------------
1. The checkpoint ``T1`` (actual destination-port arrival) comes from a fixed,
   balanced allocation: per route, a fixed number of equal time bins in 2024
   and exactly ``per_route_per_window`` bins in each 2025 validation window,
   one uniformly sampled timestamp per bin, floored to the minute.
2. Upstream: departure delay ``D1`` and ocean transit are generated, then
   ``P1 = T1 - A``, ``P0 = P1 - p1`` and ``T0 = P0 + D1`` are built backwards.
3. Downstream: the plan is propagated forward from ``P1`` with planned
   durations and the actual timeline forward from ``T1`` with actual durations,
   both through the same destination-local customs calendar.
4. Stage delays are derived from milestone timestamps only.
5. Missingness is applied last and never changes ground truth.

Randomness uses five independent streams spawned from one ``SeedSequence``.
All variates are drawn unconditionally in standardised form and scaled
afterwards, so the number of draws never depends on simulated state.

Documented simplifications:
- ``planned_truck_slot = P3`` (planned customs release). Lateness against this
  appointment drives congestion-dependent stochastic truck-slot availability:
  a synthetic approximation in which window availability is sampled
  independently per shipment (no shared capacity between shipments).
- Cargo ready before its planned truck slot may be collected early.
- Upstream factors are not seasonal; seasonality acts only through destination
  congestion and destination weather, keyed to ``T1`` in destination-local time.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import delay_propagation as dp

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Configuration (all values are synthetic assumptions)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "rng_streams": ["t1_allocation", "shipment_context", "upstream", "downstream", "missingness"],
    "period_utc": {"start": "2024-01-01T00:00:00+00:00", "end": "2026-01-01T00:00:00+00:00"},
    "allocation": {
        "train_period": ["2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"],
        "test_windows": [
            ["2025-01-01T00:00:00+00:00", "2025-05-01T00:00:00+00:00"],
            ["2025-05-01T00:00:00+00:00", "2025-09-01T00:00:00+00:00"],
            ["2025-09-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"],
        ],
        "per_route_per_window": 7,
        "t1_floor": "minute",
    },
    "routes": [
        {"id": "R1", "origin": "Mumbai", "destination": "Rotterdam", "distance_km": 10500, "planned_transit_h": 528, "records": 42, "chokepoint": True},
        {"id": "R2", "origin": "Shanghai", "destination": "Los Angeles", "distance_km": 10500, "planned_transit_h": 432, "records": 42, "chokepoint": False},
        {"id": "R3", "origin": "Singapore", "destination": "Hamburg", "distance_km": 15500, "planned_transit_h": 672, "records": 42, "chokepoint": True},
        {"id": "R4", "origin": "Chennai", "destination": "Dubai", "distance_km": 3000, "planned_transit_h": 216, "records": 42, "chokepoint": False},
        {"id": "R5", "origin": "Shenzhen", "destination": "New York", "distance_km": 19000, "planned_transit_h": 840, "records": 41, "chokepoint": True},
        {"id": "R6", "origin": "Mumbai", "destination": "Singapore", "distance_km": 4000, "planned_transit_h": 288, "records": 41, "chokepoint": False},
    ],
    "destinations": {
        "Rotterdam": {"tz": "Europe/Amsterdam", "base_congestion": 0.40, "planned_dwell_h": 60, "winter_weather": True, "inland_km": [50, 400]},
        "Los Angeles": {"tz": "America/Los_Angeles", "base_congestion": 0.55, "planned_dwell_h": 72, "winter_weather": False, "inland_km": [30, 600]},
        "Hamburg": {"tz": "Europe/Berlin", "base_congestion": 0.45, "planned_dwell_h": 60, "winter_weather": True, "inland_km": [50, 500]},
        "Dubai": {"tz": "Asia/Dubai", "base_congestion": 0.35, "planned_dwell_h": 48, "winter_weather": False, "inland_km": [20, 250]},
        "New York": {"tz": "America/New_York", "base_congestion": 0.50, "planned_dwell_h": 72, "winter_weather": True, "inland_km": [30, 450]},
        "Singapore": {"tz": "Asia/Singapore", "base_congestion": 0.30, "planned_dwell_h": 36, "winter_weather": False, "inland_km": [10, 60]},
    },
    "calendar": {"non_working_weekdays": [5, 6], "workday_start_hour": 8},
    "ocean": {
        "min_frac_of_planned": 0.85, "sigma_frac": 0.025, "slack_frac": 0.03, "recovery_fraction": [0.2, 0.6],
        "weather_coef_h": 36, "disruption_prob": 0.04, "disruption_h": [24, 120],
    },
    "departure": {
        "noise_sd_h": 3, "congestion_coef_h": 24, "weather_threshold": 0.7, "weather_delay_h": [6, 24],
        "carrier_exp_scale_h": 6, "min_h": -6,
    },
    "carriers": {
        "Carrier A": {"share": 0.35, "k": 0.5},
        "Carrier B": {"share": 0.30, "k": 1.0},
        "Carrier C": {"share": 0.20, "k": 1.5},
        "Carrier D": {"share": 0.15, "k": 2.0},
    },
    "factors": {
        "origin_congestion_beta": [2, 5], "origin_weather_beta": [2, 4], "enroute_weather_beta": [2, 5],
        "dest_weather_beta": [2, 5], "dest_weather_winter_amp": 0.08, "dest_weather_mean_ref": 0.3,
        "congestion_season_amp": 0.10, "congestion_peak_month": 9, "congestion_noise_sd": 0.08,
        "disruption_bunching": 0.15, "dwell_drift_sd": 0.10,
    },
    "port": {
        "min_h": 6, "dwell_multiplier": [0.85, 1.15], "berth_tolerance_h": 12,
        "missed_berth_gamma_shape": 2, "missed_berth_scale_h": 6, "missed_berth_congestion_mult": 3,
        "on_time_berth_exp_scale_h": 4, "early_absorption_frac": 0.5, "early_grace_h": 6,
        "congestion_handling_coef": 0.8, "weather_handling_coef": 0.3, "noise_frac": 0.10,
        "storm_threshold": 0.75, "storm_delay_h": [12, 36],
    },
    "cargo": {
        "mix": {"general": 0.55, "reefer": 0.15, "hazardous": 0.10, "high_value": 0.20},
        "planned_customs_processing_h": {"general": 24, "reefer": 12, "hazardous": 36, "high_value": 30},
    },
    "customs": {
        "min_processing_h": 2, "docs_complete_prob": 0.85, "inspection_intercept": -2.44,
        "inspection_cargo_coef": {"general": 0.0, "reefer": -0.4, "hazardous": 1.0, "high_value": 0.7},
        "inspection_docs_coef": 1.2, "inspection_gamma": [2, 18], "incomplete_docs_h": [12, 48],
        "congestion_coef": 0.3, "noise_frac": 0.15,
    },
    "inland": {
        "planning_speed_kmh": 50, "min_speed_kmh": 60, "min_handling_h": 1, "handling_allowance_h": [4, 10],
        "buffer_frac": 0.15, "recovery_fraction": [0.3, 0.7], "noise_frac_of_drive": 0.10,
        "slot_window_h": 6, "slot_availability_max": 0.90, "slot_availability_congestion_coef": 0.60,
        "slot_availability_min": 0.30, "max_windows_skipped": 8,
    },
    "missingness": {"dest_congestion_obs": 0.10, "actual_departure": 0.08},
    "reporting": {"delayed_threshold_h": 24},
}

CONFIG_NOTES: dict[str, str] = {
    "status": "All values are synthetic assumptions, not verified maritime schedules or operational statistics.",
    "ports": "Port names are labels only.",
    "units": "Durations in hours unless suffixed otherwise; timestamps stored and computed in UTC.",
    "checkpoint": "T1 = actual destination-port arrival. Upstream timestamps are built backwards from T1.",
    "customs_calendar": "Synthetic rule: customs processing may only start Mon-Fri at or after 08:00 destination-local time; public holidays are ignored.",
    "truck_slot": "planned_truck_slot = P3 (planned customs release). Lateness L = T3 - P3 is measured against this appointment.",
    "truck_slot_availability": (
        "Congestion-dependent stochastic truck-slot availability (synthetic approximation). Pickup windows of "
        "slot_window_h hours start at P3; window 0 is the booked window. If L < window, wait = 0. Otherwise "
        "j = floor(L / window); N consecutive windows starting with window j are unavailable, with "
        "q = clip(q_max - kappa * congestion_dwell, q_min, q_max) and N = min(N_max, floor(ln u / ln(1 - q))), "
        "u ~ U(0,1]. N = 0 gives wait 0, otherwise wait = (j + N) * window - L. After N_max skipped windows the "
        "next window is treated as available, so wait <= N_max * window. Availability is sampled independently "
        "per shipment: there is no shared capacity or competition between shipments."
    ),
    "truck_slot_expected_wait": (
        "Capped geometric: E[N] = sum_{k=1..N_max} (1-q)^k. At L = window, E[wait] = window * E[N]. "
        "Theoretical E[wait] at L = window: q=0.90 -> 0.67h, q=0.72 -> 2.33h, q=0.54 -> 5.10h,"
        "q=0.36 -> 10.37h, q=0.30 -> 13.19h. These are properties of the mechanism, not dataset targets."
    ),
    "truck_slot_change_log": (
        "Replaces an earlier fixed rebooking jump of U(12,36)*(1+congestion) whenever L > 6h, which produced a "
        "structural gap in final delay. The new mechanism is not tuned to any model metric or delay distribution."
    ),
    "early_pickup": "Cargo ready before its planned truck slot may be collected early (deliberate simplification).",
    "seasonality": "Only destination congestion and destination weather are seasonal, keyed to T1's destination-local month.",
    "missingness": "MCAR, applied after ground truth. Masked actual departures leave P0, P1, T1 and arrival_delay_h observable.",
}

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

ID_META_COLUMNS = ["shipment_id", "route_id", "validation_window"]
CONTEXT_COLUMNS = [
    "origin_port", "destination_port", "dest_timezone", "carrier", "cargo_type", "documentation_complete",
    "route_distance_km", "has_chokepoint", "inland_distance_km",
]
PLAN_COLUMNS = [
    "scheduled_departure", "scheduled_port_arrival", "planned_port_exit", "planned_customs_release",
    "scheduled_delivery", "planned_transit_h", "planned_port_dwell_h", "planned_customs_processing_h",
    "planned_customs_wait_h", "planned_customs_h", "planned_inland_h",
]
OBSERVED_AT_CHECKPOINT_COLUMNS = [
    "actual_departure", "actual_port_arrival", "departure_delay_h", "ocean_transit_delay_h", "arrival_delay_h",
    "origin_congestion_index", "origin_weather_severity", "enroute_weather_severity", "chokepoint_disruption",
    "dest_congestion_index", "dest_weather_severity", "projected_nonworking_wait_h",
]
POST_CHECKPOINT_COLUMNS = [
    "actual_port_exit", "actual_customs_processing_start", "actual_customs_release", "actual_delivery",
    "actual_customs_wait_h", "customs_inspection", "booked_window_missed",
    "port_processing_delay_h", "customs_delay_h", "inland_delivery_delay_h",
]
TARGET_COLUMNS = ["downstream_delay_h", "final_delivery_delay_h", "is_delayed_24h"]
LATENT_COLUMNS = [
    "_latent_actual_departure", "_latent_departure_delay_h", "_latent_ocean_transit_delay_h",
    "_latent_dest_congestion_true", "_latent_congestion_dwell", "_latent_missed_berth_window",
    "_latent_berth_wait_h", "_latent_early_absorption_h", "_latent_storm_closure_h",
    "_latent_ocean_recovery_h", "_latent_truck_wait_h", "_latent_windows_skipped",
    "_latent_slot_availability", "_latent_inland_recovery_h",
]
ALL_COLUMNS = (
    ID_META_COLUMNS + CONTEXT_COLUMNS + PLAN_COLUMNS + OBSERVED_AT_CHECKPOINT_COLUMNS
    + POST_CHECKPOINT_COLUMNS + TARGET_COLUMNS + LATENT_COLUMNS
)
CHECKPOINT_AVAILABLE_COLUMNS = CONTEXT_COLUMNS + PLAN_COLUMNS + OBSERVED_AT_CHECKPOINT_COLUMNS

#: Primary model inputs (GBR and Ridge). `planned_customs_processing_h` and
#: `planned_customs_h` are excluded: processing is an exact function of cargo_type.
MODEL_NUMERIC_FEATURES = [
    "arrival_delay_h", "dest_congestion_index", "dest_weather_severity", "planned_port_dwell_h",
    "projected_nonworking_wait_h", "inland_distance_km", "planned_inland_h", "documentation_complete",
]
MODEL_CATEGORICAL_FEATURES = ["destination_port", "cargo_type"]
MODEL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
#: Used only in the carrier ablation experiment.
ABLATION_FEATURES = ["carrier"]
MODEL_TARGET = "downstream_delay_h"

TIMESTAMP_COLUMNS = [
    "scheduled_departure", "scheduled_port_arrival", "planned_port_exit", "planned_customs_release",
    "scheduled_delivery", "actual_departure", "actual_port_arrival", "actual_port_exit",
    "actual_customs_processing_start", "actual_customs_release", "actual_delivery", "_latent_actual_departure",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(value: str) -> datetime:
    return dp.ensure_utc(datetime.fromisoformat(value))


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _calendar(config: dict) -> tuple[frozenset[int], int]:
    cal = config["calendar"]
    return frozenset(cal["non_working_weekdays"]), int(cal["workday_start_hour"])


def validate_config(config: dict) -> None:
    """Basic structural checks on a configuration."""
    windows = config["allocation"]["test_windows"]
    per_window = config["allocation"]["per_route_per_window"]
    for route in config["routes"]:
        if route["destination"] not in config["destinations"]:
            raise ValueError(f"route {route['id']} destination {route['destination']!r} has no destination config")
        if route["records"] <= per_window * len(windows):
            raise ValueError(f"route {route['id']} has too few records for the validation windows")
    for dest in config["destinations"].values():
        dp.get_zone(dest["tz"])
    for name, weights in (("carriers", [c["share"] for c in config["carriers"].values()]),
                          ("cargo mix", list(config["cargo"]["mix"].values()))):
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError(f"{name} shares must sum to 1")
    if set(config["cargo"]["mix"]) != set(config["cargo"]["planned_customs_processing_h"]):
        raise ValueError("cargo mix and planned customs processing must cover the same cargo types")


def validation_window(t1: datetime, config: dict) -> int:
    """Return the 1-based validation window containing ``t1``, or 0 if it is in no window."""
    t1 = dp.ensure_utc(t1)
    for i, (start, end) in enumerate(config["allocation"]["test_windows"], start=1):
        if _ts(start) <= t1 < _ts(end):
            return i
    return 0


def spawn_streams(seed: int, names: list[str]) -> dict[str, np.random.Generator]:
    """Independent named generators from one ``SeedSequence``."""
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children)}


# ---------------------------------------------------------------------------
# Checkpoint allocation
# ---------------------------------------------------------------------------

def _sample_period(rng: np.random.Generator, start: datetime, end: datetime, n_bins: int) -> list[datetime]:
    width_h = dp.hours_between(start, end) / n_bins
    out = []
    for i in range(n_bins):
        t = dp.add_hours(start, float(rng.uniform(i * width_h, (i + 1) * width_h)))
        out.append(t.replace(second=0, microsecond=0))  # floor to minute; start is minute-aligned
    return out


def allocate_checkpoints(config: dict, rng: np.random.Generator) -> list[tuple[dict, datetime]]:
    """Return ``(route, T1)`` pairs. Counts are fixed by design; only within-bin times are random."""
    alloc = config["allocation"]
    per_window = alloc["per_route_per_window"]
    windows = [(_ts(s), _ts(e)) for s, e in alloc["test_windows"]]
    train_start, train_end = (_ts(x) for x in alloc["train_period"])
    pairs: list[tuple[dict, datetime]] = []
    for route in config["routes"]:
        n_train = route["records"] - per_window * len(windows)
        times = _sample_period(rng, train_start, train_end, n_train)
        for start, end in windows:
            times += _sample_period(rng, start, end, per_window)
        pairs += [(route, t) for t in times]
    pairs.sort(key=lambda p: (p[1], p[0]["id"]))
    return pairs


# ---------------------------------------------------------------------------
# Random draws (standardised; independent of simulated state)
# ---------------------------------------------------------------------------

def draw_context(rng: np.random.Generator, route: dict, config: dict) -> dict:
    """Pre-checkpoint shipment context drawn from the ``shipment_context`` stream."""
    dest = config["destinations"][route["destination"]]
    f = config["factors"]
    carriers = list(config["carriers"])
    cargo_types = list(config["cargo"]["mix"])
    return {
        "carrier": carriers[rng.choice(len(carriers), p=[config["carriers"][c]["share"] for c in carriers])],
        "cargo_type": cargo_types[rng.choice(len(cargo_types), p=list(config["cargo"]["mix"].values()))],
        "documentation_complete": bool(rng.random() < config["customs"]["docs_complete_prob"]),
        "inland_distance_km": float(rng.uniform(*dest["inland_km"])),
        "handling_allowance_h": float(rng.uniform(*config["inland"]["handling_allowance_h"])),
        "dwell_multiplier": float(rng.uniform(*config["port"]["dwell_multiplier"])),
        "origin_congestion": float(rng.beta(*f["origin_congestion_beta"])),
        "origin_weather": float(rng.beta(*f["origin_weather_beta"])),
        "enroute_weather": float(rng.beta(*f["enroute_weather_beta"])),
        "disruption_u": float(rng.random()),
        "disruption_h": float(rng.uniform(*config["ocean"]["disruption_h"])),
        "dest_weather_base": float(rng.beta(*f["dest_weather_beta"])),
        "congestion_noise_z": float(rng.standard_normal()),
    }


def draw_upstream(rng: np.random.Generator, config: dict) -> dict:
    """Standardised variates for departure and ocean transit (``upstream`` stream)."""
    return {
        "departure_noise_z": float(rng.standard_normal()),
        "departure_weather_h": float(rng.uniform(*config["departure"]["weather_delay_h"])),
        "carrier_exp": float(rng.standard_exponential()),
        "ocean_noise_z": float(rng.standard_normal()),
        "ocean_recovery_fraction": float(rng.uniform(*config["ocean"]["recovery_fraction"])),
    }


def draw_downstream(rng: np.random.Generator, config: dict) -> dict:
    """Standardised variates for port, customs and inland stages (``downstream`` stream)."""
    return {
        "drift_z": float(rng.standard_normal()),
        "berth_gamma": float(rng.standard_gamma(config["port"]["missed_berth_gamma_shape"])),
        "berth_exp": float(rng.standard_exponential()),
        "port_noise_z": float(rng.standard_normal()),
        "storm_h": float(rng.uniform(*config["port"]["storm_delay_h"])),
        "inspection_u": float(rng.random()),
        "inspection_gamma": float(rng.standard_gamma(config["customs"]["inspection_gamma"][0])),
        "docs_penalty_h": float(rng.uniform(*config["customs"]["incomplete_docs_h"])),
        "customs_noise_z": float(rng.standard_normal()),
        # Exactly one uniform per shipment, drawn unconditionally; mapped to (0, 1].
        "slot_u": float(1.0 - rng.random()),
        "inland_recovery_fraction": float(rng.uniform(*config["inland"]["recovery_fraction"])),
        "inland_noise_z": float(rng.standard_normal()),
    }


def slot_availability(congestion_dwell: float, config: dict) -> float:
    """Truck-window availability ``q = clip(q_max - kappa * congestion, q_min, q_max)``."""
    inl = config["inland"]
    q_min, q_max = inl["slot_availability_min"], inl["slot_availability_max"]
    return min(q_max, max(q_min, q_max - inl["slot_availability_congestion_coef"] * congestion_dwell))


def sample_windows_skipped(u: float, q: float, n_max: int) -> int:
    """Capped geometric number of unavailable windows from a uniform ``u`` in (0, 1].

    ``P(N >= k) = (1 - q)^k`` for ``k <= n_max``; ``N = n_max`` absorbs the tail.
    """
    if not 0.0 < u <= 1.0:
        raise ValueError(f"u must be in (0, 1], got {u}")
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    return int(min(n_max, math.floor(math.log(u) / math.log(1.0 - q))))


# ---------------------------------------------------------------------------
# Deterministic simulation of one shipment
# ---------------------------------------------------------------------------

def simulate_shipment(route: dict, t1: datetime, ctx: dict, up: dict, down: dict, config: dict) -> dict:
    """Simulate one shipment's full ground truth from its draws. Deterministic."""
    dest_name = route["destination"]
    dest = config["destinations"][dest_name]
    tz = dest["tz"]
    nwd, start_hour = _calendar(config)
    f, oc, dep, port, cus, inl = (config[k] for k in ("factors", "ocean", "departure", "port", "customs", "inland"))

    # --- Upstream: departure and ocean transit -----------------------------
    p1 = float(route["planned_transit_h"])
    k = config["carriers"][ctx["carrier"]]["k"]
    d1 = max(
        float(dep["min_h"]),
        dep["noise_sd_h"] * up["departure_noise_z"]
        + dep["congestion_coef_h"] * ctx["origin_congestion"] ** 2
        + (up["departure_weather_h"] if ctx["origin_weather"] > dep["weather_threshold"] else 0.0)
        + k * dep["carrier_exp_scale_h"] * up["carrier_exp"],
    )
    disruption = bool(route["chokepoint"] and ctx["disruption_u"] < oc["disruption_prob"])
    ocean_recovery = dp.recovery_h(d1, oc["slack_frac"] * p1, up["ocean_recovery_fraction"])
    dev2 = (
        oc["sigma_frac"] * p1 * up["ocean_noise_z"]
        + oc["weather_coef_h"] * ctx["enroute_weather"] ** 2
        + (ctx["disruption_h"] if disruption else 0.0)
        - ocean_recovery
    )
    a1 = dp.actual_stage_duration_h(p1, dev2, oc["min_frac_of_planned"] * p1)
    upstream = dp.upstream_from_checkpoint(t1, d1, p1, a1)
    arrival_delay = upstream.arrival_delay_h

    # --- Conditions at arrival (destination-local month) --------------------
    month = dp.to_local(t1, tz).month
    base_c = dest["base_congestion"]
    congestion = _clip01(
        base_c
        + f["congestion_season_amp"] * math.cos(2 * math.pi * (month - f["congestion_peak_month"]) / 12)
        + (f["disruption_bunching"] if disruption else 0.0)
        + f["congestion_noise_sd"] * ctx["congestion_noise_z"]
    )
    winter = math.cos(2 * math.pi * (month - 1) / 12) if dest["winter_weather"] else 0.0
    dest_weather = _clip01(ctx["dest_weather_base"] + f["dest_weather_winter_amp"] * winter)
    congestion_dwell = _clip01(congestion + f["dwell_drift_sd"] * down["drift_z"])

    # --- Port stage ----------------------------------------------------------
    p2 = dest["planned_dwell_h"] * ctx["dwell_multiplier"]
    missed_berth = dp.missed_window(arrival_delay, port["berth_tolerance_h"])
    if missed_berth:
        berth_wait = down["berth_gamma"] * port["missed_berth_scale_h"] * (1 + port["missed_berth_congestion_mult"] * congestion_dwell)
    else:
        berth_wait = down["berth_exp"] * port["on_time_berth_exp_scale_h"] * congestion_dwell
    early_absorption = port["early_absorption_frac"] * max(0.0, -arrival_delay - port["early_grace_h"])
    handling = p2 * (
        port["congestion_handling_coef"] * (congestion_dwell - base_c)
        + port["weather_handling_coef"] * (dest_weather - f["dest_weather_mean_ref"])
    )
    storm = down["storm_h"] if dest_weather > port["storm_threshold"] else 0.0
    dev3 = berth_wait + early_absorption + handling + storm + port["noise_frac"] * p2 * down["port_noise_z"]
    a2 = dp.actual_stage_duration_h(p2, dev3, port["min_h"])

    # --- Customs processing --------------------------------------------------
    cargo = ctx["cargo_type"]
    p3_proc = float(config["cargo"]["planned_customs_processing_h"][cargo])
    inspect_p = _sigmoid(
        cus["inspection_intercept"] + cus["inspection_cargo_coef"][cargo]
        + (0.0 if ctx["documentation_complete"] else cus["inspection_docs_coef"])
    )
    inspected = bool(down["inspection_u"] < inspect_p)
    dev4 = (
        (down["inspection_gamma"] * cus["inspection_gamma"][1] if inspected else 0.0)
        + (0.0 if ctx["documentation_complete"] else down["docs_penalty_h"])
        + cus["congestion_coef"] * (congestion_dwell - base_c) * p3_proc
        + cus["noise_frac"] * p3_proc * down["customs_noise_z"]
    )
    a3_proc = dp.actual_stage_duration_h(p3_proc, dev4, cus["min_processing_h"])

    # --- Plan (forward from P1, same calendar) -------------------------------
    km = ctx["inland_distance_km"]
    drive_plan = km / inl["planning_speed_kmh"] + ctx["handling_allowance_h"]
    p4 = (1 + inl["buffer_frac"]) * drive_plan
    m4 = km / inl["min_speed_kmh"] + inl["min_handling_h"]
    plan = dp.downstream_timeline(upstream.planned_arrival, p2, p3_proc, p4, tz, nwd, start_hour)
    planned_truck_slot = plan.customs_release  # documented assumption: slot = P3

    # --- Actual customs and inland ------------------------------------------
    t2 = dp.add_hours(t1, a2)
    customs_actual = dp.customs_timing(t2, a3_proc, tz, nwd, start_hour)
    t3 = customs_actual.release
    slot_lateness = dp.hours_between(planned_truck_slot, t3)
    availability = slot_availability(congestion_dwell, config)
    windows_skipped = sample_windows_skipped(down["slot_u"], availability, inl["max_windows_skipped"])
    booked_window_missed = slot_lateness >= inl["slot_window_h"]
    truck_wait = dp.truck_window_wait_h(t3, planned_truck_slot, inl["slot_window_h"], windows_skipped)
    lateness_at_pickup = slot_lateness + truck_wait
    inland_recovery = dp.recovery_h(lateness_at_pickup, inl["buffer_frac"] * drive_plan, down["inland_recovery_fraction"])
    dev5 = truck_wait + inl["noise_frac_of_drive"] * (km / inl["planning_speed_kmh"]) * down["inland_noise_z"] - inland_recovery
    a4 = dp.actual_stage_duration_h(p4, dev5, m4)
    actual = dp.downstream_timeline(t1, a2, a3_proc, a4, tz, nwd, start_hour)
    if actual.customs_release != t3:
        raise RuntimeError("inconsistent customs release")  # defensive; should never happen

    # --- Accounting from milestones -----------------------------------------
    planned_ms = [upstream.planned_departure, *plan.milestones]
    actual_ms = [upstream.actual_departure, *actual.milestones]
    d = dp.stage_delays_from_milestones(planned_ms, actual_ms)
    final = dp.final_delay_h(d)

    return {
        "route_id": route["id"],
        "validation_window": validation_window(t1, config),
        "origin_port": route["origin"],
        "destination_port": dest_name,
        "dest_timezone": tz,
        "carrier": ctx["carrier"],
        "cargo_type": cargo,
        "documentation_complete": ctx["documentation_complete"],
        "route_distance_km": float(route["distance_km"]),
        "has_chokepoint": bool(route["chokepoint"]),
        "inland_distance_km": km,
        # plan
        "scheduled_departure": planned_ms[0],
        "scheduled_port_arrival": planned_ms[1],
        "planned_port_exit": planned_ms[2],
        "planned_customs_release": planned_ms[3],
        "scheduled_delivery": planned_ms[4],
        "planned_transit_h": p1,
        "planned_port_dwell_h": p2,
        "planned_customs_processing_h": p3_proc,
        "planned_customs_wait_h": plan.customs.wait_h,
        "planned_customs_h": plan.customs.duration_h,
        "planned_inland_h": p4,
        # observed at checkpoint (before masking)
        "actual_departure": actual_ms[0],
        "actual_port_arrival": actual_ms[1],
        "departure_delay_h": d[0],
        "ocean_transit_delay_h": d[1],
        "arrival_delay_h": dp.arrival_delay_h(d),
        "origin_congestion_index": ctx["origin_congestion"],
        "origin_weather_severity": ctx["origin_weather"],
        "enroute_weather_severity": ctx["enroute_weather"],
        "chokepoint_disruption": disruption,
        "dest_congestion_index": congestion,
        "dest_weather_severity": dest_weather,
        "projected_nonworking_wait_h": dp.projected_customs_wait_h(t1, p2, tz, nwd, start_hour),
        # post-checkpoint outcomes
        "actual_port_exit": actual_ms[2],
        "actual_customs_processing_start": customs_actual.processing_start,
        "actual_customs_release": actual_ms[3],
        "actual_delivery": actual_ms[4],
        "actual_customs_wait_h": customs_actual.wait_h,
        "customs_inspection": inspected,
        "booked_window_missed": booked_window_missed,
        "port_processing_delay_h": d[2],
        "customs_delay_h": d[3],
        "inland_delivery_delay_h": d[4],
        # targets
        "downstream_delay_h": dp.downstream_delay_h(d),
        "final_delivery_delay_h": final,
        "is_delayed_24h": final > config["reporting"]["delayed_threshold_h"],
        # latent (audit only)
        "_latent_actual_departure": actual_ms[0],
        "_latent_departure_delay_h": d[0],
        "_latent_ocean_transit_delay_h": d[1],
        "_latent_dest_congestion_true": congestion,
        "_latent_congestion_dwell": congestion_dwell,
        "_latent_missed_berth_window": missed_berth,
        "_latent_berth_wait_h": berth_wait,
        "_latent_early_absorption_h": early_absorption,
        "_latent_storm_closure_h": storm,
        "_latent_ocean_recovery_h": ocean_recovery,
        "_latent_truck_wait_h": truck_wait,
        "_latent_windows_skipped": windows_skipped,
        "_latent_slot_availability": availability,
        "_latent_inland_recovery_h": inland_recovery,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def apply_missingness(df: pd.DataFrame, rng: np.random.Generator, config: dict) -> pd.DataFrame:
    """Mask observed columns (MCAR). Ground-truth and latent columns are untouched."""
    out = df.copy()
    n = len(out)
    miss_cong = rng.random(n) < config["missingness"]["dest_congestion_obs"]
    miss_dep = rng.random(n) < config["missingness"]["actual_departure"]
    out.loc[miss_cong, "dest_congestion_index"] = np.nan
    out.loc[miss_dep, ["departure_delay_h", "ocean_transit_delay_h"]] = np.nan
    out.loc[miss_dep, "actual_departure"] = pd.NaT
    return out


def generate_shipments(config: dict | None = None, seed: int | None = None) -> pd.DataFrame:
    """Generate the synthetic shipment dataset (one row per shipment)."""
    config = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
    if seed is not None:
        config["seed"] = int(seed)
    validate_config(config)
    rngs = spawn_streams(config["seed"], config["rng_streams"])

    pairs = allocate_checkpoints(config, rngs["t1_allocation"])
    rows = []
    for i, (route, t1) in enumerate(pairs, start=1):
        ctx = draw_context(rngs["shipment_context"], route, config)
        up = draw_upstream(rngs["upstream"], config)
        down = draw_downstream(rngs["downstream"], config)
        row = simulate_shipment(route, t1, ctx, up, down, config)
        row["shipment_id"] = f"SHP-{i:04d}"
        rows.append(row)

    df = pd.DataFrame(rows)[ALL_COLUMNS]
    for col in TIMESTAMP_COLUMNS:
        df[col] = pd.to_datetime(df[col], utc=True)
    return apply_missingness(df, rngs["missingness"], config)


def column_roles() -> dict[str, list[str]]:
    """Column groups, written into the configuration JSON for documentation."""
    return {
        "id_meta": ID_META_COLUMNS,
        "context": CONTEXT_COLUMNS,
        "plan": PLAN_COLUMNS,
        "observed_at_checkpoint": OBSERVED_AT_CHECKPOINT_COLUMNS,
        "post_checkpoint_outcomes": POST_CHECKPOINT_COLUMNS,
        "targets": TARGET_COLUMNS,
        "latent_audit_only": LATENT_COLUMNS,
        "model_features_numeric": MODEL_NUMERIC_FEATURES,
        "model_features_categorical": MODEL_CATEGORICAL_FEATURES,
        "ablation_only_features": ABLATION_FEATURES,
        "model_target": [MODEL_TARGET],
    }


def save_dataset(df: pd.DataFrame, config: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``shipments.csv`` and ``generation_config.json`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "shipments.csv"
    json_path = out / "generation_config.json"
    df.to_csv(csv_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    payload = {
        "notes": CONFIG_NOTES,
        "config": config,
        "schema": column_roles(),
        "output": {"rows": int(len(df)), "csv": csv_path.name},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def load_shipments(path: str | Path) -> pd.DataFrame:
    """Read a generated CSV, restoring UTC timestamps."""
    df = pd.read_csv(path)
    for col in TIMESTAMP_COLUMNS:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic shipment dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args(argv)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["seed"] = args.seed
    df = generate_shipments(config)
    csv_path, json_path = save_dataset(df, config, args.out_dir)
    print(f"wrote {len(df)} rows to {csv_path} and config to {json_path}")


if __name__ == "__main__":
    main()
