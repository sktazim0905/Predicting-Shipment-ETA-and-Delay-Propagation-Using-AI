"""Timestamp, calendar and stage-delay arithmetic for multi-stage shipments.

This module holds the deterministic mechanics that the synthetic data
generator (and later the feature builder) rely on. It contains no random
sampling: stochastic quantities are drawn elsewhere and passed in explicitly.

Milestones and stages
---------------------
Milestones ``k = 0..4``::

    0 departure  1 port arrival  2 port exit  3 customs release  4 delivery

Stages ``k = 1..4`` span ``milestone[k-1] -> milestone[k]``::

    1 ocean transit  2 port processing  3 customs  4 inland delivery

Definitions (all values in hours, all timestamps timezone-aware UTC)::

    planned stage duration   p_k = P_k - P_{k-1}
    actual stage duration    a_k = T_k - T_{k-1}        (a_k >= m_k > 0)
    stage delays             D1 = T_0 - P_0             (departure offset)
                             D2..D5 = a_k - p_k         (k = 1..4)
    cumulative delay         C_j = T_j - P_j
    arrival delay            A = C_1 = D1 + D2
    downstream delay         Y = D3 + D4 + D5 = C_4 - C_1
    final delay              F = C_4 = A + Y

Because consecutive milestone intervals partition the timeline, the sum
telescopes and every hour is attributed to exactly one stage.

Customs calendar rule (synthetic assumption, not a statement about real
customs operating hours): processing may only *start* on a working day at or
after ``workday_start_hour`` in destination-local time; once started it runs
continuously. Any wait before the start belongs to the customs stage. The same
rule is applied to the planned and the actual timeline so that a weekend that
is already in the plan is not counted as delay.

All arithmetic is done in UTC. Destination-local time is used only to decide
weekdays and the local start-of-day.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import AbstractSet, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = timezone.utc

#: Python weekday numbers (Monday=0 ... Sunday=6) on which customs processing
#: may not start. Synthetic default applied to every destination port.
DEFAULT_NON_WORKING_WEEKDAYS: frozenset[int] = frozenset({5, 6})

#: Destination-local hour at which customs processing may start on a working day.
DEFAULT_WORKDAY_START_HOUR: int = 8

N_MILESTONES: int = 5
N_STAGE_DELAYS: int = 5  # D1 (departure offset) + four stage-duration delays


# ---------------------------------------------------------------------------
# Validation and UTC / timezone primitives
# ---------------------------------------------------------------------------

def ensure_utc(ts: datetime, name: str = "timestamp") -> datetime:
    """Return ``ts`` converted to UTC, rejecting naive datetimes.

    Raises:
        TypeError: if ``ts`` is not a ``datetime``.
        ValueError: if ``ts`` is naive (has no UTC offset).
    """
    if not isinstance(ts, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(ts).__name__}")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware; got naive {ts!r}")
    return ts.astimezone(UTC)


def get_zone(tz_key: str) -> ZoneInfo:
    """Return the ``ZoneInfo`` for an IANA key, raising ``ValueError`` if invalid."""
    if not isinstance(tz_key, str) or not tz_key:
        raise ValueError(f"timezone key must be a non-empty string, got {tz_key!r}")
    try:
        return ZoneInfo(tz_key)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone key: {tz_key!r}") from exc


def to_local(ts: datetime, tz_key: str) -> datetime:
    """Convert an aware timestamp to the local time of ``tz_key``."""
    return ensure_utc(ts).astimezone(get_zone(tz_key))


def add_hours(ts: datetime, hours: float) -> datetime:
    """Add ``hours`` to ``ts`` using absolute (UTC) time and return a UTC datetime.

    Doing this in UTC avoids Python's wall-clock arithmetic on ZoneInfo-aware
    datetimes, which would be wrong across DST transitions.
    """
    _require_finite(hours, "hours")
    return ensure_utc(ts) + timedelta(hours=hours)


def hours_between(start: datetime, end: datetime) -> float:
    """Return ``end - start`` in hours (negative if ``end`` precedes ``start``)."""
    return (ensure_utc(end, "end") - ensure_utc(start, "start")).total_seconds() / 3600.0


def _require_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def validate_duration_h(value: float, name: str = "duration_h", *, allow_zero: bool = False) -> float:
    """Validate a duration in hours: finite and positive (or non-negative if ``allow_zero``)."""
    v = _require_finite(value, name)
    if v < 0 or (v == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {bound}, got {v}")
    return v


def _validate_calendar(non_working_weekdays: AbstractSet[int], workday_start_hour: int) -> None:
    if not set(non_working_weekdays) <= set(range(7)):
        raise ValueError(f"non_working_weekdays must be within 0..6, got {sorted(non_working_weekdays)}")
    if len(set(non_working_weekdays)) == 7:
        raise ValueError("non_working_weekdays cannot cover all seven days")
    if isinstance(workday_start_hour, bool) or not isinstance(workday_start_hour, int):
        raise TypeError("workday_start_hour must be an int")
    if not 0 <= workday_start_hour <= 23:
        raise ValueError(f"workday_start_hour must be in 0..23, got {workday_start_hour}")


def _local_wall_time_to_utc(day: date, hour: int, zone: ZoneInfo) -> datetime:
    """Convert a local wall-clock time to UTC, rejecting non-existent (DST gap) times.

    Ambiguous times (DST fall-back) resolve to the first occurrence (``fold=0``).
    """
    local = datetime.combine(day, time(hour), tzinfo=zone)
    as_utc = local.astimezone(UTC)
    if as_utc.astimezone(zone).replace(tzinfo=None) != local.replace(tzinfo=None):
        raise ValueError(f"local time {local.replace(tzinfo=None)} does not exist in {zone.key}")
    return as_utc


# ---------------------------------------------------------------------------
# Customs calendar
# ---------------------------------------------------------------------------

def next_working_start(
    ts: datetime,
    tz_key: str,
    non_working_weekdays: AbstractSet[int] = DEFAULT_NON_WORKING_WEEKDAYS,
    workday_start_hour: int = DEFAULT_WORKDAY_START_HOUR,
) -> datetime:
    """Earliest UTC instant >= ``ts`` at which customs processing may start.

    If ``ts`` falls on a destination-local working day at or after the start
    hour, it is returned unchanged (in UTC). Otherwise the result is the start
    hour of the same day (if still ahead) or of the next working day.
    """
    _validate_calendar(non_working_weekdays, workday_start_hour)
    ts_utc = ensure_utc(ts)
    zone = get_zone(tz_key)
    local = ts_utc.astimezone(zone)

    if local.weekday() not in non_working_weekdays and local.hour >= workday_start_hour:
        return ts_utc

    day = local.date()
    for _ in range(8):
        if day.weekday() not in non_working_weekdays:
            candidate = _local_wall_time_to_utc(day, workday_start_hour, zone)
            if candidate >= ts_utc:
                return candidate
        day += timedelta(days=1)
    raise RuntimeError("no working day found within 8 days")  # unreachable after validation


def non_working_wait_h(
    ts: datetime,
    tz_key: str,
    non_working_weekdays: AbstractSet[int] = DEFAULT_NON_WORKING_WEEKDAYS,
    workday_start_hour: int = DEFAULT_WORKDAY_START_HOUR,
) -> float:
    """Hours between ``ts`` and the next permitted customs processing start (>= 0)."""
    start = next_working_start(ts, tz_key, non_working_weekdays, workday_start_hour)
    return hours_between(ts, start)


@dataclass(frozen=True)
class CustomsTiming:
    """Timing of one customs stage (planned or actual). Timestamps are UTC."""

    lodged: datetime            # port exit = customs lodgement (milestone 2)
    processing_start: datetime  # first permitted start per the calendar rule
    release: datetime           # customs release (milestone 3)
    wait_h: float               # non-working-day wait, part of the customs stage
    processing_h: float         # continuous processing time

    @property
    def duration_h(self) -> float:
        """Total customs stage duration: ``wait_h + processing_h``."""
        return hours_between(self.lodged, self.release)


def customs_timing(
    lodged: datetime,
    processing_h: float,
    tz_key: str,
    non_working_weekdays: AbstractSet[int] = DEFAULT_NON_WORKING_WEEKDAYS,
    workday_start_hour: int = DEFAULT_WORKDAY_START_HOUR,
) -> CustomsTiming:
    """Apply the customs calendar rule to a lodgement time and processing duration.

    Use the same function for the planned and the actual timeline.
    """
    processing_h = validate_duration_h(processing_h, "processing_h")
    lodged_utc = ensure_utc(lodged, "lodged")
    start = next_working_start(lodged_utc, tz_key, non_working_weekdays, workday_start_hour)
    return CustomsTiming(
        lodged=lodged_utc,
        processing_start=start,
        release=add_hours(start, processing_h),
        wait_h=hours_between(lodged_utc, start),
        processing_h=processing_h,
    )


# ---------------------------------------------------------------------------
# Stage durations and delays
# ---------------------------------------------------------------------------

def actual_stage_duration_h(planned_h: float, deviation_h: float, min_h: float) -> float:
    """Actual stage duration ``max(min_h, planned_h + deviation_h)``.

    The deviation may be negative (early / recovered stage), but the result is
    always at least the physical minimum ``min_h > 0``.
    """
    planned_h = validate_duration_h(planned_h, "planned_h")
    min_h = validate_duration_h(min_h, "min_h")
    deviation_h = _require_finite(deviation_h, "deviation_h")
    return max(min_h, planned_h + deviation_h)


def stage_delay_h(actual_h: float, planned_h: float) -> float:
    """Stage delay ``actual_h - planned_h``; negative means the stage ran short."""
    return validate_duration_h(actual_h, "actual_h") - validate_duration_h(planned_h, "planned_h")


def recovery_h(upstream_delay_h: float, capacity_h: float, fraction: float) -> float:
    """Hours recovered against an upstream delay.

    Returns ``fraction * min(max(upstream_delay_h, 0), capacity_h)``. Only
    positive (late) upstream delay can be recovered, and never more than the
    available slack ``capacity_h``. The caller subtracts the result from the
    stage deviation before calling :func:`actual_stage_duration_h`.
    """
    upstream_delay_h = _require_finite(upstream_delay_h, "upstream_delay_h")
    capacity_h = validate_duration_h(capacity_h, "capacity_h", allow_zero=True)
    fraction = _require_finite(fraction, "fraction")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return fraction * min(max(upstream_delay_h, 0.0), capacity_h)


def missed_window(lateness_h: float, tolerance_h: float) -> bool:
    """True if ``lateness_h`` strictly exceeds the allowed ``tolerance_h`` (>= 0)."""
    lateness_h = _require_finite(lateness_h, "lateness_h")
    tolerance_h = validate_duration_h(tolerance_h, "tolerance_h", allow_zero=True)
    return lateness_h > tolerance_h


# ---------------------------------------------------------------------------
# Timelines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpstreamTimeline:
    """Departure and port-arrival milestones built backwards from the checkpoint."""

    planned_departure: datetime  # P0
    actual_departure: datetime   # T0
    planned_arrival: datetime    # P1
    actual_arrival: datetime     # T1 (checkpoint)
    departure_delay_h: float     # D1
    transit_delay_h: float       # D2
    arrival_delay_h: float       # A = D1 + D2


def upstream_from_checkpoint(
    actual_arrival: datetime,
    departure_delay_h: float,
    planned_transit_h: float,
    actual_transit_h: float,
) -> UpstreamTimeline:
    """Build P0, T0, P1 backwards from the sampled checkpoint T1.

    ``P1 = T1 - A``, ``P0 = P1 - p1`` and ``T0 = P0 + D1``, where
    ``D2 = a1 - p1`` and ``A = D1 + D2``. This guarantees
    ``T1 = T0 + a1`` exactly.
    """
    t1 = ensure_utc(actual_arrival, "actual_arrival")
    d1 = _require_finite(departure_delay_h, "departure_delay_h")
    p1 = validate_duration_h(planned_transit_h, "planned_transit_h")
    a1 = validate_duration_h(actual_transit_h, "actual_transit_h")
    d2 = a1 - p1
    arrival_delay = d1 + d2
    planned_arrival = add_hours(t1, -arrival_delay)
    planned_departure = add_hours(planned_arrival, -p1)
    actual_departure = add_hours(planned_departure, d1)
    return UpstreamTimeline(
        planned_departure=planned_departure,
        actual_departure=actual_departure,
        planned_arrival=planned_arrival,
        actual_arrival=t1,
        departure_delay_h=d1,
        transit_delay_h=d2,
        arrival_delay_h=arrival_delay,
    )


@dataclass(frozen=True)
class DownstreamTimeline:
    """Milestones 1..4 simulated forward from port arrival (planned or actual)."""

    port_arrival: datetime     # milestone 1
    port_exit: datetime        # milestone 2
    customs: CustomsTiming     # wait + processing; release is milestone 3
    delivery: datetime         # milestone 4

    @property
    def customs_release(self) -> datetime:
        return self.customs.release

    @property
    def milestones(self) -> tuple[datetime, datetime, datetime, datetime]:
        """Milestones 1..4 as UTC datetimes."""
        return (self.port_arrival, self.port_exit, self.customs.release, self.delivery)

    @property
    def stage_durations_h(self) -> tuple[float, float, float]:
        """Durations of the port, customs and inland stages."""
        m = self.milestones
        return tuple(hours_between(a, b) for a, b in zip(m[:-1], m[1:]))  # type: ignore[return-value]


def downstream_timeline(
    port_arrival: datetime,
    port_dwell_h: float,
    customs_processing_h: float,
    inland_h: float,
    tz_key: str,
    non_working_weekdays: AbstractSet[int] = DEFAULT_NON_WORKING_WEEKDAYS,
    workday_start_hour: int = DEFAULT_WORKDAY_START_HOUR,
) -> DownstreamTimeline:
    """Propagate port -> customs -> inland forward from a port-arrival time.

    Call it once with ``P1`` and planned durations (the plan) and once with
    ``T1`` and actual durations (the outcome). ``inland_h`` must already
    include any truck-slot rebooking wait. Customs non-working-day waits are
    added here, inside the customs stage only.
    """
    arrival = ensure_utc(port_arrival, "port_arrival")
    port_exit = add_hours(arrival, validate_duration_h(port_dwell_h, "port_dwell_h"))
    customs = customs_timing(
        port_exit, customs_processing_h, tz_key, non_working_weekdays, workday_start_hour
    )
    delivery = add_hours(customs.release, validate_duration_h(inland_h, "inland_h"))
    return DownstreamTimeline(port_arrival=arrival, port_exit=port_exit, customs=customs, delivery=delivery)


def projected_customs_wait_h(
    actual_arrival: datetime,
    planned_port_dwell_h: float,
    tz_key: str,
    non_working_weekdays: AbstractSet[int] = DEFAULT_NON_WORKING_WEEKDAYS,
    workday_start_hour: int = DEFAULT_WORKDAY_START_HOUR,
) -> float:
    """Checkpoint feature: projected non-working wait if the port stage runs to plan.

    Uses only information available at port arrival: the actual arrival ``T1``
    and the *planned* dwell ``p2``. It never uses the actual port exit.
    """
    projected_exit = add_hours(actual_arrival, validate_duration_h(planned_port_dwell_h, "planned_port_dwell_h"))
    return non_working_wait_h(projected_exit, tz_key, non_working_weekdays, workday_start_hour)


def truck_window_wait_h(
    cargo_ready: datetime,
    planned_slot: datetime,
    window_h: float,
    windows_skipped: int,
) -> float:
    """Truck pickup wait under congestion-dependent stochastic slot availability.

    Pickup windows are contiguous intervals ``[planned_slot + j*window_h,
    planned_slot + (j+1)*window_h)``. Window ``j = 0`` is the booked window.
    With lateness ``L = cargo_ready - planned_slot`` (hours):

    - ``L < window_h`` (early or within the booked window): wait 0. Early cargo
      is collected immediately.
    - otherwise ``j = floor(L / window_h)`` and ``N = windows_skipped`` is the
      number of consecutive unavailable windows starting with window ``j``
      (sampled by the caller). ``N = 0`` gives wait 0; ``N >= 1`` gives
      ``(j + N) * window_h - L``, which lies in ``((N-1)*window_h, N*window_h]``.

    Deterministic: availability sampling happens in the generator.
    """
    window_h = validate_duration_h(window_h, "window_h")
    if isinstance(windows_skipped, bool) or not isinstance(windows_skipped, numbers.Integral):
        raise TypeError("windows_skipped must be an integer")
    if windows_skipped < 0:
        raise ValueError(f"windows_skipped must be >= 0, got {windows_skipped}")
    lateness = hours_between(planned_slot, cargo_ready)
    if lateness < window_h or windows_skipped == 0:
        return 0.0
    j = math.floor(lateness / window_h)
    return (j + int(windows_skipped)) * window_h - lateness


# ---------------------------------------------------------------------------
# Delay accounting
# ---------------------------------------------------------------------------

def _validate_milestones(milestones: Sequence[datetime], name: str) -> list[datetime]:
    if len(milestones) != N_MILESTONES:
        raise ValueError(f"{name} must contain {N_MILESTONES} milestones, got {len(milestones)}")
    out = [ensure_utc(m, f"{name}[{i}]") for i, m in enumerate(milestones)]
    for i in range(1, N_MILESTONES):
        if out[i] <= out[i - 1]:
            raise ValueError(
                f"{name} must be strictly increasing; milestone {i} ({out[i]}) "
                f"is not after milestone {i - 1} ({out[i - 1]})"
            )
    return out


def stage_delays_from_milestones(
    planned: Sequence[datetime], actual: Sequence[datetime]
) -> tuple[float, float, float, float, float]:
    """Stage delays ``(D1, D2, D3, D4, D5)`` from planned and actual milestones 0..4.

    ``D1 = T0 - P0``; ``D_{k+1} = (T_k - T_{k-1}) - (P_k - P_{k-1})`` for
    ``k = 1..4``. Both sequences must be strictly increasing, which enforces
    positive planned and actual stage durations.
    """
    p = _validate_milestones(planned, "planned")
    t = _validate_milestones(actual, "actual")
    d1 = hours_between(p[0], t[0])
    rest = [hours_between(t[k - 1], t[k]) - hours_between(p[k - 1], p[k]) for k in range(1, N_MILESTONES)]
    return (d1, rest[0], rest[1], rest[2], rest[3])


def cumulative_delays(stage_delays: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Cumulative delays ``(C0..C4)`` where ``C_j = D1 + ... + D_{j+1}``."""
    if len(stage_delays) != N_STAGE_DELAYS:
        raise ValueError(f"expected {N_STAGE_DELAYS} stage delays, got {len(stage_delays)}")
    total = 0.0
    out = []
    for i, d in enumerate(stage_delays):
        total += _require_finite(d, f"stage_delays[{i}]")
        out.append(total)
    return tuple(out)  # type: ignore[return-value]


def arrival_delay_h(stage_delays: Sequence[float]) -> float:
    """Arrival delay ``A = D1 + D2`` (cumulative delay at the checkpoint)."""
    return cumulative_delays(stage_delays)[1]


def downstream_delay_h(stage_delays: Sequence[float]) -> float:
    """Downstream delay ``Y = D3 + D4 + D5`` (the model target)."""
    c = cumulative_delays(stage_delays)
    return c[4] - c[1]


def final_delay_h(stage_delays: Sequence[float]) -> float:
    """Final delivery delay ``F = D1 + ... + D5``."""
    return cumulative_delays(stage_delays)[4]


def predicted_eta(scheduled_delivery: datetime, arrival_delay: float, predicted_downstream_h: float) -> datetime:
    """Predicted ETA ``P4 + A + Y_hat`` as a UTC datetime."""
    return add_hours(
        scheduled_delivery,
        _require_finite(arrival_delay, "arrival_delay") + _require_finite(predicted_downstream_h, "predicted_downstream_h"),
    )
