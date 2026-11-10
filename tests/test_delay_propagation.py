"""Unit tests for src.delay_propagation.

Expected calendar values were computed by hand from the IANA rules:
- Europe/Amsterdam: CET (+1) in winter, CEST (+2) from 2025-03-30 01:00 UTC
  until 2025-10-26 01:00 UTC.
- America/New_York: EST (-5) until 2025-03-09 07:00 UTC, then EDT (-4).
- America/Los_Angeles: PST (-8) in January.
- Asia/Dubai: +4 all year. Asia/Singapore: +8 all year.
Weekdays use Python numbering (Monday=0 ... Sunday=6).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src import delay_propagation as dp

UTC = timezone.utc
DEST_TZ = {
    "Rotterdam": "Europe/Amsterdam",
    "Los Angeles": "America/Los_Angeles",
    "Hamburg": "Europe/Berlin",
    "Dubai": "Asia/Dubai",
    "New York": "America/New_York",
    "Singapore": "Asia/Singapore",
}


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


# ---------------------------------------------------------------------------
# UTC / timezone primitives
# ---------------------------------------------------------------------------

class TestTimePrimitives:
    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            dp.ensure_utc(datetime(2025, 1, 1, 12))

    def test_non_datetime_rejected(self):
        with pytest.raises(TypeError):
            dp.ensure_utc("2025-01-01T12:00Z")  # type: ignore[arg-type]

    def test_aware_timestamp_converted_to_utc(self):
        ts = datetime(2025, 1, 1, 12, tzinfo=timezone(timedelta(hours=4)))
        assert dp.ensure_utc(ts) == utc(2025, 1, 1, 8)
        assert dp.ensure_utc(ts).tzinfo is UTC

    @pytest.mark.parametrize("key", ["Europe/Rotterdam", "Mars/Olympus", "", "UTC+1"])
    def test_invalid_timezone_key_rejected(self, key):
        with pytest.raises(ValueError):
            dp.get_zone(key)

    def test_all_destination_keys_valid_and_unique(self):
        assert len(set(DEST_TZ.values())) == len(DEST_TZ)
        for key in DEST_TZ.values():
            assert dp.get_zone(key).key == key

    def test_add_hours_is_absolute_across_dst(self):
        # 00:30 UTC = 01:30 CET; one absolute hour later local clocks read 03:30 CEST.
        start = utc(2025, 3, 30, 0, 30)
        end = dp.add_hours(start, 1.0)
        assert end == utc(2025, 3, 30, 1, 30)
        assert dp.hours_between(start, end) == pytest.approx(1.0)
        assert dp.to_local(start, "Europe/Amsterdam").hour == 1
        assert dp.to_local(end, "Europe/Amsterdam").hour == 3

    def test_add_hours_rejects_non_finite(self):
        with pytest.raises(ValueError):
            dp.add_hours(utc(2025, 1, 1), math.nan)

    def test_hours_between_negative_when_reversed(self):
        assert dp.hours_between(utc(2025, 1, 2), utc(2025, 1, 1)) == pytest.approx(-24.0)


# ---------------------------------------------------------------------------
# Durations, delays, recovery, windows
# ---------------------------------------------------------------------------

class TestDurations:
    def test_normal_positive_deviation(self):
        assert dp.actual_stage_duration_h(48.0, 6.0, 4.0) == pytest.approx(54.0)

    def test_negative_deviation_allowed(self):
        actual = dp.actual_stage_duration_h(48.0, -10.0, 4.0)
        assert actual == pytest.approx(38.0)
        assert dp.stage_delay_h(actual, 48.0) == pytest.approx(-10.0)

    def test_minimum_duration_enforced(self):
        actual = dp.actual_stage_duration_h(10.0, -50.0, 3.0)
        assert actual == pytest.approx(3.0)
        assert dp.stage_delay_h(actual, 10.0) == pytest.approx(-7.0)

    @pytest.mark.parametrize("planned", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_planned_duration(self, planned):
        with pytest.raises(ValueError):
            dp.actual_stage_duration_h(planned, 0.0, 1.0)

    @pytest.mark.parametrize("min_h", [0.0, -2.0])
    def test_invalid_minimum(self, min_h):
        with pytest.raises(ValueError):
            dp.actual_stage_duration_h(10.0, 0.0, min_h)

    def test_non_numeric_rejected(self):
        with pytest.raises(TypeError):
            dp.actual_stage_duration_h("10", 0.0, 1.0)  # type: ignore[arg-type]

    def test_stage_delay_rejects_non_positive_actual(self):
        with pytest.raises(ValueError):
            dp.stage_delay_h(0.0, 10.0)

    def test_recovery_only_for_late_and_capped(self):
        assert dp.recovery_h(20.0, 12.0, 0.5) == pytest.approx(6.0)   # capped by slack
        assert dp.recovery_h(8.0, 12.0, 0.5) == pytest.approx(4.0)
        assert dp.recovery_h(-5.0, 12.0, 0.5) == 0.0                   # early: nothing to recover
        assert dp.recovery_h(10.0, 0.0, 0.5) == 0.0

    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_recovery_fraction_bounds(self, fraction):
        with pytest.raises(ValueError):
            dp.recovery_h(10.0, 5.0, fraction)

    def test_missed_window_strict(self):
        assert dp.missed_window(12.5, 12.0)
        assert not dp.missed_window(12.0, 12.0)
        assert not dp.missed_window(-30.0, 12.0)



class TestTruckWindowWait:
    SLOT = utc(2025, 2, 3, 10)

    def wait(self, lateness_h: float, n: int, window: float = 6.0) -> float:
        return dp.truck_window_wait_h(dp.add_hours(self.SLOT, lateness_h), self.SLOT, window, n)

    @pytest.mark.parametrize("n", [0, 1, 8])
    def test_early_cargo_zero_wait(self, n):
        assert self.wait(-5.0, n) == 0.0

    @pytest.mark.parametrize("lateness", [0.0, 3.0, 5.999])
    @pytest.mark.parametrize("n", [0, 1, 8])
    def test_within_booked_window_zero_wait(self, lateness, n):
        assert self.wait(lateness, n) == 0.0

    def test_exact_window_boundary_belongs_to_next_window(self):
        assert self.wait(6.0, 0) == 0.0
        assert self.wait(6.0, 1) == pytest.approx(6.0)   # window 1 full -> window 2 starts at 12h
        assert self.wait(5.999, 1) == 0.0                # still inside the booked window

    def test_zero_skipped_windows_zero_wait(self):
        assert self.wait(17.5, 0) == 0.0

    @pytest.mark.parametrize(
        "lateness, n, expected",
        [(7.0, 1, 5.0), (7.0, 2, 11.0), (11.5, 1, 0.5), (12.0, 1, 6.0), (13.0, 2, 11.0)],
    )
    def test_skipped_window_waits(self, lateness, n, expected):
        assert self.wait(lateness, n) == pytest.approx(expected)

    def test_bounds_hold(self):
        for lateness in np.linspace(6.0, 60.0, 541):
            for n in range(0, 9):
                w = self.wait(float(lateness), n)
                if n == 0:
                    assert w == 0.0
                else:
                    assert (n - 1) * 6.0 - 1e-9 < w <= n * 6.0 + 1e-9

    def test_maximum_wait_is_48h(self):
        waits = [self.wait(float(x), 8) for x in np.linspace(6.0, 100.0, 941)]
        assert max(waits) <= 48.0 + 1e-9
        assert max(waits) == pytest.approx(48.0, abs=1e-6)

    @pytest.mark.parametrize("n, exc", [(-1, ValueError), (1.0, TypeError), (True, TypeError)])
    def test_invalid_windows_skipped(self, n, exc):
        with pytest.raises(exc):
            dp.truck_window_wait_h(self.SLOT, self.SLOT, 6.0, n)

    def test_invalid_window_and_naive_timestamps(self):
        with pytest.raises(ValueError):
            dp.truck_window_wait_h(self.SLOT, self.SLOT, 0.0, 1)
        with pytest.raises(ValueError):
            dp.truck_window_wait_h(datetime(2025, 2, 3, 12), self.SLOT, 6.0, 1)

    def test_numpy_integer_accepted(self):
        assert self.wait(7.0, np.int64(1)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Customs calendar (destination-local)
# ---------------------------------------------------------------------------

class TestCalendar:
    @pytest.mark.parametrize(
        "ts, tz, expected, wait",
        [
            # Wed 11:00 CET, working hours -> unchanged
            (utc(2025, 1, 8, 10), "Europe/Amsterdam", utc(2025, 1, 8, 10), 0.0),
            # Wed 06:00 CET, before start -> 08:00 CET same day
            (utc(2025, 1, 8, 5), "Europe/Amsterdam", utc(2025, 1, 8, 7), 2.0),
            # Mon exactly 08:00 CET -> unchanged
            (utc(2025, 1, 13, 7), "Europe/Amsterdam", utc(2025, 1, 13, 7), 0.0),
            # Sat 13:00 CET -> Mon 08:00 CET
            (utc(2025, 1, 11, 12), "Europe/Amsterdam", utc(2025, 1, 13, 7), 43.0),
            # Fri 22:00 UTC is already Sat 02:00 in Dubai -> Mon 08:00 Dubai
            (utc(2025, 1, 3, 22), "Asia/Dubai", utc(2025, 1, 6, 4), 54.0),
            # Sat 05:00 UTC is still Fri 21:00 PST in Los Angeles -> working, unchanged
            (utc(2025, 1, 4, 5), "America/Los_Angeles", utc(2025, 1, 4, 5), 0.0),
            # Sun 20:00 UTC = Mon 04:00 SGT -> Mon 08:00 SGT
            (utc(2025, 1, 5, 20), "Asia/Singapore", utc(2025, 1, 6, 0), 4.0),
        ],
    )
    def test_next_working_start_local_time(self, ts, tz, expected, wait):
        assert dp.next_working_start(ts, tz) == expected
        assert dp.non_working_wait_h(ts, tz) == pytest.approx(wait)

    def test_dst_spring_forward_europe(self):
        # Sat 13:00 CET; Monday 08:00 is CEST (+2) -> 06:00 UTC. A fixed +1 offset would give 07:00.
        ts = utc(2025, 3, 29, 12)
        assert dp.next_working_start(ts, "Europe/Amsterdam") == utc(2025, 3, 31, 6)
        assert dp.non_working_wait_h(ts, "Europe/Amsterdam") == pytest.approx(42.0)

    def test_dst_fall_back_europe(self):
        # Sat 14:00 CEST; Monday 08:00 is CET (+1) -> 07:00 UTC.
        ts = utc(2025, 10, 25, 12)
        assert dp.next_working_start(ts, "Europe/Berlin") == utc(2025, 10, 27, 7)
        assert dp.non_working_wait_h(ts, "Europe/Berlin") == pytest.approx(43.0)

    def test_dst_spring_forward_new_york(self):
        # Sat 12:00 EST; Monday 08:00 EDT (-4) -> 12:00 UTC.
        ts = utc(2025, 3, 8, 17)
        assert dp.next_working_start(ts, "America/New_York") == utc(2025, 3, 10, 12)
        assert dp.non_working_wait_h(ts, "America/New_York") == pytest.approx(43.0)

    def test_nonexistent_local_start_time_rejected(self):
        # 02:00 on 2025-03-09 does not exist in New York (clocks jump 02:00 -> 03:00).
        ts = utc(2025, 3, 9, 5)  # Sun 00:00 EST
        with pytest.raises(ValueError, match="does not exist"):
            dp.next_working_start(ts, "America/New_York", non_working_weekdays=frozenset(), workday_start_hour=2)

    @pytest.mark.parametrize(
        "days, hour",
        [(frozenset(range(7)), 8), (frozenset({7}), 8), (frozenset({5, 6}), 24), (frozenset({5, 6}), -1)],
    )
    def test_invalid_calendar_rejected(self, days, hour):
        with pytest.raises(ValueError):
            dp.next_working_start(utc(2025, 1, 8, 10), "Asia/Dubai", days, hour)

    def test_custom_non_working_days(self):
        # Friday-Saturday weekend: Fri 10:00 Dubai (06:00 UTC) -> Sun 08:00 Dubai (04:00 UTC).
        ts = utc(2025, 1, 3, 6)
        assert dp.next_working_start(ts, "Asia/Dubai", frozenset({4, 5})) == utc(2025, 1, 5, 4)

    def test_customs_timing_composition(self):
        ct = dp.customs_timing(utc(2025, 1, 11, 12), 30.0, "Europe/Amsterdam")
        assert ct.processing_start == utc(2025, 1, 13, 7)
        assert ct.release == utc(2025, 1, 14, 13)
        assert ct.wait_h == pytest.approx(43.0)
        assert ct.duration_h == pytest.approx(ct.wait_h + ct.processing_h)

    def test_customs_timing_rejects_zero_processing(self):
        with pytest.raises(ValueError):
            dp.customs_timing(utc(2025, 1, 8, 10), 0.0, "Asia/Dubai")

    def test_customs_timing_rejects_naive(self):
        with pytest.raises(ValueError):
            dp.customs_timing(datetime(2025, 1, 8, 10), 5.0, "Asia/Dubai")

    def test_projected_wait_uses_planned_dwell_only(self):
        # T1 Fri 11:00 CET + planned 24h dwell -> Sat 10:00 UTC -> Mon 07:00 UTC = 45h.
        t1 = utc(2025, 1, 10, 10)
        assert dp.projected_customs_wait_h(t1, 24.0, "Europe/Amsterdam") == pytest.approx(45.0)
        assert dp.projected_customs_wait_h(t1, 2.0, "Europe/Amsterdam") == 0.0


# ---------------------------------------------------------------------------
# Timelines and delay accounting
# ---------------------------------------------------------------------------

def _full_milestones(up: dp.UpstreamTimeline, down: dp.DownstreamTimeline, planned: bool) -> list[datetime]:
    first = up.planned_departure if planned else up.actual_departure
    return [first, *down.milestones]


class TestUpstream:
    def test_backward_construction_identities(self):
        t1 = utc(2025, 6, 15, 9)
        up = dp.upstream_from_checkpoint(t1, departure_delay_h=10.0, planned_transit_h=528.0, actual_transit_h=540.0)
        assert up.transit_delay_h == pytest.approx(12.0)
        assert up.arrival_delay_h == pytest.approx(22.0)
        assert dp.hours_between(up.planned_departure, up.actual_departure) == pytest.approx(10.0)  # D1
        assert dp.hours_between(up.actual_departure, up.actual_arrival) == pytest.approx(540.0)    # a1
        assert dp.hours_between(up.planned_departure, up.planned_arrival) == pytest.approx(528.0)  # p1
        assert dp.hours_between(up.planned_arrival, up.actual_arrival) == pytest.approx(22.0)      # A
        assert up.actual_arrival == t1

    def test_early_arrival(self):
        t1 = utc(2025, 2, 1, 0)
        up = dp.upstream_from_checkpoint(t1, departure_delay_h=-3.0, planned_transit_h=216.0, actual_transit_h=200.0)
        assert up.arrival_delay_h == pytest.approx(-19.0)
        assert up.planned_arrival > up.actual_arrival
        assert up.actual_departure < up.planned_departure

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            dp.upstream_from_checkpoint(utc(2025, 1, 1), 0.0, 100.0, 0.0)
        with pytest.raises(ValueError):
            dp.upstream_from_checkpoint(datetime(2025, 1, 1), 0.0, 100.0, 100.0)


class TestAccounting:
    def _build(self, t1, tz, plan, actual, d1=4.0, p1=432.0, a1=444.0):
        up = dp.upstream_from_checkpoint(t1, d1, p1, a1)
        plan_down = dp.downstream_timeline(up.planned_arrival, *plan, tz)
        act_down = dp.downstream_timeline(up.actual_arrival, *actual, tz)
        return up, plan_down, act_down

    def test_telescoping_and_single_attribution(self):
        up, pdn, adn = self._build(
            utc(2025, 1, 9, 12), "Europe/Amsterdam", plan=(48.0, 24.0, 36.0), actual=(70.0, 40.0, 30.0)
        )
        P = _full_milestones(up, pdn, planned=True)
        T = _full_milestones(up, adn, planned=False)
        d = dp.stage_delays_from_milestones(P, T)
        c = dp.cumulative_delays(d)

        for j in range(5):  # C_j = T_j - P_j
            assert c[j] == pytest.approx(dp.hours_between(P[j], T[j]))
        assert dp.arrival_delay_h(d) == pytest.approx(up.arrival_delay_h)
        assert dp.final_delay_h(d) == pytest.approx(dp.hours_between(P[4], T[4]))
        assert dp.downstream_delay_h(d) == pytest.approx(c[4] - c[1])
        # Port stage carries only the dwell deviation; customs waits stay in D4.
        assert d[2] == pytest.approx(70.0 - 48.0)
        assert d[3] == pytest.approx((adn.customs.wait_h - pdn.customs.wait_h) + (40.0 - 24.0))
        assert d[4] == pytest.approx(30.0 - 36.0)
        # Stage durations add up to the actual elapsed time with nothing counted twice.
        assert sum(adn.stage_durations_h) == pytest.approx(dp.hours_between(T[1], T[4]))

    def test_planned_weekend_is_not_delay(self):
        # Plan and actual both lodge on Saturday; actual is 3h later but still before Monday.
        # Same processing time -> same release, so the 3h upstream lateness is absorbed in D4.
        tz = "Europe/Amsterdam"
        p_arrival = utc(2025, 1, 10, 20)  # Fri 21:00 CET
        plan = dp.downstream_timeline(p_arrival, 12.0, 10.0, 20.0, tz)       # lodged Sat 08:00 UTC
        act = dp.downstream_timeline(dp.add_hours(p_arrival, 3.0), 12.0, 10.0, 20.0, tz)
        assert plan.customs.wait_h > 0
        assert act.customs_release == plan.customs_release
        assert act.delivery == plan.delivery
        d4 = act.stage_durations_h[1] - plan.stage_durations_h[1]
        assert d4 == pytest.approx(-3.0)

    def test_lateness_causes_weekend_shift(self):
        # Plan lodges Fri 10:00 CET (no wait); 20h of extra port dwell pushes lodgement to Sat.
        tz = "Europe/Amsterdam"
        p_arrival = utc(2025, 1, 9, 9)  # Thu 10:00 CET
        plan = dp.downstream_timeline(p_arrival, 24.0, 8.0, 12.0, tz)
        act = dp.downstream_timeline(p_arrival, 44.0, 8.0, 12.0, tz)
        assert plan.customs.wait_h == 0.0
        assert act.customs.wait_h == pytest.approx(dp.hours_between(utc(2025, 1, 11, 5), utc(2025, 1, 13, 7)))
        P = [dp.add_hours(p_arrival, -100.0), *plan.milestones]
        T = [dp.add_hours(p_arrival, -100.0), *act.milestones]
        d = dp.stage_delays_from_milestones(P, T)
        assert d[2] == pytest.approx(20.0)                     # port
        assert d[3] == pytest.approx(act.customs.wait_h)       # customs: weekend wait only
        assert d[4] == pytest.approx(0.0)                      # inland
        assert dp.final_delay_h(d) == pytest.approx(20.0 + act.customs.wait_h)

    def test_milestones_must_be_strictly_increasing(self):
        base = [utc(2025, 1, 1) + timedelta(days=i) for i in range(5)]
        bad = list(base)
        bad[2] = bad[1]
        with pytest.raises(ValueError, match="strictly increasing"):
            dp.stage_delays_from_milestones(base, bad)
        with pytest.raises(ValueError):
            dp.stage_delays_from_milestones(base[:4], base[:4])

    def test_cumulative_requires_five_values(self):
        with pytest.raises(ValueError):
            dp.cumulative_delays([1.0, 2.0])

    def test_predicted_eta(self):
        eta = dp.predicted_eta(utc(2025, 3, 1, 12), arrival_delay=10.0, predicted_downstream_h=14.0)
        assert eta == utc(2025, 3, 2, 12)


def test_randomised_timelines_are_consistent():
    """Property check over many random shipments with an explicitly seeded generator."""
    rng = np.random.default_rng(20250101)
    tzs = list(DEST_TZ.values())
    start = utc(2024, 1, 1)
    for _ in range(500):
        tz = tzs[rng.integers(len(tzs))]
        t1 = dp.add_hours(start, float(rng.uniform(0, 731 * 24)))
        p = [float(rng.uniform(200, 840)), float(rng.uniform(24, 120)),
             float(rng.uniform(8, 72)), float(rng.uniform(6, 96))]
        mins = [0.6 * p[0], 4.0, 2.0, 2.0]
        devs = rng.normal(0.0, [48.0, 30.0, 24.0, 24.0]) - rng.uniform(0, 200, size=4) * rng.integers(0, 2, size=4)
        a = [dp.actual_stage_duration_h(p[k], float(devs[k]), mins[k]) for k in range(4)]
        d1 = float(rng.normal(0, 12))

        up = dp.upstream_from_checkpoint(t1, d1, p[0], a[0])
        plan = dp.downstream_timeline(up.planned_arrival, p[1], p[2], p[3], tz)
        act = dp.downstream_timeline(up.actual_arrival, a[1], a[2], a[3], tz)
        P = _full_milestones(up, plan, planned=True)
        T = _full_milestones(up, act, planned=False)

        for k in range(1, 5):  # positive actual and planned durations
            assert T[k] > T[k - 1]
            assert P[k] > P[k - 1]
        assert all(ak >= mk for ak, mk in zip(a, mins))

        d = dp.stage_delays_from_milestones(P, T)
        assert d[0] == pytest.approx(d1, abs=1e-6)
        assert d[1] == pytest.approx(a[0] - p[0], abs=1e-6)
        assert d[2] == pytest.approx(a[1] - p[1], abs=1e-6)
        assert dp.arrival_delay_h(d) == pytest.approx(up.arrival_delay_h, abs=1e-6)
        assert dp.final_delay_h(d) == pytest.approx(dp.hours_between(P[4], T[4]), abs=1e-6)
        assert dp.final_delay_h(d) == pytest.approx(
            dp.arrival_delay_h(d) + dp.downstream_delay_h(d), abs=1e-9
        )
        # Customs processing never starts on a local non-working day or before 08:00.
        local_start = dp.to_local(act.customs.processing_start, tz)
        assert local_start.weekday() not in dp.DEFAULT_NON_WORKING_WEEKDAYS
        assert local_start.hour >= dp.DEFAULT_WORKDAY_START_HOUR
