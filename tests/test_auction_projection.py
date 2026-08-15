"""اختبارات ملف آسيا→لندن الممتد وانتقال القيمة."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from nq.auction_behavior.events import build_behavior_events
from nq.auction_behavior.projection import (
    AsiaLondonProjectionConfig,
    build_asia_london_projection,
)
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_TICK = round(0.25 / PRICE_SCALE)
_BASE = round(20_000.0 / PRICE_SCALE)
_MINUTE = 60 * 1_000_000_000


def _ns_et(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    local = datetime(year, month, day, hour, minute, tzinfo=_ET)
    return int(local.astimezone(UTC).timestamp() * 1_000_000_000)


def _asia_london_repricing_stream() -> pl.DataFrame:
    events: list[tuple[str, str, int, int, int]] = []
    times: list[int] = []
    asia = _ns_et(2024, 6, 3, 20, 0)
    london = _ns_et(2024, 6, 4, 3, 0)
    # ثلاثة براميل آسيا: ملف ضيق ومكتمل حول 20,000.
    for bar in range(3):
        for j, (offset, size) in enumerate(((-1, 15), (0, 50), (1, 15))):
            events.append(("T", "B" if j % 2 else "A", _BASE + offset * _TICK, size, 0))
            times.append(asia + bar * 3 * _MINUTE + j * 1_000_000)
    # لندن: اختبار خفيف، ثم حجم ينتقل 20 تيك للأعلى ويثبت هناك.
    london_buckets = ((5, 5), (20, 180), (20, 140), (20, 140))
    for bar, (offset, size) in enumerate(london_buckets):
        for j in range(3):
            events.append(("T", "B", _BASE + offset * _TICK, size, 0))
            times.append(london + bar * 3 * _MINUTE + j * 1_000_000)
    return make_stream(events, event_ts=times, sequence=list(range(1, len(events) + 1)))


def _config() -> AsiaLondonProjectionConfig:
    return AsiaLondonProjectionConfig(
        interval_ns=3 * _MINUTE,
        anchor_stable_ticks=4.0,
        acceptance_migration_ticks=4.0,
        repricing_ticks=8.0,
        overlap_high=0.6,
        overlap_low=0.5,
        outside_acceptance_share=0.5,
        repricing_confirm_buckets=2,
        min_asia_coverage_ratio=0.0,
    )


def test_projection_keeps_asia_anchor_and_extends_without_reset() -> None:
    projection = build_asia_london_projection(_asia_london_repricing_stream(), config=_config())
    asia = projection.filter(pl.col("projection_stage") == "asia_build")
    london = projection.filter(pl.col("projection_stage") == "london_extend")
    assert asia.height == 3
    assert london.height == 4
    assert london["asia_poc"].n_unique() == 1
    assert london["asia_vah"].n_unique() == 1
    assert london["asia_val"].n_unique() == 1
    # الملف المركب لا يبدأ من صفر في لندن؛ حجمه يتضمن آسيا ثم يزداد فقط.
    asia_total = int(asia["composite_total_volume"][-1])
    london_totals = london["composite_total_volume"].to_list()
    assert london_totals[0] > asia_total
    assert london_totals == sorted(london_totals)


def test_projection_distinguishes_testing_acceptance_and_repriced_balance() -> None:
    projection = build_asia_london_projection(_asia_london_repricing_stream(), config=_config())
    london = projection.filter(pl.col("projection_stage") == "london_extend")
    phases = london["auction_phase"].to_list()
    assert phases[0] == "expansion_testing"
    assert "expansion_accepting" in phases
    assert phases[-1] == "repriced_balance"
    assert london["proj_expansion_active"][0] == 1.0
    assert london["proj_break_level_distance_ticks"][0] > 0.0
    assert london["proj_value_transferred"][-1] == 1.0
    assert london["proj_poc_shift_ticks"][-1] >= 8.0
    assert 0.0 <= london["proj_hvn_retention_ratio"][-1] <= 1.0
    assert london["proj_va_center_shift_ticks"][-1] > 0.0
    event_source = london.with_columns(pl.lit(1).alias("_story"))
    events = build_behavior_events(event_source, group_col="_story")
    assert events["evt_expansion_accepting"].sum() == 1.0
    assert events["evt_expansion_continue"].sum() == 1.0
    assert events["evt_value_transfer"].sum() == 1.0


@pytest.mark.leakage
def test_future_london_volume_does_not_change_prior_projection() -> None:
    source = _asia_london_repricing_stream()
    baseline = build_asia_london_projection(source, config=_config())
    cutoff = int(baseline[AVAILABILITY_TS][-2])
    changed = source.with_columns(
        pl.when(pl.col("event_ts") > cutoff)
        .then(pl.col("price") + 80 * _TICK)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    perturbed = build_asia_london_projection(changed, config=_config())
    cols = [
        AVAILABILITY_TS,
        "composite_poc",
        "composite_vah",
        "composite_val",
        "auction_phase",
        "proj_poc_shift_ticks",
    ]
    assert (
        baseline.filter(pl.col(AVAILABILITY_TS) <= cutoff)
        .select(cols)
        .equals(perturbed.filter(pl.col(AVAILABILITY_TS) <= cutoff).select(cols))
    )


def test_projection_rejects_invalid_thresholds_and_empty_input() -> None:
    with pytest.raises(ValueError, match="overlap"):
        AsiaLondonProjectionConfig(overlap_low=0.8, overlap_high=0.5)
    empty = build_asia_london_projection(make_stream([]), config=_config())
    assert empty.height == 0
    assert "auction_phase" in empty.columns


def test_projection_schema_survives_more_than_100_asia_buckets() -> None:
    asia = _ns_et(2024, 6, 3, 18, 0)
    london = _ns_et(2024, 6, 4, 3, 0)
    events = [("T", "B", _BASE + (i % 3) * _TICK, 2, 0) for i in range(106)]
    times = [asia + i * 3 * _MINUTE for i in range(106)]
    events.append(("T", "B", _BASE + 8 * _TICK, 3, 0))
    times.append(london)
    frame = make_stream(events, event_ts=times, sequence=list(range(1, len(events) + 1)))
    projection = build_asia_london_projection(frame, config=_config())
    london_rows = projection.filter(pl.col("projection_stage") == "london_extend")
    assert projection.height == 107
    assert london_rows.height == 1
    assert london_rows["asia_poc"][0] is not None


def test_default_marks_partial_asia_as_incomplete_anchor() -> None:
    projection = build_asia_london_projection(_asia_london_repricing_stream())
    london = projection.filter(pl.col("projection_stage") == "london_extend")
    assert london.height > 0
    assert set(london["auction_phase"].to_list()) == {"incomplete_asia_anchor"}
    assert london["proj_anchor_complete"].sum() == 0.0
    assert london["proj_expansion_active"].sum() == 0.0


def _dense_truncated_asia_london_stream() -> pl.DataFrame:
    """آسيا كثيفة من 20:00 ET (ملف مقصوص) حتى 03:00، ثم لندن قصيرة حتى 05:00."""
    events: list[tuple[str, str, int, int, int]] = []
    times: list[int] = []
    asia = _ns_et(2024, 6, 3, 20, 0)
    london = _ns_et(2024, 6, 4, 3, 0)
    london_end = _ns_et(2024, 6, 4, 5, 0)
    t = asia
    while t < london:
        events.append(("T", "B", _BASE, 2, 0))
        times.append(t)
        t += 3 * _MINUTE
    t = london
    while t < london_end:
        events.append(("T", "B", _BASE + 8 * _TICK, 3, 0))
        times.append(t)
        t += 3 * _MINUTE
    return make_stream(events, event_ts=times, sequence=list(range(1, len(events) + 1)))


def test_truncated_file_window_completes_dense_asia_anchor() -> None:
    projection = build_asia_london_projection(_dense_truncated_asia_london_stream())
    london = projection.filter(pl.col("projection_stage") == "london_extend")
    assert london.height > 0
    assert float(london["proj_asia_coverage_ratio"][0]) >= 0.80
    assert float(london["proj_anchor_complete"][0]) == 1.0
    assert "incomplete_asia_anchor" not in set(london["auction_phase"].to_list())
    assert int(london[AVAILABILITY_TS][-1]) < _ns_et(2024, 6, 4, 9, 30)
