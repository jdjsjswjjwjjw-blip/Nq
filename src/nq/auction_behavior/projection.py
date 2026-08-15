"""إسقاط سببي لملف آسيا المكتمل على مزاد لندن المتطوّر.

يبقى ملف آسيا مرساة ثابتة، بينما يمتد ملف مركّب من بداية آسيا إلى نهاية كل
برميل مكتمل في لندن. لا يوجد تصفير عند انتقال آسيا→لندن، ولا تُنسب نتيجة
برميل لندن إلى وقت أسبق من ``bucket_end``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.session import (
    VP_LIQUIDITY_SESSION,
    VpLiquiditySession,
    session_date_from_ns,
    vp_liquidity_session_from_ns,
)
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades
from nq.simulation.volume_profile import DevelopingVolumeProfile, ValueArea, classify_nodes

_DEFAULT_INTERVAL_NS = 3 * 60 * 1_000_000_000
_NQ_TICK_FIXED = float(round(0.25 / PRICE_SCALE))

PROJECTION_NUMERIC_COLUMNS = (
    "proj_poc_shift_ticks",
    "proj_hvn_shift_ticks",
    "proj_poc_step_ticks",
    "proj_hvn_step_ticks",
    "proj_hvn_retention_ratio",
    "proj_migration_speed_ticks",
    "proj_va_overlap",
    "proj_va_center_shift_ticks",
    "proj_va_width_ratio",
    "proj_break_level_distance_ticks",
    "proj_asia_coverage_ratio",
    "proj_anchor_complete",
    "proj_outside_volume_share",
    "proj_anchor_hvn_retained",
    "proj_anchor_stable",
    "proj_expansion_active",
    "proj_expansion_testing",
    "proj_expansion_accepting",
    "proj_repriced_balance",
    "proj_value_transferred",
    "proj_rejection_to_asia",
    "proj_break_direction",
)


@dataclass(frozen=True, slots=True)
class AsiaLondonProjectionConfig:
    """عتبات وصف المزاد المركب، بوحدة تيكات NQ عندما يلزم."""

    interval_ns: int = _DEFAULT_INTERVAL_NS
    value_fraction: float = 0.7
    hvn_tolerance_ticks: float = 4.0
    anchor_stable_ticks: float = 4.0
    acceptance_migration_ticks: float = 4.0
    repricing_ticks: float = 12.0
    overlap_high: float = 0.60
    overlap_low: float = 0.35
    outside_acceptance_share: float = 0.55
    stability_window: int = 3
    repricing_confirm_buckets: int = 2
    min_asia_coverage_ratio: float = 0.80

    def __post_init__(self) -> None:
        if self.interval_ns < 1:
            raise ValueError("interval_ns must be >= 1")
        if not 0.0 < self.value_fraction <= 1.0:
            raise ValueError("value_fraction must be in (0, 1]")
        if self.hvn_tolerance_ticks < 0 or self.anchor_stable_ticks < 0:
            raise ValueError("tick tolerances must be non-negative")
        if self.acceptance_migration_ticks < 0 or self.repricing_ticks < 0:
            raise ValueError("migration thresholds must be non-negative")
        if not 0.0 <= self.overlap_low <= self.overlap_high <= 1.0:
            raise ValueError("require 0 <= overlap_low <= overlap_high <= 1")
        if not 0.0 <= self.outside_acceptance_share <= 1.0:
            raise ValueError("outside_acceptance_share must be in [0, 1]")
        if self.stability_window < 1 or self.repricing_confirm_buckets < 1:
            raise ValueError("stability windows must be >= 1")
        if not 0.0 <= self.min_asia_coverage_ratio <= 1.0:
            raise ValueError("min_asia_coverage_ratio must be in [0, 1]")


def _empty_projection() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            AVAILABILITY_TS: pl.Int64(),
            BUCKET_START: pl.Int64(),
            BUCKET_END: pl.Int64(),
            "projection_story_date": pl.Utf8(),
            "projection_stage": pl.Utf8(),
            "auction_phase": pl.Utf8(),
            "close": pl.Int64(),
            "asia_poc": pl.Int64(),
            "asia_vah": pl.Int64(),
            "asia_val": pl.Int64(),
            "asia_primary_hvn": pl.Int64(),
            "composite_poc": pl.Int64(),
            "composite_vah": pl.Int64(),
            "composite_val": pl.Int64(),
            "composite_primary_hvn": pl.Int64(),
            "composite_total_volume": pl.Int64(),
            **{c: pl.Float64() for c in PROJECTION_NUMERIC_COLUMNS},
        }
    )


def _primary_hvn(profile: DevelopingVolumeProfile, area: ValueArea) -> tuple[int, list[int]]:
    nodes = classify_nodes(profile.to_frame())
    hvn = nodes.filter(pl.col("is_hvn"))
    # العقدة الأساسية = أعلى حجم عالميًا حتى لو وقعت على حافة التوزيع؛
    # classify_nodes يستبعد الحواف لأنها بلا جارَين، وهذا لا يجعلها غير مهمة.
    primary = nodes.with_columns(
        (pl.col("price") - pl.lit(area.poc)).abs().alias("_poc_distance")
    ).sort(["volume", "_poc_distance", "price"], descending=[True, False, False])
    primary_price = int(primary["price"][0])
    prices = {int(x) for x in hvn["price"].to_list()}
    prices.add(primary_price)
    return primary_price, sorted(prices)


def _overlap(a_val: int, a_vah: int, b_val: int, b_vah: int) -> float:
    intersection = max(0, min(a_vah, b_vah) - max(a_val, b_val))
    union = max(a_vah, b_vah) - min(a_val, b_val)
    return 1.0 if union <= 0 else float(intersection / union)


def _break_direction(high: int, low: int, close: int, anchor: ValueArea) -> int:
    above = high > anchor.vah
    below = low < anchor.val
    if above and not below:
        return 1
    if below and not above:
        return -1
    if not above and not below:
        return 0
    if close > anchor.vah:
        return 1
    if close < anchor.val:
        return -1
    up_excursion = high - anchor.vah
    down_excursion = anchor.val - low
    return 1 if up_excursion >= down_excursion else -1


def _phase(
    *,
    break_direction: int,
    close: int,
    anchor: ValueArea,
    composite: ValueArea,
    anchor_stable: bool,
    accepting: bool,
    transferred: bool,
) -> str:
    if break_direction == 0:
        return "london_prebreak"
    returned = anchor.val <= close <= anchor.vah
    if returned and anchor_stable:
        return "rejection_return_to_asia"
    if transferred and composite.val <= close <= composite.vah:
        return "repriced_balance"
    if accepting:
        return "expansion_accepting"
    return "expansion_testing"


def _bucket_groups(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    trades = extract_trades(add_time_bucket(frame, interval_ns=interval_ns)).sort(EVENT_TS)
    if trades.height == 0:
        return trades
    times = [int(x) for x in trades[EVENT_TS].to_list()]
    n = len(times)
    if progress is not None:
        progress.op(f"projection: session labels for {n:,} trades")
    dates: list[str] = []
    sessions: list[int] = []
    for i, ts in enumerate(times, start=1):
        dates.append(session_date_from_ns(ts))
        sessions.append(int(vp_liquidity_session_from_ns(ts)))
        if progress is not None:
            progress.heartbeat(i, n, label="projection-session-labels")
    return trades.with_columns(
        pl.Series("projection_story_date", dates, dtype=pl.Utf8),
        pl.Series(VP_LIQUIDITY_SESSION, sessions, dtype=pl.Int8),
    )


def _asia_row(
    *,
    bucket_start: int,
    bucket_end: int,
    story_date: str,
    close: int,
    area: ValueArea,
    primary_hvn: int,
) -> dict[str, Any]:
    return {
        AVAILABILITY_TS: bucket_end,
        BUCKET_START: bucket_start,
        BUCKET_END: bucket_end,
        "projection_story_date": story_date,
        "projection_stage": "asia_build",
        "auction_phase": "asia_build",
        "close": close,
        "asia_poc": None,
        "asia_vah": None,
        "asia_val": None,
        "asia_primary_hvn": None,
        "composite_poc": area.poc,
        "composite_vah": area.vah,
        "composite_val": area.val,
        "composite_primary_hvn": primary_hvn,
        "composite_total_volume": area.total_volume,
        **{c: 0.0 for c in PROJECTION_NUMERIC_COLUMNS},
    }


def build_asia_london_projection(  # noqa: PLR0912, PLR0915
    mbo: pl.DataFrame,
    *,
    config: AsiaLondonProjectionConfig | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يبني ملفًا مركبًا Asia→London ويصنف انتقال القيمة دون قرارات تداول."""
    cfg = config or AsiaLondonProjectionConfig()
    if progress is not None:
        progress.op(
            f"build_asia_london_projection: mbo={mbo.height:,} interval_ns={cfg.interval_ns}"
        )
    trades = _bucket_groups(mbo, interval_ns=cfg.interval_ns, progress=progress)
    if trades.height == 0:
        if progress is not None:
            progress.op("projection: no trades")
        return _empty_projection()

    asia_id = int(VpLiquiditySession.ASIA)
    london_id = int(VpLiquiditySession.LONDON)
    stories = list(trades.group_by("projection_story_date", maintain_order=True))
    n_stories = len(stories)
    packed: list[tuple[str, pl.DataFrame, list[tuple[object, pl.DataFrame]]]] = []
    n_buckets = 0
    n_trades = 0
    for story_key, story in stories:
        story_date = str(story_key[0] if isinstance(story_key, tuple) else story_key)
        relevant = story.filter(pl.col(VP_LIQUIDITY_SESSION).is_in([asia_id, london_id]))
        buckets = list(relevant.group_by(BUCKET_START, maintain_order=True))
        n_buckets += len(buckets)
        n_trades += int(relevant.height)
        packed.append((story_date, relevant, buckets))
    if progress is not None:
        progress.op(
            f"projection stories={n_stories} buckets={n_buckets:,} asia_london_trades={n_trades:,}"
        )
    rows: list[dict[str, Any]] = []
    story_i = 0
    bucket_i = 0
    trade_i = 0
    for story_date, relevant, buckets in packed:
        story_i += 1
        if progress is not None:
            progress.heartbeat(story_i, n_stories, label="projection-stories")
        if relevant.height == 0:
            continue
        running = DevelopingVolumeProfile(fraction=cfg.value_fraction)
        anchor: ValueArea | None = None
        anchor_primary = 0
        anchor_hvns: list[int] = []
        break_dir = 0
        london_volume = 0
        london_above = 0
        london_below = 0
        prev_poc: int | None = None
        prev_hvn: int | None = None
        migration_steps: list[float] = []
        repricing_stable = 0
        asia_bucket_count = 0
        asia_coverage = 0.0
        anchor_complete = False

        for bucket_key, raw_bucket in buckets:
            bucket_i += 1
            if progress is not None:
                progress.heartbeat(bucket_i, max(n_buckets, 1), label="projection-buckets")
            bucket_start = int(bucket_key[0] if isinstance(bucket_key, tuple) else bucket_key)
            bucket = raw_bucket.sort(EVENT_TS)
            session_id = int(bucket[VP_LIQUIDITY_SESSION][0])
            if session_id == london_id and anchor is None:
                anchor = running.value_area()
                expected_asia_buckets = max(
                    1,
                    round((9 * 60 * 60 * 1_000_000_000) / cfg.interval_ns),
                )
                asia_coverage = min(1.0, asia_bucket_count / expected_asia_buckets)
                anchor_complete = asia_coverage >= cfg.min_asia_coverage_ratio
                if anchor is not None:
                    anchor_primary, anchor_hvns = _primary_hvn(running, anchor)
            for price, size in bucket.select("price", "size").iter_rows():
                trade_i += 1
                if progress is not None:
                    progress.heartbeat(trade_i, max(n_trades, 1), label="projection-trades")
                px, sz = int(price), int(size)
                running.add_trade(px, sz)
                if session_id == london_id and anchor is not None:
                    london_volume += sz
                    london_above += sz if px > anchor.vah else 0
                    london_below += sz if px < anchor.val else 0
            area = running.value_area()
            if area is None:  # pragma: no cover
                continue
            primary_hvn, current_hvns = _primary_hvn(running, area)
            close = int(bucket["price"][-1])
            high_value = bucket["price"].max()
            low_value = bucket["price"].min()
            if high_value is None or low_value is None:  # pragma: no cover
                continue
            high = int(np.asarray(high_value).item())
            low = int(np.asarray(low_value).item())
            bucket_end = int(bucket[BUCKET_END][0])
            if session_id == asia_id:
                asia_bucket_count += 1
                rows.append(
                    _asia_row(
                        bucket_start=bucket_start,
                        bucket_end=bucket_end,
                        story_date=story_date,
                        close=close,
                        area=area,
                        primary_hvn=primary_hvn,
                    )
                )
                prev_poc, prev_hvn = area.poc, primary_hvn
                continue
            if anchor is None:
                continue

            if break_dir == 0:
                break_dir = _break_direction(high, low, close, anchor) if anchor_complete else 0
            poc_shift = (area.poc - anchor.poc) / _NQ_TICK_FIXED
            hvn_shift = (primary_hvn - anchor_primary) / _NQ_TICK_FIXED
            poc_step = 0.0 if prev_poc is None else (area.poc - prev_poc) / _NQ_TICK_FIXED
            hvn_step = 0.0 if prev_hvn is None else (primary_hvn - prev_hvn) / _NQ_TICK_FIXED
            prev_poc, prev_hvn = area.poc, primary_hvn
            migration_steps.append(abs(float(poc_step)))
            recent = migration_steps[-cfg.stability_window :]
            migration_speed = float(np.mean(recent)) if recent else 0.0
            va_overlap = _overlap(anchor.val, anchor.vah, area.val, area.vah)
            outside = london_above if break_dir > 0 else london_below if break_dir < 0 else 0
            outside_share = float(outside / london_volume) if london_volume > 0 else 0.0
            tolerance = cfg.hvn_tolerance_ticks * _NQ_TICK_FIXED
            retained_count = sum(
                any(abs(current - asia_hvn) <= tolerance for current in current_hvns)
                for asia_hvn in anchor_hvns
            )
            retention_ratio = retained_count / max(1, len(anchor_hvns))
            retained = retained_count > 0
            anchor_center = (anchor.vah + anchor.val) * 0.5
            composite_center = (area.vah + area.val) * 0.5
            va_center_shift = (composite_center - anchor_center) / _NQ_TICK_FIXED
            anchor_width = max(anchor.vah - anchor.val, _NQ_TICK_FIXED)
            va_width_ratio = float((area.vah - area.val) / anchor_width)
            break_level = anchor.vah if break_dir > 0 else anchor.val
            break_distance = (
                ((close - break_level) * break_dir) / _NQ_TICK_FIXED if break_dir != 0 else 0.0
            )
            anchor_stable = bool(
                abs(poc_shift) <= cfg.anchor_stable_ticks
                and abs(hvn_shift) <= cfg.anchor_stable_ticks
                and va_overlap >= cfg.overlap_high
            )
            direction_migration = poc_shift * break_dir
            direction_hvn = hvn_shift * break_dir
            accepting = bool(
                break_dir != 0
                and direction_migration >= cfg.acceptance_migration_ticks
                and direction_hvn >= 0.0
                and outside_share >= cfg.outside_acceptance_share
            )
            shifted = bool(
                break_dir != 0
                and direction_migration >= cfg.repricing_ticks
                and direction_hvn >= cfg.acceptance_migration_ticks
                and va_overlap <= cfg.overlap_low
            )
            step_stable = bool(
                abs(poc_step) <= cfg.anchor_stable_ticks
                and abs(hvn_step) <= cfg.anchor_stable_ticks
            )
            repricing_stable = repricing_stable + 1 if shifted and step_stable else 0
            transferred = repricing_stable >= cfg.repricing_confirm_buckets
            phase = (
                _phase(
                    break_direction=break_dir,
                    close=close,
                    anchor=anchor,
                    composite=area,
                    anchor_stable=anchor_stable,
                    accepting=accepting,
                    transferred=transferred,
                )
                if anchor_complete
                else "incomplete_asia_anchor"
            )
            rows.append(
                {
                    AVAILABILITY_TS: bucket_end,
                    BUCKET_START: bucket_start,
                    BUCKET_END: bucket_end,
                    "projection_story_date": story_date,
                    "projection_stage": "london_extend",
                    "auction_phase": phase,
                    "close": close,
                    "asia_poc": anchor.poc,
                    "asia_vah": anchor.vah,
                    "asia_val": anchor.val,
                    "asia_primary_hvn": anchor_primary,
                    "composite_poc": area.poc,
                    "composite_vah": area.vah,
                    "composite_val": area.val,
                    "composite_primary_hvn": primary_hvn,
                    "composite_total_volume": area.total_volume,
                    "proj_poc_shift_ticks": float(poc_shift),
                    "proj_hvn_shift_ticks": float(hvn_shift),
                    "proj_poc_step_ticks": float(poc_step),
                    "proj_hvn_step_ticks": float(hvn_step),
                    "proj_hvn_retention_ratio": float(retention_ratio),
                    "proj_migration_speed_ticks": migration_speed,
                    "proj_va_overlap": va_overlap,
                    "proj_va_center_shift_ticks": float(va_center_shift),
                    "proj_va_width_ratio": va_width_ratio,
                    "proj_break_level_distance_ticks": float(break_distance),
                    "proj_asia_coverage_ratio": float(asia_coverage),
                    "proj_anchor_complete": float(anchor_complete),
                    "proj_outside_volume_share": outside_share,
                    "proj_anchor_hvn_retained": float(retained),
                    "proj_anchor_stable": float(anchor_stable),
                    "proj_expansion_active": float(phase.startswith("expansion_")),
                    "proj_expansion_testing": float(phase == "expansion_testing"),
                    "proj_expansion_accepting": float(phase == "expansion_accepting"),
                    "proj_repriced_balance": float(phase == "repriced_balance"),
                    "proj_value_transferred": float(transferred),
                    "proj_rejection_to_asia": float(phase == "rejection_return_to_asia"),
                    "proj_break_direction": float(break_dir),
                }
            )
    if not rows:
        if progress is not None:
            progress.op("projection: no asia/london rows")
        return _empty_projection()
    out = pl.DataFrame(rows, schema=_empty_projection().schema, strict=False).sort(AVAILABILITY_TS)
    if progress is not None:
        progress.op(
            f"projection done: rows={out.height:,} stories={n_stories} buckets={bucket_i:,}"
        )
    return out


__all__ = [
    "PROJECTION_NUMERIC_COLUMNS",
    "AsiaLondonProjectionConfig",
    "build_asia_london_projection",
]
