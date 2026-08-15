"""موثوقية/دليل أوامر إحصائية — evidence وليس حكم حذف.

المسار: Raw MBO يبقى · هذه الميزات تُلحق جنبًا إلى جنب.
``deceptive_score`` وملحقاته = evidence؛ النموذج الشرطي يتعلّم فائدتها.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.time import sort_causal
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_START, add_time_bucket
from nq.simulation.deceptive_liquidity import score_deceptive_events

RELIABILITY_COLUMNS = (
    "rel_mean_deceptive_score",
    "rel_credibility",
    "rel_short_life_rate",
    "rel_nonparticipate_rate",
    "rel_spoof_rate",
    "rel_flicker_rate",
    "rel_bait_rate",
    "rel_storm_rate",
    "rel_trade_share",
    "rel_cancel_share",
    "rel_add_share",
    "rel_evidence_strength",
)

_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value


@dataclass(frozen=True, slots=True)
class ReliabilityConfig:
    interval_ns: int = 30 * 1_000_000_000


def attach_reliability_evidence(
    mbo: pl.DataFrame,
    *,
    interval_ns: int | None = None,
    scored: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يجمّع أدلة الموثوقية على براميل ``availability_ts`` بدون إسقاط أحداث."""
    cfg_interval = int(interval_ns) if interval_ns is not None else ReliabilityConfig().interval_ns
    empty = pl.DataFrame(
        schema={AVAILABILITY_TS: pl.Int64(), **{c: pl.Float64() for c in RELIABILITY_COLUMNS}}
    )
    n_mbo = 0 if mbo is None else int(mbo.height)
    if progress is not None:
        progress.op(f"attach_reliability_evidence: mbo={n_mbo:,}")
    if mbo is None or mbo.height == 0:
        if progress is not None:
            progress.op("reliability: empty mbo")
        return empty

    scored_frame = scored if scored is not None else score_deceptive_events(mbo, progress=progress)
    work = sort_causal(scored_frame)
    bucketed = add_time_bucket(work, interval_ns=cfg_interval)
    action = pl.col("action").cast(pl.Utf8)

    def _mean_or_zero(name: str, alias: str) -> pl.Expr:
        if name in bucketed.columns:
            return pl.col(name).mean().alias(alias)
        return pl.lit(0.0).alias(alias)

    agg = (
        bucketed.group_by(BUCKET_START, maintain_order=True)
        .agg(
            pl.col(AVAILABILITY_TS).first().alias(AVAILABILITY_TS),
            pl.col("deceptive_score").mean().alias("rel_mean_deceptive_score")
            if "deceptive_score" in bucketed.columns
            else pl.lit(0.0).alias("rel_mean_deceptive_score"),
            _mean_or_zero("spoof_flag", "rel_spoof_rate"),
            _mean_or_zero("flicker_flag", "rel_short_life_rate"),
            _mean_or_zero("flicker_flag", "rel_flicker_rate"),
            _mean_or_zero("nonparticipate_flag", "rel_nonparticipate_rate"),
            _mean_or_zero("bait_modify_flag", "rel_bait_rate"),
            _mean_or_zero("storm_flag", "rel_storm_rate"),
            action.is_in([_TRADE, _FILL]).mean().alias("rel_trade_share"),
            (action == _CANCEL).mean().alias("rel_cancel_share"),
            (action == _ADD).mean().alias("rel_add_share"),
        )
        .with_columns(
            (1.0 - pl.col("rel_mean_deceptive_score").fill_null(0.0))
            .clip(0.0, 1.0)
            .alias("rel_credibility"),
            (2.0 * pl.col("rel_mean_deceptive_score").fill_null(0.0) - 1.0)
            .abs()
            .alias("rel_evidence_strength"),
        )
        .select(AVAILABILITY_TS, *RELIABILITY_COLUMNS)
        .sort(AVAILABILITY_TS)
    )
    out = agg.with_columns(pl.col(c).fill_null(0.0) for c in RELIABILITY_COLUMNS)
    if progress is not None:
        progress.op(f"reliability bars={out.height:,}")
    return out


__all__ = [
    "RELIABILITY_COLUMNS",
    "ReliabilityConfig",
    "attach_reliability_evidence",
]
