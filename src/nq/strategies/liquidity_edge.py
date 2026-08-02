"""توافق خلفي: إدج السيولة = امتداد داخل ``vp_auction`` (ليست استراتيجية منفصلة).

``run_liquidity_edge_research`` يستدعي ``run_vp_auction_research(with_execution=True)``
ويُرجع غلافًا بنفس الحقول القديمة حتى لا يتكسر الاستدعاء الخارجي.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nq.simulation.deceptive_liquidity import DeceptiveLiquidityConfig
from nq.simulation.edge_execution_plan import EdgeSearchSpec
from nq.strategies.vp_auction import VpAuctionResearchResult, run_vp_auction_research


@dataclass(frozen=True, slots=True)
class LiquidityEdgeResult:
    """غلاف توافق — المصدر الحقيقي: ``VpAuctionResearchResult``."""

    cleaned_mbo_rows: int
    raw_mbo_rows: int
    search_table: pl.DataFrame
    best_spec: EdgeSearchSpec | None
    best_row: dict[str, float | str]
    trades: pl.DataFrame
    trade_summary: dict[str, float]
    report_md: str
    vp: VpAuctionResearchResult


def run_liquidity_edge_research(
    nq: pl.DataFrame | str | Path,
    *,
    interval_ns: int = 1_000_000_000,
    max_rows: int | None = None,
    train_frac: float = 0.6,
    min_oos_trades: int = 3,
    min_oos_rr: float = 2.0,
    drop_deceptive: bool = True,
    deceptive: DeceptiveLiquidityConfig | None = None,
    grid: tuple[EdgeSearchSpec, ...] | None = None,
    output_dir: Path | str | None = None,
    quiet: bool = False,
) -> LiquidityEdgeResult:
    """يفوّض بالكامل لاستراتيجية Volume Profile المتصلة."""
    vp = run_vp_auction_research(
        nq,
        max_rows=max_rows,
        output_dir=output_dir,
        quiet=quiet,
        with_execution=True,
        drop_deceptive=drop_deceptive,
        deceptive=deceptive,
        edge_grid=grid,
        edge_train_frac=train_frac,
        min_oos_trades=min_oos_trades,
        min_oos_rr=min_oos_rr,
        interval_ns=interval_ns,
        n_permutations=100,
        n_splits=2,
    )
    return LiquidityEdgeResult(
        cleaned_mbo_rows=vp.cleaned_mbo_rows,
        raw_mbo_rows=vp.raw_mbo_rows,
        search_table=vp.edge_search_table,
        best_spec=vp.best_edge_spec,
        best_row=vp.best_edge_row,
        trades=vp.edge_trades,
        trade_summary=vp.edge_summary,
        report_md=vp.report.to_markdown(),
        vp=vp,
    )


__all__ = [
    "LiquidityEdgeResult",
    "run_liquidity_edge_research",
]
