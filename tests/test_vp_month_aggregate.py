"""اختبارات تجميع نتيجة شهر VP من أيام معزولة."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nq.research.vp_month_aggregate import write_vp_month_aggregate


def _write_day(root: Path, day_id: str, *, exp: float, n: float, ic: float, p: float) -> None:
    d = root / day_id
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "best_signal": ["vp_balance"],
            "oos_ic": [ic],
            "oos_pvalue": [p],
            "oos_n": [100],
            "exploratory_full_sample": [False],
            "with_execution": [True],
            "raw_mbo_rows": [1000],
            "cleaned_mbo_rows": [1000],
            "best_edge_spec": ["hold2_rr3_buf1_rr_multiple"],
            "edge_oos_expectancy": [exp],
            "edge_oos_rr": [3.0],
            "edge_n_trades": [n],
            "wf_before_execution": [True],
        }
    ).write_parquet(d / "vp_oos_summary.parquet")
    # n active trades with positive pnl when exp>0
    pnl = 0.1 if exp > 0 else -0.1
    rows = max(1, int(n))
    pl.DataFrame(
        {
            "edge_signal": [1.0] * rows,
            "edge_pnl": [pnl] * rows,
            "edge_rr": [3.0] * rows,
            "edge_hit": [1.0 if exp > 0 else -1.0] * rows,
        }
    ).write_parquet(d / "edge_trades.parquet")


def test_write_vp_month_aggregate_pooled_verdict(tmp_path: Path) -> None:
    _write_day(tmp_path, "2026-05-01", exp=0.01, n=20, ic=0.02, p=0.4)
    _write_day(tmp_path, "2026-05-02", exp=0.02, n=25, ic=0.01, p=0.6)
    path = write_vp_month_aggregate(tmp_path)
    assert path.name == "FINAL_RESULT.md"
    text = path.read_text(encoding="utf-8")
    assert "EDGE_FOUND_POOLED" in text or "NO_EDGE_POOLED" in text or "INSUFFICIENT" in text
    assert (tmp_path / "day_results.parquet").exists()
    assert (tmp_path / "aggregate_edge_trades.parquet").exists()
    assert (tmp_path / "FINAL_RESULT.parquet").exists()
    days = pl.read_parquet(tmp_path / "day_results.parquet")
    assert days.height == 2
    pooled = pl.read_parquet(tmp_path / "aggregate_edge_trades.parquet")
    assert pooled.height == 45
    assert "day_id" in pooled.columns
