"""اختبارات بحث فرضيات Failed FVG (walk-forward + منع التسريب)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import make_generator
from nq.simulation.fvg import NS_30M, NS_PER_MIN, build_ohlcv_bars, failed_fvg_features
from nq.strategies.fvg_hypothesis import (
    FvgHypothesisSpec,
    default_fvg_grid,
    materialize_fvg_hypotheses,
    search_fail_fvg_hypotheses,
    walk_forward_select_hypotheses,
)
from tests.mbo_factory import make_stream
from tests.test_coverage import _paired_streams


def _trades_at_prices(
    prices: list[int],
    *,
    start_ts: int = 0,
    step_ns: int = NS_30M // 4,
) -> pl.DataFrame:
    events = [("T", "B", p, 2, 0) for p in prices]
    ts = [start_ts + i * step_ns for i in range(len(prices))]
    return make_stream(events, event_ts=ts, sequence=list(range(1, len(prices) + 1)))


def test_default_fvg_grid_nonempty_and_causal_intervals() -> None:
    grid = default_fvg_grid()
    assert len(grid) >= 8
    for spec in grid:
        assert spec.h1_interval_ns >= spec.signal_interval_ns
        assert spec.fvg_window_ns >= NS_PER_MIN


def test_materialize_hypotheses_past_stable_under_future_perturbation() -> None:
    rng = np.random.default_rng(0)
    n = 200
    base = int(20_000 / PRICE_SCALE)
    prices = [base + int(rng.integers(-5, 6) * (1 / PRICE_SCALE)) for _ in range(n)]
    frame = _trades_at_prices(prices, step_ns=NS_30M // 3)
    clock = build_ohlcv_bars(frame, interval_ns=NS_30M)
    specs = (
        FvgHypothesisSpec(
            name="a",
            h1_interval_ns=60 * NS_PER_MIN,
            signal_interval_ns=30 * NS_PER_MIN,
            fvg_window_ns=90 * NS_PER_MIN,
            vol_price_mult=1.2,
            vol_volume_mult=1.3,
        ),
        FvgHypothesisSpec(
            name="b",
            h1_interval_ns=30 * NS_PER_MIN,
            signal_interval_ns=15 * NS_PER_MIN,
            fvg_window_ns=60 * NS_PER_MIN,
            vol_price_mult=1.1,
            vol_volume_mult=1.2,
        ),
    )
    baseline = materialize_fvg_hypotheses(frame, specs, clock=clock)
    assert baseline.height > 0
    cut_ts = int(baseline[AVAILABILITY_TS][baseline.height // 2])
    past = baseline.filter(pl.col(AVAILABILITY_TS) <= cut_ts)

    event_ts = frame["event_ts"].to_list()
    new_prices = frame["price"].to_list()
    for i, ts in enumerate(event_ts):
        if ts > cut_ts:
            new_prices[i] = base + int(rng.integers(50, 80) * (1 / PRICE_SCALE))
    perturbed = frame.with_columns(pl.Series("price", new_prices))
    after = materialize_fvg_hypotheses(perturbed, specs, clock=clock)
    after_past = after.filter(pl.col(AVAILABILITY_TS) <= cut_ts)
    for col in (specs[0].column(), specs[1].column()):
        assert past[col].to_list() == after_past[col].to_list()


def test_search_fail_fvg_hypotheses_smoke() -> None:
    nq, _ = _paired_streams(2500, seed=90)
    tiny = (
        FvgHypothesisSpec(
            name="tiny_a",
            h1_interval_ns=60 * NS_PER_MIN,
            signal_interval_ns=30 * NS_PER_MIN,
            fvg_window_ns=90 * NS_PER_MIN,
        ),
        FvgHypothesisSpec(
            name="tiny_b",
            h1_interval_ns=30 * NS_PER_MIN,
            signal_interval_ns=15 * NS_PER_MIN,
            fvg_window_ns=60 * NS_PER_MIN,
            vol_price_mult=1.1,
            vol_volume_mult=1.2,
        ),
    )
    result = search_fail_fvg_hypotheses(
        nq,
        specs=tiny,
        interval_ns=10_000,
        use_ssl_gate=False,
        n_splits=2,
        n_permutations=50,
        rng=make_generator(7),
        quiet=True,
    )
    assert result.features.height > 0
    assert len(result.candidate_columns) == 2
    assert result.report is not None
    md = result.report.to_markdown()
    assert "Failed FVG Hypothesis Search" in md


def test_walk_forward_selects_train_best_candidate_not_first() -> None:
    n = 90
    returns = 0.001 * np.sin(np.linspace(0.0, 8.0 * np.pi, n - 1))
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1.0 + float(ret)))

    good = np.concatenate([returns, np.array([0.0])])
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: np.arange(n, dtype=np.int64),
            "nq_close": prices,
            "bad": -good,
            "good": good,
        }
    )

    fold_df, _oos_ic, _oos_p, _oos_n, best = walk_forward_select_hypotheses(
        frame,
        ["bad", "good"],
        price_col="nq_close",
        horizon=1,
        n_splits=3,
        embargo=0,
        purge_samples=0,
        n_permutations=10,
        rng=np.random.default_rng(3),
    )

    assert fold_df.height > 0
    assert set(fold_df["selected"].to_list()) == {"good"}
    assert best == "good"


def test_failed_fvg_baseline_still_works() -> None:
    base = int(20_000 / PRICE_SCALE)
    prices = [base + (i % 7) * int(0.25 / PRICE_SCALE) for i in range(80)]
    frame = _trades_at_prices(prices, step_ns=NS_30M // 2)
    a = failed_fvg_features(frame)
    assert "fail_fvg" in a.columns
