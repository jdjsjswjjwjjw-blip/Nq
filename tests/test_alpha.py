"""اختبارات المحطة 8: إشارات الألفا والمخرجات النهائية."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.alpha import (
    align_forward_returns,
    discover_alpha_from_features,
    evaluate_signal,
    run_research_pipeline,
    screen_signals,
)
from nq.alpha.discovery import _fold_sign_stability
from nq.core.determinism import make_generator
from nq.models.splitting import WalkForwardFold
from tests.mbo_factory import Event, make_stream


def test_align_forward_returns() -> None:
    fwd = align_forward_returns([100.0, 110.0, 121.0], horizon=1)
    assert np.isclose(fwd[0], 0.1)
    assert np.isclose(fwd[1], 0.1)
    assert np.isnan(fwd[2])  # لا مستقبل للأخيرة


def test_evaluate_predictive_vs_noise_signal() -> None:
    rng = make_generator(0)
    n = 300
    fwd = rng.normal(0, 1, n)
    predictive = fwd + rng.normal(0, 0.3, n)  # مرتبط بالعائد الأمامي
    noise = rng.normal(0, 1, n)  # مستقل

    good = evaluate_signal("predictive", predictive, fwd, n_permutations=500, rng=rng)
    bad = evaluate_signal("noise", noise, fwd, n_permutations=500, rng=rng)
    assert good.ic > bad.ic
    assert good.ic_pvalue < 0.05
    assert bad.ic_pvalue > 0.05


def test_sparse_event_evaluation_counts_only_actual_events() -> None:
    values = np.zeros(100, dtype=np.float64)
    returns = np.zeros(100, dtype=np.float64)
    active = np.arange(0, 100, 10)
    values[active] = np.where(active % 20 == 0, 1.0, -1.0)
    returns[active] = values[active] * np.linspace(0.1, 1.0, active.size)

    result = evaluate_signal(
        "event", values, returns, active_only=True, n_permutations=20, rng=make_generator(19)
    )

    assert result.n == active.size


def test_fold_stability_rejects_sign_flips_hidden_by_aggregate() -> None:
    rng = make_generator(20)
    forward = rng.normal(size=40)
    values = forward.copy()
    folds: list[WalkForwardFold] = []
    for fold_i, start in enumerate(range(0, 40, 10)):
        stop = start + 10
        if fold_i % 2:
            values[start:stop] *= -1.0
        folds.append(
            WalkForwardFold(
                train_idx=np.arange(0, start, dtype=np.intp),
                test_idx=np.arange(start, stop, dtype=np.intp),
            )
        )

    valid, consistency, stable = _fold_sign_stability(values, forward, folds, active_only=False)

    assert valid == 4
    assert consistency == 0.5
    assert stable is False


def test_screen_signals_multiple_testing() -> None:
    rng = make_generator(1)
    n = 300
    fwd = rng.normal(0, 1, n)
    predictive = fwd + rng.normal(0, 0.3, n)
    evals = [
        evaluate_signal("predictive", predictive, fwd, n_permutations=500, rng=rng),
        evaluate_signal("noise1", rng.normal(0, 1, n), fwd, n_permutations=500, rng=rng),
        evaluate_signal("noise2", rng.normal(0, 1, n), fwd, n_permutations=500, rng=rng),
    ]
    screened = screen_signals(evals, alpha=0.05)
    selected = screened.filter(screened["selected"])["name"].to_list()
    assert "predictive" in selected
    assert "noise1" not in selected


def test_discover_alpha_from_features_report() -> None:
    rng = make_generator(2)
    n = 200
    price = np.cumsum(rng.normal(0, 1, n)) + 1000.0
    fwd = align_forward_returns(price, horizon=1)
    predictive = np.nan_to_num(fwd)  # إشارة مرتبطة بالعائد الأمامي
    frame = pl.DataFrame(
        {
            "availability_ts": list(range(n)),
            "nq_close": price,
            "good": predictive,
            "bad": rng.normal(0, 1, n),
        }
    )
    discovery = discover_alpha_from_features(
        frame,
        signal_columns=["good", "bad"],
        price_col="nq_close",
        horizon=1,
        n_permutations=500,
        rng=rng,
    )
    assert "good" in discovery.selected
    md = discovery.report.to_markdown()
    assert "Novel Alpha" in md
    # كل استنتاج في التقرير موثّق بدليل قابل للتتبّع
    assert len(discovery.report.verified) == len(discovery.selected)


def _market(prices: list[int], *, symbol: str, instrument_id: int) -> pl.DataFrame:
    events: list[Event] = []
    ts: list[int] = []
    seq: list[int] = []
    oid = 1
    t = 0
    for i, price in enumerate(prices):
        events.append(("A", "B", price - 1_000_000, 5, oid))
        events.append(("A", "A", price + 1_000_000, 5, oid + 1))
        events.append(("T", "B", price, 3, 0))
        ts.extend([t, t + 1, t + 2])
        seq.extend([3 * i + 1, 3 * i + 2, 3 * i + 3])
        oid += 2
        t += 100
    return make_stream(
        events, instrument_id=instrument_id, symbol=symbol, event_ts=ts, sequence=seq
    )


def test_pipeline_reproducible_from_raw_mbo() -> None:
    prices = [100_000_000 + i * 250_000 for i in range(12)]
    nq = _market(prices, symbol="NQ", instrument_id=1)
    mnq = _market(prices, symbol="MNQ", instrument_id=2)

    a = run_research_pipeline(
        nq,
        mnq,
        interval_ns=100,
        horizon=1,
        n_permutations=300,
        rng=make_generator(9),
        quiet=True,
    )
    b = run_research_pipeline(
        nq,
        mnq,
        interval_ns=100,
        horizon=1,
        n_permutations=300,
        rng=make_generator(9),
        quiet=True,
    )
    # نفس البيانات الخام + نفس البذرة -> نفس المخرجات بالضبط (قابلية إعادة الإنتاج)
    assert a.evaluations.equals(b.evaluations)
    assert a.selected == b.selected
