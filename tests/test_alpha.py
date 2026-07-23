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
from nq.core.determinism import make_generator
from nq.statistics.resampling import block_permutation
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


def test_block_permutation_preserves_contiguous_time_blocks() -> None:
    arr = np.arange(12, dtype=np.float64)
    permuted = block_permutation(arr, block_size=3, rng=np.random.default_rng(3))
    assert sorted(permuted.reshape(-1, 3)[:, 0].astype(int).tolist()) == [0, 3, 6, 9]
    assert all(np.all(np.diff(block) == 1) for block in permuted.reshape(-1, 3))


def test_evaluate_signal_supports_block_permutation_null() -> None:
    rng = make_generator(4)
    n = 80
    fwd = rng.normal(0, 1, n)
    signal = fwd + rng.normal(0, 0.2, n)
    result = evaluate_signal(
        "blocked",
        signal,
        fwd,
        n_permutations=100,
        permutation_block_size=5,
        rng=rng,
    )
    assert result.n == n


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


def _alpha_oos_perturbation_frame(*, oos_sign: float) -> pl.DataFrame:
    n = 90
    returns = np.zeros(n)
    returns[:60] = np.tile([0.01, -0.01], 30)
    returns[60:89] = oos_sign * np.tile([0.01, -0.01], 15)[:29]

    price = np.empty(n)
    price[0] = 100.0
    for i in range(n - 1):
        price[i + 1] = price[i] * (1.0 + returns[i])

    train_val_signal = np.zeros(n)
    train_val_signal[:60] = returns[:60]
    oos_only_signal = np.zeros(n)
    oos_only_signal[60:89] = returns[60:89]
    return pl.DataFrame(
        {
            "availability_ts": np.arange(n),
            "nq_close": price,
            "train_val": train_val_signal,
            "oos_only": oos_only_signal,
        }
    )


def test_alpha_selection_is_invariant_to_final_oos_label_perturbation() -> None:
    base = discover_alpha_from_features(
        _alpha_oos_perturbation_frame(oos_sign=1.0),
        signal_columns=["train_val", "oos_only"],
        price_col="nq_close",
        horizon=1,
        n_permutations=300,
        rng=np.random.default_rng(7),
    )
    perturbed = discover_alpha_from_features(
        _alpha_oos_perturbation_frame(oos_sign=-1.0),
        signal_columns=["train_val", "oos_only"],
        price_col="nq_close",
        horizon=1,
        n_permutations=300,
        rng=np.random.default_rng(7),
    )

    assert base.selected == perturbed.selected == ["train_val"]
    assert "oos_only" not in base.selected


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
