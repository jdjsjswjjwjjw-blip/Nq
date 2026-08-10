"""اختبارات خصمية لإصلاحات التدقيق التقني P0/P1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nq.alpha.discovery import discover_alpha_from_features
from nq.alpha.signals import evaluate_signal, evaluate_signal_intraday
from nq.contracts.mbo import MBO_SCHEMA
from nq.core.temporal_policy import TemporalPolicy
from nq.coverage.metrics import measure_qduf
from nq.ingestion.databento import normalize_databento_frame
from nq.ingestion.reader import load_mbo_frame
from nq.models.splitting import purged_walk_forward_split
from nq.orderbook import OrderBook, reconstruct
from nq.strategies.fvg_hypothesis import walk_forward_select_hypotheses
from nq.validation.leakage import LeakageError, assert_temporal_split
from tests.mbo_factory import make_stream


def test_wf_selects_best_not_first_candidate() -> None:
    """مرشح أول سيئ + ثانٍ مرتبط بالهدف → يجب اختيار الثاني."""
    n = 120
    rng = np.random.default_rng(0)
    noise = rng.normal(size=n)
    signal_good = np.cumsum(rng.normal(size=n))
    # أسعار تتحرك مع الإشارة الجيدة
    prices = 100.0 + np.cumsum(0.1 * np.diff(signal_good, prepend=signal_good[0])) + 0.01 * noise
    features = pl.DataFrame(
        {
            "availability_ts": np.arange(n, dtype=np.int64) * 1_000_000_000,
            "nq_close": prices,
            "bad_first": noise,  # أول عمود — عديم الفائدة
            "good_second": signal_good,
        }
    )
    folds, oos_ic, _p, oos_n, best = walk_forward_select_hypotheses(
        features,
        ["bad_first", "good_second"],
        price_col="nq_close",
        horizon=1,
        n_splits=3,
        n_permutations=50,
        selection_aware_null=False,
        rng=rng,
    )
    assert best == "good_second"
    assert folds.height >= 1
    train_abs_max = float(np.max(folds["train_ic"].abs().to_numpy().astype(np.float64)))
    assert train_abs_max < 1e17  # ليس -1e18
    assert oos_n > 0
    del oos_ic


def test_short_signal_positive_pnl_keeps_positive_sharpe() -> None:
    """بيع رابح في سوق هابط لا يُقلب إلى خسارة عبر ضرب الاتجاه مرتين."""
    n = 40
    bid = np.linspace(100.0, 90.0, n)
    ask = bid + 0.25
    signal = np.full(n, -1.0)  # short always
    ev = evaluate_signal_intraday(
        "short",
        signal,
        bid,
        ask,
        horizon=1,
        slippage_ticks=0.0,
        tick_size=0.25,
        n_permutations=20,
        rng=np.random.default_rng(0),
    )
    assert ev.mean_strategy_return > 0.0
    assert ev.sharpe > 0.0


def test_directional_strategy_returns_not_resign() -> None:
    values = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    mid_fwd = np.array([-0.01] * 10)  # mid falling
    already = np.array([0.01] * 10)  # short PnL already positive
    ev = evaluate_signal(
        "s",
        values,
        mid_fwd,
        strategy_returns=already,
        n_permutations=10,
        rng=np.random.default_rng(0),
        min_samples=8,
    )
    assert ev.mean_strategy_return == pytest.approx(0.01)


def test_split_no_shared_timestamp() -> None:
    times = np.array([0, 0, 0, 10, 10, 20, 20, 30, 30, 40, 40, 50], dtype=np.int64)
    folds = purged_walk_forward_split(times, n_splits=2, embargo=0, min_train_size=2)
    assert folds
    for fold in folds:
        train_ts = set(times[fold.train_idx].tolist())
        test_ts = set(times[fold.test_idx].tolist())
        assert train_ts.isdisjoint(test_ts)
        assert_temporal_split(times[fold.train_idx], times[fold.test_idx], embargo=0)


def test_assert_temporal_split_rejects_equal_ts() -> None:
    with pytest.raises(LeakageError):
        assert_temporal_split([1.0, 2.0, 3.0], [3.0, 4.0], embargo=0)


def test_databento_cancel_partial_and_fill_noop_on_stream() -> None:
    frame = make_stream(
        [
            ("A", "B", 100, 10, 1),
            ("F", "N", 0, 3, 1),
            ("C", "N", 0, 3, 1),
            ("T", "A", 100, 3, 0),
        ],
        event_ts=[0, 1, 1, 1],
        sequence=[1, 2, 2, 2],
    )

    result = reconstruct(frame)
    assert result.book.best_bid() == (100, 7)


def test_ts_in_delta_never_becomes_ingest_ts() -> None:
    frame = pl.DataFrame(
        {
            "ts_event": [100, 200],
            "ts_recv": [110, 210],
            "ts_in_delta": [999, 888],
            "sequence": [1, 2],
            "instrument_id": [1, 1],
            "symbol": ["NQ", "NQ"],
            "action": ["A", "A"],
            "side": ["B", "A"],
            "price": [20_000_000_000, 20_000_000_100],
            "size": [1, 1],
            "order_id": [1, 2],
            "flags": [0, 0],
        }
    )

    out = normalize_databento_frame(frame)
    assert out["ingest_ts"].to_list() == [110, 210]
    assert "ts_in_delta" not in out.columns


def test_rtype_alone_does_not_become_flags() -> None:
    frame = pl.DataFrame(
        {
            "ts_event": [100],
            "ts_recv": [101],
            "sequence": [1],
            "instrument_id": [1],
            "symbol": ["NQ"],
            "action": ["A"],
            "side": ["B"],
            "price": [20_000_000_000],
            "size": [1],
            "order_id": [1],
            "rtype": [160],
        }
    )

    out = normalize_databento_frame(frame)
    assert out["flags"].to_list() == [0]


def test_max_rows_after_causal_sort(tmp_path: Path) -> None:
    """قصّ max_rows يأخذ أقدم الأحداث سببيًا وليس رأس الملف غير المرتّب."""

    path = tmp_path / "unordered.parquet"
    pl.DataFrame(
        {
            "event_ts": [30, 10, 20],
            "ingest_ts": [30, 10, 20],
            "sequence": [3, 1, 2],
            "instrument_id": [1, 1, 1],
            "symbol": ["NQ", "NQ", "NQ"],
            "action": ["A", "A", "A"],
            "side": ["B", "B", "B"],
            "price": [100, 101, 102],
            "size": [1, 1, 1],
            "order_id": [3, 1, 2],
            "flags": [0, 0, 0],
        },
        schema=MBO_SCHEMA,
    ).write_parquet(path)
    out = load_mbo_frame(path, max_rows=2)
    assert out.height == 2
    assert out["event_ts"].to_list() == [10, 20]
    assert out["order_id"].to_list() == [1, 2]


def test_temporal_policy_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[temporal]\nembargo_ns = 1\nhorizon = 1\nweird_key = 9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        TemporalPolicy.from_config(path)


def test_vp_auction_toml_embargo_applied() -> None:

    policy = TemporalPolicy.for_run(
        interval_ns=30_000_000_000,
        window=5,
        horizon=1,
        config_path=Path("configs/vp_auction.toml"),
    )
    assert policy.embargo_ns == 30_000_000_000


def test_discover_alpha_refuses_full_sample_when_no_folds() -> None:
    """عيّنة أصغر من الحد الأدنى للطيّات → لا ادّعاء ألفا داخل العيّنة."""

    frame = pl.DataFrame(
        {
            "availability_ts": list(range(5)),
            "nq_close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "sig": [1.0, -1.0, 1.0, -1.0, 1.0],
        }
    )
    discovery = discover_alpha_from_features(
        frame,
        signal_columns=["sig"],
        price_col="nq_close",
        n_splits=3,
        n_permutations=10,
        rng=np.random.default_rng(0),
    )
    assert discovery.evaluations.height == 0
    assert discovery.selected == []


def test_modify_unknown_no_ghost_order() -> None:
    book = OrderBook()
    book.apply("M", "B", 99, 5, 4242)
    assert book.unknown_order_refs == 1
    assert book.orders == {}
    assert book.best_bid() is None


def test_qduf_negative_r2_mbo_does_not_explode() -> None:
    """R² سالب لوصف MBO لا ينتج نسبة QDUF متفجّرة."""

    n = 80
    rng = np.random.default_rng(0)
    times = np.arange(n, dtype=np.int64) * 1_000_000_000
    noise = rng.normal(size=n)
    features = pl.DataFrame(
        {
            "availability_ts": times,
            "nq_close": 100.0 + np.cumsum(rng.normal(size=n)),
            "feat_a": noise,
        }
    )
    # واصفات MBO القياسية بأسماء العقد — ضوضاء مستقلة عن العائد.
    descriptors = pl.DataFrame(
        {
            "availability_ts": times,
            "add_count": rng.integers(0, 5, size=n),
            "cancel_count": rng.integers(0, 5, size=n),
            "fill_count": rng.integers(0, 3, size=n),
            "trade_count": rng.integers(0, 3, size=n),
            "trade_volume": rng.integers(0, 10, size=n),
            "cancel_ratio": rng.random(size=n),
            "depth_change": rng.normal(size=n),
            "spread": rng.integers(1, 4, size=n),
            "mid": 100.0 + rng.normal(size=n),
        }
    )
    result = measure_qduf(
        features,
        descriptors,
        n_splits=3,
        embargo=0,
        n_permutations=20,
        rng=rng,
    )
    assert np.isfinite(result.value)
    assert result.value <= 1.0 + 1e-9
    assert result.value >= -1e-9
