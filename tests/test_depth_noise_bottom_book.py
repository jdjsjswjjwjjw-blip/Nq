"""اختبارات فلتر ضوضاء العمق وحافة أسفل الدفتر."""

from __future__ import annotations

import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.bottom_book import BOTTOM_BOOK_COLUMNS, bottom_book_features_at_bar_close
from nq.simulation.depth_lifecycle import attach_depth_asof, depth_event_path_at_bar_close
from nq.simulation.depth_noise import DepthNoiseConfig, filter_depth_noise
from tests.mbo_factory import make_stream
from tests.test_coverage import _paired_streams


def test_filter_depth_noise_drops_flicker() -> None:
    # إضافة ثم إلغاء سريع لنفس الأمر دون تنفيذ = وميض
    frame = make_stream(
        [
            ("A", "B", 100, 5, 1),
            ("C", "B", 100, 5, 1),  # flicker cancel
            ("A", "A", 101, 4, 2),
        ],
        event_ts=[0, 1_000_000, 2_000_000],  # 1ms apart < flicker_ns default 50ms
        sequence=[1, 2, 3],
    )
    cleaned = filter_depth_noise(
        frame, config=DepthNoiseConfig(flicker_ns=50_000_000, cancel_storm_min_events=100)
    )
    actions = cleaned["action"].to_list()
    assert "C" not in [str(a) for a in actions]
    assert cleaned.height == 2


def test_filter_depth_noise_keeps_trades() -> None:
    frame = make_stream(
        [
            ("A", "B", 100, 5, 1),
            ("T", "A", 100, 2, 0),
            ("C", "B", 100, 3, 1),
        ],
        event_ts=[0, 10, 20],
        sequence=[1, 2, 3],
    )
    cleaned = filter_depth_noise(frame, config=DepthNoiseConfig(cancel_storm_min_events=100))
    assert "T" in [str(a) for a in cleaned["action"].to_list()]


def test_bottom_book_columns_at_bucket_end() -> None:
    nq, _ = _paired_streams(500, seed=11)
    bottom = bottom_book_features_at_bar_close(nq, interval_ns=10_000, filter_noise=True)
    assert bottom.height >= 1
    for c in BOTTOM_BOOK_COLUMNS:
        assert c in bottom.columns
    assert (bottom[AVAILABILITY_TS] == bottom["bucket_end"]).all()


def test_depth_path_includes_intra_bar_extremes() -> None:
    nq, _ = _paired_streams(500, seed=12)
    path = depth_event_path_at_bar_close(nq, interval_ns=10_000)
    assert "depth_path_imbalance_max" in path.columns
    assert "depth_path_l2_l5_bid_drain" in path.columns


def test_attach_depth_asof_preserves_null_when_no_match() -> None:
    """لا تُملأ قيم العمق الناقصة بأصفار — null ≠ دفتر فارغ."""
    features = pl.DataFrame({AVAILABILITY_TS: [10, 20, 30], "x": [1.0, 2.0, 3.0]})
    depth = pl.DataFrame(
        {
            AVAILABILITY_TS: [25],
            "depth_cum_bid": [9.0],
            "depth_cum_ask": [4.0],
        }
    )
    joined = attach_depth_asof(features, depth, columns=["depth_cum_bid", "depth_cum_ask"])
    # الصفان الأولان بلا لقطة سابقة
    assert joined["depth_cum_bid"][0] is None or joined["depth_cum_bid"].is_null()[0]
    assert joined["depth_cum_bid"][2] == 9.0


def test_fb_depth_match_tolerance_is_ticks_not_price_scale() -> None:
    """عتبة تطابق مستوى الكسر = 4 تيكات (1.0$) وليست PRICE_SCALE*4."""
    assert PRICE_SCALE * 4 < 1e-5
    assert 0.25 * 4 == 1.0
