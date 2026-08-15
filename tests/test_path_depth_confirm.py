"""تأكيد مسار لندن+عمق: قياس مستمر بلا بوابات IF وبلا ريتست كامل."""

from __future__ import annotations

import polars as pl
import pytest

from nq.auction_behavior.path_confirm import (
    PATH_CONFIRM_COLUMNS,
    attach_path_depth_confirmation,
)
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION
from nq.validation.leakage import assert_causal_order

_TICK = float(round(0.25 / PRICE_SCALE))
_VAH = 100.0
_VAL = 80.0


def _london_path_frame(
    *, closes: list[float], follow: list[float], defend: list[float]
) -> pl.DataFrame:
    n = len(closes)
    return pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(1, n + 1)),
            VP_LIQUIDITY_SESSION: [1] * n,
            "_liquidity_run": [1] * n,
            "close": closes,
            "asia_vah": [_VAH] * n,
            "asia_val": [_VAL] * n,
            "asia_poc": [90.0] * n,
            "lf_liquidity_migration": follow,
            "lf_break_level_trade_intensity": follow,
            "lf_liquidity_withdrawal": follow,
            "lf_near_vah_cancel_ratio": follow,
            "lf_near_val_cancel_ratio": [0.0] * n,
            "lf_near_hvn_cancel_ratio": follow,
            "lf_refill_rate": defend,
            "lf_absorption_proxy": defend,
            "lf_queue_survival_rate": defend,
            "lf_near_vah_add_intensity": defend,
            "lf_near_val_add_intensity": [0.0] * n,
            "lf_near_poc_add_intensity": defend,
            "proj_poc_shift_ticks": [6.0] * n,
            "proj_outside_volume_share": [0.6] * n,
            "proj_va_overlap": [0.4] * n,
        }
    )


def test_path_confirm_has_no_retest_gate_and_measures_small_correction() -> None:
    """كسر + تصحيح بسيط + عمق متابع → تقدم، من غير عمود ريتست."""
    tick = _TICK
    closes = [
        _VAH + 8 * tick,
        _VAH + 12 * tick,
        _VAH + 10 * tick,
    ]
    frame = _london_path_frame(closes=closes, follow=[0.8, 0.9, 0.85], defend=[0.05, 0.04, 0.05])
    out = attach_path_depth_confirmation(frame)
    assert "vp_fsm_retest" not in frame.columns
    for col in PATH_CONFIRM_COLUMNS:
        assert col in out.columns
    assert float(out["path_beyond_asia_ticks"][0]) > 0.0
    assert float(out["path_extreme_ticks"][2]) >= float(out["path_beyond_asia_ticks"][2])
    assert float(out["path_correction_ticks"][2]) > 0.0
    assert float(out["path_inside_asia_va"][2]) == pytest.approx(0.0)
    assert float(out["path_held_frac"][2]) > 0.5
    assert float(out["path_depth_confirm"][2]) > 0.5
    assert float(out["path_change_progress"][2]) > 0.0


def test_asia_bars_carry_zero_path_scores() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2],
            VP_LIQUIDITY_SESSION: [0, 0],
            "close": [_VAH + 20.0, _VAH + 40.0],
            "asia_vah": [_VAH, _VAH],
            "asia_val": [_VAL, _VAL],
        }
    )
    out = attach_path_depth_confirmation(frame)
    assert float(out["path_beyond_asia_ticks"].sum()) == pytest.approx(0.0)
    assert float(out["path_change_progress"].sum()) == pytest.approx(0.0)


def test_return_into_asia_raises_fail_without_if_gate() -> None:
    tick = _TICK
    closes = [_VAH + 10 * tick, (_VAH + _VAL) * 0.5]
    frame = _london_path_frame(closes=closes, follow=[0.2, 0.05], defend=[0.1, 0.9])
    out = attach_path_depth_confirmation(frame)
    assert float(out["path_inside_asia_va"][1]) > float(out["path_inside_asia_va"][0])
    assert float(out["path_change_fail"][1]) >= float(out["path_change_fail"][0])
    assert float(out["path_held_frac"][1]) < float(out["path_held_frac"][0])


@pytest.mark.leakage
def test_future_depth_does_not_change_prior_path_confirm() -> None:
    tick = _TICK
    closes = [_VAH + 6 * tick, _VAH + 9 * tick, _VAH + 11 * tick]
    base = _london_path_frame(closes=closes, follow=[0.4, 0.4, 0.4], defend=[0.1, 0.1, 0.1])
    baseline = attach_path_depth_confirmation(base)
    changed = base.with_columns(
        pl.when(pl.col(AVAILABILITY_TS) > 2)
        .then(pl.col("lf_liquidity_migration") + 5.0)
        .otherwise(pl.col("lf_liquidity_migration"))
        .alias("lf_liquidity_migration")
    )
    perturbed = attach_path_depth_confirmation(changed)
    prior = baseline.filter(pl.col(AVAILABILITY_TS) <= 2).select(
        AVAILABILITY_TS, "path_depth_confirm", "path_change_progress"
    )
    after = perturbed.filter(pl.col(AVAILABILITY_TS) <= 2).select(
        AVAILABILITY_TS, "path_depth_confirm", "path_change_progress"
    )
    assert prior.equals(after)
    assert_causal_order(baseline[AVAILABILITY_TS].to_numpy())
