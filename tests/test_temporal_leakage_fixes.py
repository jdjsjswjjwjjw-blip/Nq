"""اختبارات انحدار لإصلاحات تدقيق التسريب الزمني.

ثلاث ثغرات أُغلقت — هذه الاختبارات تمنع عودتها:

1. ``auction_signals_from_states`` كان يُسمّي VA البرميل الحالي ``decision_*``
   عند غياب الأعمدة (states يدوية) — الآن يشتقها بتأخير صف واحد.
2. ``market_truth._thesis_direction`` كان يقارن الإغلاق بـPOC البرميل نفسه —
   الآن يستخدم ``decision_poc`` (أو تأخير صف عند غيابه).
3. ``_descriptive_probabilities`` (سقوط الطيّات) كان يعطي ثقة موجبة لمعدلات
   محسوبة على كل الإطار بما فيه المستقبل — الآن ``confidence=0.0`` إلزاميًا.
"""

from __future__ import annotations

import polars as pl
import pytest

from nq.auction_behavior.model import estimate_behavior_probabilities
from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.auction import auction_signals_from_states
from nq.simulation.market_truth import _thesis_direction


def _manual_states(n: int = 4) -> pl.DataFrame:
    """states يدوية بلا decision_* — تحاكي مستهلكًا قديمًا."""
    return pl.DataFrame(
        {
            "bucket_start": [i * 100 for i in range(n)],
            "bucket_end": [(i + 1) * 100 for i in range(n)],
            AVAILABILITY_TS: [(i + 1) * 100 for i in range(n)],
            # VA الحالية تتغير كل برميل حتى يفضح أي استخدام نفس البار
            "vah": [110 + 10 * i for i in range(n)],
            "poc": [100 + 10 * i for i in range(n)],
            "val": [90 + 10 * i for i in range(n)],
            "close": [100 + 10 * i for i in range(n)],
            "high": [111 + 10 * i for i in range(n)],
            "low": [89 + 10 * i for i in range(n)],
            "open": [100 + 10 * i for i in range(n)],
            "bucket_volume": [100] * n,
            "buy_volume": [60] * n,
            "sell_volume": [40] * n,
            "delta": [20] * n,
            "range": [22] * n,
            "is_balanced": [False] * n,
            "is_expansion": [False] * n,
            "close_in_value": [True] * n,
            "pullback_defended": [False] * n,
            "made_new_high": [False] * n,
            "made_new_low": [False] * n,
            "poc_migration": [0] * n,
            "excess_upper": [0] * n,
            "excess_lower": [0] * n,
            "in_value_fraction": [0.5] * n,
            "balance_confidence": [0.0] * n,
            "absorb": [0.0] * n,
            "look_fail": [0.0] * n,
            "vp_liquidity_session": [0] * n,
        }
    )


def test_signals_fallback_decision_bounds_are_lagged_not_same_bar() -> None:
    """غياب decision_* في states يدوية يجب ألا يسرّب VA نفس البار."""
    states = _manual_states()
    signals = auction_signals_from_states(states, fixed_range_decisions=False)
    scale = 1e-9
    # البرميل 0 لا يملك ملفًا مكتملًا سابقًا → حدوده null وليست vah[0]
    assert signals["vp_upper"][0] is None
    # البرميل i يقرأ VA البرميل i-1 (المكتمل)، وليس VA نفسه
    for i in range(1, states.height):
        assert signals["vp_upper"][i] == pytest.approx(states["vah"][i - 1] * scale)
        assert signals["vp_mid"][i] == pytest.approx(states["poc"][i - 1] * scale)
        assert signals["vp_lower"][i] == pytest.approx(states["val"][i - 1] * scale)
        # وبالتحديد: ليست حدود نفس البار
        assert signals["vp_upper"][i] != pytest.approx(states["vah"][i] * scale)


def test_thesis_direction_uses_decision_poc_not_current_poc() -> None:
    """الثيسيس يقارن الإغلاق بـPOC القرار المتأخر، لا POC نفس البار."""
    states = pl.DataFrame(
        {
            # close=100: فوق decision_poc (90) لكنه تحت POC البرميل الحالي (110)
            "close": [100.0],
            "poc": [110.0],
            "decision_poc": [90.0],
            "is_balanced": [False],
        }
    )
    direction = _thesis_direction(states)
    # لو استخدم poc الحالي لكان الاتجاه -1؛ الصحيح سببيًا: +1
    assert direction.to_list() == [1.0]


def test_thesis_direction_without_decision_poc_lags_one_row() -> None:
    """states يدوية بلا decision_poc: تأخير صف واحد، والصف الأول غير حاسم."""
    states = pl.DataFrame(
        {
            "close": [100.0, 100.0],
            "poc": [110.0, 90.0],
            "is_balanced": [False, False],
        }
    )
    direction = _thesis_direction(states)
    # الصف 0: لا POC سابق → 0. الصف 1: مقابل poc[0]=110 → هابط -1
    # (لو استخدم poc نفس البار (90) لأعطى +1)
    assert direction.to_list() == [0.0, -1.0]


def test_descriptive_fallback_probabilities_have_zero_confidence() -> None:
    """سقوط الطيّات → معدلات وصفية على كل الإطار: الثقة يجب أن تكون صفرًا."""
    n = 4  # أقل من أن يكوّن طيّات walk-forward
    blended = pl.DataFrame(
        {
            AVAILABILITY_TS: [i * 100 for i in range(n)],
            "vp_balance": [1.0] * n,
            "vp_imbalance": [0.0] * n,
            "signal_quality": [0.9] * n,
        }
    )
    events = pl.DataFrame(
        {
            AVAILABILITY_TS: [i * 100 for i in range(n)],
            "evt_true_break": [1.0] * n,
            "evt_failed_breakout": [0.0] * n,
            "evt_retest_success": [0.0] * n,
            "evt_retest_fail": [0.0] * n,
            "evt_expansion_continue": [0.0] * n,
            "evt_return_to_value": [0.0] * n,
        }
    )
    probs, folds = estimate_behavior_probabilities(
        blended,
        events,
        n_splits=3,
        min_train_size=1_000_000,  # يستحيل تكوين طيّة → المسار الوصفي
    )
    assert folds.height == 0
    assert probs.confidence == 0.0
    assert "not OOS" in probs.detail
