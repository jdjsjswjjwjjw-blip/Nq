"""إثبات السببية ومنع التسريب الزمني في إعادة البناء.

القاعدة: حالة الدفتر عند الحدث ``k`` يجب أن تعتمد على الأحداث حتى ``k`` فقط.
لذا إعادة البناء على البادئة ``[0 : k]`` يجب أن تُنتج نفس سلسلة top-of-book
التي تُنتجها إعادة البناء الكاملة على البادئة نفسها — أي أن أحداث المستقبل لا
تؤثّر في الماضي إطلاقًا.
"""

from __future__ import annotations

import pytest

from nq.orderbook import reconstruct
from tests.mbo_factory import random_add_cancel_stream


@pytest.mark.leakage
@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_reconstruction_is_future_independent(seed: int) -> None:
    frame = random_add_cancel_stream(400, seed=seed)
    full = reconstruct(frame).top_of_book

    for cut in (50, 137, 250, 399):
        prefix = reconstruct(frame.slice(0, cut)).top_of_book
        assert prefix.equals(full.slice(0, cut)), (
            f"future events leaked into the past at cut={cut} (seed={seed})"
        )


@pytest.mark.leakage
def test_prefix_final_state_matches_full_run_snapshot() -> None:
    frame = random_add_cancel_stream(300, seed=3)
    full = reconstruct(frame).top_of_book
    cut = 180
    prefix_book = reconstruct(frame.slice(0, cut)).book

    row = full.slice(cut - 1, 1)
    expected_bid = row["best_bid"].to_list()[0]
    got = prefix_book.best_bid()
    got_bid = got[0] if got is not None else None
    assert got_bid == expected_bid
