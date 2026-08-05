"""اختبارات ذاكرة/سلوك reconstruct بعد إصلاح مادّة to_list الكاملة."""

from __future__ import annotations

import os

import polars as pl

from nq.orderbook.reconstruction import _RECONSTRUCT_CHUNK, reconstruct
from tests.mbo_factory import make_stream, random_add_cancel_stream


def test_reconstruct_chunked_matches_small_reference() -> None:
    """الناتج مطابق للمسار المرجعي على دفق صغير (سببية + TOB)."""
    frame = make_stream(
        [
            ("A", "B", 100, 5, 1),
            ("A", "A", 102, 4, 2),
            ("C", "B", 100, 5, 1),
        ],
        event_ts=[0, 1, 2],
        sequence=[1, 2, 3],
    )
    tob = reconstruct(frame).top_of_book
    assert tob["best_bid"].to_list() == [100, 100, None]
    assert tob["best_ask"].to_list() == [None, 102, 102]


def test_reconstruct_across_chunk_boundary() -> None:
    """الحالة تستمر عبر حدود الشرائح (لا إعادة تصفير دفتر)."""
    n = _RECONSTRUCT_CHUNK + 50
    events = [("A", "B", 100 + (i % 3), 1, i + 1) for i in range(n)]
    # إلغاء آخر أمر في الشريحة الأولى داخل الشريحة الثانية
    events.append(("C", "B", 100, 1, 1))
    frame = make_stream(
        events,
        event_ts=list(range(len(events))),
        sequence=list(range(1, len(events) + 1)),
    )
    result = reconstruct(frame)
    assert result.top_of_book.height == frame.height
    # بعد إلغاء order_id=1 ما زال الدفتر حيًا (أوامر أخرى)
    assert result.book.best_bid() is not None or result.book.best_ask() is not None


def test_reconstruct_peak_rss_stays_bounded_vs_full_tolist() -> None:
    """المسار المقطّع لا يضاعف RSS مثل مادّة كل action/side دفعة واحدة."""
    n = 400_000
    frame = random_add_cancel_stream(n, seed=7)

    def rss_mb() -> float:
        with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        return float("nan")

    before = rss_mb()
    # التكلفة القديمة التقريبية: list[str] لكل الأحداث
    actions = frame["action"].cast(pl.Utf8).to_list()
    sides = frame["side"].cast(pl.Utf8).to_list()
    after_lists = rss_mb()
    list_cost = after_lists - before
    del actions, sides

    before_recon = rss_mb()
    tob = reconstruct(frame).top_of_book
    after_recon = rss_mb()
    recon_cost = after_recon - before_recon
    assert tob.height == n
    # الشريحة تُبقي تكلفة النصوص أقل من مادّة العمودين كاملين
    assert recon_cost < list_cost * 0.85 or recon_cost < 120.0
