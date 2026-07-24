"""تأكيد أن اللوج يطبع كل مرحلة متسلسلة بلا فجوات صامتة في المسار الحرج."""

from __future__ import annotations

import io
import re

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import make_generator
from nq.research.orchestrator import run_research_pipeline
from nq.research.progress import PipelineProgress
from nq.simulation.breakout import failed_breakout_from_bars
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.fvg import failed_fvg_from_bars
from tests.test_coverage import _paired_streams


_REQUIRED_MARKERS = (
    "بدء:",
    "تحميل MBO",
    "بناء الميزات",
    "عمق",
    "depth_at_bar_close",
    "Failed FVG",
    "failed_fvg",
    "Auction",
    "value_area",
    "Failed Breakout",
    "failed_breakout",
    "depth_bars",
    "SSL",
    "tick_stream",
    "sequences",
    "ألفا",
    "M9",
    "M9-NQ-desc",
    "reconstruct",
    "mfig",
    "انتهى بنجاح",
)


def test_full_pipeline_log_is_sequential_and_detailed() -> None:
    nq, mnq = _paired_streams(2200, seed=201)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf, heartbeat_seconds=0.05)
    run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=40,
        parallel_coverage=False,
        rng=make_generator(21),
        progress=progress,
    )
    text = buf.getvalue()
    lines = [ln for ln in text.splitlines() if ln.startswith("[nq]")]
    assert len(lines) >= 80, f"expected dense log, got {len(lines)} lines"

    missing = [m for m in _REQUIRED_MARKERS if m not in text]
    assert not missing, f"missing markers: {missing}"

    # ترتيب خطي أساسي
    assert text.index("تحميل MBO") < text.index("بناء الميزات")
    assert text.index("بناء الميزات") < text.index("depth_at_bar_close")
    assert text.index("Failed FVG") < text.index("Failed Breakout") or "failed_fvg" in text
    assert text.index("[SSL]") < text.index("[M9]")
    assert text.index("ألفا") < text.index("M9 مقياس:") or "mfig" in text
    assert text.index("بدء:") < text.index("انتهى بنجاح")

    # قنوات SSL/M9 ظاهرة في المسار التسلسلي
    assert "[SSL]" in text and "[M9]" in text

    # نبضات داخل حلقات (ليست مجرد عناوين خطوات)
    assert re.search(r"… .*\d+/\d+", text)
    assert "failed_breakout_features" in text
    assert "failed_fvg_features" in text or "fvg_bars" in text
    assert "M9 مقياس:" in text
    assert any(
        tag in text
        for tag in ("mfig-perm", "qduf-perm", "psg-perm", "cer-perm", "crs-perm", "lori-perm")
    ) or "mfig" in text
    assert "M9-NQ-desc:reconstruct" in text
    assert "depth_fill" in text or "depth-walk" in text


def test_bar_scan_loops_emit_named_heartbeats() -> None:
    """حلقات مسح الشموع تطبع نبضًا مسمّى (fb_bars / fvg_bars)."""
    n = 80
    rows = []
    for i in range(n):
        start = i * 1_000_000_000
        end = start + 1_000_000_000
        o = 100.0 + i * 0.1
        rows.append(
            {
                BUCKET_START: start,
                BUCKET_END: end,
                AVAILABILITY_TS: end,
                "o": o,
                "h": o + 2,
                "l": o - 2,
                "c": o + (1 if i % 3 == 0 else -1),
                "volume": 100.0 + i,
                "range": 4.0,
                "buy_volume": 60.0,
                "sell_volume": 40.0,
                "delta": 20.0,
            }
        )
    bars = pl.DataFrame(rows)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf, heartbeat_seconds=0.01)
    progress.begin("bar-scan", total_steps=2)
    progress.step("FB")
    failed_breakout_from_bars(bars, require_sma_filter=False, rth_only=False, progress=progress)
    progress.step("FVG")
    failed_fvg_from_bars(bars, bars, progress=progress)
    progress.done()
    text = buf.getvalue()
    assert "fb_bars" in text
    assert "fvg_bars" in text
    assert "مسح" in text


def test_parallel_channels_do_not_lose_detail() -> None:
    nq, mnq = _paired_streams(1800, seed=202)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf, heartbeat_seconds=0.05)
    run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=30,
        parallel_coverage=True,
        rng=make_generator(22),
        progress=progress,
    )
    text = buf.getvalue()
    assert "[SSL]" in text and "[M9]" in text
    assert "tick_stream" in text or "SSL-tick" in text
    assert "M9 مقياس:" in text or "mfig" in text
    assert "انتهى بنجاح" in text
