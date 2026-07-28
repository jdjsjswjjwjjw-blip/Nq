"""اختبارات طباعة تقدّم الخط الموحّد."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl

from nq.core.determinism import make_generator
from nq.research.orchestrator import PipelineConfig, run_research_pipeline
from nq.research.progress import PipelineProgress
from nq.simulation.cross_market import cross_market_features
from nq.simulation.depth_lifecycle import depth_at_bar_close
from nq.strategies.breakout_hypothesis import (
    BreakoutHypothesisSpec,
    materialize_breakout_hypotheses,
)
from nq.strategies.depth_entry_filter import (
    attach_depth_path_to_features,
    generate_depth_entry_candidates,
)
from nq.strategies.fvg_hypothesis import search_fail_fvg_hypotheses
from tests.test_coverage import _paired_streams


def test_pipeline_progress_prints_ordered_steps() -> None:
    nq, mnq = _paired_streams(1800, seed=91)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=100,
        parallel_coverage=False,
        rng=make_generator(11),
        progress=progress,
    )
    text = buf.getvalue()
    assert "[nq] ========== بدء: الخط الموحّد MBO → تقرير ==========" in text
    assert "تهيئة الحتمية + تحميل MBO" in text
    assert "بناء الميزات" in text
    assert "تشغيل SSL" in text
    assert "اكتشاف الألفا" in text
    assert "تشغيل المراقب M9" in text
    assert "دمج التقرير الموحّد" in text
    assert "انتهى بنجاح" in text
    assert text.index("بدء:") < text.index("انتهى بنجاح")
    assert text.index("تحميل MBO") < text.index("تشغيل SSL")
    assert text.index("تشغيل SSL") < text.index("اكتشاف الألفا")


def test_pipeline_progress_quiet_prints_nothing() -> None:
    nq, mnq = _paired_streams(1200, seed=92)
    buf = io.StringIO()
    # quiet يتجاوز أي كائن progress صريح
    run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=50,
        parallel_coverage=False,
        rng=make_generator(12),
        progress=PipelineProgress(enabled=True, stream=buf),
        quiet=True,
    )
    assert buf.getvalue() == ""


def test_pipeline_progress_fail_marks_step() -> None:
    progress = PipelineProgress(enabled=True, stream=io.StringIO())
    progress.begin("اختبار فشل", total_steps=2)
    progress.step("خطوة خطرة")
    try:
        raise ValueError("boom")
    except ValueError as exc:
        progress.fail(exc)
    assert isinstance(progress.stream, io.StringIO)
    text = progress.stream.getvalue()
    assert "✗ فشل في الخطوة: خطوة خطرة" in text
    assert "ValueError: boom" in text
    assert "توقف بخطأ" in text


def test_progress_helper_duration_and_notes() -> None:
    buf = io.StringIO()
    p = PipelineProgress(enabled=True, stream=buf)
    p.begin("demo", total_steps=1)
    p.step("عمل", "تفاصيل")
    p.note("ملاحظة داخلية")
    p.op("عملية دقيقة")
    p.heartbeat(50, 100, label="demo_loop", force=True)
    p.done("ok")
    text = buf.getvalue()
    assert "[1/1] عمل — تفاصيل" in text
    assert "ملاحظة داخلية" in text
    assert "عملية دقيقة" in text
    assert "demo_loop" in text
    assert "50/100" in text
    assert "انتهى بنجاح: demo" in text


def test_pipeline_progress_writes_progress_log(tmp_path: Path) -> None:
    nq, mnq = _paired_streams(1200, seed=93)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    out = tmp_path / "run"
    run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=50,
        parallel_coverage=False,
        rng=make_generator(13),
        progress=progress,
        output_dir=out,
    )
    log_file = out / "progress.log"
    assert log_file.is_file()
    text = log_file.read_text(encoding="utf-8")
    assert "tick_stream" in text or "بناء الميزات" in text
    assert "انتهى بنجاح" in text


def test_tick_stream_emits_heartbeats() -> None:
    nq, mnq = _paired_streams(800, seed=94)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    progress.begin("tick", total_steps=1)
    progress.step("stream")
    run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=40,
        parallel_coverage=False,
        rng=make_generator(14),
        progress=progress,
    )
    text = buf.getvalue()
    assert "tick_stream" in text
    assert "آلة الحالة" in text or "بدء آلة الحالة" in text


def test_pipeline_progress_prints_alpha_and_m9_ops() -> None:
    """كل إشارة ألفا + كل مقياس M9 يُطبعان أثناء التشغيل التسلسلي."""
    nq, mnq = _paired_streams(1600, seed=95)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=40,
        parallel_coverage=False,
        rng=make_generator(15),
        progress=progress,
    )
    text = buf.getvalue()
    assert "ألفا [" in text
    assert "M9 مقياس:" in text
    assert "mfig" in text
    assert "qduf" in text
    # مسار tick دائمًا يطبع ops حتى عند تخطّي الطيّات لصغر العيّنة
    assert "SSL-tick" in text


def test_fvg_search_passes_progress_into_ssl(tmp_path: Path) -> None:
    """بحث FVG يمرّر progress إلى SSL-tick ويكتب progress.log."""
    nq, mnq = _paired_streams(2000, seed=96)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    out = tmp_path / "fvg_search"
    search_fail_fvg_hypotheses(
        nq,
        mnq,
        interval_ns=10_000,
        use_ssl_gate=True,
        n_splits=2,
        n_permutations=30,
        ssl_window=3,
        output_dir=out,
        progress=progress,
        rng=make_generator(16),
    )
    text = buf.getvalue()
    assert "بحث فرضيات Failed FVG" in text
    assert "SSL-tick" in text
    assert "walk-forward" in text.lower() or "WF fold" in text
    assert (out / "progress.log").is_file()
    assert "انتهى بنجاح" in text


def test_bucket_ssl_emits_fold_progress() -> None:
    nq, mnq = _paired_streams(1600, seed=97)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    cfg = PipelineConfig(
        interval_ns=10_000,
        n_permutations=40,
        parallel_coverage=False,
        ssl_mode="bucket",
    )
    run_research_pipeline(
        nq,
        mnq,
        config=cfg,
        rng=make_generator(17),
        progress=progress,
    )
    text = buf.getvalue()
    assert "SSL-bucket" in text
    assert "ألفا [" in text
    assert "M9 مقياس:" in text


def test_progress_channel_prefixes_lines() -> None:
    buf = io.StringIO()
    p = PipelineProgress(enabled=True, stream=buf)
    with p.channel("SSL"):
        p.op("داخل SSL")
    with p.channel("M9"):
        p.op("داخل M9")
    text = buf.getvalue()
    assert "[SSL]" in text
    assert "[M9]" in text
    assert "داخل SSL" in text
    assert "داخل M9" in text
    assert "قناة [SSL] بدأت" in text
    assert "قناة [M9] انتهت" in text


def test_depth_and_materialize_emit_heartbeats() -> None:
    nq, mnq = _paired_streams(600, seed=101)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    progress.begin("depth+mat", total_steps=2)
    progress.step("عمق")
    depth_at_bar_close(nq, interval_ns=10_000, n_levels=3, progress=progress)
    clock = cross_market_features(nq, mnq, interval_ns=10_000, lead_lag_window=2, latency_ns=0)
    progress.step("تجسيد")
    specs = (
        BreakoutHypothesisSpec(
            name="t1",
            signal_interval_ns=10_000,
            trend_interval_ns=20_000,
            lookback=3,
            range_mult=1.0,
            vol_mode="bar",
            vol_window=3,
            vol_mult=1.0,
            require_sma_filter=False,
        ),
        BreakoutHypothesisSpec(
            name="t2",
            signal_interval_ns=10_000,
            trend_interval_ns=20_000,
            lookback=4,
            range_mult=1.1,
            vol_mode="delta",
            vol_window=3,
            vol_mult=1.1,
            require_sma_filter=False,
        ),
    )
    materialize_breakout_hypotheses(nq, specs, clock=clock, progress=progress)
    progress.done()
    text = buf.getvalue()
    assert "depth_bars" in text or "depth_at_bar_close" in text
    assert "materialize_FB" in text or "فرضية [" in text
    assert "… " in text


def test_pipeline_progress_prints_depth_ops() -> None:
    nq, mnq = _paired_streams(1600, seed=102)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=30,
        parallel_coverage=False,
        rng=make_generator(18),
        progress=progress,
    )
    text = buf.getvalue()
    assert "depth_at_bar_close" in text or "عمق" in text
    assert "[SSL]" in text
    assert "[M9]" in text
    assert "mfig-perm" in text or "M9 مقياس:" in text


def test_heartbeat_channels_do_not_mute_each_other() -> None:
    """نبض قناة لا يكتم نبض قناة أخرى عند نفس العداد."""
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf, heartbeat_seconds=60.0)
    progress.begin("قنوات", total_steps=1)
    progress.step("متوازي")
    with progress.channel("SSL"):
        progress.heartbeat(10, 100, label="loop", force=True)
    with progress.channel("M9"):
        progress.heartbeat(10, 100, label="loop", force=True)
    text = buf.getvalue()
    assert "[SSL]" in text and "[M9]" in text
    assert text.count("10/100") >= 2


def test_depth_filter_progress_emits_asof_and_generate() -> None:
    nq, _ = _paired_streams(900, seed=55)
    clock = cross_market_features(nq, nq, interval_ns=10_000, lead_lag_window=2, latency_ns=0)
    features = clock.with_columns(pl.lit(1.0).alias("sig"))
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    progress.begin("عمق", total_steps=1)
    progress.step("فلتر")
    features = attach_depth_path_to_features(features, nq, interval_ns=10_000, progress=progress)
    generate_depth_entry_candidates(features, ["sig"], progress=progress)
    progress.done()
    text = buf.getvalue()
    assert "depth_path" in text or "depth_event_path" in text
    assert "asof مسار العمق" in text
    assert "توليد مرشّحي عمق" in text
