"""اختبارات طباعة تقدّم الخط الموحّد."""

from __future__ import annotations

import io

from nq.core.determinism import make_generator
from nq.research.orchestrator import run_research_pipeline
from nq.research.progress import PipelineProgress
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


def test_pipeline_progress_writes_progress_log(tmp_path) -> None:
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
    assert "SSL-tick fold" in text or "SSL-bucket fold" in text


def test_fvg_search_passes_progress_into_ssl(tmp_path) -> None:
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
    from nq.research.orchestrator import PipelineConfig

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
    assert "fold" in text
