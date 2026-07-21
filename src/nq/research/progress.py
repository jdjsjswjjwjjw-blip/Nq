"""طباعة تقدّم الخط الموحّد — خطوة بخطوة من البداية للنهاية.

الهدف: يعرف المشغّل وصلت فين، وكم استغرقت كل خطوة، وأين فشل التشغيل
لو حصل خطأ. الطباعة على stderr حتى لا تختلط مع تقرير Markdown على stdout.
"""

from __future__ import annotations

import sys
import time
from collections.abc import TextIO
from dataclasses import dataclass, field


def _fmt_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60.0)
    return f"{int(minutes)}m{secs:04.1f}s"


@dataclass
class PipelineProgress:
    """مسجّل تقدّم بسيط للتشغيل التفاعلي.

    Parameters
    ----------
    enabled:
        عند ``False`` لا يطبع شيئًا (مناسب للاختبارات الصامتة).
    stream:
        مجرى الكتابة — افتراضيًا ``stderr``.
    """

    enabled: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    _title: str = field(default="", init=False, repr=False)
    _total: int | None = field(default=None, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)
    _t0: float = field(default=0.0, init=False, repr=False)
    _step_t0: float = field(default=0.0, init=False, repr=False)
    _current: str = field(default="", init=False, repr=False)

    def begin(self, title: str, *, total_steps: int | None = None) -> None:
        self._title = title
        self._total = total_steps
        self._index = 0
        self._t0 = time.perf_counter()
        self._step_t0 = self._t0
        self._current = ""
        self._emit(f"========== بدء: {title} ==========")
        if total_steps is not None:
            self._emit(f"عدد الخطوات المتوقعة: {total_steps}")

    def step(self, name: str, detail: str = "") -> None:
        """يعلن بدء خطوة جديدة (ويُغلق زمنيًا الخطوة السابقة إن وُجدت)."""
        now = time.perf_counter()
        if self._current:
            elapsed = now - self._step_t0
            self._emit(f"  ✓ انتهى: {self._current} ({_fmt_duration(elapsed)})")
        self._index += 1
        self._current = name
        self._step_t0 = now
        prefix = self._prefix()
        msg = f"{prefix} {name}"
        if detail:
            msg = f"{msg} — {detail}"
        self._emit(f"→ {msg} ...")

    def note(self, message: str) -> None:
        """ملاحظة داخل الخطوة الحالية بدون إغلاقها."""
        self._emit(f"  · {message}")

    def done(self, detail: str = "") -> None:
        now = time.perf_counter()
        if self._current:
            elapsed = now - self._step_t0
            self._emit(f"  ✓ انتهى: {self._current} ({_fmt_duration(elapsed)})")
            self._current = ""
        total = now - self._t0 if self._t0 else 0.0
        suffix = f" — {detail}" if detail else ""
        self._emit(
            f"========== انتهى بنجاح: {self._title or 'pipeline'} "
            f"({_fmt_duration(total)}){suffix} =========="
        )

    def fail(self, error: BaseException) -> None:
        now = time.perf_counter()
        step = self._current or "(قبل أي خطوة)"
        step_elapsed = now - self._step_t0 if self._step_t0 else 0.0
        total = now - self._t0 if self._t0 else 0.0
        self._emit(
            f"✗ فشل في الخطوة: {step} "
            f"({type(error).__name__}: {error}) "
            f"[خطوة {_fmt_duration(step_elapsed)} | إجمالي {_fmt_duration(total)}]"
        )
        self._emit(
            f"========== توقف بخطأ: {self._title or 'pipeline'} =========="
        )

    def _prefix(self) -> str:
        if self._total is not None and self._total > 0:
            return f"[{self._index}/{self._total}]"
        return f"[{self._index}]"

    def _emit(self, message: str) -> None:
        if not self.enabled:
            return
        print(f"[nq] {message}", file=self.stream, flush=True)


def resolve_progress(
    progress: PipelineProgress | bool | None,
    *,
    quiet: bool = False,
) -> PipelineProgress:
    """يحوّل إعدادات الواجهة إلى ``PipelineProgress`` جاهز."""
    if quiet or progress is False:
        return PipelineProgress(enabled=False)
    if isinstance(progress, PipelineProgress):
        return progress
    return PipelineProgress(enabled=True)


__all__ = [
    "PipelineProgress",
    "resolve_progress",
]
