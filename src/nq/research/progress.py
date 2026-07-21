"""طباعة تقدّم الخط الموحّد — سطر بسطر من البداية للنهاية.

كل عملية تُطبع فورًا على stderr (وإلى ``progress.log`` إن وُجد مسار مخرجات).
داخل الحلقات الطويلة يُطبع نبض تقدّم (نسبة + سرعة + ETA) حتى لا يحدث صمت.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m{secs:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{secs:02.0f}s"


def _fmt_rate(done: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "?"
    rate = done / elapsed
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.2f}M/s"
    if rate >= 1_000:
        return f"{rate / 1_000:.1f}k/s"
    return f"{rate:.0f}/s"


@dataclass
class PipelineProgress:
    """مسجّل تقدّم تفاعلي — كل سطر يُFLUSH فورًا.

    Parameters
    ----------
    enabled:
        عند ``False`` لا يطبع شيئًا.
    stream:
        المجرى الأساسي (افتراضيًا stderr).
    log_path:
        إن وُجد يُكتب نفس السطور إلى ملف ``progress.log``.
    heartbeat_seconds:
        الحد الأدنى بين نبضات التقدّم الزمنية داخل الحلقات.
    """

    enabled: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    log_path: Path | None = None
    heartbeat_seconds: float = 2.0
    _title: str = field(default="", init=False, repr=False)
    _total: int | None = field(default=None, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)
    _t0: float = field(default=0.0, init=False, repr=False)
    _step_t0: float = field(default=0.0, init=False, repr=False)
    _current: str = field(default="", init=False, repr=False)
    _log_file: TextIO | None = field(default=None, init=False, repr=False)
    _last_heartbeat: float = field(default=0.0, init=False, repr=False)
    _last_heartbeat_done: int = field(default=-1, init=False, repr=False)

    def attach_log(self, path: Path | str) -> None:
        """يفتح ملف لوج إضافي ويكتب فيه كل السطور."""
        if not self.enabled:
            return
        log = Path(path)
        log.parent.mkdir(parents=True, exist_ok=True)
        if self._log_file is not None:
            self._log_file.close()
        self.log_path = log
        self._log_file = log.open("w", encoding="utf-8")
        self.line(f"ملف التقدّم: {log.resolve()}")

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def begin(self, title: str, *, total_steps: int | None = None) -> None:
        self._title = title
        self._total = total_steps
        self._index = 0
        self._t0 = time.perf_counter()
        self._step_t0 = self._t0
        self._current = ""
        self._last_heartbeat = self._t0
        self._last_heartbeat_done = -1
        self.line(f"========== بدء: {title} ==========")
        if total_steps is not None:
            self.line(f"عدد الخطوات المتوقعة: {total_steps}")

    def step(self, name: str, detail: str = "") -> None:
        """يعلن بدء خطوة جديدة (ويُغلق زمنيًا الخطوة السابقة إن وُجدت)."""
        now = time.perf_counter()
        if self._current:
            elapsed = now - self._step_t0
            self.line(f"  ✓ انتهى: {self._current} ({_fmt_duration(elapsed)})")
        self._index += 1
        self._current = name
        self._step_t0 = now
        self._last_heartbeat = now
        self._last_heartbeat_done = -1
        prefix = self._prefix()
        msg = f"{prefix} {name}"
        if detail:
            msg = f"{msg} — {detail}"
        self.line(f"→ {msg} ...")

    def note(self, message: str) -> None:
        """ملاحظة داخل الخطوة الحالية."""
        self.line(f"  · {message}")

    def op(self, message: str) -> None:
        """عملية واحدة مطبوعة سطرًا بسطر (أدق من note)."""
        self.line(f"    - {message}")

    def heartbeat(
        self,
        done: int,
        total: int,
        *,
        label: str = "",
        force: bool = False,
        every: int | None = None,
    ) -> None:
        """نبض تقدّم داخل حلقة طويلة — نسبة + سرعة + ETA."""
        if not self.enabled:
            return
        if total <= 0:
            return
        done = min(max(done, 0), total)
        now = time.perf_counter()
        count_every = every if every is not None else max(1, total // 100)
        due_count = done == total or done == 0 or (done % count_every == 0)
        due_time = (now - self._last_heartbeat) >= self.heartbeat_seconds
        if not force and not due_count and not due_time:
            return
        if done == self._last_heartbeat_done and not force and done != total:
            return

        elapsed = now - self._step_t0 if self._step_t0 else now - self._t0
        pct = 100.0 * done / total
        rate = _fmt_rate(done, elapsed) if done > 0 else "?"
        remain = "?"
        if done > 0 and elapsed > 0 and done < total:
            eta = (total - done) * (elapsed / done)
            remain = _fmt_duration(eta)
        elif done >= total:
            remain = "0s"
        tag = f"{label} " if label else ""
        self.line(
            f"  … {tag}{done:,}/{total:,} ({pct:5.1f}%) "
            f"· {rate} · مرّ {_fmt_duration(elapsed)} · متبقي ~{remain}"
        )
        self._last_heartbeat = now
        self._last_heartbeat_done = done

    def done(self, detail: str = "") -> None:
        now = time.perf_counter()
        if self._current:
            elapsed = now - self._step_t0
            self.line(f"  ✓ انتهى: {self._current} ({_fmt_duration(elapsed)})")
            self._current = ""
        total = now - self._t0 if self._t0 else 0.0
        suffix = f" — {detail}" if detail else ""
        self.line(
            f"========== انتهى بنجاح: {self._title or 'pipeline'} "
            f"({_fmt_duration(total)}){suffix} =========="
        )
        self.close()

    def fail(self, error: BaseException) -> None:
        now = time.perf_counter()
        step = self._current or "(قبل أي خطوة)"
        step_elapsed = now - self._step_t0 if self._step_t0 else 0.0
        total = now - self._t0 if self._t0 else 0.0
        self.line(
            f"✗ فشل في الخطوة: {step} "
            f"({type(error).__name__}: {error}) "
            f"[خطوة {_fmt_duration(step_elapsed)} | إجمالي {_fmt_duration(total)}]"
        )
        self.line(f"========== توقف بخطأ: {self._title or 'pipeline'} ==========")
        self.close()

    def line(self, message: str) -> None:
        """طباعة سطر واحد فورية (stderr + ملف اللوج)."""
        if not self.enabled:
            return
        text = f"[nq] {message}"
        print(text, file=self.stream, flush=True)
        if self._log_file is not None:
            self._log_file.write(text + "\n")
            self._log_file.flush()

    def _prefix(self) -> str:
        if self._total is not None and self._total > 0:
            return f"[{self._index}/{self._total}]"
        return f"[{self._index}]"


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


def iter_with_progress(
    items: Sequence[object] | Iterable[object],
    progress: PipelineProgress | None,
    *,
    label: str,
    total: int | None = None,
    every: int | None = None,
) -> Iterable[object]:
    """يغلف تكرارًا عاديًا بنبضات تقدّم."""
    log = progress if progress is not None else PipelineProgress(enabled=False)
    seq = list(items) if total is None and not hasattr(items, "__len__") else items
    n = total if total is not None else len(seq)  # type: ignore[arg-type]
    log.op(f"بدء {label}: {n:,} عنصر")
    for i, item in enumerate(seq, start=1):  # type: ignore[arg-type]
        yield item
        log.heartbeat(i, n, label=label, every=every)
    if n > 0:
        log.heartbeat(n, n, label=label, force=True)


__all__ = [
    "PipelineProgress",
    "iter_with_progress",
    "resolve_progress",
]
