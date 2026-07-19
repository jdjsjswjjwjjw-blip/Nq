"""أداة اختبار التسريب الزمني العامة (General Temporal-Leakage Test Tool).

هذه الأداة هي البوابة الحاكمة الأولى في النظام: تُستخدم في كل محطة لإثبات أن
أي حساب/ميزة/نموذج يعتمد **فقط** على الماضي (point-in-time) ولا يتسرّب إليه
المستقبل بأي شكل.

توفّر الأداة مستويين من الفحص:

1. فحوص بنيوية (assertions) سريعة ورخيصة:
   * ``assert_causal_order`` — الطوابع الزمنية غير متناقصة (ترتيب سببي).
   * ``assert_availability_not_before_event`` — الإتاحة لا تسبق وقوع الحدث.
   * ``assert_temporal_split`` — تقسيم زمني صارم مع فترة حظر (embargo).

2. فحص سلوكي قوي (الأداة الذهبية):
   * ``detect_leakage_by_perturbation`` — يُثبت السببية عمليًا عبر تشويش
     المستقبل والتأكد من عدم تغيّر مخرجات الماضي إطلاقًا.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]

#: أقل عدد أحداث يُجرى عليه اختبار التسريب (نحتاج ماضيًا ومستقبلًا على الأقل).
_MIN_EVENTS_FOR_TEST: int = 2


class LeakageError(AssertionError):
    """يُرفع عند اكتشاف تسريب زمني (تأثير المستقبل على الماضي)."""


def _as_1d(values: Sequence[float] | npt.NDArray[np.generic]) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D sequence, got shape {arr.shape}")
    return arr


def assert_causal_order(
    timestamps: Sequence[float] | npt.NDArray[np.generic],
    *,
    strict: bool = False,
) -> None:
    """يتحقق من أن الطوابع الزمنية مرتّبة سببيًا (غير متناقصة).

    عند ``strict=True`` يشترط تزايدًا صارمًا (لا تكرار للطابع نفسه)، وهو مفيد
    عندما يُفترض أن يكون كل حدث فريدًا زمنيًا (بعد دمج التسلسل).
    """
    ts = _as_1d(timestamps)
    if ts.size <= 1:
        return
    diffs = np.diff(ts)
    bad = diffs < 0 if not strict else diffs <= 0
    if bool(np.any(bad)):
        idx = int(np.argmax(bad))
        kind = "strictly increasing" if strict else "non-decreasing"
        raise LeakageError(
            f"causal-order violation: timestamps not {kind} at index {idx + 1} "
            f"({ts[idx]!r} -> {ts[idx + 1]!r})."
        )


def assert_availability_not_before_event(
    event_ts: Sequence[float] | npt.NDArray[np.generic],
    availability_ts: Sequence[float] | npt.NDArray[np.generic],
) -> None:
    """يتحقق من ``availability_ts >= event_ts`` عنصرًا بعنصر (point-in-time).

    لا يمكن لأي معلومة أن تصبح متاحة قبل وقوع الحدث الذي اشتُقّت منه.
    """
    ev = _as_1d(event_ts)
    av = _as_1d(availability_ts)
    if ev.shape != av.shape:
        raise ValueError(f"event_ts and availability_ts must align, got {ev.shape} vs {av.shape}")
    bad = av < ev
    if bool(np.any(bad)):
        idx = int(np.argmax(bad))
        raise LeakageError(
            f"point-in-time violation: availability_ts < event_ts at index {idx} "
            f"({av[idx]!r} < {ev[idx]!r})."
        )


def assert_temporal_split(
    train_timestamps: Sequence[float] | npt.NDArray[np.generic],
    test_timestamps: Sequence[float] | npt.NDArray[np.generic],
    *,
    embargo: float = 0.0,
) -> None:
    """يتحقق من تقسيم زمني صارم: نهاية التدريب + الحظر <= بداية الاختبار.

    يمنع تداخل نافذتي التدريب والاختبار (walk-forward) ويفرض فترة حظر
    (``embargo``) لعزل أي أثر متبقٍّ عند الحدود.
    """
    if embargo < 0:
        raise ValueError(f"embargo must be non-negative, got {embargo}")
    train = _as_1d(train_timestamps)
    test = _as_1d(test_timestamps)
    if train.size == 0 or test.size == 0:
        return
    train_end = float(np.max(train))
    test_start = float(np.min(test))
    if test_start < train_end + embargo:
        raise LeakageError(
            "temporal-split violation: test window overlaps train (+embargo). "
            f"train_end={train_end!r}, embargo={embargo!r}, test_start={test_start!r}."
        )


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """تقرير نتيجة اختبار التسريب بالتشويش."""

    leaked: bool
    n_trials: int
    cuts_tested: tuple[int, ...]
    max_abs_diff: float
    first_violation_cut: int | None

    def raise_for_leakage(self) -> None:
        """يرفع ``LeakageError`` إذا اكتُشف تسريب."""
        if self.leaked:
            raise LeakageError(
                "temporal leakage detected: perturbing the future changed a past "
                f"output at cut={self.first_violation_cut} "
                f"(max_abs_diff={self.max_abs_diff:.3e})."
            )


def _default_perturb(future: FloatArray, rng: np.random.Generator) -> FloatArray:
    """تشويش افتراضي: إعادة عيّنة داخل التوزيع مع تشويش قوي يضمن الاختلاف."""
    if future.size == 0:
        return future
    resampled = rng.permutation(future, axis=0)
    scale = float(np.std(future)) or 1.0
    return resampled + rng.standard_normal(future.shape) * scale + 1.0


def detect_leakage_by_perturbation(
    compute_fn: Callable[[FloatArray], FloatArray],
    data: Sequence[float] | npt.NDArray[np.generic],
    *,
    cuts: Sequence[int] | None = None,
    n_trials: int = 16,
    rng: np.random.Generator | None = None,
    perturb_fn: Callable[[FloatArray, np.random.Generator], FloatArray] | None = None,
    atol: float = 1e-9,
    rtol: float = 0.0,
) -> LeakageReport:
    """يُثبت سببية ``compute_fn`` عمليًا عبر تشويش المستقبل.

    الفكرة: دالة سببية يجب أن يعتمد مخرجها عند الزمن ``t`` على المدخلات حتى
    ``t`` فقط. لذا عند تثبيت الماضي وتشويش المستقبل (المدخلات بعد نقطة القطع
    ``cut``)، يجب ألّا تتغيّر مخرجات الماضي ``output[: cut + 1]`` إطلاقًا.
    أي تغيّر = تسريب زمني.

    المعاملات:
        compute_fn: دالة تأخذ مصفوفة مدخلات مرتّبة زمنيًا وتُعيد مخرجًا
            بمحاذاة صف-بصف (نفس الطول).
        data: المدخل المرجعي (1-D، مرتّب سببيًا).
        cuts: نقاط القطع المُختبَرة؛ افتراضيًا أرباع طول البيانات.
        n_trials: عدد محاولات التشويش لكل نقطة قطع.
        rng: مولّد عشوائي حتمي (للتكرار).
        perturb_fn: دالة تشويش المستقبل؛ افتراضيًا إعادة عيّنة + تشويش قوي.
        atol, rtol: تفاوتات المقارنة العددية.

    يُعيد ``LeakageReport`` (استخدم ``.raise_for_leakage()`` للفرض).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    x = _as_1d(data)
    n = x.size
    if n < _MIN_EVENTS_FOR_TEST:
        return LeakageReport(
            leaked=False,
            n_trials=n_trials,
            cuts_tested=(),
            max_abs_diff=0.0,
            first_violation_cut=None,
        )

    generator = rng if rng is not None else np.random.default_rng(0)
    perturb = perturb_fn if perturb_fn is not None else _default_perturb

    if cuts is None:
        cuts = [n // 4, n // 2, (3 * n) // 4]
    valid_cuts = tuple(sorted({c for c in cuts if 0 <= c < n - 1}))

    baseline = np.asarray(compute_fn(x), dtype=np.float64)
    if baseline.shape[0] != n:
        raise ValueError(
            f"compute_fn output length {baseline.shape[0]} != input length {n} "
            "(output must align row-by-row with the causal input)."
        )

    max_abs_diff = 0.0
    first_violation_cut: int | None = None

    for cut in valid_cuts:
        prefix = slice(0, cut + 1)
        base_prefix = baseline[prefix]
        for _ in range(n_trials):
            perturbed = x.copy()
            perturbed[cut + 1 :] = perturb(perturbed[cut + 1 :], generator)
            out = np.asarray(compute_fn(perturbed), dtype=np.float64)
            diff = float(np.max(np.abs(out[prefix] - base_prefix))) if cut >= 0 else 0.0
            max_abs_diff = max(max_abs_diff, diff)
            if not np.allclose(out[prefix], base_prefix, atol=atol, rtol=rtol):
                first_violation_cut = cut
                break
        if first_violation_cut is not None:
            break

    return LeakageReport(
        leaked=first_violation_cut is not None,
        n_trials=n_trials,
        cuts_tested=valid_cuts,
        max_abs_diff=max_abs_diff,
        first_violation_cut=first_violation_cut,
    )
