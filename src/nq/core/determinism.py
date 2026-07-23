"""الحتمية وقابلية إعادة الإنتاج (Determinism & Reproducibility).

كل نتيجة علمية يجب أن تكون قابلة لإعادة الإنتاج بدقّة. تُثبّت هنا بذور
مولّدات الأرقام العشوائية عبر ``random`` و ``numpy`` من بذرة عامة واحدة.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """تثبّت كل مصادر العشوائية من بذرة واحدة وتُعيد مولّد ``numpy`` حتميًّا.

    * ``random`` (مكتبة بايثون القياسية).
    * ``numpy`` (الحالة العامة القديمة + ``PYTHONHASHSEED``).

    يُعاد ``numpy.random.Generator`` مُبذّر ليُستخدم صراحةً (النمط المُوصى به)
    بدل الاعتماد على الحالة العامة.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # تثبيت الحالة العامة القديمة مقصود لضمان حتمية أي كود يعتمد عليها.
    np.random.seed(seed)  # noqa: NPY002
    return np.random.default_rng(seed)


def make_generator(seed: int) -> np.random.Generator:
    """يُنشئ ``numpy.random.Generator`` حتميًّا دون لمس الحالة العامة.

    يُفضّل هذا في المكوّنات المتوازية/المعزولة لتفادي التداخل بين المكوّنات.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.default_rng(seed)
