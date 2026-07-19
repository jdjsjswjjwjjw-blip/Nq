"""Nq — MBO-only quantitative market-microstructure research engine (NQ / MNQ).

هذه الحزمة تبني النظام محطةً بمحطة وفق ``README.md``. تُفرض في كل وحدة
المبادئ الحاكمة الأربعة:

1. منع التسريب الزمني نهائيًا (zero temporal leakage / point-in-time only).
2. صرامة كمية وعلمية بلا أخطاء (quantitative & scientific rigor).
3. أداء عالٍ لبيانات ضخمة (vectorized, columnar, streaming).
4. MBO فقط (MBO-only data source).
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__"]
