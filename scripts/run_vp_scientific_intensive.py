#!/usr/bin/env python3
"""يشغّل البطارية العلمية المكثّفة لـ VP ويكتب تقرير نتيجة كامل.

استخدام:
    python scripts/run_vp_scientific_intensive.py
    python scripts/run_vp_scientific_intensive.py \
        --out /opt/cursor/artifacts/vp_scientific_report.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/vp_scientific_report.md"),
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    suites = [
        "tests/test_vp_scientific_intensive.py",
        "tests/test_auction.py",
        "tests/test_vp_auction_strategy.py",
        "tests/test_liquidity_edge.py",
        "tests/test_leakage.py",
        "tests/test_reconstruction.py",
        "tests/test_reconstruction_causality.py",
        "tests/test_depth_lifecycle.py",
        "tests/test_coverage.py",
        "tests/test_vp_month_aggregate.py",
    ]

    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *suites,
        "-v",
        "--tb=short",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    # full suite too
    t1 = time.perf_counter()
    full = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line"],
        capture_output=True,
        text=True,
        check=False,
    )
    full_elapsed = time.perf_counter() - t1

    body = f"""# تقرير علمي مكثّف — استراتيجية Volume Profile / Auction

## الهدف
تعزيز آلة استخراج **setup يومي** من مسار الفوليوم (كسر → بناء → انطلاق → إدج)
والتأكد أن الآلة تمشي صح قبل أي اعتماد تداولي.

## نطاق التحقق
1. مبادئ زمنية (سببية / availability / تشويش لاحق)
2. FSM: كسر ≠ توسّع؛ بناء حول الفوليوم؛ لا قتل مبكر
3. دفتر موحّد + تضليل مرة واحدة
4. WF قبل التنفيذ + حتمية البحث
5. شبكة إدج هيكلية (R:R)
6. ضغط يوم اصطناعي كثيف

## نتيجة البطارية العلمية المركّزة
- exit_code: `{proc.returncode}`
- wall_s: `{elapsed:.2f}`

```
{proc.stdout[-12000:]}
```

### stderr (إن وُجد)
```
{proc.stderr[-4000:]}
```

## نتيجة pytest الكامل للمشروع
- exit_code: `{full.returncode}`
- wall_s: `{full_elapsed:.2f}`

```
{full.stdout[-4000:]}
```

## حكم
"""
    if proc.returncode == 0 and full.returncode == 0:
        body += (
            "**PASS — الآلة العلمية خضراء.**\n\n"
            "هذا يثبت صحّة المسار السببي وطبقات VP/FSM/Edge على بيانات اصطناعية "
            "مُتحكَّم بها. **لا يساوي إثبات ربحية يومية على بيانات حية** — الخطوة "
            "التالية بعد السحب من `main`: تشغيل شهر MES الحقيقي بـ `--jobs 4..6`.\n"
        )
        verdict = 0
    else:
        body += "**FAIL — راجع المخرجات أعلاه قبل أي اعتماد تداولي.**\n"
        verdict = 1

    args.out.write_text(body, encoding="utf-8")
    print(body)
    print(f"\n[wrote] {args.out}")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
