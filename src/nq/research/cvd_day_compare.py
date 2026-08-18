"""مقارنة يومين بنفس تعريفات CVD. وصف، بلا قفل إشارة.

نفس العتبات الأوليّة: تعاكس قوي 500/80، توسّع 1.5× وسيط مدى 5د،
وقطع الانفجار 1000/0.20/10 من يوم 17 كـ OOS وليست قفلًا.
لا تُنسخ ساعات 10:24/11:51/12:45 إلى يوم آخر.
جلسة قصيرة (افتتاح الأحد) ليست يوم RTH مكافئ.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import EVENT_TS
from nq.research.clock_flow import (
    EXPAND_MULT,
    HYP_BURST_CVD,
    HYP_BURST_IMB,
    HYP_BURST_RANGE,
    LAYER_ID,
    STRONG_MNQ_ABS,
    STRONG_NQ_ABS,
    TZ_NAME,
    scan_cvd_align_expansion,
    scan_cvd_opposite,
    scan_tape_bins,
    write_cvd_align_expansion_report,
    write_cvd_opposite_report,
    write_tape_bins_report,
)
from nq.research.mbo_trade_overlap import prepare_mbo_events, prepare_trades_tape

LAYER_COMPARE: Final = "cvd_day_compare"
RTH_START: Final = dt.time(9, 30)
RTH_END: Final = dt.time(16, 0)
SHORT_SESSION_HOURS: Final = 6.0
_TRADE = MboAction.TRADE.value
_HOUR_NS: Final = 3_600_000_000_000
COMPARE_KEYS: Final[tuple[str, ...]] = (
    "n_mbo",
    "n_mnq_t",
    "n_nq_t",
    "t_hours",
    "has_rth",
    "n_bins_5m",
    "n_delta_opposite",
    "delta_opposite_rate",
    "n_strong",
    "n_aligned",
    "n_wide",
    "wide_rate",
    "n_mnq_joins_nq",
    "n_rth_mnq_joins_nq",
    "n_rth_mnq_joins_nq_wide",
    "rth_mnq_joins_wide_rate",
    "median_bin_range",
    "n_bins_60s",
    "n_hyp_cvd",
    "n_hyp_all_three",
    "hyp_cvd_rate",
    "hyp_all_three_rate",
    "median_next5m_hyp_cvd",
    "median_next5m_all_three",
)


class DayScan(NamedTuple):
    """نتائج يوم واحد: تغطية + تعاكس + توافق + انفجار 60ث."""

    label: str
    coverage: dict[str, Any]
    opposite: pl.DataFrame
    opposite_diag: dict[str, Any]
    align: pl.DataFrame
    align_diag: dict[str, Any]
    tape: pl.DataFrame
    tape_diag: dict[str, Any]
    summary: dict[str, Any]


def _fmt_ts(ts: int | None, tz_name: str) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts / 1_000_000_000, tz=ZoneInfo(tz_name)).isoformat()


def _clock_time(clock: str | None) -> dt.time | None:
    if not clock:
        return None
    stamp = dt.datetime.fromisoformat(str(clock))
    return stamp.time()


def in_rth(clock: str | None) -> bool:
    """``[09:30, 16:00)`` بتوقيت ساعة الصف، عادة نيويورك."""

    stamp = _clock_time(clock)
    if stamp is None:
        return False
    return RTH_START <= stamp < RTH_END


def _t_span(frame: pl.DataFrame, tz_name: str) -> dict[str, Any]:
    trades = frame.filter(pl.col("action") == _TRADE)
    if trades.height == 0:
        return {"n_t": 0, "tmin": None, "tmax": None, "hours": 0.0, "has_rth": False}
    lo = int(trades.select(pl.col(EVENT_TS).min()).item())
    hi = int(trades.select(pl.col(EVENT_TS).max()).item())
    clock = (
        pl.from_epoch(pl.col(EVENT_TS).cast(pl.Int64), time_unit="ns")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(tz_name)
    )
    minutes = clock.dt.hour().cast(pl.Int32) * 60 + clock.dt.minute().cast(pl.Int32)
    rth_lo = RTH_START.hour * 60 + RTH_START.minute
    rth_hi = RTH_END.hour * 60 + RTH_END.minute
    has_rth = trades.filter((minutes >= rth_lo) & (minutes < rth_hi)).height > 0
    return {
        "n_t": trades.height,
        "tmin": _fmt_ts(lo, tz_name),
        "tmax": _fmt_ts(hi, tz_name),
        "hours": float(hi - lo) / float(_HOUR_NS),
        "has_rth": has_rth,
    }


def describe_tape_coverage(
    mnq_mbo: pl.DataFrame,
    mnq_trades: pl.DataFrame,
    nq_trades: pl.DataFrame,
    *,
    tz_name: str = TZ_NAME,
    label: str = "day",
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> dict[str, Any]:
    """مدى ``T`` وعدد الصفوف. يفضح ملفًا ناقصًا قبل مقارنة RTH."""

    book = prepare_mbo_events(mnq_mbo)
    tape = prepare_trades_tape(mnq_trades)
    nq = prepare_trades_tape(nq_trades)
    if start_ts is not None:
        book = book.filter(pl.col(EVENT_TS) >= int(start_ts))
        tape = tape.filter(pl.col(EVENT_TS) >= int(start_ts))
        nq = nq.filter(pl.col(EVENT_TS) >= int(start_ts))
    if end_ts is not None:
        book = book.filter(pl.col(EVENT_TS) < int(end_ts))
        tape = tape.filter(pl.col(EVENT_TS) < int(end_ts))
        nq = nq.filter(pl.col(EVENT_TS) < int(end_ts))
    mbo_t = _t_span(book, tz_name)
    tape_t = _t_span(tape, tz_name)
    nq_t = _t_span(nq, tz_name)
    hours = float(mbo_t["hours"])
    has_rth = bool(mbo_t["has_rth"])
    if hours < SHORT_SESSION_HOURS:
        coverage_class = "short_session"
    elif not has_rth:
        coverage_class = "overnight_only"
    else:
        coverage_class = "includes_rth"
    return {
        "layer": LAYER_COMPARE,
        "label": label,
        "tz": tz_name,
        "n_mbo": book.height,
        "n_mnq_t": int(mbo_t["n_t"]),
        "n_mnq_tape_t": int(tape_t["n_t"]),
        "n_nq_t": int(nq_t["n_t"]),
        "mnq_tmin": mbo_t["tmin"],
        "mnq_tmax": mbo_t["tmax"],
        "nq_tmin": nq_t["tmin"],
        "nq_tmax": nq_t["tmax"],
        "t_hours": hours,
        "has_rth": has_rth,
        "coverage_class": coverage_class,
        "not_matched_rth": coverage_class != "includes_rth",
        "not_pattern": True,
        "not_copy_clocks": True,
    }


def _hm(clock: object) -> str:
    text = str(clock) if clock is not None else ""
    if "T" in text:
        return text.split("T", 1)[1][:5]
    return text


def _rth_join_lines(table: pl.DataFrame) -> list[str]:
    lines = [
        "## RTH MNQ→NQ (descriptive)",
        "",
        "| opp | align | wide | move5 | rng5 | MNQΔ opp | NQΔ opp | MNQΔ al |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    n = 0
    if table.height:
        for row in table.iter_rows(named=True):
            if not _mnq_joins_nq(row):
                continue
            clock = str(row["opp_clock"]) if row.get("opp_clock") is not None else None
            if not in_rth(clock):
                continue
            n += 1
            move = float(row["move_5"])
            span = float(row["range_5"])
            move_s = "nan" if math.isnan(move) else f"{move:.2f}"
            span_s = "nan" if math.isnan(span) else f"{span:.2f}"
            lines.append(
                f"| {_hm(row['opp_clock'])} | {_hm(row['align_clock'])} | "
                f"{row['wide_vs_median']} | {move_s} | {span_s} | "
                f"{row['mnq_cvd_delta']} | {row['nq_cvd_delta']} | {row['align_mnq_delta']} |"
            )
    if n == 0:
        lines.append("(none)")
    lines.extend(
        [
            "",
            "Not a lock. Clocks are this day's events, not copied from another day.",
            "",
        ]
    )
    return lines


def _mnq_joins_nq(row: Mapping[str, Any]) -> bool:
    if not bool(row.get("aligned")):
        return False
    align_sign = row.get("align_sign")
    nq_sign = row.get("nq_opp_sign")
    if align_sign is None or nq_sign is None:
        return False
    return int(align_sign) == int(nq_sign)


def _rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _median(values: list[float]) -> float:
    finite = [x for x in values if not math.isnan(x)]
    if not finite:
        return float("nan")
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def summarize_align_day(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """عدادات التوافق/التوسّع بما فيها MNQ→NQ في RTH."""

    n_strong = int(diagnostics.get("n_strong_episodes") or 0)
    n_aligned = int(diagnostics.get("n_aligned") or 0)
    n_wide = int(diagnostics.get("n_wide_vs_median") or 0)
    n_joins = 0
    n_rth_joins = 0
    n_rth_joins_wide = 0
    if table.height:
        for row in table.iter_rows(named=True):
            if not _mnq_joins_nq(row):
                continue
            n_joins += 1
            if in_rth(str(row["opp_clock"]) if row.get("opp_clock") is not None else None):
                n_rth_joins += 1
                if bool(row.get("wide_vs_median")):
                    n_rth_joins_wide += 1
    median_range = diagnostics.get("median_bin_range")
    return {
        "n_bins_5m": int(diagnostics.get("n_bins") or 0),
        "n_delta_opposite": int(diagnostics.get("n_delta_opposite") or 0),
        "delta_opposite_rate": _rate(
            int(diagnostics.get("n_delta_opposite") or 0),
            int(diagnostics.get("n_bins") or 0),
        ),
        "n_strong": n_strong,
        "n_aligned": n_aligned,
        "n_wide": n_wide,
        "wide_rate": _rate(n_wide, n_aligned),
        "n_mnq_joins_nq": n_joins,
        "n_rth_mnq_joins_nq": n_rth_joins,
        "n_rth_mnq_joins_nq_wide": n_rth_joins_wide,
        "rth_mnq_joins_wide_rate": _rate(n_rth_joins_wide, n_rth_joins),
        "median_bin_range": (
            float(median_range) if isinstance(median_range, int | float) else float("nan")
        ),
        "strong_mnq": int(diagnostics.get("strong_mnq") or STRONG_MNQ_ABS),
        "strong_nq": int(diagnostics.get("strong_nq") or STRONG_NQ_ABS),
        "expand_mult": float(diagnostics.get("expand_mult") or EXPAND_MULT),
        "not_pattern": True,
    }


def summarize_tape_hyp(table: pl.DataFrame, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """قطع الانفجار 1000/0.20/10 كـ OOS، ليست قفلًا."""

    n_bins = int(diagnostics.get("n_bins") or 0)
    n_cvd = int(diagnostics.get("n_hyp_cvd") or 0)
    n_all = int(diagnostics.get("n_hyp_all_three") or 0)
    next_cvd: list[float] = []
    next_all: list[float] = []
    if table.height:
        cvd_f = table["mnq_cvd_delta"].abs() >= HYP_BURST_CVD
        imb_f = table["mnq_imb"].abs() >= HYP_BURST_IMB
        rng_f = table["range"] >= HYP_BURST_RANGE
        flagged = table.with_columns(
            cvd_f.alias("_cvd"),
            (cvd_f & imb_f & rng_f).alias("_all"),
        )
        for row in flagged.iter_rows(named=True):
            nxt = float(row["next_move_5m"])
            if bool(row["_cvd"]):
                next_cvd.append(nxt)
            if bool(row["_all"]):
                next_all.append(nxt)
    return {
        "n_bins_60s": n_bins,
        "n_hyp_cvd": n_cvd,
        "n_hyp_all_three": n_all,
        "hyp_cvd_rate": _rate(n_cvd, n_bins),
        "hyp_all_three_rate": _rate(n_all, n_bins),
        "median_next5m_hyp_cvd": _median(next_cvd),
        "median_next5m_all_three": _median(next_all),
        "hyp_cvd": HYP_BURST_CVD,
        "hyp_imb": HYP_BURST_IMB,
        "hyp_range": HYP_BURST_RANGE,
        "not_burst_lock": True,
        "oos_of_0817_cuts": True,
    }


def merge_day_summary(
    coverage: Mapping[str, Any],
    align: Mapping[str, Any],
    tape: Mapping[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "layer": LAYER_COMPARE,
        "label": coverage.get("label"),
        "coverage_class": coverage.get("coverage_class"),
        "not_matched_rth": coverage.get("not_matched_rth"),
        "not_pattern": True,
        "not_copy_clocks": True,
        "not_lstm": True,
        "not_live_overlay": True,
        "n_mbo": coverage.get("n_mbo"),
        "n_mnq_t": coverage.get("n_mnq_t"),
        "n_nq_t": coverage.get("n_nq_t"),
        "t_hours": coverage.get("t_hours"),
        "has_rth": coverage.get("has_rth"),
        "mnq_tmin": coverage.get("mnq_tmin"),
        "mnq_tmax": coverage.get("mnq_tmax"),
    }
    out.update(align)
    out.update(tape)
    return out


def scan_cvd_day(
    mnq_mbo: pl.DataFrame,
    mnq_trades: pl.DataFrame,
    nq_trades: pl.DataFrame,
    *,
    label: str,
    tz_name: str = TZ_NAME,
    bin_s: int = 300,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> DayScan:
    """تعاكس 5د + توافق/توسّع + شرائح 60ث. بلا ساعات منسوخة."""

    coverage = describe_tape_coverage(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        tz_name=tz_name,
        label=label,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    opposite, opp_diag = scan_cvd_opposite(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        bin_s=bin_s,
        tz_name=tz_name,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    align, align_diag = scan_cvd_align_expansion(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        bin_s=bin_s,
        tz_name=tz_name,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    tape, tape_diag = scan_tape_bins(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        bin_s=60,
        tz_name=tz_name,
        start_ts=start_ts,
        end_ts=end_ts,
        inner_s=5,
        label=f"{label}_60s",
    )
    summary = merge_day_summary(
        coverage,
        summarize_align_day(align, align_diag),
        summarize_tape_hyp(tape, tape_diag),
    )
    summary["label"] = label
    summary["start_ts"] = start_ts
    summary["end_ts"] = end_ts
    return DayScan(
        label=label,
        coverage=coverage,
        opposite=opposite,
        opposite_diag=dict(opp_diag),
        align=align,
        align_diag=dict(align_diag),
        tape=tape,
        tape_diag=dict(tape_diag),
        summary=summary,
    )


def compare_day_metrics(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    a_label: str = "a",
    b_label: str = "b",
) -> pl.DataFrame:
    """جدول مقياس مقابل مقياس. ليس إشارة."""

    rows: list[dict[str, Any]] = []
    for key in COMPARE_KEYS:
        left = a.get(key)
        right = b.get(key)
        delta: float | None
        if isinstance(left, bool) or isinstance(right, bool):
            delta = None
        elif isinstance(left, int | float) and isinstance(right, int | float):
            if math.isnan(float(left)) or math.isnan(float(right)):
                delta = None
            else:
                delta = float(right) - float(left)
        else:
            delta = None
        rows.append(
            {
                "metric": key,
                a_label: left,
                b_label: right,
                "delta_b_minus_a": delta,
            }
        )
    return pl.DataFrame(rows)


def write_day_scan_report(scan: DayScan, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(
        json.dumps(scan.coverage, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(
        json.dumps(scan.summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_cvd_opposite_report(scan.opposite, scan.opposite_diag, out / "opposite")
    write_cvd_align_expansion_report(scan.align, scan.align_diag, out / "align")
    write_tape_bins_report({"day60": scan.tape}, {"day60": scan.tape_diag}, out / "burst60")
    lines = [
        f"# CVD day scan: {scan.label}",
        "",
        f"coverage={scan.coverage.get('coverage_class')} "
        f"t_hours={scan.coverage.get('t_hours')} has_rth={scan.coverage.get('has_rth')}.",
        f"T span MNQ {scan.coverage.get('mnq_tmin')} → {scan.coverage.get('mnq_tmax')}.",
        "Same a priori defs as the other day. Not a pattern lock. Clocks not copied.",
        "",
        f"5m bins={scan.summary.get('n_bins_5m')} "
        f"Δ opposite={scan.summary.get('n_delta_opposite')} "
        f"strong={scan.summary.get('n_strong')} aligned={scan.summary.get('n_aligned')} "
        f"wide={scan.summary.get('n_wide')}.",
        f"MNQ→NQ={scan.summary.get('n_mnq_joins_nq')} "
        f"RTH MNQ→NQ={scan.summary.get('n_rth_mnq_joins_nq')} "
        f"RTH wide={scan.summary.get('n_rth_mnq_joins_nq_wide')}.",
        f"60s bins={scan.summary.get('n_bins_60s')} "
        f"|Δ|>=1000={scan.summary.get('n_hyp_cvd')} "
        f"all_three={scan.summary.get('n_hyp_all_three')} "
        f"median next5m all_three={scan.summary.get('median_next5m_all_three')}.",
        "",
        "Burst cuts are OOS of 2026-08-17 11:51, not a lock.",
        "",
        *_rth_join_lines(scan.align),
    ]
    (out / "CVD_DAY.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def write_cvd_day_compare_report(
    table: pl.DataFrame,
    *,
    a_summary: Mapping[str, Any],
    b_summary: Mapping[str, Any],
    output_dir: Path | str,
    a_label: str = "a",
    b_label: str = "b",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "cvd_day_compare.parquet")
    payload = {
        "layer": LAYER_COMPARE,
        "a_label": a_label,
        "b_label": b_label,
        "a": dict(a_summary),
        "b": dict(b_summary),
        "not_pattern": True,
        "not_copy_clocks": True,
        "not_matched_rth": bool(a_summary.get("not_matched_rth"))
        or bool(b_summary.get("not_matched_rth")),
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        f"# CVD day compare: {a_label} vs {b_label}",
        "",
        f"{a_label} coverage={a_summary.get('coverage_class')} "
        f"t_hours={a_summary.get('t_hours')} has_rth={a_summary.get('has_rth')}.",
        f"{b_label} coverage={b_summary.get('coverage_class')} "
        f"t_hours={b_summary.get('t_hours')} has_rth={b_summary.get('has_rth')}.",
        (
            "Both include RTH. Same a priori defs. Not a lock."
            if bool(a_summary.get("has_rth"))
            and bool(b_summary.get("has_rth"))
            and a_summary.get("coverage_class") == "includes_rth"
            and b_summary.get("coverage_class") == "includes_rth"
            else "Same a priori defs. A short Sunday open is not an RTH match. Not a lock."
        ),
        "",
        f"| metric | {a_label} | {b_label} | Δ |",
        "|---|---:|---:|---:|",
    ]
    for row in table.iter_rows(named=True):
        left = row[a_label]
        right = row[b_label]
        delta = row["delta_b_minus_a"]
        delta_s = (
            ""
            if delta is None
            else ("nan" if isinstance(delta, float) and math.isnan(delta) else f"{delta:.4g}")
        )
        lines.append(f"| {row['metric']} | {left} | {right} | {delta_s} |")
    lines.append("")
    lines.append(f"layer={LAYER_ID}. not_pattern=true. clocks not copied.")
    lines.append("")
    (out / "CVD_DAY_COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "COMPARE_KEYS",
    "LAYER_COMPARE",
    "RTH_END",
    "RTH_START",
    "DayScan",
    "compare_day_metrics",
    "describe_tape_coverage",
    "in_rth",
    "scan_cvd_day",
    "summarize_align_day",
    "summarize_tape_hyp",
    "write_cvd_day_compare_report",
    "write_day_scan_report",
]
