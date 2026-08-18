"""عمر أوامر MNQ MBO حول حزم T عنيفة. وصف لا سبب، وليس سبوفينج.

مرور واحد على MBO. الإغلاق في ``[start, end)``. اللقطة الحيّة عند ``burst_ts``.
``fleeting`` حدس عمر < 2ث بلا تنفيذ، ليس حكمًا قانونيًا.
احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import EVENT_TS, SEQUENCE
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.mbo_trade_overlap import prepare_mbo_events
from nq.research.order_lifecycle import FLEETING_NS, ULTRAFAST_NS

LAYER_ID = "burst_lifetime"
SECOND_NS: Final = 1_000_000_000
_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_MODIFY = MboAction.MODIFY.value
_CLEAR = MboAction.CLEAR.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value


@dataclass(frozen=True, slots=True)
class BurstWindow:
    name: str
    start_ts: int
    end_ts: int
    burst_ts: int


@dataclass(slots=True)
class _Live:
    oid: int
    add_ts: int
    size: int
    executed: int
    side: str


class _Walk:
    def __init__(self, windows: Sequence[BurstWindow]) -> None:
        self.win_of = [(int(w.start_ts), int(w.end_ts), w.name) for w in windows]
        self.bursts = sorted((int(w.burst_ts), w.name) for w in windows)
        self.burst_i = 0
        self.live: dict[int, _Live] = {}
        self.closed: dict[str, list[tuple[int, int, bool, str]]] = {w.name: [] for w in windows}
        self.counts: dict[str, dict[str, int]] = {
            w.name: {
                "n_add": 0,
                "n_cancel": 0,
                "n_trade": 0,
                "add_size": 0,
                "cancel_size": 0,
                "trade_size": 0,
            }
            for w in windows
        }
        self.live_shot: dict[str, list[int]] = {w.name: [] for w in windows}

    def names_at(self, ts: int) -> list[str]:
        return [name for lo, hi, name in self.win_of if lo <= ts < hi]

    def snapshot(self, now: int) -> None:
        while self.burst_i < len(self.bursts) and now >= self.bursts[self.burst_i][0]:
            burst_ts, name = self.bursts[self.burst_i]
            self.live_shot[name] = [burst_ts - o.add_ts for o in self.live.values()]
            self.burst_i += 1

    def bump(self, names: list[str], key: str, size: int) -> None:
        for name in names:
            self.counts[name][key] += 1
            self.counts[name][key.replace("n_", "") + "_size"] += max(size, 0)

    def close(self, order: _Live, ts: int, names: list[str]) -> None:
        life = ts - order.add_ts
        for name in names:
            born_here = False
            for lo, hi, wname in self.win_of:
                if wname == name:
                    born_here = lo <= order.add_ts < hi
                    break
            self.closed[name].append((life, order.executed, born_here, order.side))
        self.live.pop(order.oid, None)

    def on_event(self, action: str, ts: int, size: int, side: str, oid: int) -> None:
        self.snapshot(ts)
        names = self.names_at(ts)
        if action == _CLEAR:
            self.live.clear()
        elif action == _ADD:
            self._on_add(names, ts, size, side, oid)
        elif action in {_TRADE, _FILL}:
            self._on_trade(names, ts, size, oid)
        elif action == _MODIFY:
            order = self.live.get(oid)
            if order is not None:
                order.size = max(size, 0)
        elif action == _CANCEL:
            self._on_cancel(names, ts, size, oid)

    def _on_add(self, names: list[str], ts: int, size: int, side: str, oid: int) -> None:
        self.bump(names, "n_add", size)
        if oid <= 0:
            return
        old = self.live.pop(oid, None)
        if old is not None:
            self.close(old, ts, names)
        self.live[oid] = _Live(oid, ts, max(size, 0), 0, side)

    def _on_trade(self, names: list[str], ts: int, size: int, oid: int) -> None:
        self.bump(names, "n_trade", size)
        order = self.live.get(oid)
        if order is None:
            return
        take = max(size, 0)
        order.executed += take
        order.size = max(order.size - take, 0)
        if order.size == 0:
            self.close(order, ts, names)

    def _on_cancel(self, names: list[str], ts: int, size: int, oid: int) -> None:
        self.bump(names, "n_cancel", size)
        order = self.live.get(oid)
        if order is None:
            return
        qty = size if size > 0 else order.size
        remaining = max(order.size - min(qty, order.size), 0)
        if remaining > 0:
            order.size = remaining
            return
        self.close(order, ts, names)


def _pct(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _share(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _summarize(
    name: str,
    records: list[tuple[int, int, bool, str]],
    live_ages_ns: list[int],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    arr = np.asarray([r[0] for r in records], dtype=np.float64)
    exe = np.asarray([r[1] for r in records], dtype=np.int64)
    born_a = np.asarray([r[2] for r in records], dtype=bool)
    n = int(arr.size)
    fleeting = int(((arr < FLEETING_NS) & (exe == 0)).sum()) if n else 0
    ultra = int((arr < ULTRAFAST_NS).sum()) if n else 0
    genuine = int((exe > 0).sum()) if n else 0
    rest_life = arr[~born_a] if n else arr
    live = np.asarray(live_ages_ns, dtype=np.float64)
    n_bid = int(sum(1 for rec in records if rec[3] == "B"))
    return {
        "name": name,
        "n_closed": n,
        "n_born_in_window": int(born_a.sum()) if n else 0,
        "n_resting_death": int((~born_a).sum()) if n else 0,
        "n_genuine": genuine,
        "n_fleeting": fleeting,
        "n_ultrafast": ultra,
        "n_closed_bid": n_bid,
        "median_life_ms": _pct(arr, 0.5) / 1_000_000.0,
        "p25_life_ms": _pct(arr, 0.25) / 1_000_000.0,
        "p75_life_ms": _pct(arr, 0.75) / 1_000_000.0,
        "median_resting_life_ms": _pct(rest_life, 0.5) / 1_000_000.0,
        "fleeting_share": _share(fleeting, n),
        "genuine_share": _share(genuine, n),
        "n_live_at_burst": len(live_ages_ns),
        "median_live_age_ms": _pct(live, 0.5) / 1_000_000.0,
        "n_add": counts["n_add"],
        "n_cancel": counts["n_cancel"],
        "n_trade": counts["n_trade"],
        "add_size": counts["add_size"],
        "cancel_size": counts["cancel_size"],
        "trade_size": counts["trade_size"],
        "cancel_to_add": _share(counts["cancel_size"], counts["add_size"]),
        "cancel_to_trade": _share(counts["cancel_size"], counts["trade_size"]),
        "not_spoofing": True,
    }


def score_burst_lifetimes(
    mbo: pl.DataFrame,
    windows: Sequence[BurstWindow],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """مرور واحد: إغلاقات داخل كل نافذة + عمر الدفتر الحي عند لحظة الحزمة."""

    if not windows:
        raise ValueError("windows must not be empty")
    book = prepare_mbo_events(mbo)
    assert_single_day_mbo(book)
    last_end = max(int(w.end_ts) for w in windows)
    work = book.filter(pl.col(EVENT_TS) < last_end).sort([EVENT_TS, SEQUENCE])
    walk = _Walk(windows)
    if work.height:
        for action, ts_raw, size_raw, side_raw, oid_raw in zip(
            work["action"].to_list(),
            work[EVENT_TS].to_list(),
            work["size"].to_list(),
            work["side"].to_list(),
            work["order_id"].to_list(),
            strict=True,
        ):
            walk.on_event(str(action), int(ts_raw), int(size_raw), str(side_raw), int(oid_raw))
    walk.snapshot(last_end)
    table = pl.DataFrame(
        [
            _summarize(w.name, walk.closed[w.name], walk.live_shot[w.name], walk.counts[w.name])
            for w in windows
        ]
    )
    diagnostics = {
        "layer": LAYER_ID,
        "n_windows": len(windows),
        "n_events": work.height,
        "not_pattern": True,
        "not_spoofing": True,
        "not_cause_lock": True,
        "fleeting_is_legal_spoofing": False,
        "not_lstm": True,
        "not_live_overlay": True,
        "note": "lifetime of orders that fully cancel or fully fill in the window",
    }
    return table, diagnostics


def write_burst_lifetime_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "burst_lifetime.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    def _ms(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bool) or not isinstance(value, int | float):
            return str(value)
        number = float(value)
        if math.isnan(number):
            return "nan"
        return f"{number:.1f}"

    lines = [
        "# MNQ MBO order lifetime around violent T packets",
        "",
        "Descriptive. Fleeting = full cancel, no fill, lifetime < 2s (heuristic, not spoofing).",
        "Resting death = add before the window, full cancel/fill inside it.",
        "Live age = orders still on the book at burst_ts. Not a cause lock.",
        "",
        "| name | closed | born | rest_die | genuine | fleeting | med_ms | rest_med_ms | "
        "live_n | live_med_ms | C/A | C/T |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if table.height:
        for row in table.iter_rows(named=True):
            lines.append(
                f"| {row['name']} | {row['n_closed']} | {row['n_born_in_window']} | "
                f"{row['n_resting_death']} | {row['n_genuine']} | {row['n_fleeting']} | "
                f"{_ms(row['median_life_ms'])} | {_ms(row['median_resting_life_ms'])} | "
                f"{row['n_live_at_burst']} | {_ms(row['median_live_age_ms'])} | "
                f"{_ms(row['cancel_to_add'])} | {_ms(row['cancel_to_trade'])} |"
            )
    (out / "BURST_LIFETIME.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


__all__ = [
    "LAYER_ID",
    "BurstWindow",
    "score_burst_lifetimes",
    "write_burst_lifetime_report",
]
