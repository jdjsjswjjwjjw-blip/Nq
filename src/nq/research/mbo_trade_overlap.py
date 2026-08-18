"""مقارنة شريط TRADE بملف MBO لنفس جلسة اليوم: تنفيذ فوري مقابل أمر معلّق.

ملف Trades (كلّه ``T``، بلا ``order_id``) هو شريط التنفيذ العلني. أحداث ``T``
في MBO هي نفس الطبعات عادةً؛ ``F`` هي ملء الأوامر المعلّقة التي اصطدم بها
العدواني. يوم جلسة واحد، بلا لصق MBO عبر الأيام، بلا دفتر، ليست overlay حيّة.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import EVENT_TS, INGEST_TS, SEQUENCE
from nq.research.mbo_sequence_mlp import assert_single_day_mbo

LAYER_ID = "mbo_trade_overlap"
_ADD = MboAction.ADD.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_CANCEL = MboAction.CANCEL.value
_MODIFY = MboAction.MODIFY.value
_NEEDED: Final[tuple[str, ...]] = (
    EVENT_TS,
    SEQUENCE,
    "action",
    "side",
    "price",
    "size",
)


@dataclass(frozen=True, slots=True)
class MboTradeOverlap:
    """نسب تنفيذ يوم واحد: أوامر MBO مقابل شريط TRADE."""

    n_mbo: int
    n_add: int
    n_cancel: int
    n_modify: int
    n_mbo_t: int
    n_mbo_f: int
    n_trades: int
    add_size: int
    cancel_size: int
    mbo_t_size: int
    mbo_f_size: int
    trades_size: int
    n_oid_added: int
    n_oid_filled: int
    n_oid_added_and_filled: int
    n_oid_traded: int
    n_oid_traded_never_added: int
    n_mbo_t_rested: int
    n_mbo_t_immediate: int
    n_mbo_t_oid_zero: int
    n_mbo_f_rested: int
    n_mbo_f_no_prior_add: int
    n_seq_overlap: int
    n_trades_seq_in_mbo_t: int
    n_mbo_t_seq_in_trades: int
    n_trades_unmatched_seq: int

    @property
    def pct_mbo_events_that_are_prints(self) -> float:
        return _ratio(self.n_mbo_t, self.n_mbo)

    @property
    def pct_adds_that_fill(self) -> float:
        return _ratio(self.n_oid_added_and_filled, self.n_oid_added)

    @property
    def qty_fill_over_add(self) -> float:
        return _ratio(self.mbo_f_size, self.add_size)

    @property
    def pct_mbo_t_immediate(self) -> float:
        return _ratio(self.n_mbo_t_immediate, self.n_mbo_t)

    @property
    def pct_mbo_t_rested(self) -> float:
        return _ratio(self.n_mbo_t_rested, self.n_mbo_t)

    @property
    def pct_mbo_f_rested(self) -> float:
        return _ratio(self.n_mbo_f_rested, self.n_mbo_f)

    @property
    def pct_trades_matched_to_mbo_t(self) -> float:
        return _ratio(self.n_trades_seq_in_mbo_t, self.n_trades)

    @property
    def pct_mbo_t_matched_to_trades(self) -> float:
        return _ratio(self.n_mbo_t_seq_in_trades, self.n_mbo_t)


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _to_epoch_ns(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    if name not in frame.columns:
        return frame
    dtype = frame.schema[name]
    if dtype == pl.Int64:
        return frame
    return frame.with_columns(pl.col(name).cast(pl.Int64).alias(name))


def _rename_databento(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame
    if "ts_event" in work.columns and EVENT_TS not in work.columns:
        work = work.rename({"ts_event": EVENT_TS})
    if "ts_recv" in work.columns and INGEST_TS not in work.columns:
        work = work.rename({"ts_recv": INGEST_TS})
    return work


def prepare_mbo_events(frame: pl.DataFrame) -> pl.DataFrame:
    """أعمدة المقارنة من MBO خام أو عقد قانوني. يرفض لصق أيام على ``ingest_ts``."""

    work = _rename_databento(frame)
    if "order_id" not in work.columns:
        raise ValueError("MBO frame missing order_id")
    missing = [name for name in _NEEDED if name not in work.columns]
    if missing:
        raise ValueError(f"MBO frame missing columns {missing}")
    work = _to_epoch_ns(work, EVENT_TS)
    if INGEST_TS in work.columns:
        work = _to_epoch_ns(work, INGEST_TS)
    else:
        work = work.with_columns(pl.col(EVENT_TS).alias(INGEST_TS))
    work = work.with_columns(
        pl.col("action").cast(pl.Utf8).str.to_uppercase().alias("action"),
        pl.col("side").cast(pl.Utf8).alias("side"),
        pl.col("order_id").cast(pl.Int64).alias("order_id"),
        pl.col("size").cast(pl.Int64).alias("size"),
        pl.col(SEQUENCE).cast(pl.Int64).alias(SEQUENCE),
        pl.col(EVENT_TS).cast(pl.Int64).alias(EVENT_TS),
        pl.col(INGEST_TS).cast(pl.Int64).alias(INGEST_TS),
    )
    assert_single_day_mbo(work)
    return work.select(
        [
            EVENT_TS,
            INGEST_TS,
            SEQUENCE,
            "action",
            "side",
            "price",
            "size",
            "order_id",
        ]
    )


def prepare_trades_tape(frame: pl.DataFrame) -> pl.DataFrame:
    """شريط TRADE: ``T`` فقط. ``order_id`` اختياري (غائب في مخطط Databento trades)."""

    work = _rename_databento(frame)
    missing = [name for name in _NEEDED if name not in work.columns]
    if missing:
        raise ValueError(f"trades frame missing columns {missing}")
    work = _to_epoch_ns(work, EVENT_TS)
    if INGEST_TS in work.columns:
        work = _to_epoch_ns(work, INGEST_TS)
    else:
        work = work.with_columns(pl.col(EVENT_TS).alias(INGEST_TS))
    if "order_id" not in work.columns:
        work = work.with_columns(pl.lit(0, dtype=pl.Int64).alias("order_id"))
    work = work.with_columns(
        pl.col("action").cast(pl.Utf8).str.to_uppercase().alias("action"),
        pl.col("side").cast(pl.Utf8).alias("side"),
        pl.col("order_id").cast(pl.Int64).alias("order_id"),
        pl.col("size").cast(pl.Int64).alias("size"),
        pl.col(SEQUENCE).cast(pl.Int64).alias(SEQUENCE),
        pl.col(EVENT_TS).cast(pl.Int64).alias(EVENT_TS),
        pl.col(INGEST_TS).cast(pl.Int64).alias(INGEST_TS),
    )
    assert_single_day_mbo(work)
    return work.select(
        [
            EVENT_TS,
            INGEST_TS,
            SEQUENCE,
            "action",
            "side",
            "price",
            "size",
            "order_id",
        ]
    )


def _first_add_by_oid(mbo: pl.DataFrame) -> pl.DataFrame:
    return (
        mbo.filter((pl.col("action") == _ADD) & (pl.col("order_id") > 0))
        .group_by("order_id")
        .agg(pl.col(SEQUENCE).min().alias("add_seq"))
    )


def _classify_prior_add(events: pl.DataFrame, first_add: pl.DataFrame) -> pl.DataFrame:
    return events.join(first_add, on="order_id", how="left").with_columns(
        (
            (pl.col("order_id") > 0)
            & pl.col("add_seq").is_not_null()
            & (pl.col("add_seq") < pl.col(SEQUENCE))
        ).alias("rested")
    )


def compare_mbo_trades(mbo: pl.DataFrame, trades: pl.DataFrame) -> MboTradeOverlap:
    """نسب التنفيذ والمرور على الدفتر ليوم جلسة واحد."""

    book = prepare_mbo_events(mbo)
    tape = prepare_trades_tape(trades)
    first_add = _first_add_by_oid(book)
    adds = book.filter(pl.col("action") == _ADD)
    cancels = book.filter(pl.col("action") == _CANCEL)
    modifies = book.filter(pl.col("action") == _MODIFY)
    prints = _classify_prior_add(book.filter(pl.col("action") == _TRADE), first_add)
    fills = _classify_prior_add(book.filter(pl.col("action") == _FILL), first_add)

    oid_added = adds.filter(pl.col("order_id") > 0).select("order_id").unique()
    oid_filled = fills.filter(pl.col("order_id") > 0).select("order_id").unique()
    oid_traded = prints.filter(pl.col("order_id") > 0).select("order_id").unique()
    n_oid_added = oid_added.height
    n_oid_filled = oid_filled.height
    n_oid_added_and_filled = oid_added.join(oid_filled, on="order_id", how="inner").height
    n_oid_traded = oid_traded.height
    n_oid_traded_never_added = oid_traded.join(oid_added, on="order_id", how="anti").height

    n_mbo_t_oid_zero = prints.filter(pl.col("order_id") <= 0).height
    n_mbo_t_rested = prints.filter(pl.col("rested")).height
    n_mbo_t_immediate = prints.filter(~pl.col("rested")).height
    n_mbo_f_rested = fills.filter(pl.col("rested")).height
    n_mbo_f_no_prior_add = fills.filter(~pl.col("rested")).height

    mbo_t_seq = prints.select(SEQUENCE).unique()
    tape_seq = tape.select(SEQUENCE).unique()
    n_seq_overlap = mbo_t_seq.join(tape_seq, on=SEQUENCE, how="inner").height
    n_trades_seq_in_mbo_t = tape.select(SEQUENCE).join(mbo_t_seq, on=SEQUENCE, how="inner").height
    n_mbo_t_seq_in_trades = prints.select(SEQUENCE).join(tape_seq, on=SEQUENCE, how="inner").height
    n_trades_unmatched_seq = tape.height - n_trades_seq_in_mbo_t

    return MboTradeOverlap(
        n_mbo=book.height,
        n_add=adds.height,
        n_cancel=cancels.height,
        n_modify=modifies.height,
        n_mbo_t=prints.height,
        n_mbo_f=fills.height,
        n_trades=tape.height,
        add_size=int(adds.select(pl.col("size").sum()).item() or 0),
        cancel_size=int(cancels.select(pl.col("size").sum()).item() or 0),
        mbo_t_size=int(prints.select(pl.col("size").sum()).item() or 0),
        mbo_f_size=int(fills.select(pl.col("size").sum()).item() or 0),
        trades_size=int(tape.select(pl.col("size").sum()).item() or 0),
        n_oid_added=n_oid_added,
        n_oid_filled=n_oid_filled,
        n_oid_added_and_filled=n_oid_added_and_filled,
        n_oid_traded=n_oid_traded,
        n_oid_traded_never_added=n_oid_traded_never_added,
        n_mbo_t_rested=n_mbo_t_rested,
        n_mbo_t_immediate=n_mbo_t_immediate,
        n_mbo_t_oid_zero=n_mbo_t_oid_zero,
        n_mbo_f_rested=n_mbo_f_rested,
        n_mbo_f_no_prior_add=n_mbo_f_no_prior_add,
        n_seq_overlap=n_seq_overlap,
        n_trades_seq_in_mbo_t=n_trades_seq_in_mbo_t,
        n_mbo_t_seq_in_trades=n_mbo_t_seq_in_trades,
        n_trades_unmatched_seq=n_trades_unmatched_seq,
    )


def overlap_table(result: MboTradeOverlap) -> pl.DataFrame:
    """جدول النسب للعرض. ``pct`` نسبة من المقام المذكور في ``of``."""

    rows: list[dict[str, Any]] = [
        {
            "metric": "mbo_rows",
            "count": result.n_mbo,
            "pct": 1.0,
            "of": "mbo_rows",
        },
        {
            "metric": "mbo_add",
            "count": result.n_add,
            "pct": _ratio(result.n_add, result.n_mbo),
            "of": "mbo_rows",
        },
        {
            "metric": "mbo_cancel",
            "count": result.n_cancel,
            "pct": _ratio(result.n_cancel, result.n_mbo),
            "of": "mbo_rows",
        },
        {
            "metric": "mbo_modify",
            "count": result.n_modify,
            "pct": _ratio(result.n_modify, result.n_mbo),
            "of": "mbo_rows",
        },
        {
            "metric": "mbo_T_prints",
            "count": result.n_mbo_t,
            "pct": result.pct_mbo_events_that_are_prints,
            "of": "mbo_rows",
        },
        {
            "metric": "mbo_F_resting_fills",
            "count": result.n_mbo_f,
            "pct": _ratio(result.n_mbo_f, result.n_mbo),
            "of": "mbo_rows",
        },
        {
            "metric": "trades_file_rows",
            "count": result.n_trades,
            "pct": result.pct_trades_matched_to_mbo_t,
            "of": "matched_to_mbo_T_by_sequence",
        },
        {
            "metric": "unique_oids_added",
            "count": result.n_oid_added,
            "pct": 1.0,
            "of": "unique_oids_added",
        },
        {
            "metric": "unique_oids_added_then_filled",
            "count": result.n_oid_added_and_filled,
            "pct": result.pct_adds_that_fill,
            "of": "unique_oids_added",
        },
        {
            "metric": "unique_oids_filled",
            "count": result.n_oid_filled,
            "pct": _ratio(result.n_oid_filled, result.n_oid_added),
            "of": "unique_oids_added",
        },
        {
            "metric": "mbo_T_rested_then_printed",
            "count": result.n_mbo_t_rested,
            "pct": result.pct_mbo_t_rested,
            "of": "mbo_T",
        },
        {
            "metric": "mbo_T_immediate_no_prior_add",
            "count": result.n_mbo_t_immediate,
            "pct": result.pct_mbo_t_immediate,
            "of": "mbo_T",
        },
        {
            "metric": "mbo_T_order_id_zero",
            "count": result.n_mbo_t_oid_zero,
            "pct": _ratio(result.n_mbo_t_oid_zero, result.n_mbo_t),
            "of": "mbo_T",
        },
        {
            "metric": "mbo_F_had_prior_add",
            "count": result.n_mbo_f_rested,
            "pct": result.pct_mbo_f_rested,
            "of": "mbo_F",
        },
        {
            "metric": "mbo_F_no_prior_add",
            "count": result.n_mbo_f_no_prior_add,
            "pct": _ratio(result.n_mbo_f_no_prior_add, result.n_mbo_f),
            "of": "mbo_F",
        },
        {
            "metric": "T_oids_never_added",
            "count": result.n_oid_traded_never_added,
            "pct": _ratio(result.n_oid_traded_never_added, result.n_oid_traded),
            "of": "unique_T_oids",
        },
        {
            "metric": "qty_fill_over_add",
            "count": result.mbo_f_size,
            "pct": result.qty_fill_over_add,
            "of": "add_size",
        },
        {
            "metric": "qty_print_T_over_fill_F",
            "count": result.mbo_t_size,
            "pct": _ratio(result.mbo_t_size, result.mbo_f_size),
            "of": "mbo_F_size",
        },
        {
            "metric": "trades_size_over_mbo_T_size",
            "count": result.trades_size,
            "pct": _ratio(result.trades_size, result.mbo_t_size),
            "of": "mbo_T_size",
        },
        {
            "metric": "unique_seq_overlap_T_and_tape",
            "count": result.n_seq_overlap,
            "pct": _ratio(result.n_seq_overlap, result.n_mbo_t),
            "of": "mbo_T",
        },
        {
            "metric": "trades_unmatched_by_sequence",
            "count": result.n_trades_unmatched_seq,
            "pct": _ratio(result.n_trades_unmatched_seq, result.n_trades),
            "of": "trades_file_rows",
        },
    ]
    return pl.DataFrame(rows)


def write_overlap_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out / "mbo_trade_overlap.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# MBO vs TRADE overlap (one session day)",
        "",
        "Trades file = public T prints (no order_id). MBO T ≈ same prints.",
        "MBO F = fills of resting orders those prints hit.",
        "rested = same order_id had an earlier Add (sequence strictly before).",
        "immediate = T/F with no prior Add (aggressor / hidden / missing A).",
        "Not a live overlay. Not a spoofing judgment.",
        "",
        "| metric | count | pct | of |",
        "|---|---:|---:|---|",
    ]
    for row in table.iter_rows(named=True):
        pct = row["pct"]
        pct_s = "nan" if pct is None else f"{float(pct):.6f}"
        lines.append(f"| {row['metric']} | {row['count']} | {pct_s} | {row['of']} |")
    lines.append("")
    (out / "MBO_TRADE_OVERLAP.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def overlap_diagnostics(result: MboTradeOverlap, **extra: Any) -> dict[str, Any]:
    payload = asdict(result)
    payload.update(
        {
            "layer": LAYER_ID,
            "pct_adds_that_fill": result.pct_adds_that_fill,
            "pct_mbo_t_immediate": result.pct_mbo_t_immediate,
            "pct_mbo_t_rested": result.pct_mbo_t_rested,
            "pct_mbo_f_rested": result.pct_mbo_f_rested,
            "pct_trades_matched_to_mbo_t": result.pct_trades_matched_to_mbo_t,
            "qty_fill_over_add": result.qty_fill_over_add,
        }
    )
    payload.update(extra)
    return payload


__all__ = [
    "LAYER_ID",
    "MboTradeOverlap",
    "compare_mbo_trades",
    "overlap_diagnostics",
    "overlap_table",
    "prepare_mbo_events",
    "prepare_trades_tape",
    "write_overlap_report",
]
