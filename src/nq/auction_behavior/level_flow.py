"""تدفق أوامر مُثبَّت على مستويات القرار (VAH/VAL/POC/HVN/كسر/ريتست).

لكل برميل إشارة: أحداث MBO داخل البرميل قرب المستوى.
Raw MBO يبقى؛ هذه ميزات evidence تُلحق جنبًا — بلا حذف أوامر.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nq.contracts.mbo import PRICE_SCALE, MboAction
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.time import sort_causal
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket

LEVEL_FLOW_COLUMNS = (
    "lf_near_vah_add_intensity",
    "lf_near_vah_cancel_intensity",
    "lf_near_vah_trade_intensity",
    "lf_near_vah_cancel_ratio",
    "lf_near_vah_exec_cancel_ratio",
    "lf_near_val_add_intensity",
    "lf_near_val_cancel_intensity",
    "lf_near_val_trade_intensity",
    "lf_near_val_cancel_ratio",
    "lf_near_val_exec_cancel_ratio",
    "lf_near_poc_add_intensity",
    "lf_near_poc_cancel_intensity",
    "lf_near_poc_trade_intensity",
    "lf_near_poc_cancel_ratio",
    "lf_near_hvn_add_intensity",
    "lf_near_hvn_cancel_intensity",
    "lf_near_hvn_trade_intensity",
    "lf_near_hvn_cancel_ratio",
    "lf_break_level_trade_intensity",
    "lf_break_level_cancel_intensity",
    "lf_retest_level_cancel_ratio",
    "lf_retest_level_trade_intensity",
    "lf_mean_order_lifetime_ns",
    "lf_queue_survival_rate",
    "lf_partial_exec_rate",
    "lf_refill_rate",
    "lf_absorption_proxy",
    "lf_liquidity_withdrawal",
    "lf_liquidity_migration",
    "lf_arrival_intensity",
    "lf_cancel_intensity",
    "lf_abnormal_cancel_burst",
)

_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_MODIFY = MboAction.MODIFY.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_CLEAR = MboAction.CLEAR.value
_NS = 1_000_000_000.0


@dataclass(frozen=True, slots=True)
class LevelFlowConfig:
    """قرب المستوى بوحدة تيكات السعر الخام."""

    near_ticks: float = 4.0
    interval_ns: int = 30 * 1_000_000_000


def _empty(keys: pl.DataFrame) -> pl.DataFrame:
    base = (
        keys.select(AVAILABILITY_TS)
        if AVAILABILITY_TS in keys.columns
        else pl.DataFrame(schema={AVAILABILITY_TS: pl.Int64()})
    )
    return base.with_columns(pl.lit(0.0).alias(c) for c in LEVEL_FLOW_COLUMNS)


def _order_lifecycle_by_bucket(events: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR0912, PLR0915
    """آلة حالات سببية عبر البراميل متوافقة مع دلالات MBO في ``OrderBook``.

    ``CANCEL.size`` كمية ملغاة وقد يكون جزئيًا، و``TRADE/FILL`` دليل تنفيذ عند
    السعر لكنه لا يغيّر حالة الدفتر بنفسه. المخرجات تُنسب إلى برميل الإغلاق فقط.
    الأوامر التي يمسحها ``CLEAR`` أو تنتهي مع العينة تظل censored ولا تتحول إلى
    إلغاء اصطناعي.
    """
    schema = {
        BUCKET_START: pl.Int64(),
        "lf_mean_order_lifetime_ns": pl.Float64(),
        "lf_queue_survival_rate": pl.Float64(),
        "lf_partial_exec_rate": pl.Float64(),
        "lf_refill_rate": pl.Float64(),
    }
    if events.height == 0 or "order_id" not in events.columns:
        return pl.DataFrame(schema=schema)

    work = events.sort([EVENT_TS, "order_id"])
    # حالة حية لكل order_id؛ price index يربط trade (غالبًا order_id=0)
    # بالأوامر الظاهرة عند المستوى من دون ادعاء تخصيص تنفيذ لأمر بعينه.
    open_ts: dict[int, int] = {}
    current_size: dict[int, float] = {}
    order_price: dict[int, float] = {}
    orders_at_price: dict[float, set[int]] = {}
    had_execution: dict[int, bool] = {}
    had_refill: dict[int, bool] = {}
    survived_change: dict[int, bool] = {}
    rows: list[dict[str, float | int]] = []

    actions = work["action"].to_list()
    oids = work["order_id"].to_list()
    times = work[EVENT_TS].to_list()
    sizes = work["size"].to_list()
    buckets = work[BUCKET_START].to_list()
    prices = work["price"].to_list() if "price" in work.columns else [None] * work.height
    near_flags = (
        work["_near_level"].fill_null(False).to_list()
        if "_near_level" in work.columns
        else [True] * work.height
    )

    def _drop(oid: int) -> None:
        price = order_price.pop(oid, None)
        if price is not None:
            at_price = orders_at_price.get(price)
            if at_price is not None:
                at_price.discard(oid)
                if not at_price:
                    orders_at_price.pop(price, None)
        open_ts.pop(oid, None)
        current_size.pop(oid, None)
        had_execution.pop(oid, None)
        had_refill.pop(oid, None)
        survived_change.pop(oid, None)

    for action, oid_raw, ts_raw, size_raw, buck_raw, price_raw, near_raw in zip(
        actions, oids, times, sizes, buckets, prices, near_flags, strict=True
    ):
        ts = int(ts_raw)
        size = float(size_raw) if size_raw is not None else 0.0
        buck = int(buck_raw)
        price = float(price_raw) if price_raw is not None else None
        act = str(action)

        if act == _CLEAR:
            # reset إداري: كل الأعمار المفتوحة censored وليست cancels مرصودة.
            for live_oid in tuple(open_ts):
                _drop(live_oid)
            continue

        if act in (_TRADE, _FILL):
            touched: set[int] = set()
            if price is not None:
                touched.update(orders_at_price.get(price, set()))
            if oid_raw is not None and int(oid_raw) in open_ts:
                touched.add(int(oid_raw))
            for live_oid in touched:
                had_execution[live_oid] = True
            # وفق عقد الدفتر، التغيير الكمي يأتي في CANCEL مستقل.
            continue

        if oid_raw is None:
            continue
        oid = int(oid_raw)

        if act == _ADD:
            if not bool(near_raw):
                continue
            if oid in open_ts:
                # duplicate ADD يستبدل الحالة القديمة؛ القديمة censored.
                _drop(oid)
            open_ts[oid] = ts
            current_size[oid] = max(size, 0.0)
            if price is not None:
                order_price[oid] = price
                orders_at_price.setdefault(price, set()).add(oid)
            had_execution[oid] = False
            had_refill[oid] = False
            survived_change[oid] = False
            continue
        if oid not in open_ts:
            # أمر بلا ADD مرئي في النافذة — تجاهل
            continue
        if act == _MODIFY:
            old_size = current_size.get(oid, 0.0)
            if size > old_size:
                had_refill[oid] = True
            survived_change[oid] = True
            current_size[oid] = max(size, 0.0)
            if price is not None and price != order_price.get(oid):
                old_price = order_price.get(oid)
                if old_price is not None:
                    old_set = orders_at_price.get(old_price)
                    if old_set is not None:
                        old_set.discard(oid)
                        if not old_set:
                            orders_at_price.pop(old_price, None)
                order_price[oid] = price
                orders_at_price.setdefault(price, set()).add(oid)
            continue
        if act == _CANCEL:
            remaining_before = current_size.get(oid, 0.0)
            cancel_qty = size if size > 0 else remaining_before
            remaining = max(remaining_before - min(cancel_qty, remaining_before), 0.0)
            if remaining > 0:
                current_size[oid] = remaining
                survived_change[oid] = True
                continue
            life = float(ts - open_ts[oid])
            # الإسناد لزمن الإغلاق (سببي)
            rows.append(
                {
                    BUCKET_START: buck,
                    "_life": life,
                    "_survived": float(survived_change.get(oid, False)),
                    "_partial": float(had_execution.get(oid, False)),
                    "_refill": float(had_refill.get(oid, False)),
                }
            )
            _drop(oid)

    if not rows:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(rows)
        .group_by(BUCKET_START, maintain_order=True)
        .agg(
            pl.col("_life").mean().alias("lf_mean_order_lifetime_ns"),
            pl.col("_survived").mean().alias("lf_queue_survival_rate"),
            pl.col("_partial").mean().alias("lf_partial_exec_rate"),
            pl.col("_refill").mean().alias("lf_refill_rate"),
        )
    )


def attach_level_flow_features(  # noqa: PLR0912, PLR0915
    mbo: pl.DataFrame,
    states: pl.DataFrame,
    *,
    config: LevelFlowConfig | None = None,
) -> pl.DataFrame:
    """يبني ميزات شدة/بقاء/امتصاص قرب حدود القرار لكل برميل.

    يستخدم فقط أحداثًا داخل البرميل ومستويات ``decision_*`` المعروفة عند نهايته.
    """
    cfg = config or LevelFlowConfig()
    if states.height == 0 or AVAILABILITY_TS not in states.columns:
        return _empty(states)
    need = ("decision_vah", "decision_val", "decision_poc")
    if any(c not in states.columns for c in need):
        return _empty(states)
    if mbo.height == 0 or EVENT_TS not in mbo.columns:
        return _empty(states)

    tick = float(round(0.25 / PRICE_SCALE))
    near = float(cfg.near_ticks) * tick
    work = sort_causal(mbo)
    bucketed = add_time_bucket(work, interval_ns=cfg.interval_ns)

    if BUCKET_START not in states.columns:
        st = states.select(
            AVAILABILITY_TS,
            "decision_vah",
            "decision_val",
            "decision_poc",
            *([c for c in ("asia_primary_hvn", "composite_primary_hvn") if c in states.columns]),
        ).sort(AVAILABILITY_TS)
        st = st.with_columns(
            (pl.col(AVAILABILITY_TS) - int(cfg.interval_ns)).alias(BUCKET_START),
            pl.col(AVAILABILITY_TS).alias(BUCKET_END),
        )
    else:
        st = states.select(
            AVAILABILITY_TS,
            BUCKET_START,
            *([BUCKET_END] if BUCKET_END in states.columns else []),
            "decision_vah",
            "decision_val",
            "decision_poc",
            *([c for c in ("asia_primary_hvn", "composite_primary_hvn") if c in states.columns]),
        ).sort(BUCKET_START)
        if BUCKET_END not in st.columns:
            st = st.with_columns((pl.col(BUCKET_START) + int(cfg.interval_ns)).alias(BUCKET_END))

    if "composite_primary_hvn" in st.columns:
        hvn = pl.col("composite_primary_hvn").cast(pl.Float64).fill_null(pl.col("decision_poc"))
    elif "asia_primary_hvn" in st.columns:
        hvn = pl.col("asia_primary_hvn").cast(pl.Float64).fill_null(pl.col("decision_poc"))
    else:
        hvn = pl.col("decision_poc").cast(pl.Float64)
    st = st.with_columns(hvn.alias("_hvn"))

    event_cols = [EVENT_TS, BUCKET_START, "action", "price", "size"]
    if "order_id" in bucketed.columns:
        event_cols.append("order_id")
    events = bucketed.select(
        EVENT_TS,
        BUCKET_START,
        pl.col("action").cast(pl.Utf8).alias("action"),
        pl.col("price").cast(pl.Float64).alias("price"),
        pl.col("size").cast(pl.Float64).alias("size"),
        *(
            [pl.col("order_id").cast(pl.UInt64).alias("order_id")]
            if "order_id" in bucketed.columns
            else []
        ),
    )
    joined = events.join(
        st.select(
            BUCKET_START,
            AVAILABILITY_TS,
            pl.col("decision_vah").cast(pl.Float64).alias("_vah"),
            pl.col("decision_val").cast(pl.Float64).alias("_val"),
            pl.col("decision_poc").cast(pl.Float64).alias("_poc"),
            pl.col("_hvn"),
            pl.col(BUCKET_END).alias("_bend"),
        ),
        on=BUCKET_START,
        how="inner",
    ).filter(pl.col(EVENT_TS) <= pl.col("_bend"))

    if joined.height == 0:
        return _empty(states)

    dur = (pl.col("_bend") - pl.col(BUCKET_START)).cast(pl.Float64).clip(lower_bound=1.0) / _NS

    def _near(level: str) -> pl.Expr:
        return (pl.col("price") - pl.col(level)).abs() <= near

    is_add = pl.col("action") == _ADD
    is_cancel = pl.col("action") == _CANCEL
    is_trade = pl.col("action").is_in([_TRADE, _FILL])
    is_modify = pl.col("action") == _MODIFY
    near_any = _near("_vah") | _near("_val") | _near("_poc") | _near("_hvn")

    aggs: list[pl.Expr] = [
        pl.col(AVAILABILITY_TS).first().alias(AVAILABILITY_TS),
        dur.first().alias("_dur"),
        is_add.sum().alias("_a_all"),
        is_cancel.sum().alias("_c_all"),
        is_trade.sum().alias("_t_all"),
        (is_add & near_any).sum().alias("_a_near"),
        (is_cancel & near_any).sum().alias("_c_near"),
        (is_trade & near_any).sum().alias("_t_near"),
        (is_modify & near_any).sum().alias("_m_near"),
    ]
    for tag, level in (
        ("vah", "_vah"),
        ("val", "_val"),
        ("poc", "_poc"),
        ("hvn", "_hvn"),
    ):
        mask = _near(level)
        aggs.extend(
            [
                (is_add & mask).sum().alias(f"_a_{tag}"),
                (is_cancel & mask).sum().alias(f"_c_{tag}"),
                (is_trade & mask).sum().alias(f"_t_{tag}"),
            ]
        )
    aggs.append(((is_trade & (_near("_vah") | _near("_val"))).sum()).alias("_t_break"))
    aggs.append(((is_cancel & (_near("_vah") | _near("_val"))).sum()).alias("_c_break"))
    aggs.append(((is_cancel & _near("_poc")).sum()).alias("_c_retest"))
    aggs.append(((is_add & _near("_poc")).sum()).alias("_a_retest"))
    aggs.append(((is_trade & _near("_poc")).sum()).alias("_t_retest"))

    grouped = joined.group_by(BUCKET_START, maintain_order=True).agg(aggs)

    # عمر الأمر عبر البراميل: آلة حالات سببية order_id → يُصدَر عند الإغلاق
    life_frame = pl.DataFrame(
        schema={
            BUCKET_START: pl.Int64(),
            "lf_mean_order_lifetime_ns": pl.Float64(),
            "lf_queue_survival_rate": pl.Float64(),
            "lf_partial_exec_rate": pl.Float64(),
            "lf_refill_rate": pl.Float64(),
        }
    )
    if "order_id" in joined.columns:
        # افتح lifecycle فقط إذا كان ADD قرب مستوى القرار وقت ظهوره، ثم واصل
        # تتبعه حتى الإغلاق ولو تحرك decision_* لاحقًا إلى مستوى آخر.
        life_frame = _order_lifecycle_by_bucket(joined.with_columns(near_any.alias("_near_level")))

    def _intensity(count: str) -> pl.Expr:
        return (pl.col(count).cast(pl.Float64) / pl.col("_dur")).fill_null(0.0)

    def _ratio(numer: str, denom_a: str, denom_b: str | None = None) -> pl.Expr:
        if denom_b is None:
            tot = pl.col(numer) + pl.col(denom_a)
            return (
                pl.when(tot > 0)
                .then(pl.col(numer).cast(pl.Float64) / tot.cast(pl.Float64))
                .otherwise(0.0)
            )
        den = pl.col(denom_a) + pl.col(denom_b)
        return (
            pl.when(den > 0)
            .then(pl.col(numer).cast(pl.Float64) / den.cast(pl.Float64))
            .otherwise(0.0)
        )

    def _exec_cancel(trade: str, cancel: str) -> pl.Expr:
        return (
            pl.when(pl.col(cancel) > 0)
            .then(pl.col(trade).cast(pl.Float64) / pl.col(cancel).cast(pl.Float64))
            .otherwise(pl.when(pl.col(trade) > 0).then(1.0).otherwise(0.0))
        )

    out = grouped.with_columns(
        _intensity("_a_vah").alias("lf_near_vah_add_intensity"),
        _intensity("_c_vah").alias("lf_near_vah_cancel_intensity"),
        _intensity("_t_vah").alias("lf_near_vah_trade_intensity"),
        _ratio("_c_vah", "_a_vah").alias("lf_near_vah_cancel_ratio"),
        _exec_cancel("_t_vah", "_c_vah").alias("lf_near_vah_exec_cancel_ratio"),
        _intensity("_a_val").alias("lf_near_val_add_intensity"),
        _intensity("_c_val").alias("lf_near_val_cancel_intensity"),
        _intensity("_t_val").alias("lf_near_val_trade_intensity"),
        _ratio("_c_val", "_a_val").alias("lf_near_val_cancel_ratio"),
        _exec_cancel("_t_val", "_c_val").alias("lf_near_val_exec_cancel_ratio"),
        _intensity("_a_poc").alias("lf_near_poc_add_intensity"),
        _intensity("_c_poc").alias("lf_near_poc_cancel_intensity"),
        _intensity("_t_poc").alias("lf_near_poc_trade_intensity"),
        _ratio("_c_poc", "_a_poc").alias("lf_near_poc_cancel_ratio"),
        _intensity("_a_hvn").alias("lf_near_hvn_add_intensity"),
        _intensity("_c_hvn").alias("lf_near_hvn_cancel_intensity"),
        _intensity("_t_hvn").alias("lf_near_hvn_trade_intensity"),
        _ratio("_c_hvn", "_a_hvn").alias("lf_near_hvn_cancel_ratio"),
        _intensity("_t_break").alias("lf_break_level_trade_intensity"),
        _intensity("_c_break").alias("lf_break_level_cancel_intensity"),
        _ratio("_c_retest", "_a_retest").alias("lf_retest_level_cancel_ratio"),
        _intensity("_t_retest").alias("lf_retest_level_trade_intensity"),
        _intensity("_a_all").alias("lf_arrival_intensity"),
        _intensity("_c_all").alias("lf_cancel_intensity"),
        # امتصاص: تداول عالي مع إلغاء منخفض قرب الحدود
        (
            (pl.col("_t_near").cast(pl.Float64) + 1.0) / (pl.col("_c_near").cast(pl.Float64) + 1.0)
        ).alias("lf_absorption_proxy"),
        # سحب سيولة: إلغاءات قرب المستوى بدون تداول
        (
            pl.when((pl.col("_c_near") + pl.col("_a_near")) > 0)
            .then(
                pl.col("_c_near").cast(pl.Float64)
                / (pl.col("_c_near") + pl.col("_a_near") + pl.col("_t_near")).cast(pl.Float64)
            )
            .otherwise(0.0)
        ).alias("lf_liquidity_withdrawal"),
        # هجرة سيولة تقريبية: تعديلات + فرق إضافة بين POC وHVN
        (
            (pl.col("_m_near").cast(pl.Float64) + (pl.col("_a_poc") - pl.col("_a_hvn")).abs())
            / pl.col("_dur")
        ).alias("lf_liquidity_migration"),
        # انفجار إلغاء غير طبيعي داخل البرميل
        (
            pl.when(pl.col("_a_all") > 0)
            .then(pl.col("_c_all").cast(pl.Float64) / pl.col("_a_all").cast(pl.Float64))
            .otherwise(pl.col("_c_all").cast(pl.Float64))
            .clip(0.0, 10.0)
        ).alias("lf_abnormal_cancel_burst"),
    )

    out = out.join(life_frame, on=BUCKET_START, how="left").with_columns(
        pl.col("lf_mean_order_lifetime_ns").fill_null(0.0),
        pl.col("lf_queue_survival_rate").fill_null(0.0),
        pl.col("lf_partial_exec_rate").fill_null(0.0),
        pl.col("lf_refill_rate").fill_null(0.0),
    )

    out = out.select(AVAILABILITY_TS, *LEVEL_FLOW_COLUMNS)
    return (
        states.select(AVAILABILITY_TS)
        .join(out, on=AVAILABILITY_TS, how="left")
        .with_columns(pl.col(c).fill_null(0.0) for c in LEVEL_FLOW_COLUMNS)
        .unique(subset=[AVAILABILITY_TS], keep="last")
        .sort(AVAILABILITY_TS)
    )


__all__ = [
    "LEVEL_FLOW_COLUMNS",
    "LevelFlowConfig",
    "attach_level_flow_features",
]
