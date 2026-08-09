"""مُحاكي تدفّق الأوامر (Order Flow Simulator).

يقيس ضغط الشراء/البيع العدواني، بادئ الصفقة (trade initiation)، اختلال تدفّق
الأوامر (OFI)، واستهلاك السيولة.

ملخّص التدفّق لكل نافذة:

* ``buy_volume`` / ``sell_volume`` — الحجم العدواني (شراء يرفع العرض / بيع يضرب الطلب).
* ``delta`` = ``buy_volume - sell_volume`` و ``cumulative_delta`` (سببي).
* ``buy_trades`` / ``sell_trades`` — عدد الصفقات البادئة شراءً/بيعًا (initiation).
* ``consumption`` = ``buy_volume + sell_volume`` — إجمالي السيولة المُستهلَكة عدوانيًا.

**أداة تسارع الأوامر → اختلال مبكر** (مستقلة عن ``vp_fsm_accel``):

* ``order_accel_rate`` — معدّل الاستهلاك العدواني نسبةً لمتوسط نوافذ سابقة
  (سببي، بلا البرميل الحالي في الأساس)، موقّع باتجاه ``delta``.
* ``early_imbalance`` — إشارة اختلال **في بدايته**: تسارع أوامر عدوانية بينما
  الهيكل ما زال متوازنًا على VP، أو على أول برميل قلب للتوازن→اختلال مع تسارع.

اختلال تدفّق الأوامر ``OFI`` (Cont, Kukanov & Stoica, 2014) يُحسب من تغيّرات
قمّة الدفتر (top-of-book) حدثًا بحدث:

    e_n = 1[Pᵇₙ ≥ Pᵇₙ₋₁]·qᵇₙ − 1[Pᵇₙ ≤ Pᵇₙ₋₁]·qᵇₙ₋₁
        − 1[Pᵃₙ ≤ Pᵃₙ₋₁]·qᵃₙ + 1[Pᵃₙ ≥ Pᵃₙ₋₁]·qᵃₙ₋₁

حيث Pᵇ/qᵇ سعر/حجم أفضل طلب و Pᵃ/qᵃ أفضل عرض. كل المقادير سببية تمامًا.
"""

from __future__ import annotations

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades

#: نوافذ سابقة لتقدير أساس الاستهلاك (لا تشمل البرميل الحالي).
_DEFAULT_ORDER_ACCEL_LOOKBACK = 3
#: عتبة التسارع: استهلاك الحالي ≥ مضاعف × متوسط الأساس.
_DEFAULT_ORDER_ACCEL_MULT = 1.5

ORDER_ACCEL_COLUMNS = (
    "order_accel_rate",
    "early_imbalance",
)


def order_flow_summary(frame: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يلخّص تدفّق الأوامر العدواني لكل نافذة زمنية (متاح عند ``bucket_end``)."""
    trades = extract_trades(add_time_bucket(frame, interval_ns=interval_ns))
    buckets = (
        trades.sort(EVENT_TS)
        .group_by(BUCKET_START, maintain_order=True)
        .agg(
            pl.col("buy_volume").sum(),
            pl.col("sell_volume").sum(),
            (pl.col("buy_volume") > 0).sum().alias("buy_trades"),
            (pl.col("sell_volume") > 0).sum().alias("sell_trades"),
            pl.col(BUCKET_END).first(),
            pl.col(AVAILABILITY_TS).first(),
        )
    )
    delta = pl.col("buy_volume") - pl.col("sell_volume")
    consumption = pl.col("buy_volume") + pl.col("sell_volume")
    return (
        buckets.with_columns(
            delta.alias("delta"),
            consumption.alias("consumption"),
        )
        .sort(BUCKET_START)
        .with_columns(pl.col("delta").cum_sum().alias("cumulative_delta"))
    )


def order_flow_imbalance(top_of_book: pl.DataFrame) -> pl.DataFrame:
    """يحسب OFI حدثًا بحدث من سلسلة top-of-book ومجموعه التراكمي.

    المدخل يجب أن يحوي ``best_bid``, ``bid_size``, ``best_ask``, ``ask_size``
    (مخرج ``nq.orderbook.reconstruct``). يُضيف عمودَي ``ofi`` و ``ofi_cumulative``
    و ``availability_ts`` (= ``event_ts``؛ المقدار معروف لحظة الحدث).
    """
    pb = pl.col("best_bid")
    qb = pl.col("bid_size").cast(pl.Int64)
    pa = pl.col("best_ask")
    qa = pl.col("ask_size").cast(pl.Int64)
    pb1, qb1 = pb.shift(1), qb.shift(1)
    pa1, qa1 = pa.shift(1), qa.shift(1)

    bid_part = pl.when(pb >= pb1).then(qb).otherwise(0) - pl.when(pb <= pb1).then(qb1).otherwise(0)
    ask_part = -pl.when(pa <= pa1).then(qa).otherwise(0) + pl.when(pa >= pa1).then(qa1).otherwise(0)
    ofi = (bid_part + ask_part).fill_null(0)

    return top_of_book.with_columns(ofi.alias("ofi")).with_columns(
        pl.col("ofi").cum_sum().alias("ofi_cumulative"),
        pl.col(EVENT_TS).alias(AVAILABILITY_TS),
    )


def ofi_by_bucket(top_of_book: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يجمع OFI الحدثي إلى مجموع لكل نافذة زمنية (متاح عند ``bucket_end``)."""
    per_event = order_flow_imbalance(top_of_book)
    bucketed = add_time_bucket(per_event, interval_ns=interval_ns)
    return (
        bucketed.sort(EVENT_TS)
        .group_by(BUCKET_START, maintain_order=True)
        .agg(
            pl.col("ofi").sum().alias("ofi"),
            pl.col(BUCKET_END).first(),
            pl.col(BUCKET_END).first().alias(AVAILABILITY_TS),
        )
        .sort(BUCKET_START)
        .with_columns(pl.col("ofi").cum_sum().alias("ofi_cumulative"))
        .select(BUCKET_START, "ofi", "ofi_cumulative", BUCKET_END, AVAILABILITY_TS)
    )


def order_acceleration_columns(
    frame: pl.DataFrame,
    *,
    lookback: int = _DEFAULT_ORDER_ACCEL_LOOKBACK,
    accel_mult: float = _DEFAULT_ORDER_ACCEL_MULT,
    consumption_col: str = "consumption",
    delta_col: str = "delta",
    balanced_col: str | None = "is_balanced",
    session_col: str | None = None,
) -> pl.DataFrame:
    """يعرّف معدّل تسارع الأوامر وإشارة الاختلال المبكر على براميل مرتّبة سببيًا.

    المدخل يحتاج ``delta`` و ``consumption`` (أو ``buy_volume``+``sell_volume``).
    إن وُجد ``is_balanced``: ``early_imbalance`` يُشعل عند التسارع داخل التوازن
    أو على أول برميل قلب (بداية الاختلال). بلا عمود توازن: التسارع وحده يكفي.
    ``session_col`` اختياري لتصفير الأساس عند انتقال الجلسة.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if accel_mult <= 0:
        raise ValueError(f"accel_mult must be > 0, got {accel_mult}")

    work = frame
    if consumption_col not in work.columns:
        if "buy_volume" in work.columns and "sell_volume" in work.columns:
            work = work.with_columns(
                (
                    pl.col("buy_volume").cast(pl.Float64) + pl.col("sell_volume").cast(pl.Float64)
                ).alias(consumption_col)
            )
        else:
            raise ValueError(
                f"need {consumption_col!r} or buy_volume+sell_volume; got {list(work.columns)}"
            )
    if delta_col not in work.columns:
        raise ValueError(f"need {delta_col!r} in frame columns")

    cons = pl.col(consumption_col).cast(pl.Float64)
    delta = pl.col(delta_col).cast(pl.Float64)
    # أساس سببي: متوسط الاستهلاك في النوافذ السابقة فقط (shift قبل rolling).
    past_mean = cons.shift(1).rolling_mean(window_size=lookback, min_samples=1)
    same_session = pl.lit(True)
    if session_col is not None and session_col in work.columns:
        same_session = pl.col(session_col).shift(1).fill_null(pl.col(session_col)) == pl.col(
            session_col
        )

    direction = pl.when(delta > 0).then(1.0).when(delta < 0).then(-1.0).otherwise(0.0).alias("_dir")
    rate_raw = (
        pl.when(same_session & (past_mean > 0)).then(cons / past_mean).otherwise(0.0).alias("_rate")
    )

    out = work.with_columns(direction, rate_raw).with_columns(
        (pl.col("_dir") * pl.col("_rate")).alias("order_accel_rate"),
    )

    accelerating = pl.col("_rate") >= float(accel_mult)
    directed = pl.col("_dir") != 0.0
    if balanced_col is not None and balanced_col in out.columns:
        bal = pl.col(balanced_col).fill_null(value=False)
        prev_bal = bal.shift(1).fill_null(value=True)
        onset = prev_bal & ~bal
        early_mask = accelerating & directed & (bal | onset)
    else:
        early_mask = accelerating & directed

    return (
        out.with_columns(
            pl.when(early_mask).then(pl.col("_dir")).otherwise(0.0).alias("early_imbalance"),
        )
        .drop("_dir", "_rate")
        .select(*ORDER_ACCEL_COLUMNS)
    )


__all__ = [
    "ORDER_ACCEL_COLUMNS",
    "ofi_by_bucket",
    "order_acceleration_columns",
    "order_flow_imbalance",
    "order_flow_summary",
]
