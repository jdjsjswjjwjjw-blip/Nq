"""مُحاكي تدفّق الأوامر (Order Flow Simulator).

يقيس ضغط الشراء/البيع العدواني، بادئ الصفقة (trade initiation)، اختلال تدفّق
الأوامر (OFI)، واستهلاك السيولة.

ملخّص التدفّق لكل نافذة:

* ``buy_volume`` / ``sell_volume`` — الحجم العدواني (شراء يرفع العرض / بيع يضرب الطلب).
* ``delta`` = ``buy_volume - sell_volume`` و ``cumulative_delta`` (سببي).
* ``buy_trades`` / ``sell_trades`` — عدد الصفقات البادئة شراءً/بيعًا (initiation).
* ``consumption`` = ``buy_volume + sell_volume`` — إجمالي السيولة المُستهلَكة عدوانيًا.

اختلال تدفّق الأوامر ``OFI`` (Cont, Kukanov & Stoica, 2014) يُحسب من تغيّرات
قمّة الدفتر (top-of-book) حدثًا بحدث:

    e_n = 1[Pᵇₙ ≥ Pᵇₙ₋₁]·qᵇₙ − 1[Pᵇₙ ≤ Pᵇₙ₋₁]·qᵇₙ₋₁
        − 1[Pᵃₙ ≤ Pᵃₙ₋₁]·qᵃₙ + 1[Pᵃₙ ≥ Pᵃₙ₋₁]·qᵃₙ₋₁

حيث Pᵇ/qᵇ سعر/حجم أفضل طلب و Pᵃ/qᵃ أفضل عرض. كل المقادير سببية تمامًا.
"""

from __future__ import annotations

import polars as pl

from nq.contracts.instruments import require_single_contract_identity
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.session import TRADING_SESSION_ID, add_session_columns
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades


def order_flow_summary(frame: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يلخّص تدفّق الأوامر العدواني لكل نافذة زمنية (متاح عند ``bucket_end``)."""
    require_single_contract_identity(frame, context="order_flow_summary")
    trades = extract_trades(add_time_bucket(frame, interval_ns=interval_ns))
    buckets = trades.group_by(BUCKET_START).agg(
        pl.col("buy_volume").sum(),
        pl.col("sell_volume").sum(),
        (pl.col("buy_volume") > 0).sum().alias("buy_trades"),
        (pl.col("sell_volume") > 0).sum().alias("sell_trades"),
        pl.col(BUCKET_END).first(),
        pl.col(AVAILABILITY_TS).first(),
    )
    delta = pl.col("buy_volume") - pl.col("sell_volume")
    consumption = pl.col("buy_volume") + pl.col("sell_volume")
    return (
        add_session_columns(buckets, time_col=BUCKET_END)
        .with_columns(
            delta.alias("delta"),
            consumption.alias("consumption"),
        )
        .sort([TRADING_SESSION_ID, BUCKET_START])
        .with_columns(pl.col("delta").cum_sum().over(TRADING_SESSION_ID).alias("cumulative_delta"))
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

    with_session = add_session_columns(
        top_of_book.with_columns(ofi.alias("ofi")),
        time_col=EVENT_TS,
    )
    return with_session.with_columns(
        pl.col("ofi").cum_sum().over(TRADING_SESSION_ID).alias("ofi_cumulative"),
        pl.col(EVENT_TS).alias(AVAILABILITY_TS),
    )


def ofi_by_bucket(top_of_book: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يجمع OFI الحدثي إلى مجموع لكل نافذة زمنية (متاح عند ``bucket_end``)."""
    per_event = order_flow_imbalance(top_of_book)
    bucketed = add_time_bucket(per_event, interval_ns=interval_ns)
    return (
        bucketed.group_by(BUCKET_START)
        .agg(
            pl.col("ofi").sum().alias("ofi"),
            pl.col(BUCKET_END).first(),
            pl.col(BUCKET_END).first().alias(AVAILABILITY_TS),
        )
        .pipe(add_session_columns, time_col=BUCKET_END)
        .sort([TRADING_SESSION_ID, BUCKET_START])
        .with_columns(pl.col("ofi").cum_sum().over(TRADING_SESSION_ID).alias("ofi_cumulative"))
        .select(
            BUCKET_START,
            TRADING_SESSION_ID,
            "ofi",
            "ofi_cumulative",
            BUCKET_END,
            AVAILABILITY_TS,
        )
    )


__all__ = ["ofi_by_bucket", "order_flow_imbalance", "order_flow_summary"]
