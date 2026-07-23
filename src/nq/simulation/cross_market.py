"""مُحاكي عبر السوقين (Cross-Market Simulator) — NQ ↔ MNQ.

يقارن السوق القائد (NQ، الكبير الأعلى سيولة) بالسوق التابع (MNQ، المصغّر الذي
يغلب فيه تجّار التجزئة)، على شبكة زمنية موحّدة (نوافذ ``interval_ns``)، ويشتق:

* القيادة/التأخّر (Lead/Lag): ارتباطات متدحرجة بين عوائد السوقين مع إزاحة زمنية
  لتحديد أيّهما يقود (``nq_leads_corr`` مقابل ``mnq_leads_corr``، وإشارة ``lead_lag``).
* التباعد (Divergence): اختلاف إشارة العائد بين السوقين.
* فشل التأكيد (Confirmation Failure): سوق يصنع نهايةً جديدة (قمة/قاع **داخل
  جلسة ET الحالية**) دون أن يؤكّدها الآخر.
* مصيدة المتداولين (Trader Trap): إشارة سببية لاحتمال إيقاع المتداولين العدوانيين
  في MNQ — حين يصنع MNQ نهايةً جديدة بدلتا عدوانية قويّة أحادية الاتجاه بينما
  يفشل NQ في التأكيد (تباعد قيادي).
* ``session_phase`` / ``minutes_since_rth_open`` / ``session_date`` — سياق جلسة.
* محاذاة MNQ مع تأخير ``latency_ns`` (NQ يقود MNQ).

منع التسريب: كل الميزات مُجمّعة على نوافذ ومتاحة عند ``bucket_end``، وكل
الإزاحات/الارتباطات المتدحرجة تستخدم الماضي فقط. لا تُحسب أي نتيجة مستقبلية
(مثل "تأكيد الانعكاس") ضمن الميزات — إشارة المصيدة هي إعداد (setup) سببي فقط.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, INGEST_TS
from nq.core.session import TRADING_SESSION_ID, add_session_columns, trading_session_id_from_ns
from nq.orderbook import reconstruct
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket
from nq.simulation.order_flow import order_flow_summary

_DEFAULT_LEAD_LAG_WINDOW = 20
_DEFAULT_MIN_TRAP_DELTA = 1
_MIN_LEAD_LAG_WINDOW = 2
AvailabilityMode = Literal["ingest", "event"]


def _bucket_availability(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    availability_mode: AvailabilityMode,
) -> pl.DataFrame:
    bucketed = add_time_bucket(frame, interval_ns=interval_ns)
    if availability_mode == "event" or INGEST_TS not in bucketed.columns:
        return bucketed.group_by(BUCKET_START).agg(
            pl.col(BUCKET_END).first(),
            pl.col(BUCKET_END).first().alias(AVAILABILITY_TS),
        )
    if availability_mode != "ingest":
        raise ValueError(
            f"availability_mode must be 'ingest' or 'event', got {availability_mode!r}"
        )
    return (
        bucketed.group_by(BUCKET_START)
        .agg(
            pl.col(BUCKET_END).first(),
            pl.col(INGEST_TS).max().alias("_max_ingest_ts"),
        )
        .with_columns(
            pl.max_horizontal(pl.col(BUCKET_END), pl.col("_max_ingest_ts")).alias(AVAILABILITY_TS)
        )
        .drop("_max_ingest_ts")
    )


def _market_windows(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    availability_mode: AvailabilityMode,
) -> pl.DataFrame:
    """يبني سلسلة نافذية لسوق واحد: سعر الإغلاق (mid) والدلتا العدوانية."""
    tob = reconstruct(frame).top_of_book
    both = pl.col("best_bid").is_not_null() & pl.col("best_ask").is_not_null()
    tob = add_time_bucket(
        tob.with_columns(
            pl.when(both)
            .then((pl.col("best_bid") + pl.col("best_ask")) / 2.0)
            .otherwise(None)
            .alias("mid")
        ),
        interval_ns=interval_ns,
    )
    close = (
        tob.filter(pl.col("mid").is_not_null())
        .group_by(BUCKET_START)
        .agg(
            pl.col("mid").last().alias("close"),
            pl.col("best_bid").last().alias("bid"),
            pl.col("best_ask").last().alias("ask"),
            pl.col(BUCKET_END).first(),
        )
    )
    flow = order_flow_summary(frame, interval_ns=interval_ns).select(BUCKET_START, "delta")
    availability = _bucket_availability(
        frame, interval_ns=interval_ns, availability_mode=availability_mode
    ).select(BUCKET_START, AVAILABILITY_TS)
    return (
        close.join(flow, on=BUCKET_START, how="full", coalesce=True)
        .join(availability, on=BUCKET_START, how="left")
        .sort(BUCKET_START)
    )


def _rolling_corr(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """ارتباط متدحرج سببي (يستخدم الماضي فقط) عبر نافذة بطول ``window``."""
    mean_x = x.rolling_mean(window_size=window, min_samples=window)
    mean_y = y.rolling_mean(window_size=window, min_samples=window)
    mean_xy = (x * y).rolling_mean(window_size=window, min_samples=window)
    mean_xx = (x * x).rolling_mean(window_size=window, min_samples=window)
    mean_yy = (y * y).rolling_mean(window_size=window, min_samples=window)
    cov = mean_xy - mean_x * mean_y
    std_x = (mean_xx - mean_x * mean_x).sqrt()
    std_y = (mean_yy - mean_y * mean_y).sqrt()
    denom = std_x * std_y
    return pl.when(denom > 0).then(cov / denom).otherwise(None)


def _align_markets(
    nq_w: pl.DataFrame,
    mnq_w: pl.DataFrame,
    *,
    latency_ns: int,
) -> pl.DataFrame:
    """يحاذي NQ مع MNQ مع تأخير سببي ``latency_ns`` (MNQ عند t−latency)."""
    nq_renamed = nq_w.rename(
        {
            "close": "nq_close",
            "bid": "nq_bid",
            "ask": "nq_ask",
            "delta": "nq_delta",
            AVAILABILITY_TS: "nq_availability_ts",
        }
    )
    mnq_renamed = mnq_w.rename(
        {
            "close": "mnq_close",
            "delta": "mnq_delta",
            BUCKET_START: "mnq_bucket_start",
            BUCKET_END: "mnq_bucket_end",
            AVAILABILITY_TS: "mnq_availability_ts",
        }
    )

    nq_renamed = nq_renamed.with_columns(
        (pl.col("nq_availability_ts") - latency_ns).alias("_mnq_available_by")
    )
    aligned = nq_renamed.sort("_mnq_available_by").join_asof(
        mnq_renamed.sort("mnq_availability_ts"),
        left_on="_mnq_available_by",
        right_on="mnq_availability_ts",
        strategy="backward",
    )

    return (
        aligned.sort(BUCKET_START)
        .with_columns(
            pl.col("nq_availability_ts").alias(AVAILABILITY_TS),
            pl.col("nq_delta").fill_null(0),
            pl.col("mnq_delta").fill_null(0),
        )
        .drop("_mnq_available_by", strict=False)
    )


def single_market_features(
    nq: pl.DataFrame,
    *,
    interval_ns: int,
    availability_mode: AvailabilityMode = "ingest",
) -> pl.DataFrame:
    """يبني ساعة NQ فقط دون أي أعمدة MNQ أو cross-market."""
    windows = _market_windows(
        nq,
        interval_ns=interval_ns,
        availability_mode=availability_mode,
    ).rename(
        {
            "close": "nq_close",
            "bid": "nq_bid",
            "ask": "nq_ask",
            "delta": "nq_delta",
        }
    )
    if windows.height == 0:
        return windows
    sessions = [trading_session_id_from_ns(int(t)) for t in windows[AVAILABILITY_TS].to_list()]
    features = windows.with_columns(
        pl.col("nq_close").diff().alias("nq_return"),
        pl.Series(TRADING_SESSION_ID, sessions, dtype=pl.Utf8()),
    )
    prev_max = (
        pl.col("nq_close").cum_max().over(TRADING_SESSION_ID).shift(1).over(TRADING_SESSION_ID)
    )
    prev_min = (
        pl.col("nq_close").cum_min().over(TRADING_SESSION_ID).shift(1).over(TRADING_SESSION_ID)
    )
    features = features.with_columns(
        (pl.col("nq_close") > prev_max).fill_null(value=False).alias("nq_new_high"),
        (pl.col("nq_close") < prev_min).fill_null(value=False).alias("nq_new_low"),
        pl.col("nq_delta").fill_null(0),
    )
    return add_session_columns(features.drop(TRADING_SESSION_ID), time_col=AVAILABILITY_TS)


def cross_market_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    lead_lag_window: int = _DEFAULT_LEAD_LAG_WINDOW,
    min_trap_delta: int = _DEFAULT_MIN_TRAP_DELTA,
    latency_ns: int = 0,
    availability_mode: AvailabilityMode = "ingest",
) -> pl.DataFrame:
    """يشتق ميزات عبر السوقين على شبكة زمنية موحّدة (متاحة عند ``bucket_end``)."""
    if lead_lag_window < _MIN_LEAD_LAG_WINDOW:
        raise ValueError(f"lead_lag_window must be >= 2, got {lead_lag_window}")
    if latency_ns < 0:
        raise ValueError(f"latency_ns must be non-negative, got {latency_ns}")

    nq_w = _market_windows(nq, interval_ns=interval_ns, availability_mode=availability_mode)
    mnq_w = _market_windows(mnq, interval_ns=interval_ns, availability_mode=availability_mode)
    aligned = _align_markets(nq_w, mnq_w, latency_ns=latency_ns)
    if aligned.height == 0:
        return aligned

    nq_ret = pl.col("nq_close").diff()
    mnq_ret = pl.col("mnq_close").diff()
    # قمم/قيعان جلسة CME الحالية فقط — لا cum_max عالمي ولا تاريخ تقويم ET.
    session_ids = [trading_session_id_from_ns(int(t)) for t in aligned[AVAILABILITY_TS].to_list()]
    aligned = aligned.with_columns(
        nq_ret.alias("nq_return"),
        mnq_ret.alias("mnq_return"),
        pl.Series(TRADING_SESSION_ID, session_ids, dtype=pl.Utf8()),
    )

    nq_prev_max = (
        pl.col("nq_close").cum_max().over(TRADING_SESSION_ID).shift(1).over(TRADING_SESSION_ID)
    )
    nq_prev_min = (
        pl.col("nq_close").cum_min().over(TRADING_SESSION_ID).shift(1).over(TRADING_SESSION_ID)
    )
    mnq_prev_max = (
        pl.col("mnq_close").cum_max().over(TRADING_SESSION_ID).shift(1).over(TRADING_SESSION_ID)
    )
    mnq_prev_min = (
        pl.col("mnq_close").cum_min().over(TRADING_SESSION_ID).shift(1).over(TRADING_SESSION_ID)
    )

    nq_new_high = pl.col("nq_close") > nq_prev_max
    nq_new_low = pl.col("nq_close") < nq_prev_min
    mnq_new_high = pl.col("mnq_close") > mnq_prev_max
    mnq_new_low = pl.col("mnq_close") < mnq_prev_min

    aligned = aligned.with_columns(
        nq_new_high.fill_null(value=False).alias("nq_new_high"),
        nq_new_low.fill_null(value=False).alias("nq_new_low"),
        mnq_new_high.fill_null(value=False).alias("mnq_new_high"),
        mnq_new_low.fill_null(value=False).alias("mnq_new_low"),
        _rolling_corr(pl.col("nq_return").shift(1), pl.col("mnq_return"), lead_lag_window).alias(
            "nq_leads_corr"
        ),
        _rolling_corr(pl.col("mnq_return").shift(1), pl.col("nq_return"), lead_lag_window).alias(
            "mnq_leads_corr"
        ),
    )

    divergence = (pl.col("nq_return").sign() * pl.col("mnq_return").sign()) < 0
    confirmation_failure = (
        (pl.col("nq_new_high") & ~pl.col("mnq_new_high"))
        | (pl.col("mnq_new_high") & ~pl.col("nq_new_high"))
        | (pl.col("nq_new_low") & ~pl.col("mnq_new_low"))
        | (pl.col("mnq_new_low") & ~pl.col("nq_new_low"))
    )
    lead_lag = (
        pl.when(pl.col("nq_leads_corr") > pl.col("mnq_leads_corr"))
        .then(1)
        .when(pl.col("nq_leads_corr") < pl.col("mnq_leads_corr"))
        .then(-1)
        .otherwise(0)
    )
    trap_setup = (
        pl.when(
            pl.col("mnq_new_high")
            & (pl.col("mnq_delta") >= min_trap_delta)
            & ~pl.col("nq_new_high")
        )
        .then(1)
        .when(
            pl.col("mnq_new_low") & (pl.col("mnq_delta") <= -min_trap_delta) & ~pl.col("nq_new_low")
        )
        .then(-1)
        .otherwise(0)
    )

    result = aligned.with_columns(
        divergence.fill_null(value=False).alias("divergence"),
        confirmation_failure.alias("confirmation_failure"),
        lead_lag.alias("lead_lag"),
        trap_setup.alias("trap_setup"),
    )
    return add_session_columns(result.drop(TRADING_SESSION_ID), time_col=AVAILABILITY_TS)


__all__ = ["cross_market_features", "single_market_features"]
