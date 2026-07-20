"""مُحاكي عبر السوقين (Cross-Market Simulator) — NQ ↔ MNQ.

يقارن السوق القائد (NQ، الكبير الأعلى سيولة) بالسوق التابع (MNQ، المصغّر الذي
يغلب فيه تجّار التجزئة)، على شبكة زمنية موحّدة (نوافذ ``interval_ns``)، ويشتق:

* القيادة/التأخّر (Lead/Lag): ارتباطات متدحرجة بين عوائد السوقين مع إزاحة زمنية
  لتحديد أيّهما يقود (``nq_leads_corr`` مقابل ``mnq_leads_corr``، وإشارة ``lead_lag``).
* التباعد (Divergence): اختلاف إشارة العائد بين السوقين.
* فشل التأكيد (Confirmation Failure): سوق يصنع نهايةً جديدة (قمة/قاع) دون أن
  يؤكّدها الآخر.
* مصيدة المتداولين (Trader Trap): إشارة سببية لاحتمال إيقاع المتداولين العدوانيين
  في MNQ — حين يصنع MNQ نهايةً جديدة بدلتا عدوانية قويّة أحادية الاتجاه بينما
  يفشل NQ في التأكيد (تباعد قيادي).

منع التسريب: كل الميزات مُجمّعة على نوافذ ومتاحة عند ``bucket_end``، وكل
الإزاحات/الارتباطات المتدحرجة تستخدم الماضي فقط. لا تُحسب أي نتيجة مستقبلية
(مثل "تأكيد الانعكاس") ضمن الميزات — إشارة المصيدة هي إعداد (setup) سببي فقط.
"""

from __future__ import annotations

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.orderbook import reconstruct
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket
from nq.simulation.order_flow import order_flow_summary

_DEFAULT_LEAD_LAG_WINDOW = 20
_DEFAULT_MIN_TRAP_DELTA = 1
_MIN_LEAD_LAG_WINDOW = 2


def _market_windows(frame: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
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
    return close.join(flow, on=BUCKET_START, how="full", coalesce=True).sort(BUCKET_START)


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


def cross_market_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    lead_lag_window: int = _DEFAULT_LEAD_LAG_WINDOW,
    min_trap_delta: int = _DEFAULT_MIN_TRAP_DELTA,
) -> pl.DataFrame:
    """يشتق ميزات عبر السوقين على شبكة زمنية موحّدة (متاحة عند ``bucket_end``)."""
    if lead_lag_window < _MIN_LEAD_LAG_WINDOW:
        raise ValueError(f"lead_lag_window must be >= 2, got {lead_lag_window}")

    nq_w = _market_windows(nq, interval_ns=interval_ns).rename(
        {"close": "nq_close", "bid": "nq_bid", "ask": "nq_ask", "delta": "nq_delta"}
    )
    mnq_w = _market_windows(mnq, interval_ns=interval_ns).rename(
        {"close": "mnq_close", "delta": "mnq_delta", BUCKET_END: "mnq_bucket_end"}
    )
    aligned = (
        nq_w.join(mnq_w, on=BUCKET_START, how="inner")
        .sort(BUCKET_START)
        .with_columns(
            pl.coalesce(pl.col(BUCKET_END), pl.col("mnq_bucket_end")).alias(BUCKET_END),
            pl.col("nq_delta").fill_null(0),
            pl.col("mnq_delta").fill_null(0),
        )
        .drop("mnq_bucket_end")
    )
    if aligned.height == 0:
        return aligned

    nq_ret = pl.col("nq_close").diff()
    mnq_ret = pl.col("mnq_close").diff()
    aligned = aligned.with_columns(nq_ret.alias("nq_return"), mnq_ret.alias("mnq_return"))

    nq_new_high = pl.col("nq_close") > pl.col("nq_close").cum_max().shift(1)
    nq_new_low = pl.col("nq_close") < pl.col("nq_close").cum_min().shift(1)
    mnq_new_high = pl.col("mnq_close") > pl.col("mnq_close").cum_max().shift(1)
    mnq_new_low = pl.col("mnq_close") < pl.col("mnq_close").cum_min().shift(1)

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

    return aligned.with_columns(
        divergence.fill_null(value=False).alias("divergence"),
        confirmation_failure.alias("confirmation_failure"),
        lead_lag.alias("lead_lag"),
        trap_setup.alias("trap_setup"),
        pl.col(BUCKET_END).alias(AVAILABILITY_TS),
    )


__all__ = ["cross_market_features"]
