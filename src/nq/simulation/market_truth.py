"""حكم السوق اللحظي — صح/غلط الثيسيس مع هولد وتغيّر مجمّع/لحظي.

لا يلاحق كل أمر: الدخول مسموح فقط بعد **هولد** يؤكد سيولة حقيقية
(``real_liquidity_ratio`` مرتفع + درجة تضليل منخفضة) واستقرار اتجاه الثيسيس.

* ``market_verdict``: ``+1`` السوق صادق مع الثيسيس، ``-1`` كاذب، ``0`` غير حاسم.
* ``delta_instant`` / ``delta_cum``: تغيّر سعري لحظي ومجمّع أثناء نافذة الهولد.
* الحكم عند نهاية الهولد فقط (سببي: ``availability_ts = hold_end``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.simulation.auction import auction_states
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.deceptive_liquidity import (
    DECEPTIVE_FEATURE_COLUMNS,
    DeceptiveLiquidityConfig,
    deceptive_features_by_bucket,
)

MARKET_TRUTH_COLUMNS: Final[tuple[str, ...]] = (
    "thesis_dir",
    "hold_ok",
    "delta_instant",
    "delta_cum",
    "market_verdict",
    "market_true",
    "market_false",
    "entry_gate",
)


@dataclass(frozen=True, slots=True)
class MarketTruthConfig:
    """عتبات الهولد وحكم صدق السوق."""

    hold_buckets: int = 3
    min_real_liquidity: float = 0.55
    max_deceptive_score: float = 0.45
    #: الحد الأدنى |Δ سعر| بالتيك أثناء الهولد لاعتبار الحكم حاسمًا.
    min_move_ticks: float = 4.0
    tick_size: float = 0.25


def _thesis_direction(states: pl.DataFrame) -> pl.Series:
    """اتجاه الثيسيس من المزاد: اختلال صاعد/هابط عبر close مقابل POC."""
    # +1 = ثيسيس صاعد (اختلال مع إغلاق فوق POC أو تمدّد لأعلى)
    # -1 = ثيسيس هابط
    close = states["close"].cast(pl.Float64)
    poc = states["poc"].cast(pl.Float64)
    imbalanced = ~states["is_balanced"]
    above = close > poc
    below = close < poc
    dir_expr = (
        pl.when(imbalanced & above)
        .then(1.0)
        .when(imbalanced & below)
        .then(-1.0)
        .otherwise(0.0)
    )
    return states.select(dir_expr.alias("thesis_dir"))["thesis_dir"]


def build_market_truth_frame(  # noqa: PLR0912, PLR0915
    mbo: pl.DataFrame,
    *,
    interval_ns: int,
    truth: MarketTruthConfig | None = None,
    deceptive: DeceptiveLiquidityConfig | None = None,
    score_mbo: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
    auction: pl.DataFrame | None = None,
    deceptive_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """يبني إطار حكم السوق + بوابة دخول (بدون ملاحقة كل أمر).

    ``mbo``: مصدر حالات المزاد (عادة بعد تنظيف الدفتر).
    ``score_mbo``: مصدر درجات التضليل للهولد — يجب أن يكون **الخام قبل الإسقاط**
    وإلا تصبح بوابات السيولة الحقيقية بلا معنى بعد التنظيف.
    ``auction`` / ``deceptive_frame``: اختياري لإعادة الاستخدام عبر شبكة بحث الإدج
    (تجنّب إعادة بناء المزاد/التضليل لكل مواصفة).
    """
    cfg = truth if truth is not None else MarketTruthConfig()
    if cfg.hold_buckets < 1:
        raise ValueError(f"hold_buckets must be >= 1, got {cfg.hold_buckets}")

    if progress is not None and (auction is None or deceptive_frame is None):
        progress.op("market_truth: auction_states + deceptive buckets")
    states = (
        auction
        if auction is not None
        else auction_states(mbo, interval_ns=interval_ns, progress=progress)
    )
    if deceptive_frame is not None:
        deco = deceptive_frame
    else:
        score_src = score_mbo if score_mbo is not None else mbo
        deco = deceptive_features_by_bucket(
            score_src, interval_ns=interval_ns, config=deceptive, progress=progress
        )
    if states.height == 0:
        return pl.DataFrame(
            schema={
                AVAILABILITY_TS: pl.Int64(),
                BUCKET_START: pl.Int64(),
                BUCKET_END: pl.Int64(),
                "close": pl.Float64(),
                "poc": pl.Float64(),
                "vah": pl.Float64(),
                "val": pl.Float64(),
                "is_balanced": pl.Boolean(),
                "is_expansion": pl.Boolean(),
                "pullback_defended": pl.Boolean(),
                **{c: pl.Float64() for c in DECEPTIVE_FEATURE_COLUMNS},
                **{c: pl.Float64() for c in MARKET_TRUTH_COLUMNS},
            }
        )

    merged = (
        states.join(deco, on=BUCKET_START, how="left", suffix="_d")
        .sort(BUCKET_START)
        .with_columns(
            pl.col("deceptive_score").fill_null(0.0),
            pl.col("real_liquidity_ratio").fill_null(1.0),
            pl.col("noise_instant").fill_null(0.0),
            pl.col("noise_cum").fill_null(0.0),
            pl.col("deceptive_volume_share").fill_null(0.0),
            pl.col("deceptive_cancel_rate").fill_null(0.0),
        )
    )
    # تفضيل availability من نهاية برميل المزاد
    if f"{AVAILABILITY_TS}_d" in merged.columns:
        merged = merged.drop(f"{AVAILABILITY_TS}_d")
    if f"{BUCKET_END}_d" in merged.columns:
        merged = merged.drop(f"{BUCKET_END}_d")

    thesis = _thesis_direction(merged).to_numpy()
    close = merged["close"].to_numpy().astype(np.float64)
    real_liq = merged["real_liquidity_ratio"].to_numpy().astype(np.float64)
    dec_score = merged["deceptive_score"].to_numpy().astype(np.float64)
    n = len(close)
    hold = int(cfg.hold_buckets)
    tick = float(cfg.tick_size)
    min_move = cfg.min_move_ticks * tick

    hold_ok = np.zeros(n, dtype=np.float64)
    delta_instant = np.full(n, np.nan, dtype=np.float64)
    delta_cum = np.full(n, np.nan, dtype=np.float64)
    verdict = np.zeros(n, dtype=np.float64)
    market_true = np.zeros(n, dtype=np.float64)
    market_false = np.zeros(n, dtype=np.float64)
    entry_gate = np.zeros(n, dtype=np.float64)
    # availability عند اكتمال الهولد
    avail = merged[AVAILABILITY_TS].to_numpy().astype(np.int64).copy()
    bucket_end = merged[BUCKET_END].to_numpy().astype(np.int64)

    for t in range(n):
        end = t + hold - 1
        if end >= n:
            continue
        # هولد: كل براميل النافذة سيولة حقيقية + تضليل منخفض + ثيسيس غير صفر
        window_thesis = thesis[t : end + 1]
        if not np.all(window_thesis == window_thesis[0]) or window_thesis[0] == 0.0:
            continue
        if np.any(real_liq[t : end + 1] < cfg.min_real_liquidity):
            continue
        if np.any(dec_score[t : end + 1] > cfg.max_deceptive_score):
            continue
        hold_ok[end] = 1.0
        # تغيّر لحظي = آخر برميل؛ مجمّع = من أول برميل هولد لآخره
        # الأسعار في auction بالوحدات الثابتة — نحوّل لسعر حقيقي
        c0 = close[t] * PRICE_SCALE
        c1 = close[end] * PRICE_SCALE
        c_prev = close[end - 1] * PRICE_SCALE if end > t else c0
        delta_cum[end] = c1 - c0
        delta_instant[end] = c1 - c_prev
        direction = float(window_thesis[0])
        move = c1 - c0
        if abs(move) < min_move:
            verdict[end] = 0.0
        elif move * direction > 0:
            verdict[end] = 1.0
            market_true[end] = 1.0
        else:
            verdict[end] = -1.0
            market_false[end] = 1.0
        # بوابة دخول: هولد ناجح + سوق صادق (ليس محايدًا ولا كاذبًا)
        if verdict[end] > 0.0:
            entry_gate[end] = 1.0
        # الإتاحة = نهاية برميل اكتمال الهولد
        avail[end] = int(bucket_end[end])

    out = merged.with_columns(
        pl.Series("thesis_dir", thesis, dtype=pl.Float64),
        pl.Series("hold_ok", hold_ok, dtype=pl.Float64),
        pl.Series("delta_instant", delta_instant, dtype=pl.Float64),
        pl.Series("delta_cum", delta_cum, dtype=pl.Float64),
        pl.Series("market_verdict", verdict, dtype=pl.Float64),
        pl.Series("market_true", market_true, dtype=pl.Float64),
        pl.Series("market_false", market_false, dtype=pl.Float64),
        pl.Series("entry_gate", entry_gate, dtype=pl.Float64),
        pl.Series(AVAILABILITY_TS, avail, dtype=pl.Int64),
    )
    return out


__all__ = [
    "MARKET_TRUTH_COLUMNS",
    "MarketTruthConfig",
    "build_market_truth_frame",
]
