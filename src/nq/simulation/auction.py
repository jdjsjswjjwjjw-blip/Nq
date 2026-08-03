"""مُحاكي المزاد (Auction Market Simulator).

يستند إلى نظرية المزاد ومنطقة القيمة لوصف حالة السوق لكل نافذة زمنية:

* التوازن/الاختلال (Balance / Imbalance): السوق **متوازن** حين يُغلق داخل منطقة
  القيمة الجلسية المتطوّرة (قبول القيمة، ``close_in_value``) دون تمدّد مدى،
  ومع بقاء حصّة حجم **النافذة الحالية** داخل [VAL,VAH] فوق العتبة. و**مختلّ**
  حين يُغلق خارج القيمة أو مع تمدّد. ``in_value_fraction`` ذو معنى فقط مع
  VA تراكمي (لا micro-profile معزول لكل نافذة).
* التمدّد (Expansion): ``expansion_ratio = range_t / range_{t-1}``.
* دفاع الارتداد (Pullback Defense): نهاية جديدة ثم إغلاق داخل القيمة.
* آلة حالات المزاد (FSM): توازن → كسر → تسارع → ريتست يحمي المركز → توسّع.

كل الحالات سببية: كل صف يعتمد على نافذته والنوافذ السابقة فقط، ومتاح عند
``bucket_end``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.time import sort_causal
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_START, add_time_bucket, extract_trades
from nq.simulation.volume_profile import developing_value_area

_DEFAULT_BALANCE_THRESHOLD = 0.6
_DEFAULT_EXPANSION_THRESHOLD = 1.5
#: أقصى براميل بعد الكسر لاعتبار الريتست (على فريم 1ث ≈ نصف دقيقة).
_DEFAULT_RETEST_WINDOW = 30
#: نافذة حجم للتسارع السببي.
_DEFAULT_ACCEL_LOOKBACK = 10
_DEFAULT_ACCEL_MULT = 1.5

_FSM_SIGNAL_COLUMNS = (
    "vp_fsm_break",
    "vp_fsm_accel",
    "vp_fsm_retest",
    "vp_fsm_expand",
    "vp_auction_setup",
)


def auction_states(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    fraction: float = 0.7,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
    expansion_threshold: float = _DEFAULT_EXPANSION_THRESHOLD,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يصنّف حالة المزاد لكل نافذة زمنية (متاح عند ``bucket_end``).

    يستخدم منطقة قيمة **تراكمية** عبر النوافذ (قبول/رفض القيمة الجلسي).
    """
    dva = developing_value_area(
        frame,
        interval_ns=interval_ns,
        fraction=fraction,
        cumulative=True,
        progress=progress,
    )
    if dva.height == 0:
        return dva.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("high"),
            pl.lit(None, dtype=pl.Int64).alias("low"),
            pl.lit(None, dtype=pl.Int64).alias("close"),
        )

    trades = extract_trades(add_time_bucket(sort_causal(frame), interval_ns=interval_ns))

    stats = trades.sort(EVENT_TS).group_by(BUCKET_START, maintain_order=True).agg(
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("size").cast(pl.Int64).sum().alias("bucket_volume"),
    )

    # حجم صفقات النافذة الحالية داخل منطقة القيمة الجلسية المتطوّرة
    va_bounds = dva.select(BUCKET_START, "vah", "val")
    in_value = (
        trades.join(va_bounds, on=BUCKET_START, how="left")
        .filter((pl.col("price") >= pl.col("val")) & (pl.col("price") <= pl.col("vah")))
        .group_by(BUCKET_START)
        .agg(pl.col("size").cast(pl.Int64).sum().alias("in_value_volume"))
    )

    merged = (
        dva.join(stats, on=BUCKET_START, how="left")
        .join(in_value, on=BUCKET_START, how="left")
        .with_columns(pl.col("in_value_volume").fill_null(0))
        .sort(BUCKET_START)
    )

    price_range = pl.col("high") - pl.col("low")
    prev_range = price_range.shift(1)
    prev_high = pl.col("high").shift(1)
    prev_low = pl.col("low").shift(1)
    in_value_fraction = (
        pl.when(pl.col("bucket_volume") > 0)
        .then(pl.col("in_value_volume") / pl.col("bucket_volume"))
        .otherwise(0.0)
    )
    expansion_ratio = (
        pl.when((prev_range.is_not_null()) & (prev_range > 0))
        .then(price_range / prev_range)
        .otherwise(None)
    )
    made_new_high = (prev_high.is_not_null()) & (pl.col("high") > prev_high)
    made_new_low = (prev_low.is_not_null()) & (pl.col("low") < prev_low)
    closed_in_value = (pl.col("close") >= pl.col("val")) & (pl.col("close") <= pl.col("vah"))
    is_expansion = expansion_ratio.is_not_null() & (expansion_ratio >= expansion_threshold)

    return merged.with_columns(
        price_range.alias("range"),
        in_value_fraction.alias("in_value_fraction"),
        expansion_ratio.alias("expansion_ratio"),
        made_new_high.alias("made_new_high"),
        made_new_low.alias("made_new_low"),
        closed_in_value.alias("close_in_value"),
        is_expansion.alias("is_expansion"),
    ).with_columns(
        # مع VA تراكمي: حصّة حجم النافذة داخل القيمة مقياس حيّ (قد تكون < fraction)
        (
            pl.col("close_in_value")
            & ~pl.col("is_expansion")
            & (pl.col("in_value_fraction") >= balance_threshold)
        ).alias("is_balanced"),
        ((pl.col("made_new_high") | pl.col("made_new_low")) & pl.col("close_in_value")).alias(
            "pullback_defended"
        ),
    )


def auction_fsm_columns(
    states: pl.DataFrame,
    *,
    retest_window: int = _DEFAULT_RETEST_WINDOW,
    accel_lookback: int = _DEFAULT_ACCEL_LOOKBACK,
    accel_mult: float = _DEFAULT_ACCEL_MULT,
) -> pl.DataFrame:
    """آلة حالات مزاد سببية على براميل ``auction_states``.

    التسلسل:
    توازن → كسر حدود VA → تسارع حجم → ريتست يحمي المركز → توسّع مع الاتجاه.

    ``vp_auction_setup`` = اتجاه الإعداد المكتمل (+1/−1) عند اكتمال السلسلة؛ 0 وإلا.
    """
    n = states.height
    empty = {c: pl.Series(c, [0.0] * n if n else [], dtype=pl.Float64) for c in _FSM_SIGNAL_COLUMNS}
    if n == 0:
        return pl.DataFrame(empty)

    balanced = states["is_balanced"].to_numpy()
    close = states["close"].to_numpy().astype(np.float64)
    vah = states["vah"].to_numpy().astype(np.float64)
    val = states["val"].to_numpy().astype(np.float64)
    vol = states["bucket_volume"].to_numpy().astype(np.float64)
    expansion = states["is_expansion"].to_numpy()
    pullback = states["pullback_defended"].to_numpy()
    in_value = states["close_in_value"].to_numpy()

    brk = np.zeros(n, dtype=np.float64)
    accel = np.zeros(n, dtype=np.float64)
    retest = np.zeros(n, dtype=np.float64)
    expand = np.zeros(n, dtype=np.float64)
    setup = np.zeros(n, dtype=np.float64)

    pending_dir = 0.0
    pending_i = -1
    saw_accel = False
    saw_retest = False

    for i in range(n):
        prev_bal = bool(balanced[i - 1]) if i > 0 else False
        # كسر: كان متوازنًا والآن إغلاق خارج VA
        if prev_bal and not bool(balanced[i]):
            if close[i] > vah[i]:
                brk[i] = 1.0
            elif close[i] < val[i]:
                brk[i] = -1.0
            if brk[i] != 0.0:
                pending_dir = float(brk[i])
                pending_i = i
                saw_accel = False
                saw_retest = False

        if pending_dir == 0.0 or pending_i < 0:
            continue
        age = i - pending_i
        if age > retest_window and not saw_retest:
            # انتهت نافذة الريتست بلا حماية
            pending_dir = 0.0
            pending_i = -1
            continue

        # تسارع حجم سببي بعد الكسر
        if age >= 1 and not saw_accel:
            start = max(0, i - accel_lookback)
            hist = vol[start:i]
            base = float(np.mean(hist)) if hist.size else 0.0
            if base > 0 and vol[i] >= accel_mult * base:
                accel[i] = pending_dir
                saw_accel = True

        # ريتست: العودة لحدود القيمة بعد الكسر مع دفاع/قبول القيمة.
        if age >= 1 and saw_accel and not saw_retest:
            va_w = max(float(vah[i] - val[i]), 1.0)
            if pending_dir > 0:
                near_break = abs(float(close[i] - vah[i])) <= 0.15 * va_w
            else:
                near_break = abs(float(close[i] - val[i])) <= 0.15 * va_w
            if near_break and (bool(pullback[i]) or bool(in_value[i])):
                retest[i] = pending_dir
                saw_retest = True

        # توسّع مع اتجاه الكسر بعد الريتست
        if saw_retest and bool(expansion[i]):
            if pending_dir > 0 and close[i] >= vah[i]:
                expand[i] = pending_dir
                setup[i] = pending_dir
                pending_dir = 0.0
                pending_i = -1
                saw_accel = False
                saw_retest = False
            elif pending_dir < 0 and close[i] <= val[i]:
                expand[i] = pending_dir
                setup[i] = pending_dir
                pending_dir = 0.0
                pending_i = -1
                saw_accel = False
                saw_retest = False

    return pl.DataFrame(
        {
            "vp_fsm_break": brk,
            "vp_fsm_accel": accel,
            "vp_fsm_retest": retest,
            "vp_fsm_expand": expand,
            "vp_auction_setup": setup,
        }
    )


def auction_signal_frame(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    fraction: float = 0.7,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
    expansion_threshold: float = _DEFAULT_EXPANSION_THRESHOLD,
    retest_window: int = _DEFAULT_RETEST_WINDOW,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """إشارات بحثية من Volume Profile + المزاد (توازن/اختلال/تمدّد + FSM).

    جاهزة للدمج asof في إطار البحث الموحّد:

    * ``vp_balance`` — ``+1`` متوازن، ``-1`` مختلّ.
    * ``vp_imbalance`` — ``1`` عند الاختلال، ``0`` وإلا.
    * ``vp_expansion`` — ``1`` عند تمدّد المدى، ``0`` وإلا.
    * ``vp_close_in_value`` — قبول الإغلاق داخل منطقة القيمة.
    * ``vp_in_value_frac`` — حصّة الحجم داخل [VAL, VAH].
    * ``vp_pullback_defense`` — دفاع ارتداد إلى القيمة.
    * ``vp_poc_migration`` — إزاحة POC عن النافذة السابقة (سببي).
    * ``vp_flip_to_imbalance`` — انتقال من توازن → اختلال.
    * ``vp_fsm_*`` / ``vp_auction_setup`` — سلسلة المزاد السببية.
    """
    if progress is not None:
        progress.op(f"auction_signal_frame: بناء حالات المزاد · interval_ns={interval_ns}")
    states = auction_states(
        frame,
        interval_ns=interval_ns,
        fraction=fraction,
        balance_threshold=balance_threshold,
        expansion_threshold=expansion_threshold,
        progress=progress,
    )
    base_schema = {
        AVAILABILITY_TS: pl.Int64(),
        "vp_balance": pl.Float64(),
        "vp_imbalance": pl.Float64(),
        "vp_expansion": pl.Float64(),
        "vp_close_in_value": pl.Float64(),
        "vp_in_value_frac": pl.Float64(),
        "vp_pullback_defense": pl.Float64(),
        "vp_poc_migration": pl.Float64(),
        "vp_flip_to_imbalance": pl.Float64(),
        **{c: pl.Float64() for c in _FSM_SIGNAL_COLUMNS},
    }
    if states.height == 0:
        return pl.DataFrame(schema=base_schema)

    classic = (
        states.sort(BUCKET_START)
        .with_columns(
            pl.when(pl.col("is_balanced")).then(1.0).otherwise(-1.0).alias("vp_balance"),
            pl.when(~pl.col("is_balanced")).then(1.0).otherwise(0.0).alias("vp_imbalance"),
            pl.when(pl.col("is_expansion")).then(1.0).otherwise(0.0).alias("vp_expansion"),
            pl.when(pl.col("close_in_value")).then(1.0).otherwise(0.0).alias("vp_close_in_value"),
            pl.col("in_value_fraction").cast(pl.Float64).alias("vp_in_value_frac"),
            pl.when(pl.col("pullback_defended"))
            .then(1.0)
            .otherwise(0.0)
            .alias("vp_pullback_defense"),
            pl.col("poc_migration").cast(pl.Float64).alias("vp_poc_migration"),
            (pl.col("is_balanced").shift(1).fill_null(value=False) & ~pl.col("is_balanced"))
            .cast(pl.Float64)
            .alias("vp_flip_to_imbalance"),
        )
        .select(
            AVAILABILITY_TS,
            "vp_balance",
            "vp_imbalance",
            "vp_expansion",
            "vp_close_in_value",
            "vp_in_value_frac",
            "vp_pullback_defense",
            "vp_poc_migration",
            "vp_flip_to_imbalance",
        )
    )
    fsm = auction_fsm_columns(states.sort(BUCKET_START), retest_window=retest_window)
    return classic.hstack(fsm)


__all__ = [
    "auction_fsm_columns",
    "auction_signal_frame",
    "auction_states",
]
