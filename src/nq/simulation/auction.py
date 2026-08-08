"""مُحاكي المزاد (Auction Market Simulator).

يستند إلى نظرية المزاد ومنطقة القيمة لوصف حالة السوق.

**إطاران زمنيان (افتراضي):**
* رينج / Volume Profile: ``5`` دقائق — حدود VP الثلاثة + نظام توازن/اختلال.
* فعل / دخول: ``30`` ثانية — ارتدادات من سوق متوازن، وكسر+بناء+انطلاق من سوق مختلّ.

Volume Profile يُعرَّف دائمًا بـ **ثلاث حدود**:
* ``upper`` = VAH (حد علوي لمنطقة القيمة)
* ``mid`` = POC (حد متوسط / مركز القبول)
* ``lower`` = VAL (حد سفلي لمنطقة القيمة)

بعد كسر التوازن لا يُعلَن التوسّع من أول وخزة: يُراقَب طور بناء حول مرجع
الفوليوم (خط أو خطين أو حول المنتصف). الرجوع داخل القيمة لا يقتل السياق.
الانطلاق عند اتساع صريح خارج الحد؛ وإعادة التوازن الحقيقية تقتل السياق فقط
بعد عدة براميل متتالية متوازنة.

ملف الحجم تراكمي **داخل جلسة السيولة** (آسيا / لندن / نيويورك بتوقيت ET)
ويُصفَّر عند انتقال الجلسة حتى لا يُخلط قبول جلسة بأخرى.

كل الحالات سببية: كل صف يعتمد على نوافذه المكتملة فقط، ومتاح عند
``bucket_end`` (join_asof خلفي من 5د → 30ث).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.session import VP_LIQUIDITY_SESSION
from nq.core.time import sort_causal
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades
from nq.simulation.order_flow import order_flow_summary
from nq.simulation.volume_profile import developing_value_area
from nq.simulation.vp_fixed_range import (
    VP_FIXED_RANGE_COLUMNS,
    VpFixedRangeConfig,
    attach_vp_fixed_range,
)

_NS: int = 1_000_000_000
#: رينج VP / قبول القيمة — 5 دقائق.
VP_PROFILE_INTERVAL_NS: int = 5 * 60 * _NS
#: ساعة الفعل: ارتداد / كسر / دخول — 30 ثانية.
VP_SIGNAL_INTERVAL_NS: int = 30 * _NS

_DEFAULT_BALANCE_THRESHOLD = 0.6
_DEFAULT_EXPANSION_THRESHOLD = 1.5
#: نافذة ريتست الكلاسيكي بعد الكسر (براميل إشارة 30ث) — لا تُستخدم لقتل البناء.
_DEFAULT_RETEST_WINDOW = 8
#: أقصى عمر لسياق ما بعد الكسر قبل إعادة الضبط (≈ 30ث×48 ≈ 24د).
_DEFAULT_BUILD_MAX_AGE = 48
#: عدد براميل ``is_balanced`` المتتالية لاعتبار إعادة توازن حقيقية (قتل السياق).
_DEFAULT_REBALANCE_CONFIRM = 3
#: نافذة حجم للتسارع السببي على ساعة الإشارة.
_DEFAULT_ACCEL_LOOKBACK = 3
_DEFAULT_ACCEL_MULT = 1.5
#: قرب الريتست/الارتداد من المركز (mid/POC) كنسبة من عرض منطقة القيمة.
_DEFAULT_RETEST_MID_FRAC = 0.25
#: قرب أي مرجع فوليوم (VAL/POC/VAH) لاعتبار «لعب حول الهيكل» — بلا افتراض عدد خطوط.
_DEFAULT_BUILD_ANCHOR_FRAC = 0.35
#: قرب الحد للكشف عن امتصاص / Look-above-and-fail.
_DEFAULT_BOUND_TOUCH_FRAC = 0.2

_FSM_SIGNAL_COLUMNS = (
    "vp_fsm_break",
    "vp_fsm_build",
    "vp_fsm_accel",
    "vp_fsm_retest",
    "vp_fsm_expand",
    "vp_auction_setup",
)

#: ثلاث حدود VP الصريحة (علوي / متوسط / سفلي) بالدولار.
_VP_BOUND_COLUMNS = (
    "vp_upper",
    "vp_mid",
    "vp_lower",
)
#: مسافات نسبية عن الحدود الثلاثة: (close − bound) / max(VAH−VAL, ε).
_VP_REL_BOUND_COLUMNS = (
    "vp_rel_upper",
    "vp_rel_mid",
    "vp_rel_lower",
)
#: Excess من رينج 5د + تأكيد تدفق أوامر على 30ث (من الورقتين).
_VP_STRUCTURE_OF_COLUMNS = (
    "vp_excess_upper",
    "vp_excess_lower",
    "vp_of_delta",
    "vp_absorb",
    "vp_look_fail",
)

_PROFILE_JOIN_COLS = (
    "poc",
    "vah",
    "val",
    "poc_migration",
    "is_balanced",
    "is_expansion",
    "in_value_fraction",
    "excess_upper",
    "excess_lower",
    VP_LIQUIDITY_SESSION,
    *VP_FIXED_RANGE_COLUMNS,
)


def auction_states(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    fraction: float = 0.7,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
    expansion_threshold: float = _DEFAULT_EXPANSION_THRESHOLD,
    reset_by_liquidity_session: bool = True,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يصنّف حالة المزاد لكل نافذة زمنية (متاح عند ``bucket_end``).

    يستخدم منطقة قيمة **تراكمية داخل جلسة السيولة** (آسيا/لندن/نيويورك)
    مع تصفير عند انتقال الجلسة — حتى لا يُخلط قبول جلسة بأخرى.
    لرينج الاستراتيجية استدعِ بـ ``VP_PROFILE_INTERVAL_NS`` (5د).
    """
    if progress is not None:
        progress.op(
            "auction_states: developing_value_area · "
            f"interval_ns={interval_ns} · reset_session={reset_by_liquidity_session}"
        )
    dva = developing_value_area(
        frame,
        interval_ns=interval_ns,
        fraction=fraction,
        cumulative=True,
        reset_by_liquidity_session=reset_by_liquidity_session,
        progress=progress,
    )
    if dva.height == 0:
        return dva.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("high"),
            pl.lit(None, dtype=pl.Int64).alias("low"),
            pl.lit(None, dtype=pl.Int64).alias("close"),
            pl.lit(0, dtype=pl.Int64).alias("excess_upper"),
            pl.lit(0, dtype=pl.Int64).alias("excess_lower"),
        )

    trades = extract_trades(add_time_bucket(sort_causal(frame), interval_ns=interval_ns))

    stats = (
        trades.sort(EVENT_TS)
        .group_by(BUCKET_START, maintain_order=True)
        .agg(
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").cast(pl.Int64).sum().alias("bucket_volume"),
        )
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
        # Excess: تطرف فوق VAH / تحت VAL (ورقة VP)
        (pl.col("high") - pl.col("vah")).clip(lower_bound=0).alias("excess_upper"),
        (pl.col("val") - pl.col("low")).clip(lower_bound=0).alias("excess_lower"),
    ).with_columns(
        (
            pl.col("close_in_value")
            & ~pl.col("is_expansion")
            & (pl.col("in_value_fraction") >= balance_threshold)
        ).alias("is_balanced"),
        ((pl.col("made_new_high") | pl.col("made_new_low")) & pl.col("close_in_value")).alias(
            "pullback_defended"
        ),
    )


def auction_action_states(
    frame: pl.DataFrame,
    *,
    profile_interval_ns: int = VP_PROFILE_INTERVAL_NS,
    signal_interval_ns: int = VP_SIGNAL_INTERVAL_NS,
    fraction: float = 0.7,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
    expansion_threshold: float = _DEFAULT_EXPANSION_THRESHOLD,
    bound_touch_frac: float = _DEFAULT_BOUND_TOUCH_FRAC,
    reset_by_liquidity_session: bool = True,
    fixed_range: bool = True,
    fixed_range_config: VpFixedRangeConfig | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """براميل فعل ``30ث`` مع حدود/نظام رينج ``5د`` + تأكيد تدفق أوامر.

    * الرينج (VAH/POC/VAL + Excess + توازن) من ``profile_interval_ns``.
    * الإغلاق/الدلتا/امتصاص/Look-fail من ``signal_interval_ns``.
    * افتراضيًا: تصفير الفوليوم عند انتقال آسيا/لندن/نيويورك.
    * افتراضيًا: رينج ثابت من expansion مقبول + عرضي (قرار عند الخروج فقط).
    """
    if signal_interval_ns < 1 or profile_interval_ns < 1:
        raise ValueError("profile_interval_ns and signal_interval_ns must be >= 1")
    if profile_interval_ns < signal_interval_ns:
        raise ValueError(
            f"profile_interval_ns ({profile_interval_ns}) must be >= "
            f"signal_interval_ns ({signal_interval_ns})"
        )

    if progress is not None:
        progress.op(
            "auction_action_states: "
            f"profile={profile_interval_ns // _NS}s · signal={signal_interval_ns // _NS}s · "
            f"reset_session={reset_by_liquidity_session} · fixed_range={fixed_range}"
        )

    profile = auction_states(
        frame,
        interval_ns=profile_interval_ns,
        fraction=fraction,
        balance_threshold=balance_threshold,
        expansion_threshold=expansion_threshold,
        reset_by_liquidity_session=reset_by_liquidity_session,
        progress=progress,
    )
    if profile.height and fixed_range:
        profile = attach_vp_fixed_range(
            profile,
            frame,
            interval_ns=profile_interval_ns,
            config=fixed_range_config,
            progress=progress,
        )
    elif profile.height:
        # أعمدة فارغة للحفاظ على مخطط الانضمام
        zeros = pl.DataFrame(
            {
                BUCKET_START: profile[BUCKET_START],
                **{
                    c: pl.Series(c, [0.0] * profile.height, dtype=pl.Float64)
                    for c in VP_FIXED_RANGE_COLUMNS
                    if c
                    not in (
                        "vp_fr_start_ts",
                        "vp_fr_end_ts",
                        "vp_fr_upper",
                        "vp_fr_mid",
                        "vp_fr_lower",
                    )
                },
                "vp_fr_upper": pl.Series("vp_fr_upper", [None] * profile.height, dtype=pl.Float64),
                "vp_fr_mid": pl.Series("vp_fr_mid", [None] * profile.height, dtype=pl.Float64),
                "vp_fr_lower": pl.Series("vp_fr_lower", [None] * profile.height, dtype=pl.Float64),
                "vp_fr_start_ts": pl.Series("vp_fr_start_ts", [0] * profile.height, dtype=pl.Int64),
                "vp_fr_end_ts": pl.Series("vp_fr_end_ts", [0] * profile.height, dtype=pl.Int64),
            }
        )
        profile = profile.join(zeros, on=BUCKET_START, how="left")
    empty = pl.DataFrame(
        schema={
            AVAILABILITY_TS: pl.Int64(),
            BUCKET_START: pl.Int64(),
            BUCKET_END: pl.Int64(),
            "high": pl.Int64(),
            "low": pl.Int64(),
            "close": pl.Int64(),
            "bucket_volume": pl.Int64(),
            "buy_volume": pl.Int64(),
            "sell_volume": pl.Int64(),
            "delta": pl.Int64(),
            "poc": pl.Int64(),
            "vah": pl.Int64(),
            "val": pl.Int64(),
            "poc_migration": pl.Int64(),
            "excess_upper": pl.Int64(),
            "excess_lower": pl.Int64(),
            "is_balanced": pl.Boolean(),
            "is_expansion": pl.Boolean(),
            "in_value_fraction": pl.Float64(),
            "close_in_value": pl.Boolean(),
            "made_new_high": pl.Boolean(),
            "made_new_low": pl.Boolean(),
            "pullback_defended": pl.Boolean(),
            "absorb": pl.Float64(),
            "look_fail": pl.Float64(),
            "range": pl.Int64(),
            "expansion_ratio": pl.Float64(),
            VP_LIQUIDITY_SESSION: pl.Int8(),
            "vp_fr_active": pl.Float64(),
            "vp_fr_accepted_expansion": pl.Float64(),
            "vp_fr_in_balance": pl.Float64(),
            "vp_fr_exit": pl.Float64(),
            "vp_fr_upper": pl.Float64(),
            "vp_fr_mid": pl.Float64(),
            "vp_fr_lower": pl.Float64(),
            "vp_fr_start_ts": pl.Int64(),
            "vp_fr_end_ts": pl.Int64(),
        }
    )
    if profile.height == 0:
        return empty

    trades = extract_trades(add_time_bucket(sort_causal(frame), interval_ns=signal_interval_ns))
    if trades.height == 0:
        return empty

    of = order_flow_summary(frame, interval_ns=signal_interval_ns).select(
        BUCKET_START,
        "buy_volume",
        "sell_volume",
        "delta",
    )
    signal = (
        trades.sort(EVENT_TS)
        .group_by(BUCKET_START, maintain_order=True)
        .agg(
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").cast(pl.Int64).sum().alias("bucket_volume"),
            pl.col(BUCKET_END).first().alias(BUCKET_END),
            pl.col(AVAILABILITY_TS).first().alias(AVAILABILITY_TS),
        )
        .join(of, on=BUCKET_START, how="left")
        .with_columns(
            pl.col("buy_volume").fill_null(0),
            pl.col("sell_volume").fill_null(0),
            pl.col("delta").fill_null(0),
        )
        .sort(AVAILABILITY_TS)
    )

    profile_right = profile.select(AVAILABILITY_TS, *_PROFILE_JOIN_COLS).sort(AVAILABILITY_TS)
    merged = signal.join_asof(profile_right, on=AVAILABILITY_TS, strategy="backward")
    merged = merged.filter(pl.col("vah").is_not_null() & pl.col("val").is_not_null())
    if merged.height == 0:
        return empty

    prev_high = pl.col("high").shift(1)
    prev_low = pl.col("low").shift(1)
    price_range = pl.col("high") - pl.col("low")
    prev_range = price_range.shift(1)
    closed_in_value = (pl.col("close") >= pl.col("val")) & (pl.col("close") <= pl.col("vah"))
    made_new_high = (prev_high.is_not_null()) & (pl.col("high") > prev_high)
    made_new_low = (prev_low.is_not_null()) & (pl.col("low") < prev_low)
    expansion_ratio = (
        pl.when((prev_range.is_not_null()) & (prev_range > 0))
        .then(price_range.cast(pl.Float64) / prev_range.cast(pl.Float64))
        .otherwise(None)
    )
    va_w = pl.max_horizontal(
        pl.col("vah").cast(pl.Float64) - pl.col("val").cast(pl.Float64),
        pl.lit(1.0),
    )
    touch = float(bound_touch_frac)
    near_lower = (
        pl.col("low").cast(pl.Float64) - pl.col("val").cast(pl.Float64)
    ).abs() <= touch * va_w
    near_upper = (
        pl.col("high").cast(pl.Float64) - pl.col("vah").cast(pl.Float64)
    ).abs() <= touch * va_w
    # امتصاص شرائي عند VAL: بيع عدواني كثيف بلا كسر الإغلاق تحت القيمة.
    absorb_buy = near_lower & (pl.col("sell_volume") > pl.col("buy_volume")) & closed_in_value
    # امتصاص بيعي عند VAH: شراء عدواني كثيف بلا إغلاق فوق القيمة.
    absorb_sell = near_upper & (pl.col("buy_volume") > pl.col("sell_volume")) & closed_in_value
    # Look above/below and fail (ورقة VP): اختراق المدى ثم إغلاق داخل + ضغط معاكس.
    look_fail_up = (pl.col("high") > pl.col("vah")) & closed_in_value & (pl.col("delta") < 0)
    look_fail_dn = (pl.col("low") < pl.col("val")) & closed_in_value & (pl.col("delta") > 0)

    return (
        merged.sort(BUCKET_START)
        .with_columns(
            price_range.alias("range"),
            expansion_ratio.alias("expansion_ratio"),
            made_new_high.alias("made_new_high"),
            made_new_low.alias("made_new_low"),
            closed_in_value.alias("close_in_value"),
            pl.col("is_balanced").fill_null(value=False),
            pl.col("is_expansion").fill_null(value=False),
            pl.col("in_value_fraction").fill_null(0.0),
            pl.col("poc_migration").fill_null(0),
            pl.col("excess_upper").fill_null(0),
            pl.col("excess_lower").fill_null(0),
            pl.col(VP_LIQUIDITY_SESSION).fill_null(0).cast(pl.Int8),
            pl.col("vp_fr_active").fill_null(0.0),
            pl.col("vp_fr_accepted_expansion").fill_null(0.0),
            pl.col("vp_fr_in_balance").fill_null(0.0),
            pl.col("vp_fr_exit").fill_null(0.0),
            pl.col("vp_fr_start_ts").fill_null(0),
            pl.col("vp_fr_end_ts").fill_null(0),
            pl.when(absorb_buy)
            .then(1.0)
            .when(absorb_sell)
            .then(-1.0)
            .otherwise(0.0)
            .alias("absorb"),
            pl.when(look_fail_up)
            .then(-1.0)
            .when(look_fail_dn)
            .then(1.0)
            .otherwise(0.0)
            .alias("look_fail"),
        )
        .with_columns(
            (
                pl.col("is_balanced")
                & (
                    ((pl.col("made_new_high") | pl.col("made_new_low")) & pl.col("close_in_value"))
                    | (pl.col("absorb") != 0.0)
                    | (pl.col("look_fail") != 0.0)
                )
            ).alias("pullback_defended"),
        )
    )


def _near_volume_anchor(
    close: float,
    vah: float,
    poc: float,
    val: float,
    *,
    anchor_frac: float,
) -> bool:
    """قرب مرن لأي مرجع فوليوم (خط واحد أو بين خطين) — بلا فرع على عدد الخطوط."""
    va_w = max(vah - val, 1.0)
    if val <= close <= vah:
        return True
    dist = min(abs(close - poc), abs(close - vah), abs(close - val))
    return dist <= anchor_frac * va_w


def auction_fsm_columns(  # noqa: PLR0912, PLR0915
    states: pl.DataFrame,
    *,
    retest_window: int = _DEFAULT_RETEST_WINDOW,
    build_max_age: int = _DEFAULT_BUILD_MAX_AGE,
    rebalance_confirm: int = _DEFAULT_REBALANCE_CONFIRM,
    accel_lookback: int = _DEFAULT_ACCEL_LOOKBACK,
    accel_mult: float = _DEFAULT_ACCEL_MULT,
    retest_mid_frac: float = _DEFAULT_RETEST_MID_FRAC,
    build_anchor_frac: float = _DEFAULT_BUILD_ANCHOR_FRAC,
) -> pl.DataFrame:
    """آلة حالات على براميل الفعل (30ث) بحدود الرينج (5د) + تأكيد تدفق.

    الأطوار بعد كسر التوازن:

    1. ``break`` — أول خروج مؤكَّد بالدلتا (مراقبة فقط، ليس توسّعًا).
    2. ``build`` — لعب/تراكم حول مرجع فوليوم (POC أو حدود أو بينهما)؛
       الرجوع داخل القيمة **لا** يقتل السياق ولا يعني فشل التوسّع.
    3. ``expand`` / ``setup`` — انطلاق صريح: اتساع + قبول خارج الحد باتجاه الكسر.
    4. إعادة توازن حقيقية = ``rebalance_confirm`` براميل متتالية ``is_balanced``.
    """
    if rebalance_confirm < 1:
        raise ValueError(f"rebalance_confirm must be >= 1, got {rebalance_confirm}")
    if build_max_age < 1:
        raise ValueError(f"build_max_age must be >= 1, got {build_max_age}")

    n = states.height
    empty = {c: pl.Series(c, [0.0] * n if n else [], dtype=pl.Float64) for c in _FSM_SIGNAL_COLUMNS}
    if n == 0:
        return pl.DataFrame(empty)

    balanced = states["is_balanced"].to_numpy()
    close = states["close"].to_numpy().astype(np.float64)
    vah = states["vah"].to_numpy().astype(np.float64)
    poc = states["poc"].to_numpy().astype(np.float64)
    val = states["val"].to_numpy().astype(np.float64)
    vol = states["bucket_volume"].to_numpy().astype(np.float64)
    expansion = states["is_expansion"].to_numpy()
    pullback = states["pullback_defended"].to_numpy()
    in_value = states["close_in_value"].to_numpy()
    delta = (
        states["delta"].to_numpy().astype(np.float64)
        if "delta" in states.columns
        else np.zeros(n, dtype=np.float64)
    )
    absorb = (
        states["absorb"].to_numpy().astype(np.float64)
        if "absorb" in states.columns
        else np.zeros(n, dtype=np.float64)
    )
    look_fail = (
        states["look_fail"].to_numpy().astype(np.float64)
        if "look_fail" in states.columns
        else np.zeros(n, dtype=np.float64)
    )
    sessions = (
        states[VP_LIQUIDITY_SESSION].to_numpy().astype(np.int64)
        if VP_LIQUIDITY_SESSION in states.columns
        else np.zeros(n, dtype=np.int64)
    )
    fr_active = (
        states["vp_fr_active"].to_numpy().astype(np.float64)
        if "vp_fr_active" in states.columns
        else np.zeros(n, dtype=np.float64)
    )
    fr_upper = (
        states["vp_fr_upper"].to_numpy().astype(np.float64)
        if "vp_fr_upper" in states.columns
        else np.full(n, np.nan)
    )
    fr_mid = (
        states["vp_fr_mid"].to_numpy().astype(np.float64)
        if "vp_fr_mid" in states.columns
        else np.full(n, np.nan)
    )
    fr_lower = (
        states["vp_fr_lower"].to_numpy().astype(np.float64)
        if "vp_fr_lower" in states.columns
        else np.full(n, np.nan)
    )
    fr_exit = (
        states["vp_fr_exit"].to_numpy().astype(np.float64)
        if "vp_fr_exit" in states.columns
        else np.zeros(n, dtype=np.float64)
    )
    fr_in_bal = (
        states["vp_fr_in_balance"].to_numpy().astype(np.float64)
        if "vp_fr_in_balance" in states.columns
        else np.zeros(n, dtype=np.float64)
    )

    brk = np.zeros(n, dtype=np.float64)
    build = np.zeros(n, dtype=np.float64)
    accel = np.zeros(n, dtype=np.float64)
    retest = np.zeros(n, dtype=np.float64)
    expand = np.zeros(n, dtype=np.float64)
    setup = np.zeros(n, dtype=np.float64)

    pending_dir = 0.0
    pending_i = -1
    saw_accel = False
    saw_retest = False
    saw_build = False
    balance_streak = 0

    def _reset_pending() -> None:
        nonlocal pending_dir, pending_i, saw_accel, saw_retest, saw_build, balance_streak
        pending_dir = 0.0
        pending_i = -1
        saw_accel = False
        saw_retest = False
        saw_build = False
        balance_streak = 0

    for i in range(n):
        # انتقال جلسة السيولة يقتل أي سياق كسر/بناء سابق (حدود جديدة).
        if i > 0 and int(sessions[i]) != int(sessions[i - 1]) and pending_dir != 0.0:
            _reset_pending()

        # حدود القرار: الرينج الثابت النشط إن وُجد، وإلا حدود الجلسة التراكمية.
        use_fr = float(fr_active[i]) > 0.0 and np.isfinite(fr_upper[i]) and np.isfinite(fr_lower[i])
        bound_hi = float(fr_upper[i]) if use_fr else float(vah[i])
        bound_mid = float(fr_mid[i]) if use_fr and np.isfinite(fr_mid[i]) else float(poc[i])
        bound_lo = float(fr_lower[i]) if use_fr else float(val[i])

        va_w = max(bound_hi - bound_lo, 1.0)
        near_mid = abs(float(close[i] - bound_mid)) <= retest_mid_frac * va_w
        near_anchor = _near_volume_anchor(
            float(close[i]),
            bound_hi,
            bound_mid,
            bound_lo,
            anchor_frac=build_anchor_frac,
        )

        # خروج صريح من الرينج الثابت = setup مرة واحدة (حافة صاعدة؛ قرار نادر)
        fr_exit_now = float(fr_exit[i])
        fr_exit_prev = float(fr_exit[i - 1]) if i > 0 else 0.0
        if fr_exit_now != 0.0 and fr_exit_prev == 0.0:
            expand[i] = fr_exit_now
            setup[i] = fr_exit_now
            brk[i] = fr_exit_now
            _reset_pending()
            continue

        # أثناء الرينج الثابت: لا مسار كسر→توسّع كلاسيكي — بناء داخل التوازن فقط.
        if use_fr:
            if float(fr_in_bal[i]) > 0.0 or near_anchor or bool(in_value[i]):
                build[i] = 1.0
            if bool(balanced[i]):
                if absorb[i] != 0.0:
                    retest[i] = float(absorb[i])
                elif look_fail[i] != 0.0:
                    retest[i] = float(look_fail[i])
                elif bool(pullback[i]) and near_mid:
                    retest[i] = 1.0 if close[i] >= bound_mid else -1.0
            _reset_pending()
            continue

        # --- بلا سياق كسر: ارتدادات سوق متوازن فقط ---
        if pending_dir == 0.0:
            balance_streak = 0
            if bool(balanced[i]):
                if absorb[i] != 0.0:
                    retest[i] = float(absorb[i])
                elif look_fail[i] != 0.0:
                    retest[i] = float(look_fail[i])
                elif bool(pullback[i]) and near_mid:
                    retest[i] = 1.0 if close[i] >= bound_mid else -1.0
            else:
                # مختلّ بلا سياق حيّ: أول كسر مؤكَّد بالدلتا يبدأ المراقبة فقط.
                if close[i] > bound_hi and delta[i] >= 0:
                    brk[i] = 1.0
                elif close[i] < bound_lo and delta[i] <= 0:
                    brk[i] = -1.0
                if brk[i] != 0.0:
                    pending_dir = float(brk[i])
                    pending_i = i
                    saw_accel = False
                    saw_retest = False
                    saw_build = False
                    balance_streak = 0
            continue

        # --- سياق كسر حيّ: بناء → (تسارع/ريتست اختياري) → توسّع صريح ---
        age = i - pending_i
        if age > build_max_age:
            _reset_pending()
            continue

        if bool(balanced[i]):
            balance_streak += 1
            # رجوع للتوازن اللحظي = استمرار بناء، ليس فشل توسّع.
            build[i] = pending_dir
            saw_build = True
            if balance_streak >= rebalance_confirm:
                _reset_pending()
            continue
        balance_streak = 0

        # لعب حول مرجع الفوليوم (خط/خطين/حول المنتصف) قبل الانطلاق.
        if near_anchor or bool(in_value[i]):
            build[i] = pending_dir
            saw_build = True

        if age >= 1 and not saw_accel:
            start = max(0, i - accel_lookback)
            hist = vol[start:i]
            base = float(np.mean(hist)) if hist.size else 0.0
            delta_ok = (pending_dir > 0 and delta[i] > 0) or (pending_dir < 0 and delta[i] < 0)
            if (base > 0 and vol[i] >= accel_mult * base) or delta_ok:
                accel[i] = pending_dir
                saw_accel = True

        if (
            age >= 1
            and age <= retest_window
            and saw_accel
            and not saw_retest
            and near_mid
            and (bool(pullback[i]) or bool(in_value[i]) or absorb[i] != 0.0)
        ):
            retest[i] = pending_dir
            saw_retest = True

        # توسّع فقط عند اتساع حقيقي + قبول خارج الحد — ليس أول وخزة ولا age وحدها.
        delta_ok = (pending_dir > 0 and delta[i] >= 0) or (pending_dir < 0 and delta[i] <= 0)
        long_ok = pending_dir > 0 and close[i] > bound_hi and delta_ok
        short_ok = pending_dir < 0 and close[i] < bound_lo and delta_ok
        ready = (saw_retest or saw_build) and age >= 1 and bool(expansion[i])
        if ready and (long_ok or short_ok):
            expand[i] = pending_dir
            setup[i] = pending_dir
            _reset_pending()

    return pl.DataFrame(
        {
            "vp_fsm_break": brk,
            "vp_fsm_build": build,
            "vp_fsm_accel": accel,
            "vp_fsm_retest": retest,
            "vp_fsm_expand": expand,
            "vp_auction_setup": setup,
        }
    )


def auction_signals_from_states(
    states: pl.DataFrame,
    *,
    retest_window: int = _DEFAULT_RETEST_WINDOW,
    build_max_age: int = _DEFAULT_BUILD_MAX_AGE,
    rebalance_confirm: int = _DEFAULT_REBALANCE_CONFIRM,
) -> pl.DataFrame:
    """يحوّل حالات المزاد المحسوبة مسبقًا إلى أعمدة VP/FSM دون إعادة مسح MBO."""
    fr_signal_cols = (
        "vp_fr_active",
        "vp_fr_accepted_expansion",
        "vp_fr_in_balance",
        "vp_fr_exit",
        "vp_fr_upper",
        "vp_fr_mid",
        "vp_fr_lower",
        "vp_fr_start_ts",
        "vp_fr_end_ts",
    )
    base_schema = {
        AVAILABILITY_TS: pl.Int64(),
        **{c: pl.Float64() for c in _VP_BOUND_COLUMNS},
        **{c: pl.Float64() for c in _VP_REL_BOUND_COLUMNS},
        **{c: pl.Float64() for c in _VP_STRUCTURE_OF_COLUMNS},
        "vp_balance": pl.Float64(),
        "vp_imbalance": pl.Float64(),
        "vp_expansion": pl.Float64(),
        "vp_close_in_value": pl.Float64(),
        "vp_in_value_frac": pl.Float64(),
        "vp_pullback_defense": pl.Float64(),
        "vp_poc_migration": pl.Float64(),
        "vp_flip_to_imbalance": pl.Float64(),
        "vp_liquidity_session": pl.Float64(),
        **{c: pl.Float64() for c in fr_signal_cols},
        **{c: pl.Float64() for c in _FSM_SIGNAL_COLUMNS},
    }
    if states.height == 0:
        return pl.DataFrame(schema=base_schema)

    ordered = states.sort(BUCKET_START)
    scale = float(PRICE_SCALE)
    has_fr = "vp_fr_active" in ordered.columns
    fr_exprs = (
        [
            pl.col("vp_fr_active").fill_null(0.0).cast(pl.Float64).alias("vp_fr_active"),
            pl.col("vp_fr_accepted_expansion")
            .fill_null(0.0)
            .cast(pl.Float64)
            .alias("vp_fr_accepted_expansion"),
            pl.col("vp_fr_in_balance").fill_null(0.0).cast(pl.Float64).alias("vp_fr_in_balance"),
            pl.col("vp_fr_exit").fill_null(0.0).cast(pl.Float64).alias("vp_fr_exit"),
            (pl.col("vp_fr_upper").cast(pl.Float64) * scale).alias("vp_fr_upper"),
            (pl.col("vp_fr_mid").cast(pl.Float64) * scale).alias("vp_fr_mid"),
            (pl.col("vp_fr_lower").cast(pl.Float64) * scale).alias("vp_fr_lower"),
            pl.col("vp_fr_start_ts").fill_null(0).cast(pl.Float64).alias("vp_fr_start_ts"),
            pl.col("vp_fr_end_ts").fill_null(0).cast(pl.Float64).alias("vp_fr_end_ts"),
        ]
        if has_fr
        else [pl.lit(0.0).alias(c) for c in fr_signal_cols]
    )
    classic = (
        ordered.with_columns(
            (pl.col("vah").cast(pl.Float64) * scale).alias("vp_upper"),
            (pl.col("poc").cast(pl.Float64) * scale).alias("vp_mid"),
            (pl.col("val").cast(pl.Float64) * scale).alias("vp_lower"),
            pl.max_horizontal(
                pl.col("vah").cast(pl.Float64) - pl.col("val").cast(pl.Float64),
                pl.lit(1.0),
            ).alias("_va_w"),
            pl.col("close").cast(pl.Float64).alias("_close"),
            pl.col("bucket_volume").cast(pl.Float64).alias("_bvol"),
        )
        .with_columns(
            ((pl.col("_close") - pl.col("vah")) / pl.col("_va_w")).alias("vp_rel_upper"),
            ((pl.col("_close") - pl.col("poc")) / pl.col("_va_w")).alias("vp_rel_mid"),
            ((pl.col("_close") - pl.col("val")) / pl.col("_va_w")).alias("vp_rel_lower"),
            (pl.col("excess_upper").cast(pl.Float64) / pl.col("_va_w")).alias("vp_excess_upper"),
            (pl.col("excess_lower").cast(pl.Float64) / pl.col("_va_w")).alias("vp_excess_lower"),
            pl.when(pl.col("_bvol") > 0)
            .then(pl.col("delta").cast(pl.Float64) / pl.col("_bvol"))
            .otherwise(0.0)
            .alias("vp_of_delta"),
            pl.col("absorb").cast(pl.Float64).alias("vp_absorb"),
            pl.col("look_fail").cast(pl.Float64).alias("vp_look_fail"),
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
            pl.col(VP_LIQUIDITY_SESSION).cast(pl.Float64).alias("vp_liquidity_session"),
            *fr_exprs,
        )
        .select(
            AVAILABILITY_TS,
            *_VP_BOUND_COLUMNS,
            *_VP_REL_BOUND_COLUMNS,
            *_VP_STRUCTURE_OF_COLUMNS,
            "vp_balance",
            "vp_imbalance",
            "vp_expansion",
            "vp_close_in_value",
            "vp_in_value_frac",
            "vp_pullback_defense",
            "vp_poc_migration",
            "vp_flip_to_imbalance",
            "vp_liquidity_session",
            *fr_signal_cols,
        )
    )
    fsm = auction_fsm_columns(
        ordered,
        retest_window=retest_window,
        build_max_age=build_max_age,
        rebalance_confirm=rebalance_confirm,
    )
    return classic.hstack(fsm)


def auction_signal_frame(
    frame: pl.DataFrame,
    *,
    interval_ns: int | None = None,
    profile_interval_ns: int = VP_PROFILE_INTERVAL_NS,
    signal_interval_ns: int | None = None,
    fraction: float = 0.7,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
    expansion_threshold: float = _DEFAULT_EXPANSION_THRESHOLD,
    retest_window: int = _DEFAULT_RETEST_WINDOW,
    build_max_age: int = _DEFAULT_BUILD_MAX_AGE,
    rebalance_confirm: int = _DEFAULT_REBALANCE_CONFIRM,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """إشارات بحثية: رينج 5د + فعل 30ث (ارتداد متوازن / كسر+بناء+انطلاق).

    ``interval_ns`` مرادف قديم لـ ``signal_interval_ns`` (ساعة البحث/الفعل).
    """
    sig_iv = int(
        signal_interval_ns
        if signal_interval_ns is not None
        else (interval_ns if interval_ns is not None else VP_SIGNAL_INTERVAL_NS)
    )
    prof_iv = int(profile_interval_ns)
    if progress is not None:
        progress.op(f"auction_signal_frame: رينج={prof_iv // _NS}s · فعل={sig_iv // _NS}s")
    states = auction_action_states(
        frame,
        profile_interval_ns=prof_iv,
        signal_interval_ns=sig_iv,
        fraction=fraction,
        balance_threshold=balance_threshold,
        expansion_threshold=expansion_threshold,
        progress=progress,
    )
    return auction_signals_from_states(
        states,
        retest_window=retest_window,
        build_max_age=build_max_age,
        rebalance_confirm=rebalance_confirm,
    )


__all__ = [
    "VP_FIXED_RANGE_COLUMNS",
    "VP_PROFILE_INTERVAL_NS",
    "VP_SIGNAL_INTERVAL_NS",
    "VpFixedRangeConfig",
    "attach_vp_fixed_range",
    "auction_action_states",
    "auction_fsm_columns",
    "auction_signal_frame",
    "auction_signals_from_states",
    "auction_states",
]
