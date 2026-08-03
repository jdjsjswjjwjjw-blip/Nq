"""مُحاكي المزاد (Auction Market Simulator).

يستند إلى نظرية المزاد ومنطقة القيمة لوصف حالة السوق.

**إطاران زمنيان (افتراضي):**
* رينج / Volume Profile: ``5`` دقائق — حدود VP الثلاثة + نظام توازن/اختلال.
* فعل / دخول: ``30`` ثانية — ارتدادات من سوق متوازن، وكسر+دخول من سوق مختلّ.

Volume Profile يُعرَّف دائمًا بـ **ثلاث حدود**:
* ``upper`` = VAH (حد علوي لمنطقة القيمة)
* ``mid`` = POC (حد متوسط / مركز القبول)
* ``lower`` = VAL (حد سفلي لمنطقة القيمة)

كل الحالات سببية: كل صف يعتمد على نوافذه المكتملة فقط، ومتاح عند
``bucket_end`` (join_asof خلفي من 5د → 30ث).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.time import sort_causal
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades
from nq.simulation.order_flow import order_flow_summary
from nq.simulation.volume_profile import developing_value_area

_NS: int = 1_000_000_000
#: رينج VP / قبول القيمة — 5 دقائق.
VP_PROFILE_INTERVAL_NS: int = 5 * 60 * _NS
#: ساعة الفعل: ارتداد / كسر / دخول — 30 ثانية.
VP_SIGNAL_INTERVAL_NS: int = 30 * _NS

_DEFAULT_BALANCE_THRESHOLD = 0.6
_DEFAULT_EXPANSION_THRESHOLD = 1.5
#: نافذة ريتست بعد الكسر بعدد براميل الإشارة (30ث → برميل واحد ≈ 30ث).
_DEFAULT_RETEST_WINDOW = 2
#: نافذة حجم للتسارع السببي على ساعة الإشارة.
_DEFAULT_ACCEL_LOOKBACK = 3
_DEFAULT_ACCEL_MULT = 1.5
#: قرب الريتست/الارتداد من المركز (mid/POC) كنسبة من عرض منطقة القيمة.
_DEFAULT_RETEST_MID_FRAC = 0.25
#: قرب الحد للكشف عن امتصاص / Look-above-and-fail.
_DEFAULT_BOUND_TOUCH_FRAC = 0.2

_FSM_SIGNAL_COLUMNS = (
    "vp_fsm_break",
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
    لرينج الاستراتيجية استدعِ بـ ``VP_PROFILE_INTERVAL_NS`` (5د).
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
            pl.lit(0, dtype=pl.Int64).alias("excess_upper"),
            pl.lit(0, dtype=pl.Int64).alias("excess_lower"),
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
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """براميل فعل ``30ث`` مع حدود/نظام رينج ``5د`` + تأكيد تدفق أوامر.

    * الرينج (VAH/POC/VAL + Excess + توازن) من ``profile_interval_ns``.
    * الإغلاق/الدلتا/امتصاص/Look-fail من ``signal_interval_ns``.
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
            f"profile={profile_interval_ns // _NS}s · signal={signal_interval_ns // _NS}s"
        )

    profile = auction_states(
        frame,
        interval_ns=profile_interval_ns,
        fraction=fraction,
        balance_threshold=balance_threshold,
        expansion_threshold=expansion_threshold,
        progress=progress,
    )
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
        (pl.col("low").cast(pl.Float64) - pl.col("val").cast(pl.Float64)).abs()
        <= touch * va_w
    )
    near_upper = (
        (pl.col("high").cast(pl.Float64) - pl.col("vah").cast(pl.Float64)).abs()
        <= touch * va_w
    )
    # امتصاص شرائي عند VAL: بيع عدواني كثيف بلا كسر الإغلاق تحت القيمة.
    absorb_buy = near_lower & (pl.col("sell_volume") > pl.col("buy_volume")) & closed_in_value
    # امتصاص بيعي عند VAH: شراء عدواني كثيف بلا إغلاق فوق القيمة.
    absorb_sell = near_upper & (pl.col("buy_volume") > pl.col("sell_volume")) & closed_in_value
    # Look above/below and fail (ورقة VP): اختراق المدى ثم إغلاق داخل + ضغط معاكس.
    look_fail_up = (
        (pl.col("high") > pl.col("vah")) & closed_in_value & (pl.col("delta") < 0)
    )
    look_fail_dn = (
        (pl.col("low") < pl.col("val")) & closed_in_value & (pl.col("delta") > 0)
    )

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


def auction_fsm_columns(  # noqa: PLR0912, PLR0915
    states: pl.DataFrame,
    *,
    retest_window: int = _DEFAULT_RETEST_WINDOW,
    accel_lookback: int = _DEFAULT_ACCEL_LOOKBACK,
    accel_mult: float = _DEFAULT_ACCEL_MULT,
    retest_mid_frac: float = _DEFAULT_RETEST_MID_FRAC,
) -> pl.DataFrame:
    """آلة حالات على براميل الفعل (30ث) بحدود الرينج (5د) + تأكيد تدفق.

    * متوازن: ارتداد/امتصاص/Look-fail نحو mid أو الحدود → ``vp_fsm_retest``.
    * مختلّ: كسر مؤكَّد بالدلتا → تسارع → ريتست mid → دخول مع الدلتا.
    """
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
        va_w = max(float(vah[i] - val[i]), 1.0)
        near_mid = abs(float(close[i] - poc[i])) <= retest_mid_frac * va_w

        # --- متوازن: ارتدادات 30ث (دفاع هيكل + امتصاص تدفق) ---
        if bool(balanced[i]):
            pending_dir = 0.0
            pending_i = -1
            saw_accel = False
            saw_retest = False
            if absorb[i] != 0.0:
                retest[i] = float(absorb[i])
            elif look_fail[i] != 0.0:
                retest[i] = float(look_fail[i])
            elif bool(pullback[i]) and near_mid:
                retest[i] = 1.0 if close[i] >= poc[i] else -1.0
            continue

        # --- مختلّ: كسر + دخول على 30ث (تأكيد دلتا عدوانية) ---
        prev_bal = bool(balanced[i - 1]) if i > 0 else False
        if prev_bal or pending_dir == 0.0:
            if close[i] > vah[i] and delta[i] >= 0:
                brk[i] = 1.0
            elif close[i] < val[i] and delta[i] <= 0:
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
            pending_dir = 0.0
            pending_i = -1
            continue

        if age >= 1 and not saw_accel:
            start = max(0, i - accel_lookback)
            hist = vol[start:i]
            base = float(np.mean(hist)) if hist.size else 0.0
            # تسارع حجم أو دلتا موافقة للاتجاه
            delta_ok = (pending_dir > 0 and delta[i] > 0) or (pending_dir < 0 and delta[i] < 0)
            if (base > 0 and vol[i] >= accel_mult * base) or delta_ok:
                accel[i] = pending_dir
                saw_accel = True

        if (
            age >= 1
            and saw_accel
            and not saw_retest
            and near_mid
            and (bool(pullback[i]) or bool(in_value[i]) or absorb[i] != 0.0)
        ):
            retest[i] = pending_dir
            saw_retest = True

        if saw_retest and (bool(expansion[i]) or age >= 1):
            delta_ok = (pending_dir > 0 and delta[i] >= 0) or (
                pending_dir < 0 and delta[i] <= 0
            )
            long_ok = pending_dir > 0 and close[i] >= vah[i] and delta_ok
            short_ok = pending_dir < 0 and close[i] <= val[i] and delta_ok
            if long_ok or short_ok:
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
    interval_ns: int | None = None,
    profile_interval_ns: int = VP_PROFILE_INTERVAL_NS,
    signal_interval_ns: int | None = None,
    fraction: float = 0.7,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
    expansion_threshold: float = _DEFAULT_EXPANSION_THRESHOLD,
    retest_window: int = _DEFAULT_RETEST_WINDOW,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """إشارات بحثية: رينج 5د + فعل 30ث (ارتداد متوازن / كسر+دخول مختلّ).

    ``interval_ns`` مرادف قديم لـ ``signal_interval_ns`` (ساعة البحث/الفعل).
    """
    sig_iv = int(signal_interval_ns if signal_interval_ns is not None else (
        interval_ns if interval_ns is not None else VP_SIGNAL_INTERVAL_NS
    ))
    prof_iv = int(profile_interval_ns)
    if progress is not None:
        progress.op(
            "auction_signal_frame: "
            f"رينج={prof_iv // _NS}s · فعل={sig_iv // _NS}s"
        )
    states = auction_action_states(
        frame,
        profile_interval_ns=prof_iv,
        signal_interval_ns=sig_iv,
        fraction=fraction,
        balance_threshold=balance_threshold,
        expansion_threshold=expansion_threshold,
        progress=progress,
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
        **{c: pl.Float64() for c in _FSM_SIGNAL_COLUMNS},
    }
    if states.height == 0:
        return pl.DataFrame(schema=base_schema)

    scale = float(PRICE_SCALE)
    classic = (
        states.sort(BUCKET_START)
        .with_columns(
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
        )
    )
    fsm = auction_fsm_columns(states.sort(BUCKET_START), retest_window=retest_window)
    return classic.hstack(fsm)


__all__ = [
    "VP_PROFILE_INTERVAL_NS",
    "VP_SIGNAL_INTERVAL_NS",
    "auction_action_states",
    "auction_fsm_columns",
    "auction_signal_frame",
    "auction_states",
]
