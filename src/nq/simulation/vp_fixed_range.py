"""رينج فوليوم ثابت (Fixed Range) حسب خطة التوازن/الاختلال.

القواعد (سببية، داخل جلسة السيولة):

1. البداية = آخر **expansion مقبول** (اندفاع ثم قبول/عرضي).
2. يجمع الشموع العرضية بعده؛ النهاية تتحرك لقدام **داخل التوازن فقط**.
3. الرجوع جوّه التوازن بعد وخزة = بناء — لا تصفير ولا مدّ النهاية داخل الاختلال.
4. عند خروج صريح مقبول: إشارة خروج وتقفل الرينج؛ رينج جديد فقط من expansion مقبول لاحق.
5. انتقال جلسة السيولة يقفل الرينج دون قرار.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.core.session import VP_LIQUIDITY_SESSION
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades
from nq.simulation.volume_profile import DevelopingVolumeProfile

_DEFAULT_ACCEPT_WINDOW = 3
_DEFAULT_BALANCE_FRAC = 0.6
_DEFAULT_EXIT_EXPANSION = 1.5

VP_FIXED_RANGE_COLUMNS = (
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


@dataclass(frozen=True, slots=True)
class VpFixedRangeConfig:
    """إعدادات رينج الفوليوم الثابت."""

    accept_window: int = _DEFAULT_ACCEPT_WINDOW
    balance_frac: float = _DEFAULT_BALANCE_FRAC
    exit_expansion_ratio: float = _DEFAULT_EXIT_EXPANSION
    value_fraction: float = 0.7


def _empty_fr(n: int) -> dict[str, pl.Series]:
    z = [0.0] * n
    zi = [0] * n
    return {
        "vp_fr_active": pl.Series("vp_fr_active", z, dtype=pl.Float64),
        "vp_fr_accepted_expansion": pl.Series("vp_fr_accepted_expansion", z, dtype=pl.Float64),
        "vp_fr_in_balance": pl.Series("vp_fr_in_balance", z, dtype=pl.Float64),
        "vp_fr_exit": pl.Series("vp_fr_exit", z, dtype=pl.Float64),
        "vp_fr_upper": pl.Series("vp_fr_upper", [None] * n, dtype=pl.Float64),
        "vp_fr_mid": pl.Series("vp_fr_mid", [None] * n, dtype=pl.Float64),
        "vp_fr_lower": pl.Series("vp_fr_lower", [None] * n, dtype=pl.Float64),
        "vp_fr_start_ts": pl.Series("vp_fr_start_ts", zi, dtype=pl.Int64),
        "vp_fr_end_ts": pl.Series("vp_fr_end_ts", zi, dtype=pl.Int64),
    }


def _bucket_trade_lists(
    trades: pl.DataFrame,
    bucket_starts: list[int],
) -> dict[int, list[tuple[int, int]]]:
    """يجمّع صفقات كل برميل: bucket_start -> [(price, size), ...]."""
    if trades.height == 0 or not bucket_starts:
        return {}
    out: dict[int, list[tuple[int, int]]] = {int(b): [] for b in bucket_starts}
    if BUCKET_START not in trades.columns:
        return out
    for row in trades.select(BUCKET_START, "price", "size").iter_rows():
        b, px, sz = int(row[0]), int(row[1]), int(row[2])
        if b in out:
            out[b].append((px, sz))
    return out


def attach_vp_fixed_range(  # noqa: PLR0912, PLR0915
    profile_states: pl.DataFrame,
    mbo: pl.DataFrame,
    *,
    interval_ns: int,
    config: VpFixedRangeConfig | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يلحق أعمدة الرينج الثابت على براميل الرينج (5د) سببيًا."""
    cfg = config if config is not None else VpFixedRangeConfig()
    if interval_ns < 1:
        raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")
    n = profile_states.height
    if n == 0:
        return profile_states.hstack(list(_empty_fr(0).values()))

    ordered = profile_states.sort(BUCKET_START)
    bucket_starts = ordered[BUCKET_START].to_list()
    bucket_ends = ordered[BUCKET_END].to_list()
    closes = ordered["close"].to_numpy().astype(np.float64)
    ranges = ordered["range"].to_numpy().astype(np.float64)
    is_exp = ordered["is_expansion"].to_numpy()
    is_bal = ordered["is_balanced"].to_numpy()
    in_val_frac = ordered["in_value_fraction"].to_numpy().astype(np.float64)
    close_in_val = ordered["close_in_value"].to_numpy()
    sessions = (
        ordered[VP_LIQUIDITY_SESSION].to_numpy().astype(np.int64)
        if VP_LIQUIDITY_SESSION in ordered.columns
        else np.zeros(n, dtype=np.int64)
    )

    trades = extract_trades(add_time_bucket(mbo, interval_ns=interval_ns))
    by_bucket = _bucket_trade_lists(trades, [int(x) for x in bucket_starts])

    active = np.zeros(n, dtype=np.float64)
    accepted = np.zeros(n, dtype=np.float64)
    in_balance = np.zeros(n, dtype=np.float64)
    exit_sig = np.zeros(n, dtype=np.float64)
    upper = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    start_ts = np.zeros(n, dtype=np.int64)
    end_ts = np.zeros(n, dtype=np.int64)

    running = DevelopingVolumeProfile(fraction=cfg.value_fraction)
    range_open = False
    start_i = -1
    end_i = -1
    pending_exp_i = -1
    fr_vah = fr_poc = fr_val = None

    def _add_bucket(i: int) -> None:
        for px, sz in by_bucket.get(int(bucket_starts[i]), []):
            running.add_trade(px, sz)

    def _publish_va(i: int) -> None:
        nonlocal fr_vah, fr_poc, fr_val
        va = running.value_area()
        if va is None:
            return
        fr_vah, fr_poc, fr_val = float(va.vah), float(va.poc), float(va.val)
        upper[i] = fr_vah
        mid[i] = fr_poc
        lower[i] = fr_val

    def _close_range(*, keep_pending: bool = False) -> None:
        nonlocal range_open, start_i, end_i, pending_exp_i, fr_vah, fr_poc, fr_val
        range_open = False
        start_i = -1
        end_i = -1
        if not keep_pending:
            pending_exp_i = -1
        fr_vah = fr_poc = fr_val = None
        running.clear()

    if progress is not None:
        progress.op(f"vp_fixed_range: bars={n:,}")

    for i in range(n):
        # انتقال جلسة: اقفل بلا قرار
        if range_open and i > 0 and int(sessions[i]) != int(sessions[i - 1]):
            _close_range()
        if pending_exp_i >= 0 and int(sessions[i]) != int(sessions[pending_exp_i]):
            pending_exp_i = -1

        # آخر expansion مرشّح (ليس الأول فقط) بانتظار قبول عرضي
        if not range_open and bool(is_exp[i]) and not bool(is_bal[i]):
            pending_exp_i = i

        # قبول expansion: عرضي/توازن خلال النافذة — العلم على برميل القبول (سببي)
        if not range_open and pending_exp_i >= 0 and i > pending_exp_i:
            age = i - pending_exp_i
            if age > cfg.accept_window:
                pending_exp_i = -1
            else:
                accepted_here = bool(is_bal[i]) or (
                    bool(close_in_val[i]) and float(in_val_frac[i]) >= cfg.balance_frac
                )
                if accepted_here:
                    running.clear()
                    start_i = pending_exp_i
                    for j in range(pending_exp_i, i + 1):
                        _add_bucket(j)
                    end_i = i
                    range_open = True
                    accepted[i] = 1.0  # نقطة المعرفة = برميل القبول لا برميل الاندفاع
                    pending_exp_i = -1
                    _publish_va(i)

        if not range_open:
            continue

        if fr_vah is None or fr_val is None or fr_poc is None:
            # لا منطقة قيمة بعد — أبقِ الرينج مغلقًا منطقيًا
            _close_range()
            continue

        close_in_fr = fr_val <= float(closes[i]) <= fr_vah
        # مدّ النهاية فقط داخل التوازن (is_balanced) — ليس بمجرد الإغلاق داخل FR
        lateral = bool(is_bal[i])
        outside = float(closes[i]) > fr_vah or float(closes[i]) < fr_val
        exp_now = bool(is_exp[i])
        prev_r = float(ranges[i - 1]) if i > 0 and ranges[i - 1] > 0 else 0.0
        range_blow = prev_r > 0 and float(ranges[i]) >= cfg.exit_expansion_ratio * prev_r
        clear_exit = outside and (exp_now or range_blow)

        if clear_exit:
            direction = 1.0 if float(closes[i]) > fr_vah else -1.0
            exit_sig[i] = direction
            # حدود الرينج للسياق على برميل الخروج؛ active=0 حتى لا يلصق use_fr عبر asof
            upper[i], mid[i], lower[i] = fr_vah, fr_poc, fr_val
            start_ts[i] = int(bucket_starts[start_i])
            end_ts[i] = int(bucket_ends[end_i])
            # برميل الخروج نفسه قد يصبح مرشّح expansion للدورة التالية
            seed_pending = exp_now
            _close_range(keep_pending=False)
            if seed_pending:
                pending_exp_i = i
            continue

        # رينج حيّ فقط هنا
        active[i] = 1.0
        start_ts[i] = int(bucket_starts[start_i])
        end_ts[i] = int(bucket_ends[end_i])
        upper[i], mid[i], lower[i] = fr_vah, fr_poc, fr_val

        if lateral:
            # مدّ النهاية داخل التوازن فقط؛ عند اللحاق تخطَّ حجم الوخزات خارج FR
            if i > end_i:
                for j in range(end_i + 1, i + 1):
                    j_close = float(closes[j])
                    j_in = fr_val <= j_close <= fr_vah
                    if bool(is_bal[j]) or j_in:
                        _add_bucket(j)
                end_i = i
                _publish_va(i)
            in_balance[i] = 1.0
            end_ts[i] = int(bucket_ends[end_i])
            if fr_vah is not None:
                upper[i], mid[i], lower[i] = fr_vah, fr_poc, fr_val
        elif close_in_fr and not bool(is_exp[i]):
            # إغلاق داخل FR بلا توازن معلن = بناء؛ لا تمدّ النهاية
            in_balance[i] = 0.0
            end_ts[i] = int(bucket_ends[end_i])
        else:
            # وخزة/تلاعب خارجي بدون قبول خروج — أبقِ الرينج، لا تمدّ النهاية
            in_balance[i] = 0.0
            end_ts[i] = int(bucket_ends[end_i])

    cols = {
        "vp_fr_active": pl.Series("vp_fr_active", active),
        "vp_fr_accepted_expansion": pl.Series("vp_fr_accepted_expansion", accepted),
        "vp_fr_in_balance": pl.Series("vp_fr_in_balance", in_balance),
        "vp_fr_exit": pl.Series("vp_fr_exit", exit_sig),
        "vp_fr_upper": pl.Series("vp_fr_upper", upper),
        "vp_fr_mid": pl.Series("vp_fr_mid", mid),
        "vp_fr_lower": pl.Series("vp_fr_lower", lower),
        "vp_fr_start_ts": pl.Series("vp_fr_start_ts", start_ts),
        "vp_fr_end_ts": pl.Series("vp_fr_end_ts", end_ts),
    }
    out = ordered.hstack(list(cols.values()))
    return out.with_columns(
        [
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c in ("vp_fr_upper", "vp_fr_mid", "vp_fr_lower")
        ]
    )


__all__ = [
    "VP_FIXED_RANGE_COLUMNS",
    "VpFixedRangeConfig",
    "attach_vp_fixed_range",
]
