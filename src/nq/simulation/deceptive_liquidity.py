"""فلتر سيولة تضليلية علمي — كشف أوامر وهمية سببيًا (درجة + إسقاط).

العقد (قبل التنفيذ التنفيذي):

1. **الهدف:** تنظيف المدخلات *و* درجة ``deceptive_score`` كبوابة — الاتنان معًا.
2. **تعريف قابل للقياس (سببي فقط):** كل حكم عند ``t`` يستخدم أحداثًا بـ
   ``event_ts <= t`` فقط.
3. **لا يُمس:** TRADE/FILL — حقيقة الشريط تبقى دائمًا.
4. **إثبات الفائدة:** ميزات مجمّعة على البرميل تُقارن A/B في طبقة الإدج.

إشارات الخداع (مرشّحات قوية، ليست شعارات):

* عمر قصير + إلغاء بلا fill → ما تحوّلت لسيولة حقيقية.
* حجم كبير بعيد عن mid ثم اختفى → ضغط ظاهري بلا التزام.
* تعديلات متكررة بلا تطابق → تحريك طعم.
* وصول السعر للمستوى دون مشاركة الأمر → سيولة «ما اتكسرتش».
* إلغاء جماعي متزامن بعيد عن الداخل → عاصفة تضليل.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from nq.contracts.mbo import PRICE_SCALE, MboAction
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.time import sort_causal
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket

_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_MODIFY = MboAction.MODIFY.value

_DEFAULT_TICK: Final = 0.25
_TICK_FIXED: Final = round(_DEFAULT_TICK / PRICE_SCALE)

DECEPTIVE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "deceptive_score",
    "deceptive_volume_share",
    "real_liquidity_ratio",
    "deceptive_cancel_rate",
    "noise_instant",
    "noise_cum",
    "spoof_flag",
    "flicker_flag",
    "bait_modify_flag",
    "nonparticipate_flag",
    "storm_flag",
)


@dataclass(frozen=True, slots=True)
class DeceptiveLiquidityConfig:
    """عتبات علمية لدرجة التضليل والإسقاط."""

    short_life_ns: int = 80_000_000  # 80ms — عمر بلا fill
    spoof_ticks_from_mid: int = 4
    spoof_min_size: int = 5
    spoof_cancel_ns: int = 300_000_000  # 0.3s
    bait_modify_min: int = 3
    storm_window_ns: int = 500_000_000
    storm_cancel_ratio: float = 0.7
    storm_min_events: int = 10
    storm_ticks_from_inside: int = 3
    drop_score: float = 0.65
    #: أوزان مكوّنات الدرجة (تُطبَّع تلقائيًا).
    w_short_life: float = 0.25
    w_spoof: float = 0.30
    w_bait: float = 0.15
    w_nonparticipate: float = 0.15
    w_storm: float = 0.15


def _normalize_weights(cfg: DeceptiveLiquidityConfig) -> tuple[float, float, float, float, float]:
    raw = (
        cfg.w_short_life,
        cfg.w_spoof,
        cfg.w_bait,
        cfg.w_nonparticipate,
        cfg.w_storm,
    )
    s = float(sum(raw))
    if s <= 0:
        raise ValueError("deceptive score weights must sum to > 0")
    return tuple(x / s for x in raw)  # type: ignore[return-value]


def score_deceptive_events(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    config: DeceptiveLiquidityConfig | None = None,
) -> pl.DataFrame:
    """يُلحق بكل حدث MBO درجة تضليل سببية ومكوّناتها.

    الصفقات/التنفيذات تحصل على درجة 0 وتُبقى دائمًا. الإضافات تُسجَّل؛ الحكم
    النهائي عند الإلغاء/بعد وصول السعر للمستوى.
    """
    cfg = config if config is not None else DeceptiveLiquidityConfig()
    if frame.height == 0:
        empty = {c: pl.Series(c, [], dtype=pl.Float64) for c in DECEPTIVE_FEATURE_COLUMNS}
        return frame.hstack(list(empty.values())) if frame.width else pl.DataFrame(
            schema={**{c: pl.Float64() for c in DECEPTIVE_FEATURE_COLUMNS}}
        )

    w = _normalize_weights(cfg)
    work = sort_causal(frame)
    actions = work["action"].cast(pl.Utf8).to_list()
    sides = work["side"].cast(pl.Utf8).to_list()
    prices = work["price"].to_list()
    sizes = work["size"].to_list()
    order_ids = work["order_id"].to_list()
    event_ts = work[EVENT_TS].to_list()
    n = len(actions)

    # oid -> (add_ts, price, side, size, modify_count, touched_by_trade)
    live: dict[int, tuple[int, int, str, int, int, bool]] = {}
    executed_oids: set[int] = set()
    # price -> set of resting oids (للمشاركة عند وصول الصفقة)
    by_price: dict[int, set[int]] = {}

    best_bid: int | None = None
    best_ask: int | None = None
    recent: list[tuple[int, bool]] = []  # (ts, is_cancel)
    head = 0

    scores = [0.0] * n
    spoof_f = [0.0] * n
    flicker_f = [0.0] * n
    bait_f = [0.0] * n
    nonpart_f = [0.0] * n
    storm_f = [0.0] * n
    drop_mask = [False] * n

    def _mid() -> float | None:
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def _refresh_bbo() -> None:
        nonlocal best_bid, best_ask
        bids = [meta[1] for meta in live.values() if meta[2] == "B"]
        asks = [meta[1] for meta in live.values() if meta[2] == "A"]
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None

    for i in range(n):
        action = str(actions[i])
        side = str(sides[i])
        price = int(prices[i])
        size = int(sizes[i])
        oid = int(order_ids[i])
        ts = int(event_ts[i])

        cutoff = ts - cfg.storm_window_ns
        while head < len(recent) and recent[head][0] < cutoff:
            head += 1
        window = recent[head:]
        n_win = len(window)
        n_cancel = sum(1 for _, c in window if c)
        in_storm = (
            n_win >= cfg.storm_min_events
            and (n_cancel / float(n_win)) >= cfg.storm_cancel_ratio
        )

        if action in (_TRADE, _FILL):
            executed_oids.add(oid)
            # أوامر راقدة على نفس السعر ولم تُنفَّذ → مرشّح عدم مشاركة
            resting = by_price.get(price, set())
            for other in list(resting):
                if other == oid or other in executed_oids:
                    continue
                meta = live.get(other)
                if meta is None:
                    continue
                add_ts, _px, _sd, _sz, mods, _touched = meta
                live[other] = (add_ts, _px, _sd, _sz, mods, True)
            recent.append((ts, False))
            continue

        if action == _ADD:
            live[oid] = (ts, price, side, size, 0, False)
            by_price.setdefault(price, set()).add(oid)
            _refresh_bbo()
            mid = _mid()
            # إضافة ضخمة بعيدة أثناء عاصفة = تضليل فوري (إسقاط الإضافة)
            if in_storm and mid is not None and size >= cfg.spoof_min_size:
                dist = abs(price - mid) / float(_TICK_FIXED)
                if dist >= cfg.spoof_ticks_from_mid:
                    score = w[1] + w[4]
                    scores[i] = min(1.0, score)
                    spoof_f[i] = 1.0
                    storm_f[i] = 1.0
                    if scores[i] >= cfg.drop_score:
                        drop_mask[i] = True
                        by_price.get(price, set()).discard(oid)
                        live.pop(oid, None)
                        _refresh_bbo()
            recent.append((ts, False))
            continue

        if action == _MODIFY:
            meta = live.get(oid)
            if meta is not None:
                add_ts, old_px, old_side, _old_sz, mods, touched = meta
                by_price.get(old_px, set()).discard(oid)
                live[oid] = (add_ts, price, old_side, size, mods + 1, touched)
                by_price.setdefault(price, set()).add(oid)
                _refresh_bbo()
                if mods + 1 >= cfg.bait_modify_min and oid not in executed_oids:
                    bait_f[i] = 1.0
                    scores[i] = max(scores[i], w[2])
            recent.append((ts, False))
            continue

        if action == _CANCEL:
            meta = live.get(oid)
            short_life = 0.0
            spoof = 0.0
            bait = 0.0
            nonpart = 0.0
            storm = 1.0 if in_storm else 0.0
            if meta is not None:
                add_ts, add_px, add_side, add_sz, mods, touched = meta
                dt = ts - add_ts
                if dt <= cfg.short_life_ns and oid not in executed_oids:
                    short_life = 1.0
                    flicker_f[i] = 1.0
                mid = _mid()
                if (
                    mid is not None
                    and add_sz >= cfg.spoof_min_size
                    and dt <= cfg.spoof_cancel_ns
                    and oid not in executed_oids
                ):
                    dist = abs(add_px - mid) / float(_TICK_FIXED)
                    if dist >= cfg.spoof_ticks_from_mid:
                        spoof = 1.0
                        spoof_f[i] = 1.0
                if mods >= cfg.bait_modify_min and oid not in executed_oids:
                    bait = 1.0
                    bait_f[i] = 1.0
                if touched and oid not in executed_oids:
                    nonpart = 1.0
                    nonpart_f[i] = 1.0
                # عاصفة بعيدة عن الداخل
                if in_storm and mid is not None:
                    dist = abs(add_px - mid) / float(_TICK_FIXED)
                    if dist >= cfg.storm_ticks_from_inside:
                        storm = 1.0
                by_price.get(add_px, set()).discard(oid)
                live.pop(oid, None)
                _refresh_bbo()
            storm_f[i] = storm
            score = (
                w[0] * short_life
                + w[1] * spoof
                + w[2] * bait
                + w[3] * nonpart
                + w[4] * storm
            )
            scores[i] = float(min(1.0, score))
            if scores[i] >= cfg.drop_score:
                drop_mask[i] = True
            recent.append((ts, True))
            continue

        recent.append((ts, False))

    return work.with_columns(
        pl.Series("deceptive_score", scores, dtype=pl.Float64),
        pl.Series("spoof_flag", spoof_f, dtype=pl.Float64),
        pl.Series("flicker_flag", flicker_f, dtype=pl.Float64),
        pl.Series("bait_modify_flag", bait_f, dtype=pl.Float64),
        pl.Series("nonparticipate_flag", nonpart_f, dtype=pl.Float64),
        pl.Series("storm_flag", storm_f, dtype=pl.Float64),
        pl.Series("_deceptive_drop", drop_mask, dtype=pl.Boolean),
    )


def filter_deceptive_liquidity(
    frame: pl.DataFrame,
    *,
    config: DeceptiveLiquidityConfig | None = None,
) -> pl.DataFrame:
    """يُسقط أحداث ADD/CANCEL/MODIFY ذات درجة تضليل ≥ العتبة؛ يُبقي الصفقات."""
    scored = score_deceptive_events(frame, config=config)
    if scored.height == 0:
        return scored.drop([c for c in scored.columns if c.startswith("_deceptive")], strict=False)
    actions = scored["action"].cast(pl.Utf8)
    keep = (~scored["_deceptive_drop"]) | actions.is_in([_TRADE, _FILL])
    out = scored.filter(keep)
    drop_cols = [c for c in out.columns if c.startswith("_deceptive")]
    return out.drop(drop_cols) if drop_cols else out


def deceptive_features_by_bucket(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    config: DeceptiveLiquidityConfig | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يجمّع درجة التضليل على براميل زمنية (متاح عند ``bucket_end``).

    * ``noise_instant`` — متوسط درجة التضليل في البرميل الحالي.
    * ``noise_cum`` — متوسط تراكمي سببي عبر البراميل (تغيّر مجمّع).
    * ``real_liquidity_ratio`` — 1 − حصة الحجم المضلل عند الإلغاءات/الإضافات.
    """
    if progress is not None:
        progress.op(f"deceptive_features_by_bucket: interval_ns={interval_ns}")
    scored = score_deceptive_events(frame, config=config)
    if scored.height == 0:
        return pl.DataFrame(
            schema={
                AVAILABILITY_TS: pl.Int64(),
                BUCKET_START: pl.Int64(),
                BUCKET_END: pl.Int64(),
                **{c: pl.Float64() for c in DECEPTIVE_FEATURE_COLUMNS},
            }
        )

    bucketed = add_time_bucket(scored, interval_ns=interval_ns)
    # حجم «مضلل» ≈ size عند أحداث بدرجة عالية (CANCEL/ADD المسقطة منطقيًا)
    high = pl.col("deceptive_score") >= 0.5
    agg = (
        bucketed.group_by(BUCKET_START, maintain_order=True)
        .agg(
            pl.col(BUCKET_END).first().alias(BUCKET_END),
            pl.col(AVAILABILITY_TS).first().alias(AVAILABILITY_TS),
            pl.col("deceptive_score").mean().alias("deceptive_score"),
            pl.col("spoof_flag").max().alias("spoof_flag"),
            pl.col("flicker_flag").max().alias("flicker_flag"),
            pl.col("bait_modify_flag").max().alias("bait_modify_flag"),
            pl.col("nonparticipate_flag").max().alias("nonparticipate_flag"),
            pl.col("storm_flag").max().alias("storm_flag"),
            pl.when(high)
            .then(pl.col("size").cast(pl.Float64))
            .otherwise(0.0)
            .sum()
            .alias("_dec_sz"),
            pl.col("size").cast(pl.Float64).sum().alias("_all_sz"),
            (
                (pl.col("action").cast(pl.Utf8) == _CANCEL)
                & (pl.col("deceptive_score") >= 0.5)
            )
            .mean()
            .alias("deceptive_cancel_rate"),
        )
        .sort(BUCKET_START)
        .with_columns(
            pl.when(pl.col("_all_sz") > 0)
            .then(pl.col("_dec_sz") / pl.col("_all_sz"))
            .otherwise(0.0)
            .alias("deceptive_volume_share"),
        )
        .with_columns(
            (1.0 - pl.col("deceptive_volume_share")).alias("real_liquidity_ratio"),
            pl.col("deceptive_score").alias("noise_instant"),
        )
        .with_columns(
            (
                pl.col("noise_instant").cum_sum()
                / pl.int_range(1, pl.len() + 1).cast(pl.Float64)
            ).alias("noise_cum"),
        )
        .drop(["_dec_sz", "_all_sz"])
    )
    # ضمان ترتيب الأعمدة
    cols = [
        AVAILABILITY_TS,
        BUCKET_START,
        BUCKET_END,
        *DECEPTIVE_FEATURE_COLUMNS,
    ]
    return agg.select([c for c in cols if c in agg.columns])


__all__ = [
    "DECEPTIVE_FEATURE_COLUMNS",
    "DeceptiveLiquidityConfig",
    "deceptive_features_by_bucket",
    "filter_deceptive_liquidity",
    "score_deceptive_events",
]
