"""خطط دخول/خروج تنفيذية بـ R:R هيكلي (مش نقطتين).

المستويات من Volume Profile الجلسي (VAL/VAH/POC) + ثيسيس المزاد بعد هولد:

* الدخول عند إغلاق برميل اكتمال الهولد (``entry_gate=1``).
* الوقف خلف الحد المعاكس لمنطقة القيمة (+ حاجز تيك).
* الهدف: الحد المقابل أو مضاعف R — أيهما يحقّق ``min_rr``.
* لا صفقة إن كان R:R المخطط < ``min_rr`` أو السوق محكوم كاذبًا.

البحث: شبكة ``(hold_buckets × min_rr × stop_buffer_ticks × target_mode)``
مع تقييم عائد محقّق بم stub مسار سعري سببي (إغلاقات البراميل اللاحقة).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.research.progress import ProgressLike
from nq.simulation.auction import (
    VP_PROFILE_INTERVAL_NS,
    auction_action_states,
)
from nq.simulation.deceptive_liquidity import (
    DeceptiveLiquidityConfig,
    deceptive_features_by_bucket,
)
from nq.simulation.market_truth import MarketTruthConfig, build_market_truth_frame

FloatArray = npt.NDArray[np.float64]
TargetMode = Literal["va_opposite", "rr_multiple"]

EDGE_TRADE_COLUMNS: Final[tuple[str, ...]] = (
    "edge_signal",
    "edge_entry",
    "edge_stop",
    "edge_target",
    "edge_rr",
    "edge_risk",
    "edge_reward",
    "edge_pnl",
    "edge_hit",  # +1 target / -1 stop / 0 timeout
)

_TRAIN_FRAC_MIN: Final = 0.1
_TRAIN_FRAC_MAX: Final = 0.9
_MIN_TRUTH_ROWS_FOR_OOS: Final = 20


@dataclass(frozen=True, slots=True)
class EdgeExecConfig:
    """إعداد خطة تنفيذ الإدج."""

    min_rr: float = 2.5
    stop_buffer_ticks: float = 2.0
    target_mode: TargetMode = "va_opposite"
    rr_multiple: float = 3.0
    max_hold_buckets: int = 30
    tick_size: float = 0.25


@dataclass(frozen=True, slots=True)
class EdgeSearchSpec:
    """فرضية بحث لشبكة الدخول/الخروج."""

    name: str
    hold_buckets: int
    min_rr: float
    stop_buffer_ticks: float
    target_mode: TargetMode
    rr_multiple: float = 3.0

    def exec_config(self) -> EdgeExecConfig:
        return EdgeExecConfig(
            min_rr=self.min_rr,
            stop_buffer_ticks=self.stop_buffer_ticks,
            target_mode=self.target_mode,
            rr_multiple=self.rr_multiple,
        )

    def truth_config(self) -> MarketTruthConfig:
        return MarketTruthConfig(hold_buckets=self.hold_buckets)


def default_edge_search_grid() -> tuple[EdgeSearchSpec, ...]:
    """شبكة بحث محافظة — R:R قوي، هولد يمنع ملاحقة كل أمر."""
    specs: list[EdgeSearchSpec] = []
    for hold in (2, 3, 5):
        for min_rr in (2.0, 2.5, 3.0, 4.0):
            for buf in (1.0, 2.0, 4.0):
                for mode in ("va_opposite", "rr_multiple"):
                    name = f"hold{hold}_rr{min_rr:g}_buf{buf:g}_{mode}"
                    specs.append(
                        EdgeSearchSpec(
                            name=name,
                            hold_buckets=hold,
                            min_rr=min_rr,
                            stop_buffer_ticks=buf,
                            target_mode=mode,
                            rr_multiple=max(min_rr, 3.0),
                        )
                    )
    return tuple(specs)


def _plan_levels(
    *,
    direction: float,
    entry: float,
    vah: float,
    val: float,
    cfg: EdgeExecConfig,
) -> tuple[float, float, float, float] | None:
    """يُعيد (stop, target, risk, reward) بالسعر الحقيقي أو None إن R:R ضعيف."""
    tick = cfg.tick_size
    buf = cfg.stop_buffer_ticks * tick
    if direction > 0:
        stop = min(val, entry) - buf
        risk = entry - stop
        if risk <= 0:
            return None
        if cfg.target_mode == "va_opposite":
            target = max(vah, entry + cfg.min_rr * risk)
        else:
            target = entry + cfg.rr_multiple * risk
        reward = target - entry
    elif direction < 0:
        stop = max(vah, entry) + buf
        risk = stop - entry
        if risk <= 0:
            return None
        if cfg.target_mode == "va_opposite":
            target = min(val, entry - cfg.min_rr * risk)
        else:
            target = entry - cfg.rr_multiple * risk
        reward = entry - target
    else:
        return None
    if risk <= 0 or reward / risk < cfg.min_rr:
        return None
    return stop, target, risk, reward


def simulate_edge_trades(  # noqa: PLR0915
    truth: pl.DataFrame,
    *,
    exec_cfg: EdgeExecConfig | None = None,
) -> pl.DataFrame:
    """يبني إشارات تنفيذ + يحاكي المسار حتى وقف/هدف/انتهاء الهولد الأقصى."""
    cfg = exec_cfg if exec_cfg is not None else EdgeExecConfig()
    n = truth.height
    empty_cols = {
        c: pl.Series(c, [np.nan] * n if n else [], dtype=pl.Float64) for c in EDGE_TRADE_COLUMNS
    }
    if n == 0:
        return truth.hstack(list(empty_cols.values()))

    gate = truth["entry_gate"].to_numpy().astype(np.float64)
    thesis = truth["thesis_dir"].to_numpy().astype(np.float64)
    close = truth["close"].to_numpy().astype(np.float64) * PRICE_SCALE
    vah = truth["vah"].to_numpy().astype(np.float64) * PRICE_SCALE
    val = truth["val"].to_numpy().astype(np.float64) * PRICE_SCALE
    verdict = truth["market_verdict"].to_numpy().astype(np.float64)

    signal = np.zeros(n, dtype=np.float64)
    entry_a = np.full(n, np.nan)
    stop_a = np.full(n, np.nan)
    target_a = np.full(n, np.nan)
    rr_a = np.full(n, np.nan)
    risk_a = np.full(n, np.nan)
    reward_a = np.full(n, np.nan)
    pnl_a = np.full(n, np.nan)
    hit_a = np.zeros(n, dtype=np.float64)

    i = 0
    while i < n:
        if gate[i] < 1.0 or verdict[i] < 0.0 or thesis[i] == 0.0:
            i += 1
            continue
        planned = _plan_levels(
            direction=float(thesis[i]),
            entry=float(close[i]),
            vah=float(vah[i]),
            val=float(val[i]),
            cfg=cfg,
        )
        if planned is None:
            i += 1
            continue
        stop, target, risk, reward = planned
        direction = float(thesis[i])
        signal[i] = direction
        entry_a[i] = close[i]
        stop_a[i] = stop
        target_a[i] = target
        risk_a[i] = risk
        reward_a[i] = reward
        rr_a[i] = reward / risk

        # محاكاة مسار لاحق سببيًا
        hit = 0.0
        pnl = 0.0
        last = min(n - 1, i + cfg.max_hold_buckets)
        for j in range(i + 1, last + 1):
            px = close[j]
            if direction > 0:
                if px <= stop:
                    hit = -1.0
                    pnl = (stop - close[i]) / close[i]
                    break
                if px >= target:
                    hit = 1.0
                    pnl = (target - close[i]) / close[i]
                    break
            else:
                if px >= stop:
                    hit = -1.0
                    pnl = (close[i] - stop) / close[i]
                    break
                if px <= target:
                    hit = 1.0
                    pnl = (close[i] - target) / close[i]
                    break
        else:
            # انتهاء الوقت عند آخر إغلاق
            px = close[last]
            pnl = direction * (px - close[i]) / close[i]
            hit = 0.0
        hit_a[i] = hit
        pnl_a[i] = pnl
        # لا تداخل صفقات — انتقل بعد نافذة الصفقة
        i = last + 1

    return truth.with_columns(
        pl.Series("edge_signal", signal),
        pl.Series("edge_entry", entry_a),
        pl.Series("edge_stop", stop_a),
        pl.Series("edge_target", target_a),
        pl.Series("edge_rr", rr_a),
        pl.Series("edge_risk", risk_a),
        pl.Series("edge_reward", reward_a),
        pl.Series("edge_pnl", pnl_a),
        pl.Series("edge_hit", hit_a),
    )


def summarize_edge_trades(trades: pl.DataFrame) -> dict[str, float]:
    """ملخص كمي للصفقات ذات الإشارة غير الصفرية."""
    if trades.height == 0 or "edge_signal" not in trades.columns:
        return {
            "n_trades": 0.0,
            "win_rate": 0.0,
            "avg_rr_planned": 0.0,
            "expectancy": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
        }
    active = trades.filter(pl.col("edge_signal") != 0.0)
    n = active.height
    if n == 0:
        return {
            "n_trades": 0.0,
            "win_rate": 0.0,
            "avg_rr_planned": 0.0,
            "expectancy": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
        }
    pnl = active["edge_pnl"].drop_nulls().to_numpy()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    return {
        "n_trades": float(n),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_rr_planned": float(cast(Any, active["edge_rr"].mean()) or 0.0),
        "expectancy": float(np.nanmean(pnl)) if len(pnl) else 0.0,
        "avg_pnl": float(np.nanmean(pnl)) if len(pnl) else 0.0,
        "profit_factor": float(pf) if np.isfinite(pf) else 99.0,
    }


def run_edge_plan(
    mbo: pl.DataFrame,
    *,
    interval_ns: int,
    truth_cfg: MarketTruthConfig | None = None,
    exec_cfg: EdgeExecConfig | None = None,
    deceptive: DeceptiveLiquidityConfig | None = None,
    score_mbo: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
    auction: pl.DataFrame | None = None,
    deceptive_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """خط واحد: حقيقة السوق → خطة تنفيذ → محاكاة.

    ``auction`` / ``deceptive_frame`` اختياريان لإعادة استخدام بناء لليوم
    (تجنّب إعادة تسجيل التضليل بعد الفلتر/البحث).
    """
    truth = build_market_truth_frame(
        mbo,
        interval_ns=interval_ns,
        truth=truth_cfg,
        deceptive=deceptive,
        score_mbo=score_mbo,
        progress=progress,
        auction=auction,
        deceptive_frame=deceptive_frame,
    )
    return simulate_edge_trades(truth, exec_cfg=exec_cfg)


def score_edge_spec_oos(
    mbo: pl.DataFrame,
    spec: EdgeSearchSpec,
    *,
    interval_ns: int,
    train_frac: float = 0.6,
    deceptive: DeceptiveLiquidityConfig | None = None,
    score_mbo: pl.DataFrame | None = None,
    truth_frame: pl.DataFrame | None = None,
    auction: pl.DataFrame | None = None,
    deceptive_frame: pl.DataFrame | None = None,
) -> dict[str, float | str]:
    """تقييم فرضية بمحاكاة مستقلة على التدريب ثم الاختبار (بلا تسرّب عبر القطع)."""
    if not _TRAIN_FRAC_MIN < train_frac < _TRAIN_FRAC_MAX:
        raise ValueError(
            f"train_frac must be in ({_TRAIN_FRAC_MIN}, {_TRAIN_FRAC_MAX}), got {train_frac}"
        )
    empty: dict[str, float | str] = {
        "name": spec.name,
        "train_expectancy": 0.0,
        "oos_expectancy": 0.0,
        "oos_win_rate": 0.0,
        "oos_n": 0.0,
        "oos_avg_rr": 0.0,
        "oos_profit_factor": 0.0,
    }
    truth = truth_frame
    if truth is None:
        truth = build_market_truth_frame(
            mbo,
            interval_ns=interval_ns,
            truth=spec.truth_config(),
            deceptive=deceptive,
            score_mbo=score_mbo,
            auction=auction,
            deceptive_frame=deceptive_frame,
        )
    if truth.height < _MIN_TRUTH_ROWS_FOR_OOS:
        return empty
    cut = int(truth.height * train_frac)
    purge = int(spec.exec_config().max_hold_buckets)
    train_truth = truth.head(cut)
    test_start = min(truth.height, cut + purge)
    test_truth = truth.slice(test_start)
    train = summarize_edge_trades(simulate_edge_trades(train_truth, exec_cfg=spec.exec_config()))
    test = summarize_edge_trades(simulate_edge_trades(test_truth, exec_cfg=spec.exec_config()))
    return {
        "name": spec.name,
        "train_expectancy": train["expectancy"],
        "oos_expectancy": test["expectancy"],
        "oos_win_rate": test["win_rate"],
        "oos_n": test["n_trades"],
        "oos_avg_rr": test["avg_rr_planned"],
        "oos_profit_factor": test["profit_factor"],
    }


def search_best_edge_spec(  # noqa: PLR0912
    mbo: pl.DataFrame,
    *,
    interval_ns: int,
    grid: tuple[EdgeSearchSpec, ...] | None = None,
    train_frac: float = 0.6,
    deceptive: DeceptiveLiquidityConfig | None = None,
    score_mbo: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
    min_oos_trades: int = 3,
    min_oos_rr: float = 2.0,
    auction: pl.DataFrame | None = None,
    deceptive_frame: pl.DataFrame | None = None,
    scored: pl.DataFrame | None = None,
    profile_interval_ns: int = VP_PROFILE_INTERVAL_NS,
) -> tuple[pl.DataFrame, EdgeSearchSpec | None, dict[str, float | str]]:
    """يبحث عن أفضل دخول/خروج؛ يعيد ``best=None`` إن لم تُحقَّق القيود.

    ``auction`` / ``deceptive_frame`` / ``scored`` اختياريان لإعادة استخدام
    بناء اليوم. الرينج الافتراضي 5د على ساعة فعل ``interval_ns`` (30ث).
    """
    specs = grid if grid is not None else default_edge_search_grid()
    rows: list[dict[str, float | str]] = []
    if progress is not None:
        progress.op(f"edge search: {len(specs)} مواصفات")
        if deceptive_frame is not None:
            progress.op(f"edge search: reuse deceptive_frame · buckets={deceptive_frame.height:,}")
        elif scored is not None:
            progress.op(f"edge search: reuse scored · rows={scored.height:,}")
        else:
            progress.op("edge search: بناء auction + deceptive (قد يعيد التسجيل إن لم يُمرَّر scored)")
    states = (
        auction
        if auction is not None
        else auction_action_states(
            mbo,
            profile_interval_ns=profile_interval_ns,
            signal_interval_ns=interval_ns,
            progress=progress,
        )
    )
    if deceptive_frame is not None:
        deco = deceptive_frame
    else:
        score_src = score_mbo if score_mbo is not None else mbo
        deco = deceptive_features_by_bucket(
            score_src,
            interval_ns=interval_ns,
            config=deceptive,
            progress=progress,
            scored=scored,
        )
    if progress is not None:
        progress.op(f"edge search: جاهز · auction={states.height:,} · deco={deco.height:,}")
    # hold_buckets هو الفرق الوحيد في MarketTruthConfig عبر الشبكة الافتراضية.
    truth_by_hold: dict[int, pl.DataFrame] = {}
    for i, spec in enumerate(specs):
        hold = int(spec.hold_buckets)
        if hold not in truth_by_hold:
            if progress is not None:
                progress.op(f"edge search: market_truth hold={hold}")
            truth_by_hold[hold] = build_market_truth_frame(
                mbo,
                interval_ns=interval_ns,
                truth=spec.truth_config(),
                auction=states,
                deceptive_frame=deco,
            )
        row = score_edge_spec_oos(
            mbo,
            spec,
            interval_ns=interval_ns,
            train_frac=train_frac,
            truth_frame=truth_by_hold[hold],
        )
        rows.append(row)
        if progress is not None:
            progress.heartbeat(i + 1, len(specs), label="edge-spec")
    table = pl.DataFrame(rows) if rows else pl.DataFrame()
    if table.height == 0:
        return table, None, {}
    eligible = table.filter(
        (pl.col("oos_n") >= float(min_oos_trades)) & (pl.col("oos_avg_rr") >= float(min_oos_rr))
    )
    if eligible.height == 0:
        return table, None, {}
    best_row = eligible.sort("oos_expectancy", descending=True).row(0, named=True)
    best_spec = next((s for s in specs if s.name == best_row["name"]), None)
    return table, best_spec, dict(best_row)


__all__ = [
    "EDGE_TRADE_COLUMNS",
    "EdgeExecConfig",
    "EdgeSearchSpec",
    "default_edge_search_grid",
    "run_edge_plan",
    "score_edge_spec_oos",
    "search_best_edge_spec",
    "simulate_edge_trades",
    "summarize_edge_trades",
]
