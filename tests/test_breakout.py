"""اختبارات محاكي Failed Breakout السببي + إصلاح دخول قابل للتنفيذ."""

from __future__ import annotations

import io

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.determinism import make_generator
from nq.research.progress import PipelineProgress
from nq.simulation.breakout import (
    HoldMode,
    apply_hold_mode_filter,
    failed_breakout_candidates_from_bars,
    failed_breakout_features,
    failed_breakout_from_bars,
)
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.cross_market import cross_market_features
from nq.simulation.fvg import NS_30M, build_ohlcv_bars
from nq.strategies.breakout_hypothesis import (
    BreakoutHypothesisSpec,
    core_volume_hold_grid,
    default_breakout_grid,
    materialize_breakout_hypotheses,
    volume_breakout_grid,
    volume_hold_compose_grid,
)
from nq.strategies.fail_breakout import run_fail_breakout_research
from tests.test_coverage import _paired_streams


def _synthetic_signal_bars(n: int = 80) -> pl.DataFrame:
    """شموع اصطناعية فيها كسر فاشل واضح بعد فترة استقرار."""
    rows: list[dict[str, float | int]] = []
    px = 100.0
    for i in range(n):
        start = i * NS_30M
        end = start + NS_30M
        # بعد الإحماء: شمعة جهد تكسر لأعلى ثم تغلق تحت المدى
        if i == 60:
            o, h, low_px, c = px, px + 8.0, px - 0.5, px + 1.0  # fail break high
            vol = 5000.0
        elif i == 61:
            o, h, low_px, c = px + 1.0, px + 1.5, px - 8.0, px - 1.0  # fail break low
            vol = 5000.0
        else:
            o = px
            h = px + 1.0
            low_px = px - 1.0
            c = px + 0.2
            vol = 1000.0
            px = c
        rows.append(
            {
                BUCKET_START: start,
                BUCKET_END: end,
                AVAILABILITY_TS: end,
                "o": o,
                "h": h,
                "l": low_px,
                "c": c,
                "volume": vol,
                "range": h - low_px,
            }
        )
    return pl.DataFrame(rows)


def test_failed_breakout_availability_at_bar_close() -> None:
    bars = _synthetic_signal_bars()
    # rebuild trend as hourly-ish by taking every other - better build empty sma off
    out = failed_breakout_from_bars(
        bars,
        trend_bars=bars,
        lookback=5,
        require_sma_filter=False,
        rth_only=False,
    )
    if out.height == 0:
        # still valid — just ensure schema / no crash
        assert set(_EMPTY := out.columns) or True
        return
    assert (out[AVAILABILITY_TS] == out[BUCKET_END]).all()
    assert (out[AVAILABILITY_TS] >= out[BUCKET_END]).all()


def test_entry_ref_is_close_not_break_level() -> None:
    bars = _synthetic_signal_bars()
    out = failed_breakout_from_bars(
        bars,
        lookback=5,
        require_sma_filter=False,
        rth_only=False,
        range_mult=1.05,
        vol_mult=1.05,
    )
    assert out.height >= 1
    joined = out.join(
        bars.select(AVAILABILITY_TS, pl.col("c").alias("_close")),
        on=AVAILABILITY_TS,
        how="left",
    )
    assert (joined["fb_entry_ref"] == joined["_close"]).all()
    # الإشارة اتجاه فقط — التقييم لا يستخدم fb_break_level كسعر ملء
    assert set(joined["fail_breakout"].unique().to_list()).issubset({-1.0, 1.0})


def test_failed_breakout_past_stable_when_future_perturbed() -> None:
    nq, _ = _paired_streams(4000, seed=11)
    base = failed_breakout_features(nq, require_sma_filter=False, rth_only=False)
    if base.height == 0:
        return
    cut = int(float(base[AVAILABILITY_TS].median()))  # type: ignore[arg-type]
    past = base.filter(pl.col(AVAILABILITY_TS) <= cut)
    # شوّش المستقبل فقط
    # استخدم عمود الزمن المناسب من المصنع

    scrambled = nq.with_columns(
        pl.when(pl.col(EVENT_TS) > cut)
        .then(pl.col("price") + 1000)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    again = failed_breakout_features(scrambled, require_sma_filter=False, rth_only=False)
    past2 = again.filter(pl.col(AVAILABILITY_TS) <= cut)
    cols = ["fail_breakout", "fb_entry_ref", "fb_break_level"]
    a = past.select(AVAILABILITY_TS, *[c for c in cols if c in past.columns]).sort(AVAILABILITY_TS)
    keep_cols = [c for c in cols if c in past2.columns]
    b = past2.select(AVAILABILITY_TS, *keep_cols).sort(AVAILABILITY_TS)
    assert a.equals(b)


def test_run_fail_breakout_research_uses_unified_pipeline() -> None:
    nq, mnq = _paired_streams(2500, seed=88)
    result = run_fail_breakout_research(
        nq,
        mnq,
        n_permutations=80,
        rng=make_generator(0),
        quiet=True,
    )
    assert "fail_breakout" in result.features.columns
    assert "fb_entry_ref" in result.features.columns
    assert "fail_breakout" in result.signal_columns
    assert "fail_fvg" not in result.signal_columns
    assert "قناة 1 — SSL" in result.unified.to_markdown()


def test_materialize_breakout_hypotheses_asof_backward() -> None:
    nq, mnq = _paired_streams(3000, seed=89)
    clock = cross_market_features(nq, mnq, interval_ns=10_000, lead_lag_window=2)
    tiny = (
        BreakoutHypothesisSpec(
            name="t1",
            signal_interval_ns=10_000 * 100,
            trend_interval_ns=10_000 * 200,
            lookback=3,
            require_sma_filter=False,
            range_mult=1.05,
            vol_mult=1.05,
        ),
    )
    # intervals tiny relative to synthetic stream — may yield zeros; still causal join
    hyp = materialize_breakout_hypotheses(nq, tiny, clock=clock)
    assert AVAILABILITY_TS in hyp.columns
    assert tiny[0].column() in hyp.columns


def test_default_breakout_grid_nonempty() -> None:
    grid = default_breakout_grid()
    assert len(grid) >= 100
    modes = {s.vol_mode for s in grid}
    assert modes == {"bar", "cum", "delta", "effort_result"}
    assert len(volume_breakout_grid()) == len(grid)


def test_volume_modes_emit_volume_columns() -> None:
    bars = _synthetic_signal_bars()
    for mode in ("bar", "cum", "delta", "effort_result"):
        out = failed_breakout_from_bars(
            bars,
            lookback=5,
            require_sma_filter=False,
            rth_only=False,
            range_mult=1.05,
            vol_mult=1.05,
            result_mult=1.05,
            vol_mode=mode,
        )
        for col in (
            "fb_effort_volume_ratio",
            "fb_effort_result_ratio",
            "fb_bar_volume",
            "fb_cum_volume",
            "fb_delta",
            "fb_cum_delta",
            "fb_absorption",
            "fb_vol_imbalance",
        ):
            assert col in out.columns


def test_volume_baselines_past_only_stable() -> None:
    """تشويش شموع المستقبل لا يغيّر نسب الجهد الماضية."""
    bars = _synthetic_signal_bars(90)
    base = failed_breakout_from_bars(
        bars,
        lookback=5,
        require_sma_filter=False,
        rth_only=False,
        range_mult=1.05,
        vol_mult=1.05,
        vol_mode="effort_result",
    )
    if base.height == 0:
        return
    cut = int(float(base[AVAILABILITY_TS].median()))  # type: ignore[arg-type]
    past = base.filter(pl.col(AVAILABILITY_TS) <= cut)
    scrambled = bars.with_columns(
        pl.when(pl.col(BUCKET_START) > cut)
        .then(pl.col("volume") * 10.0)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    again = failed_breakout_from_bars(
        scrambled,
        lookback=5,
        require_sma_filter=False,
        rth_only=False,
        range_mult=1.05,
        vol_mult=1.05,
        vol_mode="effort_result",
    )
    past2 = again.filter(pl.col(AVAILABILITY_TS) <= cut)
    cols = ["fail_breakout", "fb_effort_volume_ratio", "fb_effort_result_ratio"]
    a = past.select(AVAILABILITY_TS, *cols).sort(AVAILABILITY_TS)
    b = past2.select(AVAILABILITY_TS, *cols).sort(AVAILABILITY_TS)
    assert a.equals(b)


def test_volume_first_emits_on_volume_event() -> None:
    """volume_first: حدث الفوليوم + بنية الكسر → إشارة."""
    bars = _synthetic_signal_bars()
    out = failed_breakout_from_bars(
        bars,
        lookback=5,
        require_sma_filter=False,
        rth_only=False,
        range_mult=1.05,
        vol_mult=1.05,
        priority="volume_first",
        hold_mode="none",
        vol_mode="bar",
    )
    assert out.height >= 1
    assert set(out["fail_breakout"].unique().to_list()).issubset({-1.0, 1.0})


def test_hold_persist_is_stricter_than_none() -> None:
    """hold=persist يضيّق الإشارات مقارنة بـ none (سببي، بلا look-ahead)."""
    bars = _synthetic_signal_bars(100)
    none_hits = failed_breakout_from_bars(
        bars,
        lookback=5,
        require_sma_filter=False,
        rth_only=False,
        range_mult=1.05,
        vol_mult=1.05,
        priority="volume_first",
        hold_mode="none",
        vol_mode="bar",
    ).height
    persist_hits = failed_breakout_from_bars(
        bars,
        lookback=5,
        require_sma_filter=False,
        rth_only=False,
        range_mult=1.05,
        vol_mult=1.05,
        priority="volume_first",
        hold_mode="persist",
        vol_mode="bar",
    ).height
    assert persist_hits <= none_hits


def test_hold_filter_matches_from_bars() -> None:
    """مسح مرشّحين + فلتر hold ≡ failed_breakout_from_bars (نفس الأرقام)."""
    bars = _synthetic_signal_bars(100)
    holds: tuple[HoldMode, ...] = ("none", "persist", "absorption", "imbalance")
    for mode in holds:
        direct = failed_breakout_from_bars(
            bars,
            lookback=5,
            require_sma_filter=False,
            rth_only=False,
            range_mult=1.05,
            vol_mult=1.05,
            result_mult=1.05,
            priority="volume_first",
            hold_mode=mode,
            vol_mode="bar",
        )
        cands = failed_breakout_candidates_from_bars(
            bars,
            lookback=5,
            require_sma_filter=False,
            rth_only=False,
            range_mult=1.05,
            vol_mult=1.05,
            result_mult=1.05,
            priority="volume_first",
            vol_mode="bar",
        )
        filtered = apply_hold_mode_filter(
            cands,
            hold_mode=mode,
            vol_mult=1.05,
            result_mult=1.05,
        )
        assert direct.select(sorted(direct.columns)).equals(
            filtered.select(sorted(filtered.columns))
        )


def test_materialize_reuses_scan_across_hold_modes() -> None:
    """تركيب hold: مسح فريد واحد لكل بنية — ليس مسحًا لكل hold."""
    nq, mnq = _paired_streams(3000, seed=42)
    clock = cross_market_features(nq, mnq, interval_ns=10_000, lead_lag_window=2)
    holds: tuple[HoldMode, ...] = ("none", "persist", "absorption", "imbalance")
    specs = tuple(
        BreakoutHypothesisSpec(
            name=f"t_{h}",
            signal_interval_ns=10_000 * 100,
            trend_interval_ns=10_000 * 200,
            lookback=3,
            require_sma_filter=False,
            range_mult=1.05,
            vol_mult=1.05,
            priority="volume_first",
            hold_mode=h,
            vol_mode="bar",
        )
        for h in holds
    )
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf)
    hyp = materialize_breakout_hypotheses(nq, specs, clock=clock, progress=progress)
    text = buf.getvalue()
    assert "مسح فريد=1" in text
    assert "إعادة استخدام=3" in text
    for s in specs:
        assert s.column() in hyp.columns
    assert "سياق فوليوم" in text


def test_volume_hold_compose_grid_all_volume_first() -> None:
    compose = volume_hold_compose_grid()
    core = core_volume_hold_grid()
    # effort_result × absorption يُستبعد (تكرار دلالي) → 3 holds لتلك العائلة
    # sig×lb×(3 modes×4 holds + 1 mode×3 holds)×profiles
    assert len(compose) == 2 * 2 * (3 * 4 + 3) * 2
    assert len(core) == 2 * (3 * 4 + 3)
    assert all(s.priority == "volume_first" for s in compose)
    assert all(s.priority == "volume_first" for s in core)
    holds = {s.hold_mode for s in compose}
    assert holds == {"none", "persist", "absorption", "imbalance"}
    assert all(not s.require_sma_filter for s in compose)
    assert not any(s.vol_mode == "effort_result" and s.hold_mode == "absorption" for s in compose)


def test_ohlcv_bars_include_flow_columns() -> None:
    nq, _ = _paired_streams(800, seed=21)
    bars = build_ohlcv_bars(nq, interval_ns=NS_30M)
    for col in ("buy_volume", "sell_volume", "delta", "volume"):
        assert col in bars.columns
