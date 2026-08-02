"""تدقيق ما بعد الدمج — أولويات 1–4 (NaN عمق / noise+bb / streaming / fill+TF)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import nq.research.orchestrator as orch
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core import align_horizon_to_context, resolve_grid_context_interval
from nq.core.determinism import make_generator
from nq.research.orchestrator import (
    _DEFAULT_SIGNAL_COLUMNS,
    PipelineConfig,
    _attach_failed_breakout,
    _build_research_features,
    _resolve_signal_columns,
    run_research_pipeline,
)
from nq.simulation.bottom_book import BOTTOM_BOOK_COLUMNS
from nq.simulation.breakout import FB_PULSE_ZERO_FILL
from nq.simulation.common import BUCKET_END
from tests.test_coverage import _paired_streams

# ─── P1: NaN لـ fb_depth_at_break + سياسة نبضة ─────────────────────────────


def test_fb_pulse_zero_fill_excludes_depth_and_effort() -> None:
    assert "fail_breakout" in FB_PULSE_ZERO_FILL
    assert "fb_delta" in FB_PULSE_ZERO_FILL
    for col in (
        "fb_depth_at_break",
        "fb_depth_imbalance",
        "fb_depth_cum_bid",
        "fb_depth_cum_ask",
        "fb_effort_range_ratio",
        "fb_effort_volume_ratio",
        "fb_effort_result_ratio",
        "fb_bar_volume",
        "fb_cum_volume",
        "fb_break_level",
        "fb_entry_ref",
        "fb_risk_pts",
    ):
        assert col not in FB_PULSE_ZERO_FILL


def test_attach_failed_breakout_empty_depth_uses_null_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_ts = 30 * 60 * 1_000_000_000
    fb = pl.DataFrame(
        {
            AVAILABILITY_TS: [signal_ts],
            "fail_breakout": [1.0],
            "fb_break_level": [100.0],
            "fb_entry_ref": [99.5],
            "fb_effort_range_ratio": [1.2],
            "fb_effort_volume_ratio": [1.1],
            "fb_effort_result_ratio": [0.8],
            "fb_bar_volume": [5000.0],
            "fb_cum_volume": [8000.0],
            "fb_delta": [100.0],
            "fb_cum_delta": [150.0],
            "fb_vol_imbalance": [0.2],
            "fb_absorption": [0.3],
            "fb_risk_pts": [2.0],
        }
    )
    monkeypatch.setattr(orch, "failed_breakout_features", lambda *a, **k: fb)
    features = pl.DataFrame({AVAILABILITY_TS: [signal_ts, signal_ts + 1_000_000_000]})
    empty_depth = pl.DataFrame({BUCKET_END: pl.Series([], dtype=pl.Int64)})
    out = _attach_failed_breakout(features, pl.DataFrame(), depth_30m=empty_depth)

    assert out.height == 2
    pulse_row = out.filter(pl.col(AVAILABILITY_TS) == signal_ts)
    other = out.filter(pl.col(AVAILABILITY_TS) != signal_ts)
    assert pulse_row["fail_breakout"][0] == 1.0
    assert pulse_row["fb_depth_at_break"][0] is None
    assert other["fail_breakout"][0] == 0.0
    assert other["fb_depth_at_break"][0] is None
    assert pulse_row["fb_effort_volume_ratio"][0] == pytest.approx(1.1)
    assert other["fb_effort_volume_ratio"][0] is None


def test_attach_failed_breakout_no_level_match_is_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_ts = 30 * 60 * 1_000_000_000
    fb = pl.DataFrame(
        {
            AVAILABILITY_TS: [signal_ts],
            "fail_breakout": [-1.0],
            "fb_break_level": [500.0],  # بعيد عن السلم
            "fb_entry_ref": [499.0],
            "fb_effort_range_ratio": [1.0],
            "fb_effort_volume_ratio": [1.0],
            "fb_effort_result_ratio": [1.0],
            "fb_bar_volume": [1000.0],
            "fb_cum_volume": [1000.0],
            "fb_delta": [-50.0],
            "fb_cum_delta": [-50.0],
            "fb_vol_imbalance": [-0.1],
            "fb_absorption": [0.0],
            "fb_risk_pts": [1.0],
        }
    )
    monkeypatch.setattr(orch, "failed_breakout_features", lambda *a, **k: fb)
    depth = pl.DataFrame(
        {
            BUCKET_END: [signal_ts],
            AVAILABILITY_TS: [signal_ts],
            "depth_cum_bid": [10.0],
            "depth_cum_ask": [12.0],
            "depth_imbalance": [-0.1],
            **{f"depth_bid_px_{k}": [100.0 - 0.25 * (k - 1)] for k in range(1, 6)},
            **{f"depth_bid_sz_{k}": [float(k)] for k in range(1, 6)},
            **{f"depth_ask_px_{k}": [100.25 + 0.25 * (k - 1)] for k in range(1, 6)},
            **{f"depth_ask_sz_{k}": [float(k + 10)] for k in range(1, 6)},
        }
    )
    features = pl.DataFrame({AVAILABILITY_TS: [signal_ts]})
    out = _attach_failed_breakout(features, pl.DataFrame(), depth_30m=depth)
    val = out["fb_depth_at_break"][0]
    assert val is None or (isinstance(val, float) and np.isnan(val))


# ─── P2: DepthNoise + bottom_book على المسار الموحّد ─────────────────────────


def test_unified_pipeline_attaches_bottom_book_columns() -> None:
    nq, mnq = _paired_streams(2500, seed=91)
    result = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=100,
        parallel_coverage=False,
        rng=make_generator(91),
        quiet=True,
        config=PipelineConfig(
            interval_ns=10_000,
            n_permutations=100,
            parallel_coverage=False,
            quiet=True,
            filter_depth_noise=True,
            include_bottom_book=True,
            include_failed_breakout=True,
            include_failed_fvg=False,
            include_auction_vp=False,
        ),
    )
    for col in BOTTOM_BOOK_COLUMNS:
        assert col in result.features.columns, col
    assert "fb_depth_at_break" in result.features.columns


def test_build_research_features_can_skip_bottom_book() -> None:
    nq, mnq = _paired_streams(800, seed=92)
    cfg = PipelineConfig(
        interval_ns=10_000,
        quiet=True,
        filter_depth_noise=False,
        include_bottom_book=False,
        include_failed_breakout=False,
        include_failed_fvg=False,
        include_auction_vp=False,
        feature_mode="batch",
    )
    features, _ = _build_research_features(nq, mnq, cfg)
    assert "bb_l2_l5_bid" not in features.columns


# ─── P3: [streaming] micro / research interval في from_toml ──────────────────


def test_pipeline_config_from_toml_streaming_interval_priority(tmp_path: Path) -> None:
    path = tmp_path / "pipe.toml"
    path.write_text(
        """
[temporal]
interval_ns = 2_000_000_000
horizon = 1

[streaming]
research_interval_ns = 500_000_000
micro_interval_ns = 100_000_000
use_micro_interval = false

[depth]
filter_noise = false
include_bottom_book = false
""",
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_toml(path)
    assert cfg.interval_ns == 500_000_000
    assert cfg.research_interval_ns == 500_000_000
    assert cfg.micro_interval_ns == 100_000_000
    assert cfg.use_micro_interval is False
    assert cfg.filter_depth_noise is False
    assert cfg.include_bottom_book is False

    path.write_text(
        """
[temporal]
interval_ns = 2_000_000_000

[streaming]
research_interval_ns = 500_000_000
micro_interval_ns = 100_000_000
use_micro_interval = true
""",
        encoding="utf-8",
    )
    cfg_micro = PipelineConfig.from_toml(path)
    assert cfg_micro.interval_ns == 100_000_000
    assert cfg_micro.use_micro_interval is True


def test_pipeline_config_default_toml_wires_streaming_and_depth() -> None:
    cfg = PipelineConfig.from_toml(Path("configs/default.toml"))
    assert cfg.research_interval_ns == 1_000_000_000
    assert cfg.micro_interval_ns == 100_000_000
    assert cfg.use_micro_interval is False
    assert cfg.interval_ns == 1_000_000_000
    assert cfg.filter_depth_noise is True
    assert cfg.include_bottom_book is True


# ─── P4: fill نبضة فقط + max(TF) صريح ───────────────────────────────────────


def test_resolve_grid_context_interval_mixed_tf() -> None:
    ns_15 = 15 * 60 * 1_000_000_000
    ns_30 = 30 * 60 * 1_000_000_000
    ctx, mixed = resolve_grid_context_interval([ns_15, ns_30], default_ns=ns_30)
    assert ctx == ns_30
    assert mixed is True
    ctx2, mixed2 = resolve_grid_context_interval([ns_15, ns_15], default_ns=ns_30)
    assert ctx2 == ns_15
    assert mixed2 is False


def test_align_horizon_to_context_scales_when_horizon_one() -> None:
    research = 1_000_000_000
    ctx = 30 * 60 * 1_000_000_000
    scaled = align_horizon_to_context(1, research_interval_ns=research, context_interval_ns=ctx)
    kept = align_horizon_to_context(5, research_interval_ns=research, context_interval_ns=ctx)
    assert scaled == 1800
    assert kept == 5


# ─── بقايا الطبقات: أنطولوجيا vp_* + TOML research ─────────────────────────


def test_default_alpha_signals_prefer_vp_ontology_not_streaming_va() -> None:
    assert "vp_balance" in _DEFAULT_SIGNAL_COLUMNS
    assert "in_value_area" not in _DEFAULT_SIGNAL_COLUMNS
    assert "near_vah" not in _DEFAULT_SIGNAL_COLUMNS
    frame = pl.DataFrame(
        {
            "nq_delta": [1.0],
            "vp_balance": [1.0],
            "in_value_area": [1.0],
            "near_vah": [1.0],
            "fail_fvg": [0.0],
        }
    )
    resolved = _resolve_signal_columns(frame, None)
    assert "vp_balance" in resolved
    assert "in_value_area" not in resolved
    assert "near_vah" not in resolved


def test_research_toml_wires_streaming_depth_and_vp_ontology() -> None:
    cfg = PipelineConfig.from_toml(Path("configs/research.toml"))
    assert cfg.research_interval_ns == 1_000_000_000
    assert cfg.micro_interval_ns == 100_000_000
    assert cfg.filter_depth_noise is True
    assert cfg.include_bottom_book is True
    assert cfg.signal_columns is not None
    assert "vp_balance" in cfg.signal_columns
    assert "in_value_area" not in cfg.signal_columns
