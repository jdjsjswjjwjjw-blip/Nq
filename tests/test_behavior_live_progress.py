"""نبض تقدّم مسار فهم سلوك المزاد — لا خطوة ولا حلقة معالجة صامتة."""

from __future__ import annotations

import io

from nq.auction_behavior import BehaviorConfig, run_auction_behavior_analysis
from nq.research.progress import PipelineProgress
from tests.test_auction_behavior import _dense_trade_stream

_EXPECTED_STEPS = (
    "asia_london_projection",
    "auction_action_states",
    "session_summaries",
    "order_flow_scores",
    "level_flow",
    "auction_signals_from_states",
    "join_flow_and_projection",
    "behavior_events",
    "quality_structure_memory",
    "base_rate_probabilities",
    "behavior_science",
    "validate_behavior_frame",
)

_EXPECTED_OPS = (
    "build_asia_london_projection",
    "score_deceptive_events",
    "attach_reliability_evidence",
    "level_flow",
    "auction_signals_from_states",
    "build_behavior_events",
    "attach_market_memory",
    "attach_sequence_memory",
    "estimate_behavior_probabilities",
    "run_behavior_science",
    "validate_behavior_frame",
)


def test_behavior_pipeline_prints_every_step_and_processing_pulse() -> None:
    frame = _dense_trade_stream(n_bars=48, bar_ns=200)
    buf = io.StringIO()
    progress = PipelineProgress(enabled=True, stream=buf, heartbeat_seconds=0.0)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=True,
            include_level_flow=True,
            include_reliability_evidence=True,
            include_science=True,
            n_splits=3,
            min_train_size=6,
            quiet=False,
        ),
        progress=progress,
        quiet=False,
    )
    text = buf.getvalue()
    assert "[nq] ========== بدء: auction_behavior ==========" in text
    for step in _EXPECTED_STEPS:
        assert step in text, step
    for op in _EXPECTED_OPS:
        assert op in text, op
    assert "    - " in text
    assert "  … " in text
    assert "انتهى بنجاح: auction_behavior" in text
    assert result.diagnostics["deceptive_filtered"] is False
    assert text.index("asia_london_projection") < text.index("behavior_science")
    assert text.index("behavior_events") < text.index("validate_behavior_frame")


def test_behavior_pipeline_quiet_prints_nothing() -> None:
    frame = _dense_trade_stream(n_bars=24, bar_ns=200)
    buf = io.StringIO()
    run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            include_science=False,
            quiet=True,
        ),
        progress=PipelineProgress(enabled=True, stream=buf),
        quiet=True,
    )
    assert buf.getvalue() == ""
