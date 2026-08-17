"""تسلسل MBO داخل البرميل: الترتيب يحمل ما تفقده مجاميع 30ث."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

import nq.research
from nq.core.determinism import make_generator
from nq.research.mbo_sequence_mlp import (
    BIN_NS,
    HOLDOUT_START_DATE,
    N_BINS,
    Y_TARGET,
    assert_single_day_mbo,
    build_intra_bar_sequence,
    collapse_sequence,
    fit_predict_logistic,
    fit_predict_mlp,
    write_mbo_sequence_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")


def _setup_ts() -> int:
    stamp = dt.datetime(2025, 6, 3, 4, 0, 30, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "run_mbo_sequence_mlp" not in nq.research.__all__
    assert not hasattr(nq.research, "run_mbo_sequence_mlp")


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 4, 4, 0, tzinfo=_ET)
    ts = [
        int(day_a.timestamp() * 1_000_000_000),
        int(day_b.timestamp() * 1_000_000_000),
    ]
    mbo = make_stream(
        [("A", "B", 20_000_000_000, 1, 1), ("C", "N", 0, 0, 1)],
        event_ts=ts,
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)


def test_ignores_events_after_setup() -> None:
    t = _setup_ts()
    mbo = make_stream(
        [
            ("A", "B", 20_000_000_000, 2, 1),
            ("C", "N", 0, 1, 1),
        ],
        event_ts=[t - BIN_NS, t + BIN_NS],
    )
    seq = build_intra_bar_sequence(mbo, t)
    assert seq.shape == (N_BINS, 8)
    assert float(seq[:, 1].sum()) == 0.0
    assert float(seq[:, 0].sum()) == 1.0


def test_mlp_reads_cancel_order_that_aggregates_miss() -> None:
    rng = make_generator(3)
    n_each = 20
    x = np.zeros((n_each * 2, N_BINS, 8), dtype=np.float64)
    y = np.zeros(n_each * 2, dtype=np.float64)
    for i in range(n_each):
        x[i, -5:, 1] = 2.0
        y[i] = 1.0
        j = n_each + i
        x[j, :5, 1] = 2.0
        y[j] = 0.0
    train = np.arange(0, n_each * 2, 2)
    test = np.arange(1, n_each * 2, 2)
    p_agg = fit_predict_logistic(collapse_sequence(x), y, train)[test]
    p_mlp = fit_predict_mlp(x, y, train, rng=rng, epochs=25)[test]
    agg_spread = float(np.max(p_agg) - np.min(p_agg))
    pos = p_mlp[y[test] > 0.5]
    neg = p_mlp[y[test] < 0.5]
    assert agg_spread < 0.05
    assert float(np.min(pos) - np.max(neg)) > 0.05
    assert Y_TARGET == "y_phase_extend"
    assert HOLDOUT_START_DATE == "2025-09-01"


def test_report_writes(tmp_path: Path) -> None:
    scored = pl.DataFrame(
        {
            "setup_availability_ts": [1],
            "y": [1.0],
            "p_aggregate": [0.4],
            "p_mlp": [0.7],
            "fold": [0],
        }
    )
    diag = {
        "aggregate": {"n": 1.0, "auc": 0.5, "brier_skill": 0.0},
        "mlp": {"n": 1.0, "auc": 0.7, "brier_skill": 0.1},
    }
    out = write_mbo_sequence_report(scored, diag, tmp_path)
    text = (out / "MBO_SEQUENCE.md").read_text(encoding="utf-8")
    assert "mlp sequence" in text
