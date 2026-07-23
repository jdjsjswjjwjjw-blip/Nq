"""اختبارات السياسة الزمنية (Temporal Policy)."""

from __future__ import annotations

import numpy as np

from nq.core.temporal_policy import TemporalPolicy


def test_purge_samples_for_window() -> None:
    policy = TemporalPolicy(window=5, stride=1)
    assert policy.purge_samples() == 4


def test_embargo_uses_ns_for_production_times() -> None:
    policy = TemporalPolicy(embargo_ns=1_000_000_000, window=5, stride=1)
    times = np.array([1_700_000_000_000_000_000, 1_700_000_000_001_000_000], dtype=np.int64)
    embargo = policy.embargo_time_units(interval_ns=1_000_000_000, times=times)
    assert embargo >= policy.embargo_ns


def test_embargo_scales_down_for_test_times() -> None:
    policy = TemporalPolicy(embargo_ns=1_000_000_000, window=5, stride=1)
    times = np.arange(100, dtype=np.int64)
    embargo = policy.embargo_time_units(interval_ns=10, times=times)
    assert embargo < policy.embargo_ns
    assert embargo == (5 - 1) * 10 + 10
