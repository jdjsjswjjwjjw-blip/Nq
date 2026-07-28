"""اختبارات المعالجة الصحيحة ضمن القدرة (capacity-correct)."""

from __future__ import annotations

from nq.research.capacity import (
    LEAN_GATE_QUANTILES,
    RECOMMENDED_MAX_ROWS,
    SEARCH_N_PERMUTATIONS,
    UNDERSTAND_N_PERMUTATIONS,
)
from nq.strategies.breakout_hypothesis import core_breakout_grid
from nq.strategies.fvg_hypothesis import core_fvg_grid, default_fvg_grid


def test_capacity_constants_are_lean() -> None:
    assert RECOMMENDED_MAX_ROWS == 500_000
    assert SEARCH_N_PERMUTATIONS <= 100
    assert UNDERSTAND_N_PERMUTATIONS <= SEARCH_N_PERMUTATIONS
    assert LEAN_GATE_QUANTILES == (0.7,)


def test_core_grids_smaller_than_full() -> None:
    assert 8 <= len(core_breakout_grid()) <= 12
    assert 12 <= len(core_fvg_grid()) <= 24
    assert len(core_fvg_grid()) < len(default_fvg_grid())
