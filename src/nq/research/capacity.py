"""Capacity-correct processing defaults (useful work within resource limits).

Principle
---------
Correct causal protocol on a **bounded** sample beats brute-force volume.
These knobs do **not** weaken PIT / purged walk-forward / asof-backward rules;
they only avoid wasted permutations and oversized candidate grids.
"""

from __future__ import annotations

from typing import Final

# Recommended MBO row cap for interactive / search runs (not a hard engine limit).
RECOMMENDED_MAX_ROWS: Final[int] = 500_000

# Hypothesis search: full permutation tests only on pooled OOS (selection uses IC).
SEARCH_N_PERMUTATIONS: Final[int] = 100

# Post-hoc understanding diagnostics (never alters best_oos_spec).
UNDERSTAND_N_PERMUTATIONS: Final[int] = 50

# Lean depth / SSL enhancement quantiles (past-only thresholds unchanged).
LEAN_GATE_QUANTILES: Final[tuple[float, ...]] = (0.7,)
LEAN_SSL_Z_COLS: Final[tuple[str, ...]] = ("z0",)

# Unified pipeline formal report (configs) — keep stricter when not searching.
UNIFIED_N_PERMUTATIONS: Final[int] = 200

__all__ = [
    "LEAN_GATE_QUANTILES",
    "LEAN_SSL_Z_COLS",
    "RECOMMENDED_MAX_ROWS",
    "SEARCH_N_PERMUTATIONS",
    "UNDERSTAND_N_PERMUTATIONS",
    "UNIFIED_N_PERMUTATIONS",
]
