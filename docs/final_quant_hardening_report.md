# Final Quant Hardening Report

Date: 2026-07-23

Scope: final remediation of the remaining partially-fixed quantitative risks from
`docs/f006_f014_remediation_report.md`: F-007, F-008, and F-013 only. No project
redesign, profitability optimization, or strategy-parameter tuning was performed.

## Red-Test Evidence

Initial targeted regressions were added before production changes and failed as expected:

- `tests/test_volume_profile.py` failed during collection because
  `nq.contracts.instruments` did not exist.
- `tests/test_execution.py` failed during collection because
  `realistic_depth_execution_simulation` was not exported.
- After implementation, the targeted suite passed: `42 passed`.

## F-007 - Contract / Session Reset Semantics

status: FIXED

root cause:

- Stateful processors reset on CME trading-session changes but did not explicitly audit
  contract identity changes.
- `build_tick_stream` overwrote source `instrument_id` with canonical NQ/MNQ stream IDs
  before lifecycle tracking, so source instrument changes could be hidden.
- Volume Profile, order-flow cumulative delta, depth state, and strict reconstruction could
  process mixed contract identities unless the caller manually separated the data.

files changed:

- `src/nq/contracts/instruments.py`
- `src/nq/contracts/__init__.py`
- `src/nq/models/tick_stream.py`
- `src/nq/orderbook/reconstruction.py`
- `src/nq/simulation/order_flow.py`
- `src/nq/simulation/depth_lifecycle.py`
- `src/nq/simulation/volume_profile.py`
- `tests/test_tick_stream.py`
- `tests/test_reconstruction.py`
- `tests/test_order_flow.py`

regression tests:

- `tests/test_tick_stream.py::test_tick_stream_rejects_contract_roll_without_explicit_config`
- `tests/test_tick_stream.py::test_tick_stream_resets_books_and_state_on_explicit_contract_roll`
- `tests/test_reconstruction.py::test_strict_reconstruction_rejects_contract_identity_change`
- `tests/test_order_flow.py::test_order_flow_rejects_contract_roll_without_explicit_lifecycle_config`

before behavior:

- Contract rolls could be accepted silently in tick-stream and aggregate state.
- Old order-book/VP/regime/trap state could carry into a new contract.

after behavior:

- Mixed contract identities now fail safely by default in stateful VP, order-flow, depth,
  and strict reconstruction paths.
- `build_tick_stream(..., allow_contract_roll=True)` performs an explicit lifecycle reset:
  NQ/MNQ books, developing VP, regime tracker, MNQ cumulative signed volume, session highs/lows,
  and previous NQ mid are reset at source contract identity changes.
- Strict reconstruction rejects contract identity changes in research/backtest mode.

remaining limitations:

- No roll calendar is invented. Unsupported automatic roll decisions require explicit data or
  configuration.
- Order books reset on explicit contract identity changes and MBO book events, not on every
  session boundary absent exchange reset data.

## F-008 - Instrument Metadata / Volume Profile Tick Semantics

status: FIXED

root cause:

- Volume Profile near-level logic used a local fixed-scale approximation instead of the
  repository price scale and contract tick size.
- NQ/MNQ tick size, point value, and price-scale semantics were not centralized.

files changed:

- `src/nq/contracts/instruments.py`
- `src/nq/contracts/__init__.py`
- `src/nq/simulation/volume_profile.py`
- `src/nq/models/tick_stream.py`
- `src/nq/simulation/execution/costs.py`
- `src/nq/simulation/execution/spread.py`
- `src/nq/simulation/execution/intraday.py`
- `src/nq/simulation/execution/depth_fill.py`
- `src/nq/alpha/signals.py`
- `src/nq/alpha/discovery.py`
- `src/nq/research/orchestrator.py`
- `tests/test_volume_profile.py`

regression tests:

- `tests/test_volume_profile.py::test_developing_volume_profile_near_levels_use_nq_tick_size`
- `tests/test_volume_profile.py::test_instrument_metadata_distinguishes_nq_and_mnq_contract_specs`

before behavior:

- A valid two-tick NQ/MNQ distance could be classified as not near VAH/VAL because the
  threshold used a local approximate constant.

after behavior:

- Central metadata supports NQ and MNQ root/contract parsing, tick size, point value,
  fixed-price tick size, price scale, and contract identity.
- VP near-level logic uses `tick_size_fixed` from metadata.
- Execution/research tick-size defaults now inherit from `NQ_METADATA.tick_size` instead of
  scattered `0.25` literals.

remaining limitations:

- Only NQ/MNQ are supported. Other symbols fail with an explicit metadata error until configured.

## F-013 - Realistic Causal Execution

status: FIXED

root cause:

- Realistic execution returned only forward-return arrays and did not expose fill timestamps,
  filled quantity, or rejection state.
- Depth execution could use L1 fallback when visible depth was insufficient, hiding liquidity
  failures.
- Partial fills were not reported.

files changed:

- `src/nq/simulation/execution/intraday.py`
- `src/nq/simulation/execution/depth_fill.py`
- `src/nq/simulation/execution/__init__.py`
- `src/nq/simulation/__init__.py`
- `tests/test_execution.py`

regression tests:

- `tests/test_execution.py::test_realistic_execution_reports_fill_timestamps_and_order_size`
- `tests/test_execution.py::test_depth_execution_rejects_insufficient_liquidity_even_with_l1_fallback_by_default`
- `tests/test_execution.py::test_depth_execution_allows_partial_fills_with_reported_qty_and_timestamps`

before behavior:

- Decision-row labels and realistic execution were separated only at the return-array level.
- There was no explicit fill timestamp or filled quantity report.
- Thin visible depth could be hidden by fallback bid/ask fills.

after behavior:

- Added `realistic_execution_simulation` for L1 causal market-order simulation with explicit
  decision -> latency -> eligible fill timing, order size, bid/ask crossing, fill timestamps,
  filled quantity, rejection flags, slippage, and commission bps.
- Added `realistic_depth_execution_simulation` for visible-depth walking with size-aware VWAP,
  partial-entry handling, insufficient-liquidity rejection, fill timestamps, and filled quantity.
- `execution_forward_returns_depth` now rejects insufficient visible liquidity by default even
  when fallback quotes are supplied; L1 fallback requires explicit opt-in.
- Existing research forward-return labels remain available as separate APIs.

remaining limitations:

- Queue position, order resting, cancels after submission, and exchange-specific fee schedules
  are not simulated.
- Commission is modeled as `commission_bps`, not a per-contract fee schedule.

## Verification

- Targeted regressions: PASS, `42 passed`.
- Adjacent affected suites: PASS, `63 passed`.
- `pytest -ra`: PASS, `295 passed in 94.32s`.
- `pytest -m leakage`: PASS, `5 passed, 290 deselected in 1.21s`.
- `pytest --cov --cov-report=term-missing`: PASS, `295 passed in 130.56s`.
- Coverage: total `83%`.

