# F-006-F-014 Quantitative Remediation Report

Date: 2026-07-23

Scope: continued remediation from `docs/deep_leakage_and_integrity_audit.md` after
`docs/f001_f005_remediation_report.md`. F-001-F-005 were not revisited except where
full-suite verification exposed an unrelated numerical invariant in preprocessing.

## Red-Test Evidence

- Initial targeted regressions failed as expected:
  - `tests/test_alpha.py` could not import `block_permutation`.
  - `tests/test_execution.py` could not import `realistic_execution_forward_returns`.
  - Remaining targeted files showed 9 failures: missing `strict=True` reconstruction,
    unknown modify creating liquidity, Databento sequence tied to row order, bucket-local VP,
    cross-session cumulative delta, SSL future-missingness feature selection, missing bearish
    trap low-side state, and ambiguous multi-instrument FeatureStore snapshots.
- Additional F-011 duplicate-order regression failed before fix:
  - `tests/test_reconstruction.py::test_duplicate_add_does_not_duplicate_resting_liquidity`
    showed `unknown_order_refs == 0` for a duplicate `ADD` order id.

## Verification

- `pytest -ra`: PASS, `286 passed in 65.65s`.
- `pytest -m leakage`: PASS, `5 passed, 281 deselected in 1.21s`.
- `pytest --cov --cov-report=term-missing`: PASS, `286 passed`, total coverage `83%`.
- `ruff check src tests`: FAIL, 37 existing lint findings left untouched.
- `mypy`: FAIL, 91 existing type findings in 12 files left untouched.

## F-011 - Strict MBO / Order-Book Reconstruction Integrity

status: FIXED

root cause: `OrderBook.apply` accepted unknown `MODIFY` as new resting liquidity and duplicate
`ADD` could overwrite an order id while duplicating depth. `reconstruct` had no strict
research/backtest mode, and `IntegrityReport.ok` did not reject sequence skips or crossed books.

files changed:
- `src/nq/orderbook/book.py`: duplicate `ADD` and unknown `MODIFY` now increment
  `unknown_order_refs` and do not mutate depth.
- `src/nq/orderbook/integrity.py`: added `strict_ok` and `strict_failures`.
- `src/nq/orderbook/reconstruction.py`: added `StrictReconstructionError` and `strict=True`.
- `src/nq/orderbook/__init__.py`: exports strict error.
- `tests/test_reconstruction.py`, `tests/test_book.py`.

regression test:
- `tests/test_reconstruction.py::test_strict_reconstruction_rejects_crossed_book`
- `tests/test_reconstruction.py::test_strict_reconstruction_rejects_sequence_skips`
- `tests/test_reconstruction.py::test_unknown_modify_does_not_create_resting_liquidity`
- `tests/test_reconstruction.py::test_duplicate_add_does_not_duplicate_resting_liquidity`

before behavior: crossed books and sequence skips were reported but permissive; strict mode did
not exist; unknown modify/duplicate add could create artificial liquidity.

after behavior: strict reconstruction rejects non-monotonic/unknown/skipped/crossed streams;
unknown modify and duplicate add no longer create resting liquidity.

remaining limitations: strict mode is opt-in, so exploratory callers must request `strict=True`
for hard rejection; permissive mode still returns integrity counters.

## F-012 - Duplicate Timestamps And Deterministic Event Ordering

status: FIXED

root cause: Databento frames without native `sequence` received synthetic sequence numbers in
input row order. Generic causal sorting used only `(event_ts, sequence)`.

files changed:
- `src/nq/ingestion/databento.py`: synthetic sequence now follows source metadata order.
- `src/nq/core/time.py`: causal sort uses available tie-breakers after `(event_ts, sequence)`.
- `src/nq/models/tick_stream.py`: combined NQ/MNQ stream uses `sort_causal`.
- `tests/test_databento.py`.

regression test:
- `tests/test_databento.py::test_databento_synthesized_sequence_uses_source_metadata_not_row_order`

before behavior: reversing identical Databento rows changed synthetic sequence order.

after behavior: synthetic sequence is deterministic from `event_ts`, `ingest_ts`, publisher,
instrument, order/action/side/price/size/flags metadata, with original row only as final tie.

remaining limitations: if all source metadata is truly identical, stable input row order remains
the unavoidable last tie-breaker.

## F-007 - Session / Contract State Reset Semantics

status: PARTIALLY FIXED

root cause: several cumulative/rolling state features accumulated across CME session boundaries.

files changed:
- `src/nq/simulation/order_flow.py`: bucket and event OFI cumulatives reset by
  `trading_session_id`.
- `src/nq/simulation/footprint.py`: footprint cumulative delta resets by session.
- `src/nq/simulation/breakout.py`: Failed Breakout rolling baselines and cumulative delta reset
  by session.
- `src/nq/simulation/auction.py`: auction previous-range/high/low and flip state reset by session.
- `tests/test_order_flow.py`.

regression test:
- `tests/test_order_flow.py::test_order_flow_cumulative_delta_resets_at_cme_session_boundary`

before behavior: two trades across 16:59 ET and 18:01 ET produced cumulative delta `[5, 12]`.

after behavior: the same stream produces `[5, 7]`, proving CME session reset.

remaining limitations: contract-roll resets need explicit contract/roll-calendar metadata beyond
the available `instrument_id`/session fields.

## F-008 - Volume Profile Semantic Correctness

status: PARTIALLY FIXED

root cause: `developing_value_area` computed independent bucket-local profiles rather than a
session-developing profile available at each bucket close.

files changed:
- `src/nq/simulation/volume_profile.py`: maintains a `DevelopingVolumeProfile` per
  `trading_session_id` and resets POC migration per session.
- `src/nq/simulation/auction.py`: consumes the session-aware VP output.
- `tests/test_volume_profile.py`.

regression test:
- `tests/test_volume_profile.py::test_developing_value_area_accumulates_within_session`

before behavior: after heavy volume at 100 and one later trade at 110, bucket 2 reported POC 110.

after behavior: bucket 2 reports POC 100 because the session-developing profile still has dominant
volume at 100.

remaining limitations: VP near-price threshold still uses the existing local fixed scale constant;
per-contract tick-size parameterization remains future work.

## F-009 - SSL Full-Frame Feature-Selection Leakage

status: FIXED

root cause: `_feature_columns` filtered columns by full-frame null fraction before walk-forward
folds, so future non-null periods could decide whether a column existed for earlier folds.

files changed:
- `src/nq/models/ssl_pipeline.py`: automatic selection now includes numeric columns without
full-frame missingness filtering; fold-local scaler/PCA remain causal.
- `src/nq/models/preprocessing.py`: added a tiny relative variance floor for near-constant
training columns found by invariant testing.
- `tests/test_models_ssl.py`, `tests/test_invariants.py`.

regression test:
- `tests/test_models_ssl.py::test_ssl_feature_selection_is_invariant_to_future_missingness`
- existing `tests/test_invariants.py::test_causal_scaler_zero_mean`

before behavior: changing only future null/non-null values changed selected feature columns.

after behavior: selected columns are invariant to future missingness; null imputation occurs after
selection, and scaling remains fit only on training folds.

remaining limitations: if a caller explicitly supplies `feature_columns` selected by a leaky
external process, the pipeline cannot prove that provenance.

## F-010 - IID Permutation / Multiple-Testing Bias

status: FIXED

root cause: alpha IC and structural coverage nulls permuted individual rows despite time-series
autocorrelation.

files changed:
- `src/nq/statistics/resampling.py`: added `block_permutation`.
- `src/nq/statistics/__init__.py`: exports `block_permutation`.
- `src/nq/alpha/signals.py`: `evaluate_signal` and intraday evaluation use block permutation.
- `src/nq/coverage/metrics.py`: MFIG, CER, and QDUF use block permutation nulls.
- `tests/test_alpha.py`.

regression test:
- `tests/test_alpha.py::test_block_permutation_preserves_contiguous_time_blocks`
- `tests/test_alpha.py::test_evaluate_signal_supports_block_permutation_null`

before behavior: permutation nulls destroyed all local temporal dependence.

after behavior: affected research metrics permute contiguous blocks; BH multiple-testing screen
remains in place.

remaining limitations: generic `permutation_test` remains available for genuinely independent
two-sample tests; callers must choose block-aware tests for autocorrelated market series.

## F-013 - Realistic Causal Execution

status: PARTIALLY FIXED

root cause: execution labels entered at the same row that generated the decision and depth fills
used current-row depth as immediately fillable.

files changed:
- `src/nq/simulation/execution/intraday.py`: added
  `realistic_execution_forward_returns` with decision -> latency -> entry -> exit timeline.
- `src/nq/simulation/execution/depth_fill.py`: depth fills now honor `latency_steps`.
- `src/nq/simulation/execution/__init__.py`, `src/nq/simulation/__init__.py`: exports.
- `src/nq/alpha/signals.py`: intraday alpha evaluation uses realistic execution by default.
- `tests/test_execution.py`.

regression test:
- `tests/test_execution.py::test_realistic_execution_applies_latency_before_entry`

before behavior: a decision at `t` could fill at `ask[t]` / `bid[t]`.

after behavior: default intraday evaluation fills no earlier than `t + latency_steps`; old
`execution_forward_returns` is documented as a research forward-return label path.

remaining limitations: queue priority and partial-fill simulation are still not modeled; thin
depth can reject via `NaN`, but there is no order-resting lifecycle.

## F-014 - Bearish Trap Logic

status: FIXED

root cause: `_trap_setup` only tracked highs, making the bearish branch compare bearish movement
against prior highs rather than prior lows.

files changed:
- `src/nq/models/tick_stream.py`: tracks `nq_low`/`mnq_low` per CME session and uses lower-low
  nonconfirmation for bearish traps.
- `tests/test_tick_stream.py`.

regression test:
- `tests/test_tick_stream.py::test_bearish_trap_uses_lower_low_nonconfirmation`

before behavior: bearish trap logic was effectively unreachable or semantically inverted.

after behavior: bearish trap fires when MNQ breaks a prior low on negative delta while NQ does not
confirm with a lower low.

remaining limitations: trap threshold remains a simple signed-volume threshold, not a calibrated
microstructure model.

## F-006 - Multi-Instrument FeatureStore Isolation

status: FIXED

root cause: `snapshot_series` pivoted by `feature` only, so same-named features for multiple
instruments collapsed into one wide column.

files changed:
- `src/nq/features/store.py`: ambiguous multi-instrument snapshots require explicit
  `instrument_id`.
- `tests/test_feature_store.py`.

regression test:
- `tests/test_feature_store.py::test_snapshot_requires_instrument_scope_for_ambiguous_feature_names`

before behavior: NQ/MNQ same-named features could silently overwrite/aggregate via pivot.

after behavior: ambiguous `snapshot_series` and `point_in_time_join` raise until scoped by
`instrument_id`; scoped snapshots retain the correct instrument value.

remaining limitations: cross-instrument composite features should still use explicit feature names
or `instrument_id=0` by producer convention.

## Additional Verification-Driven Repair

status: FIXED

root cause: full-suite Hypothesis testing found near-constant scaler columns could amplify
floating-point noise and violate the zero-mean invariant.

files changed:
- `src/nq/models/preprocessing.py`.

regression test:
- `tests/test_invariants.py::test_causal_scaler_zero_mean`

before behavior: a generated near-constant matrix produced transformed column mean `-1.0`.

after behavior: effectively constant columns use a relative variance floor and transform stably.

remaining limitations: none identified for the invariant; this is numerical hygiene, not parameter
tuning.

## Remaining Tooling Debt Not Fixed

`ruff check src tests` still fails with existing lint/style debt, including:
- `src/nq/orderbook/depth.py` redundant `int(round(...))`.
- `src/nq/research/progress.py` magic-number / comparison style findings.
- `src/nq/research/orchestrator.py`, `src/nq/simulation/breakout.py`, and strategy search modules
  complexity thresholds.
- `tests/test_breakout.py`, `tests/test_pipeline_progress.py`, and
  `tests/test_ssl_enhancements.py` unused imports/local imports/line-length/name style findings.

`mypy` still fails with existing type debt, mostly:
- progress objects typed as `object` while calling `.op()` / `.heartbeat()`.
- unused or mismatched `type: ignore[union-attr]` comments.
- missing generic args and several test typing issues.

These were not remediated because the request explicitly said not to fix unrelated lint/type debt.
