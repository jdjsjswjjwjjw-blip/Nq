# Phase 18 Master Quant Audit

Date: 2026-07-23

Repository path: `/Users/abdallah/Desktop/Trading/Nq-main`

This report separates verified implementation bugs from risks requiring more market
data, exchange/channel metadata, or broader empirical validation.

## Summary

| Bucket | Count |
| --- | ---: |
| CRITICAL verified bugs | 1 |
| HIGH verified bugs | 2 |
| MEDIUM verified bugs | 0 |
| HIGH risks requiring evidence | 2 |
| MEDIUM risks requiring evidence | 5 |
| LOW risks requiring evidence | 1 |

## Verified Bugs

### QF-001

| Field | Value |
| --- | --- |
| Severity | CRITICAL |
| Category | Hypothesis selection / validation |
| File | `src/nq/strategies/fvg_hypothesis.py` |
| Function/Class | `walk_forward_select_hypotheses` |
| Exact lines | 278-292 |
| Evidence | `best_ic` is initialized to `-1e18`; the update condition compares `abs(ic) > abs(best_ic)`. No finite IC can exceed `1e18`, so `best_name` remains `cols[0]` for every fold. |
| Why it is wrong | Candidate selection is advertised as train-fold IC selection, but the implementation deterministically selects the first candidate. |
| Quantitative consequence | Failed FVG and Failed Breakout OOS results can be materially wrong because the reported selected hypothesis is list-order dependent. |
| Leakage consequence | Not direct look-ahead leakage, but it invalidates walk-forward model-selection claims and can hide leakage by never selecting the actual train-best candidate. |
| Recommended fix | Initialize with `None` or evaluate the first candidate before comparisons. Preserve deterministic tie-breaking. |
| Required regression test | Synthetic candidate frame where the second candidate has materially higher train IC than the first and must be selected in every fold. |

Affected callers:

- `src/nq/strategies/fvg_hypothesis.py:489-500`
- `src/nq/strategies/breakout_hypothesis.py:506-517`

### QF-002

| Field | Value |
| --- | --- |
| Severity | HIGH |
| Category | Target leakage / walk-forward purge |
| File | `src/nq/strategies/fvg_hypothesis.py`, `src/nq/models/splitting.py`, `src/nq/core/temporal_policy.py` |
| Function/Class | `walk_forward_select_hypotheses`, `purged_walk_forward_split`, `TemporalPolicy` |
| Exact lines | `fvg_hypothesis.py:240`, `fvg_hypothesis.py:257-263`, `splitting.py:71-76`, `temporal_policy.py:70-74` |
| Evidence | `align_forward_returns` creates labels using `price[t+horizon]`, but the split only receives timestamp embargo and `purge_samples`. The splitter has no explicit `label_horizon`, and `TemporalPolicy.purge_samples()` only accounts for overlapping SSL windows. |
| Why it is wrong | Training rows near the test boundary can remain in train even when their label consumes a future price inside the test block if `horizon > purge_samples`. |
| Quantitative consequence | Train-fold IC can be inflated and candidate choice can be optimized using test-period prices. |
| Leakage consequence | Direct target leakage across the train/test boundary for configurable horizons. |
| Recommended fix | Add horizon-aware purge semantics to the splitter, or pass an effective label purge derived from `horizon` into all forward-return hypothesis-selection calls. |
| Required regression test | A split with `horizon=H` must prove `max(train_idx + H) < min(test_idx)` for every accepted fold. |

### QF-003

| Field | Value |
| --- | --- |
| Severity | HIGH |
| Category | Databento MBO semantics / order-book reconstruction |
| File | `src/nq/orderbook/book.py` |
| Function/Class | `OrderBook.apply` |
| Exact lines | 67-87 |
| Evidence | The `CANCEL` branch pops the entire order and removes full remaining size; the `FILL` branch reduces book state. Databento's official order-tracking and limit-order-book examples describe `C` as partial or full cancellation by event size, while `T`, `F`, and `N` do not update resting book state because the book update is carried by other records. Sources: https://databento.com/docs/examples/order-book/order-tracking and https://databento.com/docs/examples/order-book/limit-order-book |
| Why it is wrong | Partial cancel events are over-applied, and fill events can be double-counted against book depth when paired with the corresponding book-update event. |
| Quantitative consequence | Top-of-book depth, cumulative depth, queue depth, depth imbalance, VP-at-depth features, and depth-walk execution labels can be materially distorted. |
| Leakage consequence | Not a temporal leak, but incorrect state mutation corrupts all point-in-time depth features and execution realism. |
| Recommended fix | Apply cancel size against the existing order and level, remove only when remaining size is zero, and treat `F` as a Databento no-op for resting book state. Invalid over-cancels should fail loudly. |
| Required regression test | Partial cancel preserves the residual order; full cancel removes it; fill does not mutate the book; fill followed by the corresponding update does not double-remove depth. |

## Risks Requiring Further Evidence

### QR-004

| Field | Value |
| --- | --- |
| Severity | HIGH |
| Category | Duplicate timestamp and multi-stream ordering |
| File | `src/nq/core/time.py`, `src/nq/ingestion/databento.py`, `src/nq/models/tick_stream.py` |
| Function/Class | `sort_causal`, `normalize_databento_frame`, `build_tick_stream` |
| Exact lines | `time.py:15-17`, `databento.py:111-114`, `tick_stream.py:325-328` |
| Evidence | Canonical local ordering is `(event_ts, sequence)`. If `sequence` is absent, normalization creates one from current row order. Tick-stream construction overwrites instrument IDs and then sorts concatenated NQ/MNQ by only `event_ts, sequence`. |
| Why it may be wrong | MBO can have duplicate timestamps, and Databento provides `publisher_id`, `channel_id`, `ts_recv`, and venue sequence semantics that are not preserved in the internal contract. |
| Quantitative consequence | Same-timestamp NQ/MNQ state can be ordered arbitrarily, altering cross-market traps, SSL tick windows, and depth snapshots. |
| Leakage consequence | Boundary leakage can occur when contemporaneous train/test events with the same timestamp are split by index without a deterministic full ordering or timestamp-boundary guard. |
| Recommended fix | Preserve source ordering metadata, require real sequence for Databento MBO when possible, and add duplicate timestamp tests for same-instrument and cross-instrument streams. |
| Required regression test | Duplicate `event_ts` with same or missing sequence must produce deterministic, documented output and must not split same-time events across train/test when causality requires grouping. |

### QR-005

| Field | Value |
| --- | --- |
| Severity | HIGH |
| Category | Session and futures state |
| File | `src/nq/models/tick_stream.py`, `src/nq/core/session.py` |
| Function/Class | `build_tick_stream`, `_tick_row`, session utilities |
| Exact lines | `tick_stream.py:331-336`, `tick_stream.py:372-392`, `session.py:41-77` |
| Evidence | `DevelopingVolumeProfile`, `CausalRegimeTracker`, `nq_high`, `mnq_high`, `mnq_low`, and `mnq_signed` are initialized once and carried through the whole stream. Session utilities exist, but tick-stream state does not reset by session or contract. |
| Why it may be wrong | Many futures features should reset at session, RTH/ETH boundary, UTC snapshot boundary, or contract roll. |
| Quantitative consequence | Prior-session volume profile, highs/lows, signed volume, and regimes can influence current-session signals. |
| Leakage consequence | If a feature is intended to be session-developing, carrying final prior-session state is not look-ahead, but it is a state-definition violation and can create false research claims. |
| Recommended fix | Define state reset policy per feature: continuous ETH, RTH session, calendar session, or contract. Make it config-driven and test it. |
| Required regression test | Two-session fixture where session-reset features at the second session open do not contain first-session VP/high/low/delta state. |

### QR-006

| Field | Value |
| --- | --- |
| Severity | MEDIUM |
| Category | SSL preprocessing |
| File | `src/nq/models/ssl_pipeline.py` |
| Function/Class | `_feature_columns`, `run_ssl_pipeline` |
| Exact lines | 153-170, 222-239 |
| Evidence | Feature-column selection uses full-frame null fraction before splitting, and selected feature nulls are filled with zero before sequence construction. Fold scaler and PCA are train-only, but feature inclusion is not. |
| Why it may be wrong | Future missingness can influence which columns are trained and tested. Zero-fill can also encode structural missingness as a numeric value without an explicit mask. |
| Quantitative consequence | SSL metrics and embeddings may depend on full-period feature availability patterns. |
| Leakage consequence | Potential low-to-medium selection leakage through full-frame preprocessing. |
| Recommended fix | Select feature columns inside folds or freeze a declared feature set from config; add missingness indicators where zero-fill is intended. |
| Required regression test | Future-only null perturbation must not change selected columns or historical embeddings for timestamps before the perturbation. |

### QR-007

| Field | Value |
| --- | --- |
| Severity | MEDIUM |
| Category | Volume Profile semantics |
| File | `src/nq/simulation/volume_profile.py`, `src/nq/simulation/auction.py` |
| Function/Class | `developing_value_area`, `DevelopingVolumeProfile`, `auction_states` |
| Exact lines | `volume_profile.py:24-25`, `volume_profile.py:96-146`, `volume_profile.py:162-223`, `auction.py:46-77` |
| Evidence | `DevelopingVolumeProfile` is event-by-event cumulative, but `developing_value_area` computes a separate value area per bucket. `_PRICE_SCALE = 1_000_000` is an approximate fixed-point dollar step, not centralized contract metadata. |
| Why it may be wrong | A bucket-local value area is not the same as a developing session POC/VAH/VAL. NQ tick-size and point-value metadata should be exact and centralized. |
| Quantitative consequence | Auction balance/imbalance and VP migration can be interpreted as session-developing when they are actually bucket-local. |
| Leakage consequence | No direct future leakage observed because outputs are bucket-close available, but semantic mismatch can invalidate strategy interpretation. |
| Recommended fix | Rename bucket-local functions or implement a true session-developing profile keyed by session and exact tick metadata. |
| Required regression test | A multi-bucket fixture where session-developing POC differs from bucket-local POC and each function returns the documented one. |

### QR-008

| Field | Value |
| --- | --- |
| Severity | MEDIUM |
| Category | Cross-market availability |
| File | `src/nq/simulation/cross_market.py` |
| Function/Class | `_align_markets`, `cross_market_features` |
| Exact lines | 78-113, 137-205 |
| Evidence | With positive `latency_ns`, NQ rows align to MNQ at `bucket_start - latency_ns` by backward asof. Otherwise, NQ/MNQ are inner-joined on bucket start. The contract uses `event_ts`, not `ts_recv`, as feature availability. |
| Why it may be wrong | Availability for live/replay research may need `ts_recv` or a feed-latency model, not just exchange event time. |
| Quantitative consequence | Lead/lag and trap features can be optimistic or overly conservative depending on the intended trading clock. |
| Leakage consequence | Potential cross-market look-ahead if a market's event timestamp precedes when the strategy could have received it. |
| Recommended fix | Document `event_ts` versus `ingest_ts` research modes and enforce joins on the chosen availability clock. |
| Required regression test | NQ/MNQ fixture with `event_ts` and delayed `ingest_ts`; prove features use only the configured availability clock. |

### QR-009

| Field | Value |
| --- | --- |
| Severity | MEDIUM |
| Category | Invalid MBO states |
| File | `src/nq/orderbook/book.py`, `src/nq/orderbook/reconstruction.py` |
| Function/Class | `OrderBook.apply`, `reconstruct` |
| Exact lines | `book.py:67-80`, `book.py:90-98`, `reconstruction.py:147-151` |
| Evidence | Unknown cancel/fill/modify references increment `unknown_order_refs`; unknown modify also creates a new order. The pipeline does not fail by default on this integrity condition. |
| Why it may be wrong | The user requirement says invalid states should fail loudly instead of silently corrupting research. Counting is useful, but continuing can pollute depth features. |
| Quantitative consequence | Missing snapshots, partial files, or contract roll boundaries can be silently mixed into research features. |
| Leakage consequence | Not a look-ahead leak, but state corruption can dominate signal estimates. |
| Recommended fix | Add strict mode for production research that raises on unknown references or requires a valid Databento snapshot/reset boundary before continuing. |
| Required regression test | Strict reconstruction fails on unknown cancel/modify/fill; non-strict mode preserves current diagnostics for exploratory use. |

### QR-010

| Field | Value |
| --- | --- |
| Severity | MEDIUM |
| Category | Statistical selection bias |
| File | `src/nq/strategies/fvg_hypothesis.py`, `src/nq/strategies/breakout_hypothesis.py` |
| Function/Class | `default_fvg_grid`, `volume_breakout_grid`, `exploratory_screen_candidates` |
| Exact lines | `fvg_hypothesis.py:77-110`, `fvg_hypothesis.py:349-376`, `breakout_hypothesis.py:96-180`, `breakout_hypothesis.py:524-533` |
| Evidence | FVG tests many grid variants; Breakout tests a larger grid and optional SSL enhancements. Exploratory screening evaluates all candidates on the full feature period. |
| Why it may be wrong | Repeated OOS reuse and large hypothesis grids can create false discoveries unless there is a final untouched OOS period or nested validation. |
| Quantitative consequence | Reported IC/p-values can understate multiple-testing and selection risk. |
| Leakage consequence | Not direct feature leakage, but test-set reuse contaminates research conclusions. |
| Recommended fix | Separate train, validation, and final OOS; reserve a final untouched period and report family-wise hypothesis counts. |
| Required regression test | Report metadata includes number of candidates tried and final OOS period; selection must not use final OOS metrics. |

### QR-011

| Field | Value |
| --- | --- |
| Severity | LOW |
| Category | Reproducibility |
| File | `pyproject.toml`, local environment |
| Function/Class | Package management |
| Exact lines | Baseline report `docs/deep_audit/00_baseline.md` |
| Evidence | Python default is 3.10.9, project requires `>=3.11`, `uv` is missing, no lock file was found, and dependency declarations are lower bounds only. |
| Why it may be wrong | Exact research reproducibility requires a deterministic dependency set and verified interpreter. |
| Quantitative consequence | Different Polars/PyArrow/NumPy versions can change rolling, join, null, dtype, and floating-point behavior. |
| Leakage consequence | None directly, but reproducibility gaps make leakage fixes hard to verify. |
| Recommended fix | Create a Python 3.11 virtualenv and lock dependencies with the repository's chosen package manager. Do not blindly upgrade. |
| Required regression test | CI installs from the lock file and runs `ruff`, `mypy`, and `pytest` in Python 3.11. |

## Items Verified As Mostly Correct So Far

- `CausalStandardScaler.fit` and `PCAEncoder.fit` are called inside SSL fold evaluation
  on fold-train matrices only.
- `build_sequences` creates past-to-present windows and timestamps the sample at the
  window end.
- Failed FVG and Failed Breakout use completed bars with `availability_ts = bucket_end`.
- Failed FVG effort baselines use shifted rolling statistics.
- Failed Breakout breakout levels use prior bars only, not the signal bar.
- Feature joins inspected so far use backward `join_asof`.

These are not final safety claims. Prefix-invariance and future-perturbation tests
still need to be expanded before final verification.

