# Final Adversarial Quantitative Audit

Date: 2026-07-23

Scope: independent adversarial audit of the end-to-end research path:

`raw MBO -> normalization -> ordering -> order-book reconstruction -> streaming features -> Volume Profile / order flow / depth -> FVG / breakout -> SSL -> alpha discovery -> hypothesis selection -> walk-forward -> execution -> statistics -> final report`

This audit did not modify production code or tests. Evidence below comes from direct code tracing and one-off adversarial probes run against the current implementation.

## Executive Verdict

The system is **not currently leakage-safe**.

The largest remaining quantitative risk is that the default streaming and several bucketed feature paths ignore raw `ingest_ts` when assigning feature `availability_ts`. A delayed event can be incorporated into features, order-book state, Volume Profile, FVG/breakout bars, depth, and cross-market MNQ state before it was actually received.

## Findings Summary

| ID | Severity | Status | Summary |
|---|---|---|---|
| A-001 | CRITICAL | VERIFIED BUG | Default streaming/tick path leaks delayed MNQ/NQ events by using `event_ts` as availability. |
| A-002 | CRITICAL | VERIFIED BUG | Bucketed OHLC, order-flow, depth, and related features use bucket end/event time, not max included `ingest_ts`. |
| A-003 | HIGH | VERIFIED BUG | In-memory DataFrame pipeline inputs bypass MBO loading/sorting, and OHLC bar construction is order-dependent and roll-blending. |
| A-004 | HIGH | VERIFIED BUG | Walk-forward splitter can put duplicate availability timestamps in both train and test. |
| A-005 | HIGH | VERIFIED BUG | Alpha holdout boundaries depend on final full-sample length; prefix extension changes selected candidates. |
| A-006 | HIGH | VERIFIED BUG | Depth execution simulator fills crossed/invalid books instead of rejecting them. |
| A-007 | MEDIUM | RISK | Tick stream resets session-scoped profile/regime/delta, but keeps order-book state across Globex session changes unless source emits clear. |
| A-008 | MEDIUM | RISK | Exploratory hypothesis screens use full-sample labels and emit `selected` flags that can be mistaken for OOS-safe selection. |
| A-009 | MEDIUM | VERIFIED BUG | M9 coverage report corrects multiple testing only over already-triggered metrics, omitting the full tested family. |
| A-010 | LOW | RISK | FeatureStore exact duplicate feature snapshots at the same availability timestamp use `last`, which is deterministic only with stable ingestion order. |
| A-011 | LOW | VERIFIED BUG | Tick SSL masked reconstruction error compares scaled reconstructions to raw structural-mask targets. |

## Detailed Findings

### A-001 - Streaming/tick cross-market delayed-ingest leakage

Severity: **CRITICAL**  
Status: **VERIFIED BUG**

Exact file: `src/nq/models/tick_stream.py`  
Functions: `build_tick_stream`, `_tick_row`  
Lines: 320-343, 366-373, 422-424, 452-520

Exact file: `src/nq/features/streaming.py`  
Functions: `streaming_event_features`, `sample_streaming_to_interval`, `build_streaming_research_features`  
Lines: 48-91, 94-120, 123-194

Evidence:

```text
MNQ event: event_ts=100, ingest_ts=200
NQ events: event_ts=110,111,112, ingest_ts=110,111,112

Probe output:
[
  {'event_ts': 100, 'availability_ts': 100, 'instrument_id': 2, 'mnq_signed_vol': 7.0},
  {'event_ts': 110, 'availability_ts': 110, 'instrument_id': 1, 'mnq_signed_vol': 7.0},
  {'event_ts': 111, 'availability_ts': 111, 'instrument_id': 1, 'mnq_signed_vol': 7.0},
  {'event_ts': 112, 'availability_ts': 112, 'instrument_id': 1, 'mnq_signed_vol': 7.0}
]
```

Why it is wrong:

The temporal contract defines `ingest_ts` as receive time and `availability_ts` as when a derived value can be used. `build_tick_stream` sorts combined NQ/MNQ by exchange event time and writes `availability_ts = event_ts`, so delayed MNQ information is injected into NQ features before receive time.

Quantitative consequence:

Default streaming research can trade NQ using MNQ order flow that had not arrived yet. This can inflate cross-market trap, lead-lag, SSL, alpha discovery, and final reports. This is a direct temporal leakage path.

Regression test needed:

Create NQ/MNQ events where MNQ has earlier exchange time but later `ingest_ts` than NQ. Assert NQ rows before MNQ ingest do not include MNQ state, deltas, traps, or embeddings derived from the delayed MNQ event.

Recommended fix:

Add an explicit availability mode to tick-stream construction. In realistic/research-safe mode, process or expose event-derived state no earlier than `ingest_ts`; cross-market state must be gated by receive availability, not exchange event time. Preserve exchange sequence for per-market reconstruction, but do not publish feature rows before all included source events are available.

### A-002 - Bucketed features publish delayed events before receive time

Severity: **CRITICAL**  
Status: **VERIFIED BUG**

Exact file: `src/nq/simulation/common.py`  
Function: `add_time_bucket`  
Lines: 34-48

Exact file: `src/nq/simulation/fvg.py`  
Function: `build_ohlcv_bars`  
Lines: 46-105

Exact file: `src/nq/simulation/order_flow.py`  
Function: `order_flow_summary`  
Lines: 32-54

Exact file: `src/nq/simulation/depth_lifecycle.py`  
Functions: `depth_event_series`, `depth_at_bar_close`  
Lines: 98-125, 128-188

Related impacted paths: `failed_fvg_features`, `failed_breakout_features`, `auction_signal_frame`, `_attach_causal_depth`, and any downstream alpha/execution path using those features.

Evidence:

```text
Delayed trade: event_ts=5, ingest_ts=100, interval_ns=10

OHLC output:
{'bucket_end': 10, 'availability_ts': 10, 'c': 200.0, 'volume': 4.0}

Order-flow output:
{'bucket_end': 10, 'availability_ts': 10, 'delta': 4, 'cumulative_delta': 4}

Delayed depth: max ingest_ts=101
depth_at_bar_close output:
{'bucket_end': 10, 'availability_ts': 10, 'nq_bid': 199.0, 'nq_ask': 201.0}
```

Why it is wrong:

The feature value includes events that were not received until `ingest_ts=100/101`, but the feature is published at bucket end `10`. `availability_ts >= event_ts` is necessary but not sufficient; for realistic point-in-time use it must also be no earlier than the receive time of the source rows used in the value.

Quantitative consequence:

OHLC close, FVG, failed breakout, Volume Profile, cumulative delta, order-flow, book depth, bid/ask, and depth-based execution labels can all include late-arriving information before it was available. This can materially improve backtest timing, fills, and signal selection.

Regression test needed:

For every bucketed feature builder, create rows with `event_ts` inside bucket but `ingest_ts > bucket_end`. Assert output `availability_ts >= max(ingest_ts)` for the rows used, or assert the delayed rows are excluded until their receive time under the selected mode.

Recommended fix:

Centralize bucket availability semantics. For each aggregate bucket, set `availability_ts = max(bucket_end, max(ingest_ts of included rows))` in realistic/research-safe mode. For per-event features, set `availability_ts = max(event_ts, ingest_ts)`. Propagate this through FVG, breakout, VP, order-flow, depth, and execution inputs.

### A-003 - In-memory DataFrame path bypasses ingestion sorting; OHLC is order-dependent and contract-blending

Severity: **HIGH**  
Status: **VERIFIED BUG**

Exact file: `src/nq/research/orchestrator.py`  
Function: `_load_pipeline_frames`  
Lines: 721-750

Exact file: `src/nq/simulation/fvg.py`  
Function: `build_ohlcv_bars`  
Lines: 46-105

Exact file: `src/nq/simulation/breakout.py`  
Function: `failed_breakout_features`  
Lines: 373-411

Evidence:

```text
Unsorted normalized MBO in same bar:
input order bars: [{'o': 100.0, 'h': 200.0, 'l': 100.0, 'c': 200.0}]
causal sorted bars: [{'o': 200.0, 'h': 200.0, 'l': 100.0, 'c': 100.0}]

Mixed contracts:
mixed contract bars produced:
[{'o': 100.0, 'h': 300.0, 'l': 100.0, 'c': 300.0, 'volume': 2.0}]
```

Why it is wrong:

`load_mbo_frame` sorts and validates, but `_load_pipeline_frames` returns `pl.DataFrame` inputs as-is when `max_rows` is `None`. `build_ohlcv_bars` then uses `first()`/`last()` inside a group without sorting causal order and without requiring a single contract identity. Direct DataFrame use is part of the real public research pipeline.

Quantitative consequence:

Open/close, FVG, failed breakout, trend filters, and forward labels can change purely from row order. Contract roll rows can be blended into a single candle, creating artificial range, breakouts, gaps, or FVGs.

Regression test needed:

Run `run_research_pipeline` or `_attach_failed_fvg/_attach_failed_breakout` with an in-memory unsorted normalized frame and assert either rejection or identical results to `load_mbo_frame(frame)`. Add a mixed-contract frame and assert fail-safe rejection unless explicit lifecycle config is supplied.

Recommended fix:

Always route in-memory MBO frames through `load_mbo_frame` or a shared `_prepare_frame` equivalent. Add `sort_causal` and `require_single_contract_identity` at stateful OHLC/FVG/breakout entry points.

### A-004 - Duplicate timestamps can contaminate walk-forward train/test

Severity: **HIGH**  
Status: **VERIFIED BUG**

Exact file: `src/nq/models/splitting.py`  
Function: `purged_walk_forward_split`  
Lines: 26-88

Exact file: `src/nq/alpha/discovery.py`  
Function: `discover_alpha_from_features`  
Lines: 247-259

Evidence:

```text
times = [0, 1, 1, 1, 2, 3, 4, 5]
purged_walk_forward_split(...):
shared timestamp fold: train_idx [0, 1], test_idx [2, 3], shared [1]
```

Why it is wrong:

The splitter partitions by row index after checking non-decreasing timestamps. If duplicate availability timestamps straddle a boundary, simultaneous samples can be in both train and test. Several alpha/hypothesis paths also sort only by `availability_ts`, not by a complete deterministic availability-order key.

Quantitative consequence:

Models and selectors can train on data at the same effective decision time used for validation/OOS. This is especially risky for tick/event data with many same-timestamp events or bucketed features collapsed to the same availability time.

Regression test needed:

Construct duplicated `availability_ts` around every split boundary and assert no timestamp appears in both train and test, or require an explicit strictly increasing availability/order key.

Recommended fix:

Group split boundaries by availability timestamp, or introduce a stable `availability_seq`/source-order key and define whether equal timestamps are simultaneous or ordered. If simultaneous, keep the whole group in either train or test. If ordered, pass the complete ordering key into the splitter and purge on that key.

### A-005 - Prefix extension changes alpha candidate selection

Severity: **HIGH**  
Status: **VERIFIED BUG**

Exact file: `src/nq/alpha/discovery.py`  
Functions: `_temporal_holdout_indices`, `discover_alpha_from_features`  
Lines: 68-102, 255-327

Exact file: `src/nq/models/splitting.py`  
Function: `purged_walk_forward_split`  
Lines: 66-86

Evidence:

```text
Final-OOS perturbation sanity:
selected before: ['stable_alpha']
selected after final perturb: ['stable_alpha']

Prefix extension:
300-row selected: ['early_alpha']
450-row selected: ['late_alpha', 'early_alpha']
```

Why it is wrong:

The final-OOS perturbation did not change selection, which is good. However, extending the dataset changes train/validation/final boundaries because holdouts are computed as proportions of the full sample. That means the selection decision for a prefix is not invariant to adding future rows.

Quantitative consequence:

Research conclusions for data up to time `T` can change because rows after `T` exist, even if their labels are not directly used in the final-OOS perturbation. This creates future-dependent research design and can support data-snooping through repeated sample-window changes.

Regression test needed:

Run alpha discovery on `data[:T]` and `data[:T+K]` with frozen split cutoffs. Assert selections and preprocessing for rows `<=T` are unchanged. Also test SSL fold assignment prefix invariance if embeddings are consumed downstream.

Recommended fix:

Require explicit absolute train/validation/final cut timestamps or persisted split manifests for research runs. Do not infer split boundaries from final dataset length when reporting prefix-period selections.

### A-006 - Depth execution fills invalid crossed books

Severity: **HIGH**  
Status: **VERIFIED BUG**

Exact file: `src/nq/simulation/execution/depth_fill.py`  
Function: `realistic_depth_execution_simulation`  
Lines: 146-280

Evidence:

```text
bid_px > ask_px at every row:
bid_px = [[101.0], [101.5], [102.0]]
ask_px = [[100.0], [100.5], [101.0]]

Output:
crossed depth long returns: [0.014925373134328358, nan, nan]
long_rejected: [False, False, False], entry=1, exit=2
crossed depth short returns: [0.0049261083743842365, nan, nan]
short_rejected: [False, False, False]
```

Why it is wrong:

The L1 execution simulator rejects invalid/crossed L1 quotes, but the size-aware depth simulator does not validate book monotonicity, positive sizes, or `best_bid < best_ask` before walking depth.

Quantitative consequence:

Corrupted or crossed depth can generate executable returns at impossible prices, including both long and short profits from the same invalid state.

Regression test needed:

Feed crossed, locked, non-monotonic, zero-size, and insufficient-liquidity depth matrices into `realistic_depth_execution_simulation`. Assert rejection or explicit data-quality error.

Recommended fix:

Validate depth snapshots before filling: finite prices, positive sizes, strictly ordered levels, and `best_bid < best_ask`. In strict research mode, fail the run; in simulation mode, reject the affected orders with explicit rejection reasons.

### A-007 - Session boundary keeps order-book state without explicit clear

Severity: **MEDIUM**  
Status: **RISK**

Exact file: `src/nq/models/tick_stream.py`  
Function: `build_tick_stream`  
Lines: 449-488

Evidence:

Probe across 2024-01-02 18:00 ET Globex session boundary:

```text
Rows after session change still showed prior NQ best bid/ask:
{'event_ts': 1704236401000000000,
 'nq_best_bid_norm': 5.0,
 'nq_best_ask_norm': 5.05,
 'nq_bid_size_log': 1.3862943611198906,
 'nq_ask_size_log': 1.3862943611198906}
```

Why it is risky:

The code resets `session_state` at a session change, but not `nq_book`/`mnq_book`. If source MBO emits reliable `CLEAR` actions across maintenance/reopen, this can be correct. If the sample lacks clear events or starts mid-session, stale liquidity can persist across a boundary.

Quantitative consequence:

Depth, spread, trap, and execution features may rely on stale pre-boundary book levels, improving fill availability or suppressing data-quality failures.

Regression test needed:

Create a known maintenance-boundary fixture with and without explicit Databento clear/reset records. Assert fail-safe behavior when clear metadata is absent or incomplete.

Recommended fix:

Do not invent a roll/session clear calendar. Add an explicit configuration for session book lifecycle semantics. In strict mode, require clear/reset metadata or reject streams that cross maintenance boundaries without explicit lifecycle evidence.

### A-008 - Full-sample exploratory screens can be mistaken for OOS-safe selection

Severity: **MEDIUM**  
Status: **RISK**

Exact file: `src/nq/strategies/fvg_hypothesis.py`  
Functions: `exploratory_screen_candidates`, `search_fail_fvg_hypotheses`  
Lines: 354-381, 513-520

Exact file: `src/nq/strategies/breakout_hypothesis.py`  
Function: `search_fail_breakout_hypotheses`  
Lines: 522-531

Evidence:

`exploratory_screen_candidates` computes forward returns over the entire provided feature frame and applies BH selection to all candidate columns. Search pipelines then store/write that exploratory screen alongside walk-forward results.

Why it is risky:

It is labeled exploratory and does not drive `walk_forward_select_hypotheses`, but the artifact contains full-sample `selected` flags. Any downstream report, human process, or later code that treats those flags as candidates for trading would be selecting on OOS/final labels.

Quantitative consequence:

Data-snooping and multiple-testing bias can enter through a documented artifact even if the primary walk-forward selector is clean.

Regression test needed:

Assert that exploratory screens are either opt-in, clearly tagged `exploratory_full_sample_only`, excluded from final recommendations, and not consumed by any trading/backtest path.

Recommended fix:

Rename outputs to make them non-actionable, remove `selected` semantics from exploratory artifacts, or compute exploratory screens only inside train folds.

### A-009 - M9 multiple-testing correction omits non-triggered tests

Severity: **MEDIUM**  
Status: **VERIFIED BUG**

Exact file: `src/nq/coverage/monitor.py`  
Function: `build_coverage_report`  
Lines: 45-87

Evidence:

Code builds the correction family only from pre-triggered metrics:

```python
triggered = [r for r in results if r.triggered]
pvalues = {r.name: r.pvalue for r in triggered}
verified = verify_hypotheses(pvalues, alpha=alpha) if pvalues else None
```

Probe with 20 metrics, one raw `p=0.04`, nineteen raw `p=0.99`:

```text
verified_count 1
alerts [('m0', 'medium')]
metrics_rows 20
```

Why it is wrong:

The family for multiple-testing correction must include all hypotheses tested, not just hypotheses that already passed raw alpha. Filtering first makes BH/verification anti-conservative.

Quantitative consequence:

M9 can overstate structural coverage findings and create false confidence in model diagnostics. It does not directly select alpha in the current pipeline, but it can bias final research conclusions.

Regression test needed:

Build a metric set where one raw p-value passes alpha but fails BH after including the full family. Assert no verified alert is emitted.

Recommended fix:

Pass all metric p-values into `verify_hypotheses`, then emit alerts only for metrics rejected after correction and satisfying their effect-size trigger.

### A-010 - FeatureStore exact duplicate snapshots use ingestion-order `last`

Severity: **LOW**  
Status: **RISK**

Exact file: `src/nq/features/store.py`  
Functions: `as_of`, `snapshot_series`  
Lines: 171-217

Evidence:

`as_of` filters to max `availability_ts` and then uses `.unique(..., keep="last")`; `snapshot_series` pivots duplicate values with `aggregate_function="last"`.

Why it is risky:

For exact duplicates of `(feature, instrument_id, availability_ts)` with conflicting values, the chosen value depends on ingestion order rather than source metadata.

Quantitative consequence:

This is unlikely to create look-ahead by itself, but it can make research non-reproducible under duplicate data ingestion.

Regression test needed:

Insert conflicting duplicate feature observations at the same availability timestamp in different ingestion orders and assert deterministic rejection or deterministic source-priority resolution.

Recommended fix:

Reject conflicting duplicate feature observations by default. If replacement is intended, require explicit versioning or source priority.

### A-011 - SSL tick masked MSE mixes scaled and raw spaces

Severity: **LOW**  
Status: **VERIFIED BUG**

Exact file: `src/nq/models/ssl_pipeline.py`  
Function: `_evaluate_ssl_tick_fold`  
Lines: 316-373

Evidence:

`test_3d` is built from raw flattened windows before scaling. The encoder reconstructs `x_test` after causal scaling, then reshapes the scaled reconstruction and compares it to a structural mask batch built from raw `test_3d`.

Why it is wrong:

The masked reconstruction error is not measured in one consistent feature space.

Quantitative consequence:

SSL report metrics can be numerically invalid or misleading. This is not a direct leakage bug because scaler/PCA are still fit on train only, but it weakens confidence in SSL quality claims.

Regression test needed:

Use a simple known tick-feature matrix and assert masked MSE is computed either entirely in scaled space or entirely after inverse-transforming reconstruction.

Recommended fix:

Build structural masks on scaled `x_test.reshape(...)`, or inverse-transform reconstructions before comparing to raw targets.

## Subsystems That Survived Adversarial Testing

### Strict MBO/order-book reconstruction

Status: **VERIFIED SAFE** for tested corruptions.

Files/functions:

- `src/nq/orderbook/reconstruction.py::reconstruct` lines 131-189
- `src/nq/orderbook/integrity.py::check_integrity` lines 62-93
- `src/nq/orderbook/book.py::OrderBook.apply` lines 57-112

Probe results:

```text
unknown_modify REJECTED StrictReconstructionError unknown_order_refs=1
unknown_cancel REJECTED StrictReconstructionError unknown_order_refs=1
duplicate_add REJECTED StrictReconstructionError unknown_order_refs=1
sequence_gap REJECTED StrictReconstructionError sequence_skips=1
crossed_book REJECTED StrictReconstructionError crossed_book_events=1
mixed_contract_same_instrument REJECTED StrictReconstructionError contract roll / identity change
```

Limits: strict reconstruction safety does not automatically protect all downstream feature builders that bypass strict reconstruction or publish using event-time availability.

### Databento normalization and file loading

Status: **VERIFIED SAFE** for causal ordering and basic temporal contract when `load_mbo_frame` is used.

Files/functions:

- `src/nq/ingestion/databento.py::normalize_databento_frame` lines 118-156
- `src/nq/ingestion/reader.py::load_mbo_frame` lines 85-121
- `src/nq/core/time.py::sort_causal` lines 26-35
- `src/nq/contracts/mbo.py::validate_mbo_frame` lines 92-131

The file-loading path normalizes, validates `ingest_ts >= event_ts`, and sorts causally. The unsafe case is the real orchestrator's in-memory DataFrame bypass described in A-003.

### Batch cross-market delay handling

Status: **VERIFIED SAFE** for the tested delayed MNQ receive case.

Files/functions:

- `src/nq/simulation/cross_market.py::_bucket_availability` lines 40-66
- `src/nq/simulation/cross_market.py::_align_markets` lines 122-166
- `src/nq/simulation/cross_market.py::cross_market_features` lines 209-306

Probe result:

```text
MNQ event_ts same as NQ, MNQ ingest delayed to 200+
Output at NQ availability 110:
mnq_availability_ts=None, mnq_close=None, mnq_delta=0, trap_setup=0
```

Limit: this safety does not apply to the default streaming/tick path in A-001.

### Alpha final-OOS selection freeze

Status: **VERIFIED SAFE** for direct final-OOS label perturbation; **not prefix-invariant** per A-005.

File/function:

- `src/nq/alpha/discovery.py::discover_alpha_from_features` lines 269-365

Probe result:

```text
Final-OOS-only signal was not selected.
Perturbing final-OOS labels did not change selected signals:
selected before: ['stable_alpha']
selected after final perturb: ['stable_alpha']
```

Limit: split cutoffs depend on full sample length.

### SSL train/test preprocessing

Status: **VERIFIED SAFE** for fit-on-train scaling/PCA/world-model calls.

Files/functions:

- `src/nq/models/ssl_pipeline.py::_evaluate_ssl_fold` lines 74-117
- `src/nq/models/ssl_pipeline.py::_evaluate_ssl_tick_fold` lines 316-373
- `src/nq/models/preprocessing.py::CausalStandardScaler` lines 15-49
- `src/nq/models/encoder.py::PCAEncoder` lines 34-78

Scalers, PCA, and world model are fit on train folds only. Remaining concerns are prefix-dependent folds (A-005/A-004) and tick metric scale mismatch (A-011).

### FVG and Failed Breakout timing after causal bars exist

Status: **VERIFIED SAFE** for future perturbation of already-causal bars/features, based on existing regression tests and code path.

Files/functions:

- `src/nq/simulation/fvg.py::detect_h1_fvgs` lines 109-151
- `src/nq/simulation/fvg.py::failed_fvg_from_bars` lines 267-323
- `src/nq/simulation/breakout.py::failed_breakout_from_bars` lines 186-370

Existing tests:

- `tests/test_fvg.py::test_failed_fvg_past_stable_when_future_perturbed`
- `tests/test_breakout.py::test_failed_breakout_past_stable_when_future_perturbed`

Limit: upstream bar construction has A-002/A-003 issues.

### Volume Profile semantic logic

Status: **VERIFIED SAFE** for session-scoped developing value area once event availability is assumed correct.

Files/functions:

- `src/nq/simulation/volume_profile.py::DevelopingVolumeProfile` lines 99-158
- `src/nq/simulation/volume_profile.py::developing_value_area` lines 174-254
- `src/nq/contracts/instruments.py` lines 17-33

The VP code centralizes NQ/MNQ tick metadata and session-scopes developing profiles. Limit: the bucket availability still inherits A-002 if source rows are delayed.

### FeatureStore point-in-time joins

Status: **VERIFIED SAFE** for normal point-in-time as-of filtering.

Files/functions:

- `src/nq/features/store.py::as_of` lines 171-193
- `src/nq/features/store.py::point_in_time_join` lines 219-241

The store filters `availability_ts <= timestamp` and uses backward as-of joins. Exact duplicate conflicts are only a low reproducibility risk (A-010).

### L1 realistic execution simulator

Status: **VERIFIED SAFE** for decision-latency and invalid L1 rejection in the inspected path.

File/function:

- `src/nq/simulation/execution/intraday.py::realistic_execution_simulation` lines 130-227

Limit: depth execution has A-006.

## Requested Verification Commands

These commands are to be run after the audit report is created:

```bash
pytest -ra
pytest -m leakage
pytest --cov --cov-report=term-missing
```

Results:

- `pytest -ra`: **failed in the ambient shell environment** before collection, because `pytest` resolved to system Python 3.8.3 without project dependencies (`polars`, `nq`, `zoneinfo`). No tests executed in that environment.
- `.venv/bin/pytest -ra`: **295 passed** in 81.38s on Python 3.11.3.
- `.venv/bin/pytest -m leakage`: **5 passed, 290 deselected** in 1.13s.
- `.venv/bin/pytest --cov --cov-report=term-missing`: **295 passed** in 132.00s; total coverage **83%**.
