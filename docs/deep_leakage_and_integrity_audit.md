# Deep Leakage and Integrity Audit

Date: 2026-07-23

Scope: implementation audit of the current `Nq-main` project for temporal leakage,
target leakage, train/test contamination, market-data causality, walk-forward
validation, order-book reconstruction, execution realism, and feature availability.

This audit does not trust README text, comments, or test names as proof. Findings are
based on source inspection, test execution, and targeted diagnostic snippets. No
production code was modified for this audit.

## Verification Run

Commands executed:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m leakage
```

Results:

- Full test suite: `267 passed in 132.00s`
- Marked leakage tests: `5 passed, 262 deselected in 1.40s`

Important limitation: the marked leakage suite is narrow. It primarily covers
reconstruction causality. It does not prove alpha discovery, cross-market research,
session handling, execution realism, or feature-store behavior are leakage-safe.

Graph/indexing note: Graphify was used only as a navigation aid. The existing
`graphify-out/graph.json` was not overwritten because the generated graph would have
reduced node count. Direct source inspection is the source of truth for this audit.

## Executive Risk Summary

The highest-risk issue is not future-looking feature calculation inside the core FVG
and breakout feature builders. Those are mostly careful. The highest-risk issue is
selection leakage: alpha discovery and exploratory screens evaluate many candidate
signals against full-sample forward labels without a nested, purged, train-only
selection process.

The second major risk is synthetic cross-market contamination. Several research paths
run in `nq_only` mode by passing the same NQ frame as both NQ and MNQ, while configs
still include MNQ/cross-market signals. This can create false confirmation and invalid
cross-market conclusions.

The third major risk is session and availability semantics. Session IDs are calendar
dates rather than CME futures trading sessions, and some cross-market alignment uses
exchange event time instead of explicit receive/decision availability time.

## Findings

### F-001 - CRITICAL - Full-sample alpha discovery uses all forward labels for signal selection

- File: `src/nq/alpha/discovery.py`
- Function: `discover_alpha_from_features`
- Lines: 87-167, especially 93-114, 150-163, 167
- Also called from: `src/nq/research/orchestrator.py`
- Caller lines: 580-597 and 609-626
- Evidence:
  - `discover_alpha_from_features` builds forward execution labels across the full
    input frame when `mode == "execution"` at lines 93-114.
  - It builds midpoint forward-return labels across the full input frame when
    `mode == "mid"` at lines 150-163.
  - Every candidate signal is evaluated on the same full frame and then passed to
    `screen_signals` at line 167.
  - The orchestrator calls this directly from the full pipeline feature frame, with no
    purged train/validation/test split around the discovery stage.
- Why it is wrong:
  - The system uses future labels from the entire sample to decide which signals are
    interesting.
  - Benjamini-Hochberg correction does not fix temporal selection leakage. It only
    adjusts p-values under the tested statistical assumptions.
- How it could bias results:
  - The chosen signals are already optimized on the reported period.
  - Reported alpha quality can look materially stronger than out-of-sample live
    performance.
  - Multiple signal families, horizons, and execution modes compound the selection
    bias.
- Recommended fix:
  - Replace full-sample discovery with nested purged walk-forward selection.
  - Select candidate signals only on training/validation folds.
  - Report final metrics only on untouched out-of-sample folds.
  - Keep a final holdout period that is not used for candidate generation, threshold
    selection, horizon selection, or model selection.
  - Report the full number of tested candidates, horizons, families, and rejected
    hypotheses.

### F-002 - HIGH - `nq_only` mode duplicates NQ as MNQ while still allowing cross-market signals

- Files:
  - `src/nq/research/orchestrator.py`
  - `src/nq/strategies/fail_fvg.py`
  - `src/nq/strategies/fail_breakout.py`
  - `src/nq/strategies/vp_auction.py`
  - `src/nq/strategies/fvg_hypothesis.py`
  - `src/nq/strategies/breakout_hypothesis.py`
  - `configs/research.toml`
  - `configs/fail_fvg.toml`
  - `configs/fail_breakout.toml`
  - `configs/vp_auction.toml`
- Functions:
  - `_load_pipeline_frames`
  - `run_fail_fvg_research`
  - `run_fail_breakout_research`
  - `run_vp_auction_research`
  - `search_fail_fvg_hypotheses`
  - `search_fail_breakout_hypotheses`
- Lines:
  - `src/nq/research/orchestrator.py`: 708-728
  - `src/nq/strategies/fail_fvg.py`: 85-100
  - `src/nq/strategies/fail_breakout.py`: 94-109
  - `src/nq/strategies/vp_auction.py`: 80-95
  - `src/nq/strategies/fvg_hypothesis.py`: 422-446
  - `src/nq/strategies/breakout_hypothesis.py`: 399-431
  - `configs/research.toml`: 37-68
  - `configs/fail_fvg.toml`: 45-50
  - `configs/fail_breakout.toml`: 32-65
  - `configs/vp_auction.toml`: 46-50
- Evidence:
  - `_load_pipeline_frames` returns `(nq_frame, nq_frame)` when
    `cross_market_mode == "nq_only"`.
  - Strategy entry points set `partner = mnq if mnq is not None else nq`.
  - Hypothesis search paths pass the same frame into `cross_market_features` when MNQ
    is absent.
  - Config files still include features such as `mnq_delta`, `trap_setup`, and
    `lead_lag` while defaulting to `cross_market_mode = "nq_only"`.
  - Diagnostic snippet:

    ```text
    rows 1
    nq_equals_mnq_close True
    nq_delta_equals_mnq_delta True
    ```

- Why it is wrong:
  - Cross-market features imply independent information from related instruments.
    Passing the same NQ data as MNQ destroys that assumption.
  - The feature names still communicate MNQ/lead-lag/trap semantics even though the
    source is mirrored NQ.
- How it could bias results:
  - Divergence, confirmation, and trap features can become self-confirmation features.
  - The system may select cross-market-looking signals that cannot exist in live NQ/MNQ
    deployment.
  - Apparent robustness across NQ/MNQ can be entirely synthetic.
- Recommended fix:
  - In `nq_only` mode, disable all MNQ/cross-market features and signals.
  - Alternatively, mark synthetic self-partner columns explicitly and exclude them from
    alpha discovery and production research.
  - Add tests proving `nq_only` output contains no `mnq_*`, `lead_lag_*`, or
    cross-market trap features unless an explicit synthetic-fixture flag is set.

### F-003 - HIGH - CME futures sessions use ET calendar date instead of trading session ID

- File: `src/nq/core/session.py`
- Functions:
  - `session_date_from_ns`
  - `session_phase_from_ns`
- Lines: 63-77
- Downstream files:
  - `src/nq/simulation/cross_market.py`
  - `src/nq/models/tick_stream.py`
- Downstream lines:
  - `src/nq/simulation/cross_market.py`: 139-150
  - `src/nq/models/tick_stream.py`: 329-392
- Evidence:
  - `session_date_from_ns` converts event time to America/New_York and returns
    `local.date().isoformat()`.
  - `session_phase_from_ns` only labels RTH-style phases and does not produce a CME
    Globex trade date/session ID.
  - Diagnostic snippet:

    ```text
    2024-07-15T17:59:00-04:00 session_date=2024-07-15 phase=ETH
    2024-07-15T18:30:00-04:00 session_date=2024-07-15 phase=ETH
    2024-07-16T09:45:00-04:00 session_date=2024-07-16 phase=OPEN
    ```

- Why it is wrong:
  - CME index futures trading sessions do not map cleanly to local calendar dates.
    The evening Globex open belongs to the next trading session, while this function
    keeps it on the same calendar date.
- How it could bias results:
  - Session highs/lows, confirmation-failure features, trap setup, volume-profile
    state, and regime state can reset at the wrong boundary.
  - The model may see prior-session state as if it belonged to the current session, or
    split one trading session across two dates.
  - Walk-forward folds and daily reports can be misaligned around evening Globex,
    midnight, DST, holidays, and contract rolls.
- Recommended fix:
  - Introduce a central CME session calendar that returns a true futures trading
    session ID.
  - Separate `session_phase` from `trading_session_id`.
  - Add tests for 16:00, 17:00, 18:00 ET, midnight, RTH open, DST transitions,
    holidays, and roll dates.
  - Reset session-scoped features by `trading_session_id`, not by local date.

### F-004 - HIGH - `max_rows` is applied before causal sorting despite docstring promise

- File: `src/nq/ingestion/reader.py`
- Functions:
  - `_read_columnar`
  - `load_mbo_frame`
- Lines:
  - `_read_columnar`: 43-63
  - `load_mbo_frame`: 101-127
- Evidence:
  - The docstring says `max_rows` is applied after sorting.
  - The file path path passes `max_rows` into `_read_columnar`, which applies
    `n_rows`/`head` during read before `_prepare_frame` performs sorting.
  - The DataFrame path also applies `.head(max_rows)` before `_prepare_frame`.
  - Diagnostic snippet:

    ```text
    input event_ts: [30, 10, 20]
    load_mbo_frame(..., max_rows=2) event_ts: [10, 30]
    expected sorted-first head: [10, 20]
    ```

- Why it is wrong:
  - A smoke or research slice can drop earlier events and retain later events if raw
    input is not already sorted.
- How it could bias results:
  - Chronological validation and feature warm-up become invalid on unsorted inputs.
  - Early smoke results can include future rows while omitting earlier rows.
  - Leakage diagnostics that use `max_rows` can pass on a non-causal slice.
- Recommended fix:
  - Sort and validate first, then apply `max_rows`.
  - If read pushdown is needed for very large files, first validate that the source is
    sorted by `event_ts` and `sequence`; otherwise reject `max_rows` with a clear
    error.
  - Prefer explicit time slicing for large-market-data diagnostics.

### F-005 - HIGH - Cross-market alignment uses event/bucket time, not explicit live availability time

- File: `src/nq/simulation/cross_market.py`
- Functions:
  - `_align_markets`
  - `cross_market_features`
- Lines:
  - `_align_markets`: 78-113
  - `cross_market_features`: 116-205
- Related files:
  - `src/nq/strategies/fvg_hypothesis.py`
  - `src/nq/strategies/breakout_hypothesis.py`
- Related lines:
  - `src/nq/strategies/fvg_hypothesis.py`: 440-446
  - `src/nq/strategies/breakout_hypothesis.py`: 425-431
- Evidence:
  - `_align_markets` aligns bucket starts and supports a synthetic `latency_ns`
    offset.
  - It does not use `ingest_ts`, `ts_recv`, or a decision-time availability column.
  - Search runners hard-code `latency_ns=0` when calling `cross_market_features`.
- Why it is wrong:
  - In live trading, the causal clock is what has been received and processed before
    the decision, not merely the exchange event timestamp.
  - Same-bucket NQ/MNQ joins can include contemporaneous partner-market information
    that was not available when the order decision was made.
- How it could bias results:
  - Cross-market signals can be stronger in backtests than in live replay.
  - Lead-lag and trap features can use partner-market updates that would arrive after
    the decision.
- Recommended fix:
  - Define an explicit feature clock: `event_ts` for offline exchange-time studies and
    `ingest_ts`/`ts_recv` for live/replay decision features.
  - Emit `availability_ts` for every cross-market feature.
  - Pass configured latency from strategy configs into hypothesis search.
  - Add tests with delayed MNQ receive timestamps proving features are not visible
    before the configured decision clock.

### F-006 - MEDIUM - FeatureStore collapses instruments in wide snapshots

- File: `src/nq/features/store.py`
- Functions:
  - `as_of`
  - `snapshot_series`
  - `point_in_time_join`
- Lines:
  - `as_of`: 151-173
  - `snapshot_series`: 175-196
  - `point_in_time_join`: 198-220
- Evidence:
  - `as_of` correctly partitions latest rows by `feature` and `instrument_id`.
  - `snapshot_series` pivots only by `availability_ts` and `feature`, using
    `aggregate_function="last"`.
  - If multiple instruments share the same feature and availability timestamp,
    one instrument silently overwrites/collapses the other in the wide output.
  - Diagnostic snippet:

    ```text
    snapshot_series -> [{'availability_ts': 110, 'imbalance': -99.0}]
    as_of -> [
      {'feature':'imbalance','instrument_id':1,'value':10.0},
      {'feature':'imbalance','instrument_id':2,'value':-99.0}
    ]
    ```

- Why it is wrong:
  - Wide feature snapshots are ambiguous when `instrument_id=None` and multiple
    instruments exist.
- How it could bias results:
  - NQ training rows can receive MNQ feature values or vice versa.
  - The overwritten value depends on data order, which is unstable and can contaminate
    train/test features.
- Recommended fix:
  - Require `instrument_id` for wide snapshots, or encode instrument into output column
    names, such as `nq_imbalance` and `mnq_imbalance`.
  - Make `point_in_time_join(..., instrument_id=None)` fail when more than one
    instrument exists.
  - Add a regression test for same-timestamp same-feature multi-instrument snapshots.

### F-007 - MEDIUM - Tick-stream state is global across sessions and contracts

- File: `src/nq/models/tick_stream.py`
- Function: `build_tick_stream`
- Lines: 329-392
- Related files and lines:
  - `src/nq/simulation/order_flow.py`: 49, 89, 94
  - `src/nq/simulation/failed_breakout.py`: 106
- Evidence:
  - `build_tick_stream` creates one NQ book, one MNQ book, one developing volume
    profile, one regime tracker, and cumulative/high/low variables before the event
    loop.
  - The loop does not reset those states on session, trade date, holiday, or contract
    roll boundaries.
  - `cumulative_delta` and related order-flow summaries are also cumulative over the
    provided frame unless the caller pre-slices correctly.
- Why it is wrong:
  - Many features are session-scoped by trading definition, not file-scoped.
  - A file spanning multiple sessions or contracts carries old state into new sessions.
- How it could bias results:
  - Prior-session volume profile, delta, high/low, and regime information can affect
    later-session signals.
  - Backtests can benefit from state that would be reset or unavailable in a real
    deployment.
- Recommended fix:
  - Define reset semantics per feature: continuous order book, trading-session volume
    profile, RTH-only profile, Globex profile, and contract-roll reset.
  - Reset session-scoped state by the corrected CME `trading_session_id`.
  - Add two-session tests proving cumulative and profile features reset exactly where
    intended.

### F-008 - MEDIUM - Volume Profile/Auction implementation is causal but semantically bucket-local

- File: `src/nq/simulation/volume_profile.py`
- Functions:
  - `DevelopingVolumeProfile.update`
  - `developing_value_area`
- Lines:
  - `DevelopingVolumeProfile`: 96-146
  - `developing_value_area`: 162-223
  - `_PRICE_SCALE`: 25
- Related file: `src/nq/simulation/auction.py`
- Related lines: 46-77 and 163-190
- Evidence:
  - `DevelopingVolumeProfile` is truly event-by-event and causal.
  - `developing_value_area` groups by bucket and computes value area independently
    inside each bucket.
  - `poc_migration` is a diff of bucket-local POCs, not a diff of session-developing
    POCs.
  - Price scaling uses a local approximate constant rather than centralized tick-size
    metadata.
- Why it is wrong:
  - The name and downstream auction feature interpretation suggest a developing
    session profile, but the implementation is bucket-local.
  - This is not direct look-ahead, but it is a feature-definition integrity problem.
- How it could bias results:
  - VP/Auction strategy results may be attributed to market-profile behavior that the
    code is not actually measuring.
  - Parameter tuning on this feature can select artifacts of bucket size rather than
    stable auction structure.
- Recommended fix:
  - Either rename the features as bucket-local profile features, or implement true
    session-developing VP keyed by corrected CME session ID.
  - Centralize NQ/MNQ tick size, point value, and price scale.
  - Add tests where bucket-local POC differs from session-developing POC.

### F-009 - MEDIUM - SSL feature-column selection uses full-frame missingness before folds

- File: `src/nq/models/ssl_pipeline.py`
- Functions:
  - `_feature_columns`
  - `run_ssl_pipeline`
  - `fit_transform_fold`
- Lines:
  - `_feature_columns`: 153-170
  - `run_ssl_pipeline`: 222-239
  - `fit_transform_fold`: 86-90 and 334-338
- Evidence:
  - `_feature_columns` selects feature columns using null fraction over the entire
    feature frame.
  - This happens before sequence construction and fold processing.
  - Fold-local scaler/PCA fitting is correctly isolated, but the feature set itself is
    chosen with future missingness information.
- Why it is wrong:
  - Feature availability in later periods can decide whether a feature exists in
    earlier training folds.
- How it could bias results:
  - Past fold embeddings can depend on future data quality/availability.
  - This can subtly improve stability in historical reports versus live deployment.
- Recommended fix:
  - Freeze feature columns from configuration, or select columns inside each training
    fold only.
  - Add missingness indicators where useful.
  - Add a prefix-perturbation test proving future missingness does not change earlier
    fold feature sets.

### F-010 - MEDIUM - Statistical tests assume IID permutations and exploratory screens reuse the research sample

- Files:
  - `src/nq/alpha/signals.py`
  - `src/nq/coverage/metrics.py`
  - `src/nq/strategies/fvg_hypothesis.py`
  - `src/nq/strategies/breakout_hypothesis.py`
- Functions:
  - `evaluate_signal`
  - `measure_mfig`
  - `measure_cer`
  - `measure_qduf`
  - `exploratory_screen_candidates`
  - `exploratory_screen_breakout_candidates`
- Lines:
  - `src/nq/alpha/signals.py`: 101-107
  - `src/nq/coverage/metrics.py`: 165-172, 238-246, 567-588
  - `src/nq/strategies/fvg_hypothesis.py`: 354-381 and 508-517
  - `src/nq/strategies/breakout_hypothesis.py`: 524-533
- Evidence:
  - `evaluate_signal` permutes forward returns IID.
  - Coverage metrics also permute returns/features over the full available sample.
  - Exploratory screens are executed after walk-forward output and evaluate the full
    candidate table.
- Why it is wrong:
  - Market returns, features, and labels are serially correlated and regime dependent.
  - IID shuffling breaks time structure and usually understates uncertainty.
  - Reusing the same OOS/research sample for repeated candidate exploration creates
    multiple-testing bias.
- How it could bias results:
  - p-values can look too strong.
  - Candidate families may be selected because they fit one historical sample.
  - The final report can unintentionally combine confirmatory and exploratory evidence.
- Recommended fix:
  - Use block bootstrap, stationary bootstrap, or purged fold-level permutation tests.
  - Separate exploratory screens from confirmatory reporting.
  - Maintain an untouched final holdout.
  - Report family-wise candidate counts and all parameter grids tried.

### F-011 - MEDIUM - MBO reconstruction integrity is observable but not strict enough for production research

- Files:
  - `src/nq/orderbook/book.py`
  - `src/nq/orderbook/integrity.py`
  - `src/nq/orderbook/reconstruction.py`
  - `tests/test_integrity.py`
- Functions:
  - `OrderBook.apply`
  - `BookIntegrity.ok`
  - `reconstruct`
- Lines:
  - `src/nq/orderbook/book.py`: 70-103
  - `src/nq/orderbook/integrity.py`: 36-43
  - `src/nq/orderbook/reconstruction.py`: 147-151
  - `tests/test_integrity.py`: 29-37
- Evidence:
  - Unknown `CANCEL` increments a counter and continues.
  - Unknown `MODIFY` increments a counter but creates/inserts an order.
  - `BookIntegrity.ok` does not fail on sequence gaps or crossed book events.
  - Existing tests explicitly treat sequence skips alone as not failing integrity.
- Why it is wrong:
  - For production-grade order-book research, missing initial snapshots, unknown order
    references, crossed books, and sequence gaps can invalidate depth and fill
    features.
  - Unknown modify-as-add can invent displayed liquidity when the input is incomplete.
- How it could bias results:
  - Depth, imbalance, queue, and execution labels can be computed from a corrupted book.
  - Backtests can receive fills against liquidity that did not exist.
- Recommended fix:
  - Add a strict reconstruction mode for research/backtest production runs.
  - Fail or quarantine sessions with unknown references, crossed books, or sequence
    gaps unless explicitly accepted by a data-quality policy.
  - Require clean snapshot/reset boundaries for partial files.
  - Surface integrity counters in all research outputs.

### F-012 - MEDIUM - Duplicate timestamp and same-time ordering are under-specified

- Files:
  - `src/nq/core/time.py`
  - `src/nq/ingestion/databento.py`
  - `src/nq/models/tick_stream.py`
- Functions:
  - `sort_causal`
  - `normalise_databento_mbo`
  - `build_tick_stream`
- Lines:
  - `src/nq/core/time.py`: 15-17
  - `src/nq/ingestion/databento.py`: 111-114
  - `src/nq/models/tick_stream.py`: 325-328
- Evidence:
  - `sort_causal` sorts only by `event_ts` and `sequence`.
  - Databento ingestion synthesizes `sequence` when it is absent.
  - Cross-market tick stream concatenates NQ and MNQ and sorts by the same two columns.
- Why it is wrong:
  - Same-timestamp events across venues/instruments can be arbitrarily ordered if
    source sequence or receive-time metadata is missing.
  - Synthetic sequence is not equivalent to exchange or feed ordering.
- How it could bias results:
  - Same-time events may cross train/test boundaries incorrectly.
  - NQ/MNQ lead-lag and trap state can depend on arbitrary row order.
  - Replays can become non-deterministic across file formats or scan order.
- Recommended fix:
  - Preserve and use source metadata such as `ts_recv`, publisher/channel, original row
    ordinal, and instrument/source tie-breakers.
  - Reject missing exchange sequence in strict mode for MBO reconstruction.
  - Keep same-timestamp event groups together at validation split boundaries.

### F-013 - MEDIUM - Execution simulation is simplified and likely optimistic

- Files:
  - `src/nq/simulation/execution/intraday.py`
  - `src/nq/simulation/execution/depth_fill.py`
  - `src/nq/alpha/discovery.py`
- Functions:
  - `execution_forward_returns`
  - `depth_walk_forward_returns`
  - `discover_alpha_from_features`
- Lines:
  - `src/nq/simulation/execution/intraday.py`: 42-64
  - `src/nq/simulation/execution/depth_fill.py`: 82-104
  - `src/nq/alpha/discovery.py`: 95-114
- Evidence:
  - Entry uses current ask/bid plus slippage and exit uses future bid/ask.
  - Depth-walk labels use visible future depth at the exit timestamp.
  - No queue position, partial fill probability, order latency, cancellation latency,
    market impact, or realistic commission default is enforced in most configs.
- Why it is wrong:
  - The labels are useful research approximations, but they are not a production fill
    simulator.
  - Current-row entry assumes the order can be placed and filled immediately at the
    observed decision-row price.
- How it could bias results:
  - Strategy EV can be inflated, especially for short horizons and high-turnover
    signals.
  - Depth-based labels may assume exit liquidity that would not be executable for the
    strategy's queue position or order size.
- Recommended fix:
  - Add an explicit decision-to-order-to-fill clock.
  - Enter on the next eligible book/trade after configured latency.
  - Model queue position, partial fills, order size, market impact, and actual NQ/MNQ
    commissions/fees.
  - Keep approximate labels named as research labels unless the stricter simulator is
    used.

### F-014 - LOW - Bearish trap setup branch appears unreachable after high/low update

- File: `src/nq/models/tick_stream.py`
- Function: `_trap_setup`
- Lines: 192-206
- Caller lines: 273-292
- Evidence:
  - `nq_high`, `mnq_high`, and `mnq_low` are updated before `_trap_setup` is called.
  - The bearish branch checks `mnq_mid <= mnq_high and nq_mid > nq_high`.
  - Since `nq_high` has already been updated to at least `nq_mid`, `nq_mid > nq_high`
    cannot be true after the update.
  - `mnq_low` is passed but not used.
- Why it is wrong:
  - This is not a look-ahead issue, but the signal logic likely does not implement the
    intended bearish trap.
- How it could bias results:
  - The strategy can accidentally over-represent one side of the trap logic.
  - Research conclusions about trap features may be based on a broken asymmetric
    signal.
- Recommended fix:
  - Compare against previous highs/lows before updating the state, or use low-side
    logic for bearish traps.
  - Add unit tests where bullish and bearish traps are both expected to trigger.

## Important Parts That Are Proven Correct Or Mostly Correct

These items passed source-level causality review, subject to the caveats listed above.

### MBO contract validation

- File: `src/nq/contracts/mbo.py`
- Function: `validate_mbo_frame`
- Lines: 92-130
- Evidence:
  - Required schema is enforced.
  - `event_ts` and `ingest_ts` are cast to integer timestamps.
  - `ingest_ts >= event_ts` is enforced.
- Assessment:
  - This is a good causality guard for raw MBO input. It should be kept and extended
    with stricter session/order-quality checks.

### Causal sorting primitive

- File: `src/nq/core/time.py`
- Function: `sort_causal`
- Lines: 15-17
- Evidence:
  - Uses stable sort by `event_ts` and `sequence`.
- Assessment:
  - Correct as a base primitive when real sequence metadata exists. It is not
    sufficient by itself for cross-instrument same-timestamp ordering or synthetic
    sequence cases.

### Reconstruction single-instrument guard

- File: `src/nq/orderbook/reconstruction.py`
- Function: `reconstruct`
- Lines: 106-110
- Evidence:
  - Rejects multi-instrument frames.
  - Asserts the frame is sorted before replay.
- Assessment:
  - Good safety boundary. Production strictness should be raised as described in
    F-011.

### Depth lifecycle availability

- File: `src/nq/simulation/depth_lifecycle.py`
- Function: `depth_at_bar_close`
- Lines: 160-181
- Evidence:
  - Emits bar-close depth features after processing events through bucket end.
  - Sets `availability_ts` to `bucket_end`.
- Assessment:
  - This is causally correct for bar-close features if the strategy only consumes them
    at or after `bucket_end`.

### OHLCV bar availability

- File: `src/nq/simulation/fvg.py`
- Function: `build_ohlcv_bars`
- Lines: 71-91
- Evidence:
  - Groups trades by closed buckets.
  - Emits `availability_ts = bucket_end`.
- Assessment:
  - Correct bar-level availability convention.

### FVG feature availability

- File: `src/nq/simulation/fvg.py`
- Functions:
  - `detect_h1_fvgs`
  - `_with_effort_features`
  - `_pick_failed_fvg`
- Lines:
  - `detect_h1_fvgs`: 119-151
  - `_with_effort_features`: 154-157
  - `_pick_failed_fvg`: 236-244
- Evidence:
  - FVGs become available only after the relevant higher-timeframe bar end.
  - Effort features use shifted volume before rolling means.
  - Failed-FVG selection requires FVG availability no later than signal time.
- Assessment:
  - This is one of the stronger causal implementations in the project.

### Breakout feature availability

- File: `src/nq/simulation/breakout.py`
- Functions:
  - `_with_volume_baselines`
  - `detect_breakouts`
  - `detect_failed_breakouts`
- Lines:
  - `_with_volume_baselines`: 84-122
  - prior-range logic: 283-285
  - signal availability: 334-354
- Evidence:
  - Volume baselines use shifted inputs.
  - Breakout ranges use prior bars.
  - Signals are emitted with bar availability timestamps.
- Assessment:
  - Mostly causal at the bar level, assuming input bars themselves are causally built.

### Purged walk-forward splitter

- File: `src/nq/models/splitting.py`
- Function: `purged_walk_forward_split`
- Lines: 26-88
- Evidence:
  - Requires monotonic timestamps.
  - Applies embargo after validation.
  - Purges training samples whose label horizon overlaps validation start.
- Assessment:
  - Correct core splitter design. Risk remains where callers do model/signal selection
    outside this splitter.

### Hypothesis walk-forward calls pass label horizon

- File: `src/nq/strategies/fvg_hypothesis.py`
- Function: `walk_forward_select_hypotheses`
- Lines: 257-263
- Evidence:
  - Calls `purged_walk_forward_split` with `label_horizon=horizon`.
- Assessment:
  - Good use of purging for the explicit hypothesis-selection path.

### Train-only scaler primitive and SSL fold fitting

- Files:
  - `src/nq/models/preprocessing.py`
  - `src/nq/models/ssl_pipeline.py`
- Functions:
  - `CausalStandardScaler.fit`
  - `fit_transform_fold`
- Lines:
  - `src/nq/models/preprocessing.py`: 29-44
  - `src/nq/models/ssl_pipeline.py`: 86-90 and 334-338
- Evidence:
  - Scaler statistics are fit on passed training frames.
  - SSL fold scaler/PCA fitting is local to the fold.
- Assessment:
  - This avoids the worst global-normalization leakage. The full-frame feature-column
    selection issue in F-009 still needs correction.

### Sequence windows are past-to-present

- File: `src/nq/models/windowing.py`
- Function: `build_sequences`
- Lines: 46-85
- Evidence:
  - Each sequence uses rows `start:end`.
  - Sequence timestamp is taken from the final row of the window.
- Assessment:
  - Correct temporal orientation for representation learning, assuming input rows are
    correctly sorted and feature availability is respected.

### Asof joins are generally backward

- Files:
  - `src/nq/research/orchestrator.py`
  - `src/nq/models/ssl_pipeline.py`
- Functions:
  - `attach_depth_asof`
  - `apply_causal_ssl_gate`
- Lines:
  - `src/nq/research/orchestrator.py`: 214, 273, 299, 396
  - `src/nq/models/ssl_pipeline.py`: 178-190
- Evidence:
  - Feature joins use `join_asof(..., strategy="backward")`.
  - SSL gate threshold uses shifted rolling quantile.
- Assessment:
  - This is the correct join direction for causal feature attachment. It depends on all
    joined feature tables having valid `availability_ts`.

## Recommended Audit Gate Before Any Fixes

Before interpreting any backtest or alpha report as research-valid, the project should
pass these gates:

1. No full-sample signal selection. All discovery must be nested and purged.
2. `nq_only` mode must not emit MNQ/cross-market signals.
3. All session-scoped features must use a CME trading-session ID.
4. Cross-market features must be keyed by explicit availability/receive time.
5. Production research reconstruction must run in strict integrity mode.
6. Execution labels must clearly distinguish research approximation from live-fill
   simulation.
7. Final reports must separate exploratory screens from confirmatory OOS results.

## Bottom Line

Core low-level causality is better than average in several places: shifted rolling
features, backward asof joins, bar-close availability, fold-local scaling, and the
purged splitter are all strong building blocks.

However, the current project is not yet leakage-safe at the system level. The critical
blocker is full-sample alpha/signal discovery. The high-risk structural blockers are
synthetic NQ-as-MNQ cross-market features, incorrect CME session IDs, pre-sort
`max_rows`, and ambiguous event-time cross-market availability.

