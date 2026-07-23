# Phase 2 System Architecture Audit

Date: 2026-07-23

Repository path: `/Users/abdallah/Desktop/Trading/Nq-main`

Graphify baseline used: `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`.

Important caveat: Graphify was useful for navigation and clustering, but direct source
inspection is the authority. The Phase 1 graph had many duplicate/collapsed structural
edges, so call direction and call cardinality were verified from source before use.

## Entry Points

| Entrypoint | Primary call | Purpose |
| --- | --- | --- |
| `scripts/run_week.py` | `nq.research.orchestrator.run_research_pipeline` | Unified NQ/MNQ MBO to report pipeline. |
| `scripts/run_fail_fvg.py` | `nq.strategies.fvg_hypothesis.search_fail_fvg_hypotheses` or `run_fail_fvg_research` | Failed FVG strategy and hypothesis search. |
| `scripts/run_fail_breakout.py` | `nq.strategies.breakout_hypothesis.search_fail_breakout_hypotheses` or `run_fail_breakout_research` | Failed Breakout strategy and hypothesis search. |
| `scripts/run_vp_auction.py` | `nq.strategies.vp_auction.run_vp_auction_research` | Volume Profile and auction state research. |

All runtime scripts enforce Python `>=3.11` before importing project code.

## Data Flow

```text
Databento or contract-shaped MBO
-> load_mbo_frame
-> normalize_databento_frame, sanitize_mbo_frame, validate_mbo_frame
-> sort_causal(event_ts, sequence)
-> OrderBook event application
-> depth, top-of-book, order flow, OHLCV bars, volume profile
-> streaming or batch research features
-> availability_ts-aligned joins
-> SSL, M9 coverage, alpha screen
-> strategy hypothesis search and execution labels
-> research reports and parquet outputs
```

## Ingestion And Temporal Contract

Raw input is loaded by `src/nq/ingestion/reader.py`.

- `load_mbo_frame` reads DataFrames or `.parquet`, `.arrow`, `.ipc`, `.csv`, `.zst`.
- `_prepare_frame` detects Databento, normalizes, selects the MBO contract, validates,
  then calls `sort_causal`.
- `normalize_databento_frame` maps `ts_event -> event_ts`, `ts_recv -> ingest_ts`,
  maps actions/sides, scales float prices to fixed-point integers, and creates
  `sequence = arange(...)` if the input has no sequence.
- `MBO_SCHEMA` requires `event_ts`, `ingest_ts`, `sequence`, `instrument_id`,
  `symbol`, `action`, `side`, `price`, `size`, `order_id`, `flags`.
- `validate_mbo_frame` enforces schema and `ingest_ts >= event_ts`.
- The canonical local ordering rule is `sort_causal(frame).sort(["event_ts", "sequence"])`.

Temporal implication: `event_ts` is the exchange/matching timestamp, while
`availability_ts` is the timestamp used by higher-level features and joins. In this
repo, most features are available either at event time or at completed bucket close.
The code does not currently carry `publisher_id` or `channel_id` through the internal
contract, so multi-publisher/channel MBO ordering is not fully represented.

## Order Book And Depth

Order book state lives in `src/nq/orderbook/book.py`.

- `OrderBook` maintains `bids`, `asks`, and `orders[order_id]`.
- `reconstruct` applies each event and records top-of-book after each event.
- `depth_event_series` emits a depth snapshot after each event with
  `availability_ts = event_ts`.
- `depth_at_bar_close` updates the book through all events inside a bucket, then emits
  the snapshot at `availability_ts = bucket_end`.
- `attach_depth_asof` attaches depth to features with backward `join_asof`.

Stateful components:

- `OrderBook`: book levels and per-order state.
- `IntegrityReport`: unknown order refs and crossed book events.
- `DepthSnapshot`: flattened visible depth and L1 state.

## Feature Construction

The unified feature assembly is in `src/nq/research/orchestrator.py`.

Default path:

1. `_load_pipeline_frames` loads NQ and MNQ.
2. `_build_research_features` uses `build_streaming_research_features` when
   `feature_mode="streaming"`.
3. Streaming features call `build_tick_stream`, update NQ/MNQ books event by event,
   maintain volume-profile and regime state, then sample the last state per interval.
4. Batch features use `cross_market_features`.
5. `_attach_causal_depth` attaches bar-close depth.
6. `_attach_failed_fvg`, `_attach_auction_vp`, and `_attach_failed_breakout` attach
   strategy/auction features by backward `join_asof`.

Streaming path details:

- `build_tick_stream` sorts NQ and MNQ separately, overwrites local `instrument_id`
  with `1` and `2`, concatenates, then sorts by `event_ts, sequence`.
- `_tick_row` applies the relevant book update, updates the NQ developing volume
  profile on NQ trades, updates MNQ signed volume on MNQ trades, updates regime
  state, and emits a feature row with `availability_ts = event_ts`.
- `sample_streaming_to_interval` groups events into buckets and takes the last known
  state, then sets `availability_ts = bucket_end`.

Batch cross-market path:

- `_market_windows` reconstructs each market, computes mid close per bucket, and
  joins order-flow delta.
- `_align_markets` joins NQ and MNQ. With positive `latency_ns`, NQ rows are aligned
  to MNQ state at or before `bucket_start - latency_ns`.
- `cross_market_features` computes rolling correlations, divergence, confirmation
  failure, session highs/lows using shifted cumulative extrema by `session_date`,
  and `trap_setup`.

## Strategy Feature Flow

Failed FVG:

- `build_ohlcv_bars` extracts MBO trades and builds completed OHLCV bars.
- `detect_h1_fvgs` detects gaps only after the third completed higher-timeframe bar.
- `failed_fvg_from_bars` uses shifted historical effort baselines and considers only
  FVGs whose `availability_ts <= signal_time`.
- `materialize_fvg_hypotheses` builds candidate columns on a shared research clock
  by backward `join_asof`.

Failed Breakout:

- `build_ohlcv_bars` creates completed signal and trend bars.
- `_with_volume_baselines` shifts baseline ATR, volume, cumulative volume, delta, and
  absorption statistics before use.
- `failed_breakout_from_bars` uses prior bars only for breakout levels and emits
  signal rows at completed bar close.
- `materialize_breakout_hypotheses` builds candidate columns by backward `join_asof`.

Volume Profile / Auction:

- `DevelopingVolumeProfile` is event-by-event and cumulative for tick features.
- `developing_value_area` computes a value area per completed bucket.
- `auction_states` joins bucket-local value area with bucket stats and emits balance,
  imbalance, expansion, pullback, and migration fields.
- `auction_signal_frame` converts auction states into numeric signal columns.

## SSL And ML Flow

Bucket SSL:

1. `run_ssl_pipeline` selects numeric feature columns if none are supplied.
2. Missing feature values are filled with zero.
3. `build_sequences` creates causal windows `[i-window+1, ..., i]` and timestamps
   each sequence at the window end.
4. `_walk_forward_folds` calls `purged_walk_forward_split`.
5. `_evaluate_ssl_fold` fits `CausalStandardScaler` on fold train only.
6. `PCAEncoder` is fitted on fold train only.
7. Test-fold embeddings are produced by applying train-fitted transforms to test.

Tick SSL:

1. `run_ssl_tick_pipeline` builds a full causal tick stream.
2. `build_tick_sequences` creates tick/event windows.
3. The same fold discipline applies: scaler and PCA are fitted only on fold train.

Known limitation: `run_ssl_pipeline` performs feature-column availability selection
and zero-fill before splitting. The scaler/PCA are train-only, but the set of selected
columns can still be influenced by future missingness patterns.

## Walk-Forward And Targets

Core splitter: `src/nq/models/splitting.py::purged_walk_forward_split`.

- Expanding train, forward test blocks.
- Training timestamps must be non-decreasing and before test timestamps.
- Embargo removes training rows with timestamps inside the embargo window before the
  test start.
- `purge_samples` removes the last N train samples before test.

Target generation:

- `align_forward_returns` labels row `t` with `price[t+horizon]`.
- `execution_forward_returns` labels rows with entry at `t` and exit at `t+horizon`.
- `execution_forward_returns_depth` does the same using visible depth at entry and
  exit timestamps, with L1 fallback if depth is unavailable.

Critical implication: for labeled alpha and hypothesis selection, it is not enough
that `train_ts < test_ts`. A train label at index `i` also consumes information at
`i + horizon`; folds must remove any train row whose label reaches into the test block.

## Reporting Flow

Unified pipeline:

- `run_ssl_research_pipeline` runs SSL, M9 coverage, alpha discovery, and unifies the
  reports with `build_unified_report`.
- `run_research_pipeline` writes `report.md`, `ssl_metrics.parquet`,
  `coverage_metrics.parquet`, `alpha_evaluations.parquet`, and `features.parquet`
  when `output_dir` is provided.

Hypothesis searches:

- Failed FVG and Failed Breakout write `report.md`, `features.parquet`,
  `fold_selections.parquet`, `exploratory_screen.parquet`, and SSL metrics when
  applicable.

## Architecture Bottlenecks From Graphify

Graphify god nodes and source inspection agree that the main coupling points are:

- `PipelineConfig`
- `ResearchAssistant` and report/evidence objects
- `OrderBook`
- `run_ssl_research_pipeline`
- `AlphaDiscovery`
- `walk_forward_select_hypotheses`
- `build_tick_stream`

The highest-risk coupling is not size alone; it is semantic coupling around
`availability_ts`, forward labels, and MBO state mutation. Repairs should focus there
before cosmetic refactors.

