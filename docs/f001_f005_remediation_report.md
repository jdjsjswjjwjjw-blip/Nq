# F-001 to F-005 Remediation Report

Date: 2026-07-23

Scope: fixed only the high-priority findings requested from
`docs/deep_leakage_and_integrity_audit.md`:

- F-001: full-sample alpha discovery / selection leakage
- F-002: fake NQ-as-MNQ evidence in `nq_only`
- F-003: incorrect CME futures session identification
- F-004: `max_rows` applied before causal sorting
- F-005: cross-market alignment using event time instead of explicit availability

No strategy parameters were tuned. No profitability optimization was performed. No
existing tests were weakened or deleted.

## F-001

Status: FIXED

Original issue:

- `discover_alpha_from_features` generated forward labels, evaluated candidates, selected
  signals, and reported performance on the same full sample.

Root cause:

- The public discovery path had no train/validation/final-OOS separation.
- Final-period labels could influence `selected`.

Files changed:

- `src/nq/alpha/discovery.py`
- `tests/test_alpha.py`

Design of fix:

- Added temporal train/validation/final-OOS splitting inside
  `discover_alpha_from_features`.
- TRAIN is used for initial candidate discovery.
- VALIDATION is used for selected-signal decision.
- FINAL OOS is evaluation-only. The selected list is frozen before final-OOS evaluation.
- Forward-label overlap is purged at train->validation and validation->final boundaries by
  removing rows whose label horizon crosses the next stage.
- Tiny samples no longer fall back to full-sample selection. They emit transparent progress
  logs and return no alpha selections.

Regression test:

- `tests/test_alpha.py::test_alpha_selection_is_invariant_to_final_oos_label_perturbation`

Before behavior:

- The red test selected `["train_val", "oos_only"]`, proving an OOS-only signal could be
  selected.

After behavior:

- The same test selects only `["train_val"]`.
- Changing final-OOS labels does not change selected signals.

Verification result:

- Targeted regression passed.
- Full pytest passed: `274 passed`.

Remaining limitations:

- The split is a conservative three-stage temporal holdout, not a full nested
  walk-forward optimizer over multiple horizons/thresholds.
- Final OOS is still evaluated with IID permutation p-values from existing signal metrics.

## F-002

Status: FIXED

Original issue:

- Multiple paths used NQ as MNQ when `mnq is None` or `cross_market_mode = "nq_only"`.
- This created real-looking `mnq_*`, `lead_lag`, divergence, and trap features from
  duplicated NQ data.

Root cause:

- The orchestrator returned `(nq_frame, nq_frame)` in NQ-only mode.
- Strategy wrappers and hypothesis searches also substituted `mnq_frame = nq_frame`.

Files changed:

- `src/nq/research/orchestrator.py`
- `src/nq/simulation/cross_market.py`
- `src/nq/strategies/fail_fvg.py`
- `src/nq/strategies/fail_breakout.py`
- `src/nq/strategies/fvg_hypothesis.py`
- `src/nq/strategies/breakout_hypothesis.py`
- `tests/test_orchestrator_leakage_regressions.py`

Design of fix:

- Added `single_market_features` for genuine NQ-only feature generation.
- In orchestrator NQ-only mode, MNQ is not duplicated; an empty MNQ frame is carried only
  where function signatures still require a partner frame.
- NQ-only feature building emits no `mnq_*`, `lead_lag`, divergence, confirmation, or trap
  evidence.
- Focused strategy wrappers now use NQ-only signal lists when no real MNQ is supplied.
- Hypothesis searches use NQ-only clocks when `mnq is None`.

Regression test:

- `tests/test_orchestrator_leakage_regressions.py::test_nq_only_research_features_do_not_create_fake_mnq_evidence`

Before behavior:

- The red test found `mnq_*` columns in NQ-only research features.

After behavior:

- The feature set contains no fake MNQ/cross-market columns in NQ-only mode.

Verification result:

- Targeted regression passed.
- Adjacent orchestrator/strategy suites passed.

Remaining limitations:

- Some internal SSL/tick APIs still accept an MNQ frame argument by signature, but NQ-only
  mode no longer feeds duplicated MNQ evidence into research features or signal selection.

## F-003

Status: FIXED

Original issue:

- Session identification used ET calendar date as the session identifier.
- Evening Globex reopen was assigned to the same calendar date instead of the next CME
  futures trading session.

Root cause:

- `session_date_from_ns` returned `local.date().isoformat()`.
- Session-scoped tick-stream state did not reset on CME trading-session changes.

Files changed:

- `src/nq/core/session.py`
- `src/nq/simulation/cross_market.py`
- `src/nq/models/tick_stream.py`
- `tests/test_session.py`
- `tests/test_tick_stream.py`

Design of fix:

- Added explicit central fields:
  - `calendar_date`
  - `trading_session_id`
  - `session_phase`
  - `is_rth`
  - `is_eth`
  - `is_globex`
- `trading_session_id_from_ns` assigns timestamps at or after 18:00 ET to the next
  futures trading session.
- `session_date_from_ns` remains as a compatibility wrapper but now returns the CME
  trading session ID.
- Cross-market session highs/lows group by `trading_session_id`.
- Tick-stream session-scoped state resets when `trading_session_id` changes:
  volume profile, regime tracker, MNQ signed volume, highs/lows, and previous NQ mid.

Regression tests:

- `tests/test_session.py::test_cme_trading_session_id_boundaries`
- `tests/test_session.py::test_cme_trading_session_id_survives_dst_transition`
- `tests/test_tick_stream.py::test_tick_stream_resets_session_scoped_state_at_cme_session_boundary`

Before behavior:

- The new CME session API was missing.
- Tick-stream MNQ signed volume carried across the 18:00 ET session boundary:
  `[1.0, 2.0]`.

After behavior:

- Globex reopen at 18:30 ET on 2024-07-15 maps to trading session `2024-07-16`.
- Midnight and RTH on 2024-07-16 remain in trading session `2024-07-16`.
- Tick-stream MNQ signed volume resets across the CME session boundary:
  `[1.0, 1.0]`.

Verification result:

- Session and tick-stream regressions passed.
- Full pytest passed.

Remaining limitations:

- The session model handles the normal CME daily 18:00 ET reopen and maintenance break, but
  it does not yet encode the full CME holiday calendar, early closes, or contract-roll
  schedule.

## F-004

Status: FIXED

Original issue:

- `max_rows` was applied before causal sorting on DataFrame, parquet, CSV, IPC, and zst
  paths.

Root cause:

- `_read_columnar` used read-time `n_rows`/`head`.
- `load_mbo_frame` also applied `.head(max_rows)` before `_prepare_frame`.

Files changed:

- `src/nq/ingestion/reader.py`
- `tests/test_ingestion.py`

Design of fix:

- `_read_columnar` now reads the full raw file for supported formats.
- `load_mbo_frame` now performs:
  raw input -> normalize/sanitize -> validate -> causal sort -> `max_rows`.
- Unsafe pushdown was removed rather than guessing source sortedness.

Regression test:

- `tests/test_ingestion.py::test_max_rows_applies_after_causal_sort`

Before behavior:

- Input timestamps `[30, 10, 20]` with `max_rows=2` returned `[10, 30]`.

After behavior:

- The same input returns `[10, 20]`.

Verification result:

- Targeted regression passed.
- Full ingestion suite passed.

Remaining limitations:

- Large-file `max_rows` no longer uses read pushdown. This is intentional for temporal
  correctness. Future large-data slicing should use explicit time windows or a verified
  sorted-source contract.

## F-005

Status: FIXED

Original issue:

- Cross-market alignment used event/bucket time and assumed same event time meant same
  information availability.
- Hypothesis search paths hard-coded zero latency.

Root cause:

- Market windows did not carry source availability based on `ingest_ts`.
- `_align_markets` aligned on bucket time instead of an NQ decision availability clock.

Files changed:

- `src/nq/simulation/cross_market.py`
- `src/nq/research/orchestrator.py`
- `src/nq/strategies/fvg_hypothesis.py`
- `src/nq/strategies/breakout_hypothesis.py`
- `tests/test_cross_market_latency.py`

Design of fix:

- Added explicit `availability_mode` for cross-market features:
  - default `ingest` mode uses `max(bucket_end, max(ingest_ts in bucket))`.
  - explicit `event` mode remains available for exchange-time offline studies.
- NQ decision time is `nq_availability_ts`.
- MNQ is as-of joined only when `mnq_availability_ts <= nq_availability_ts - latency_ns`.
- Output feature `availability_ts` is the NQ decision availability timestamp.
- Hypothesis searches now accept `latency_ns` and pass it to real cross-market features.

Regression test:

- `tests/test_cross_market_latency.py::test_delayed_mnq_ingest_is_not_visible_before_nq_decision`

Before behavior:

- MNQ event at earlier exchange time but delayed ingest was visible to the NQ decision.

After behavior:

- Delayed MNQ has `mnq_close = None` before the NQ decision.
- The same MNQ event is visible when its ingest time is before the decision.

Verification result:

- Targeted regression passed.
- Cross-market latency suite passed.

Remaining limitations:

- The schema has `ingest_ts` but not a separate `ts_recv`; if Databento receive-time
  fields are later preserved separately, cross-market availability should use that
  configured clock directly.

## Test Results

Passed:

- Targeted F-001 to F-005 regressions: `7 passed`
- Adjacent suites: `45 passed`
- Leakage subset: `5 passed, 269 deselected`
- Full pytest: `274 passed in 65.62s`
- Coverage run: `274 passed`, total coverage `82%`

Commands used:

```bash
.venv/bin/python -m pytest tests/test_alpha.py::test_alpha_selection_is_invariant_to_final_oos_label_perturbation tests/test_orchestrator_leakage_regressions.py::test_nq_only_research_features_do_not_create_fake_mnq_evidence tests/test_session.py::test_cme_trading_session_id_boundaries tests/test_session.py::test_cme_trading_session_id_survives_dst_transition tests/test_ingestion.py::test_max_rows_applies_after_causal_sort tests/test_cross_market_latency.py::test_delayed_mnq_ingest_is_not_visible_before_nq_decision tests/test_tick_stream.py::test_tick_stream_resets_session_scoped_state_at_cme_session_boundary -q
.venv/bin/python -m pytest -m leakage
.venv/bin/python -m pytest -ra
.venv/bin/python -m pytest --cov --cov-report=term-missing
```

## Leakage Tests

- Existing marked leakage tests pass: `5 passed`.
- New regression tests add coverage for selection leakage, NQ-only fake MNQ evidence,
  CME trading-session boundaries, causal `max_rows`, delayed MNQ availability, and
  session-scoped tick-state reset.

## Tooling Results

The functional pytest gates pass. Repo-wide lint/type gates are not clean.

- `ruff check src tests`
  - `ruff` is not on PATH in this shell.
  - `.venv/bin/ruff check src tests` fails with 44 issues, mostly outside this
    remediation surface: existing complexity/lint in `simulation/breakout.py`,
    `simulation/execution/depth_fill.py`, `research/progress.py`,
    `tests/test_breakout.py`, `tests/test_ssl_enhancements.py`, and related files.
- `ruff format --check src tests`
  - `.venv/bin/ruff format --check src tests` fails because 5 files outside this
    remediation surface would be reformatted:
    `src/nq/ingestion/databento.py`, `src/nq/simulation/breakout.py`,
    `src/nq/strategies/ssl_enhancements.py`, `tests/test_breakout.py`,
    `tests/test_ssl_enhancements.py`.
- `mypy`
  - `.venv/bin/mypy` fails with 94 strict-typing errors, mainly progress-protocol
    `object.op` / stale `type: ignore[union-attr]` issues plus older strict test typing
    problems.

These lint/type failures were not fixed because they are outside F-001 to F-005 and would
expand the scope beyond the requested remediation.

## Files Changed

Source:

- `src/nq/alpha/discovery.py`
- `src/nq/core/session.py`
- `src/nq/ingestion/reader.py`
- `src/nq/models/tick_stream.py`
- `src/nq/research/orchestrator.py`
- `src/nq/simulation/cross_market.py`
- `src/nq/strategies/fail_fvg.py`
- `src/nq/strategies/fail_breakout.py`
- `src/nq/strategies/fvg_hypothesis.py`
- `src/nq/strategies/breakout_hypothesis.py`

Tests:

- `tests/mbo_factory.py`
- `tests/test_alpha.py`
- `tests/test_cross_market_latency.py`
- `tests/test_ingestion.py`
- `tests/test_session.py`
- `tests/test_tick_stream.py`
- `tests/test_orchestrator_leakage_regressions.py`

Docs:

- `docs/f001_f005_remediation_report.md`

## Remaining Risks

- Full CME holiday/early-close handling is not implemented.
- Alpha discovery now has causal holdout selection, but not a full nested horizon and
  threshold optimizer.
- Existing permutation tests remain IID and should be upgraded in later lower-priority
  remediation.
- NQ-only mode no longer creates fake MNQ feature evidence, but some internal APIs still
  accept an MNQ argument for compatibility.
- Repo-wide ruff/mypy debt remains outside this scope.

## Findings Not Yet Addressed

The following lower-priority audit findings were intentionally not fixed:

- F-006 FeatureStore multi-instrument snapshot collapse
- F-007 global tick-stream/session/contract state beyond the F-003 reset boundary
- F-008 Volume Profile/Auction semantic mismatch
- F-009 SSL full-frame feature-column missingness selection
- F-010 IID permutation and multiple-testing weaknesses
- F-011 MBO reconstruction strictness
- F-012 duplicate timestamp/tie ordering
- F-013 execution realism
- F-014 bearish trap branch asymmetry

