# Phase 0 Baseline

Date: 2026-07-23

Repository path: `/Users/abdallah/Desktop/Trading/Nq-main`

Remote requested by user: `https://github.com/jdjsjswjjwjjw-blip/Nq`

## Git Baseline

Local Git metadata is missing. `/Users/abdallah/Desktop/Trading/Nq-main` is not a Git worktree, so the required local commands currently fail:

```text
git status --short --branch
fatal: not a git repository (or any of the parent directories): .git

git log --oneline --decorate -20
fatal: not a git repository (or any of the parent directories): .git

git branch --show-current
fatal: not a git repository (or any of the parent directories): .git

git rev-parse HEAD
fatal: not a git repository (or any of the parent directories): .git
```

Nearby Git repositories under `/Users/abdallah/Desktop/Trading`:

```text
/Users/abdallah/Desktop/Trading/MNQVolumeAI/.git
/Users/abdallah/Desktop/Trading/QuantSystemFinal/.git
```

Remote `HEAD` observed with `git ls-remote`:

```text
5ef86c229b285c77331bf726048b602df68f5a99	HEAD
```

Risk: without local `.git`, there is no local commit SHA, local branch name, local diff base, or safe way to produce an exact Git diff summary for this folder. Treat the remote SHA above as a reference only, not proof that the current local files exactly match that commit.

## Python And Tooling

Declared project requirement from `pyproject.toml`:

```text
requires-python = ">=3.11"
```

Observed interpreters:

```text
python3 --version
Python 3.10.9

python3.11 --version
Python 3.11.3

python3.12
not found on PATH
```

Tool availability:

```text
uv --version
zsh:1: command not found: uv

which graphify
/Users/abdallah/anaconda3/bin/graphify

graphify --version
graphify 0.8.27
```

Dependency environment:

- Active default `python3` is Anaconda Python 3.10 and does not satisfy the project requirement.
- `python3.11` exists but does not have `pytest` installed.
- `uv` is not installed on PATH.
- `graphifyy==0.8.27` is installed in the Anaconda Python 3.10 environment.
- `python3` currently has at least `numpy==1.26.4`, `pytest==7.1.2`, and `zstandard==0.19.0`.
- `polars` is not installed in the default `python3` environment, blocking test collection.

Declared package dependencies:

```text
numpy>=1.26
polars>=1.0
pyarrow>=16.0
```

Declared dev dependencies:

```text
pytest>=8.0
pytest-cov>=5.0
ruff>=0.6
mypy>=1.11
hypothesis>=6.100
```

Reproducibility risk: dependency declarations are lower bounds only and there is no observed lock file (`uv.lock`, `poetry.lock`, etc.). Exact research reproducibility is not established at baseline.

## Repository Structure

Top-level directories:

```text
.github/
benchmarks/
configs/
data/
docs/
scripts/
src/
tests/
```

Primary package layout:

```text
src/nq/alpha/
src/nq/contracts/
src/nq/core/
src/nq/coverage/
src/nq/features/
src/nq/ingestion/
src/nq/models/
src/nq/orderbook/
src/nq/research/
src/nq/simulation/
src/nq/states/
src/nq/statistics/
src/nq/strategies/
src/nq/validation/
```

Approximate file counts excluding generated/cache/raw run output:

```text
total_files=190
code_files=129
doc_files=5
paper_files=0
config_files=7
```

Graphify detector baseline:

```text
total_files=127
total_words=52925
needs_graph=true
skipped_sensitive=[]
graphifyignore_patterns=24
scan_root=/Users/abdallah/Desktop/Trading/Nq-main
files_by_type={'code': 122, 'document': 5, 'paper': 0, 'image': 0, 'video': 0}
```

No research PDFs were found in this local tree at baseline. Documents available for Graphify are:

```text
.github/pull_request_template.md
.github/workflows/ci.yml
README.md
docs/architecture.md
docs/data_contracts.md
```

Generated audit output under `docs/deep_audit/` should be excluded from the Phase 1 baseline graph so reports created by this audit do not become evidence about the original system.

## Available Tests

Test files are present under `tests/`, including coverage across:

- MBO contracts and Databento ingestion
- order book reconstruction and causality
- depth lifecycle and integrity
- temporal policy and leakage utilities
- cross-market alignment and latency
- volume profile, footprint, order flow, liquidity, auction
- Failed FVG and Failed Breakout strategy flows
- SSL masking, splitting, pipeline, and enhancements
- execution simulation and statistics
- research orchestration and reporting

Baseline collection command:

```text
python3 -m pytest --collect-only -q
```

Result:

```text
no tests collected, 41 errors in 12.15s
```

Primary causes:

- `ModuleNotFoundError: No module named 'polars'`
- `ModuleNotFoundError: No module named 'nq'`

Interpretation: the current shell environment is not an installed project environment. Test failures at this stage are environment/setup failures, not evidence that implementation tests fail.

## Available Scripts

```text
scripts/run_week.py
scripts/run_fail_fvg.py
scripts/run_fail_breakout.py
scripts/run_vp_auction.py
benchmarks/bench_reconstruction.py
```

Entrypoint summary from direct script inspection:

- `run_week.py`: unified pipeline from NQ/MNQ MBO paths to `run_research_pipeline`.
- `run_fail_fvg.py`: focused Failed FVG run, with optional walk-forward hypothesis search.
- `run_fail_breakout.py`: focused Failed Breakout run, with optional volume/SSL hypothesis search.
- `run_vp_auction.py`: focused Volume Profile / auction balance-imbalance run.
- All runtime scripts enforce Python `>=3.11` before importing project modules.

## Configs

```text
configs/default.toml
configs/research.toml
configs/fail_breakout.toml
configs/fail_fvg.toml
configs/vp_auction.toml
```

Important baseline config claims from `configs/research.toml`:

- `interval_ns = 1_000_000_000`
- `horizon = 1`
- `embargo_ns = 1_000_000_000`
- `cross_market.latency_ns = 5_000_000`
- `execution.mode = "intraday"`
- `execution.tick_size = 0.25`
- `ssl.mode = "tick"`
- `ssl.window = 5`
- `features.mode = "streaming"`
- `data.cross_market_mode = "nq_only"`
- `data.max_rows = 0`

## Existing Architecture Claims

The README and `docs/architecture.md` describe this system as:

```text
MBO Raw
→ Ingestion + Order Book Reconstruction
→ Streaming State Machine or Simulation batch
→ Unified Feature Frame
→ SSL, M9 coverage monitor, alpha screen
→ ResearchAssistant
→ Unified Report
```

The codebase appears to implement corresponding modules for:

- Databento/MBO ingestion and contracts
- order book reconstruction, depth, and integrity
- streaming feature state
- footprint, volume profile, order flow, liquidity, auction, cross-market simulation
- Failed FVG and Failed Breakout strategies
- SSL masking/preprocessing/windowing/splitting/pipeline
- alpha discovery and signal evaluation
- statistical metrics, multiple testing, resampling, and hypothesis validation
- research orchestration and unified reporting

These are architecture claims only. Later audit phases must prove or refute them from source code and tests.

## Known Research Pipelines

Known runnable research paths, subject to environment setup and local MBO data availability:

```text
python scripts/run_week.py --config configs/research.toml --nq data/raw/nq.parquet --nq-only --max-rows 500000 --output data/runs/latest
python scripts/run_fail_fvg.py --nq data/raw/nq.parquet --search --max-rows 500000 --output data/runs/fail_fvg_search
python scripts/run_fail_breakout.py --nq data/raw/nq.parquet --search --max-rows 500000 --output data/runs/fail_breakout
python scripts/run_vp_auction.py --nq data/raw/nq.parquet --max-rows 500000 --output data/runs/vp_auction
```

No local real MBO data files were validated during Phase 0.

## Baseline Blockers To Resolve

1. The local folder is not a Git worktree, so original/final local Git diff guarantees are impossible unless `.git` is restored or the repo is recloned.
2. The default Python interpreter is below the project requirement.
3. No lock file pins dependencies for reproducible research.
4. The active environment cannot collect tests because `polars` and the installed package path are missing.
5. Graphify is available but on older CLI `0.8.27`; syntax must follow `graphify extract <path>`, not older or newer assumed forms.
