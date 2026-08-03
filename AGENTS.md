# AGENTS.md

## Cursor Cloud specific instructions

`nq` is a pure-Python (>=3.11, CI uses 3.12) quantitative market-microstructure
research **library** — there is no long-running server or GUI. Work is exercised
through the test suite and short driver scripts that import `nq`. Standard dev
commands live in `README.md` ("Local Dev & Quality Gates") and
`.github/workflows/ci.yml`; the quality gates are `ruff check src tests`,
`ruff format --check src tests`, `mypy`, and `pytest --cov`.

Non-obvious notes for running things here:

- Dependencies are installed into a project-local virtualenv at `.venv` (created
  by the startup update script). Run tools via that venv without activating,
  e.g. `.venv/bin/pytest`, `.venv/bin/ruff check src tests`, `.venv/bin/mypy`.
  (`python3 -m venv` requires the system `python3.12-venv` package, which is
  already present in the VM image.)
- The `tests/` directory is a package used by tests (e.g. `tests.mbo_factory`
  builds synthetic MBO streams) but the installed `nq` package does not include
  it. To run an ad-hoc driver script that imports from `tests`, run it with the
  repo root on the path: `PYTHONPATH=. .venv/bin/python your_script.py`.
- `ruff check` currently reports 2 pre-existing `PLR0917` findings in
  `src/nq/orderbook/reconstruction.py`. These come from a newer pinned ruff
  (`ruff>=0.6` resolves to 0.16.x) enabling `too-many-positional-arguments`;
  they are not caused by environment setup. `ruff format --check`, `mypy`
  (strict, clean), and `pytest` (all tests pass) are green.
- Everything is deterministic by design: use `nq.core.determinism.make_generator`
  / `seed_everything` so results reproduce from the same raw inputs + seed.
- The full research path (`nq.alpha.run_research_pipeline`) goes raw MBO →
  order book reconstruction → cross-market features → statistically screened
  alpha signals → an evidence-backed report; it is a good smoke test of the
  whole stack.
