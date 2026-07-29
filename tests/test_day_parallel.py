"""اختبارات التشغيل اليومي المتوازي — عزل سببي عبر الأيام."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nq.research.day_parallel import (
    DayInput,
    day_id_from_path,
    discover_day_inputs,
    run_fail_breakout_day_parallel,
    stable_day_seed,
)
from tests.mbo_factory import Event, make_stream


def _day_mbo(path: Path, *, n: int, seed_price: int) -> None:
    events: list[Event] = []
    ts: list[int] = []
    seq: list[int] = []
    price = seed_price
    for i in range(n):
        t = i * 50_000_000  # 50ms
        events.extend(
            [
                ("A", "B", price, 5, i * 2 + 1),
                ("A", "A", price + 1_000_000, 5, i * 2 + 2),
                ("T", "B", price, 1, 0),
            ]
        )
        ts.extend([t, t + 1, t + 2])
        seq.extend([i * 3 + 1, i * 3 + 2, i * 3 + 3])
        price += 250_000
    frame = make_stream(events, instrument_id=1, symbol="NQ", event_ts=ts, sequence=seq)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_day_id_from_path_parses_dates() -> None:
    assert day_id_from_path(Path("nq_2026-07-01.parquet")) == "2026-07-01"
    assert day_id_from_path(Path("mnq_20260702.parquet")) == "2026-07-02"
    assert day_id_from_path(Path("NQ_2026_07_03.parquet")) == "2026-07-03"


def test_stable_day_seed_deterministic_and_distinct() -> None:
    a = stable_day_seed(0, "2026-07-01")
    b = stable_day_seed(0, "2026-07-01")
    c = stable_day_seed(0, "2026-07-02")
    assert a == b
    assert a != c


def test_discover_pairs_mnq_by_name(tmp_path: Path) -> None:
    nq = tmp_path / "nq"
    mnq = tmp_path / "mnq"
    nq.mkdir()
    mnq.mkdir()
    (nq / "2026-07-01.parquet").write_bytes(b"")
    (nq / "2026-07-02.parquet").write_bytes(b"")
    (mnq / "2026-07-01.parquet").write_bytes(b"")
    # touch empty files are is_file — discover only checks is_file
    days = discover_day_inputs(
        nq_paths=[nq / "2026-07-01.parquet", nq / "2026-07-02.parquet"],
        mnq_dir=mnq,
    )
    assert len(days) == 2
    assert days[0].day_id == "2026-07-01"
    assert days[0].mnq_path is not None
    assert days[1].mnq_path is None  # no matching MNQ for day 2


def test_day_parallel_search_isolates_days(tmp_path: Path) -> None:
    d1 = tmp_path / "nq_2026-07-01.parquet"
    d2 = tmp_path / "nq_2026-07-02.parquet"
    _day_mbo(d1, n=40, seed_price=20_000_000_000)
    _day_mbo(d2, n=40, seed_price=20_100_000_000)
    out = tmp_path / "runs"
    days = (
        DayInput(day_id="2026-07-01", nq_path=d1),
        DayInput(day_id="2026-07-02", nq_path=d2),
    )
    # jobs=1 في الاختبارات لثبات بيئة CI؛ التوازي ProcessPool يُغطى عبر CLI/التشغيل
    manifest = run_fail_breakout_day_parallel(
        days,
        output_root=out,
        mode="search",
        jobs=1,
        threads_per_worker=1,
        n_splits=2,
        n_permutations=20,
        use_ssl_gate=False,
        enhance_with_ssl=False,
        use_depth_filter=False,
        quiet_workers=True,
    )
    assert (out / "manifest.json").is_file()
    assert (out / "summary.md").is_file()
    assert manifest.n_days == 2
    assert manifest.n_ok == 2
    # كل يوم مجلد منفصل — لا ملف ميزات موحّد عبر الأيام
    assert (out / "2026-07-01" / "features.parquet").is_file()
    assert (out / "2026-07-02" / "features.parquet").is_file()
    f1 = pl.read_parquet(out / "2026-07-01" / "features.parquet")
    f2 = pl.read_parquet(out / "2026-07-02" / "features.parquet")
    assert f1.height > 0 and f2.height > 0
    assert "zero_temporal_leakage" in " ".join(manifest.principles)
    assert any("no_cross_day_selection" in p for p in manifest.principles)


def test_day_parallel_payload_builds_unique_seeds() -> None:
    """ضمان أن بذور الأيام لا تتصادم (حتمية + تمايز)."""
    seeds = {stable_day_seed(42, f"2026-07-{d:02d}") for d in range(1, 31)}
    assert len(seeds) == 30
