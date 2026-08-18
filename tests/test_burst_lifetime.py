"""عمر الأمر حول حزم T: وصف لا سبب ولا سبوفينج."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import nq.research
from nq.research.burst_lifetime import (
    BurstWindow,
    score_burst_lifetimes,
    write_burst_lifetime_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_NS = 1_000_000_000


def _ns(clock: str) -> int:
    hour, minute, second = (int(p) for p in clock.split(":"))
    stamp = dt.datetime(2025, 6, 3, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "score_burst_lifetimes" not in nq.research.__all__
    assert not hasattr(nq.research, "score_burst_lifetimes")


def test_resting_death_vs_fleeting_and_live_snapshot(tmp_path: Path) -> None:
    mbo = make_stream(
        [
            ("A", "B", _PX, 8, 1),
            ("C", "B", _PX, 8, 1),
            ("A", "A", _PX, 5, 2),
            ("C", "A", _PX, 5, 2),
            ("A", "B", _PX, 4, 3),
            ("T", "B", _PX, 4, 3),
            ("A", "B", _PX, 9, 4),
        ],
        event_ts=[
            _ns("10:59:00"),
            _ns("11:00:10"),
            _ns("11:00:05"),
            _ns("11:00:05") + _NS // 20,
            _ns("11:02:10"),
            _ns("11:02:12"),
            _ns("11:04:00"),
        ],
        sequence=list(range(1, 8)),
    )
    windows = (
        BurstWindow("win_a", _ns("11:00:00"), _ns("11:01:00"), _ns("11:00:20")),
        BurstWindow("win_b", _ns("11:02:00"), _ns("11:03:00"), _ns("11:02:11")),
        BurstWindow("control", _ns("11:05:00"), _ns("11:06:00"), _ns("11:05:30")),
    )
    table, diag = score_burst_lifetimes(mbo, windows)
    assert diag["not_cause_lock"] is True
    assert diag["fleeting_is_legal_spoofing"] is False
    a = table.filter(table["name"] == "win_a").row(0, named=True)
    b = table.filter(table["name"] == "win_b").row(0, named=True)
    ctrl = table.filter(table["name"] == "control").row(0, named=True)
    assert a["n_resting_death"] == 1
    assert a["n_fleeting"] == 1
    assert a["median_resting_life_ms"] > a["p25_life_ms"]
    assert b["n_genuine"] == 1
    assert ctrl["n_live_at_burst"] == 1
    assert ctrl["median_live_age_ms"] > 30_000
    written = write_burst_lifetime_report(table, diag, tmp_path)
    text = (written / "BURST_LIFETIME.md").read_text(encoding="utf-8")
    assert "Not a cause lock" in text
    assert (written / "burst_lifetime.parquet").exists()
