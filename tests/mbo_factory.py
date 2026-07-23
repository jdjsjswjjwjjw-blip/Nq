"""مصنع بيانات MBO للاختبارات (test-only MBO builder)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.mbo import MBO_SCHEMA

# حدث مختصر: (action, side, price, size, order_id)
Event = tuple[str, str, int, int, int]


def make_stream(
    events: list[Event],
    *,
    instrument_id: int = 1,
    symbol: str = "NQ",
    event_ts: list[int] | None = None,
    ingest_ts: list[int] | None = None,
    sequence: list[int] | None = None,
) -> pl.DataFrame:
    """يبني إطار MBO صالحًا من قائمة أحداث مختصرة."""
    n = len(events)
    ts = event_ts if event_ts is not None else list(range(n))
    ingest = ingest_ts if ingest_ts is not None else ts
    seq = sequence if sequence is not None else list(range(1, n + 1))
    return pl.DataFrame(
        {
            "event_ts": ts,
            "ingest_ts": ingest,
            "sequence": seq,
            "instrument_id": [instrument_id] * n,
            "symbol": [symbol] * n,
            "action": [e[0] for e in events],
            "side": [e[1] for e in events],
            "price": [e[2] for e in events],
            "size": [e[3] for e in events],
            "order_id": [e[4] for e in events],
            "flags": [0] * n,
        },
        schema=MBO_SCHEMA,
    )


def random_add_cancel_stream(n: int, *, seed: int = 0) -> pl.DataFrame:
    """يولّد تدفّق أوامر عشوائيًا (إضافة/إلغاء) صالحًا ومرتّبًا سببيًا."""
    rng = np.random.default_rng(seed)
    active: list[int] = []
    events: list[Event] = []
    next_id = 1
    for _ in range(n):
        add = not active or rng.random() < 0.6
        if add:
            side = "B" if rng.random() < 0.5 else "A"
            price = int(rng.integers(19_990, 20_010)) * 1_000_000
            size = int(rng.integers(1, 10))
            oid = next_id
            next_id += 1
            active.append(oid)
            events.append(("A", side, price, size, oid))
        else:
            idx = int(rng.integers(0, len(active)))
            oid = active.pop(idx)
            events.append(("C", "N", 0, 0, oid))
    return make_stream(events)
