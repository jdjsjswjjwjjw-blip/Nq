"""قياس إنتاجية إعادة بناء دفتر الأوامر (throughput benchmark).

يُشغّل يدويًا (خارج CI):

    python benchmarks/bench_reconstruction.py --events 2000000

يقيس عدد الأحداث المُعالَجة في الثانية مع وبدون تسجيل top-of-book.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from nq.contracts.mbo import MBO_SCHEMA
from nq.orderbook import reconstruct


def _synthetic_stream(n: int, *, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    prices = (rng.integers(19_990, 20_010, size=n) * 1_000_000).astype(np.int64)
    sizes = rng.integers(1, 10, size=n).astype(np.uint32)
    sides = np.where(rng.random(n) < 0.5, "B", "A")
    return pl.DataFrame(
        {
            "event_ts": np.arange(n, dtype=np.int64),
            "ingest_ts": np.arange(n, dtype=np.int64),
            "sequence": np.arange(1, n + 1, dtype=np.uint64),
            "instrument_id": np.ones(n, dtype=np.uint32),
            "symbol": ["NQ"] * n,
            "action": ["A"] * n,
            "side": sides,
            "price": prices,
            "size": sizes,
            "order_id": np.arange(1, n + 1, dtype=np.uint64),
            "flags": np.zeros(n, dtype=np.uint8),
        },
        schema=MBO_SCHEMA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = _synthetic_stream(args.events, seed=args.seed)

    for label, record in (("with top-of-book", True), ("state-only", False)):
        start = time.perf_counter()
        reconstruct(frame, record_top_of_book=record)
        elapsed = time.perf_counter() - start
        rate = args.events / elapsed if elapsed else float("inf")
        print(f"{label:>18}: {args.events:,} events in {elapsed:.3f}s -> {rate:,.0f} ev/s")


if __name__ == "__main__":
    main()
