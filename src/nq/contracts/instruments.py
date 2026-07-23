"""Instrument metadata and contract lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from nq.contracts.mbo import PRICE_SCALE

SYMBOL: Final = "symbol"
INSTRUMENT_ID: Final = "instrument_id"
CONTRACT_ID: Final = "contract_id"


@dataclass(frozen=True, slots=True)
class InstrumentMetadata:
    """Canonical exchange metadata needed by leakage-safe simulations."""

    root_symbol: str
    tick_size: float
    point_value: float
    price_scale: float = PRICE_SCALE

    @property
    def tick_size_fixed(self) -> int:
        """Tick size in the repository's fixed-point integer price units."""
        return int(round(self.tick_size / self.price_scale))


NQ_METADATA: Final = InstrumentMetadata(root_symbol="NQ", tick_size=0.25, point_value=20.0)
MNQ_METADATA: Final = InstrumentMetadata(root_symbol="MNQ", tick_size=0.25, point_value=2.0)
_METADATA_BY_ROOT: Final = {
    NQ_METADATA.root_symbol: NQ_METADATA,
    MNQ_METADATA.root_symbol: MNQ_METADATA,
}


def root_symbol(symbol: str) -> str:
    """Return the supported futures root from a raw symbol/contract string."""
    normalized = str(symbol).strip().upper()
    if normalized.startswith("MNQ"):
        return "MNQ"
    if normalized.startswith("NQ"):
        return "NQ"
    raise ValueError(f"unsupported instrument symbol {symbol!r}; explicit metadata is required")


def instrument_metadata(symbol: str) -> InstrumentMetadata:
    """Return canonical metadata for a supported NQ/MNQ root or contract."""
    return _METADATA_BY_ROOT[root_symbol(symbol)]


def first_instrument_metadata(frame: pl.DataFrame, *, default_symbol: str = "NQ") -> InstrumentMetadata:
    """Infer metadata from the first available symbol, falling back to an explicit default."""
    if SYMBOL in frame.columns and frame.height > 0:
        for symbol in frame[SYMBOL].to_list():
            if str(symbol).strip():
                return instrument_metadata(str(symbol))
    return instrument_metadata(default_symbol)


def contract_identity(*, symbol: object | None, instrument_id: object | None) -> str:
    """Build an explicit contract identity from source symbol and instrument id."""
    symbol_text = "" if symbol is None else str(symbol).strip().upper()
    instrument_text = "" if instrument_id is None else str(int(instrument_id))
    if symbol_text and instrument_text:
        return f"{symbol_text}#{instrument_text}"
    if symbol_text:
        return symbol_text
    if instrument_text:
        return f"instrument_id:{instrument_text}"
    raise ValueError("contract identity requires symbol or instrument_id metadata")


def contract_identity_values(
    frame: pl.DataFrame,
    *,
    symbol_col: str = SYMBOL,
    instrument_col: str = INSTRUMENT_ID,
) -> list[str]:
    """Return per-row source contract identities for lifecycle auditing."""
    if frame.height == 0:
        return []
    symbols = frame[symbol_col].to_list() if symbol_col in frame.columns else [None] * frame.height
    instruments = (
        frame[instrument_col].to_list() if instrument_col in frame.columns else [None] * frame.height
    )
    return [
        contract_identity(symbol=symbol, instrument_id=instrument_id)
        for symbol, instrument_id in zip(symbols, instruments, strict=True)
    ]


def unique_contract_identities(frame: pl.DataFrame) -> tuple[str, ...]:
    """Distinct contract identities in first-seen order."""
    return tuple(dict.fromkeys(contract_identity_values(frame)))


def require_single_contract_identity(
    frame: pl.DataFrame,
    *,
    context: str,
) -> None:
    """Fail safely when a stateful calculation spans a contract roll."""
    identities = unique_contract_identities(frame)
    if len(identities) > 1:
        raise ValueError(
            f"{context}: contract roll / contract identity change detected "
            f"({', '.join(identities)}); explicit contract lifecycle configuration is required"
        )


__all__ = [
    "CONTRACT_ID",
    "INSTRUMENT_ID",
    "MNQ_METADATA",
    "NQ_METADATA",
    "SYMBOL",
    "InstrumentMetadata",
    "contract_identity",
    "contract_identity_values",
    "first_instrument_metadata",
    "instrument_metadata",
    "require_single_contract_identity",
    "root_symbol",
    "unique_contract_identities",
]
