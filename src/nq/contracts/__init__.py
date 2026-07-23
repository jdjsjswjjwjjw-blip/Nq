"""عقود البيانات (Data Contracts) — التعريف الصارم لبنية بيانات MBO والحقول الزمنية.

Data contracts are the single, versioned definition of every field entering the
system. All higher layers derive *exclusively* from these contracts.
"""

from __future__ import annotations

from nq.contracts.instruments import (
    CONTRACT_ID,
    INSTRUMENT_ID,
    MNQ_METADATA,
    NQ_METADATA,
    SYMBOL,
    InstrumentMetadata,
    contract_identity,
    contract_identity_values,
    first_instrument_metadata,
    instrument_metadata,
    require_single_contract_identity,
    root_symbol,
    unique_contract_identities,
)
from nq.contracts.mbo import (
    MBO_SCHEMA,
    MboAction,
    MboEvent,
    MboSide,
    validate_mbo_frame,
)
from nq.contracts.temporal import (
    AVAILABILITY_TS,
    EVENT_TS,
    INGEST_TS,
    SEQUENCE,
    TemporalFields,
)

__all__ = [
    "AVAILABILITY_TS",
    "CONTRACT_ID",
    "EVENT_TS",
    "INSTRUMENT_ID",
    "INGEST_TS",
    "MBO_SCHEMA",
    "MNQ_METADATA",
    "NQ_METADATA",
    "SEQUENCE",
    "SYMBOL",
    "InstrumentMetadata",
    "MboAction",
    "MboEvent",
    "MboSide",
    "TemporalFields",
    "contract_identity",
    "contract_identity_values",
    "first_instrument_metadata",
    "instrument_metadata",
    "require_single_contract_identity",
    "root_symbol",
    "unique_contract_identities",
    "validate_mbo_frame",
]
