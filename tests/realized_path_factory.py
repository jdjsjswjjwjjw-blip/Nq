"""Synthetic path geometry so period/year tests exercise realized-path Y."""

from __future__ import annotations

from typing import Any

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS

#: First realized transitions used in synthetic stories (not scenario phases).
RESOLVING_PATH_KINDS = (
    "further_beyond_asia",
    "return_inside_asia_va",
    "value_built",
    "value_migrated",
    "continue_direction",
    "reverse_path",
)

_EVENT_BAR = 2


def path_kind_for_index(index: int) -> str:
    return RESOLVING_PATH_KINDS[int(index) % len(RESOLVING_PATH_KINDS)]


def path_bar_fields(kind: str, bar: int) -> dict[str, float]:
    """Pulse a realized setup at bar 0, then one next-path event at bar 2.

    Break without retest is the default onset. Scenario ``proj_*`` columns are
    left to the caller — they stay annotations, not this layer's Y.
    """
    beyond = 0.0
    inside = 1.0
    if kind == "return_inside_asia_va":
        beyond = 2.0
        inside = 1.0 if bar >= _EVENT_BAR else 0.0
    elif kind == "further_beyond_asia":
        beyond = 2.0 if bar >= _EVENT_BAR else 0.0
        inside = 0.0 if bar >= _EVENT_BAR else 1.0

    event = bar == _EVENT_BAR
    return {
        "path_beyond_asia_ticks": beyond,
        "path_inside_asia_va": inside,
        "proj_poc_step_ticks": 5.0 if event and kind == "value_migrated" else 0.0,
        "vp_fsm_break": 1.0 if bar == 0 else 0.0,
        "vp_fsm_retest": 0.0,
        "vp_fsm_expand": 1.0 if event and kind == "continue_direction" else 0.0,
        "vp_look_fail": 1.0 if event and kind == "reverse_path" else 0.0,
        "vp_absorb": 1.0 if event and kind == "value_built" else 0.0,
        "vp_close_in_value": 1.0 if event and kind == "value_built" else 0.0,
        "vp_balance": 1.0 if event and kind == "value_built" else 0.0,
    }


def attach_realized_path_geometry(frame: pl.DataFrame) -> pl.DataFrame:
    """Stamp break-onset plus a cycling next-path event onto episode groups.

    Groups come from ``_behavior_story_run`` when present; otherwise each
    contiguous 8-bar block is one story so concatenated months still mix classes.
    """
    if frame.height == 0:
        return frame
    work = frame.sort(AVAILABILITY_TS) if AVAILABILITY_TS in frame.columns else frame
    n = work.height
    if "_behavior_story_run" in work.columns:
        groups = [int(x) for x in work["_behavior_story_run"].to_list()]
    else:
        groups = [i // 8 for i in range(n)]

    seen: dict[int, int] = {}
    kind_i = 0
    last_g: int | None = None
    bar = 0
    fields: list[dict[str, float]] = []
    for g in groups:
        if g not in seen:
            seen[g] = kind_i
            kind_i += 1
        if g != last_g:
            bar = 0
            last_g = g
        fields.append(path_bar_fields(path_kind_for_index(seen[g]), bar))
        bar += 1

    updates: dict[str, list[Any]] = {name: [row[name] for row in fields] for name in fields[0]}
    return work.with_columns([pl.Series(name, values) for name, values in updates.items()])
