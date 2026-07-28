"""Causal quantitative understanding layers (diagnostic only).

These analyses **never** participate in candidate selection. Metrics are
computed on purged walk-forward OOS test indices only — the same temporal
discipline as ``purged_walk_forward_split``. They explain *why* a selected
signal behaves as it does; they do not invent new alpha.

Layers
------
1. **Ablation** — peel ``__ssl`` / ``__depth__*`` / ``__enh__*`` and remeasure
   OOS Spearman IC; Benjamini–Hochberg FDR within the ablation family.
2. **Regime map** — OOS IC stratified by ``session_phase``.
3. **Gate attribution** — pass-rate of each peeled layer vs its base on OOS;
   contemporaneous |selected|↔|base| Spearman (not forward return).
4. **Temporal stability** — fold-level OOS ``test_ic`` from selection folds
   (already purged); mean, std, positive-fold rate.
5. **Depth counterfactual** — OOS IC with depth layer on vs off, plus a
   **label-permutation** null for the on−off delta (OOS labels only).
6. **SSL state link** — contemporaneous mean |z| ↔ |signal| on OOS
   (state association, **not** forward-looking alpha).

Scientific stance
-----------------
- No external narratives, no LLM text, no look-ahead joins.
- All columns must already exist as PIT columns in ``features``.
- Label permutation shuffles *future returns within OOS* only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.temporal_policy import TemporalPolicy
from nq.models.splitting import purged_walk_forward_split
from nq.research.progress import PipelineProgress, resolve_progress
from nq.statistics.metrics import information_coefficient
from nq.statistics.multiple_testing import benjamini_hochberg

DEPTH_MARK = "__depth__"
ENH_MARK = "__enh__"
SSL_SUFFIX = "__ssl"
SESSION_COL = "session_phase"
_MIN_OOS = 30
_Z_PREFIX = "z"


@dataclass(frozen=True, slots=True)
class AblationRow:
    layer: str
    ablated_column: str
    oos_ic_full: float
    oos_ic_ablated: float
    delta_ic: float
    n_oos: int
    p_value: float
    significant_bh: bool


@dataclass(frozen=True, slots=True)
class RegimeRow:
    regime: str
    oos_ic: float
    n_oos: int
    p_value: float


@dataclass(frozen=True, slots=True)
class AttributionRow:
    layer: str
    pass_rate: float
    abs_selected_base_spearman: float
    n_oos: int


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    fold_ics: tuple[float, ...]
    mean_ic: float
    std_ic: float
    positive_fold_rate: float
    n_folds: int


@dataclass(frozen=True, slots=True)
class DepthCounterfactual:
    oos_ic_with_depth: float
    oos_ic_without_depth: float
    delta_ic: float
    perm_p_value: float
    n_oos: int
    n_permutations: int
    with_column: str
    without_column: str


@dataclass(frozen=True, slots=True)
class SslStateLink:
    embedding_dim: int
    abs_z_abs_signal_spearman: float
    n_oos: int
    p_value: float


@dataclass(frozen=True, slots=True)
class UnderstandingFinding:
    claim: str
    metric: str
    value: float
    p_value: float
    significant_bh: bool
    detail: str


@dataclass(frozen=True, slots=True)
class UnderstandingReport:
    """Bundle of diagnostic layers for one selected candidate."""

    selected_column: str
    base_column: str
    layers: tuple[str, ...]
    ablation: tuple[AblationRow, ...] = ()
    regimes: tuple[RegimeRow, ...] = ()
    attribution: tuple[AttributionRow, ...] = ()
    stability: StabilitySummary | None = None
    depth_cf: DepthCounterfactual | None = None
    ssl_link: SslStateLink | None = None
    notes: tuple[str, ...] = ()
    findings: tuple[UnderstandingFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_column": self.selected_column,
            "base_column": self.base_column,
            "layers": list(self.layers),
            "ablation": [asdict(r) for r in self.ablation],
            "regimes": [asdict(r) for r in self.regimes],
            "attribution": [asdict(r) for r in self.attribution],
            "stability": asdict(self.stability) if self.stability is not None else None,
            "depth_cf": asdict(self.depth_cf) if self.depth_cf is not None else None,
            "ssl_link": asdict(self.ssl_link) if self.ssl_link is not None else None,
            "notes": list(self.notes),
            "findings": [asdict(f) for f in self.findings],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Causal quantitative understanding (OOS diagnostics)",
            "",
            f"- selected: `{self.selected_column}`",
            f"- base: `{self.base_column}`",
            f"- layers: `{list(self.layers)}`",
            "",
            "## Findings",
            "",
        ]
        if not self.findings:
            lines.append("_No findings (insufficient OOS or missing columns)._")
        for f in self.findings:
            sig = "BH*" if f.significant_bh else ""
            lines.append(
                f"- **{f.claim}** `{f.metric}`={f.value:.6g} p={f.p_value:.4g}{sig} — {f.detail}"
            )
        lines.extend(["", "## Notes", ""])
        for n in self.notes:
            lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)


def resolve_base_column(
    selected: str,
    columns: set[str] | frozenset[str],
) -> tuple[str, tuple[str, ...]]:
    """Peel ``__ssl`` / ``__depth__*`` / ``__enh__*`` from the right to recover base.

    Returns ``(base_column, layers_outer_to_inner)`` where each layer is a
    human-readable tag such as ``ssl``, ``depth:pressure_q0p7``, ``enh:z0_abs``.
    Intermediate columns must exist in ``columns`` when peeled.
    """
    layers: list[str] = []
    cur = selected
    changed = True
    while changed:
        changed = False
        if cur.endswith(SSL_SUFFIX):
            parent = cur[: -len(SSL_SUFFIX)]
            if parent in columns:
                layers.append("ssl")
                cur = parent
                changed = True
                continue
        for mark, kind in ((DEPTH_MARK, "depth"), (ENH_MARK, "enh")):
            if mark in cur:
                head, _, tail = cur.rpartition(mark)
                if head and tail and head in columns:
                    layers.append(f"{kind}:{tail}")
                    cur = head
                    changed = True
                    break
    if cur not in columns:
        return selected, ()
    return cur, tuple(layers)


def peel_one(selected: str, columns: set[str] | frozenset[str]) -> tuple[str, str] | None:
    """Remove the outermost layer; return ``(parent_column, layer_tag)``."""
    if selected.endswith(SSL_SUFFIX):
        parent = selected[: -len(SSL_SUFFIX)]
        if parent in columns:
            return parent, "ssl"
    for mark, kind in ((DEPTH_MARK, "depth"), (ENH_MARK, "enh")):
        if mark in selected:
            head, _, tail = selected.rpartition(mark)
            if head and tail and head in columns:
                return head, f"{kind}:{tail}"
    return None


def oos_test_indices(
    times: np.ndarray,
    *,
    interval_ns: int,
    ssl_window: int,
    n_splits: int,
    horizon: int = 1,
) -> np.ndarray:
    """Sorted unique row indices that fall in any purged WF **test** fold."""
    policy = TemporalPolicy.for_run(interval_ns=interval_ns, window=ssl_window, horizon=horizon)
    embargo = policy.embargo_time_units(interval_ns=interval_ns, times=times)
    folds = purged_walk_forward_split(
        times,
        n_splits=n_splits,
        embargo=embargo,
        purge_samples=policy.purge_samples(),
        min_train_size=max(10, int(times.shape[0]) // (n_splits + 2)),
    )
    if not folds:
        return np.asarray([], dtype=np.intp)
    return np.unique(np.concatenate([f.test_idx for f in folds])).astype(np.intp)


def _align_forward_returns(prices: np.ndarray, *, horizon: int = 1) -> np.ndarray:
    """Forward returns for evaluation labels only (not features)."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    p = np.asarray(prices, dtype=np.float64)
    n = p.shape[0]
    fwd = np.full(n, np.nan, dtype=np.float64)
    if n > horizon:
        base = p[: n - horizon]
        future = p[horizon:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(base != 0, (future - base) / base, np.nan)
        fwd[: n - horizon] = ret
    return fwd


def _oos_ic(
    signal: np.ndarray,
    forward: np.ndarray,
    idx: np.ndarray,
    *,
    name: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, int, float]:
    """Spearman IC + permutation p-value on ``idx`` only."""
    del name  # kept for call-site clarity / future logging
    if idx.size == 0:
        return 0.0, 0, 1.0
    v = np.asarray(signal[idx], dtype=np.float64)
    f = np.asarray(forward[idx], dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(f)
    v, f = v[mask], f[mask]
    n = int(v.shape[0])
    if n < min(_MIN_OOS, 8) or float(np.std(v)) == 0.0:
        return 0.0, n, 1.0
    observed = information_coefficient(v, f, method="spearman")
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        null[i] = information_coefficient(v, rng.permutation(f), method="spearman")
    p = float((int(np.sum(np.abs(null) >= abs(observed))) + 1) / (n_permutations + 1))
    return float(observed), n, p


def _ablation_layer(
    *,
    selected: str,
    work: pl.DataFrame,
    forward: np.ndarray,
    oos_idx: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[AblationRow, ...]:
    cols = set(work.columns)
    if selected not in cols or oos_idx.size == 0:
        return ()
    full = work[selected].to_numpy().astype(np.float64)
    full_ic, n_full, _ = _oos_ic(
        full, forward, oos_idx, name=selected, n_permutations=n_permutations, rng=rng
    )
    rows: list[AblationRow] = []
    raw_p: list[float] = []
    # Ablate each single outer-to-inner layer by comparing selected vs that parent chain
    cur = selected
    while True:
        peeled = peel_one(cur, cols)
        if peeled is None:
            break
        parent, layer = peeled
        parent_vals = work[parent].to_numpy().astype(np.float64)
        ic_a, n_a, p_a = _oos_ic(
            parent_vals,
            forward,
            oos_idx,
            name=parent,
            n_permutations=n_permutations,
            rng=rng,
        )
        # Delta = contribution of this layer: IC(with layer ancestors) - IC(without this layer)
        # Here full is always the original selected; ablated is parent of current peel step.
        # For multi-layer, also compute selected vs immediate parent for outermost,
        # then parent vs grandparent, etc.
        with_vals = work[cur].to_numpy().astype(np.float64)
        ic_with, n_with, _ = _oos_ic(
            with_vals,
            forward,
            oos_idx,
            name=cur,
            n_permutations=n_permutations,
            rng=rng,
        )
        delta = ic_with - ic_a
        rows.append(
            AblationRow(
                layer=layer,
                ablated_column=parent,
                oos_ic_full=ic_with,
                oos_ic_ablated=ic_a,
                delta_ic=delta,
                n_oos=min(n_with, n_a),
                p_value=p_a,
                significant_bh=False,
            )
        )
        raw_p.append(float(np.clip(p_a, 0.0, 1.0)))
        cur = parent

    if not rows:
        return ()
    # Also record overall selected vs ultimate base using first/last
    _ = full_ic, n_full
    bh = benjamini_hochberg(raw_p)
    out: list[AblationRow] = []
    for row, reject in zip(rows, bh.reject, strict=True):
        out.append(
            AblationRow(
                layer=row.layer,
                ablated_column=row.ablated_column,
                oos_ic_full=row.oos_ic_full,
                oos_ic_ablated=row.oos_ic_ablated,
                delta_ic=row.delta_ic,
                n_oos=row.n_oos,
                p_value=row.p_value,
                significant_bh=bool(reject),
            )
        )
    return tuple(out)


def _regime_layer(
    signal: np.ndarray,
    forward: np.ndarray,
    phases: np.ndarray,
    oos_idx: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[RegimeRow, ...]:
    if oos_idx.size == 0:
        return ()
    rows: list[RegimeRow] = []
    oos_phases = phases[oos_idx]
    for regime in sorted(set(int(x) for x in oos_phases if np.isfinite(x))):
        sub = oos_idx[oos_phases == regime]
        ic, n, p = _oos_ic(
            signal,
            forward,
            sub,
            name=f"regime_{regime}",
            n_permutations=n_permutations,
            rng=rng,
        )
        if n < _MIN_OOS:
            continue
        rows.append(RegimeRow(regime=str(regime), oos_ic=ic, n_oos=n, p_value=p))
    return tuple(rows)


def _attribution_layer(
    *,
    selected: str,
    work: pl.DataFrame,
    oos_idx: np.ndarray,
) -> tuple[AttributionRow, ...]:
    cols = set(work.columns)
    if selected not in cols or oos_idx.size == 0:
        return ()
    rows: list[AttributionRow] = []
    cur = selected
    while True:
        peeled = peel_one(cur, cols)
        if peeled is None:
            break
        parent, layer = peeled
        sel = work[cur].to_numpy().astype(np.float64)[oos_idx]
        base = work[parent].to_numpy().astype(np.float64)[oos_idx]
        finite = np.isfinite(sel) & np.isfinite(base)
        base_nz = finite & (base != 0.0)
        if int(base_nz.sum()) < _MIN_OOS:
            cur = parent
            continue
        passed = base_nz & (sel != 0.0)
        pass_rate = float(passed.sum() / base_nz.sum())
        spear = information_coefficient(
            np.abs(sel[finite]),
            np.abs(base[finite]),
            method="spearman",
        )
        rows.append(
            AttributionRow(
                layer=layer,
                pass_rate=pass_rate,
                abs_selected_base_spearman=float(spear),
                n_oos=int(finite.sum()),
            )
        )
        cur = parent
    return tuple(rows)


def _stability_layer(fold_selections: pl.DataFrame) -> StabilitySummary | None:
    if fold_selections.height == 0 or "test_ic" not in fold_selections.columns:
        return None
    ics = tuple(float(x) for x in fold_selections["test_ic"].to_list())
    arr = np.asarray(ics, dtype=np.float64)
    return StabilitySummary(
        fold_ics=ics,
        mean_ic=float(np.mean(arr)),
        std_ic=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        positive_fold_rate=float(np.mean(arr > 0.0)),
        n_folds=len(ics),
    )


def _depth_counterfactual(
    *,
    selected: str,
    work: pl.DataFrame,
    forward: np.ndarray,
    oos_idx: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> DepthCounterfactual | None:
    if DEPTH_MARK not in selected or selected not in work.columns:
        return None
    cols = set(work.columns)
    # Find depth parent: peel until we remove a depth layer
    cur = selected
    without: str | None = None
    while True:
        peeled = peel_one(cur, cols)
        if peeled is None:
            break
        parent, layer = peeled
        if layer.startswith("depth:"):
            without = parent
            break
        cur = parent
    if without is None or without not in work.columns:
        return None

    with_vals = work[selected].to_numpy().astype(np.float64)
    without_vals = work[without].to_numpy().astype(np.float64)
    ic_on, n_on, _ = _oos_ic(
        with_vals, forward, oos_idx, name=selected, n_permutations=n_permutations, rng=rng
    )
    ic_off, n_off, _ = _oos_ic(
        without_vals, forward, oos_idx, name=without, n_permutations=n_permutations, rng=rng
    )
    delta = ic_on - ic_off
    n = min(n_on, n_off)
    if n < _MIN_OOS or n_permutations <= 0 or oos_idx.size == 0:
        return DepthCounterfactual(
            oos_ic_with_depth=ic_on,
            oos_ic_without_depth=ic_off,
            delta_ic=delta,
            perm_p_value=float("nan"),
            n_oos=n,
            n_permutations=0,
            with_column=selected,
            without_column=without,
        )

    s_on = with_vals[oos_idx]
    s_off = without_vals[oos_idx]
    y = forward[oos_idx]
    finite = np.isfinite(s_on) & np.isfinite(s_off) & np.isfinite(y)
    s_on, s_off, y = s_on[finite], s_off[finite], y[finite]
    null_deltas = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(y)
        null_deltas[i] = information_coefficient(
            s_on, perm, method="spearman"
        ) - information_coefficient(s_off, perm, method="spearman")
    p = float((np.sum(np.abs(null_deltas) >= abs(delta)) + 1) / (n_permutations + 1))
    return DepthCounterfactual(
        oos_ic_with_depth=ic_on,
        oos_ic_without_depth=ic_off,
        delta_ic=delta,
        perm_p_value=p,
        n_oos=n,
        n_permutations=n_permutations,
        with_column=selected,
        without_column=without,
    )


def _ssl_state_link(
    signal: np.ndarray,
    work: pl.DataFrame,
    embeddings: pl.DataFrame | None,
    oos_idx: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> SslStateLink | None:
    if oos_idx.size == 0:
        return None
    z_cols = [c for c in work.columns if c == "z0" or (c.startswith(_Z_PREFIX) and c[1:].isdigit())]
    intensity: np.ndarray | None = None
    dim = 0
    if z_cols:
        z = work.select(z_cols).to_numpy().astype(np.float64)
        intensity = np.nanmean(np.abs(z), axis=1)
        dim = len(z_cols)
    elif embeddings is not None and embeddings.height > 0:
        emb_z = [
            c
            for c in embeddings.columns
            if c == "z0" or (c.startswith(_Z_PREFIX) and c[1:].isdigit())
        ]
        if not emb_z or AVAILABILITY_TS not in embeddings.columns:
            return None
        left = work.select(AVAILABILITY_TS).with_row_index("_i")
        right = embeddings.select(AVAILABILITY_TS, *emb_z).sort(AVAILABILITY_TS)
        joined = left.sort(AVAILABILITY_TS).join_asof(
            right, on=AVAILABILITY_TS, strategy="backward"
        )
        joined = joined.sort("_i")
        z = joined.select(emb_z).to_numpy().astype(np.float64)
        intensity = np.nanmean(np.abs(z), axis=1)
        dim = len(emb_z)
    if intensity is None:
        return None

    s = np.abs(signal[oos_idx])
    inten = intensity[oos_idx]
    finite = np.isfinite(s) & np.isfinite(inten)
    if int(finite.sum()) < _MIN_OOS:
        return None
    # Contemporaneous association — permute intensity under null (features fixed timeline).
    obs = information_coefficient(inten[finite], s[finite], method="spearman")
    null = np.empty(n_permutations, dtype=np.float64)
    y = s[finite]
    x = inten[finite]
    for i in range(n_permutations):
        null[i] = information_coefficient(rng.permutation(x), y, method="spearman")
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_permutations + 1))
    return SslStateLink(
        embedding_dim=dim,
        abs_z_abs_signal_spearman=float(obs),
        n_oos=int(finite.sum()),
        p_value=p,
    )


def run_understanding_layers(
    features: pl.DataFrame,
    *,
    selected_column: str,
    fold_selections: pl.DataFrame,
    embeddings: pl.DataFrame | None = None,
    price_col: str = "nq_close",
    horizon: int = 1,
    interval_ns: int,
    ssl_window: int = 5,
    n_splits: int = 3,
    n_permutations: int = 200,
    seed: int = 7,
    progress: PipelineProgress | bool | None = None,
    quiet: bool = False,
) -> UnderstandingReport:
    """Run all diagnostic layers for ``selected_column`` (OOS-only)."""
    log = resolve_progress(progress, quiet=quiet)
    notes: list[str] = []
    findings: list[UnderstandingFinding] = []
    rng = np.random.default_rng(seed)

    if selected_column not in features.columns:
        raise KeyError(f"selected_column not in features: {selected_column}")
    if price_col not in features.columns:
        raise KeyError(f"price_col not in features: {price_col}")
    if AVAILABILITY_TS not in features.columns:
        raise KeyError(f"features must include {AVAILABILITY_TS}")

    work = features.sort(AVAILABILITY_TS)
    colset = set(work.columns)
    base, layers = resolve_base_column(selected_column, colset)
    if base == selected_column and not layers:
        notes.append("no_gate_suffix_parsed_using_selected_as_base")

    times = work[AVAILABILITY_TS].to_numpy()
    prices = work[price_col].to_numpy().astype(np.float64)
    forward = _align_forward_returns(prices, horizon=horizon)
    oos_idx = oos_test_indices(
        times,
        interval_ns=interval_ns,
        ssl_window=ssl_window,
        n_splits=n_splits,
        horizon=horizon,
    )
    signal = work[selected_column].to_numpy().astype(np.float64)

    log.op(f"understanding: selected={selected_column!r} · oos_n={oos_idx.size:,}")

    log.op("understanding_ablation")
    ablation = _ablation_layer(
        selected=selected_column,
        work=work,
        forward=forward,
        oos_idx=oos_idx,
        n_permutations=n_permutations,
        rng=rng,
    )
    for abl in ablation:
        findings.append(
            UnderstandingFinding(
                claim=f"ablation_{abl.layer}",
                metric="oos_spearman_ic_delta",
                value=abl.delta_ic,
                p_value=abl.p_value,
                significant_bh=abl.significant_bh,
                detail=(
                    f"with={abl.oos_ic_full:.6f} without={abl.oos_ic_ablated:.6f} "
                    f"ablated_col={abl.ablated_column} n_oos={abl.n_oos}"
                ),
            )
        )

    log.op("understanding_regime")
    if SESSION_COL in work.columns:
        phases = work[SESSION_COL].to_numpy().astype(np.float64)
        regimes = _regime_layer(
            signal, forward, phases, oos_idx, n_permutations=n_permutations, rng=rng
        )
    else:
        regimes = ()
        notes.append("session_phase_missing_regime_layer_skipped")
    for reg in regimes:
        findings.append(
            UnderstandingFinding(
                claim=f"regime_{reg.regime}",
                metric="oos_spearman_ic",
                value=reg.oos_ic,
                p_value=reg.p_value,
                significant_bh=False,
                detail=f"n_oos={reg.n_oos}",
            )
        )

    log.op("understanding_attribution")
    attribution = _attribution_layer(selected=selected_column, work=work, oos_idx=oos_idx)

    log.op("understanding_stability")
    stability = _stability_layer(fold_selections)
    if stability is not None:
        findings.append(
            UnderstandingFinding(
                claim="temporal_stability_mean_fold_ic",
                metric="mean_oos_fold_ic",
                value=stability.mean_ic,
                p_value=float("nan"),
                significant_bh=False,
                detail=(
                    f"std={stability.std_ic:.6f} positive_rate={stability.positive_fold_rate:.3f} "
                    f"n_folds={stability.n_folds}"
                ),
            )
        )

    log.op("understanding_depth_cf")
    depth_cf = _depth_counterfactual(
        selected=selected_column,
        work=work,
        forward=forward,
        oos_idx=oos_idx,
        n_permutations=n_permutations,
        rng=rng,
    )
    if depth_cf is not None:
        findings.append(
            UnderstandingFinding(
                claim="depth_gate_counterfactual",
                metric="oos_ic_on_minus_off",
                value=depth_cf.delta_ic,
                p_value=depth_cf.perm_p_value,
                significant_bh=False,
                detail=(
                    f"on={depth_cf.oos_ic_with_depth:.6f} off={depth_cf.oos_ic_without_depth:.6f} "
                    f"n_perm={depth_cf.n_permutations} n_oos={depth_cf.n_oos}"
                ),
            )
        )

    log.op("understanding_ssl_link")
    ssl_link = _ssl_state_link(
        signal, work, embeddings, oos_idx, n_permutations=n_permutations, rng=rng
    )
    if ssl_link is not None:
        findings.append(
            UnderstandingFinding(
                claim="ssl_state_absz_abssignal",
                metric="oos_spearman_contemporaneous",
                value=ssl_link.abs_z_abs_signal_spearman,
                p_value=ssl_link.p_value,
                significant_bh=False,
                detail=f"dim={ssl_link.embedding_dim} n_oos={ssl_link.n_oos} (not forward alpha)",
            )
        )
        notes.append("ssl_link_is_contemporaneous_state_association_not_forward_ic")

    notes.append("all_metrics_restricted_to_purged_wf_oos_test_folds")
    notes.append("understanding_does_not_alter_candidate_selection")

    return UnderstandingReport(
        selected_column=selected_column,
        base_column=base,
        layers=layers,
        ablation=ablation,
        regimes=regimes,
        attribution=attribution,
        stability=stability,
        depth_cf=depth_cf,
        ssl_link=ssl_link,
        notes=tuple(notes),
        findings=tuple(findings),
    )


def write_understanding_outputs(
    report: UnderstandingReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write parquet tables + markdown report under ``output_dir/understanding``."""
    root = Path(output_dir) / "understanding"
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if report.ablation:
        p = root / "ablation.parquet"
        pl.DataFrame([asdict(r) for r in report.ablation]).write_parquet(p)
        paths["ablation"] = p
    if report.regimes:
        p = root / "regimes.parquet"
        pl.DataFrame([asdict(r) for r in report.regimes]).write_parquet(p)
        paths["regimes"] = p
    if report.attribution:
        p = root / "attribution.parquet"
        pl.DataFrame([asdict(r) for r in report.attribution]).write_parquet(p)
        paths["attribution"] = p
    if report.stability is not None:
        p = root / "stability.parquet"
        pl.DataFrame(
            {
                "fold": list(range(report.stability.n_folds)),
                "test_ic": list(report.stability.fold_ics),
            }
        ).write_parquet(p)
        paths["stability"] = p
    if report.depth_cf is not None:
        p = root / "depth_counterfactual.parquet"
        pl.DataFrame([asdict(report.depth_cf)]).write_parquet(p)
        paths["depth_cf"] = p
    if report.ssl_link is not None:
        p = root / "ssl_state_link.parquet"
        pl.DataFrame([asdict(report.ssl_link)]).write_parquet(p)
        paths["ssl_link"] = p

    md = root / "report.md"
    md.write_text(report.to_markdown(), encoding="utf-8")
    paths["report"] = md

    # Flatten summary for parquet (JSON-serializable scalars only)
    summary = {
        "selected_column": report.selected_column,
        "base_column": report.base_column,
        "n_layers": len(report.layers),
        "n_findings": len(report.findings),
        "mean_fold_ic": report.stability.mean_ic if report.stability else float("nan"),
        "depth_delta_ic": report.depth_cf.delta_ic if report.depth_cf else float("nan"),
        "ssl_absz_spearman": (
            report.ssl_link.abs_z_abs_signal_spearman if report.ssl_link else float("nan")
        ),
    }
    meta = root / "summary.parquet"
    pl.DataFrame([summary]).write_parquet(meta)
    paths["summary"] = meta
    return paths


__all__ = [
    "AblationRow",
    "AttributionRow",
    "DepthCounterfactual",
    "RegimeRow",
    "SslStateLink",
    "StabilitySummary",
    "UnderstandingFinding",
    "UnderstandingReport",
    "oos_test_indices",
    "peel_one",
    "resolve_base_column",
    "run_understanding_layers",
    "write_understanding_outputs",
]
