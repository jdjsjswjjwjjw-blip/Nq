"""طبقة تحقّق صارمة: خصائص (property-based) + ميتامورفية عبر كل الأدوات.

تختبر هذه الطبقة **ثوابت رياضية يجب أن تصمد على كل المدخلات** (لا أمثلة مختارة)
عبر توليد آلاف الحالات العشوائية بـ Hypothesis، إضافةً إلى علاقات ميتامورفية
تحت تحويلات معروفة. لا تلمس هذه الطبقة كود المصدر (تعيش في ``tests/`` فقط).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from nq.models.encoder import PCAEncoder
from nq.models.preprocessing import CausalStandardScaler
from nq.orderbook import reconstruct
from nq.simulation.footprint import footprint_cells, footprint_summary
from nq.simulation.order_flow import order_flow_imbalance
from nq.simulation.volume_profile import build_volume_profile, value_area
from nq.states import KMeansRegimes
from nq.statistics import benjamini_hochberg, permutation_test, sharpe_ratio
from tests.mbo_factory import Event, make_stream, random_add_cancel_stream

_SETTINGS = settings(max_examples=40, deadline=None)
_INTERVAL = 10
_TICK = 1
_BASE = 100

TradeSpec = list[tuple[str, int, int]]  # (side, price_offset, size)


@st.composite
def _trade_specs(draw: st.DrawFn) -> TradeSpec:
    n = draw(st.integers(min_value=1, max_value=30))
    return [
        (
            draw(st.sampled_from(["B", "A"])),
            draw(st.integers(min_value=-5, max_value=5)),
            draw(st.integers(min_value=1, max_value=20)),
        )
        for _ in range(n)
    ]


def _trades(spec: TradeSpec, *, shift: int = 0) -> pl.DataFrame:
    events: list[Event] = [("T", side, _BASE + off * _TICK, size, 0) for side, off, size in spec]
    ts = [i + shift for i in range(len(spec))]
    return make_stream(events, event_ts=ts, sequence=[i + 1 for i in range(len(spec))])


@st.composite
def _matrix(draw: st.DrawFn) -> npt.NDArray[np.float64]:
    rows = draw(st.integers(min_value=3, max_value=20))
    cols = draw(st.integers(min_value=1, max_value=5))
    flat = draw(
        st.lists(
            st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
            min_size=rows * cols,
            max_size=rows * cols,
        )
    )
    return np.asarray(flat, dtype=np.float64).reshape(rows, cols)


# --- طبقة المحاكاة: Footprint ------------------------------------------------


@_SETTINGS
@given(spec=_trade_specs())
def test_footprint_volume_decomposition(spec: TradeSpec) -> None:
    cells = footprint_cells(_trades(spec), interval_ns=_INTERVAL)
    total = cells["buy_volume"] + cells["sell_volume"]
    assert cells["total_volume"].to_list() == total.to_list()
    imb = cells["imbalance"].to_numpy()
    assert np.all((imb >= -1.0 - 1e-9) & (imb <= 1.0 + 1e-9))


@_SETTINGS
@given(spec=_trade_specs())
def test_footprint_cumulative_delta_is_cumsum(spec: TradeSpec) -> None:
    summary = footprint_summary(_trades(spec), interval_ns=_INTERVAL).sort("bucket_start")
    expected = np.cumsum(summary["delta"].to_numpy())
    assert summary["cumulative_delta"].to_list() == expected.tolist()


@_SETTINGS
@given(spec=_trade_specs())
def test_footprint_side_flip_negates_delta(spec: TradeSpec) -> None:
    """ميتامورفي: قلب جوانب الصفقات يقلب إشارة الدلتا."""
    flipped: TradeSpec = [("A" if s == "B" else "B", off, sz) for s, off, sz in spec]
    base = footprint_summary(_trades(spec), interval_ns=_INTERVAL).sort("bucket_start")
    flip = footprint_summary(_trades(flipped), interval_ns=_INTERVAL).sort("bucket_start")
    assert base["delta"].to_list() == [-d for d in flip["delta"].to_list()]


@_SETTINGS
@given(spec=_trade_specs(), k=st.integers(min_value=1, max_value=5))
def test_footprint_time_shift_invariance(spec: TradeSpec, k: int) -> None:
    """ميتامورفي: إزاحة الزمن بمضاعف النافذة تُبقي القيم وتُزيح الطوابع فقط."""
    shift = k * _INTERVAL
    base = footprint_summary(_trades(spec), interval_ns=_INTERVAL).sort("bucket_start")
    shifted = footprint_summary(_trades(spec, shift=shift), interval_ns=_INTERVAL).sort(
        "bucket_start"
    )
    assert base["delta"].to_list() == shifted["delta"].to_list()
    assert [b + shift for b in base["bucket_start"].to_list()] == shifted[
        "bucket_start"
    ].to_list()


# --- Volume Profile ----------------------------------------------------------


@_SETTINGS
@given(spec=_trade_specs(), fraction=st.floats(min_value=0.5, max_value=0.9))
def test_value_area_bounds_and_coverage(spec: TradeSpec, fraction: float) -> None:
    profile = build_volume_profile(_trades(spec))
    va = value_area(profile, fraction=fraction)
    assert va is not None
    assert va.val <= va.poc <= va.vah
    assert va.value_volume >= fraction * va.total_volume - 1e-9
    assert va.poc_volume == max(profile["volume"].to_list())


# --- إعادة بناء الدفتر: حفظ الحجم -------------------------------------------


@_SETTINGS
@given(n=st.integers(min_value=1, max_value=80), seed=st.integers(min_value=0, max_value=10_000))
def test_orderbook_volume_conservation(n: int, seed: int) -> None:
    """ثابت بنيوي: مجموع أحجام المستويات = مجموع أحجام الأوامر المتتبَّعة."""
    frame = random_add_cancel_stream(n, seed=seed)
    book = reconstruct(frame, record_top_of_book=False).book
    level_total = sum(book.bids.values()) + sum(book.asks.values())
    order_total = sum(size for _, _, size in book.orders.values())
    assert level_total == order_total
    assert all(v > 0 for v in book.bids.values())
    assert all(v > 0 for v in book.asks.values())


@_SETTINGS
@given(n=st.integers(min_value=2, max_value=80), seed=st.integers(min_value=0, max_value=10_000))
def test_ofi_cumulative_consistency(n: int, seed: int) -> None:
    tob = reconstruct(random_add_cancel_stream(n, seed=seed)).top_of_book
    ofi = order_flow_imbalance(tob)
    assert ofi["ofi"].to_list()[0] == 0
    assert ofi["ofi_cumulative"].to_list() == np.cumsum(ofi["ofi"].to_numpy()).tolist()


# --- الأدوات العددية ---------------------------------------------------------


@_SETTINGS
@given(x=_matrix())
def test_causal_scaler_zero_mean(x: npt.NDArray[np.float64]) -> None:
    transformed = CausalStandardScaler().fit_transform(x)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)


@_SETTINGS
@given(x=_matrix())
def test_pca_full_rank_reconstructs(x: npt.NDArray[np.float64]) -> None:
    k = min(x.shape)
    recon = PCAEncoder(n_components=k).fit(x).reconstruct(x)
    np.testing.assert_allclose(recon, x, atol=1e-6)


@_SETTINGS
@given(
    pvals=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=20,
    )
)
def test_benjamini_hochberg_bounds(pvals: list[float]) -> None:
    res = benjamini_hochberg(pvals, alpha=0.05)
    assert np.all((res.adjusted >= 0.0) & (res.adjusted <= 1.0))
    # كل مرفوضة يجب أن تكون قيمتها المُعدّلة <= alpha
    assert np.all(res.adjusted[res.reject] <= 0.05 + 1e-12)


@_SETTINGS
@given(
    returns=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=100,
    )
)
def test_sharpe_sign_matches_mean(returns: list[float]) -> None:
    arr = np.asarray(returns, dtype=np.float64)
    if float(np.std(arr, ddof=1)) == 0:
        return
    assert np.sign(sharpe_ratio(arr)) == np.sign(np.mean(arr))


@_SETTINGS
@given(
    a=st.lists(st.floats(-5, 5, allow_nan=False), min_size=3, max_size=50),
    b=st.lists(st.floats(-5, 5, allow_nan=False), min_size=3, max_size=50),
)
def test_permutation_pvalue_in_unit_interval(a: list[float], b: list[float]) -> None:
    res = permutation_test(a, b, n_permutations=200, rng=np.random.default_rng(0))
    assert 0.0 < res.pvalue <= 1.0


@_SETTINGS
@given(x=_matrix(), seed=st.integers(min_value=0, max_value=1000))
def test_kmeans_labels_valid_and_deterministic(x: npt.NDArray[np.float64], seed: int) -> None:
    k = min(3, x.shape[0])
    a = KMeansRegimes(k, seed=seed).fit_predict(x)
    b = KMeansRegimes(k, seed=seed).fit_predict(x)
    np.testing.assert_array_equal(a, b)
    assert np.all((a >= 0) & (a < k))
