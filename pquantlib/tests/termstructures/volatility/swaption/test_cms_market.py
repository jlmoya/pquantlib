"""Tests for CmsMarket — CMS-coupon quote container.

# C++ parity: ql/termstructures/volatility/swaption/cmsmarket.{hpp,cpp}
# (v1.42.1). PQuantLib lands the structural-container subset only —
# repricing requires the deferred CmsCouponPricer port.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.cms_market import (
    CmsMarket,
    SwapIndexLike,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF_DATE = Date.from_ymd(15, Month.January, 2024)


class _FakeSwapIndex:
    """SwapIndexLike test double — only exposes ``tenor()``."""

    def __init__(self, tenor: Period) -> None:
        self._tenor = tenor

    def tenor(self) -> Period:
        return self._tenor


def _make_market() -> CmsMarket:
    swap_lengths = [Period(5, TimeUnit.Years), Period(10, TimeUnit.Years)]
    swap_indexes: list[SwapIndexLike] = [
        _FakeSwapIndex(Period(2, TimeUnit.Years)),
        _FakeSwapIndex(Period(5, TimeUnit.Years)),
        _FakeSwapIndex(Period(10, TimeUnit.Years)),
    ]
    bid_ask = [
        [
            (SimpleQuote(0.0010), SimpleQuote(0.0020)),
            (SimpleQuote(0.0015), SimpleQuote(0.0025)),
            (SimpleQuote(0.0018), SimpleQuote(0.0028)),
        ],
        [
            (SimpleQuote(0.0020), SimpleQuote(0.0030)),
            (SimpleQuote(0.0025), SimpleQuote(0.0035)),
            (SimpleQuote(0.0028), SimpleQuote(0.0038)),
        ],
    ]
    pricers: list[object] = [object(), object(), object()]
    discount = FlatForward(
        reference_date=_REF_DATE, forward=SimpleQuote(0.03),
        day_counter=Actual365Fixed(),
    )
    return CmsMarket(
        swap_lengths=swap_lengths,
        swap_indexes=swap_indexes,
        bid_ask_spreads=bid_ask,
        pricers=pricers,
        discount_curve=discount,
    )


# --- construction + invariants ----------------------------------------------


def test_construction_succeeds() -> None:
    market = _make_market()
    assert market.n_exercise == 2
    assert market.n_swap_indexes == 3


def test_swap_lengths_accessor_returns_input() -> None:
    market = _make_market()
    lengths = market.swap_lengths()
    assert len(lengths) == 2
    assert lengths[0] == Period(5, TimeUnit.Years)
    assert lengths[1] == Period(10, TimeUnit.Years)


def test_swap_tenors_derived_from_indexes() -> None:
    """# C++ parity: ``swap_tenors_`` populated from each index's tenor()."""
    market = _make_market()
    tenors = market.swap_tenors()
    assert len(tenors) == 3
    assert tenors[0] == Period(2, TimeUnit.Years)
    assert tenors[1] == Period(5, TimeUnit.Years)
    assert tenors[2] == Period(10, TimeUnit.Years)


def test_bid_ask_spreads_accessor_returns_defensive_copy() -> None:
    market = _make_market()
    row0 = market.bid_ask_spreads()[0]
    # The accessor returns a fresh list — mutating it doesn't affect storage.
    row0.clear()
    assert len(market.bid_ask_spreads()[0]) == 3


def test_pricer_accessor_returns_per_swap_index() -> None:
    market = _make_market()
    p0 = market.pricer(0)
    p1 = market.pricer(1)
    assert p0 is not p1


def test_pricer_index_out_of_range() -> None:
    market = _make_market()
    with pytest.raises(LibraryException, match="out of range"):
        market.pricer(99)


def test_empty_swap_lengths_rejected() -> None:
    swap_indexes: list[SwapIndexLike] = [_FakeSwapIndex(Period(2, TimeUnit.Years))]
    discount = FlatForward(
        reference_date=_REF_DATE, forward=SimpleQuote(0.03),
        day_counter=Actual365Fixed(),
    )
    with pytest.raises(LibraryException, match="empty swap_lengths"):
        CmsMarket(
            swap_lengths=[],
            swap_indexes=swap_indexes,
            bid_ask_spreads=[],
            pricers=[object()],
            discount_curve=discount,
        )


def test_pricers_count_mismatch_rejected() -> None:
    swap_lengths = [Period(5, TimeUnit.Years)]
    swap_indexes: list[SwapIndexLike] = [
        _FakeSwapIndex(Period(2, TimeUnit.Years)),
        _FakeSwapIndex(Period(5, TimeUnit.Years)),
    ]
    discount = FlatForward(
        reference_date=_REF_DATE, forward=SimpleQuote(0.03),
        day_counter=Actual365Fixed(),
    )
    with pytest.raises(LibraryException, match="pricers"):
        CmsMarket(
            swap_lengths=swap_lengths,
            swap_indexes=swap_indexes,
            bid_ask_spreads=[
                [(SimpleQuote(0.001), SimpleQuote(0.002)),
                 (SimpleQuote(0.002), SimpleQuote(0.003))]
            ],
            pricers=[object()],  # wrong length
            discount_curve=discount,
        )


def test_bid_ask_row_size_mismatch_rejected() -> None:
    swap_lengths = [Period(5, TimeUnit.Years)]
    swap_indexes: list[SwapIndexLike] = [
        _FakeSwapIndex(Period(2, TimeUnit.Years)),
        _FakeSwapIndex(Period(5, TimeUnit.Years)),
    ]
    discount = FlatForward(
        reference_date=_REF_DATE, forward=SimpleQuote(0.03),
        day_counter=Actual365Fixed(),
    )
    with pytest.raises(LibraryException, match="row"):
        CmsMarket(
            swap_lengths=swap_lengths,
            swap_indexes=swap_indexes,
            # Row has only 1 entry but 2 swap indexes expected.
            bid_ask_spreads=[[(SimpleQuote(0.001), SimpleQuote(0.002))]],
            pricers=[object(), object()],
            discount_curve=discount,
        )


def test_reprice_raises_documented_stub() -> None:
    """The reprice() method documents that CmsCouponPricer isn't ported."""
    market = _make_market()
    with pytest.raises(LibraryException, match="CmsCouponPricer"):
        market.reprice(object(), 0.05)


def test_weighted_spread_error_raises_documented_stub() -> None:
    market = _make_market()
    with pytest.raises(LibraryException, match="CmsCouponPricer"):
        market.weighted_spread_error(object())
