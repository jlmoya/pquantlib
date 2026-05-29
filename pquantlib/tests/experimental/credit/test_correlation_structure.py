"""Tests for CorrelationTermStructure + CompoundCorrelationStructure +
BaseCorrelationStructure.

# C++ parity: ql/experimental/credit/correlationstructure.{hpp,cpp}
# + basecorrelationstructure.hpp. Tests are not cross-validated against the
# C++ probe (no closed-form C++ probe path covers these classes directly)
# but they exercise the structural invariants — interpolation values match
# explicit bilinear interpolation; observer registration propagates; loss
# levels and tenors validate.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.base_correlation_structure import (
    BaseCorrelationStructure,
)
from pquantlib.experimental.credit.correlation_structure import (
    CompoundCorrelationStructure,
    CorrelationTermStructure,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date, Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# -----------------------------------------------------------------------------
# Minimal scalar correlation curve for testing the aggregator
# -----------------------------------------------------------------------------


class _ConstantCorrelationCurve(CorrelationTermStructure):
    """Toy scalar correlation curve — returns a constant for the whole
    lifespan. Used only in CompoundCorrelationStructure tests."""

    __slots__ = ("_corr_quote", "_max_date")

    def __init__(self, ref: Date, corr_quote: SimpleQuote, max_date: Date) -> None:
        super().__init__(
            bdc=BusinessDayConvention.Following,
            calendar=TARGET(),
            day_counter=Actual365Fixed(),
            reference_date=ref,
        )
        self._corr_quote = corr_quote
        self._max_date = max_date
        self._corr_quote.register_with(self)

    def correlation_size(self) -> int:
        return 1

    def max_date(self) -> Date:
        return self._max_date

    def correlation(self) -> float:
        return self._corr_quote.value()


# -----------------------------------------------------------------------------
# CompoundCorrelationStructure tests
# -----------------------------------------------------------------------------


def test_compound_correlation_size_is_max_of_children() -> None:
    ref = Date.from_ymd(15, Month.January, 2024)
    max1 = Date.from_ymd(15, Month.January, 2025)
    max2 = Date.from_ymd(15, Month.January, 2026)
    child1 = _ConstantCorrelationCurve(ref, SimpleQuote(0.20), max1)
    child2 = _ConstantCorrelationCurve(ref, SimpleQuote(0.30), max2)
    compound = CompoundCorrelationStructure([child1, child2])
    assert compound.correlation_size() == 1
    # max_date = min of children
    assert compound.max_date() == max1


def test_compound_correlation_requires_consistent_calendar() -> None:
    ref = Date.from_ymd(15, Month.January, 2024)
    child1 = _ConstantCorrelationCurve(ref, SimpleQuote(0.20), Date.from_ymd(15, Month.January, 2025))

    # Create a curve with a different calendar
    class _OtherCurve(CorrelationTermStructure):
        __slots__ = ()

        def __init__(self) -> None:
            super().__init__(
                bdc=BusinessDayConvention.Following,
                calendar=WeekendsOnly(),
                day_counter=Actual365Fixed(),
                reference_date=ref,
            )

        def correlation_size(self) -> int:
            return 1

        def max_date(self) -> Date:
            return Date.from_ymd(15, Month.January, 2026)

    with pytest.raises(LibraryException, match="inconsistent calendars"):
        CompoundCorrelationStructure([child1, _OtherCurve()])


def test_compound_correlation_requires_at_least_one_child() -> None:
    with pytest.raises(LibraryException, match="at least one child"):
        CompoundCorrelationStructure([])


def test_compound_correlation_returns_copy_of_children() -> None:
    ref = Date.from_ymd(15, Month.January, 2024)
    child = _ConstantCorrelationCurve(ref, SimpleQuote(0.20), Date.from_ymd(15, Month.January, 2025))
    compound = CompoundCorrelationStructure([child])
    structures = compound.structures()
    structures.clear()
    # Internal list is unchanged
    assert len(compound.structures()) == 1


# -----------------------------------------------------------------------------
# BaseCorrelationStructure tests
# -----------------------------------------------------------------------------


def test_base_correlation_constructs_and_queries() -> None:
    cal = TARGET()
    dc = Actual365Fixed()
    tenors = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years), Period(5, TimeUnit.Years)]
    losses = [0.03, 0.06, 0.09]
    quotes = [
        [SimpleQuote(0.40), SimpleQuote(0.50), SimpleQuote(0.60)],  # tenor 1Y
        [SimpleQuote(0.45), SimpleQuote(0.55), SimpleQuote(0.65)],  # tenor 3Y
        [SimpleQuote(0.50), SimpleQuote(0.60), SimpleQuote(0.70)],  # tenor 5Y
    ]
    bcs = BaseCorrelationStructure(
        settlement_days=2,
        calendar=cal,
        bdc=BusinessDayConvention.Following,
        tenors=tenors,
        loss_levels=losses,
        correlation_quotes=quotes,
        day_counter=dc,
    )
    # On-grid query: tenor=1Y, loss=0.03 -> 0.40
    t0 = bcs.tranche_times()[0]
    tolerance.tight(bcs.correlation_at_time(t0, 0.03), 0.40)
    t1 = bcs.tranche_times()[1]
    tolerance.tight(bcs.correlation_at_time(t1, 0.06), 0.55)
    t2 = bcs.tranche_times()[2]
    tolerance.tight(bcs.correlation_at_time(t2, 0.09), 0.70)

    # Off-grid interpolation at the midpoint of (t0, t1) and (0.03, 0.06).
    tmid = 0.5 * (t0 + t1)
    lmid = 0.045
    # Bilinear interp expectation: average of four corner values.
    z_ll = 0.40
    z_lr = 0.45
    z_ul = 0.50
    z_ur = 0.55
    expected = 0.25 * (z_ll + z_lr + z_ul + z_ur)
    tolerance.tight(bcs.correlation_at_time(tmid, lmid), expected)


def test_base_correlation_rejects_non_monotone_loss_levels() -> None:
    quotes = [[SimpleQuote(0.40), SimpleQuote(0.50)] for _ in range(2)]
    with pytest.raises(LibraryException, match="non-increasing loss level"):
        BaseCorrelationStructure(
            settlement_days=2,
            calendar=TARGET(),
            bdc=BusinessDayConvention.Following,
            tenors=[Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)],
            loss_levels=[0.06, 0.03],  # non-monotone
            correlation_quotes=quotes,
            day_counter=Actual365Fixed(),
        )


def test_base_correlation_rejects_loss_level_above_one() -> None:
    quotes = [[SimpleQuote(0.40), SimpleQuote(0.50)] for _ in range(2)]
    with pytest.raises(LibraryException, match="100%"):
        BaseCorrelationStructure(
            settlement_days=2,
            calendar=TARGET(),
            bdc=BusinessDayConvention.Following,
            tenors=[Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)],
            loss_levels=[0.03, 1.5],
            correlation_quotes=quotes,
            day_counter=Actual365Fixed(),
        )


def test_base_correlation_quote_update_propagates() -> None:
    cal = TARGET()
    dc = Actual365Fixed()
    tenors = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)]
    losses = [0.03, 0.06]
    q00 = SimpleQuote(0.40)
    q01 = SimpleQuote(0.50)
    q10 = SimpleQuote(0.45)
    q11 = SimpleQuote(0.55)
    quotes = [[q00, q01], [q10, q11]]
    bcs = BaseCorrelationStructure(
        settlement_days=2,
        calendar=cal,
        bdc=BusinessDayConvention.Following,
        tenors=tenors,
        loss_levels=losses,
        correlation_quotes=quotes,
        day_counter=dc,
    )
    t0 = bcs.tranche_times()[0]
    tolerance.tight(bcs.correlation_at_time(t0, 0.03), 0.40)
    # Mutate quote -> matrix refresh on next call
    q00.set_value(0.42)
    tolerance.tight(bcs.correlation_at_time(t0, 0.03), 0.42)


def test_base_correlation_max_date_is_last_tranche_date() -> None:
    cal = TARGET()
    dc = Actual365Fixed()
    tenors = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)]
    losses = [0.03, 0.06]
    quotes = [
        [SimpleQuote(0.40), SimpleQuote(0.50)],
        [SimpleQuote(0.45), SimpleQuote(0.55)],
    ]
    bcs = BaseCorrelationStructure(
        settlement_days=2,
        calendar=cal,
        bdc=BusinessDayConvention.Following,
        tenors=tenors,
        loss_levels=losses,
        correlation_quotes=quotes,
        day_counter=dc,
    )
    # Last tranche date is reference_date + 3Y advanced.
    expected = cal.advance_period(bcs.reference_date(), Period(3, TimeUnit.Years), BusinessDayConvention.Following)
    assert bcs.max_date() == expected


def test_base_correlation_correlation_size_is_one() -> None:
    cal = TARGET()
    dc = Actual365Fixed()
    tenors = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)]
    losses = [0.03, 0.06]
    quotes = [
        [SimpleQuote(0.40), SimpleQuote(0.50)],
        [SimpleQuote(0.45), SimpleQuote(0.55)],
    ]
    bcs = BaseCorrelationStructure(
        settlement_days=2,
        calendar=cal,
        bdc=BusinessDayConvention.Following,
        tenors=tenors,
        loss_levels=losses,
        correlation_quotes=quotes,
        day_counter=dc,
    )
    assert bcs.correlation_size() == 1


def test_base_correlation_matrix_copy_is_defensive() -> None:
    cal = TARGET()
    dc = Actual365Fixed()
    tenors = [Period(1, TimeUnit.Years), Period(3, TimeUnit.Years)]
    losses = [0.03, 0.06]
    quotes = [
        [SimpleQuote(0.40), SimpleQuote(0.50)],
        [SimpleQuote(0.45), SimpleQuote(0.55)],
    ]
    bcs = BaseCorrelationStructure(
        settlement_days=2,
        calendar=cal,
        bdc=BusinessDayConvention.Following,
        tenors=tenors,
        loss_levels=losses,
        correlation_quotes=quotes,
        day_counter=dc,
    )
    m = bcs.correlations_matrix()
    m[0, 0] = 99.0
    # Internal matrix is unchanged.
    assert bcs.correlations_matrix()[0, 0] != 99.0
