"""Cross-cluster Protocol conformance tests for the inflation cluster.

Verifies that the L7-A concrete InflationIndex / InflationTermStructure
classes structurally satisfy the L7-B/C/D Protocols at runtime via
``isinstance`` (since they're ``@runtime_checkable``). Type-time
conformance is a separate (silent) pyright check.
"""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.inflation.eu_hicp import EUHICP, YoYEUHICP
from pquantlib.indexes.inflation.fr_hicp import FRHICP, YoYFRHICP
from pquantlib.indexes.inflation.protocols import (
    InflationIndexProtocol,
    InflationTermStructureProtocol,
)
from pquantlib.indexes.inflation.uk_hicp import UKHICP
from pquantlib.indexes.inflation.uk_rpi import UKRPI, YoYUKRPI
from pquantlib.indexes.inflation.us_cpi import USCPI, YoYUSCPI
from pquantlib.termstructures.inflation.yoy_inflation_term_structure import (
    YoYInflationTermStructure,
)
from pquantlib.termstructures.inflation.zero_inflation_term_structure import (
    ZeroInflationTermStructure,
)
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

# ---- InflationIndexProtocol -----------------------------------------


def test_all_zero_concretes_satisfy_inflation_index_protocol() -> None:
    """Every L7-A zero-inflation concrete duck-types InflationIndexProtocol."""
    for ctor in (EUHICP, FRHICP, UKRPI, UKHICP, USCPI):
        assert isinstance(ctor(), InflationIndexProtocol)


def test_all_yoy_concretes_satisfy_inflation_index_protocol() -> None:
    """Every L7-A YoY concrete duck-types InflationIndexProtocol."""
    for ctor in (YoYEUHICP, YoYFRHICP, YoYUKRPI, YoYUSCPI):
        assert isinstance(ctor(), InflationIndexProtocol)


def test_inflation_index_protocol_accessor_round_trip() -> None:
    """All Protocol-listed accessors return values of the expected type."""
    eu: InflationIndexProtocol = EUHICP()
    assert isinstance(eu.name(), str)
    assert isinstance(eu.family_name(), str)
    assert isinstance(eu.frequency(), Frequency)
    assert eu.interpolated() is False
    assert eu.revised() is False


# ---- InflationTermStructureProtocol ---------------------------------


class _StubZeroCurve(ZeroInflationTermStructure):
    def __init__(self) -> None:
        super().__init__(
            base_date=Date.from_ymd(1, Month.January, 2020),
            frequency=Frequency.Monthly,
            day_counter=Actual365Fixed(),
            reference_date=Date.from_ymd(1, Month.February, 2020),
        )
        # base_rate is optional; provide it so base_rate() doesn't raise.
        self._base_rate = 0.02

    def _zero_rate_impl(self, t: float) -> float:
        del t
        return 0.02

    def max_date(self) -> Date:
        return Date.from_ymd(31, Month.December, 2050)


class _StubYoYCurve(YoYInflationTermStructure):
    def __init__(self) -> None:
        super().__init__(
            base_date=Date.from_ymd(1, Month.January, 2020),
            base_yoy_rate=0.018,
            frequency=Frequency.Monthly,
            day_counter=Actual365Fixed(),
            reference_date=Date.from_ymd(1, Month.February, 2020),
        )

    def _yoy_rate_impl(self, t: float) -> float:
        del t
        return 0.018

    def max_date(self) -> Date:
        return Date.from_ymd(31, Month.December, 2050)


def test_zero_inflation_curve_stub_satisfies_protocol() -> None:
    assert isinstance(_StubZeroCurve(), InflationTermStructureProtocol)


def test_yoy_inflation_curve_stub_satisfies_protocol() -> None:
    assert isinstance(_StubYoYCurve(), InflationTermStructureProtocol)


def test_inflation_termstructure_protocol_accessors_thread() -> None:
    """All Protocol-listed accessors return values of the expected type."""
    ts: InflationTermStructureProtocol = _StubZeroCurve()
    assert isinstance(ts.reference_date(), Date)
    assert isinstance(ts.max_date(), Date)
    assert isinstance(ts.base_date(), Date)
    assert isinstance(ts.frequency(), Frequency)
    assert ts.observation_lag() is None
    assert ts.nominal_term_structure() is None
    assert isinstance(ts.base_rate(), float)


# ---- structural-typing negative test --------------------------------


def test_unrelated_class_does_not_satisfy_inflation_index_protocol() -> None:
    """A plain object with no Protocol surface must not duck-type."""

    class _NotAnIndex:
        pass

    assert isinstance(_NotAnIndex(), InflationIndexProtocol) is False


def test_unrelated_class_does_not_satisfy_inflation_ts_protocol() -> None:
    class _NotACurve:
        pass

    assert isinstance(_NotACurve(), InflationTermStructureProtocol) is False
