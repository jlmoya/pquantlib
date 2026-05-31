"""Generic inflation indexes — name/region/frequency/lag wiring.

# C++ parity: ql/experimental/inflation/genericindexes.hpp (v1.42.1).
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.experimental.inflation.generic_indexes import GenericCPI, YYGenericCPI
from pquantlib.indexes.inflation.region import Region
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def test_generic_region_payload() -> None:
    # C++ GenericRegion Data("Generic","GENERIC").
    assert Region.Generic.region_name() == "Generic"
    assert Region.Generic.region_code() == "GENERIC"


def test_generic_cpi_identity() -> None:
    idx = GenericCPI(
        Frequency.Monthly, False, Period(2, TimeUnit.Months), EURCurrency()
    )
    # InflationIndex.name() == "<region.name> <familyName>".
    assert idx.name() == "Generic CPI"
    assert idx.frequency() == Frequency.Monthly
    assert idx.availability_lag() == Period(2, TimeUnit.Months)
    assert idx.region() == Region.Generic
    assert not idx.revised()
    # ZeroInflationIndex is always non-interpolated.
    assert not idx.interpolated()
    assert idx.currency() == EURCurrency()


def test_yy_generic_cpi_identity() -> None:
    idx = YYGenericCPI(
        Frequency.Monthly, False, Period(3, TimeUnit.Months), Currency()
    )
    assert idx.name() == "Generic YY_CPI"
    assert idx.frequency() == Frequency.Monthly
    assert idx.availability_lag() == Period(3, TimeUnit.Months)
    # Quoted-mode YoY index is not a ratio.
    assert not idx.ratio()
    assert idx.underlying_index() is None
    # Null currency by default in the stripper's "fake index".
    assert idx.currency().empty()


def test_yy_generic_cpi_interpolated_flag() -> None:
    idx = YYGenericCPI(
        Frequency.Monthly,
        False,
        Period(3, TimeUnit.Months),
        Currency(),
        interpolated=True,
    )
    assert idx.interpolated()
