"""Generic placeholder inflation indexes (for testing / stripping).

# C++ parity: ql/experimental/inflation/genericindexes.hpp (v1.42.1) —
   ``GenericRegion`` + ``GenericCPI`` + ``YYGenericCPI``.

These exist so the YoY optionlet stripper can build a "fake" YoY index
carrying only the right frequency / lag / currency, independent of any
particular economy. The ``GenericRegion`` is modelled as
``Region.Generic`` (name ``"Generic"``, code ``"GENERIC"``) rather than a
dedicated class, matching PQuantLib's :class:`Region` ``IntEnum`` port of
the C++ region hierarchy.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.indexes.inflation.inflation_index import (
    YoYInflationIndex,
    ZeroInflationIndex,
)
from pquantlib.indexes.inflation.region import Region
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class GenericCPI(ZeroInflationIndex):
    """Generic CPI zero-inflation index (family ``"CPI"``, region Generic).

    # C++ parity: ``GenericCPI`` (genericindexes.hpp:44-53).
    """

    def __init__(
        self,
        frequency: Frequency,
        revised: bool,
        lag: Period,
        ccy: Currency,
        ts: object | None = None,
    ) -> None:
        super().__init__(
            family_name="CPI",
            region=Region.Generic,
            revised=revised,
            frequency=frequency,
            availability_lag=lag,
            currency=ccy,
            ts=ts,
        )


class YYGenericCPI(YoYInflationIndex):
    """Quoted year-on-year generic CPI (family ``"YY_CPI"``, region Generic).

    # C++ parity: ``YYGenericCPI`` (genericindexes.hpp:57-94). The C++ class
    # has a deprecated ``interpolated``-taking overload; we keep the single
    # current-form constructor (interpolation defaults to ``False``,
    # matching the non-deprecated overload).
    """

    def __init__(
        self,
        frequency: Frequency,
        revised: bool,
        lag: Period,
        ccy: Currency,
        ts: object | None = None,
        interpolated: bool = False,
    ) -> None:
        super().__init__(
            family_name="YY_CPI",
            region=Region.Generic,
            revised=revised,
            interpolated=interpolated,
            frequency=frequency,
            availability_lag=lag,
            currency=ccy,
            ts=ts,
        )
