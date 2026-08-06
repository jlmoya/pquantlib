"""OvernightIndexFutureRateHelper / SofrFutureRateHelper.

# C++ parity: ql/termstructures/yield/overnightindexfutureratehelper.{hpp,cpp} (v1.43)

Bootstrap helper quoted as a futures PRICE (100 - rate), wrapping an
:class:`~pquantlib.instruments.overnight_index_future.OvernightIndexFuture`.

``SofrFutureRateHelper`` derives the reference period from an exchange month /
year / frequency:

* quarterly contracts run from the third Wednesday of the reference month to
  the third Wednesday of the month one quarter later, and compound;
* monthly contracts run from the first of the month to the first of the next
  month, and average simply.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper, PillarChoice
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.weekday import Weekday


def _sofr_start(month: Month, year: int, freq: Frequency) -> Date:
    """C++ parity: anonymous-namespace ``getSofrStart``."""
    if freq == Frequency.Monthly:
        return Date.from_ymd(1, month, year)
    return Date.nth_weekday(3, Weekday.Wednesday, month, year)


def _sofr_end(month: Month, year: int, freq: Frequency) -> Date:
    """C++ parity: anonymous-namespace ``getSofrEnd``."""
    if freq == Frequency.Monthly:
        return Date.end_of_month(Date.from_ymd(1, month, year)) + 1
    d = _sofr_start(month, year, freq) + Period.from_frequency(freq)
    return Date.nth_weekday(3, Weekday.Wednesday, d.month(), d.year())


class OvernightIndexFutureRateHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """Bootstrap helper for a future on a compounded overnight index."""

    def __init__(
        self,
        price: Quote | float,
        value_date: Date,
        maturity_date: Date,
        overnight_index: OvernightIndex,
        convexity_adjustment: Quote | None = None,
        averaging_method: RateAveraging = RateAveraging.Compound,
        pillar: PillarChoice = PillarChoice.LastRelevantDate,
        custom_pillar_date: Date | None = None,
    ) -> None:
        super().__init__(price)
        # Local import: termstructures/ should not depend on instruments/ at
        # module-load time.
        from pquantlib.instruments.overnight_index_future import (  # noqa: PLC0415
            OvernightIndexFuture,
        )

        self._index: OvernightIndex = overnight_index
        self._averaging_method: RateAveraging = averaging_method
        self._convexity_adjustment: Quote | None = convexity_adjustment
        self._value_date: Date = value_date
        self._future_maturity_date: Date = maturity_date
        # C++ clones the index onto the (relinkable) curve handle; without
        # handles the future is rebuilt in set_term_structure instead.
        self._future: OvernightIndexFuture = OvernightIndexFuture(
            overnight_index, value_date, maturity_date, convexity_adjustment,
            averaging_method,
        )
        self._earliest_date = value_date
        self._latest_date = maturity_date

        # C++ parity: overnightindexfutureratehelper.cpp:63-83. C++ writes
        # MaturityDate and LastRelevantDate as separate cases assigning
        # maturityDate_ and latestDate_ respectively; here they are the same
        # date by construction, so the two are folded.
        if pillar in (PillarChoice.MaturityDate, PillarChoice.LastRelevantDate):
            self._pillar_date = maturity_date
        elif pillar == PillarChoice.CustomDate:
            qassert.require(
                custom_pillar_date is not None, "custom pillar date must be provided"
            )
            assert custom_pillar_date is not None
            qassert.require(
                custom_pillar_date >= value_date,
                "custom pillar date before start of reference period",
            )
            qassert.require(
                custom_pillar_date <= maturity_date,
                "custom pillar date after end of reference period",
            )
            self._pillar_date = custom_pillar_date
        else:
            qassert.fail("unknown Pillar::Choice")

    # --- BootstrapHelper interface --------------------------------------

    def set_term_structure(self, ts: YieldTermStructureProtocol) -> None:
        """Rebuild the future on an index cloned onto the curve.

        # C++ parity: ``setTermStructure`` relinks a handle the cloned index
        # already holds; without handles the index and future are rebuilt.
        """
        super().set_term_structure(ts)
        from pquantlib.instruments.overnight_index_future import (  # noqa: PLC0415
            OvernightIndexFuture,
        )

        self._future = OvernightIndexFuture(
            self._index.clone(ts),
            self._value_date,
            self._future_maturity_date,
            self._convexity_adjustment,
            self._averaging_method,
        )

    def implied_quote(self) -> float:
        # C++ parity: ``impliedQuote`` forces a recalculation, then reads NPV.
        self._future.update()
        return self._future.npv()

    # --- inspectors ------------------------------------------------------

    def convexity_adjustment(self) -> float:
        return self._future.convexity_adjustment()

    def future(self) -> object:
        return self._future


class SofrFutureRateHelper(OvernightIndexFutureRateHelper):
    """SOFR futures helper keyed on the exchange's reference month / year."""

    def __init__(
        self,
        price: Quote | float,
        reference_month: Month,
        reference_year: int,
        reference_freq: Frequency,
        convexity_adjustment: Quote | float | None = None,
        pillar: PillarChoice = PillarChoice.LastRelevantDate,
        custom_pillar_date: Date | None = None,
    ) -> None:
        qassert.require(
            reference_freq in (Frequency.Quarterly, Frequency.Monthly),
            "only monthly and quarterly SOFR futures accepted",
        )
        # C++ takes std::variant<Rate, Handle<Quote>>; the float overload is
        # wrapped into a quote by the base class, so only the Quote-or-None
        # case needs handling here.
        adjustment: Quote | None
        if convexity_adjustment is None or isinstance(convexity_adjustment, Quote):
            adjustment = convexity_adjustment
        else:
            from pquantlib.quotes.simple_quote import SimpleQuote  # noqa: PLC0415

            adjustment = SimpleQuote(float(convexity_adjustment))
        super().__init__(
            price,
            _sofr_start(reference_month, reference_year, reference_freq),
            _sofr_end(reference_month, reference_year, reference_freq),
            Sofr(),
            adjustment,
            RateAveraging.Compound
            if reference_freq == Frequency.Quarterly
            else RateAveraging.Simple,
            pillar,
            custom_pillar_date,
        )
