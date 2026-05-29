"""InflationIndex / ZeroInflationIndex / YoYInflationIndex abstracts.

# C++ parity: ql/indexes/inflationindex.{hpp,cpp} (v1.42.1).

C++ structure:

* ``InflationIndex`` extends ``Index`` and holds
  ``familyName_, region_, revised_, frequency_, availabilityLag_, currency_``.
  Its ``name()`` accessor formats as ``"<region.name()> <familyName_>"`` (e.g.
  ``"EU HICP"``); ``fixingCalendar()`` returns ``NullCalendar`` (inflation
  indexes are observed at month/quarter granularity, not daily).
* ``ZeroInflationIndex`` extends ``InflationIndex`` and adds a forecasting
  path through a ``ZeroInflationTermStructure`` handle. We keep the
  abstract slot for the term-structure handle as ``object | None`` to avoid
  circular imports between ``pquantlib.indexes.inflation`` and
  ``pquantlib.termstructures.inflation``; the concrete handle is set by
  L7-B once curves land.
* ``YoYInflationIndex`` extends ``InflationIndex`` with two construction
  modes: a *ratio* mode (built on an underlying ``ZeroInflationIndex``) and
  a *quoted* mode (own family/region/etc.). The ratio mode computes
  ``fixing(d) = underlying.fixing(d) / underlying.fixing(d - 12m) - 1.0``;
  the quoted mode stores YoY fixings directly.

Python divergences from C++:

- The C++ class hierarchy keeps ``interpolated_`` on
  ``YoYInflationIndex`` only. The L7-A spec collapses ``interpolated`` up
  onto the abstract for ergonomic uniformity. For ``ZeroInflationIndex``,
  ``interpolated()`` always returns ``False`` (zero inflation fixings are
  always non-interpolated in C++, see ``ZeroInflationIndex::needsForecast``).
- The Python ``maturity_date(fixing_date)`` convenience returns the end
  of the C++ ``inflationPeriod(fixing_date, frequency).second`` interval.
  This is not on the C++ class but is the natural ``Index.maturityDate``
  semantics for inflation indexes (the period during which the fixing is
  available).
- Forecasting paths (``forecastFixing`` / ``zeroInflation_->zeroRate``) are
  *not* implemented at the abstract level; they hook in once L7-B lands
  the curve types. Concrete ``fixing()`` on the abstract raises if no
  term-structure is set and no past fixing is registered.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, cast, runtime_checkable

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.indexes.index import Index
from pquantlib.indexes.inflation.region import Region
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class _DayCounterLike(Protocol):
    """Structural subset of DayCounter used by inflation_year_fraction.

    Local Protocol to avoid the circular import that a direct import of
    ``pquantlib.daycounters.day_counter.DayCounter`` would create
    (daycounters → currencies → indexes/inflation).
    """

    def year_fraction(self, d1: Date, d2: Date) -> float: ...


@runtime_checkable
class _ZeroInflationTSLike(Protocol):
    """Structural subset of ``ZeroInflationTermStructure`` used by
    ``ZeroInflationIndex._forecast_fixing``.

    Avoids the circular import on
    ``pquantlib.termstructures.inflation.zero_inflation_term_structure``.
    """

    def base_date(self) -> Date: ...
    def zero_rate(self, d: Date, extrapolate: bool = False) -> float: ...
    def day_counter(self) -> _DayCounterLike: ...


def inflation_year_fraction(
    f: Frequency,
    index_is_interpolated: bool,
    day_counter: _DayCounterLike,
    d1: Date,
    d2: Date,
) -> float:
    """Compute the year-fraction between two inflation dates.

    # C++ parity: ``inflationYearFraction`` in
    # ql/termstructures/inflationtermstructure.cpp (v1.42.1).

    For *interpolated* indexes, the year-fraction is the raw
    ``dayCounter.year_fraction(d1, d2)``. For *non-interpolated*
    indexes the fixings are constant within an inflation period, so the
    inflation-time between two dates is computed between the *period
    starts* containing each date.
    """
    if index_is_interpolated:
        return day_counter.year_fraction(d1, d2)
    lim_d1 = inflation_period(d1, f)
    lim_d2 = inflation_period(d2, f)
    return day_counter.year_fraction(lim_d1[0], lim_d2[0])


def inflation_period(d: Date, f: Frequency) -> tuple[Date, Date]:
    """Return ``(start, end)`` of the inflation period containing ``d``.

    # C++ parity: ql/termstructures/inflationtermstructure.cpp —
    # ``inflationPeriod(const Date&, Frequency)``. We host the helper here
    # (rather than under termstructures/inflation/) because the inflation
    # index abstract already needs it for ``maturity_date()`` and we want
    # to avoid a circular import once curves land.
    """
    month = d.month()
    year = d.year()
    f_int = int(f)
    if f in (
        Frequency.Annual,
        Frequency.Semiannual,
        Frequency.EveryFourthMonth,
        Frequency.Quarterly,
        Frequency.Bimonthly,
    ):
        n_months = 12 // f_int
        start_month_int = month - (month - 1) % n_months
        end_month_int = start_month_int + n_months - 1
    elif f == Frequency.Monthly:
        start_month_int = end_month_int = month
    else:
        qassert.fail(f"frequency not handled by inflation_period: {f}")
    start = Date.from_ymd(1, Month(start_month_int), year)
    end = Date.end_of_month(Date.from_ymd(1, Month(end_month_int), year))
    return start, end


class InflationIndex(Index):
    """Abstract base for inflation-rate indexes.

    # C++ parity: ``InflationIndex`` in ql/indexes/inflationindex.hpp. The
    # constructor signature follows the C++ one plus a Python-side
    # ``interpolated`` flag (see module docstring).
    """

    def __init__(
        self,
        family_name: str,
        region: Region,
        revised: bool,
        interpolated: bool,
        frequency: Frequency,
        availability_lag: Period,
        currency: Currency,
    ) -> None:
        super().__init__()
        self._family_name: str = family_name
        self._region: Region = region
        self._revised: bool = revised
        self._interpolated: bool = interpolated
        self._frequency: Frequency = frequency
        self._availability_lag: Period = availability_lag
        self._currency: Currency = currency
        # C++ parity: InflationIndex constructor builds
        # ``name_ = region_.name() + " " + familyName_``.
        self._name: str = f"{region.region_name()} {family_name}"

    # ---- Index interface ----------------------------------------------

    def name(self) -> str:
        """Formatted name: ``"<region.name()> <family_name>"``."""
        return self._name

    def fixing_calendar(self) -> Calendar:
        """Inflation indexes use a NullCalendar.

        # C++ parity: ``InflationIndex::fixingCalendar`` returns
        # ``NullCalendar``. Inflation fixings are monthly/quarterly and
        # not date-bound at daily granularity.
        """
        return NullCalendar()

    def is_valid_fixing_date(self, fixing_date: Date) -> bool:
        """C++ parity: ``isValidFixingDate`` is ``true`` for inflation indexes."""
        del fixing_date
        return True

    @abstractmethod
    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        """Inflation fixing on ``fixing_date``. Concrete classes implement."""

    # ---- inspectors ---------------------------------------------------

    def family_name(self) -> str:
        return self._family_name

    def region(self) -> Region:
        return self._region

    def revised(self) -> bool:
        return self._revised

    def interpolated(self) -> bool:
        """Return whether the index uses linear interpolation between fixings.

        # C++ parity: only ``YoYInflationIndex`` carries this in C++; we
        # hoist it onto the abstract per L7-A spec. ``ZeroInflationIndex``
        # always returns ``False`` (zero fixings are non-interpolated).
        """
        return self._interpolated

    def frequency(self) -> Frequency:
        return self._frequency

    def availability_lag(self) -> Period:
        return self._availability_lag

    def currency(self) -> Currency:
        return self._currency


class ZeroInflationIndex(InflationIndex):
    """Base class for zero-inflation (CPI-style) indexes.

    # C++ parity: ``ZeroInflationIndex`` in ql/indexes/inflationindex.hpp.
    # The Handle<ZeroInflationTermStructure> handle is stored as a generic
    # ``object | None`` to avoid an import cycle; L7-B will install the
    # concrete handle type via the InflationTermStructure Protocol.
    """

    def __init__(
        self,
        family_name: str,
        region: Region,
        revised: bool,
        frequency: Frequency,
        availability_lag: Period,
        currency: Currency,
        ts: object | None = None,
    ) -> None:
        # C++ parity: ZeroInflationIndex always has interpolated=False
        # (see ZeroInflationIndex::needsForecast comment).
        super().__init__(
            family_name=family_name,
            region=region,
            revised=revised,
            interpolated=False,
            frequency=frequency,
            availability_lag=availability_lag,
            currency=currency,
        )
        self._zero_inflation_ts: object | None = ts

    def zero_inflation_term_structure(self) -> object | None:
        """Return the zero-inflation term-structure handle, if any.

        # C++ parity: returns ``Handle<ZeroInflationTermStructure>``.
        # Typed as ``object`` here to break the import cycle until L7-B.
        """
        return self._zero_inflation_ts

    def set_zero_inflation_term_structure(self, ts: object | None) -> None:
        """Install (or clear) the zero-inflation forecasting curve.

        Used by ``ZeroCouponInflationSwapHelper`` to wire the curve being
        bootstrapped into the index before computing the implied quote.

        # C++ parity: the C++ helper clones the index and binds a
        # ``Handle<ZeroInflationTermStructure>`` to the bootstrapping
        # curve. Python uses in-place mutation of the index's slot since
        # the helpers ensure single-curve binding during a bootstrap run.
        """
        self._zero_inflation_ts = ts
        self.update()

    def maturity_date(self, fixing_date: Date) -> Date:
        """Return the end of the inflation period containing ``fixing_date``.

        # C++ parity: not on C++ ``ZeroInflationIndex`` directly. Mirrors
        # the natural ``Index::maturityDate`` semantics for an inflation
        # index — the end of the period during which the fixing applies.
        # Computed as ``inflationPeriod(fixing_date, frequency()).second``.
        """
        _, end = inflation_period(fixing_date, self.frequency())
        return end

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        """Look up a past fixing or forecast via the zero-inflation curve.

        # C++ parity: ``ZeroInflationIndex::fixing`` (inflationindex.cpp:200ff).
        # If the period start has a stored fixing, return it; otherwise
        # forecast via the zero-inflation term-structure handle.
        # Forecast formula:
        #   I(T) = I(baseDate) * (1 + Z1)^t1
        # where Z1 = ts.zero_rate(periodStart), t1 = inflation_year_fraction(
        # frequency, indexInterpolated=False, ts.day_counter, baseDate,
        # periodStart).
        """
        del forecast_todays_fixing
        start, _ = inflation_period(fixing_date, self.frequency())
        history = self.time_series()
        value = history[start]
        if value is not None:
            return value
        # No past fixing → forecast.
        qassert.require(
            self._zero_inflation_ts is not None,
            f"Missing {self.name()} fixing for {start} and no zero-inflation "
            "term structure to forecast from.",
        )
        return self._forecast_fixing(fixing_date)

    def _forecast_fixing(self, fixing_date: Date) -> float:
        """Forecast ``I(T)`` via the zero-inflation curve.

        # C++ parity: ``ZeroInflationIndex::forecastFixing`` (inflationindex.cpp:219-237).
        # The ts slot is typed ``object`` to avoid a hard circular import;
        # we cast to the local ``_ZeroInflationTSLike`` Protocol for the
        # call site, which captures the structural methods we need.
        """
        qassert.require(self._zero_inflation_ts is not None, "no zero TS")
        ts = cast(_ZeroInflationTSLike, self._zero_inflation_ts)
        base_date = ts.base_date()
        # Base-date fixing must exist (recurses via fixing() above).
        base_fixing = self.fixing(base_date)
        period_start, _ = inflation_period(fixing_date, self.frequency())
        z1 = ts.zero_rate(period_start, False)
        t1 = inflation_year_fraction(
            self.frequency(),
            False,
            ts.day_counter(),
            base_date,
            period_start,
        )
        if z1 <= -1.0:
            return 0.0
        return base_fixing * ((1.0 + z1) ** t1)


class YoYInflationIndex(InflationIndex):
    """Base class for year-on-year inflation indexes.

    # C++ parity: ``YoYInflationIndex`` in ql/indexes/inflationindex.hpp.
    # Two construction modes:
    #
    # * Quoted mode (this class's constructor): the YoY fixings are stored
    #   directly; ``ratio()`` returns ``False``.
    # * Ratio mode (``from_underlying`` classmethod): YoY is the ratio of
    #   two zero-inflation fixings; ``ratio()`` returns ``True``. The
    #   underlying ZeroInflationIndex provides the past fixings.
    """

    def __init__(
        self,
        family_name: str,
        region: Region,
        revised: bool,
        interpolated: bool,
        frequency: Frequency,
        availability_lag: Period,
        currency: Currency,
        ts: object | None = None,
    ) -> None:
        super().__init__(
            family_name=family_name,
            region=region,
            revised=revised,
            interpolated=interpolated,
            frequency=frequency,
            availability_lag=availability_lag,
            currency=currency,
        )
        # C++ parity: quoted-mode YoYInflationIndex sets ratio_ = false.
        self._ratio: bool = False
        self._underlying: ZeroInflationIndex | None = None
        self._yoy_inflation_ts: object | None = ts

    @classmethod
    def from_underlying(
        cls,
        underlying: ZeroInflationIndex,
        interpolated: bool = False,
        ts: object | None = None,
    ) -> YoYInflationIndex:
        """Construct a YoY index as a ratio of two zero-inflation fixings.

        # C++ parity: ``YoYInflationIndex(const ext::shared_ptr<
        # ZeroInflationIndex>&, ...)`` — the ratio-mode constructor.
        # Family/region/etc. are forwarded from the underlying, with the
        # family name prefixed by ``"YYR_"``.
        """
        instance = cls(
            family_name=f"YYR_{underlying.family_name()}",
            region=underlying.region(),
            revised=underlying.revised(),
            interpolated=interpolated,
            frequency=underlying.frequency(),
            availability_lag=underlying.availability_lag(),
            currency=underlying.currency(),
            ts=ts,
        )
        instance._ratio = True
        instance._underlying = underlying
        return instance

    def ratio(self) -> bool:
        """Return True iff this YoY index is built as a ratio of an underlying."""
        return self._ratio

    def underlying_index(self) -> ZeroInflationIndex | None:
        return self._underlying

    def yoy_inflation_term_structure(self) -> object | None:
        return self._yoy_inflation_ts

    def set_yoy_inflation_term_structure(self, ts: object | None) -> None:
        """Install (or clear) the YoY forecasting curve.

        Used by ``YearOnYearInflationSwapHelper`` during bootstrap. See
        :meth:`ZeroInflationIndex.set_zero_inflation_term_structure` for
        the rationale.
        """
        self._yoy_inflation_ts = ts
        self.update()

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        """Look up the YoY fixing on ``fixing_date``.

        # C++ parity: ``YoYInflationIndex::fixing``. Ratio mode delegates
        # to underlying past fixings; quoted mode reads from local history.
        # Forecasting via ``YoYInflationTermStructure`` lands with L7-B.
        """
        del forecast_todays_fixing
        if self._ratio:
            qassert.require(
                self._underlying is not None,
                f"ratio-mode {self.name()} has no underlying index",
            )
            assert self._underlying is not None
            one_year = Period(1, TimeUnit.Years)
            cur = self._underlying.fixing(fixing_date)
            prev = self._underlying.fixing(fixing_date - one_year)
            return cur / prev - 1.0

        start, _ = inflation_period(fixing_date, self.frequency())
        history = self.time_series()
        value = history[start]
        if value is not None:
            return value
        qassert.require(
            self._yoy_inflation_ts is None,
            f"forecast path for {self.name()} not yet wired (L7-B will land it)",
        )
        qassert.fail(
            f"Missing {self.name()} YoY fixing for {start}; "
            "store one via add_fixing before calling fixing()."
        )


