"""CPICapFloor — single-flow CPI cap or floor option.

# C++ parity: ql/instruments/cpicapfloor.{hpp,cpp} (v1.42.1).

Payoff:

    P_n(0, T) * max(y * (N * ((1 + K)^T - 1) - N * (I(T) / I(0) - 1)), 0)

where ``y = +1`` for a cap (payer pays inflation), ``-1`` for a floor.

The contract has a single cashflow on the payment date. The instrument
is shaped like an Option but does not inherit from ``Option`` (per C++
documentation).

L7-D ports the instrument shell + ``setup_arguments``. The concrete pricing
engine (analytic / interpolated-surface) is left as a Phase 8+ carve-out.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.indexes.inflation.cpi import (
    InterpolationType,
    is_interpolated,
)
from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class CPICapFloorArguments(PricingEngineArguments):
    """Engine arguments for CPICapFloor.

    # C++ parity: ``CPICapFloor::arguments`` (cpicapfloor.hpp:116-131).
    """

    def __init__(self) -> None:
        self.type: OptionType = OptionType.Call
        self.nominal: float = 0.0
        self.start_date: Date = Date()
        self.fix_date: Date = Date()
        self.pay_date: Date = Date()
        self.base_cpi: float = 0.0
        self.maturity: Date = Date()
        self.fix_calendar: Calendar | None = None
        self.pay_calendar: Calendar | None = None
        self.fix_convention: BusinessDayConvention = BusinessDayConvention.Following
        self.pay_convention: BusinessDayConvention = BusinessDayConvention.Following
        self.strike: float = 0.0
        self.index: ZeroInflationIndex | None = None
        self.observation_lag: Period = Period()
        self.observation_interpolation: InterpolationType = InterpolationType.AsIndex

    def validate(self) -> None:
        # # C++ parity: ``CPICapFloor::arguments::validate`` is a no-op.
        # # We don't add Python-side checks (matches C++).
        return


class CPICapFloorResults(InstrumentResults):
    """Results carrier — inherits the base Instrument::results contract."""


class CPICapFloor(Instrument):
    """CPI cap or floor option (single flow).

    # C++ parity: ``CPICapFloor`` (cpicapfloor.hpp:62-113).
    """

    def __init__(
        self,
        *,
        type_: OptionType,
        nominal: float,
        start_date: Date,
        base_cpi: float,
        maturity: Date,
        fix_calendar: Calendar,
        fix_convention: BusinessDayConvention,
        pay_calendar: Calendar,
        pay_convention: BusinessDayConvention,
        strike: float,
        inflation_index: ZeroInflationIndex,
        observation_lag: Period,
        observation_interpolation: InterpolationType = InterpolationType.AsIndex,
    ) -> None:
        # # C++ parity: CPICapFloor::CPICapFloor (cpicapfloor.cpp:37-70).
        # # C++ checks for non-null shared_ptrs at runtime; Python's type
        # # system catches this statically (the constructor signature
        # # requires concrete types). Skipping the redundant runtime checks.
        super().__init__()
        # C++ consistency check between obs lag and index availability.
        avail_lag_months = inflation_index.availability_lag().length
        obs_lag_months = observation_lag.length
        if not is_interpolated(observation_interpolation):
            qassert.require(
                obs_lag_months >= avail_lag_months,
                "CPI cap/floor observation lag must be at least the index "
                f"availability lag (effectively flat): {obs_lag_months}M vs "
                f"{avail_lag_months}M",
            )
        else:
            qassert.require(
                obs_lag_months > avail_lag_months,
                "CPI cap/floor observation lag must be strictly greater than "
                f"the index availability lag (effectively linear): {obs_lag_months}M "
                f"vs {avail_lag_months}M",
            )

        self._type: OptionType = type_
        self._nominal: float = nominal
        self._start_date: Date = start_date
        self._base_cpi: float = base_cpi
        self._maturity: Date = maturity
        self._fix_calendar: Calendar = fix_calendar
        self._fix_convention: BusinessDayConvention = fix_convention
        self._pay_calendar: Calendar = pay_calendar
        self._pay_convention: BusinessDayConvention = pay_convention
        self._strike: float = strike
        self._index: ZeroInflationIndex = inflation_index
        self._observation_lag: Period = observation_lag
        self._observation_interpolation: InterpolationType = observation_interpolation

    # ---- Instrument interface -----------------------------------------

    def is_expired(self) -> bool:
        """Expired iff the evaluation date is past maturity.

        # C++ parity: CPICapFloor::isExpired (cpicapfloor.cpp:84-86) uses
        # the global evaluation date. PQuantLib reads it via ObservableSettings.
        """
        today = ObservableSettings().evaluation_date_or_today()
        return today > self._maturity

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Fill the CPI cap/floor argument carrier.

        # C++ parity: CPICapFloor::setupArguments (cpicapfloor.cpp:94-116).
        """
        qassert.require(
            isinstance(args, CPICapFloorArguments),
            "wrong argument type (expected CPICapFloorArguments)",
        )
        assert isinstance(args, CPICapFloorArguments)
        args.type = self._type
        args.nominal = self._nominal
        args.start_date = self._start_date
        args.base_cpi = self._base_cpi
        args.maturity = self._maturity
        args.fix_calendar = self._fix_calendar
        args.fix_convention = self._fix_convention
        args.pay_calendar = self._pay_calendar
        args.pay_convention = self._pay_convention
        args.fix_date = self.fixing_date()
        args.pay_date = self.pay_date()
        args.strike = self._strike
        args.index = self._index
        args.observation_lag = self._observation_lag
        args.observation_interpolation = self._observation_interpolation

    # ---- inspectors ---------------------------------------------------

    def type(self) -> OptionType:
        return self._type

    def nominal(self) -> float:
        return self._nominal

    def strike(self) -> float:
        return self._strike

    def fixing_date(self) -> Date:
        """Fixing date = ``fix_calendar.adjust(maturity - observation_lag, fix_convention)``.

        # C++ parity: CPICapFloor::fixingDate (cpicapfloor.cpp:74-76).
        """
        return self._fix_calendar.adjust(
            self._maturity - self._observation_lag, self._fix_convention
        )

    def pay_date(self) -> Date:
        """Payment date = ``pay_calendar.adjust(maturity, pay_convention)``.

        # C++ parity: CPICapFloor::payDate (cpicapfloor.cpp:79-81).
        """
        return self._pay_calendar.adjust(self._maturity, self._pay_convention)

    def index(self) -> ZeroInflationIndex:
        return self._index

    def observation_lag(self) -> Period:
        return self._observation_lag


__all__ = [
    "CPICapFloor",
    "CPICapFloorArguments",
    "CPICapFloorResults",
]
