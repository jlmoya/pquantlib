"""PerpetualFutures — futures with no termination date (crypto perpetual swaps).

# C++ parity: ql/instruments/perpetualfutures.{hpp,cpp} (v1.43).

``PerpetualFuturesPayoffType``:

- ``Linear``  — the underlying is a FOR/DOM pair; margin and settlement in DOM.
- ``Inverse`` — the underlying is a FOR/DOM pair; margin and settlement in FOR.
- ``Quanto``  — margin and settlement in a third (quanto) currency.

``PerpetualFuturesFundingType``:

- ``FundingWithPreviousSpot`` —
  ``cashflow(t+1) = f_{t+1} - f_t - fr_t (f_t - x_t) - i^diff_t x_t``
- ``FundingWithCurrentSpot`` —
  ``cashflow(t+1) = f_{t+1} - f_t - fr_t x_{t+1} (f_t - x_t)/x_t - i^diff_t x_{t+1}``

where ``x_t``, ``f_t``, ``fr_t`` and ``i^diff_t`` are the spot, the futures
price, the funding rate and the interest-rate differential at ``t``.

``funding_frequency`` of zero length means continuous funding; anything else is
discrete.

For details see *Perpetual Futures Pricing*, Damien Ackerer, Julien Hugonnier,
Urban Jermann, 2024 — https://finance.wharton.upenn.edu/~jermann/AHJ-main-10.pdf

Python port notes:

- The C++ nested ``PerpetualFutures::PayoffType`` / ``FundingType`` enums
  become the module-level ``PerpetualFuturesPayoffType`` /
  ``PerpetualFuturesFundingType`` IntEnums, following the flat-naming
  convention already used for ``SwapType`` and ``RateAveraging``.
- C++ default-constructs ``arguments`` with the out-of-range sentinels
  ``PayoffType(-1)`` / ``FundingType(-1)``; Python enums are closed, so the
  unset state is ``None`` and ``validate()`` rejects it with the same message.
- The C++ ``operator<<`` overloads become ``__str__`` on the enums.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class PerpetualFuturesPayoffType(IntEnum):
    """# C++ parity: ``PerpetualFutures::PayoffType`` (perpetualfutures.hpp:66)."""

    Linear = 0
    Inverse = 1
    Quanto = 2

    def __str__(self) -> str:
        """# C++ parity: ``operator<<(ostream&, PayoffType)`` (perpetualfutures.cpp:69-80)."""
        return self.name


class PerpetualFuturesFundingType(IntEnum):
    """# C++ parity: ``PerpetualFutures::FundingType`` (perpetualfutures.hpp:67)."""

    FundingWithPreviousSpot = 0
    FundingWithCurrentSpot = 1

    def __str__(self) -> str:
        """# C++ parity: ``operator<<(ostream&, FundingType)`` (perpetualfutures.cpp:82-91)."""
        return self.name


class PerpetualFuturesArguments(PricingEngineArguments):
    """# C++ parity: ``PerpetualFutures::arguments`` (perpetualfutures.hpp:91-101)."""

    def __init__(self) -> None:
        # C++ seeds the enums with the invalid sentinels PayoffType(-1) /
        # FundingType(-1) so that a never-set-up argument set fails validate();
        # ``None`` is this port's spelling of that sentinel.
        self.payoff_type: PerpetualFuturesPayoffType | None = None
        self.funding_type: PerpetualFuturesFundingType | None = None
        self.funding_frequency: Period = Period(8, TimeUnit.Hours)
        self.cal: Calendar = NullCalendar()
        self.dc: DayCounter = ActualActual(Convention.ISDA)

    def validate(self) -> None:
        """# C++ parity: ``PerpetualFutures::arguments::validate``
        (perpetualfutures.cpp:51-67)."""
        qassert.require(
            self.payoff_type
            in (
                PerpetualFuturesPayoffType.Linear,
                PerpetualFuturesPayoffType.Inverse,
                PerpetualFuturesPayoffType.Quanto,
            ),
            "unknown payoff type",
        )
        qassert.require(
            self.funding_type
            in (
                PerpetualFuturesFundingType.FundingWithPreviousSpot,
                PerpetualFuturesFundingType.FundingWithCurrentSpot,
            ),
            "unknown funding type",
        )


class PerpetualFuturesEngine(GenericEngine[PerpetualFuturesArguments, InstrumentResults]):
    """Perpetual-futures engine base class.

    # C++ parity: ``PerpetualFutures::engine`` (perpetualfutures.hpp:103-106),
    # a ``GenericEngine<PerpetualFutures::arguments, PerpetualFutures::results>``
    # where ``results`` resolves to ``Instrument::results``.
    """

    def __init__(self) -> None:
        super().__init__(PerpetualFuturesArguments(), InstrumentResults())


class PerpetualFutures(Instrument):
    """Futures contract with no termination date.

    # C++ parity: ``PerpetualFutures`` (perpetualfutures.hpp:62-87,
    # .cpp:26-49).
    """

    def __init__(
        self,
        payoff_type: PerpetualFuturesPayoffType,
        funding_type: PerpetualFuturesFundingType = (PerpetualFuturesFundingType.FundingWithCurrentSpot),
        funding_frequency: Period | None = None,
        cal: Calendar | None = None,
        dc: DayCounter | None = None,
    ) -> None:
        # # C++ parity: ctor (perpetualfutures.cpp:26-33). The C++ defaults are
        # # Period(8, Hours), NullCalendar() and ActualActual(ISDA); Python
        # # spells them as ``None`` because they are mutable-ish objects.
        super().__init__()
        self._payoff_type: PerpetualFuturesPayoffType = payoff_type
        self._funding_type: PerpetualFuturesFundingType = funding_type
        self._funding_frequency: Period = (
            funding_frequency if funding_frequency is not None else Period(8, TimeUnit.Hours)
        )
        self._cal: Calendar = cal if cal is not None else NullCalendar()
        self._dc: DayCounter = dc if dc is not None else ActualActual(Convention.ISDA)

    # --- inspectors ----------------------------------------------------

    def payoff_type(self) -> PerpetualFuturesPayoffType:
        return self._payoff_type

    def funding_type(self) -> PerpetualFuturesFundingType:
        return self._funding_type

    def funding_frequency(self) -> Period:
        return self._funding_frequency

    def calendar(self) -> Calendar:
        return self._cal

    def day_counter(self) -> DayCounter:
        return self._dc

    # --- Instrument interface ------------------------------------------

    def is_expired(self) -> bool:
        """Never expires. # C++ parity: perpetualfutures.hpp:78."""
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """# C++ parity: ``PerpetualFutures::setupArguments``
        (perpetualfutures.cpp:35-43)."""
        qassert.require(isinstance(args, PerpetualFuturesArguments), "wrong argument type")
        assert isinstance(args, PerpetualFuturesArguments)
        args.payoff_type = self._payoff_type
        args.funding_type = self._funding_type
        args.funding_frequency = self._funding_frequency
        args.cal = self._cal
        args.dc = self._dc


__all__ = [
    "PerpetualFutures",
    "PerpetualFuturesArguments",
    "PerpetualFuturesEngine",
    "PerpetualFuturesFundingType",
    "PerpetualFuturesPayoffType",
]
