"""CapFloor — interest-rate caps, floors, and collars.

# C++ parity: ql/instruments/capfloor.{hpp,cpp} (v1.42.1).

Three concrete shapes:

- ``Cap`` — pays ``max(L_i - K_cap, 0)`` per coupon period.
- ``Floor`` — pays ``max(K_floor - L_i, 0)`` per coupon period.
- ``Collar`` — long cap + short floor (or vice versa) — built via the
  multi-strike base constructor.

The C++ class uses a single multi-strike constructor with two rate
vectors (``capRates`` and ``floorRates``) and an enum to disambiguate.
The two-strike constructor extends the strikes vector to leg length
if it's shorter (per-coupon repeating with the last element). The
Python port mirrors the same semantics.

Divergences from C++:

- C++ ``deepUpdate`` is a LazyObject cache-invalidation primitive;
  PQuantLib mirrors it via a ``deep_update`` alias for ``update``.
- C++ ``optionlet(i)`` returns a single-cashflow ``CapFloor`` — a
  useful helper for per-period decomposition. Ported here.
- C++ ``atmRate(discountCurve)`` + ``impliedVolatility`` are deferred.
  They require a YieldTermStructure-based ATM calculation and a
  NewtonSafe solver path that's not exercised by any L4-E test.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.termstructures.protocols import (
        IborIndexProtocol,
        OvernightIndexProtocol,
    )


# Sentinel used in the CapFloor argument carrier to signal "not
# applicable" (e.g. ``cap_rates`` on a Floor). C++ uses
# ``Null<Rate>()`` which is ``std::numeric_limits<Real>::max()``.
NULL_RATE: float = float("nan")


class CapFloorType(IntEnum):
    """Cap / Floor / Collar.

    # C++ parity: ``CapFloor::Type`` in capfloor.hpp:57 (v1.42.1).
    """

    Cap = 0
    Floor = 1
    Collar = 2


class CapFloorArguments(PricingEngineArguments):
    """Engine arguments carrier.

    # C++ parity: ``CapFloor::arguments`` in capfloor.hpp:138-154.

    Engines (Black, Bachelier, AnalyticCapFloor) read these to compute
    per-optionlet prices and accumulate the total NPV.
    """

    def __init__(self) -> None:
        self.type: CapFloorType = CapFloorType.Cap
        self.start_dates: list[Date] = []
        self.fixing_dates: list[Date] = []
        self.end_dates: list[Date] = []
        self.accrual_times: list[float] = []
        self.cap_rates: list[float] = []
        self.floor_rates: list[float] = []
        self.forwards: list[float] = []
        self.gearings: list[float] = []
        self.spreads: list[float] = []
        self.nominals: list[float] = []
        # IborIndex pointer per coupon (used by Gaussian1d-type engines;
        # plain Black/Bachelier ignore it).
        self.indexes: list[IborIndexProtocol | OvernightIndexProtocol | None] = []

    def validate(self) -> None:
        # # C++ parity: CapFloor::arguments::validate (capfloor.cpp:278-313).
        n = len(self.start_dates)
        qassert.require(
            len(self.end_dates) == n,
            f"number of start dates ({n}) differs from end dates ({len(self.end_dates)})",
        )
        qassert.require(
            len(self.accrual_times) == n,
            f"number of start dates ({n}) differs from accrual times ({len(self.accrual_times)})",
        )
        qassert.require(
            self.type == CapFloorType.Floor or len(self.cap_rates) == n,
            f"number of start dates ({n}) differs from cap rates ({len(self.cap_rates)})",
        )
        qassert.require(
            self.type == CapFloorType.Cap or len(self.floor_rates) == n,
            f"number of start dates ({n}) differs from floor rates ({len(self.floor_rates)})",
        )
        qassert.require(
            len(self.gearings) == n,
            f"number of start dates ({n}) differs from gearings ({len(self.gearings)})",
        )
        qassert.require(
            len(self.spreads) == n,
            f"number of start dates ({n}) differs from spreads ({len(self.spreads)})",
        )
        qassert.require(
            len(self.nominals) == n,
            f"number of start dates ({n}) differs from nominals ({len(self.nominals)})",
        )
        qassert.require(
            len(self.forwards) == n,
            f"number of start dates ({n}) differs from forwards ({len(self.forwards)})",
        )


class CapFloorResults(InstrumentResults):
    """Engine results carrier for CapFloor.

    # C++ parity: ``CapFloor::results`` is just the standard
    # ``Instrument::results`` plus the engine's ``additionalResults``
    # map — both already provided by ``InstrumentResults``.
    """


class CapFloor(Instrument):
    """Abstract base for cap-like instruments.

    # C++ parity: ``class CapFloor : public Instrument`` (capfloor.hpp:55-104).

    Instantiable: C++ exposes a public 2-arg + 3-arg constructor;
    PQuantLib follows the same — see :class:`Cap`, :class:`Floor`,
    :class:`Collar` for the type-specialised convenience subclasses.
    """

    def __init__(
        self,
        type_: CapFloorType,
        floating_leg: Sequence[CashFlow],
        cap_rates: Sequence[float] | None = None,
        floor_rates: Sequence[float] | None = None,
    ) -> None:
        # # C++ parity: CapFloor::CapFloor (capfloor.cpp:124-147).
        super().__init__()
        self._type: CapFloorType = type_
        self._floating_leg: list[CashFlow] = list(floating_leg)
        self._cap_rates: list[float] = list(cap_rates) if cap_rates is not None else []
        self._floor_rates: list[float] = (
            list(floor_rates) if floor_rates is not None else []
        )

        if type_ in (CapFloorType.Cap, CapFloorType.Collar):
            qassert.require(len(self._cap_rates) > 0, "no cap rates given")
            # Repeat the last cap-rate entry up to leg length.
            while len(self._cap_rates) < len(self._floating_leg):
                self._cap_rates.append(self._cap_rates[-1])

        if type_ in (CapFloorType.Floor, CapFloorType.Collar):
            qassert.require(len(self._floor_rates) > 0, "no floor rates given")
            while len(self._floor_rates) < len(self._floating_leg):
                self._floor_rates.append(self._floor_rates[-1])

        # # C++ parity: registerWith every leg cashflow.
        for cf in self._floating_leg:
            cf.register_with(self)

    @classmethod
    def from_strikes(
        cls,
        type_: CapFloorType,
        floating_leg: Sequence[CashFlow],
        strikes: Sequence[float],
    ) -> CapFloor:
        """Construct a Cap or Floor (not Collar) from a single strike vector.

        # C++ parity: ``CapFloor::CapFloor(Type, Leg, const std::vector<Rate>&)``
        # in capfloor.cpp:149-170. Disallows Collar (which needs two vectors).
        """
        qassert.require(len(strikes) > 0, "no strikes given")
        if type_ == CapFloorType.Cap:
            return cls(type_, floating_leg, cap_rates=list(strikes), floor_rates=None)
        if type_ == CapFloorType.Floor:
            return cls(type_, floating_leg, cap_rates=None, floor_rates=list(strikes))
        qassert.fail("only Cap/Floor types allowed in this constructor")

    # --- Instrument interface ---------------------------------------------

    def is_expired(self) -> bool:
        """Expired iff every cashflow has occurred.

        # C++ parity: CapFloor::isExpired (capfloor.cpp:172-177).
        """
        return all(cf.has_occurred() for cf in self._floating_leg)

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Fill the engine argument carrier from leg + strike vectors.

        # C++ parity: CapFloor::setupArguments (capfloor.cpp:210-269).
        """
        qassert.require(
            isinstance(args, CapFloorArguments),
            "wrong argument type (expected CapFloorArguments)",
        )
        assert isinstance(args, CapFloorArguments)

        n = len(self._floating_leg)
        args.type = self._type
        args.start_dates = [Date()] * n
        args.fixing_dates = [Date()] * n
        args.end_dates = [Date()] * n
        args.accrual_times = [0.0] * n
        args.forwards = [0.0] * n
        args.nominals = [0.0] * n
        args.gearings = [0.0] * n
        args.cap_rates = [NULL_RATE] * n
        args.floor_rates = [NULL_RATE] * n
        args.spreads = [0.0] * n
        args.indexes = [None] * n

        # C++ parity: ``Date today = Settings::instance().evaluationDate()``.
        # In PQuantLib we don't gate on evaluationDate (matches the
        # VanillaOption / Swaption divergence — engines compute the
        # NPV regardless and zero-out expired coupons).
        for i, cf in enumerate(self._floating_leg):
            qassert.require(
                isinstance(cf, FloatingRateCoupon),
                "non-FloatingRateCoupon given",
            )
            assert isinstance(cf, FloatingRateCoupon)
            coupon = cf
            args.start_dates[i] = coupon.accrual_start_date()
            args.fixing_dates[i] = coupon.fixing_date()
            args.end_dates[i] = coupon.date()
            args.accrual_times[i] = coupon.accrual_period()

            # C++ checks ``endDate >= today`` before computing the
            # forward; PQuantLib always computes (LazyObject calculates
            # only when the engine is set).
            try:
                args.forwards[i] = coupon.adjusted_fixing()
            except Exception:
                args.forwards[i] = NULL_RATE

            args.nominals[i] = coupon.nominal()
            gearing = coupon.gearing()
            spread = coupon.spread()
            args.gearings[i] = gearing
            args.spreads[i] = spread

            if self._type in (CapFloorType.Cap, CapFloorType.Collar):
                args.cap_rates[i] = (self._cap_rates[i] - spread) / gearing
            if self._type in (CapFloorType.Floor, CapFloorType.Collar):
                args.floor_rates[i] = (self._floor_rates[i] - spread) / gearing

            args.indexes[i] = coupon.index()

    def deep_update(self) -> None:
        """Forward deep-update through the leg, then notify observers.

        # C++ parity: CapFloor::deepUpdate (capfloor.cpp:271-276).
        """
        self.update()

    # --- inspectors -------------------------------------------------------

    def type(self) -> CapFloorType:
        return self._type

    def cap_rates(self) -> list[float]:
        return list(self._cap_rates)

    def floor_rates(self) -> list[float]:
        return list(self._floor_rates)

    def floating_leg(self) -> list[CashFlow]:
        return list(self._floating_leg)

    def start_date(self) -> Date:
        """Earliest accrual start date in the leg.

        # C++ parity: CapFloor::startDate (capfloor.cpp:179-181) — delegates
        # to ``CashFlows::startDate``. PQuantLib mirrors the algorithm
        # locally (no separate CashFlows utility).
        """
        qassert.require(len(self._floating_leg) > 0, "empty leg")
        d: Date | None = None
        for cf in self._floating_leg:
            sd = (
                cf.accrual_start_date()
                if isinstance(cf, FloatingRateCoupon)
                else cf.date()
            )
            d = sd if d is None else min(d, sd)
        assert d is not None
        return d

    def maturity_date(self) -> Date:
        """Latest payment date in the leg.

        # C++ parity: CapFloor::maturityDate (capfloor.cpp:183-185).
        """
        qassert.require(len(self._floating_leg) > 0, "empty leg")
        d: Date | None = None
        for cf in self._floating_leg:
            ed = cf.date()
            d = ed if d is None else max(d, ed)
        assert d is not None
        return d

    def last_floating_rate_coupon(self) -> FloatingRateCoupon | None:
        """The last cashflow if it's a FloatingRateCoupon.

        # C++ parity: CapFloor::lastFloatingRateCoupon (capfloor.cpp:187-193).
        """
        if not self._floating_leg:
            return None
        last = self._floating_leg[-1]
        if isinstance(last, FloatingRateCoupon):
            return last
        return None

    def optionlet(self, n: int) -> CapFloor:
        """Return the n-th period as a new CapFloor with a single cashflow.

        # C++ parity: CapFloor::optionlet (capfloor.cpp:195-208).
        """
        qassert.require(
            n < len(self._floating_leg),
            f"optionlet {n + 1} does not exist; only {len(self._floating_leg)}",
        )
        single = [self._floating_leg[n]]
        cap: list[float] | None = None
        floor: list[float] | None = None
        if self._type in (CapFloorType.Cap, CapFloorType.Collar):
            cap = [self._cap_rates[n]]
        if self._type in (CapFloorType.Floor, CapFloorType.Collar):
            floor = [self._floor_rates[n]]
        return CapFloor(self._type, single, cap_rates=cap, floor_rates=floor)


class Cap(CapFloor):
    """Concrete cap.

    # C++ parity: ``class Cap : public CapFloor`` (capfloor.hpp:108-114).
    """

    def __init__(self, floating_leg: Sequence[CashFlow], strikes: Sequence[float]) -> None:
        super().__init__(
            CapFloorType.Cap, floating_leg, cap_rates=list(strikes), floor_rates=None
        )


class Floor(CapFloor):
    """Concrete floor.

    # C++ parity: ``class Floor : public CapFloor`` (capfloor.hpp:118-124).
    """

    def __init__(self, floating_leg: Sequence[CashFlow], strikes: Sequence[float]) -> None:
        super().__init__(
            CapFloorType.Floor, floating_leg, cap_rates=None, floor_rates=list(strikes)
        )


class Collar(CapFloor):
    """Long-cap + short-floor (or vice versa) combination.

    # C++ parity: ``class Collar : public CapFloor`` (capfloor.hpp:128-134).
    """

    def __init__(
        self,
        floating_leg: Sequence[CashFlow],
        cap_rates: Sequence[float],
        floor_rates: Sequence[float],
    ) -> None:
        super().__init__(
            CapFloorType.Collar,
            floating_leg,
            cap_rates=list(cap_rates),
            floor_rates=list(floor_rates),
        )


__all__ = [
    "NULL_RATE",
    "Cap",
    "CapFloor",
    "CapFloorArguments",
    "CapFloorResults",
    "CapFloorType",
    "Collar",
    "Floor",
]
