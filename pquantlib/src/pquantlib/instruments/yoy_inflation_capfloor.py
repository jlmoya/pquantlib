"""YoYInflationCapFloor — YoY inflation cap / floor / collar instrument.

# C++ parity: ql/instruments/inflationcapfloor.{hpp,cpp} (v1.42.1).

Three concrete shapes (mirroring C++):

- ``YoYInflationCap`` — pays ``max(YoY(T_i) - K_cap, 0) * nominal * gearing
  * accrual_period`` discounted on a nominal curve.
- ``YoYInflationFloor`` — pays ``max(K_floor - YoY(T_i), 0) * ...``.
- ``YoYInflationCollar`` — long cap + short floor, with two strike vectors.

``YoYInflationCapFloor`` takes a leg of YoY-inflation cashflows + cap/floor
strike vectors. Each strike vector is repeated up to leg length if shorter
(matches C++).

Coupon seam: the YoY leg is typed as ``Sequence[YoYInflationCouponLike]``
where ``YoYInflationCouponLike`` is a Protocol exposing the methods the
engine reads. L7-C will provide ``YoYInflationCoupon`` concretes that
structurally satisfy this Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date

# Sentinel for "not applicable" cap/floor rates (mirrors CapFloor.NULL_RATE
# from L4-E). Engines compare against this via math.isnan.
NULL_RATE: float = float("nan")


@runtime_checkable
class YoYInflationCouponLike(Protocol):
    """Structural API the YoY cap/floor engine needs from a leg cashflow.

    The L7-C ``YoYInflationCoupon`` will satisfy this Protocol at merge
    time; until then, tests use a lightweight scaffold (see
    ``_ScaffoldYoYInflationCoupon`` below).
    """

    def accrual_start_date(self) -> Date: ...
    def accrual_end_date(self) -> Date: ...
    def fixing_date(self) -> Date: ...
    def date(self) -> Date: ...
    def accrual_period(self) -> float: ...
    def nominal(self) -> float: ...
    def gearing(self) -> float: ...
    def spread(self) -> float: ...
    def has_occurred(self, ref_date: Date | None = None) -> bool: ...
    def register_with(self, observer: object) -> None: ...


class YoYInflationCapFloorType(IntEnum):
    """Cap / Floor / Collar.

    # C++ parity: ``YoYInflationCapFloor::Type`` (inflationcapfloor.hpp:57).
    """

    Cap = 0
    Floor = 1
    Collar = 2


class YoYInflationCapFloorArguments(PricingEngineArguments):
    """Engine arguments carrier for YoY cap/floor.

    # C++ parity: ``YoYInflationCapFloor::arguments`` (inflationcapfloor.hpp:133-149).
    """

    def __init__(self) -> None:
        self.type: YoYInflationCapFloorType = YoYInflationCapFloorType.Cap
        self.start_dates: list[Date] = []
        self.fixing_dates: list[Date] = []
        self.pay_dates: list[Date] = []
        self.accrual_times: list[float] = []
        self.cap_rates: list[float] = []
        self.floor_rates: list[float] = []
        self.gearings: list[float] = []
        self.spreads: list[float] = []
        self.nominals: list[float] = []

    def validate(self) -> None:
        # # C++ parity: YoYInflationCapFloor::arguments::validate
        # # (inflationcapfloor.cpp:181-212).
        n = len(self.start_dates)
        qassert.require(
            len(self.pay_dates) == n,
            f"number of start dates ({n}) differs from pay dates ({len(self.pay_dates)})",
        )
        qassert.require(
            len(self.accrual_times) == n,
            f"number of start dates ({n}) differs from accrual times "
            f"({len(self.accrual_times)})",
        )
        qassert.require(
            self.type == YoYInflationCapFloorType.Floor or len(self.cap_rates) == n,
            f"number of start dates ({n}) differs from cap rates ({len(self.cap_rates)})",
        )
        qassert.require(
            self.type == YoYInflationCapFloorType.Cap or len(self.floor_rates) == n,
            f"number of start dates ({n}) differs from floor rates "
            f"({len(self.floor_rates)})",
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


class YoYInflationCapFloorResults(InstrumentResults):
    """Results carrier for YoY cap/floor.

    # C++ parity: ``YoYInflationCapFloor::results`` is just the standard
    # ``Instrument::results`` plus the engine's ``additionalResults``.
    """


class YoYInflationCapFloor(Instrument):
    """Abstract YoY cap/floor instrument.

    # C++ parity: ``class YoYInflationCapFloor : public Instrument``
    # (inflationcapfloor.hpp:55-98).
    """

    def __init__(
        self,
        type_: YoYInflationCapFloorType,
        yoy_leg: Sequence[YoYInflationCouponLike],
        cap_rates: Sequence[float] | None = None,
        floor_rates: Sequence[float] | None = None,
    ) -> None:
        # # C++ parity: YoYInflationCapFloor::YoYInflationCapFloor
        # # (inflationcapfloor.cpp:44-67).
        super().__init__()
        self._type: YoYInflationCapFloorType = type_
        self._yoy_leg: list[YoYInflationCouponLike] = list(yoy_leg)
        self._cap_rates: list[float] = list(cap_rates) if cap_rates is not None else []
        self._floor_rates: list[float] = (
            list(floor_rates) if floor_rates is not None else []
        )

        if type_ in (YoYInflationCapFloorType.Cap, YoYInflationCapFloorType.Collar):
            qassert.require(len(self._cap_rates) > 0, "no cap rates given")
            while len(self._cap_rates) < len(self._yoy_leg):
                self._cap_rates.append(self._cap_rates[-1])
        if type_ in (YoYInflationCapFloorType.Floor, YoYInflationCapFloorType.Collar):
            qassert.require(len(self._floor_rates) > 0, "no floor rates given")
            while len(self._floor_rates) < len(self._yoy_leg):
                self._floor_rates.append(self._floor_rates[-1])

        # # C++ parity: registerWith every leg cashflow.
        for cf in self._yoy_leg:
            cf.register_with(self)

    @classmethod
    def from_strikes(
        cls,
        type_: YoYInflationCapFloorType,
        yoy_leg: Sequence[YoYInflationCouponLike],
        strikes: Sequence[float],
    ) -> YoYInflationCapFloor:
        """Single-strike-vector constructor (Cap or Floor only).

        # C++ parity: ``YoYInflationCapFloor::YoYInflationCapFloor(Type, Leg,
        # const std::vector<Rate>&)`` (inflationcapfloor.cpp:69-92).
        """
        qassert.require(len(strikes) > 0, "no strikes given")
        if type_ == YoYInflationCapFloorType.Cap:
            return cls(type_, yoy_leg, cap_rates=list(strikes), floor_rates=None)
        if type_ == YoYInflationCapFloorType.Floor:
            return cls(type_, yoy_leg, cap_rates=None, floor_rates=list(strikes))
        qassert.fail("only Cap/Floor types allowed in this constructor")

    # --- Instrument interface ------------------------------------------

    def is_expired(self) -> bool:
        """Expired iff every cashflow has occurred.

        # C++ parity: YoYInflationCapFloor::isExpired (inflationcapfloor.cpp:94-99).
        """
        return all(cf.has_occurred() for cf in self._yoy_leg)

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Fill the YoY cap/floor argument carrier.

        # C++ parity: YoYInflationCapFloor::setupArguments
        # (inflationcapfloor.cpp:133-179).
        """
        qassert.require(
            isinstance(args, YoYInflationCapFloorArguments),
            "wrong argument type (expected YoYInflationCapFloorArguments)",
        )
        assert isinstance(args, YoYInflationCapFloorArguments)

        n = len(self._yoy_leg)
        args.type = self._type
        args.start_dates = [Date()] * n
        args.fixing_dates = [Date()] * n
        args.pay_dates = [Date()] * n
        args.accrual_times = [0.0] * n
        args.nominals = [0.0] * n
        args.gearings = [0.0] * n
        args.spreads = [0.0] * n
        args.cap_rates = [NULL_RATE] * n
        args.floor_rates = [NULL_RATE] * n

        for i, coupon in enumerate(self._yoy_leg):
            args.start_dates[i] = coupon.accrual_start_date()
            args.fixing_dates[i] = coupon.fixing_date()
            args.pay_dates[i] = coupon.date()
            args.accrual_times[i] = coupon.accrual_period()
            args.nominals[i] = coupon.nominal()
            gearing = coupon.gearing()
            spread = coupon.spread()
            args.gearings[i] = gearing
            args.spreads[i] = spread
            if self._type in (
                YoYInflationCapFloorType.Cap,
                YoYInflationCapFloorType.Collar,
            ):
                args.cap_rates[i] = (self._cap_rates[i] - spread) / gearing
            if self._type in (
                YoYInflationCapFloorType.Floor,
                YoYInflationCapFloorType.Collar,
            ):
                args.floor_rates[i] = (self._floor_rates[i] - spread) / gearing

    # --- inspectors ----------------------------------------------------

    def type(self) -> YoYInflationCapFloorType:
        return self._type

    def cap_rates(self) -> list[float]:
        return list(self._cap_rates)

    def floor_rates(self) -> list[float]:
        return list(self._floor_rates)

    def yoy_leg(self) -> list[YoYInflationCouponLike]:
        return list(self._yoy_leg)

    def start_date(self) -> Date:
        """Earliest accrual start date in the leg.

        # C++ parity: YoYInflationCapFloor::startDate (inflationcapfloor.cpp:101-103).
        """
        qassert.require(len(self._yoy_leg) > 0, "empty leg")
        d: Date | None = None
        for cf in self._yoy_leg:
            sd = cf.accrual_start_date()
            d = sd if d is None else min(d, sd)
        assert d is not None
        return d

    def maturity_date(self) -> Date:
        """Latest payment date in the leg.

        # C++ parity: YoYInflationCapFloor::maturityDate (inflationcapfloor.cpp:105-107).
        """
        qassert.require(len(self._yoy_leg) > 0, "empty leg")
        d: Date | None = None
        for cf in self._yoy_leg:
            ed = cf.date()
            d = ed if d is None else max(d, ed)
        assert d is not None
        return d

    def last_yoy_inflation_coupon(self) -> YoYInflationCouponLike | None:
        """Last coupon in the leg.

        # C++ parity: YoYInflationCapFloor::lastYoYInflationCoupon
        # (inflationcapfloor.cpp:109-115).
        """
        if not self._yoy_leg:
            return None
        return self._yoy_leg[-1]

    def optionlet(self, n: int) -> YoYInflationCapFloor:
        """Return the n-th period as a new YoYInflationCapFloor.

        # C++ parity: YoYInflationCapFloor::optionlet (inflationcapfloor.cpp:117-131).
        """
        qassert.require(
            n < len(self._yoy_leg),
            f"optionlet {n + 1} does not exist; only {len(self._yoy_leg)}",
        )
        single = [self._yoy_leg[n]]
        cap: list[float] | None = None
        floor: list[float] | None = None
        if self._type in (YoYInflationCapFloorType.Cap, YoYInflationCapFloorType.Collar):
            cap = [self._cap_rates[n]]
        if self._type in (YoYInflationCapFloorType.Floor, YoYInflationCapFloorType.Collar):
            floor = [self._floor_rates[n]]
        return YoYInflationCapFloor(self._type, single, cap_rates=cap, floor_rates=floor)


class YoYInflationCap(YoYInflationCapFloor):
    """Concrete YoY cap. # C++ parity: inflationcapfloor.hpp:102-108."""

    def __init__(
        self,
        yoy_leg: Sequence[YoYInflationCouponLike],
        strikes: Sequence[float],
    ) -> None:
        super().__init__(
            YoYInflationCapFloorType.Cap,
            yoy_leg,
            cap_rates=list(strikes),
            floor_rates=None,
        )


class YoYInflationFloor(YoYInflationCapFloor):
    """Concrete YoY floor. # C++ parity: inflationcapfloor.hpp:112-118."""

    def __init__(
        self,
        yoy_leg: Sequence[YoYInflationCouponLike],
        strikes: Sequence[float],
    ) -> None:
        super().__init__(
            YoYInflationCapFloorType.Floor,
            yoy_leg,
            cap_rates=None,
            floor_rates=list(strikes),
        )


class YoYInflationCollar(YoYInflationCapFloor):
    """Concrete YoY collar (long cap + short floor).

    # C++ parity: inflationcapfloor.hpp:122-129.
    """

    def __init__(
        self,
        yoy_leg: Sequence[YoYInflationCouponLike],
        cap_rates: Sequence[float],
        floor_rates: Sequence[float],
    ) -> None:
        super().__init__(
            YoYInflationCapFloorType.Collar,
            yoy_leg,
            cap_rates=list(cap_rates),
            floor_rates=list(floor_rates),
        )


# Re-export CashFlow type for convenience (instrument users may construct
# the leg with mixed CashFlow types).
__all__ = [
    "NULL_RATE",
    "CashFlow",
    "YoYInflationCap",
    "YoYInflationCapFloor",
    "YoYInflationCapFloorArguments",
    "YoYInflationCapFloorResults",
    "YoYInflationCapFloorType",
    "YoYInflationCollar",
    "YoYInflationCouponLike",
    "YoYInflationFloor",
]
