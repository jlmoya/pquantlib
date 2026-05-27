"""Swap — abstract interest-rate swap (two or more legs).

# C++ parity: ql/instruments/swap.{hpp,cpp} (v1.42.1).

The C++ ``Swap`` derives from ``Instrument`` and carries:

* ``legs_`` — ``std::vector<Leg>`` (each leg is a ``std::vector<shared_ptr<CashFlow>>``).
* ``payer_`` — ``std::vector<Real>`` (sign multiplier per leg, -1.0 if the
  leg is paid by us, +1.0 if received).
* Mutable result caches: ``legNPV_`` / ``legBPS_`` / ``startDiscounts_`` /
  ``endDiscounts_`` / ``npvDateDiscount_``.

The C++ class is abstract in practice — concrete subclasses override
``setupArguments`` for product-specific argument carriers, but the
``Swap::engine`` plays well with ``DiscountingSwapEngine`` directly.

Python port:

- ``Type`` IntEnum mirrors C++ enum (Receiver=-1, Payer=1).
- Legs and payer multipliers are passed in constructor (either as
  ``(first_leg, second_leg)`` shorthand or full multi-leg ``(legs, payer_bools)``).
- The ``SwapArguments`` carrier holds ``legs`` + ``payer`` lists; engines
  (e.g. ``DiscountingSwapEngine``) write into ``SwapResults`` which
  carries per-leg NPV/BPS + start/end discount arrays + npvDateDiscount.
- We expose ``fetch_results`` so the inherited result fields are
  populated for ``Instrument.npv()`` etc. Subclass intermediates
  (``FixedVsFloatingSwap``) override to read their custom result fields.
- ``Observable`` wiring: C++ ``registerWith(cashflow)`` is mirrored by
  ``cf.register_with(self)`` so leg-level updates propagate.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon import Coupon
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow

# Leg is a sequence of cashflows. Use Sequence (covariant) at the input
# boundary so callers can pass narrower-typed lists (e.g.
# ``list[FixedRateCoupon]``); internally we store ``list[CashFlow]`` for
# mutation. Use ``LegInput`` for parameters and ``Leg`` for stored data.
type Leg = list["CashFlow"]
type LegInput = Sequence["CashFlow"]


def leg_start_date(leg: LegInput) -> Date:
    """Equivalent of C++ ``CashFlows::startDate`` (cashflows.cpp:78-98).

    Returns ``min(accrual_start_date)`` over Coupon flows, falling back
    to the earliest cashflow ``date()`` for non-coupon legs.
    """
    qassert.require(len(leg) > 0, "empty leg")
    # Initialize with a coupon-aware sentinel.
    d: Date | None = None
    for cf in leg:
        sd = cf.accrual_start_date() if isinstance(cf, Coupon) else cf.date()
        d = sd if d is None else min(d, sd)
    assert d is not None
    return d


def leg_maturity_date(leg: LegInput) -> Date:
    """Equivalent of C++ ``CashFlows::maturityDate`` (cashflows.cpp:100-117).

    Returns ``max(accrual_end_date)`` over Coupon flows, falling back
    to the latest cashflow ``date()`` for non-coupon legs.
    """
    qassert.require(len(leg) > 0, "empty leg")
    d: Date | None = None
    for cf in leg:
        ed = cf.accrual_end_date() if isinstance(cf, Coupon) else cf.date()
        d = ed if d is None else max(d, ed)
    assert d is not None
    return d


class SwapType(IntEnum):
    """Swap side — referenced to the fixed leg in most concrete swaps.

    # C++ parity: ``Swap::Type`` enum (swap.hpp). C++ uses {Receiver=-1,
    # Payer=1}; we preserve the int values so existing code can multiply
    # by ``int(SwapType.Payer)`` directly if needed.
    """

    Receiver = -1
    Payer = 1


class SwapArguments(PricingEngineArguments):
    """Arguments carrier for swap pricing engines.

    # C++ parity: ``Swap::arguments`` (swap.hpp:142-147). Holds the
    # legs and the payer multipliers.
    """

    def __init__(self) -> None:
        self.legs: list[Leg] = []
        self.payer: list[float] = []

    def validate(self) -> None:
        qassert.require(
            len(self.legs) == len(self.payer),
            "number of legs and multipliers differ",
        )


class SwapResults(InstrumentResults):
    """Results carrier for swap pricing engines.

    # C++ parity: ``Swap::results`` (swap.hpp:149-156). Per-leg NPV/BPS
    # + start/end discount factors + valuation-date discount.
    """

    def __init__(self) -> None:
        super().__init__()
        self.leg_npv: list[float] = []
        self.leg_bps: list[float] = []
        self.start_discounts: list[float] = []
        self.end_discounts: list[float] = []
        self.npv_date_discount: float | None = None

    def reset(self) -> None:
        super().reset()
        self.leg_npv = []
        self.leg_bps = []
        self.start_discounts = []
        self.end_discounts = []
        self.npv_date_discount = None


class SwapEngine(GenericEngine[SwapArguments, SwapResults]):
    """Base swap pricing engine.

    # C++ parity: ``Swap::engine`` typedef of GenericEngine<Swap::arguments,
    # Swap::results>. Subclasses (DiscountingSwapEngine) implement
    # ``calculate``.
    """

    def __init__(self) -> None:
        super().__init__(SwapArguments(), SwapResults())


class Swap(Instrument):
    """Abstract interest-rate swap.

    The cashflows in the first leg are paid; cashflows in the second leg
    are received (matches C++ semantics in the two-leg constructor).
    Use :meth:`from_legs` for the two-leg shorthand or :meth:`from_multi`
    for the explicit per-leg payer multi-leg flavour.
    """

    def __init__(self, n_legs: int = 0) -> None:
        """Protected constructor — used by subclasses that build legs themselves.

        # C++ parity: ``Swap::Swap(Size legs)`` (swap.cpp:62-66). Allocates
        # the leg / payer / result vectors at the right size; the subclass
        # populates legs_[i] before calculations run.
        """
        super().__init__()
        self._legs: list[Leg] = [[] for _ in range(n_legs)]
        self._payer: list[float] = [1.0] * n_legs
        self._leg_npv: list[float | None] = [None] * n_legs
        self._leg_bps: list[float | None] = [None] * n_legs
        self._start_discounts: list[float | None] = [None] * n_legs
        self._end_discounts: list[float | None] = [None] * n_legs
        self._npv_date_discount: float | None = None

    @classmethod
    def from_legs(cls, first_leg: LegInput, second_leg: LegInput) -> Swap:
        """Two-leg shorthand: first leg is paid (-1), second leg is received (+1).

        # C++ parity: ``Swap::Swap(const Leg&, const Leg&)``.
        """
        s = cls(2)
        s._legs = [list(first_leg), list(second_leg)]
        s._payer = [-1.0, 1.0]
        for cf in s._legs[0]:
            cf.register_with(s)
        for cf in s._legs[1]:
            cf.register_with(s)
        return s

    @classmethod
    def from_multi(cls, legs: Sequence[LegInput], payer: Sequence[bool]) -> Swap:
        """Multi-leg constructor; ``payer[i]=True`` means the i-th leg is paid (-1).

        # C++ parity: ``Swap::Swap(const std::vector<Leg>&, const std::vector<bool>&)``.
        """
        qassert.require(
            len(legs) == len(payer),
            f"size mismatch between payer ({len(payer)}) and legs ({len(legs)})",
        )
        s = cls(len(legs))
        s._legs = [list(leg) for leg in legs]
        s._payer = [-1.0 if p else 1.0 for p in payer]
        for leg in s._legs:
            for cf in leg:
                cf.register_with(s)
        return s

    # --- Instrument interface ----------------------------------------------

    def is_expired(self) -> bool:
        """Expired iff every cashflow in every leg has occurred.

        # C++ parity: ``Swap::isExpired`` (swap.cpp:68-76). C++ calls
        # ``hasOccurred()`` with the global evaluation date implicitly;
        # PQuantLib's CashFlow.has_occurred uses an explicit ref_date —
        # for the Instrument lifecycle (called from ``calculate``) we
        # treat no ref_date as never-occurred, so a Swap with at least
        # one cashflow is never reported as expired purely on that
        # basis. Concrete-engine code consults the curve reference date
        # when reading flows.
        """
        for leg in self._legs:
            for cf in leg:
                if not cf.has_occurred():
                    return False
        return True

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy legs + payer multipliers into the engine's argument carrier.

        # C++ parity: ``Swap::setupArguments`` (swap.cpp:87-93).
        """
        qassert.require(
            isinstance(args, SwapArguments), "wrong argument type"
        )
        assert isinstance(args, SwapArguments)
        args.legs = [list(leg) for leg in self._legs]
        args.payer = list(self._payer)

    def fetch_results(self, results) -> None:  # type: ignore[no-untyped-def]
        """Pull leg NPVs + BPS + discount caches out of the engine results.

        # C++ parity: ``Swap::fetchResults`` (swap.cpp:95-140).
        """
        super().fetch_results(results)
        qassert.require(
            isinstance(results, SwapResults), "wrong result type"
        )
        assert isinstance(results, SwapResults)
        n = len(self._legs)
        if results.leg_npv:
            qassert.require(
                len(results.leg_npv) == n, "wrong number of leg NPV returned"
            )
            self._leg_npv = list(results.leg_npv)
        else:
            self._leg_npv = [None] * n
        if results.leg_bps:
            qassert.require(
                len(results.leg_bps) == n, "wrong number of leg BPS returned"
            )
            self._leg_bps = list(results.leg_bps)
        else:
            self._leg_bps = [None] * n
        if results.start_discounts:
            qassert.require(
                len(results.start_discounts) == n,
                "wrong number of leg start discounts returned",
            )
            self._start_discounts = list(results.start_discounts)
        else:
            self._start_discounts = [None] * n
        if results.end_discounts:
            qassert.require(
                len(results.end_discounts) == n,
                "wrong number of leg end discounts returned",
            )
            self._end_discounts = list(results.end_discounts)
        else:
            self._end_discounts = [None] * n
        self._npv_date_discount = results.npv_date_discount

    def setup_expired(self) -> None:
        """Zero out per-leg caches on top of the base expiration logic.

        # C++ parity: ``Swap::setupExpired`` (swap.cpp:78-85).
        """
        super().setup_expired()
        n = len(self._legs)
        self._leg_bps = [0.0] * n
        self._leg_npv = [0.0] * n
        self._start_discounts = [0.0] * n
        self._end_discounts = [0.0] * n
        self._npv_date_discount = 0.0

    # --- inspectors ---------------------------------------------------------

    def number_of_legs(self) -> int:
        return len(self._legs)

    def legs(self) -> list[Leg]:
        return self._legs

    def leg(self, j: int) -> Leg:
        qassert.require(
            j < len(self._legs), f"leg #{j} doesn't exist!"
        )
        return self._legs[j]

    def payer(self, j: int) -> bool:
        """True iff leg ``j`` is paid by the swap holder (sign -1)."""
        qassert.require(
            j < len(self._legs), f"leg #{j} doesn't exist!"
        )
        return self._payer[j] < 0.0

    def start_date(self) -> Date:
        """Earliest start date across all legs.

        # C++ parity: ``Swap::startDate`` (swap.cpp:146-152).
        """
        qassert.require(len(self._legs) > 0, "no legs given")
        d = leg_start_date(self._legs[0])
        for j in range(1, len(self._legs)):
            d = min(d, leg_start_date(self._legs[j]))
        return d

    def maturity_date(self) -> Date:
        """Latest maturity date across all legs.

        # C++ parity: ``Swap::maturityDate`` (swap.cpp:154-160).
        """
        qassert.require(len(self._legs) > 0, "no legs given")
        d = leg_maturity_date(self._legs[0])
        for j in range(1, len(self._legs)):
            d = max(d, leg_maturity_date(self._legs[j]))
        return d

    def leg_npv(self, j: int) -> float:
        """Per-leg NPV. Triggers calculation if cache is dirty."""
        qassert.require(
            j < len(self._legs), f"leg #{j} doesn't exist!"
        )
        self.calculate()
        qassert.require(
            self._leg_npv[j] is not None, "result not available"
        )
        assert self._leg_npv[j] is not None
        return self._leg_npv[j]  # type: ignore[return-value]

    def leg_bps(self, j: int) -> float:
        """Per-leg basis-point sensitivity. Triggers calculation."""
        qassert.require(
            j < len(self._legs), f"leg# {j} doesn't exist!"
        )
        self.calculate()
        qassert.require(
            self._leg_bps[j] is not None, "result not available"
        )
        assert self._leg_bps[j] is not None
        return self._leg_bps[j]  # type: ignore[return-value]

    def start_discounts(self, j: int) -> float:
        qassert.require(
            j < len(self._legs), f"leg #{j} doesn't exist!"
        )
        self.calculate()
        qassert.require(
            self._start_discounts[j] is not None, "result not available"
        )
        assert self._start_discounts[j] is not None
        return self._start_discounts[j]  # type: ignore[return-value]

    def end_discounts(self, j: int) -> float:
        qassert.require(
            j < len(self._legs), f"leg #{j} doesn't exist!"
        )
        self.calculate()
        qassert.require(
            self._end_discounts[j] is not None, "result not available"
        )
        assert self._end_discounts[j] is not None
        return self._end_discounts[j]  # type: ignore[return-value]

    def npv_date_discount(self) -> float:
        self.calculate()
        qassert.require(
            self._npv_date_discount is not None, "result not available"
        )
        assert self._npv_date_discount is not None
        return self._npv_date_discount


__all__ = [
    "Leg",
    "LegInput",
    "Swap",
    "SwapArguments",
    "SwapEngine",
    "SwapResults",
    "SwapType",
    "leg_maturity_date",
    "leg_start_date",
]
