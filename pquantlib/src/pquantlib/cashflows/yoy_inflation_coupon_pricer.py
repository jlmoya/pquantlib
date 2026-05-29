"""YoYInflationCouponPricer — analytic + Black/Bachelier YoY pricers.

# C++ parity: ql/cashflows/inflationcouponpricer.{hpp,cpp} (v1.42.1) —
# YoYInflationCouponPricer + BlackYoYInflationCouponPricer +
# UnitDisplacedBlackYoYInflationCouponPricer + BachelierYoYInflationCouponPricer.

The base ``YoYInflationCouponPricer`` already handles swaplet rate /
price (no vol surface needed). The three subclasses differ only in
``optionletPriceImp``:

* ``BlackYoYInflationCouponPricer`` — ``blackFormula(opt, K, F, stdDev)``.
* ``UnitDisplacedBlackYoYInflationCouponPricer`` — Black with
  ``K' = K + 1, F' = F + 1`` (handles negative-rate cap/floor).
* ``BachelierYoYInflationCouponPricer`` — Bachelier (normal-model) formula.

Python divergences from C++:

- The cap/floor branches in C++ check ``fixingDate <= capletVol.baseDate``
  to short-circuit to the deterministic intrinsic value. We mirror that,
  using the vol surface's ``base_date()`` if a surface is set; without
  a surface, we fall through to the implementation pricer (which raises
  for the base class).
- The C++ pricer holds raw ``Handle<YoYOptionletVolatilitySurface>``;
  ours holds ``object | None`` to avoid the L7-D coupling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.inflation_coupon_pricer import InflationCouponPricer
from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    black_formula,
)

if TYPE_CHECKING:
    from pquantlib.cashflows.inflation_coupon import InflationCoupon
    from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class YoYInflationCouponPricer(InflationCouponPricer):
    """Analytic swaplet pricer for ``YoYInflationCoupon``.

    # C++ parity: ``YoYInflationCouponPricer`` in inflationcouponpricer.hpp:88-142.

    The base class can already price swaplets (no vol needed). Cap/floor
    methods require an optionlet-volatility surface — they raise here
    unless overridden by Black/Bachelier/UnitDisplaced subclasses with a
    surface wired in.
    """

    def __init__(
        self,
        nominal_term_structure: YieldTermStructureProtocol | None = None,
        caplet_vol: object | None = None,
    ) -> None:
        super().__init__()
        self._nominal_term_structure: YieldTermStructureProtocol | None = nominal_term_structure
        self._caplet_vol: object | None = caplet_vol
        # Per-coupon state populated by ``initialize``:
        self._coupon: YoYInflationCoupon | None = None
        self._gearing: float = 1.0
        self._spread: float = 0.0
        self._discount: float | None = 1.0

    # ---- inspectors --------------------------------------------------

    def caplet_volatility(self) -> object | None:
        """C++ parity: ql/cashflows/inflationcouponpricer.hpp:97-99 (inline)."""
        return self._caplet_vol

    def nominal_term_structure(self) -> YieldTermStructureProtocol | None:
        """C++ parity: ql/cashflows/inflationcouponpricer.hpp:101-103 (inline)."""
        return self._nominal_term_structure

    def set_caplet_volatility(self, caplet_vol: object) -> None:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:52-57."""
        qassert.require(caplet_vol is not None, "empty capletVol handle")
        self._caplet_vol = caplet_vol

    # ---- pricer wiring -----------------------------------------------

    def initialize(self, coupon: InflationCoupon) -> None:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:136-154."""
        # Local import for the coupon ↔ pricer cycle.
        from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon  # noqa: PLC0415

        qassert.require(
            isinstance(coupon, YoYInflationCoupon),
            "year-on-year inflation coupon needed",
        )
        assert isinstance(coupon, YoYInflationCoupon)
        self._coupon = coupon
        self._gearing = coupon.gearing()
        self._spread = coupon.spread()
        self._payment_date = coupon.date()

        self._discount = 1.0
        if self._nominal_term_structure is None:
            self._discount = None
        else:
            ref = self._nominal_term_structure.reference_date()
            if self._payment_date > ref:
                self._discount = self._nominal_term_structure.discount(self._payment_date)

    # ---- InflationCouponPricer interface -----------------------------

    def swaplet_rate(self) -> float:
        """``gearing * adjustedFixing + spread``.

        # C++ parity: ql/cashflows/inflationcouponpricer.cpp:163-169.
        """
        return self._gearing * self.adjusted_fixing() + self._spread

    def swaplet_price(self) -> float:
        """``swaplet_rate * accrual_period * discount``.

        # C++ parity: ql/cashflows/inflationcouponpricer.cpp:157-160.
        """
        qassert.require(
            self._discount is not None,
            "no nominal term structure provided",
        )
        assert self._discount is not None
        return self.swaplet_rate() * self._coupon_required().accrual_period() * self._discount

    def caplet_price(self, effective_cap: float) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:65-68."""
        return self._gearing * self._optionlet_price(OptionType.Call, effective_cap)

    def caplet_rate(self, effective_cap: float) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:75-77."""
        return self._gearing * self._optionlet_rate(OptionType.Call, effective_cap)

    def floorlet_price(self, effective_floor: float) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:60-63."""
        return self._gearing * self._optionlet_price(OptionType.Put, effective_floor)

    def floorlet_rate(self, effective_floor: float) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:71-73."""
        return self._gearing * self._optionlet_rate(OptionType.Put, effective_floor)

    # ---- adjusted-fixing hook ----------------------------------------

    def adjusted_fixing(self, fixing: float | None = None) -> float:
        """No convexity adjustment in the base class.

        # C++ parity: ql/cashflows/inflationcouponpricer.cpp:126-133.
        """
        if fixing is None:
            fixing = self._coupon_required().index_fixing()
        return fixing

    # ---- helpers / internal ------------------------------------------

    def _coupon_required(self) -> YoYInflationCoupon:
        qassert.require(self._coupon is not None, "coupon not initialized")
        assert self._coupon is not None
        return self._coupon

    def _optionlet_price(self, option_type: OptionType, eff_strike: float) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:89-93."""
        qassert.require(
            self._discount is not None,
            "no nominal term structure provided",
        )
        assert self._discount is not None
        cpn = self._coupon_required()
        return self._optionlet_rate(option_type, eff_strike) * cpn.accrual_period() * self._discount

    def _optionlet_rate(self, option_type: OptionType, eff_strike: float) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:96-123.

        Two branches:

        * If ``fixingDate <= capletVol.baseDate()``, the amount is
          determined → return ``max(a - b, 0)`` (intrinsic value).
        * Otherwise call ``optionletPriceImp`` with ``(strike, forward,
          stdDev)`` derived from the surface.

        Without a vol surface the second branch raises; the first branch
        still requires a vol surface in C++ (``capletVol.baseDate()``).
        We mirror that constraint.
        """
        cpn = self._coupon_required()
        fixing_date = cpn.fixing_date()
        if self._caplet_vol is not None:
            base_date_fn = getattr(self._caplet_vol, "base_date", None)
            if base_date_fn is not None:
                base_date = base_date_fn()
                if fixing_date <= base_date:
                    # intrinsic value
                    if option_type == OptionType.Call:
                        a = cpn.index_fixing()
                        b = eff_strike
                    else:
                        a = eff_strike
                        b = cpn.index_fixing()
                    return max(a - b, 0.0)

        # Forward-looking case — need the vol surface to extract stdDev.
        qassert.require(
            self._caplet_vol is not None,
            "missing optionlet volatility",
        )
        # Total variance lookup — C++ uses
        # ``capletVol.totalVariance(fixingDate, effStrike, Period(0, Days))``.
        total_variance_fn = getattr(self._caplet_vol, "total_variance", None)
        qassert.require(
            total_variance_fn is not None,
            "caplet volatility surface does not expose total_variance(...)",
        )
        assert total_variance_fn is not None
        std_dev = math.sqrt(total_variance_fn(fixing_date, eff_strike))
        return self._optionlet_price_imp(
            option_type, eff_strike, self.adjusted_fixing(), std_dev
        )

    def _optionlet_price_imp(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
    ) -> float:
        """Subclass hook — raise in the base class.

        # C++ parity: ql/cashflows/inflationcouponpricer.cpp:80-86 —
        # ``QL_FAIL("you must implement this to get a vol-dependent price")``.
        """
        del option_type, strike, forward, std_dev
        msg = "you must implement optionlet_price_imp to get a vol-dependent price"
        raise LibraryException(msg)


# -----------------------------------------------------------------------
# Black / UnitDisplaced-Black / Bachelier specialisations
# -----------------------------------------------------------------------


class BlackYoYInflationCouponPricer(YoYInflationCouponPricer):
    """Lognormal-Black YoY optionlet pricer.

    # C++ parity: ``BlackYoYInflationCouponPricer`` in
    # inflationcouponpricer.hpp:146-162.
    """

    def _optionlet_price_imp(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
    ) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:178-188."""
        return black_formula(option_type, strike, forward, std_dev)


class UnitDisplacedBlackYoYInflationCouponPricer(YoYInflationCouponPricer):
    """Unit-displaced Black YoY pricer (handles negative rates).

    # C++ parity: ``UnitDisplacedBlackYoYInflationCouponPricer`` in
    # inflationcouponpricer.hpp:166-182.
    """

    def _optionlet_price_imp(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
    ) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:190-200."""
        return black_formula(option_type, strike + 1.0, forward + 1.0, std_dev)


class BachelierYoYInflationCouponPricer(YoYInflationCouponPricer):
    """Bachelier-formula (normal model) YoY pricer.

    # C++ parity: ``BachelierYoYInflationCouponPricer`` in
    # inflationcouponpricer.hpp:186-202.
    """

    def _optionlet_price_imp(
        self,
        option_type: OptionType,
        strike: float,
        forward: float,
        std_dev: float,
    ) -> float:
        """C++ parity: ql/cashflows/inflationcouponpricer.cpp:202-211."""
        return bachelier_black_formula(option_type, strike, forward, std_dev)
