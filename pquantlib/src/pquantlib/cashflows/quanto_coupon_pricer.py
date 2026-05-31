"""BlackIborQuantoCouponPricer — quanto-adjusted IBOR coupon pricer.

# C++ parity: ql/experimental/coupons/quantocouponpricer.hpp + .cpp (v1.42.1,
# 099987f0).

Applies the Hull quanto adjustment (Hull 6th ed., p. 642, generalised to
shifted-lognormal and normal caplet volatilities) to the coupon's forward
fixing, then delegates to the base IBOR pricer's ``adjustedFixing``.

# C++ parity divergence (delegation base): the Python
# :class:`~pquantlib.cashflows.coupon_pricer.BlackIborCouponPricer` does not
# carry an OptionletVolatilityStructure (cap/floor pricing is a deferred
# carve-out), so the quanto adjustment uses *this* pricer's caplet vol surface
# directly; the post-adjustment delegation goes through the base pricer's
# ``_adjusted_fixing`` (par-coupon span only — no extra Black convexity term,
# matching the deferred-cap/floor base behaviour).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType


class BlackIborQuantoCouponPricer(BlackIborCouponPricer):
    """Quanto-adjusted Black IBOR coupon pricer."""

    def __init__(
        self,
        fx_rate_black_volatility: BlackVolTermStructure,
        underlying_fx_correlation: Quote,
        caplet_volatility: OptionletVolatilityStructure,
    ) -> None:
        super().__init__()
        self._fx_rate_black_volatility: BlackVolTermStructure = fx_rate_black_volatility
        self._underlying_fx_correlation: Quote = underlying_fx_correlation
        self._caplet_volatility: OptionletVolatilityStructure = caplet_volatility

    # --- inspectors ------------------------------------------------------------

    def fx_rate_black_volatility(self) -> BlackVolTermStructure:
        return self._fx_rate_black_volatility

    def underlying_fx_correlation(self) -> Quote:
        return self._underlying_fx_correlation

    def caplet_volatility(self) -> OptionletVolatilityStructure:
        return self._caplet_volatility

    # --- quanto adjustment -----------------------------------------------------

    def quanto_adjusted_fixing(self, fixing: float | None = None) -> float:
        """Hull quanto adjustment applied to ``fixing``.

        # C++ parity: BlackIborQuantoCouponPricer::adjustedFixing
        # (quantocouponpricer.cpp lines 31-64). The post-adjustment delegation
        # to the base ``adjustedFixing`` is done by the caller / by
        # ``_adjusted_fixing``.

        For a fixing date at or before the caplet-vol reference date the
        adjustment is the identity (no forward optionality left).
        """
        if fixing is None:
            qassert.require(self._coupon is not None, "coupon not set")
            assert self._coupon is not None
            fixing = self._coupon.index_fixing()

        qassert.require(self._coupon is not None, "coupon not set")
        assert self._coupon is not None

        d1 = self._coupon.fixing_date()
        reference_date = self._caplet_volatility.reference_date()

        if d1 > reference_date:
            t1 = self._caplet_volatility.time_from_reference(d1)
            fxsigma = self._fx_rate_black_volatility.black_vol(d1, fixing, True)
            sigma = self._caplet_volatility.volatility(d1, fixing)
            rho = self._underlying_fx_correlation.value()

            # Hull 6th ed., p. 642 — generalised to shifted-lognormal / normal.
            if self._caplet_volatility.volatility_type() == VolatilityType.ShiftedLognormal:
                d_quanto_adj = math.exp(sigma * fxsigma * rho * t1)
                shift = self._caplet_volatility.displacement()
                fixing = (fixing + shift) * d_quanto_adj - shift
            else:
                d_quanto_adj = sigma * fxsigma * rho * t1
                fixing += d_quanto_adj

        return fixing

    def _adjusted_fixing(self, fixing: float | None = None) -> float:
        """Quanto-adjust first, then delegate to the base IBOR adjustment.

        # C++ parity: ``return BlackIborCouponPricer::adjustedFixing(fixing);``
        # at the tail of quantocouponpricer.cpp.
        """
        adjusted = self.quanto_adjusted_fixing(fixing)
        return super()._adjusted_fixing(adjusted)
