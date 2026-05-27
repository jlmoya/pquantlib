"""LocalVolSurface — local volatility derived from a BlackVolTermStructure.

# C++ parity: ql/termstructures/volatility/equityfx/localvolsurface.hpp +
#             localvolsurface.cpp (v1.42.1).

The Dupire formula for the local volatility from a Black-vol surface
parameterized by total variance ``w(t, y)`` with ``y = log(K / F(t))``
is::

    sigma_L^2 = (dw/dt) /
                ( 1 - (y/w) * dw/dy
                  + 0.25 * (-0.25 - 1/w + y^2/w^2) * (dw/dy)^2
                  + 0.5 * d^2 w / dy^2 )

C++ obtains the forward ``F(t)`` from a risk-free and a dividend
``YieldTermStructure`` plus an underlying spot quote — none of which
exist yet in PQuantLib (YieldTermStructure is the L2-B cluster). L2-E
is contractually independent of L2-B, so this port adopts the
**flat-curve simplification**: zero risk-free rate, zero dividend
yield, so ``F(t) = S_0`` for all t. The L2-B implementation will
generalize this (a follow-up commit will add a constructor that
accepts the two yield curves once they exist).

Derivative scheme matches C++ exactly (so the JSON reference values
agree to ~14 digits even for the trivial forward case):

- Strike step ``dy = 1e-4 * |y|`` when ``|y| > 1e-3``, else ``1e-6``.
- Time step ``dt = min(1e-4, t / 2)`` (forward FD at t=0, central
  otherwise).

The C++ class issues monotonicity guards on the time derivative;
PQuantLib mirrors them (raise on decreasing variance in time).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class LocalVolSurface(LocalVolTermStructure):
    """Local volatility surface derived from a BlackVolTermStructure (Dupire 2-D).

    Currently assumes zero risk-free rate and zero dividend yield —
    forward = spot. A future constructor (after L2-B lands
    YieldTermStructure) will accept rate / dividend curves.
    """

    def __init__(
        self,
        *,
        black_ts: BlackVolTermStructure,
        underlying: float | Quote,
    ) -> None:
        super().__init__(
            business_day_convention=black_ts.business_day_convention(),
            day_counter=black_ts.day_counter(),
        )
        self._black_ts: BlackVolTermStructure = black_ts
        self._underlying: Quote = (
            underlying if isinstance(underlying, Quote) else SimpleQuote(float(underlying))
        )
        self._black_ts.register_with(self)
        self._underlying.register_with(self)

    # --- delegated TermStructure accessors ---------------------------------

    def reference_date(self) -> Date:
        return self._black_ts.reference_date()

    def calendar(self) -> Calendar:
        return self._black_ts.calendar()

    def max_date(self) -> Date:
        return self._black_ts.max_date()

    # --- delegated VolatilityTermStructure accessors -----------------------

    def min_strike(self) -> float:
        return self._black_ts.min_strike()

    def max_strike(self) -> float:
        return self._black_ts.max_strike()

    # --- local-vol impl ----------------------------------------------------

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        # Flat-curve assumption: forward = spot.
        forward = self._underlying.value()

        strike = underlying_level
        y = math.log(strike / forward)
        dy = y * 1.0e-4 if abs(y) > 1.0e-3 else 1.0e-6
        strikep = strike * math.exp(dy)
        strikem = strike / math.exp(dy)

        bts = self._black_ts
        w = bts.black_variance_at_time(t, strike, extrapolate=True)
        wp = bts.black_variance_at_time(t, strikep, extrapolate=True)
        wm = bts.black_variance_at_time(t, strikem, extrapolate=True)
        dwdy = (wp - wm) / (2.0 * dy)
        d2wdy2 = (wp - 2.0 * w + wm) / (dy * dy)

        # Time derivative.
        if t == 0.0:
            dt = 1.0e-4
            # With flat curves, strikept = strike (dr/dq cancel).
            wpt = bts.black_variance_at_time(t + dt, strike, extrapolate=True)
            qassert.require(
                wpt >= w,
                f"decreasing variance at strike {strike} "
                f"between time {t} and time {t + dt}",
            )
            dwdt = (wpt - w) / dt
        else:
            dt = min(1.0e-4, t / 2.0)
            wpt = bts.black_variance_at_time(t + dt, strike, extrapolate=True)
            wmt = bts.black_variance_at_time(t - dt, strike, extrapolate=True)
            qassert.require(
                wpt >= w,
                f"decreasing variance at strike {strike} "
                f"between time {t} and time {t + dt}",
            )
            qassert.require(
                w >= wmt,
                f"decreasing variance at strike {strike} "
                f"between time {t - dt} and time {t}",
            )
            dwdt = (wpt - wmt) / (2.0 * dt)

        if dwdy == 0.0 and d2wdy2 == 0.0:  # avoid /w when w might be 0
            return math.sqrt(dwdt)

        den1 = 1.0 - y / w * dwdy
        den2 = 0.25 * (-0.25 - 1.0 / w + y * y / w / w) * dwdy * dwdy
        den3 = 0.5 * d2wdy2
        den = den1 + den2 + den3
        result = dwdt / den

        qassert.require(
            result >= 0.0,
            f"negative local vol^2 at strike {strike} and time {t}; "
            "the black vol surface is not smooth enough",
        )
        return math.sqrt(result)
