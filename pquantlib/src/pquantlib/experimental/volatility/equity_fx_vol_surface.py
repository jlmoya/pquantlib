"""EquityFXVolSurface — equity/FX volatility (smile) surface base.

# C++ parity: ql/experimental/volatility/equityfxvolsurface.{hpp,cpp} (v1.42.1).

This abstract class extends :class:`BlackVolSurface` with the notion of
*forward (at-the-money) volatility / variance* between two dates. As the
C++ docs note: the concept of an ATM-forward vol only makes sense in the
absence of a smile, so these methods are computed from the ATM variance
of the underlying ``BlackAtmVolCurve`` interface.

Volatilities are expressed on an annual basis.

Subclasses MUST override the same hooks as :class:`BlackVolSurface`
(``max_date``, ``min_strike``, ``max_strike``, ``_smile_section_impl``).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.experimental.volatility.black_vol_surface import BlackVolSurface
from pquantlib.time.date import Date


class EquityFXVolSurface(BlackVolSurface):
    """Abstract equity/FX volatility (smile) surface.

    Adds ``atm_forward_vol`` / ``atm_forward_variance`` over the
    :class:`BlackVolSurface` API.
    """

    # --- forward ATM vol/variance ------------------------------------------

    def atm_forward_vol(
        self, date1: Date, date2: Date, extrapolate: bool = False
    ) -> float:
        """Forward (ATM) volatility between two dates.

        # C++ parity: EquityFXVolSurface::atmForwardVol(Date, Date, bool).
        """
        qassert.require(date1 < date2, "wrong dates")
        t1 = self.time_from_reference(date1)
        t2 = self.time_from_reference(date2)
        return self.atm_forward_vol_at_time(t1, t2, extrapolate)

    def atm_forward_vol_at_time(
        self, time1: float, time2: float, extrapolate: bool = False
    ) -> float:
        """Forward (ATM) volatility between two times."""
        fwd_variance = self.atm_forward_variance_at_time(time1, time2, extrapolate)
        t = time2 - time1
        return math.sqrt(fwd_variance / t)

    def atm_forward_variance(
        self, date1: Date, date2: Date, extrapolate: bool = False
    ) -> float:
        """Forward (ATM) variance between two dates."""
        qassert.require(date1 < date2, "wrong dates")
        t1 = self.time_from_reference(date1)
        t2 = self.time_from_reference(date2)
        return self.atm_forward_variance_at_time(t1, t2, extrapolate)

    def atm_forward_variance_at_time(
        self, time1: float, time2: float, extrapolate: bool = False
    ) -> float:
        """Forward (ATM) variance between two times.

        # C++ parity: EquityFXVolSurface::atmForwardVariance(Time, Time, bool).
        """
        qassert.require(time1 < time2, "wrong times")
        var1 = self.atm_variance_at_time(time1, extrapolate)
        var2 = self.atm_variance_at_time(time2, extrapolate)
        qassert.require(var1 < var2, "non-increasing variances")
        return var2 - var1


__all__ = ["EquityFXVolSurface"]
