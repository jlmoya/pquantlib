"""BlackAtmVolCurve — Black at-the-money (no-smile) volatility curve base.

# C++ parity: ql/experimental/volatility/blackatmvolcurve.{hpp,cpp} (v1.42.1).

This abstract class defines the interface of concrete Black
at-the-money (no-smile) volatility *curves*. Volatilities are expressed
on an annual basis. It sits below the experimental ``BlackVolSurface``
(which adds a per-expiry smile section) and above the concrete
``AbcdAtmVolCurve``.

Subclasses MUST override:

- ``max_date()`` (inherited from ``TermStructure``)
- ``min_strike()`` / ``max_strike()`` (inherited from
  ``VolatilityTermStructure``)
- ``_atm_variance_impl(t)`` and ``_atm_vol_impl(t)``

The two impls must agree: ``variance(t) = vol(t)^2 * t``. The public
``atm_vol`` / ``atm_variance`` accessors (by tenor / date / time)
range-check and delegate to the impls.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility_term_structure import VolatilityTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class BlackAtmVolCurve(VolatilityTermStructure):
    """Abstract Black ATM (no-smile) volatility curve.

    Construction modes 1 (delegated) and 2 (fixed reference date) plus
    mode 3 (moving reference date via ``settlement_days``) are forwarded
    to ``VolatilityTermStructure``.
    """

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention = BusinessDayConvention.Following,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        settlement_days: int | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )

    # --- subclass-implemented hooks ----------------------------------------

    @abstractmethod
    def _atm_variance_impl(self, t: float) -> float:
        """Subclass: spot ATM variance at time ``t``.

        Range checks have already been performed; treat the call as if
        extrapolation is required.
        """

    @abstractmethod
    def _atm_vol_impl(self, t: float) -> float:
        """Subclass: spot ATM volatility at time ``t``."""

    # --- public ATM-vol API ------------------------------------------------

    def atm_vol(self, option_tenor: Period, extrapolate: bool = False) -> float:
        """Spot ATM volatility for an option tenor.

        # C++ parity: BlackAtmVolCurve::atmVol(const Period&, bool).
        """
        d = self.option_date_from_tenor(option_tenor)
        return self.atm_vol_at_date(d, extrapolate)

    def atm_vol_at_date(self, maturity: Date, extrapolate: bool = False) -> float:
        """Spot ATM volatility for an option maturity date."""
        t = self.time_from_reference(maturity)
        return self.atm_vol_at_time(t, extrapolate)

    def atm_vol_at_time(self, maturity: float, extrapolate: bool = False) -> float:
        """Spot ATM volatility for an option maturity time."""
        self.check_time_range(maturity, extrapolate)
        return self._atm_vol_impl(maturity)

    # --- public ATM-variance API -------------------------------------------

    def atm_variance(self, option_tenor: Period, extrapolate: bool = False) -> float:
        """Spot ATM variance for an option tenor."""
        d = self.option_date_from_tenor(option_tenor)
        return self.atm_variance_at_date(d, extrapolate)

    def atm_variance_at_date(self, maturity: Date, extrapolate: bool = False) -> float:
        """Spot ATM variance for an option maturity date."""
        t = self.time_from_reference(maturity)
        return self.atm_variance_at_time(t, extrapolate)

    def atm_variance_at_time(self, maturity: float, extrapolate: bool = False) -> float:
        """Spot ATM variance for an option maturity time."""
        self.check_time_range(maturity, extrapolate)
        return self._atm_variance_impl(maturity)


__all__ = ["BlackAtmVolCurve"]
