"""BlackVolSurface — Black volatility (smile) surface base.

# C++ parity: ql/experimental/volatility/blackvolsurface.{hpp,cpp} (v1.42.1).

This abstract class extends :class:`BlackAtmVolCurve` with a per-expiry
*smile section*. Concrete subclasses implement ``_smile_section_impl(t)``
to return a :class:`SmileSection` for a given option time; the ATM
variance / volatility required by the ``BlackAtmVolCurve`` interface is
then derived from the smile at its own ATM level.

Volatilities are expressed on an annual basis.

Subclasses MUST override:

- ``max_date()``, ``min_strike()``, ``max_strike()``
- ``_smile_section_impl(t) -> SmileSection``

The public ``smile_section`` accessors (by tenor / date / time)
range-check and delegate to ``_smile_section_impl``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.experimental.volatility.black_atm_vol_curve import BlackAtmVolCurve
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class BlackVolSurface(BlackAtmVolCurve):
    """Abstract Black volatility (smile) surface.

    Implements the ``BlackAtmVolCurve`` ATM hooks in terms of the smile
    section: ``atm_variance(t) = smile(t).variance(smile(t).atm_level())``
    and likewise for the volatility.
    """

    # --- subclass-implemented hook -----------------------------------------

    @abstractmethod
    def _smile_section_impl(self, t: float) -> SmileSection:
        """Subclass: smile section for option time ``t``.

        Time check has already been performed; treat the call as if
        time-extrapolation is allowed.
        """

    # --- public smile-section API ------------------------------------------

    def smile_section(self, option_tenor: Period, extrapolate: bool) -> SmileSection:
        """Smile section for an option tenor.

        # C++ parity: BlackVolSurface::smileSection(const Period&, bool).
        """
        return self.smile_section_at_date(
            self.option_date_from_tenor(option_tenor), extrapolate
        )

    def smile_section_at_date(self, d: Date, extrapolate: bool) -> SmileSection:
        """Smile section for an option date."""
        return self.smile_section_at_time(self.time_from_reference(d), extrapolate)

    def smile_section_at_time(self, t: float, extrapolate: bool) -> SmileSection:
        """Smile section for an option time."""
        self.check_time_range(t, extrapolate)
        return self._smile_section_impl(t)

    # --- BlackAtmVolCurve interface ----------------------------------------

    def _atm_variance_impl(self, t: float) -> float:
        s = self._smile_section_impl(t)
        return s.variance(s.atm_level())

    def _atm_vol_impl(self, t: float) -> float:
        s = self._smile_section_impl(t)
        return s.volatility(s.atm_level())


__all__ = ["BlackVolSurface"]
