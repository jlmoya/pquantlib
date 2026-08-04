"""PiecewiseBlackVarianceSurface — Black-vol surface built from smile sections.

# C++ parity: ql/termstructures/volatility/equityfx/piecewiseblackvariancesurface.hpp +
#             piecewiseblackvariancesurface.cpp (v1.43).

One :class:`SmileSection` per tenor, interpolated linearly in *total
variance* between tenors at a fixed strike. Unlike
:class:`~pquantlib.termstructures.volatility.equity_fx.black_variance_surface.BlackVarianceSurface`,
which interpolates a rectangular vol grid, each tenor here carries a
full smile object, so a parametric shape (SVI, SABR, …) survives intact
instead of being sampled onto a strike ladder.

That matters because this is the surface whose ``_smile_section_impl``
hands the *original* section straight back when the requested time
matches a tenor. Engines that read a smile rather than a single
volatility — :class:`~pquantlib.pricingengines.basket.gaussian_copula_spread_engine.GaussianCopulaSpreadEngine`
is the one this port needed it for — see the parametric smile itself,
not a re-interpolation of it.

Time behaviour, in the three regions (``blackVarianceImpl``):

* ``t == 0``                → 0.
* ``t <= times[0]``         → ramp linearly from the origin:
  ``var(t) = var_0(K) * t / times[0]``.
* ``t >= times[-1]``        → flat *volatility* extrapolation, i.e.
  variance grows linearly: ``var(t) = var_{n-1}(K) * t / times[-1]``.
* in between                → linear in total variance between the two
  bracketing tenors.

Not ported: the ``makeFromGrid`` static factory, which builds one
``InterpolatedSmileSection<Linear>`` per column of a vol matrix. It is a
migration convenience for callers holding a ``BlackVarianceSurface``-shaped
grid, and this port has ``BlackVarianceSurface`` itself for that; adding it
would put an un-cross-validated code path in the library, which the project's
probe-first rule does not allow. The C++ probe covers the two constructors,
both interpolation regimes and the smile-section passthrough.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.closeness import close_enough
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date


class PiecewiseBlackVarianceSurface(BlackVarianceTermStructure):
    """Black-vol surface assembled from one smile section per tenor.

    # C++ parity: ``class PiecewiseBlackVarianceSurface``.

    Args:
        reference_date: the surface's reference date.
        dates: tenor dates, strictly increasing and all after
            ``reference_date``.
        smile_sections: one section per entry of ``dates``.
        day_counter: day counter used for ``time_from_reference``.
    """

    def __init__(
        self,
        *,
        reference_date: Date,
        dates: Sequence[Date],
        smile_sections: Sequence[SmileSection],
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            day_counter=day_counter,
        )
        qassert.require(len(dates) > 0, "at least one date is required")
        qassert.require(
            len(dates) == len(smile_sections),
            f"mismatch between {len(dates)} dates and "
            f"{len(smile_sections)} smile sections",
        )

        self._max_date: Date = dates[-1]
        times: list[float] = [self.time_from_reference(dates[0])]
        qassert.require(
            times[0] > 0.0,
            f"first date ({dates[0]}) must be after reference date "
            f"({reference_date})",
        )
        for i in range(1, len(dates)):
            times.append(self.time_from_reference(dates[i]))
            qassert.require(
                times[i] > times[i - 1],
                f"dates must be sorted and unique, but date {dates[i]} "
                f"(t={times[i]}) is not after date {dates[i - 1]} "
                f"(t={times[i - 1]})",
            )

        self._times: list[float] = times
        self._smile_sections: list[SmileSection] = list(smile_sections)
        # C++ also asserts each section is non-null before registering. A
        # ``Sequence[SmileSection]`` cannot carry a null here, so the check has
        # no analogue; the registration it guarded does.
        for section in self._smile_sections:
            section.register_with(self)

    @classmethod
    def single_tenor(
        cls,
        *,
        reference_date: Date,
        date: Date,
        smile_section: SmileSection,
        day_counter: DayCounter | None = None,
    ) -> PiecewiseBlackVarianceSurface:
        """One-tenor surface.

        # C++ parity: the second ``PiecewiseBlackVarianceSurface`` constructor,
        # which delegates to the vector one with singleton arguments. Python
        # has no constructor overloading, so it becomes a named factory.
        """
        return cls(
            reference_date=reference_date,
            dates=[date],
            smile_sections=[smile_section],
            day_counter=day_counter,
        )

    # --- inspectors --------------------------------------------------------

    def max_date(self) -> Date:
        return self._max_date

    def min_strike(self) -> float:
        # C++: QL_MIN_REAL, which is ``-max()`` rather than the smallest
        # positive normal.
        return -QL_MAX_REAL

    def max_strike(self) -> float:
        return QL_MAX_REAL

    # --- core --------------------------------------------------------------

    def _section_variance(self, i: int, strike: float) -> float:
        """Total variance of section ``i`` at ``strike``.

        # C++ parity: ``PiecewiseBlackVarianceSurface::sectionVariance``. The
        # strike range is the *section's* own, which is narrower than the
        # surface's unbounded one, so it is re-checked here.
        """
        s = self._smile_sections[i]
        qassert.require(
            self.allows_extrapolation()
            or (s.min_strike() <= strike <= s.max_strike()),
            f"strike ({strike}) is outside the range of smile section {i} "
            f"[{s.min_strike()}, {s.max_strike()}]",
        )
        return s.variance(strike)

    def _black_variance_impl(self, t: float, strike: float) -> float:
        # C++ parity: ``PiecewiseBlackVarianceSurface::blackVarianceImpl``.
        if t == 0.0:
            return 0.0

        if t <= self._times[0]:
            # linear interpolation from (0, 0) to the first tenor
            return self._section_variance(0, strike) * t / self._times[0]

        if t >= self._times[-1]:
            # flat vol extrapolation beyond the last tenor
            return (
                self._section_variance(len(self._smile_sections) - 1, strike)
                * t
                / self._times[-1]
            )

        # find the enclosing interval: hi is the first tenor strictly after t
        hi = next(i for i, ti in enumerate(self._times) if ti > t)
        lo = hi - 1
        var_lo = self._section_variance(lo, strike)
        var_hi = self._section_variance(hi, strike)
        alpha = (t - self._times[lo]) / (self._times[hi] - self._times[lo])
        return var_lo + (var_hi - var_lo) * alpha

    def _smile_section_impl(self, t: float) -> SmileSection:
        """Return the tenor's own section when ``t`` lands on one.

        # C++ parity: ``PiecewiseBlackVarianceSurface::smileSectionImpl``. The
        # match is ``close_enough``, not ``==``; off-tenor times fall back to
        # the base class's adapter, which reads through ``black_vol`` and so
        # sees the interpolated variance rather than any one section.
        """
        for i, ti in enumerate(self._times):
            if ti >= t:
                if close_enough(t, ti):
                    return self._smile_sections[i]
                break
        return super()._smile_section_impl(t)


__all__ = ["PiecewiseBlackVarianceSurface"]
