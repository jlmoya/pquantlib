"""PiecewiseSpreadYieldCurve — bootstrapped multiplicative spread over a base curve.

# C++ parity: ql/termstructures/yield/piecewisespreadyieldcurve.hpp (v1.43)
#   template <class Traits, class Interpolator,
#             template <class> class Bootstrap = IterativeBootstrap>
#   class PiecewiseSpreadYieldCurve
#       : public PiecewiseYieldCurve<detail::SpreadTraits<Traits>,
#                                    Interpolator, Bootstrap>;

The bootstrap state is the *spread* discount factor, not the outright
one: the curve's discount factor is ``base.discount(t) * spread(t)``, and
the helpers are repriced through that product. Everything else — the
guesses, the brackets, the transforms — is inherited verbatim from
``Discount``; the only thing ``detail::SpreadTraits<Discount>`` changes is
which interpolated curve type sits underneath
(``InterpolatedSpreadDiscountCurve`` instead of
``InterpolatedDiscountCurve``). See spreadbootstraptraits.hpp:13-22.

# C++ parity divergence (inherited from :class:`PiecewiseYieldCurve`):
   the C++ class is templated on the ``Bootstrap`` policy; the Python
   port hard-codes ``IterativeBootstrap``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.log_linear import LogLinearInterpolation
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.yield_.interpolated_spread_discount_curve import (
    InterpolatedSpreadDiscountCurve,
)
from pquantlib.termstructures.yield_.piecewise_yield_curve import (
    InterpolationFactory,
    PiecewiseYieldCurve,
)
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class SpreadTraits(Discount):
    """Spread discount-curve traits.

    # C++ parity: ``detail::SpreadTraits<Discount>``
    # (spreadbootstraptraits.hpp:16-22). Inherits every member from
    # ``Discount`` — initial value, guesses, brackets, transforms,
    # ``updateGuess``, ``maxIterations`` — and only substitutes the curve
    # type. The subclassing here mirrors the C++ ``: Discount`` exactly.
    """


class PiecewiseSpreadYieldCurve(PiecewiseYieldCurve):
    """Piecewise-bootstrapped multiplicative spread over a base yield curve.

    Args:
        base_curve: the curve the bootstrapped spread multiplies.
        instruments: ``BootstrapHelper[YieldTermStructure]`` list.
        interpolator: factory ``(xs, ys) -> Interpolation`` over the
            spread discount factors. Defaults to log-linear, matching the
            ``SpreadDiscountCurve`` typedef (spreaddiscountcurve.hpp:75)
            — log-linear in the spread means a piecewise-constant spread
            forward.
        accuracy: Brent inner tolerance. Default ``1e-12``.

    Day counter, calendar and reference date are the base curve's, as in
    C++ (spreaddiscountcurve.hpp:117-137).
    """

    def __init__(
        self,
        base_curve: YieldTermStructure,
        instruments: Sequence[BootstrapHelper[Any]],
        interpolator: InterpolationFactory | None = None,
        accuracy: float = 1.0e-12,
    ) -> None:
        # C++ ``SpreadDiscountCurve`` = InterpolatedSpreadDiscountCurve<LogLinear>.
        if interpolator is None:
            interpolator = LogLinearInterpolation
        # The base curve must be readable before super().__init__ returns,
        # because the bootstrap is lazy and only touches it later.
        self._base: YieldTermStructure = base_curve
        super().__init__(
            traits=SpreadTraits,
            reference_date=base_curve.reference_date(),
            instruments=instruments,
            day_counter=base_curve.day_counter(),
            calendar=None,
            interpolator=interpolator,
            accuracy=accuracy,
        )

    # ---- inspectors forwarded to the base curve ---------------------------
    # C++ parity: spreaddiscountcurve.hpp:117-137 — the spread curve has no
    # day counter / calendar / reference date of its own.

    def base_curve(self) -> YieldTermStructure:
        """# C++ parity: ``InterpolatedSpreadDiscountCurve::baseCurve``."""
        return self._base

    def day_counter(self) -> DayCounter:
        return self._base.day_counter()

    def calendar(self) -> Calendar:
        return self._base.calendar()

    def reference_date(self) -> Date:
        return self._base.reference_date()

    def max_date(self) -> Date:
        """# C++ parity: spreaddiscountcurve.hpp:139-143.

        ``min(baseCurve->maxDate(), maxDate_ ? maxDate_ : dates_.back())``.
        """
        return min(self._base.max_date(), super().max_date())

    # ---- traits -> underlying curve plumbing ------------------------------

    def _underlying_class(self) -> type:
        # C++ parity: ``SpreadTraits<Discount>::curve<Interpolator>::type``
        # (spreadbootstraptraits.hpp:18-21).
        return InterpolatedSpreadDiscountCurve

    def _underlying_data_kwarg(self) -> str:
        return "dfs"

    def _seed_data(self, n: int) -> list[float]:
        """Seed spreads that pass ``InterpolatedSpreadDiscountCurve``'s checks.

        The constructor requires ``dfs[0] == 1.0`` and ``dfs[i] > 0``
        (spreaddiscountcurve.hpp:93-100); the real values are installed
        immediately afterwards by :meth:`bootstrap_install_grid`.
        """
        return [1.0] + [1.0 / (1.0 + 0.05 * i) for i in range(1, n)]

    def bootstrap_install_grid(
        self, dates: list[Date], times: list[float], data: list[float]
    ) -> None:
        """Allocate the underlying spread curve from the bootstrap grid.

        Overridden because ``InterpolatedSpreadDiscountCurve`` takes the
        base curve and no day counter / calendar / jumps — its own
        accessors delegate to the base.
        """
        self._underlying = InterpolatedSpreadDiscountCurve(  # type: ignore[assignment]
            base_curve=self._base,
            dates=list(dates),
            dfs=self._seed_data(len(dates)),
            interpolator=self._interpolator,
        )
        u: Any = self._underlying
        u._dates = list(dates)
        u._times = list(times)
        u._data = list(data)
        u._interpolation = self._interpolator(
            np.asarray(times, dtype=np.float64),
            np.asarray(data, dtype=np.float64),
        )


__all__ = ["PiecewiseSpreadYieldCurve", "SpreadTraits"]
