"""PiecewiseYieldCurve — bootstrap-from-helpers yield curve.

# C++ parity: ql/termstructures/yield/piecewiseyieldcurve.hpp (v1.42.1)

C++ ``PiecewiseYieldCurve<Traits, Interpolator, Bootstrap>`` is the
production-grade piecewise-bootstrap yield curve. It chains
``InterpolatedDiscount/Zero/ForwardCurve<Interpolator>`` with
``IterativeBootstrap<Curve>``. The traits class
(``Discount`` / ``ZeroYield`` / ``ForwardRate``) selects which curve
quantity is the bootstrap state — and therefore which
``InterpolatedXxxCurve`` is the underlying interpolation engine.

In Python we take ``traits`` as either a class (auto-instantiated) or
an instance, and pick the matching ``InterpolatedXxxCurve`` at
construction time. The bootstrap is lazy: it runs on the first call to
``discount`` / ``zero_rate`` / ``forward_rate``.

The class subclasses ``YieldTermStructure`` directly and forwards
``discount_impl`` to the underlying interpolated curve. The
:class:`BootstrapCurveProtocol` surface (``data_live``, ``set_data_at``,
``refresh_interpolation_through``, ``bootstrap_install_grid``) is
implemented by mutating the underlying interpolated curve's internal
``_dates / _times / _data`` arrays directly — this is the same
``IterativeBootstrap``-protocol contract used by L7's piecewise
inflation curves and now by the wired-up :class:`PiecewiseDefaultCurve`.

# C++ parity divergence: the C++ class supports arbitrary
   ``Bootstrap`` template parameters (``IterativeBootstrap`` vs
   ``LocalBootstrap``). The Python port hard-codes
   ``IterativeBootstrap``; ``LocalBootstrap`` is carved out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.interpolations.log_linear import LogLinearInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap.iterative_bootstrap import IterativeBootstrap
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.yield_.interpolated_discount_curve import (
    InterpolatedDiscountCurve,
)
from pquantlib.termstructures.yield_.interpolated_forward_curve import (
    InterpolatedForwardCurve,
)
from pquantlib.termstructures.yield_.interpolated_zero_curve import (
    InterpolatedZeroCurve,
)
from pquantlib.termstructures.yield_.yield_traits import (
    Discount,
    ForwardRate,
    ZeroYield,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


def _instantiate_traits(traits: Any) -> Any:
    """Accept either a class or an instance for ``traits=``.

    Mirrors how C++ template parameters are passed as *types*; the
    Python port accepts either ``Discount`` (class) or ``Discount()``
    (instance).
    """
    if isinstance(traits, type):
        return traits()
    return traits


class PiecewiseYieldCurve(YieldTermStructure):
    """Piecewise-bootstrapped yield term structure.

    Inputs:
    - ``traits``: ``Discount`` / ``ZeroYield`` / ``ForwardRate``
      (class or instance) — picks the underlying interpolated curve
      type and the bootstrap state variable.
    - ``reference_date``: curve reference date (pillar 0).
    - ``instruments``: list of ``BootstrapHelper[YieldTermStructure]``.
    - ``day_counter``: day counter for date → time conversion.
    - ``interpolator``: factory ``(xs, ys) → Interpolation``. Default
      :class:`LinearInterpolation` for ``ZeroYield`` / ``ForwardRate``
      and :class:`LogLinearInterpolation` for ``Discount``. Pass an
      explicit factory to override (e.g. a CubicNaturalSpline from
      L9-A).
    - ``accuracy``: Brent inner tolerance. Default ``1e-12``.
    - ``jumps`` / ``jump_dates``: forwarded to the underlying
      interpolated curve.

    The bootstrap is lazy: it triggers on the first ``discount`` /
    ``zero_rate`` / ``forward_rate`` evaluation.
    """

    def __init__(
        self,
        traits: Any,
        reference_date: Date,
        instruments: Sequence[BootstrapHelper[Any]],
        day_counter: DayCounter,
        calendar: Calendar | None = None,
        interpolator: InterpolationFactory | None = None,
        accuracy: float = 1.0e-12,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
    ) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            jumps=jumps,
            jump_dates=jump_dates,
        )
        qassert.require(
            len(instruments) >= 1,
            "PiecewiseYieldCurve: at least one instrument is required",
        )
        self._traits: Any = _instantiate_traits(traits)
        self._traits_class: type = traits if isinstance(traits, type) else type(traits)
        self._instruments: list[BootstrapHelper[Any]] = list(instruments)
        # TermStructure already stores day_counter; keep a typed copy
        # here for direct access during ``bootstrap_install_grid``.
        self._dc_stored: DayCounter = day_counter
        self._accuracy: float = accuracy
        self._jumps_in: list[Quote] | None = jumps
        self._jump_dates_in: list[Date] | None = jump_dates
        self._calendar_in: Calendar | None = calendar
        # Default interpolator depends on the traits.
        if interpolator is None:
            interpolator = (
                LogLinearInterpolation
                if self._traits_class is Discount
                else LinearInterpolation
            )
        self._interpolator: InterpolationFactory = interpolator
        # Lazy underlying — populated on first ``calculate``.
        self._underlying: (
            InterpolatedDiscountCurve
            | InterpolatedZeroCurve
            | InterpolatedForwardCurve
            | None
        ) = None
        self._bootstrap_done: bool = False

    # ---- BootstrapCurveProtocol ------------------------------------------

    def base_date(self) -> Date:
        # The earliest date on the bootstrap grid is the reference date.
        return self.reference_date()

    def times(self) -> list[float]:
        if self._underlying is None:
            return []
        return list(self._underlying.times())

    def data_live(self) -> list[float]:
        """Live (non-defensive) reference to the underlying _data array.

        Required by IterativeBootstrap so trait ``update_guess`` writes
        through to the curve. # C++ parity:
        ``InterpolatedCurve::data_``.
        """
        u: Any = self._underlying
        assert u is not None
        return u._data

    def set_data_at(self, i: int, level: float) -> None:
        u: Any = self._underlying
        assert u is not None
        u._data[i] = level

    def refresh_interpolation_through(self, up_to: int) -> None:
        """Rebuild the interpolation over the first ``up_to + 1`` nodes.

        # C++ parity: ``InterpolatedCurve::setupInterpolation``.
        """
        u: Any = self._underlying
        assert u is not None
        partial_t = u._times[: up_to + 1]
        partial_d = u._data[: up_to + 1]
        u._interpolation = self._interpolator(
            np.asarray(partial_t, dtype=np.float64),
            np.asarray(partial_d, dtype=np.float64),
        )
        # Notify observers so downstream lazy instruments invalidate
        # their NPV cache (helper swaps wired to this curve).
        self.notify_observers()

    def bootstrap_install_grid(
        self, dates: list[Date], times: list[float], data: list[float],
    ) -> None:
        """Allocate the underlying interpolated curve from a grid.

        Called once by ``IterativeBootstrap`` after sorting helpers.
        """
        # Construct the underlying with dummy positive entries we'll then
        # overwrite — the InterpolatedXxx constructors do their own
        # range checks (e.g. discounts must be > 0; survival probs must
        # be monotonic), so we bypass them by instantiating then
        # re-installing the arrays.
        underlying_cls = self._underlying_class()
        seed_data = self._seed_data(len(dates))
        self._underlying = underlying_cls(
            dates=list(dates), **{self._underlying_data_kwarg(): seed_data},
            day_counter=self._dc_stored,
            calendar=self._calendar_in,
            jumps=self._jumps_in,
            jump_dates=self._jump_dates_in,
            interpolator=self._interpolator,
        )
        # Overwrite arrays in place via the same Any cast used by the
        # other protocol shims (bootstrap-internal mutation of curve
        # state — pyright reports `reportPrivateUsage`).
        u: Any = self._underlying
        u._dates = list(dates)
        u._times = list(times)
        u._data = list(data)
        u._interpolation = self._interpolator(
            np.asarray(times, dtype=np.float64),
            np.asarray(data, dtype=np.float64),
        )

    # ---- traits → underlying class plumbing -------------------------------

    def _underlying_class(self) -> type:
        if self._traits_class is Discount:
            return InterpolatedDiscountCurve
        if self._traits_class is ZeroYield:
            return InterpolatedZeroCurve
        if self._traits_class is ForwardRate:
            return InterpolatedForwardCurve
        qassert.fail(
            f"PiecewiseYieldCurve: unsupported traits {self._traits_class}",
        )
        return InterpolatedDiscountCurve  # unreachable

    def _underlying_data_kwarg(self) -> str:
        if self._traits_class is Discount:
            return "dfs"
        if self._traits_class is ZeroYield:
            return "yields"
        if self._traits_class is ForwardRate:
            return "forwards"
        qassert.fail(
            f"PiecewiseYieldCurve: unsupported traits {self._traits_class}",
        )
        return "data"

    def _seed_data(self, n: int) -> list[float]:
        """Seed arrays that pass the underlying constructor validation.

        Discount factors must satisfy data[0]==1.0 and data[i]>0; we
        use a monotonically-decreasing sequence. Rates accept any
        positive sequence.
        """
        if self._traits_class is Discount:
            return [1.0] + [1.0 / (1.0 + 0.05 * i) for i in range(1, n)]
        return [0.05] * n

    # ---- bootstrap trigger -----------------------------------------------

    def _ensure_bootstrap(self) -> None:
        # If the underlying is already allocated (we may be re-entrant
        # mid-bootstrap from a helper engine call), bail out — the
        # IterativeBootstrap is using the live interpolation as it
        # solves, so a partial-progress query is meaningful.
        if self._bootstrap_done or self._underlying is not None:
            return
        bootstrapper = IterativeBootstrap(
            curve=self, instruments=self._instruments,
            traits=self._traits, accuracy=self._accuracy,
        )
        bootstrapper.calculate()
        self._bootstrap_done = True

    # ---- YieldTermStructure interface ------------------------------------

    def max_date(self) -> Date:
        # Use the latest helper's pillar as the max date — same as
        # base scaffold.
        u: Any = self._underlying
        if u is not None and u._dates:
            return u._dates[-1]
        return max(h.pillar_date() for h in self._instruments)

    def _discount_impl(self, t: float) -> float:
        # Delegate to the underlying interpolated curve's
        # ``_discount_impl`` so YieldTermStructure's jump-handling on
        # ``self`` isn't double-applied (we already forwarded jumps to
        # the underlying at construction).
        self._ensure_bootstrap()
        u: Any = self._underlying
        assert u is not None
        return u._discount_impl(t)

    # ---- inspectors -------------------------------------------------------

    def instruments(self) -> list[BootstrapHelper[Any]]:
        return list(self._instruments)

    def traits(self) -> type:
        return self._traits_class

    def dates(self) -> list[Date]:
        self._ensure_bootstrap()
        u: Any = self._underlying
        assert u is not None
        return list(u._dates)

    def data(self) -> list[float]:
        self._ensure_bootstrap()
        u: Any = self._underlying
        assert u is not None
        return list(u._data)

    def nodes(self) -> list[tuple[Date, float]]:
        self._ensure_bootstrap()
        return list(zip(self.dates(), self.data(), strict=True))


__all__ = ["PiecewiseYieldCurve"]
