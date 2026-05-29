"""PiecewiseTimeDependentHestonModel — Heston with piecewise-constant params.

# C++ parity: ql/models/equity/piecewisetimedependenthestonmodel.{hpp,cpp}
# (v1.42.1).

A Heston-type stochastic-volatility model where ``theta``, ``kappa``,
``sigma`` and ``rho`` are general ``Parameter`` instances (typically
``PiecewiseConstantParameter`` over a finite tenor grid). ``v0`` and
``s0`` remain scalars.

Reference:
- Heston, S.L. 1993. A Closed-Form Solution for Options with
  Stochastic Volatility with Applications to Bond and Currency
  Options. Review of Financial Studies, 6, 327-343.
- Elices, A. Models with time-dependent parameters using transform
  methods: application to Heston's model. http://arxiv.org/pdf/0708.2020.

Divergences from C++:
- ``Handle<...>`` collapses to a direct reference.
- The model itself is a thin scaffold over ``CalibratedModel`` and a
  ``TimeGrid`` — it never builds its own ``HestonProcess`` (this is
  why the constructor doesn't take one). The companion
  ``AnalyticPiecewiseTimeDependentHestonEngine`` does the integration
  by reading ``theta(t)``, ``kappa(t)``, ``sigma(t)``, ``rho(t)`` on
  each grid segment.
- The C++ ``HestonModel`` parent of ``PiecewiseTimeDependentHestonModel``
  is **not** ``HestonModel`` — it's ``CalibratedModel`` (the C++
  diamond hierarchy is: PiecewiseTimeDependentHestonModel → CalibratedModel).
  We mirror that directly.
"""

from __future__ import annotations

from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.models.model import CalibratedModel
from pquantlib.models.parameter import ConstantParameter, Parameter
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.time_grid import TimeGrid


class PiecewiseTimeDependentHestonModel(CalibratedModel):
    """Heston model with piecewise-constant time-dependent params.

    # C++ parity: ``class PiecewiseTimeDependentHestonModel : public
    # CalibratedModel`` in piecewisetimedependenthestonmodel.hpp:44-79
    # (v1.42.1).

    Parameter layout matches C++ ``arguments_`` exactly:

    | Index | Name  |
    |-------|-------|
    | 0     | theta |
    | 1     | kappa |
    | 2     | sigma |
    | 3     | rho   |
    | 4     | v0    (ConstantParameter / PositiveConstraint) |
    """

    __slots__ = (
        "_dividend_yield",
        "_risk_free_rate",
        "_s0",
        "_time_grid",
    )

    _N_ARGUMENTS: int = 5

    def __init__(
        self,
        *,
        risk_free_rate: YieldTermStructure,
        dividend_yield: YieldTermStructure,
        s0: Quote,
        v0: float,
        theta: Parameter,
        kappa: Parameter,
        sigma: Parameter,
        rho: Parameter,
        time_grid: TimeGrid,
    ) -> None:
        # C++ parity: piecewisetimedependenthestonmodel.cpp:36-48.
        super().__init__(self._N_ARGUMENTS)
        self._s0: Quote = s0
        self._risk_free_rate: YieldTermStructure = risk_free_rate
        self._dividend_yield: YieldTermStructure = dividend_yield
        self._time_grid: TimeGrid = time_grid

        self._arguments[0] = theta
        self._arguments[1] = kappa
        self._arguments[2] = sigma
        self._arguments[3] = rho
        self._arguments[4] = ConstantParameter(v0, PositiveConstraint())

        # C++ parity: piecewisetimedependenthestonmodel.cpp:45-47 —
        # register observers on s0 + the two yield curves.
        s0.register_with(self)
        risk_free_rate.register_with(self)
        dividend_yield.register_with(self)

    # --- piecewise inspectors -------------------------------------------

    def theta(self, t: float) -> float:
        """Long-term variance level at time ``t``.

        # C++ parity: piecewisetimedependenthestonmodel.hpp:57.
        """
        return self._arguments[0](t)

    def kappa(self, t: float) -> float:
        """Mean-reversion speed at time ``t``.

        # C++ parity: piecewisetimedependenthestonmodel.hpp:59.
        """
        return self._arguments[1](t)

    def sigma(self, t: float) -> float:
        """Vol-of-variance at time ``t``.

        # C++ parity: piecewisetimedependenthestonmodel.hpp:61.
        """
        return self._arguments[2](t)

    def rho(self, t: float) -> float:
        """Correlation at time ``t``.

        # C++ parity: piecewisetimedependenthestonmodel.hpp:63.
        """
        return self._arguments[3](t)

    def v0(self) -> float:
        """Spot variance (constant in time).

        # C++ parity: piecewisetimedependenthestonmodel.hpp:65.
        """
        return self._arguments[4](0.0)

    def s0(self) -> float:
        """Spot value.

        # C++ parity: piecewisetimedependenthestonmodel.hpp:67.
        """
        return self._s0.value()

    # --- curve / grid accessors -----------------------------------------

    def time_grid(self) -> TimeGrid:
        """The time grid for the piecewise segments.

        # C++ parity: piecewisetimedependenthestonmodel.hpp:70 +
        # piecewisetimedependenthestonmodel.cpp:50-52.
        """
        return self._time_grid

    def risk_free_rate(self) -> YieldTermStructure:
        """Risk-free yield curve.

        # C++ parity: piecewisetimedependenthestonmodel.hpp:72 +
        # piecewisetimedependenthestonmodel.cpp:59-61.
        """
        return self._risk_free_rate

    def dividend_yield(self) -> YieldTermStructure:
        """Dividend-yield curve.

        # C++ parity: piecewisetimedependenthestonmodel.hpp:71 +
        # piecewisetimedependenthestonmodel.cpp:54-56.
        """
        return self._dividend_yield


__all__ = ["PiecewiseTimeDependentHestonModel"]
