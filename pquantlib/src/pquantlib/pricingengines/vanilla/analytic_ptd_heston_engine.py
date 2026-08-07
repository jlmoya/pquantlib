"""AnalyticPTDHestonEngine — analytic piecewise-time-dependent Heston engine.

# C++ parity: ql/pricingengines/vanilla/analyticptdhestonengine.{hpp,cpp}
# (v1.43) — ``class AnalyticPTDHestonEngine : public
# GenericModelEngine<PiecewiseTimeDependentHestonModel,
# VanillaOption::arguments, VanillaOption::results>``.

Generalises :class:`~pquantlib.pricingengines.vanilla.analytic_heston_engine.AnalyticHestonEngine`
to piecewise-constant ``kappa(t)``, ``theta(t)``, ``sigma(t)``, ``rho(t)``. The
characteristic function is assembled by walking the model's ``TimeGrid``
*backwards* from maturity, accumulating the Riccati pair ``(D, C)`` with each
segment's own parameters, sampled at the segment MIDPOINT ``0.5*(begin + end)``.

Two complex-log formulations, exactly as in C++:

* ``Gatheral`` — the classic two-integral form::

      Call = spot*dd*(p1 + 0.5) - strike*dr*(p2 + 0.5)
      Put  = spot*dd*(p1 - 0.5) - strike*dr*(p2 - 0.5)

  with ``p_j = (1/pi) * Integral_0^inf Fj(phi) dphi``.

* ``AndersenPiterbarg`` — a single integral of the difference between the
  model chF and a Black control variate whose price is closed-form::

      Call = bsPrice + h_cv
      Put  = bsPrice + h_cv - dr*(fwd - strike)

References:

- Heston, S. L., 1993. *A Closed-Form Solution for Options with Stochastic
  Volatility with Applications to Bond and Currency Options.* Review of
  Financial Studies 6(2), 327-343.
- J. Gatheral, *The Volatility Surface: A Practitioner's Guide*, Wiley Finance.
- A. Elices, *Models with time-dependent parameters using transform methods:
  application to Heston's model*, http://arxiv.org/pdf/0708.2020.

Three traps this port has to keep
---------------------------------
1. ``AnalyticPTDHestonEngine::ComplexLogFormula`` is its OWN two-valued enum
   ``{Gatheral, AndersenPiterbarg}``, **not** ``AnalyticHestonEngine``'s
   eight-valued one. Only ``Integration`` is a typedef of the latter's.
2. ``Fj_Helper::operator()`` clamps ``phi`` to ``numeric_limits<float>::epsilon()``
   — FLOAT epsilon, 1.1920928955078125e-07, not double epsilon. That clamp is
   the value of the integrand at the left endpoint of every Gauss-Laguerre and
   Gauss-Lobatto rule, so using ``DBL_EPSILON`` there changes every price.
3. Spot, rates and dividends come off the **model**
   (``model_->s0()``, ``model_->riskFreeRate()``), not off a process, and the
   maturity is ``riskFreeRate()->dayCounter().yearFraction(referenceDate,
   lastDate)`` — not ``process->time()``.
"""

from __future__ import annotations

import bisect
import cmath
import math
from enum import IntEnum

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.closeness import close_enough
from pquantlib.models.equity.piecewise_time_dependent_heston_model import (
    PiecewiseTimeDependentHestonModel,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_heston_engine import Integration
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

#: ``std::numeric_limits<float>::epsilon()`` as a double — the clamp
#: ``Fj_Helper::operator()`` applies to ``phi`` (analyticptdhestonengine.cpp:80).
#: Deliberately NOT ``QL_EPSILON``: C++ names the *float* epsilon here.
_FLOAT_EPSILON: float = 1.1920928955078125e-07

#: C++ ``Null<Real>()`` == ``std::numeric_limits<float>::max()``
#: (ql/utilities/null.hpp). The first two constructors leave
#: ``andersenPiterbargEpsilon_`` at this sentinel; the Gatheral branch never
#: reads it, so that is a usable state, not a "must be set later" marker.
_NULL_REAL: float = 3.4028234663852886e38


class ComplexLogFormula(IntEnum):
    """Which complex-log formulation the engine uses.

    # C++ parity: ``enum ComplexLogFormula { Gatheral, AndersenPiterbarg }``
    # nested in ``AnalyticPTDHestonEngine`` (analyticptdhestonengine.hpp:57).
    # This is a DIFFERENT enum from ``AnalyticHestonEngine``'s eight-valued
    # one; the PTD engine understands exactly these two.
    """

    #: Gatheral form of the characteristic function, no control variate.
    Gatheral = 0
    #: Andersen-Piterbarg control variate around a Black price.
    AndersenPiterbarg = 1


class _FjHelper:
    """Gatheral-form integrand for ``j = 1`` or ``j = 2``.

    # C++ parity: ``class AnalyticPTDHestonEngine::Fj_Helper``
    # (analyticptdhestonengine.cpp:32-120) — private in C++, private here.
    """

    __slots__ = ("_divs", "_j", "_log_spot", "_log_strike", "_model", "_rates", "_term", "_v0")

    def __init__(
        self,
        model: PiecewiseTimeDependentHestonModel,
        term: float,
        strike: float,
        j: int,
    ) -> None:
        """# C++ parity: ``Fj_Helper::Fj_Helper`` (.cpp:52-74).

        The per-segment forward rates are computed ONCE here, over the whole
        grid, with both endpoints clamped to ``term``. When both clamp to the
        same value ``forwardRate`` takes its ``t2 == t1`` branch and returns the
        instantaneous forward around a ``dt = 1e-4`` window; that degenerate
        value is then multiplied by ``tau = 0`` in ``operator()`` and cannot
        affect the answer, but it must not raise.
        """
        self._j: int = j
        self._term: float = term
        self._v0: float = model.v0()
        self._log_spot: float = math.log(model.s0())
        self._log_strike: float = math.log(strike)
        self._model: PiecewiseTimeDependentHestonModel = model

        time_grid = model.time_grid()
        risk_free = model.risk_free_rate()
        dividend = model.dividend_yield()
        rates: list[float] = []
        divs: list[float] = []
        for i in range(len(time_grid) - 1):
            begin = min(term, time_grid[i])
            end = min(term, time_grid[i + 1])
            rates.append(
                risk_free.forward_rate(
                    begin, end, Compounding.Continuous, Frequency.NoFrequency
                ).rate()
            )
            divs.append(
                dividend.forward_rate(
                    begin, end, Compounding.Continuous, Frequency.NoFrequency
                ).rate()
            )
        self._rates: list[float] = rates
        self._divs: list[float] = divs

    def __call__(self, phi_in: float) -> float:
        """# C++ parity: ``Fj_Helper::operator()`` (.cpp:76-120)."""
        # C++ parity: .cpp:80 — avoid numeric overflow for phi -> 0.
        phi = max(_FLOAT_EPSILON, phi_in)

        big_d: complex = 0j
        big_c: complex = 0j
        model = self._model
        time_grid = model.time_grid()
        term = self._term
        j = self._j

        for i in range(len(time_grid) - 1, 0, -1):
            begin = time_grid[i - 1]
            if begin >= term:
                continue
            end = min(term, time_grid[i])
            tau = end - begin
            t_mid = 0.5 * (end + begin)

            rho = model.rho(t_mid)
            sigma = model.sigma(t_mid)
            kappa = model.kappa(t_mid)
            theta = model.theta(t_mid)

            sigma2 = sigma * sigma
            t0 = kappa - (rho * sigma if j == 1 else 0.0)
            rpsig = rho * sigma * phi

            t1 = t0 + complex(0.0, -rpsig)
            d = cmath.sqrt(t1 * t1 - sigma2 * phi * complex(-phi, 1.0 if j == 1 else -1.0))
            g = (t1 - d) / (t1 + d)
            gt = (t1 - d - big_d * sigma2) / (t1 + d - big_d * sigma2)
            ex = cmath.exp(-d * tau)

            big_d = (t1 + d) / sigma2 * (g - gt * ex) / (1.0 - gt * ex)

            lng = cmath.log((1.0 - gt * ex) / (1.0 - gt))

            big_c = (
                (kappa * theta) / sigma2 * ((t1 - d) * tau - 2.0 * lng)
                + complex(0.0, phi * (self._rates[i - 1] - self._divs[i - 1]) * tau)
                + big_c
            )

        return (
            cmath.exp(
                self._v0 * big_d
                + big_c
                + complex(0.0, phi * (self._log_spot - self._log_strike))
            ).imag
            / phi
        )


class _APHelper:
    """Integrand for the Andersen-Piterbarg control variate.

    # C++ parity: ``class AnalyticPTDHestonEngine::AP_Helper``
    # (analyticptdhestonengine.cpp:122-152). Note this is NOT
    # ``AnalyticHestonEngine::AP_Helper``: it is a much smaller class with a
    # fixed ``alpha = -0.5`` contour and no angled-contour variants.
    """

    __slots__ = ("_dd", "_engine", "_log_strike", "_sigma_bs", "_term")

    def __init__(
        self,
        term: float,
        s0: float,
        strike: float,
        ratio: float,
        sigma_bs: float,
        engine: AnalyticPTDHestonEngine,
    ) -> None:
        """# C++ parity: ``AP_Helper::AP_Helper`` (.cpp:124-134).

        C++ asserts ``enginePtr != nullptr``; the Python signature types
        ``engine`` as non-optional, so the annotation is the check.
        """
        self._term: float = term
        self._sigma_bs: float = sigma_bs
        self._log_strike: float = math.log(strike)
        self._dd: float = math.log(s0) - math.log(ratio)
        self._engine: AnalyticPTDHestonEngine = engine

    def __call__(self, u: float) -> float:
        """# C++ parity: ``AP_Helper::operator()`` (.cpp:136-145)."""
        z = complex(u, -0.5)
        phi_bs = cmath.exp(
            -0.5
            * self._sigma_bs
            * self._sigma_bs
            * self._term
            * (z * z + complex(-z.imag, z.real))
        )
        return (
            cmath.exp(complex(0.0, u * (self._dd - self._log_strike)))
            * (phi_bs - self._engine.ch_f(z, self._term))
            / (u * u + 0.25)
        ).real


class AnalyticPTDHestonEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """European-vanilla engine for the piecewise-time-dependent Heston model.

    # C++ parity: ``class AnalyticPTDHestonEngine`` in
    # ql/pricingengines/vanilla/analyticptdhestonengine.hpp:52-98 (v1.43).
    """

    # C++ spelling: AnalyticPTDHestonEngine::Integration is a typedef of
    # AnalyticHestonEngine::Integration (hpp:58); ::Gatheral / ::AndersenPiterbarg
    # name the engine's own enum (hpp:57).
    Integration = Integration
    Gatheral = ComplexLogFormula.Gatheral
    AndersenPiterbarg = ComplexLogFormula.AndersenPiterbarg

    def __init__(
        self,
        model: PiecewiseTimeDependentHestonModel,
        integration_order: int = 144,
    ) -> None:
        """Gauss-Laguerre quadrature of ``integration_order``, Gatheral log.

        # C++ parity: ``AnalyticPTDHestonEngine(model, integrationOrder = 144)``
        # (analyticptdhestonengine.cpp:211-222) — cpxLog = Gatheral,
        # andersenPiterbargEpsilon = Null<Real>().
        """
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._model: PiecewiseTimeDependentHestonModel = model
        self._evaluations: int = 0
        self._cpx_log: ComplexLogFormula = ComplexLogFormula.Gatheral
        self._integration: Integration = Integration.gauss_laguerre(integration_order)
        self._andersen_piterbarg_epsilon: float = _NULL_REAL
        # C++ parity: GenericModelEngine's ctor registers with the model.
        model.register_with(self)

    @classmethod
    def with_lobatto(
        cls,
        model: PiecewiseTimeDependentHestonModel,
        rel_tolerance: float,
        max_evaluations: int,
    ) -> AnalyticPTDHestonEngine:
        """Adaptive Gauss-Lobatto quadrature, Gatheral log.

        # C++ parity: ``AnalyticPTDHestonEngine(model, relTolerance,
        # maxEvaluations)`` (.cpp:224-235). C++ warns that a large
        # ``maxEvaluations`` can overflow the stack, because Lobatto recurses.
        """
        engine = cls(model)
        engine._integration = Integration.gauss_lobatto(rel_tolerance, None, max_evaluations)
        return engine

    @classmethod
    def with_integration(
        cls,
        model: PiecewiseTimeDependentHestonModel,
        cpx_log: ComplexLogFormula,
        integration: Integration,
        andersen_piterbarg_epsilon: float = 1e-8,
    ) -> AnalyticPTDHestonEngine:
        """Full control over the Fourier integration algorithm.

        # C++ parity: ``AnalyticPTDHestonEngine(model, cpxLog, itg,
        # andersenPiterbargEpsilon = 1e-8)`` (.cpp:237-249). Note the default
        # epsilon is 1e-8 here and 1e-25 in ``AnalyticHestonEngine``.
        """
        engine = cls(model)
        engine._cpx_log = cpx_log
        engine._integration = integration
        engine._andersen_piterbarg_epsilon = andersen_piterbarg_epsilon
        return engine

    # --- inspectors -------------------------------------------------------

    def model(self) -> PiecewiseTimeDependentHestonModel:
        """The underlying model."""
        return self._model

    def number_of_evaluations(self) -> int:
        """# C++ parity: ``numberOfEvaluations`` (.cpp:408-410)."""
        return self._evaluations

    # --- characteristic function ------------------------------------------

    def ln_ch_f(self, z: complex, t: float) -> complex:
        """Log of the normalised characteristic function.

        # C++ parity: ``AnalyticPTDHestonEngine::lnChF`` (.cpp:155-204).

        Walks the grid backwards from the segment containing ``t``. Note the
        loop bound is ``lower_bound(timeGrid, T)``, i.e. the first grid point
        at or beyond ``T`` — so a ``T`` landing exactly on a grid point does
        NOT walk the segment starting there.
        """
        v0 = self._model.v0()
        big_d: complex = 0j
        big_c: complex = 0j

        time_grid = self._model.time_grid()
        last_model_time = time_grid.back()
        qassert.require(
            t <= last_model_time,
            f"maturity ({t}) is too large, time grid is bounded by {last_model_time}",
        )

        # C++ parity: .cpp:170-171 — std::lower_bound over the grid times.
        last_i = bisect.bisect_left(time_grid.times, t)

        for i in range(last_i - 1, -1, -1):
            begin = time_grid[i]
            end = min(t, time_grid[i + 1])
            tau = end - begin
            t_mid = 0.5 * (end + begin)

            kappa = self._model.kappa(t_mid)
            sigma = self._model.sigma(t_mid)
            theta = self._model.theta(t_mid)
            rho = self._model.rho(t_mid)

            sigma2 = sigma * sigma

            k = kappa + rho * sigma * complex(z.imag, -z.real)
            d = cmath.sqrt(k * k + (z * z + complex(-z.imag, z.real)) * sigma2)
            g = (k - d) / (k + d)
            gt = (k - d - big_d * sigma2) / (k + d - big_d * sigma2)

            big_c += (
                kappa
                * theta
                / sigma2
                * ((k - d) * tau - 2.0 * cmath.log((1.0 - gt * cmath.exp(-d * tau)) / (1.0 - gt)))
            )

            big_d = (
                (k + d)
                / sigma2
                * (g - gt * cmath.exp(-d * tau))
                / (1.0 - gt * cmath.exp(-d * tau))
            )

        return big_d * v0 + big_c

    def ch_f(self, z: complex, t: float) -> complex:
        """Normalised characteristic function.

        # C++ parity: ``AnalyticPTDHestonEngine::chF`` (.cpp:206-209) — plain
        # ``exp(lnChF(z, t))``; unlike ``AnalyticHestonEngine::chF`` there is
        # no small-sigma expansion branch here.
        """
        return cmath.exp(self.ln_ch_f(z, t))

    # --- pricing ----------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915  (mirrors a 155-line C++ body)
        """Fill ``results.value`` for a European vanilla.

        # C++ parity: ``AnalyticPTDHestonEngine::calculate`` (.cpp:252-406).
        """
        args = self._arguments
        results = self._results
        model = self._model

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        # C++ parity: .cpp:254-255.
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not an European option"
        )
        # C++ parity: .cpp:258-260 — dynamic_pointer_cast<PlainVanillaPayoff>.
        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        v0 = model.v0()
        spot_price = model.s0()
        # C++ parity: .cpp:264.
        qassert.require(spot_price > 0.0, "negative or null underlying given")

        strike = payoff.strike()
        risk_free = model.risk_free_rate()
        # C++ parity: .cpp:267-270 — the day counter of the RISK-FREE curve.
        term = risk_free.day_counter().year_fraction(
            risk_free.reference_date(), args.exercise.last_date()
        )

        time_grid = model.time_grid()
        # C++ parity: .cpp:272-275 — close_enough, not an ad-hoc tolerance.
        qassert.require(
            term < time_grid.back() or close_enough(term, time_grid.back()),
            f"maturity ({term}) is too large, time grid is bounded by {time_grid.back()}",
        )

        risk_free_discount = risk_free.discount(args.exercise.last_date())
        dividend_discount = model.dividend_yield().discount(args.exercise.last_date())

        # C++ parity: .cpp:284 — checked AFTER the maturity guard.
        qassert.require(len(time_grid) > 1, "at least two model points needed")

        # C++ parity: .cpp:286-296 — average parameters over the grid, sampled
        # at each interval's midpoint.
        n = len(time_grid) - 1
        kappa_avg = 0.0
        theta_avg = 0.0
        sigma_avg = 0.0
        rho_avg = 0.0
        for i in range(1, n + 1):
            t_mid = 0.5 * (time_grid[i - 1] + time_grid[i])
            kappa_avg += model.kappa(t_mid)
            theta_avg += model.theta(t_mid)
            sigma_avg += model.sigma(t_mid)
            rho_avg += model.rho(t_mid)
        kappa_avg /= n
        theta_avg /= n
        sigma_avg /= n
        rho_avg /= n

        self._evaluations = 0
        results.reset()

        if self._cpx_log == ComplexLogFormula.Gatheral:
            # C++ parity: .cpp:302-327.
            c_inf = min(0.2, max(0.0001, math.sqrt(1.0 - rho_avg * rho_avg) / sigma_avg)) * (
                v0 + kappa_avg * theta_avg * term
            )

            p1 = self._integration.calculate(c_inf, _FjHelper(model, term, strike, 1)) / math.pi
            self._evaluations += self._integration.number_of_evaluations()

            p2 = self._integration.calculate(c_inf, _FjHelper(model, term, strike, 2)) / math.pi
            self._evaluations += self._integration.number_of_evaluations()

            if payoff.option_type() == OptionType.Call:
                results.value = spot_price * dividend_discount * (p1 + 0.5) - strike * (
                    risk_free_discount
                ) * (p2 + 0.5)
            elif payoff.option_type() == OptionType.Put:
                results.value = spot_price * dividend_discount * (p1 - 0.5) - strike * (
                    risk_free_discount
                ) * (p2 - 0.5)
            else:
                raise LibraryException("unknown option type")
            return

        if self._cpx_log != ComplexLogFormula.AndersenPiterbarg:
            raise LibraryException("unknown complex log formula")

        # C++ parity: .cpp:329-400.
        qassert.require(
            term <= time_grid.back(),
            f"maturity ({term}) is too large, time grid is bounded by {time_grid.back()}",
        )

        t05 = 0.5 * time_grid.at(1)
        rho05 = model.rho(t05)
        d_u_inf = -complex(math.sqrt(1 - rho05 * rho05), rho05) / model.sigma(t05)

        last_i = bisect.bisect_left(time_grid.times, term)

        c_u_inf: complex = 0j
        for i in range(last_i):
            begin = time_grid[i]
            end = min(term, time_grid[i + 1])
            tau = end - begin
            t_mid = 0.5 * (end + begin)

            kappa = model.kappa(t_mid)
            theta = model.theta(t_mid)
            sigma = model.sigma(t_mid)
            rho = model.rho(t_mid)

            c_u_inf += (
                -kappa * theta * tau / sigma * complex(math.sqrt(1 - rho * rho), rho)
            )

        ratio = risk_free_discount / dividend_discount
        fwd_price = spot_price / ratio

        epsilon = (
            self._andersen_piterbarg_epsilon
            * math.pi
            / (math.sqrt(strike * fwd_price) * risk_free_discount)
        )
        c_inf = -(c_u_inf + d_u_inf * v0).real

        def u_m() -> float:
            # C++ parity: .cpp:369-371 — a lazily invoked std::function<Real()>.
            return Integration.andersen_piterbarg_integration_limit(c_inf, epsilon, v0, term)

        v_avg = (1 - math.exp(-kappa_avg * term)) * (v0 - theta_avg) / (
            kappa_avg * term
        ) + theta_avg

        bs_price = BlackCalculator.from_type_strike(
            OptionType.Call,
            strike,
            fwd_price,
            math.sqrt(v_avg * term),
            risk_free_discount,
        ).value()

        h_cv = (
            self._integration.calculate(
                c_inf,
                _APHelper(term, spot_price, strike, ratio, math.sqrt(v_avg), self),
                u_m,
            )
            * math.sqrt(strike * fwd_price)
            * risk_free_discount
            / math.pi
        )
        self._evaluations += self._integration.number_of_evaluations()

        if payoff.option_type() == OptionType.Call:
            results.value = bs_price + h_cv
        elif payoff.option_type() == OptionType.Put:
            results.value = bs_price + h_cv - risk_free_discount * (fwd_price - strike)
        else:
            raise LibraryException("unknown option type")

    def update(self) -> None:
        """Observer.update — model parameters or curves changed."""
        self.notify_observers()


__all__ = ["AnalyticPTDHestonEngine", "ComplexLogFormula"]
