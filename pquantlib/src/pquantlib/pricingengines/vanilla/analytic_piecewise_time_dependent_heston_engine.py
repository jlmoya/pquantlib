"""AnalyticPiecewiseTimeDependentHestonEngine — segmented Heston Fourier engine.

# C++ parity: ql/pricingengines/vanilla/analyticptdhestonengine.{hpp,cpp}
# (v1.42.1).

Generalizes the L4-C ``AnalyticHestonEngine`` to allow piecewise-constant
``theta(t)``, ``kappa(t)``, ``sigma(t)``, ``rho(t)``. The characteristic
function is built by walking the time grid backwards from maturity,
accumulating the (D, C) Riccati pair using each segment's local parameters.

Pricing equation::

    Call = spot * dd * (p1 + 0.5) - strike * dr * (p2 + 0.5)
    Put  = spot * dd * (p1 - 0.5) - strike * dr * (p2 - 0.5)

where ``p_j = (1/pi) * integral_0^inf Fj_Helper(phi) dphi``.

Divergences from C++:
- Numerical integration: scipy.integrate.quad on (0, +inf), as for
  the L4-C ``AnalyticHestonEngine``.
- Only the ``Gatheral`` complex-log formula is implemented; the
  ``AndersenPiterbarg`` control-variate form is deferred (it requires
  a separate ``AP_Helper`` + ``lnChF`` machinery and a Black-vol
  baseline; not exercised by the L11-W1-D testbed).
- ``Integration`` class is not modeled; ``integration_order`` is kept
  as a kwarg for API parity but quad is adaptive.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, cast

import numpy as np
from scipy.integrate import quad  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.models.equity.piecewise_time_dependent_heston_model import (
    PiecewiseTimeDependentHestonModel,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency


class AnalyticPiecewiseTimeDependentHestonEngine(
    GenericEngine[OptionArguments, OneAssetOptionResults]
):
    """European-vanilla engine for piecewise-time-dep Heston via Fourier.

    # C++ parity: ``class AnalyticPTDHestonEngine`` in
    # analyticptdhestonengine.hpp:52-99 (v1.42.1).
    """

    def __init__(
        self,
        model: PiecewiseTimeDependentHestonModel,
        integration_order: int = 144,
    ) -> None:
        """Construct from a ``PiecewiseTimeDependentHestonModel``.

        # C++ parity: analyticptdhestonengine.cpp:211-221.
        """
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._model: PiecewiseTimeDependentHestonModel = model
        self._integration_order: int = integration_order
        self._evaluations: int = 0
        # C++ parity: registerWith(model) — engine listens for param changes.
        model.register_with(self)

    # --- inspectors -----------------------------------------------------

    def model(self) -> PiecewiseTimeDependentHestonModel:
        """The underlying model."""
        return self._model

    def number_of_evaluations(self) -> int:
        """Total cost-function evaluations during the last calculate.

        # C++ parity: analyticptdhestonengine.cpp:408-410.
        """
        return self._evaluations

    # --- Fj_Helper integrand -----------------------------------------

    def _fj(
        self,
        phi_in: float,
        *,
        j: int,
        term: float,
        log_strike: float,
        log_spot: float,
        rates: list[float],
        divs: list[float],
    ) -> float:
        """Heston Gatheral-form integrand for j=1 or j=2 with piecewise params.

        # C++ parity: ``AnalyticPTDHestonEngine::Fj_Helper::operator()``
        # in analyticptdhestonengine.cpp:76-120.

        Walks the time grid backwards from ``maturity`` accumulating
        the (D, C) Riccati pair.
        """
        # C++ parity: cpp:80 — avoid numeric overflow for phi→0 by clamping.
        phi = max(np.finfo(np.float32).eps, phi_in)

        big_d: complex = 0 + 0j
        big_c: complex = 0 + 0j
        v0 = self._model.v0()
        time_grid = self._model.time_grid()
        n = len(time_grid)

        # Walk segments backwards from last to first.
        for i in range(n - 1, 0, -1):
            begin = time_grid[i - 1]
            if begin < term:
                end = min(term, time_grid[i])
                tau = end - begin
                t_mid = 0.5 * (end + begin)

                rho = self._model.rho(t_mid)
                sigma = self._model.sigma(t_mid)
                kappa = self._model.kappa(t_mid)
                theta = self._model.theta(t_mid)

                sigma2 = sigma * sigma
                t0 = kappa - (rho * sigma if j == 1 else 0.0)
                rpsig = rho * sigma * phi

                t1 = complex(t0, -rpsig)
                sgn = 1.0 if j == 1 else -1.0
                d = np.lib.scimath.sqrt(t1 * t1 - sigma2 * phi * complex(-phi, sgn))
                g = (t1 - d) / (t1 + d)
                ex = np.exp(-d * tau)
                gt = (t1 - d - big_d * sigma2) / (t1 + d - big_d * sigma2)

                big_d = (t1 + d) / sigma2 * (g - gt * ex) / (1.0 - gt * ex)

                lng = np.lib.scimath.log((1.0 - gt * ex) / (1.0 - gt))

                big_c = (
                    (kappa * theta) / sigma2 * ((t1 - d) * tau - 2.0 * lng)
                    + complex(0.0, phi * (rates[i - 1] - divs[i - 1]) * tau)
                    + big_c
                )

        arg = v0 * big_d + big_c + complex(0.0, phi * (log_spot - log_strike))
        return float(np.exp(arg).imag / phi)

    # --- pricing ---------------------------------------------------------

    def _segmented_rates(self, term: float) -> tuple[list[float], list[float]]:
        """Per-segment forward (r, q) rates over the time grid.

        # C++ parity: analyticptdhestonengine.cpp:66-73 (Fj_Helper ctor).
        """
        tg = self._model.time_grid()
        n = len(tg)
        rates: list[float] = []
        divs: list[float] = []
        rf = self._model.risk_free_rate()
        dq = self._model.dividend_yield()
        for i in range(n - 1):
            begin = min(term, tg[i])
            end = min(term, tg[i + 1])
            # Edge: degenerate end == begin → forward over a tiny window
            # to keep forward_rate(b, e) valid.
            if end <= begin:
                begin_eff = begin
                end_eff = begin + 1e-8
            else:
                begin_eff = begin
                end_eff = end
            r = rf.forward_rate(
                begin_eff,
                end_eff,
                Compounding.Continuous,
                Frequency.NoFrequency,
                True,
            ).rate()
            q = dq.forward_rate(
                begin_eff,
                end_eff,
                Compounding.Continuous,
                Frequency.NoFrequency,
                True,
            ).rate()
            rates.append(r)
            divs.append(q)
        return rates, divs

    def _integrate_pj(
        self,
        *,
        j: int,
        term: float,
        log_strike: float,
        log_spot: float,
        rates: list[float],
        divs: list[float],
    ) -> float:
        """Integrate the j-th Heston integrand over (0, +inf)."""

        def integrand(phi: float) -> float:
            return self._fj(
                phi,
                j=j,
                term=term,
                log_strike=log_strike,
                log_spot=log_spot,
                rates=rates,
                divs=divs,
            )

        quad_result = cast(
            "tuple[float, float, dict[str, Any]]",
            quad(
                integrand,
                0.0,
                np.inf,
                epsabs=1e-10,
                epsrel=1e-10,
                limit=200,
                full_output=1,
            ),
        )
        result, _abs_err, infodict = quad_result
        neval_raw = infodict.get("neval", 0)
        with contextlib.suppress(TypeError, ValueError):
            self._evaluations += int(neval_raw)
        return float(result)

    def calculate(self) -> None:
        """Run the engine — fill ``results.value`` for a European vanilla.

        # C++ parity: ``AnalyticPTDHestonEngine::calculate`` in
        # analyticptdhestonengine.cpp:252-406 (Gatheral branch only).
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European option",
        )
        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff),
            "non-striked payoff given",
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        model = self._model
        spot = model.s0()
        qassert.require(spot > 0.0, "negative or null underlying given")

        strike = payoff.strike()
        rf = model.risk_free_rate()
        term = rf.day_counter().year_fraction(
            rf.reference_date(), args.exercise.last_date()
        )

        tg = model.time_grid()
        qassert.require(len(tg) > 1, "at least two model points needed")
        qassert.require(
            term <= tg.back() or math.isclose(term, tg.back(), abs_tol=1e-10),
            f"maturity ({term}) is too large, time grid is bounded by {tg.back()}",
        )

        risk_free_discount = rf.discount(args.exercise.last_date())
        dividend_discount = model.dividend_yield().discount(args.exercise.last_date())

        self._evaluations = 0

        rates, divs = self._segmented_rates(term)

        p1 = (
            self._integrate_pj(
                j=1,
                term=term,
                log_strike=math.log(strike),
                log_spot=math.log(spot),
                rates=rates,
                divs=divs,
            )
            / math.pi
        )
        p2 = (
            self._integrate_pj(
                j=2,
                term=term,
                log_strike=math.log(strike),
                log_spot=math.log(spot),
                rates=rates,
                divs=divs,
            )
            / math.pi
        )

        results.reset()
        if payoff.option_type() == OptionType.Call:
            results.value = spot * dividend_discount * (p1 + 0.5) - strike * risk_free_discount * (
                p2 + 0.5
            )
        elif payoff.option_type() == OptionType.Put:
            results.value = spot * dividend_discount * (p1 - 0.5) - strike * risk_free_discount * (
                p2 - 0.5
            )
        else:
            raise LibraryException(f"unknown option type: {payoff.option_type()}")

    def update(self) -> None:
        """Observer.update — model parameters or curves changed."""
        self.notify_observers()


__all__ = ["AnalyticPiecewiseTimeDependentHestonEngine"]
