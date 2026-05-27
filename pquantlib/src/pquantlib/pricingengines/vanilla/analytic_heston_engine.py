"""AnalyticHestonEngine — Heston model analytic engine via Fourier transform.

# C++ parity: ql/pricingengines/vanilla/analytichestonengine.{hpp,cpp}
# (v1.42.1) — ``class AnalyticHestonEngine : public
# GenericModelEngine<HestonModel, VanillaOption::arguments,
# VanillaOption::results>``.

Prices a plain-vanilla European option under the Heston dynamics by
inverting two characteristic functions over the positive real line::

    p_j = 1/pi * Integral_0^inf Re(exp(-i*phi*ln(K)) * f_j(phi) / (i*phi)) dphi

(written in pquantlib's `Fj_Helper` form, the integrand pulls the
`/phi` factor out of `Re(...)` and uses `Im(exp(...))/phi` directly —
this is the standard Heston "rotation count" form due to Gatheral.)

The final option price is::

    Call = spot * dd * (p1 + 0.5) - strike * dr * (p2 + 0.5)
    Put  = spot * dd * (p1 - 0.5) - strike * dr * (p2 - 0.5)

with ``dr = riskfree.discount(T)`` and ``dd = dividendYield.discount(T)``.

Divergences from C++:

* **Numerical integration**: scipy's adaptive ``scipy.integrate.quad``
  on ``(0, +inf)`` (which uses QUADPACK's QAGI under the hood) replaces
  the C++ Gauss-Laguerre quadrature. The L4-C design accepts a slight
  precision degradation (LOOSE tier, abs_tol=1e-8, rel_tol=1e-8) in
  exchange for not porting a 144-point Gauss-Laguerre weights table.
  scipy.quad usually returns ~1e-8 absolute accuracy without parameter
  tuning, which is well within the calibration tolerance needed
  downstream.
* **ComplexLogFormula**: only the ``Gatheral`` form is implemented.
  ``BranchCorrection`` (used in the legacy Heston 1993 paper) is left
  out — it requires the ``b_`` / ``g_km1`` branch tracking, which is
  numerically inferior on adaptive integrators. The Andersen-Piterbarg
  / AngledContour / OptimalCV control-variate forms are also deferred;
  they're tuning parameters that improve robustness for extreme strikes
  but Gatheral is sufficient for the standard parameter range covered
  by the L4-C testbed (and the typical Heston calibration regime).
* **AP_Helper** / **OptimalAlpha** classes are NOT ported (only used
  by the deferred control-variate forms).
* **chF / lnChF** (the bare characteristic function) are NOT exposed.
  L4-C only needs ``priceVanillaPayoff``; downstream users wanting the
  CF can compute it inline from the Gatheral parameters.
* **Constructors**: only the ``integrationOrder`` form is honoured
  (kept as ``integration_order`` kwarg for API parity but the value is
  ignored — scipy.quad is adaptive). The ``relTolerance`` /
  ``maxEvaluations`` Gauss-Lobatto form maps to scipy.quad's
  ``epsabs`` / ``epsrel`` / ``limit``.
* **Bates / add-on term hook**: a class-level ``add_on_term`` method
  exists (and returns 0.0 + 0i by default — matching the C++ virtual
  base). Sub-engines (BatesEngine, deferred to L5) can override; the
  Fj_Helper integrand picks it up.
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
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine


class AnalyticHestonEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Analytic Heston engine via Fourier transform (Gatheral form).

    # C++ parity: ``class AnalyticHestonEngine`` in
    # ql/pricingengines/vanilla/analytichestonengine.hpp:91-195 (v1.42.1).
    """

    # __slots__ would shadow GenericEngine's _arguments/_results. The
    # base class doesn't __slot__ them, so we follow the same pattern
    # — no __slots__ here.

    def __init__(
        self,
        model: HestonModel,
        integration_order: int = 144,
    ) -> None:
        """Construct from a HestonModel.

        # C++ parity: ``AnalyticHestonEngine(model, integrationOrder)``
        # in analytichestonengine.cpp:659-671.

        Parameters
        ----------
        model:
            The Heston model whose parameters drive the characteristic
            function.
        integration_order:
            Kept as a kwarg for API parity with C++. The Python port
            ignores it (scipy.integrate.quad is adaptive); a future
            refinement could route it through scipy.integrate.fixed_quad
            with a Gauss-Laguerre node table.
        """
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._model: HestonModel = model
        # Stored for API parity / introspection only.
        self._integration_order: int = integration_order
        self._evaluations: int = 0
        # C++ parity: analytichestonengine.cpp:665-670 — register with
        # model so the engine invalidates when the model's parameters
        # change (e.g. during calibration).
        model.register_with(self)

    # --- inspectors -----------------------------------------------------

    def model(self) -> HestonModel:
        """The underlying HestonModel."""
        return self._model

    def number_of_evaluations(self) -> int:
        """Count of cost-function evaluations during the last calculate.

        # C++ parity: ``AnalyticHestonEngine::numberOfEvaluations`` in
        # analytichestonengine.cpp:721-723.
        """
        return self._evaluations

    # --- characteristic-function machinery ------------------------------

    def add_on_term(self, phi: float, t: float, j: int) -> complex:
        """Sub-engine hook: extra term in the log of the characteristic function.

        # C++ parity: ``AnalyticHestonEngine::add_onTerm`` in
        # analytichestonengine.hpp:179-181 + .hpp:307-310 (inline
        # default returning ``complex<Real>(0, 0)``).

        Default is zero; BatesEngine (deferred to L5) overrides to
        inject the Merton-jump compensator into both j=1 and j=2
        integrands.
        """
        del phi, t, j  # default is parameter-independent
        return 0 + 0j

    def _fj(
        self,
        phi: float,
        *,
        j: int,
        term: float,
        spot: float,
        strike: float,
        ratio: float,
    ) -> float:
        """Heston Gatheral-form integrand for j=1 or j=2.

        # C++ parity: ``AnalyticHestonEngine::Fj_Helper::operator()(phi)``
        # in analytichestonengine.cpp:181-296 (Gatheral branch only).

        Returns the real-valued integrand to be integrated over
        ``phi`` on ``(0, +inf)``.
        """
        kappa = self._model.kappa()
        theta = self._model.theta()
        sigma = self._model.sigma()
        v0 = self._model.v0()
        rho = self._model.rho()

        x = math.log(spot)
        sx = math.log(strike)
        dd = x - math.log(ratio)
        sigma2 = sigma * sigma
        rsigma = rho * sigma
        t0 = kappa - (rsigma if j == 1 else 0.0)

        rpsig = rsigma * phi
        t1 = complex(t0, -rpsig)
        # The inner complex(-phi, +1 if j==1 else -1) factor in
        # ``t1*t1 - sigma2*phi*complex(-phi, sgn)``.
        sgn = 1.0 if j == 1 else -1.0
        d = np.lib.scimath.sqrt(t1 * t1 - sigma2 * phi * complex(-phi, sgn))
        ex = np.exp(-d * term)

        add_on = self.add_on_term(phi, term, j)

        if phi != 0.0:
            if sigma > 1e-5:
                p = (t1 - d) / (t1 + d)
                # C++ uses std::log here; the principal branch is fine
                # for Gatheral form.
                g = np.lib.scimath.log((1.0 - p * ex) / (1.0 - p))
                arg = (
                    v0 * (t1 - d) * (1.0 - ex) / (sigma2 * (1.0 - ex * p))
                    + (kappa * theta) / sigma2 * ((t1 - d) * term - 2.0 * g)
                    + complex(0.0, phi * (dd - sx))
                    + add_on
                )
                return float(np.exp(arg).imag / phi)
            # sigma ≈ 0: use the L'Hospital expansion.
            td = phi / (2.0 * t1) * complex(-phi, sgn)
            p = td * sigma2 / (t1 + d)
            g = p * (1.0 - ex)
            arg = (
                v0 * td * (1.0 - ex) / (1.0 - p * ex)
                + (kappa * theta) * (td * term - 2.0 * g / sigma2)
                + complex(0.0, phi * (dd - sx))
                + add_on
            )
            return float(np.exp(arg).imag / phi)

        # phi == 0 — L'Hospital limit.
        # C++ parity: analytichestonengine.cpp:222-242.
        if j == 1:
            kmr = rsigma - kappa
            if abs(kmr) > 1e-7:
                return (
                    dd - sx
                    + (math.exp(kmr * term) * kappa * theta
                       - kappa * theta * (kmr * term + 1.0)) / (2.0 * kmr * kmr)
                    - v0 * (1.0 - math.exp(kmr * term)) / (2.0 * kmr)
                )
            # kappa == rho*sigma — series expansion.
            return dd - sx + 0.25 * kappa * theta * term * term + 0.5 * v0 * term
        # j == 2.
        return (
            dd - sx
            - (math.exp(-kappa * term) * kappa * theta
               + kappa * theta * (kappa * term - 1.0)) / (2.0 * kappa * kappa)
            - v0 * (1.0 - math.exp(-kappa * term)) / (2.0 * kappa)
        )

    def _integrate_pj(
        self,
        *,
        j: int,
        term: float,
        spot: float,
        strike: float,
        ratio: float,
    ) -> float:
        """Integrate the j-th Heston integrand over (0, +inf).

        Returns ``int_0^inf Fj(phi) dphi`` via scipy.integrate.quad.
        Updates ``self._evaluations`` with QUADPACK's evaluation count.
        """
        def integrand(phi: float) -> float:
            return self._fj(phi, j=j, term=term, spot=spot, strike=strike, ratio=ratio)

        # scipy.integrate.quad with (0, +inf) routes through QUADPACK's
        # QAGI — adaptive double-exponential transform. abs_tol /
        # rel_tol chosen to match the L4-C LOOSE tier; integral
        # accuracy is then well under 1e-8 in absolute terms for the
        # standard Heston regime.
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

    # --- pricing ---------------------------------------------------------

    def _price_vanilla_payoff(
        self,
        *,
        payoff: PlainVanillaPayoff,
        maturity: float,
    ) -> float:
        """Closed-form price of the European vanilla under Heston.

        # C++ parity: ``AnalyticHestonEngine::priceVanillaPayoff`` in
        # analytichestonengine.cpp:725-746 + 748-859.

        Only the Gatheral branch is implemented (see module docstring).
        """
        process = self._model.process()
        spot = process.s0().value()
        qassert.require(spot > 0.0, f"negative or null underlying given: {spot}")
        strike = payoff.strike()

        dr = process.risk_free_rate().discount(maturity)
        dd = process.dividend_yield().discount(maturity)
        # C++ parity: analytichestonengine.cpp:761-762 — dd here is
        # the dividend discount via the algebraic identity
        # dr / (spot/fwd) = dd, where fwd = spot * dd / dr.
        # We use dd directly (less arithmetic, same result).
        fwd = spot * dd / dr
        ratio = spot / fwd  # = dr / dd

        self._evaluations = 0

        p1 = self._integrate_pj(j=1, term=maturity, spot=spot, strike=strike, ratio=ratio) / math.pi
        p2 = self._integrate_pj(j=2, term=maturity, spot=spot, strike=strike, ratio=ratio) / math.pi

        if payoff.option_type() == OptionType.Call:
            return spot * dd * (p1 + 0.5) - strike * dr * (p2 + 0.5)
        if payoff.option_type() == OptionType.Put:
            return spot * dd * (p1 - 0.5) - strike * dr * (p2 - 0.5)
        raise LibraryException(f"unknown option type: {payoff.option_type()}")

    def calculate(self) -> None:
        """Run the engine: fill ``results.value`` for a European vanilla.

        # C++ parity: ``AnalyticHestonEngine::calculate`` in
        # analytichestonengine.cpp:861-875.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not a European option",
        )
        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff),
            "non plain vanilla payoff given",
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        process = self._model.process()
        exercise_date = args.exercise.last_date()
        maturity = process.time(exercise_date)

        # Reset the standard Greek slots (the analytic Heston engine
        # only fills NPV, mirroring the C++ which sets results_.value
        # and leaves the Greeks as their NaN sentinels).
        results.reset()
        results.value = self._price_vanilla_payoff(
            payoff=payoff,
            maturity=maturity,
        )

    def update(self) -> None:
        """Observer.update — model parameters or curves changed.

        # C++ parity: ``GenericEngine`` mixes in ``Observer``; we just
        # notify our own observers (the instrument that owns us picks
        # up via Instrument.update).
        """
        self.notify_observers()


__all__ = ["AnalyticHestonEngine"]
