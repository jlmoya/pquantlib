"""COSHestonEngine — Fourier-cosine (Fang-Oosterlee) Heston pricer.

# C++ parity: ql/pricingengines/vanilla/coshestonengine.{hpp,cpp}
# (v1.43) — ``class COSHestonEngine : public GenericModelEngine<HestonModel,
# VanillaOption::arguments, VanillaOption::results>``.

References
----------
F. Fang, C. W. Oosterlee, *A Novel Pricing Method for European Options based on
Fourier-Cosine Series Expansions*.
Fabien Le Floc'h, *Fourier Integration and Stochastic Volatility Calibration*.

The engine expands the density on ``[a, b] = [x + c1 - L*w, x + c1 + L*w]``
(``w = sqrt(|c2|)``, ``x = log(forward/strike)``) in a cosine series of ``N``
terms and integrates the payoff term by term. ``L`` and ``N`` are the only
tuning knobs and both are honoured here.

Things a port gets wrong easily
------------------------------
* ``muT()`` — the drift ``log(dividendDiscount / riskFreeDiscount)`` — is
  private in C++ and **never called** in v1.43. The cumulants ``c1``..``c4``
  are therefore the *driftless* ones and depend only on
  ``(v0, kappa, theta, sigma, rho, t)``: no rates, no spot, no strike. Folding
  the drift into ``c1`` would look like a fix and would break every cumulant.
* ``w`` uses only ``sqrt(|c2|)``; the commented-out 4th-order refinement in the
  C++ source is deliberately NOT applied.
* When ``x >= b/2`` or ``x <= a/2`` the series is abandoned and the engine
  returns the discounted no-arbitrage BOUND (``max(spot*qf - K*df, 0)`` for a
  call, ``max(K*df - spot*qf, 0)`` for a put), not a price. Deep wings at short
  maturity hit this.
* The ``n = 0`` term is ``chF(0, T).real() * (exp(a) - 1 - a) * d`` — evaluated
  from ``chF`` rather than short-circuited to ``1``.
* The call branch is ``spot*qf - K*df*(1 - s)``, i.e. it recomputes the
  dividend discount locally; it is NOT ``K*df*s`` plus parity.

Cross-validated against C++ v1.43 in
``pquantlib/tests/pricingengines/vanilla/test_cos_heston_engine.py``
(reference ``migration-harness/references/v143/pe/heston.json``).
"""

from __future__ import annotations

import cmath
import math

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine

# The cumulant closed forms below are transliterated token-for-token from the
# Mathematica-generated C++; binding ``pow``/``exp`` to short module-level names
# keeps them readable as the same arithmetic.
_pow = math.pow
exp = math.exp


class COSHestonEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """COS-method Heston engine based on Fourier-cosine series expansions.

    # C++ parity: ``class COSHestonEngine`` in coshestonengine.hpp:52-77
    # (v1.43).
    """

    def __init__(self, model: HestonModel, l: float = 16.0, n: int = 200) -> None:  # noqa: E741
        """Construct from a Heston model.

        # C++ parity: ``COSHestonEngine(model, L = 16, N = 200)`` in
        # coshestonengine.cpp:27-37. The model parameters are CACHED at
        # construction and refreshed in :meth:`update`, exactly as in C++ —
        # a port reading them live from the model on every call would still
        # price correctly but would not reproduce C++'s observer semantics.

        Parameters
        ----------
        model:
            The Heston model.
        l:
            ``L`` — half-width of the truncation range in units of
            ``sqrt(|c2(T)|)``.
        n:
            ``N`` — number of cosine terms.
        """
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._model: HestonModel = model
        self._l: float = l
        self._n: int = n
        self._kappa: float = model.kappa()
        self._theta: float = model.theta()
        self._sigma: float = model.sigma()
        self._rho: float = model.rho()
        self._v0: float = model.v0()
        model.register_with(self)

    def model(self) -> HestonModel:
        """The underlying Heston model."""
        return self._model

    def update(self) -> None:
        """Refresh the cached model parameters and notify observers.

        # C++ parity: ``COSHestonEngine::update`` (coshestonengine.cpp:40-49).
        """
        self._kappa = self._model.kappa()
        self._theta = self._model.theta()
        self._sigma = self._model.sigma()
        self._rho = self._model.rho()
        self._v0 = self._model.v0()
        self.notify_observers()

    # --- characteristic function ------------------------------------------

    def ch_f(self, u: float, t: float) -> complex:
        """Normalized (driftless) Heston characteristic function at real ``u``.

        # C++ parity: ``COSHestonEngine::chF(Real u, Real t)``
        # (coshestonengine.cpp:126-142). Note the argument is REAL here, unlike
        # ``AnalyticHestonEngine::chF`` which takes a complex ``z``.
        """
        sigma = self._sigma
        kappa = self._kappa
        rho = self._rho
        theta = self._theta
        v0 = self._v0
        sigma2 = sigma * sigma

        d = cmath.sqrt(
            complex(kappa, -rho * sigma * u) ** 2 + complex(u * u, u) * sigma2
        )
        g = complex(kappa, -rho * sigma * u)
        big_g = (g - d) / (g + d)

        return cmath.exp(
            v0 / sigma2 * (1.0 - cmath.exp(-d * t)) / (1.0 - big_g * cmath.exp(-d * t)) * (g - d)
            + kappa
            * theta
            / sigma2
            * ((g - d) * t - 2.0 * cmath.log((1.0 - big_g * cmath.exp(-d * t)) / (1.0 - big_g)))
        )

    # --- cumulants and moments ---------------------------------------------

    def c1(self, t: float) -> float:
        """First cumulant of log(S_T/S_0) under the driftless Heston dynamics.

        # C++ parity: ``COSHestonEngine::c1`` in coshestonengine.cpp:177
        # (v1.43) — a Mathematica-generated closed form, transliterated
        # token-for-token. Depends only on the model parameters and ``t``:
        # no rates, no spot, no strike.
        """
        # No sigma / rho here: unlike c2..c4, the C++ first cumulant
        # (coshestonengine.cpp:177-181) is free of both.
        kappa = self._kappa
        theta = self._theta
        v0 = self._v0
        return (
        (-theta +exp(kappa * t) * (theta -kappa * t * theta -v0) +v0) / (2 * exp(kappa * t) *
        kappa)
        )

    def c2(self, t: float) -> float:
        """Second cumulant (variance) of log(S_T/S_0).

        # C++ parity: ``COSHestonEngine::c2`` in coshestonengine.cpp:183
        # (v1.43) — a Mathematica-generated closed form, transliterated
        # token-for-token. Depends only on the model parameters and ``t``:
        # no rates, no spot, no strike.
        """
        kappa = self._kappa
        theta = self._theta
        sigma = self._sigma
        rho = self._rho
        v0 = self._v0
        sigma2 = sigma * sigma
        kappa2 = kappa * kappa
        kappa3 = kappa2 * kappa
        return (
        (sigma2 * (theta -2 * v0) +exp(2 * kappa * t) * (8 * kappa3 * t * theta -8 * kappa2 *
        (theta +rho * sigma * t * theta -v0) +sigma2 * (-5 * theta +2 * v0) +2 * kappa * sigma *
        (8 * rho * theta +sigma * t * theta -4 * rho * v0)) +4 * exp(kappa * t) * (sigma2 * theta
        -2 * kappa2 * (-1 +rho * sigma * t) * (theta -v0) +kappa * sigma * (sigma * t * (theta
        -v0) +2 * rho * (-2 * theta +v0)))) / (8. * exp(2 * kappa * t) * kappa3)
        )

    def c3(self, t: float) -> float:
        """Third cumulant of log(S_T/S_0).

        # C++ parity: ``COSHestonEngine::c3`` in coshestonengine.cpp:199
        # (v1.43) — a Mathematica-generated closed form, transliterated
        # token-for-token. Depends only on the model parameters and ``t``:
        # no rates, no spot, no strike.
        """
        kappa = self._kappa
        theta = self._theta
        sigma = self._sigma
        rho = self._rho
        v0 = self._v0
        sigma2 = sigma * sigma
        sigma3 = sigma2 * sigma
        kappa2 = kappa * kappa
        kappa3 = kappa2 * kappa
        kappa4 = kappa3 * kappa
        rho2 = rho * rho
        return (
        -(sigma * (sigma3 * (theta -3 * v0) +exp(3 * kappa * t) * (2 * (-11 * sigma3 -24 * kappa4
        * rho * t +3 * kappa * sigma2 * (20 * rho +sigma * t) -6 * kappa2 * sigma * (5 +3 * rho *
        (4 * rho +sigma * t)) +12 * kappa3 * (sigma * t +2 * rho * (2 +rho * sigma * t))) * theta
        -6 * (2 * kappa * rho -sigma) * (4 * kappa2 -4 * kappa * rho * sigma +sigma2) * v0) +6 *
        exp(kappa * t) * sigma * (-2 * kappa2 * (-1 +rho * sigma * t) * (theta -2 * v0) +sigma2 *
        (theta -v0) +kappa * sigma * (-4 * rho * theta +sigma * t * theta +6 * rho * v0 -2 * sigma
        * t * v0)) +3 * exp(2 * kappa * t) * (2 * kappa * sigma2 * (-16 * rho * theta +sigma * t *
        (3 * theta -v0)) +8 * kappa4 * rho * t * (-2 +rho * sigma * t) * (theta -v0) +sigma3 * (5
        * theta +v0) +8 * kappa3 * (-(rho * (4 +sigma2 * t * t) * theta) +2 * sigma * t * (theta
        -v0) +2 * rho2 * sigma * t * (2 * theta -v0) +rho * (2 +sigma2 * t * t) * v0) +2 * kappa2
        * sigma * ((8 +24 * rho2 -16 * rho * sigma * t +sigma2 * t * t) * theta -(8 * rho2 -8 *
        rho * sigma * t +sigma2 * t * t) * v0)))) / (16. * exp(3 * kappa * t) * kappa * kappa4)
        )

    def c4(self, t: float) -> float:
        """Fourth cumulant of log(S_T/S_0).

        # C++ parity: ``COSHestonEngine::c4`` in coshestonengine.cpp:228
        # (v1.43) — a Mathematica-generated closed form, transliterated
        # token-for-token. Depends only on the model parameters and ``t``:
        # no rates, no spot, no strike.
        """
        kappa = self._kappa
        theta = self._theta
        sigma = self._sigma
        rho = self._rho
        v0 = self._v0
        sigma2 = sigma * sigma
        sigma3 = sigma2 * sigma
        sigma4 = sigma2 * sigma2
        kappa2 = kappa * kappa
        kappa3 = kappa2 * kappa
        kappa4 = kappa2 * kappa2
        kappa5 = kappa2 * kappa3
        kappa6 = kappa3 * kappa3
        kappa7 = kappa4 * kappa3
        rho2 = rho * rho
        rho3 = rho2 * rho
        t2 = t * t
        t3 = t2 * t
        return (
        (sigma2 * (3 * sigma4 * (theta -4 * v0) +3 * exp(4 * kappa * t) * ((-93 * sigma4 +64 *
        kappa5 * (t +4 * rho2 * t) +4 * kappa * sigma3 * (176 * rho +5 * sigma * t) -32 * kappa2 *
        sigma2 * (11 +50 * rho2 +5 * rho * sigma * t) +32 * kappa3 * sigma * (3 * sigma * t +4 *
        rho * (10 +8 * rho2 +3 * rho * sigma * t)) -32 * kappa4 * (5 +4 * rho * (6 * rho +(3 +2 *
        rho2) * sigma * t))) * theta +4 * (4 * kappa2 -4 * kappa * rho * sigma +sigma2) * (4 *
        kappa2 * (1 +4 * rho2) -20 * kappa * rho * sigma +5 * sigma2) * v0) +24 * exp(kappa * t) *
        sigma2 * (-2 * kappa2 * (-1 +rho * sigma * t) * (theta -3 * v0) +sigma2 * (theta -2 * v0)
        +kappa * sigma * (-4 * rho * theta +sigma * t * theta +10 * rho * v0 -3 * sigma * t * v0))
        +12 * exp(2 * kappa * t) * (sigma4 * (7 * theta -4 * v0) +8 * kappa4 * (1 +2 * rho * sigma
        * t * (-2 +rho * sigma * t)) * (theta -2 * v0) +2 * kappa * sigma3 * (-24 * rho * theta +5
        * sigma * t * theta +20 * rho * v0 -6 * sigma * t * v0) +4 * kappa2 * sigma2 * ((6 +20 *
        rho2 -14 * rho * sigma * t +sigma2 * t2) * theta -2 * (3 +12 * rho2 -10 * rho * sigma * t
        +sigma2 * t2) * v0) +8 * kappa3 * sigma * ((3 * sigma * t +2 * rho * (-4 +sigma * t * (4 *
        rho -sigma * t))) * theta +2 * (-3 * sigma * t +2 * rho * (3 +sigma * t * (-3 * rho +sigma
        * t))) * v0)) -8 * exp(3 * kappa * t) * (16 * kappa6 * rho2 * t2 * (-3 +rho * sigma * t) *
        (theta -v0) -3 * sigma4 * (7 * theta +2 * v0) +2 * kappa3 * sigma * ((192 * (rho +rho3) -6
        * (9 +40 * rho2) * sigma * t +42 * rho * sigma2 * t2 -sigma3 * t3) * theta +(-48 * rho3
        +18 * (1 +4 * rho2) * sigma * t -24 * rho * sigma2 * t2 +sigma3 * t3) * v0) +12 * kappa4 *
        ((-4 -24 * rho2 +8 * rho * (4 +3 * rho2) * sigma * t -(3 +14 * rho2) * sigma2 * t2 +rho *
        sigma3 * t3) * theta +(8 * rho2 -8 * rho * (2 +rho2) * sigma * t +(3 +8 * rho2) * sigma2 *
        t2 -rho * sigma3 * t3) * v0) -6 * kappa2 * sigma2 * ((15 +80 * rho2 -35 * rho * sigma * t
        +2 * sigma2 * t2) * theta +(3 +sigma * t * (7 * rho -sigma * t)) * v0) +24 * kappa5 * t *
        ((-2 +rho * (4 * sigma * t +rho * (-8 +sigma * t * (4 * rho -sigma * t)))) * theta +(2
        +rho * (-4 * sigma * t +rho * (4 +sigma * t * (-2 * rho +sigma * t)))) * v0) +3 * kappa *
        sigma3 * (sigma * t * (-9 * theta +v0) +10 * rho * (6 * theta +v0))))) / (64. * exp(4 *
        kappa * t) * kappa7)
        )

    def mu(self, t: float) -> float:
        """First cumulant. # C++ parity: ``COSHestonEngine::mu`` (.cpp:325-327)."""
        return self.c1(t)

    def var(self, t: float) -> float:
        """Second cumulant. # C++ parity: ``COSHestonEngine::var`` (.cpp:328-330)."""
        return self.c2(t)

    def skew(self, t: float) -> float:
        """# C++ parity: ``COSHestonEngine::skew`` (.cpp:331-333) — ``c3 / c2**1.5``."""
        return self.c3(t) / _pow(self.c2(t), 1.5)

    def kurtosis(self, t: float) -> float:
        """# C++ parity: ``COSHestonEngine::kurtosis`` (.cpp:334-336) — ``c4 / c2**2``."""
        c2 = self.c2(t)
        return self.c4(t) / (c2 * c2)

    # --- pricing ------------------------------------------------------------

    def calculate(self) -> None:
        """Price a European vanilla by the COS method.

        # C++ parity: ``COSHestonEngine::calculate`` (coshestonengine.cpp:52-123).
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
            "non plain vanilla payoff given",
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        process = self._model.process()
        maturity_date = args.exercise.last_date()
        maturity = process.time(maturity_date)

        cum1 = self.c1(maturity)
        # C++ parity: .cpp:69-72 — the commented-out 4th-order term is NOT used.
        w = math.sqrt(abs(self.c2(maturity)))

        k = payoff.strike()
        spot = process.s0().value()
        qassert.require(spot > 0.0, "negative or null underlying given")

        df = process.risk_free_rate().discount(maturity_date)
        qf = process.dividend_yield().discount(maturity_date)
        fwd = spot * qf / df
        x = math.log(fwd / k)

        a = x + cum1 - self._l * w
        b = x + cum1 + self._l * w

        results.reset()

        # C++ parity: .cpp:87-98 — outside the truncation range the series is
        # abandoned and the discounted no-arbitrage bound is returned.
        if x >= b / 2 or x <= a / 2:
            if payoff.option_type() == OptionType.Put:
                results.value = max(-spot * qf + k * df, 0.0)
            elif payoff.option_type() == OptionType.Call:
                results.value = max(spot * qf - k * df, 0.0)
            else:
                raise LibraryException(f"unknown payoff type: {payoff.option_type()}")
            return

        d = 1.0 / (b - a)
        exp_a = math.exp(a)
        s = self.ch_f(0.0, maturity).real * (exp_a - 1 - a) * d

        for n in range(1, self._n):
            r = n * math.pi * d
            u_n = 2.0 * d * (
                1.0 / (1.0 + r * r) * (exp_a + r * math.sin(r * a) - math.cos(r * a))
                - 1.0 / r * math.sin(r * a)
            )
            s += u_n * (self.ch_f(r, maturity) * cmath.exp(complex(0, r * (x - a)))).real

        if payoff.option_type() == OptionType.Put:
            results.value = k * df * s
        elif payoff.option_type() == OptionType.Call:
            # C++ parity: .cpp:116-120 — the dividend discount is recomputed
            # here and the call is written as spot*qf - K*df*(1-s).
            qf_call = process.dividend_yield().discount(maturity_date)
            results.value = spot * qf_call - k * df * (1 - s)
        else:
            raise LibraryException(f"unknown payoff type: {payoff.option_type()}")


__all__ = ["COSHestonEngine"]
