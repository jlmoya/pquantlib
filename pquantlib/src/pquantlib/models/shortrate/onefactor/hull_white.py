"""Hull-White extended Vasicek single-factor short-rate model.

# C++ parity: ql/models/shortrate/onefactormodels/hullwhite.{hpp,cpp} (v1.42.1).

Implements the SDE

    dr_t = (theta(t) - a r_t) dt + sigma dW_t

with constants ``a`` (mean reversion) and ``sigma`` (volatility), and a
deterministic time-dependent drift ``theta(t)`` chosen so that the
model exactly reprices the input yield curve at time 0.

Hull-White is structurally a Vasicek extension: the short rate is
written as ``r_t = phi(t) + x_t`` where ``x_t`` is a standard
Ornstein-Uhlenbeck process and ``phi(t)`` is the analytical
term-structure fitting parameter:

    phi(t) = f(t) + 0.5 * (sigma * (1 - exp(-a t)) / a)^2

(with the algebraic small-``a`` limit ``phi(t) = f(t) + 0.5*(sigma*t)^2``).

The discount-bond price is

    P(t, T, r) = A(t, T) * exp(-B(t, T) * r)

with ``B(t, T)`` the same as Vasicek and ``A(t, T)`` adjusted so that
``A(0, T) = P_market(0, T)``:

    A(t, T) = (P_mkt(0,T) / P_mkt(0,t)) * exp(B(t,T) * f(t) - 0.25 * (sigma B(t,T))^2 * B(0, 2t))

Discount-bond European options are priced via the Black formula with
the option-bond volatility taken from the yield-curve discount
factors (not the model-internal discount).

The C++ class subclasses both ``Vasicek`` (for A/B/discount/dynamics)
and ``TermStructureConsistentModel`` (for the curve handle). The
Python port preserves the double inheritance — both bases sit in the
MRO and Observable is the common Observer/Observable root.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.models.model import TermStructureConsistentModel
from pquantlib.models.parameter import (
    NullParameter,
    Parameter,
    ParameterImpl,
    TermStructureFittingParameter,
)
from pquantlib.models.shortrate.onefactor.one_factor_model import (
    ShortRateDynamics,
)
from pquantlib.models.shortrate.onefactor.vasicek import Vasicek
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class _HullWhiteFittingImpl(ParameterImpl):
    """Analytical Hull-White term-structure fitting parameter ``phi(t)``.

    # C++ parity: nested ``HullWhite::FittingParameter::Impl`` in
    # hullwhite.hpp:134-151 (v1.42.1).

    Reads the instantaneous forward rate from the curve and adds the
    closed-form correction ``0.5 * (sigma * B(t))^2`` where
    ``B(t) = (1 - exp(-a*t))/a`` (algebraic limit ``sigma * t`` for ``a≈0``).
    """

    __slots__ = ("_a", "_sigma", "_term_structure")

    def __init__(self, term_structure: YieldTermStructure, a: float, sigma: float) -> None:
        self._term_structure: YieldTermStructure = term_structure
        self._a: float = float(a)
        self._sigma: float = float(sigma)

    def value(self, params: npt.NDArray[np.float64], t: float) -> float:
        # C++ parity: hullwhite.hpp:139-146 — ``forwardRate(t, t,
        # Continuous, NoFrequency)`` reads the instantaneous forward
        # rate, then adds the variance-correction term.
        forward = self._term_structure.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        if self._a < math.sqrt(QL_EPSILON):
            temp = self._sigma * t
        else:
            temp = self._sigma * (1.0 - math.exp(-self._a * t)) / self._a
        return forward + 0.5 * temp * temp


class _HullWhiteFittingParameter(TermStructureFittingParameter):
    """Parameter wrapping ``_HullWhiteFittingImpl``.

    # C++ parity: ``class HullWhite::FittingParameter`` at hullwhite.hpp:131-157.

    Has zero free parameters; just exposes ``phi(t)`` via the Parameter
    call surface.
    """

    def __init__(self, term_structure: YieldTermStructure, a: float, sigma: float) -> None:
        super().__init__(_HullWhiteFittingImpl(term_structure, a, sigma))


class _HullWhiteDynamics(ShortRateDynamics):
    """Hull-White short-rate dynamics: OU(0) process + deterministic phi(t).

    # C++ parity: ``class HullWhite::Dynamics`` in hullwhite.hpp:110-122.

    The state ``x_t`` is a standard mean-reverting OU process with mean
    zero; the short rate is ``r_t = x_t + phi(t)`` where ``phi`` is the
    term-structure fitting parameter (held as a Parameter, evaluated
    on every call to variable / short_rate).
    """

    __slots__ = ("_fitting",)

    def __init__(self, fitting: Parameter, a: float, sigma: float) -> None:
        # C++ parity: hullwhite.hpp:112-115 — process is OU(a, sigma)
        # with default x0=0 and level=0.
        process = OrnsteinUhlenbeckProcess(speed=a, vol=sigma, x0=0.0, level=0.0)
        super().__init__(process)
        self._fitting: Parameter = fitting

    def variable(self, t: float, r: float) -> float:
        # C++ parity: hullwhite.hpp:117 — ``r - fitting(t)``.
        return r - self._fitting(t)

    def short_rate(self, t: float, variable: float) -> float:
        # C++ parity: hullwhite.hpp:118 — ``x + fitting(t)``.
        return variable + self._fitting(t)


class HullWhite(Vasicek, TermStructureConsistentModel):
    """Hull-White (extended Vasicek) short-rate model.

    # C++ parity: ``class HullWhite : public Vasicek, public
    # TermStructureConsistentModel`` in hullwhite.hpp:49-99 (v1.42.1).

    Inherits Vasicek's Parameter slots; nulls out the ``b`` and
    ``lambda`` slots (Hull-White has no constant long-run mean — that
    role is filled by the term-structure fitting parameter ``phi(t)``)
    and rebuilds the fitting parameter on every ``set_params`` call.
    """

    __slots__ = ("_phi",)

    def __init__(
        self,
        term_structure: YieldTermStructure,
        a: float = 0.1,
        sigma: float = 0.01,
    ) -> None:
        """Construct Hull-White consistent with the given curve.

        # C++ parity: hullwhite.cpp:31-41.

        The initial short rate is read from the curve (continuous,
        no-frequency forward rate at t=0). ``b`` and ``lambda`` are
        nulled out; only ``a`` and ``sigma`` remain as free parameters.
        """
        # C++ parity: hullwhite.cpp:31-34 — initialise Vasicek with
        # r0 = curve.forward_rate(0, 0, Continuous, NoFrequency).rate(),
        # a, b=0, sigma, lambda=0.
        r0 = term_structure.forward_rate(
            0.0, 0.0, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        Vasicek.__init__(self, r0=r0, a=a, b=0.0, sigma=sigma, lambda_=0.0)
        # Vasicek's super().__init__() chain reaches TermStructureConsistentModel
        # with the default term_structure=None; explicitly re-set the curve
        # afterwards. See the docstring of TermStructureConsistentModel for the
        # cooperative-super() rationale.
        self._term_structure = term_structure

        # C++ parity: hullwhite.cpp:36-37 — Hull-White has no constant
        # long-run mean (phi(t) fills that role) and no risk premium.
        # Replace the Vasicek parameters in-place with NullParameters.
        self.arguments[1] = NullParameter()
        self.arguments[3] = NullParameter()
        # Reset the aliases — they point at the Vasicek instances no longer.
        self._b_param = self.arguments[1]
        self._lambda_param = self.arguments[3]

        # The phi(t) fitting parameter is rebuilt every set_params via
        # generate_arguments(); call it once here to populate.
        self._phi: Parameter = _HullWhiteFittingParameter(
            term_structure, self.a(), self.sigma()
        )

        # C++ parity: hullwhite.cpp:40 — registerWith(termStructure).
        term_structure.register_with(self)

    # --- generated arguments hook ---------------------------------------

    def generate_arguments(self) -> None:
        """Refresh phi(t) using the current ``a`` and ``sigma``.

        # C++ parity: hullwhite.cpp:85-87.
        """
        self._phi = _HullWhiteFittingParameter(
            self.term_structure, self.a(), self.sigma()
        )

    # --- ShortRateModel surface -----------------------------------------

    def dynamics(self) -> ShortRateDynamics:
        # C++ parity: hullwhite.hpp:162-166 inline.
        return _HullWhiteDynamics(self._phi, self.a(), self.sigma())

    # --- A(t, T) override ----------------------------------------------

    def _a(self, t: float, t_maturity: float) -> float:
        """Hull-White ``A(t,T)`` — curve-consistent variant.

        # C++ parity: hullwhite.cpp:75-83.

        Different from Vasicek: rather than using the Vasicek formula
        with the (zero) ``b`` and ``lambda``, the Hull-White ``A``
        encodes curve consistency:

            A(t, T) = (D(T) / D(t)) * exp(B(t,T) * f(t) - 0.25 *
                       (sigma * B(t,T))^2 * B(0, 2t))
        """
        # B(t, T) is inherited from Vasicek (works with whatever ``a``).
        ts = self.term_structure
        d1 = ts.discount(t)
        d2 = ts.discount(t_maturity)
        forward = ts.forward_rate(
            t, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        bt = self._b(t, t_maturity)
        temp = self.sigma() * bt
        value = bt * forward - 0.25 * temp * temp * self._b(0.0, 2.0 * t)
        return math.exp(value) * d2 / d1

    # --- option pricing -------------------------------------------------

    def discount_bond_option(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """4-arg Hull-White discount-bond European option.

        # C++ parity: hullwhite.cpp:89-105.

        Uses ``term_structure.discount(...)`` for the forward/strike
        prices (rather than the model's implied discount — they
        coincide by construction but the C++ idiom uses the curve
        directly).
        """
        a = self.a()
        sigma = self.sigma()
        b = self._b(maturity, bond_maturity)
        if a < math.sqrt(QL_EPSILON):
            v = sigma * b * math.sqrt(maturity)
        else:
            v = sigma * b * math.sqrt(0.5 * (1.0 - math.exp(-2.0 * a * maturity)) / a)
        f = self.term_structure.discount(bond_maturity)
        k = self.term_structure.discount(maturity) * strike
        return black_formula(option_type, k, f, v)

    def discount_bond_option_3args(
        self,
        option_type: OptionType,
        strike: float,
        maturity: float,
        bond_start: float,
        bond_maturity: float,
    ) -> float:
        """5-arg Hull-White discount-bond European option.

        # C++ parity: hullwhite.cpp:107-131.

        Computes a richer volatility expression with the bond_start
        time; reduces to the 4-arg result when ``bond_start = maturity``.
        """
        a = self.a()
        sigma = self.sigma()
        if a < math.sqrt(QL_EPSILON):
            v = sigma * self._b(bond_start, bond_maturity) * math.sqrt(maturity)
        else:
            c = (
                math.exp(-2.0 * a * (bond_start - maturity))
                - math.exp(-2.0 * a * bond_start)
                - 2.0 * (
                    math.exp(-a * (bond_start + bond_maturity - 2.0 * maturity))
                    - math.exp(-a * (bond_start + bond_maturity))
                )
                + math.exp(-2.0 * a * (bond_maturity - maturity))
                - math.exp(-2.0 * a * bond_maturity)
            )
            # C++ parity (hullwhite.cpp:122-125): the above should be
            # positive but numerical errors can push it slightly below
            # zero; floor at 0 to avoid NaN under sqrt.
            v = sigma / (a * math.sqrt(2.0 * a)) * math.sqrt(max(c, 0.0))
        f = self.term_structure.discount(bond_maturity)
        k = self.term_structure.discount(bond_start) * strike
        return black_formula(option_type, k, f, v)

    # --- static utility -------------------------------------------------

    @staticmethod
    def convexity_bias(
        futures_price: float,
        t: float,
        t_horizon: float,
        sigma: float,
        a: float,
    ) -> float:
        """Convexity bias for futures-implied vs forward rates.

        # C++ parity: hullwhite.cpp:133-168.

        Reference: G. Kirikos, D. Novak, "Convexity Conundrums",
        Risk Magazine, March 1997.

        Args:
            futures_price: futures' market quoted price (in [0, 100]).
            t: deposit-DC year fraction to futures expiry.
            t_horizon: deposit-DC year fraction to maturity (``T >= t``).
            sigma: short-rate volatility.
            a: mean-reversion speed.

        Returns:
            The futures rate minus the equivalent forward rate.
        """
        qassert.require(
            futures_price >= 0.0,
            f"negative futures price ({futures_price}) not allowed",
        )
        qassert.require(t >= 0.0, f"negative t ({t}) not allowed")
        qassert.require(
            t_horizon >= t,
            f"T ({t_horizon}) must not be less than t ({t})",
        )
        qassert.require(sigma >= 0.0, f"negative sigma ({sigma}) not allowed")
        qassert.require(a >= 0.0, f"negative a ({a}) not allowed")

        def temp(x: float) -> float:
            return x if a < QL_EPSILON else (1.0 - math.exp(-a * x)) / a

        delta_t = t_horizon - t
        temp_delta_t = temp(delta_t)
        half_sigma_square = sigma * sigma / 2.0
        # lambda adjusts for the fact that the underlying is an interest rate.
        lam = temp(2.0 * t) * temp_delta_t
        temp_t = temp(t)
        phi = temp_t * temp_t
        z = half_sigma_square * (lam + phi)
        future_rate = (100.0 - futures_price) / 100.0
        if delta_t < QL_EPSILON:
            return z
        return (1.0 - math.exp(-z * temp_delta_t)) * (future_rate + 1.0 / delta_t)
