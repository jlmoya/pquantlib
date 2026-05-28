"""Black-Karasinski log-normal short-rate model.

# C++ parity: ql/models/shortrate/onefactormodels/blackkarasinski.{hpp,cpp} (v1.42.1).

Implements the SDE for the *log* short rate:

    d ln r_t = (theta(t) - a ln r_t) dt + sigma dW_t

i.e. ``ln r`` is an Ornstein-Uhlenbeck process and the short rate
``r_t = exp(phi(t) + x_t)`` is strictly positive. ``a`` and ``sigma``
are constants; ``phi(t)`` is the deterministic time-dependent
term-structure-fitting parameter (no closed form — must be
numerically calibrated via the tree).

Unlike Vasicek/Hull-White, the Black-Karasinski short rate has no
closed-form discount bond, so ``discount_bond`` / ``discount_bond_option``
are not provided. Pricing is done through the trinomial tree
(``tree(grid)``), which numerically fits ``phi(t_i)`` at each grid
point by Brent-solving for the theta that makes the lattice reprice
the curve's discount factor.

The dynamics class follows the C++ layout exactly: state is a centred
OU process; ``variable(t, r) = ln(r) - phi(t)`` and
``short_rate(t, x) = exp(x + phi(t))``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.methods.lattices.trinomial_tree import TrinomialTree
from pquantlib.models.model import TermStructureConsistentModel
from pquantlib.models.parameter import (
    ConstantParameter,
    Parameter,
    TermStructureFittingParameter,
    TermStructureFittingParameterImpl,
)
from pquantlib.models.shortrate.onefactor.one_factor_model import (
    OneFactorModel,
    ShortRateDynamics,
)
from pquantlib.models.shortrate.short_rate_tree import ShortRateTree
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess

if TYPE_CHECKING:
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.time_grid import TimeGrid


class _BlackKarasinskiDynamics(ShortRateDynamics):
    """Black-Karasinski short-rate dynamics: OU state + ``r = exp(phi + x)``.

    # C++ parity: ``class BlackKarasinski::Dynamics`` in
    # blackkarasinski.hpp:75-94 (v1.42.1).
    """

    __slots__ = ("_fitting",)

    def __init__(self, fitting: Parameter, a: float, sigma: float) -> None:
        # # C++ parity: blackkarasinski.hpp:80-83 — process is
        # OrnsteinUhlenbeckProcess(a, sigma) (default x0=0, level=0).
        process = OrnsteinUhlenbeckProcess(speed=a, vol=sigma, x0=0.0, level=0.0)
        super().__init__(process)
        self._fitting: Parameter = fitting

    def variable(self, t: float, r: float) -> float:
        # # C++ parity: blackkarasinski.hpp:86 — ``ln(r) - fitting(t)``.
        return math.log(r) - self._fitting(t)

    def short_rate(self, t: float, variable: float) -> float:
        # # C++ parity: blackkarasinski.hpp:88 — ``exp(x + fitting(t))``.
        return math.exp(variable + self._fitting(t))


class BlackKarasinski(OneFactorModel, TermStructureConsistentModel):
    """Black-Karasinski log-normal short-rate model.

    # C++ parity: ``class BlackKarasinski : public OneFactorModel,
    #                                       public TermStructureConsistentModel``
    # in blackkarasinski.hpp:41-63 (v1.42.1).

    Two free parameters: ``a`` (positive), ``sigma`` (positive).
    ``phi(t)`` is the term-structure fitting parameter, numerically
    calibrated by ``tree(grid)`` rather than analytically.
    """

    __slots__ = ("_a_param", "_phi", "_sigma_param")

    def __init__(
        self,
        term_structure: YieldTermStructure,
        a: float = 0.1,
        sigma: float = 0.1,
    ) -> None:
        """Construct Black-Karasinski consistent with ``term_structure``.

        # C++ parity: blackkarasinski.cpp:60-68 — OneFactorModel(2)
        # ctor, sets a / sigma to PositiveConstraint ConstantParameters,
        # and phi to a TermStructureFittingParameter (numerical impl).
        """
        OneFactorModel.__init__(self, n_arguments=2)
        # # C++ parity: blackkarasinski.cpp:62-63 — a and sigma slots.
        self.arguments[0] = ConstantParameter(a, PositiveConstraint())
        self.arguments[1] = ConstantParameter(sigma, PositiveConstraint())
        # # C++ parity: blackkarasinski.cpp:64 — phi_ = TermStructureFittingParameter(termStructure).
        # This default-constructs a NumericalImpl (no closed-form fitter).
        self._phi: Parameter = TermStructureFittingParameter(term_structure)
        # Cache typed aliases.
        self._a_param: Parameter = self.arguments[0]
        self._sigma_param: Parameter = self.arguments[1]
        # Set the term-structure handle on the cooperative TS-consistent base.
        self._term_structure = term_structure
        # # C++ parity: blackkarasinski.cpp:65 — registerWith(termStructure).
        term_structure.register_with(self)

    # --- inspectors ------------------------------------------------------

    def a(self) -> float:
        """Mean-reversion speed.

        # C++ parity: ``BlackKarasinski::a`` (blackkarasinski.hpp:55).
        """
        return self._a_param(0.0)

    def sigma(self) -> float:
        """Log-rate volatility.

        # C++ parity: ``BlackKarasinski::sigma`` (blackkarasinski.hpp:56).
        """
        return self._sigma_param(0.0)

    # --- ShortRateModel surface ------------------------------------------

    def dynamics(self) -> ShortRateDynamics:
        """Return a fresh dynamics object — invoked by the tree fitter.

        # C++ parity: ``BlackKarasinski::dynamics`` (blackkarasinski.cpp:91-100).

        The C++ source calibrates ``phi`` by building a 50-step lattice
        before returning the dynamics; we keep the same idiom for
        parity (the cached ``_phi`` already has the right fitter
        type, so the call is cheap if no calibration is needed yet).
        """
        return _BlackKarasinskiDynamics(self._phi, self.a(), self.sigma())

    def tree(self, grid: TimeGrid) -> ShortRateTree:
        """Build a numerically-fitted trinomial tree for BK.

        # C++ parity: ``BlackKarasinski::tree`` (blackkarasinski.cpp:70-89).

        Overrides ``OneFactorModel.tree``: BK has no closed-form
        discount bond, so the tree must numerically calibrate ``phi``
        via Brent at each grid time.
        """
        dyn = self.dynamics()
        # C++ uses TrinomialTree(process, grid) — *not* the is_positive
        # branch despite r > 0, because the *state* x = ln r is not
        # constrained (only the short rate exp(x + phi) is).
        trinomial = TrinomialTree(dyn.process, grid)
        # Pull the NumericalImpl out of the held phi parameter so the
        # tree can call ``set`` / ``change`` / ``reset`` on it.
        impl = self._phi.impl
        assert isinstance(impl, TermStructureFittingParameterImpl), (
            "BlackKarasinski.phi must hold a TermStructureFittingParameterImpl"
        )
        return ShortRateTree(
            trinomial,
            dyn,
            grid,
            phi=impl,
            term_structure=self.term_structure,
        )


__all__ = ["BlackKarasinski"]
