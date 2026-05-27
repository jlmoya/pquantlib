"""OneFactorAffineModel — closed-form A(t,T) and B(t,T).

# C++ parity: ql/models/shortrate/onefactormodel.hpp:117-145 (v1.42.1).

Implements the affine class of one-factor short-rate models where the
discount-bond price has the closed form

    P(t, T, r) = A(t, T) * exp(-B(t, T) * r)

Subclasses (Vasicek, HullWhite, CIR, etc.) implement the abstract
``a(t, T)`` and ``b(t, T)`` (Python idiomatic lowercase to avoid name
clashes with model parameters like ``a()`` for mean-reversion speed) —
returning the closed-form factors at maturity ``T`` given current
time ``t``.

The base implements ``discount_bond`` and ``discount`` (the implied
discount factor at time t) on top of those primitives; concrete
classes provide ``discount_bond_option`` themselves (no Jamshidian
decomposition is needed for the European bond option in the affine
case — the option pricing reduces to ``blackFormula`` once the bond
volatility is known).
"""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np
import numpy.typing as npt

from pquantlib.models.model import AffineModel
from pquantlib.models.shortrate.onefactor.one_factor_model import OneFactorModel


class OneFactorAffineModel(OneFactorModel, AffineModel):
    """Single-factor affine short-rate model.

    # C++ parity: ``class OneFactorAffineModel : public OneFactorModel,
    # public AffineModel`` in onefactormodel.hpp:126-145 (v1.42.1).

    Subclasses MUST implement ``_a(t, T)`` and ``_b(t, T)``. The base
    provides ``discount_bond(now, maturity, x)`` (scalar) and
    ``discount_bond(now, maturity, factors)`` (array), plus the
    ``discount(t)`` implied curve.

    Python note: ``A`` and ``B`` collide with mean-reversion / volatility
    accessor names (``a()``, ``b()`` on Vasicek). To keep both, we use
    ``_a(t, T)`` / ``_b(t, T)`` for the affine factors (protected
    convention) — matching C++'s ``protected`` access on the
    ``A(t, T)`` / ``B(t, T)`` virtuals.
    """

    def __init__(self, n_arguments: int) -> None:
        # C++ parity: onefactormodel.hpp:129-130 — forward to
        # OneFactorModel(n_arguments). AffineModel has no state.
        super().__init__(n_arguments)

    # --- protected affine primitives ------------------------------------

    @abstractmethod
    def _a(self, t: float, t_maturity: float) -> float:
        """Affine factor A(t, T) — multiplicative.

        # C++ parity: ``OneFactorAffineModel::A`` (protected virtual)
        # at onefactormodel.hpp:143.
        """
        ...

    @abstractmethod
    def _b(self, t: float, t_maturity: float) -> float:
        """Affine factor B(t, T) — exponential rate.

        # C++ parity: ``OneFactorAffineModel::B`` (protected virtual)
        # at onefactormodel.hpp:144.
        """
        ...

    # --- discount-bond surface ------------------------------------------

    def discount_bond_scalar(self, now: float, maturity: float, rate: float) -> float:
        """Scalar discount bond P(now, maturity, r).

        # C++ parity: ``OneFactorAffineModel::discountBond(Time, Time,
        # Rate)`` (the scalar overload) at onefactormodel.hpp:136-138.

        ``discount_bond(now, maturity, factors)`` dispatches into this
        with ``factors[0]``; we keep both surfaces for callers that
        already have a scalar in hand.
        """
        return self._a(now, maturity) * math.exp(-self._b(now, maturity) * rate)

    def discount_bond(
        self,
        now: float,
        maturity: float,
        factors: npt.NDArray[np.float64],
    ) -> float:
        """Vector-factor discount bond — defers to the scalar form.

        # C++ parity: ``OneFactorAffineModel::discountBond(Time, Time,
        # Array)`` at onefactormodel.hpp:132-134.
        """
        # C++ uses factors[0] — same here.
        return self.discount_bond_scalar(now, maturity, float(factors[0]))

    def discount(self, t: float) -> float:
        """Implied curve discount factor at ``t``.

        # C++ parity: ``OneFactorAffineModel::discount(Time)`` at
        # onefactormodel.cpp:97-101 — evaluates
        # ``discountBond(0, t, shortRate(0, process.x0()))``.
        """
        dyn = self.dynamics()
        x0 = dyn.process.x0()
        r0 = dyn.short_rate(0.0, x0)
        return self.discount_bond_scalar(0.0, t, r0)
