"""GjrGarchModel — Glosten-Jagannathan-Runkle GARCH(1,1) calibrated model.

# C++ parity: ql/models/equity/gjrgarchmodel.{hpp,cpp} (v1.42.1).

Subclass of ``CalibratedModel`` that wraps a ``GjrGarchProcess`` and
exposes its parameters (omega, alpha, beta, gamma, lambda, v0) to the
calibration optimizer as a 6-element ``ConstantParameter`` vector.

Parameter index ordering (matches C++ ``arguments_`` exactly):

| Index | Name   | Constraint                       |
|-------|--------|----------------------------------|
| 0     | omega  | PositiveConstraint               |
| 1     | alpha  | BoundaryConstraint(0.0, 1.0)     |
| 2     | beta   | BoundaryConstraint(0.0, 1.0)     |
| 3     | gamma  | BoundaryConstraint(-1.0, 1.0)    |
| 4     | lambda | NoConstraint                     |
| 5     | v0     | PositiveConstraint               |

In addition, a composite ``VolatilityConstraint`` is layered on top of
the per-Parameter constraints to enforce ``beta + gamma >= 0`` —
required for variance positivity.

``generate_arguments`` rebuilds the underlying ``GjrGarchProcess`` from
the current optimizer params on each iteration.

Reference:
- C++ note in gjrgarchmodel.hpp: "calibration is not implemented for
  GJR-GARCH" — meaning the engine round-trips through `CalibratedModel`
  but the upstream calibration testbed doesn't exercise it. We
  preserve the parameter / constraint scaffolding so a calibration
  call would converge if the engine returns market values.

Divergences from C++:
- The ``VolatilityConstraint`` nested class in C++ wraps a
  ``Constraint::Impl::test`` that checks ``beta + gamma >= 0``. Python
  ports this as a free-standing class via composition (matches
  pquantlib's L4-C ``FellerConstraint`` pattern).
- ``CompositeConstraint`` does not exist in pquantlib; we layer the
  ``VolatilityConstraint`` by replacing the model's ``_constraint``
  with a freshly-allocated ``_AndConstraint``-style adapter.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Note: ``np`` is consumed inside ``npt.NDArray[np.float64]`` aliases below.
from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    Constraint,
    NoConstraint,
    PositiveConstraint,
)
from pquantlib.models.model import CalibratedModel
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.gjr_garch_process import GjrGarchProcess


class VolatilityConstraint(Constraint):
    """``beta + gamma >= 0`` constraint guarding variance positivity.

    # C++ parity: ``GJRGARCHModel::VolatilityConstraint::Impl`` in
    # gjrgarchmodel.cpp:26-40.
    """

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        beta = float(params[2])
        gamma = float(params[3])
        return beta + gamma >= 0.0


class _AndConstraint(Constraint):
    """Conjunction of two constraints: both must pass.

    Inlined here to mirror C++ ``CompositeConstraint(c1, c2)`` (which
    lives in ql/math/optimization/constraint.hpp). We don't have a
    public ``CompositeConstraint`` in pquantlib's math/optimization
    layer; the same shape is reproduced inline. The L4-A ``_AndConstraint``
    in pquantlib.models.model is `_`-prefixed and not exported, so
    we duplicate the trivial implementation here.
    """

    __slots__ = ("_c1", "_c2")

    def __init__(self, c1: Constraint, c2: Constraint) -> None:
        super().__init__()
        self._c1: Constraint = c1
        self._c2: Constraint = c2

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        return self._c1.test(params) and self._c2.test(params)


class GjrGarchModel(CalibratedModel):
    """GJR-GARCH(1,1) calibrated model.

    # C++ parity: ``class GJRGARCHModel : public CalibratedModel`` in
    # gjrgarchmodel.hpp:42-67 (v1.42.1).
    """

    __slots__ = ("_process",)

    _N_ARGUMENTS: int = 6

    def __init__(self, process: GjrGarchProcess) -> None:
        # C++ parity: gjrgarchmodel.cpp:43-66 — CalibratedModel(6) +
        # ConstantParameter assignments with their constraints.
        super().__init__(self._N_ARGUMENTS)
        self._process: GjrGarchProcess = process

        self._arguments[0] = ConstantParameter(process.omega, PositiveConstraint())
        self._arguments[1] = ConstantParameter(
            process.alpha, BoundaryConstraint(0.0, 1.0)
        )
        self._arguments[2] = ConstantParameter(
            process.beta, BoundaryConstraint(0.0, 1.0)
        )
        self._arguments[3] = ConstantParameter(
            process.gamma, BoundaryConstraint(-1.0, 1.0)
        )
        self._arguments[4] = ConstantParameter(process.lambda_, NoConstraint())
        self._arguments[5] = ConstantParameter(process.v0, PositiveConstraint())

        # C++ parity: gjrgarchmodel.cpp:58-59 — composite constraint
        # combining the per-Parameter private constraint with the
        # variance-positivity guard.
        self._constraint = _AndConstraint(self._constraint, VolatilityConstraint())

        # C++ parity: gjrgarchmodel.cpp:61 — propagate freshly-assigned
        # parameters back into the process.
        self.generate_arguments()

        # C++ parity: gjrgarchmodel.cpp:63-65 — register observer on
        # rate / dividend curves + spot quote.
        process.risk_free_rate().register_with(self)
        process.dividend_yield().register_with(self)
        process.s0().register_with(self)

    # --- parameter accessors --------------------------------------------

    def omega(self) -> float:
        """Variance baseline coefficient (params[0] at t=0).

        # C++ parity: ``GJRGARCHModel::omega`` in gjrgarchmodel.hpp:48.
        """
        return self._arguments[0](0.0)

    def alpha(self) -> float:
        """Innovation impact coefficient (params[1] at t=0).

        # C++ parity: ``GJRGARCHModel::alpha`` in gjrgarchmodel.hpp:50.
        """
        return self._arguments[1](0.0)

    def beta(self) -> float:
        """Variance autoregression coefficient (params[2] at t=0).

        # C++ parity: ``GJRGARCHModel::beta`` in gjrgarchmodel.hpp:52.
        """
        return self._arguments[2](0.0)

    def gamma(self) -> float:
        """Negative-innovation impact coefficient (params[3] at t=0).

        # C++ parity: ``GJRGARCHModel::gamma`` in gjrgarchmodel.hpp:54.
        """
        return self._arguments[3](0.0)

    def lambda_(self) -> float:
        """Market price of risk (params[4] at t=0).

        # C++ parity: ``GJRGARCHModel::lambda`` in gjrgarchmodel.hpp:56.

        Trailing underscore because ``lambda`` is a Python keyword.
        """
        return self._arguments[4](0.0)

    def v0(self) -> float:
        """Initial daily variance (params[5] at t=0).

        # C++ parity: ``GJRGARCHModel::v0`` in gjrgarchmodel.hpp:58.
        """
        return self._arguments[5](0.0)

    def process(self) -> GjrGarchProcess:
        """The underlying ``GjrGarchProcess``.

        # C++ parity: ``GJRGARCHModel::process`` in gjrgarchmodel.hpp:61.
        """
        return self._process

    # --- generateArguments ----------------------------------------------

    def generate_arguments(self) -> None:
        """Rebuild the underlying ``GjrGarchProcess`` with current params.

        # C++ parity: ``GJRGARCHModel::generateArguments`` in
        # gjrgarchmodel.cpp:68-76.

        Constructs a fresh ``GjrGarchProcess`` from the same yield
        curves + spot quote but with parameters drawn from
        ``arguments_``.
        """
        self._process = GjrGarchProcess(
            risk_free_rate=self._process.risk_free_rate(),
            dividend_yield=self._process.dividend_yield(),
            s0=self._process.s0(),
            v0=self.v0(),
            omega=self.omega(),
            alpha=self.alpha(),
            beta=self.beta(),
            gamma=self.gamma(),
            lambda_=self.lambda_(),
            days_per_year=self._process.days_per_year,
            discretization=self._process.discretization,
        )


__all__ = ["GjrGarchModel", "VolatilityConstraint"]
