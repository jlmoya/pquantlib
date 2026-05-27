"""HestonModel — Heston stochastic-volatility model for equity options.

# C++ parity: ql/models/equity/hestonmodel.{hpp,cpp} (v1.42.1).

Subclass of ``CalibratedModel`` that wraps a ``HestonProcess`` and
exposes its parameters (theta, kappa, sigma, rho, v0) to the
calibration optimizer as a 5-element ``ConstantParameter`` vector.

Parameter index ordering (matches C++ ``arguments_`` exactly):

| Index | Name  | Constraint                       |
|-------|-------|----------------------------------|
| 0     | theta | PositiveConstraint               |
| 1     | kappa | PositiveConstraint               |
| 2     | sigma | PositiveConstraint               |
| 3     | rho   | BoundaryConstraint(-1.0, 1.0)    |
| 4     | v0    | PositiveConstraint               |

``generate_arguments`` rebuilds the underlying ``HestonProcess`` from
the optimizer's current parameter slice on each iteration — this is
how the model "republishes" the new dynamics to the engine, which
holds a reference to ``model.process()`` and recomputes the option
price.

Divergences from C++:

* ``ext::shared_ptr<HestonProcess>`` collapses to a plain reference;
  Python's GC handles ownership.
* ``HestonModel::FellerConstraint`` (a nested ``Constraint`` checking
  ``sigma^2 < 2*kappa*theta``) is provided as a free-standing class
  here for the same callsite.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    Constraint,
    PositiveConstraint,
)
from pquantlib.models.model import CalibratedModel
from pquantlib.models.parameter import ConstantParameter
from pquantlib.processes.heston_process import HestonProcess


class FellerConstraint(Constraint):
    """Feller-condition constraint: ``sigma^2 < 2 * kappa * theta``.

    # C++ parity: ``HestonModel::FellerConstraint`` in
    # ql/models/equity/hestonmodel.hpp:66-82 (v1.42.1).

    Tests the Feller condition on a ``(theta, kappa, sigma, ...)``
    parameter vector. Used as the *optional* extra constraint passed
    to ``CalibratedModel::calibrate``; not enforced by default
    (calibration sometimes needs to leave the Feller region to fit
    the smile).
    """

    def test(self, params: npt.NDArray[np.float64]) -> bool:
        theta = float(params[0])
        kappa = float(params[1])
        sigma = float(params[2])
        return sigma >= 0.0 and sigma * sigma < 2.0 * kappa * theta


class HestonModel(CalibratedModel):
    """Heston stochastic-volatility model.

    # C++ parity: ``class HestonModel : public CalibratedModel`` in
    # ql/models/equity/hestonmodel.hpp:42-64 (v1.42.1).
    """

    # Reserve a slot for the process so subclasses (BatesModel) can
    # extend ``arguments_`` without losing the reference.
    __slots__ = ("_process",)

    # The base ``arguments_`` size for HestonModel — overridden by
    # BatesModel to 8.
    _N_ARGUMENTS: int = 5

    def __init__(self, process: HestonProcess) -> None:
        # C++ parity: hestonmodel.cpp:26-27 — CalibratedModel(5) +
        # process_ initialization. Then individual ConstantParameter
        # assignments with their constraints.
        super().__init__(self._N_ARGUMENTS)
        self._process: HestonProcess = process

        self._arguments[0] = ConstantParameter(process.theta, PositiveConstraint())
        self._arguments[1] = ConstantParameter(process.kappa, PositiveConstraint())
        self._arguments[2] = ConstantParameter(process.sigma, PositiveConstraint())
        self._arguments[3] = ConstantParameter(process.rho, BoundaryConstraint(-1.0, 1.0))
        self._arguments[4] = ConstantParameter(process.v0, PositiveConstraint())

        # Note: ``_PrivateConstraint`` constructed in CalibratedModel.__init__
        # holds a reference to ``self._arguments`` (the same list mutated
        # above), so the freshly-assigned ConstantParameter slots are
        # automatically picked up — no rebuild required.

        # C++ parity: hestonmodel.cpp:38 — propagate freshly-assigned
        # parameters back into the process by reconstructing it. (The
        # initial construction trivially matches the input process so
        # this is a no-op on the first call.)
        self.generate_arguments()

        # C++ parity: hestonmodel.cpp:40-42 — register observer on
        # rate/dividend curves + spot quote. The model becomes a
        # dependent of any market-data change.
        process.risk_free_rate().register_with(self)
        process.dividend_yield().register_with(self)
        process.s0().register_with(self)

    # --- parameter accessors --------------------------------------------

    def theta(self) -> float:
        """Long-term variance level (params[0] evaluated at t=0).

        # C++ parity: ``HestonModel::theta`` in hestonmodel.hpp:47.
        """
        return self._arguments[0](0.0)

    def kappa(self) -> float:
        """Mean-reversion speed (params[1] evaluated at t=0).

        # C++ parity: ``HestonModel::kappa`` in hestonmodel.hpp:49.
        """
        return self._arguments[1](0.0)

    def sigma(self) -> float:
        """Volatility of variance (params[2] evaluated at t=0).

        # C++ parity: ``HestonModel::sigma`` in hestonmodel.hpp:51.
        """
        return self._arguments[2](0.0)

    def rho(self) -> float:
        """Spot/vol correlation (params[3] evaluated at t=0).

        # C++ parity: ``HestonModel::rho`` in hestonmodel.hpp:53.
        """
        return self._arguments[3](0.0)

    def v0(self) -> float:
        """Initial variance (params[4] evaluated at t=0).

        # C++ parity: ``HestonModel::v0`` in hestonmodel.hpp:55.
        """
        return self._arguments[4](0.0)

    def process(self) -> HestonProcess:
        """The underlying ``HestonProcess`` (updated on each calibration step).

        # C++ parity: ``HestonModel::process`` in hestonmodel.hpp:58.
        """
        return self._process

    # --- generateArguments ----------------------------------------------

    def generate_arguments(self) -> None:
        """Rebuild the underlying ``HestonProcess`` with current params.

        # C++ parity: ``HestonModel::generateArguments`` in
        # hestonmodel.cpp:45-51.

        Constructs a fresh ``HestonProcess`` from the same yield curves
        + spot quote but with parameters drawn from ``arguments_``. The
        new process is published to any engine holding a reference via
        ``self._process``.
        """
        self._process = HestonProcess(
            risk_free_rate=self._process.risk_free_rate(),
            dividend_yield=self._process.dividend_yield(),
            s0=self._process.s0(),
            v0=self.v0(),
            kappa=self.kappa(),
            theta=self.theta(),
            sigma=self.sigma(),
            rho=self.rho(),
        )


__all__ = ["FellerConstraint", "HestonModel"]
