"""VarianceGammaModel — CalibratedModel wrapping the VG structural params.

# C++ parity: ql/experimental/variancegamma/variancegammamodel.{hpp,cpp}
# (v1.42.1).

A thin :class:`CalibratedModel` over the three Variance-Gamma parameters:

* ``arguments[0]`` = sigma (PositiveConstraint),
* ``arguments[1]`` = nu (PositiveConstraint),
* ``arguments[2]`` = theta (NoConstraint).

``generate_arguments`` rebuilds the wrapped process from the current
parameter values (so observers see a process that always reflects the
latest calibrated params). Calibration of the VG model itself is not
implemented in C++ (no helper instruments) — the model is constructed
from a process and exposes the params as model arguments.

References
----------
Dilip B. Madan, Peter Carr, Eric C. Chang (1998), "The variance gamma
process and option pricing," European Finance Review, 2, 79-105.
"""

from __future__ import annotations

from typing import final

from pquantlib.experimental.variancegamma.variance_gamma_process import (
    VarianceGammaProcess,
)
from pquantlib.math.optimization.constraint import NoConstraint, PositiveConstraint
from pquantlib.models.model import CalibratedModel
from pquantlib.models.parameter import ConstantParameter


@final
class VarianceGammaModel(CalibratedModel):
    """Variance-Gamma calibrated model.

    # C++ parity: ``class VarianceGammaModel : public CalibratedModel`` in
    # variancegammamodel.hpp:41.
    """

    __slots__ = ("_process",)

    def __init__(self, process: VarianceGammaProcess) -> None:
        # C++ parity: CalibratedModel(3) + 3 ConstantParameters.
        super().__init__(3)
        self._process: VarianceGammaProcess = process
        self.arguments[0] = ConstantParameter(process.sigma(), PositiveConstraint())
        self.arguments[1] = ConstantParameter(process.nu(), PositiveConstraint())
        self.arguments[2] = ConstantParameter(process.theta(), NoConstraint())

        VarianceGammaModel.generate_arguments(self)

        process.risk_free_rate().register_with(self)
        process.dividend_yield().register_with(self)
        process.s0().register_with(self)

    # --- parameter accessors -------------------------------------------

    def sigma(self) -> float:
        # C++ parity: arguments_[0](0.0).
        return self.arguments[0](0.0)

    def nu(self) -> float:
        # C++ parity: arguments_[1](0.0).
        return self.arguments[1](0.0)

    def theta(self) -> float:
        # C++ parity: arguments_[2](0.0).
        return self.arguments[2](0.0)

    def process(self) -> VarianceGammaProcess:
        # C++ parity: process() accessor.
        return self._process

    # --- CalibratedModel hook ------------------------------------------

    def generate_arguments(self) -> None:
        """Rebuild the wrapped process from the current params.

        # C++ parity: ``VarianceGammaModel::generateArguments`` —
        # reconstructs the VarianceGammaProcess with (sigma, nu, theta).
        """
        self._process = VarianceGammaProcess(
            self._process.s0(),
            self._process.dividend_yield(),
            self._process.risk_free_rate(),
            self.sigma(),
            self.nu(),
            self.theta(),
        )


__all__ = ["VarianceGammaModel"]
