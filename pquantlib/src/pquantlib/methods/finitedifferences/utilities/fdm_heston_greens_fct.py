"""FdmHestonGreensFct — initial condition for the Heston forward equation.

# C++ parity: ql/methods/finitedifferences/utilities/fdmhestongreensfct.{hpp,cpp}
# (v1.43).

Seeds a 2-D ``(x = ln S, v)`` grid with an approximation to the Heston
transition density at a small horizon ``t``, so a forward (Fokker-Planck)
solve can start from it. Three approximations, in increasing order of cost:

* ``ZeroCorrelation`` — the product of a Gaussian in ``x`` and the exact
  square-root transition density in ``v``. Ignores ``rho``.
* ``Gaussian`` — a bivariate normal in ``(x, v)`` that does carry ``rho``.
* ``SemiAnalytical`` — ``HestonProcess.pdf``, i.e. the Broadie-Kaya
  characteristic function inverted numerically.

The result is then reweighted for whatever variable transformation the
companion ``FdmSquareRootFwdOp`` is using (``Log`` multiplies by ``v``,
``Power`` by ``v^(1 - 2 kappa theta / sigma^2)``, ``Plain`` leaves it alone).
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING, final

import numpy as np

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.operators.fdm_square_root_fwd_op import (
    TransformationType,
)
from pquantlib.methods.finitedifferences.utilities.square_root_process_rnd_calculator import (
    SquareRootProcessRNDCalculator,
)
from pquantlib.time.compounding import Compounding

if TYPE_CHECKING:
    from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
    from pquantlib.processes.heston_process import HestonProcess

_TWO_PI: float = 2.0 * math.pi
#: ``M_1_SQRTPI * M_SQRT1_2`` in C++'s mathconstants.hpp = 1/sqrt(2 pi).
_ONE_OVER_SQRT_TWO_PI: float = 1.0 / math.sqrt(_TWO_PI)


class FdmHestonGreensFctAlgorithm(IntEnum):
    """# C++ parity: ``FdmHestonGreensFct::Algorithm``."""

    ZeroCorrelation = 0
    Gaussian = 1
    SemiAnalytical = 2


@final
class FdmHestonGreensFct:
    """Green's-function seed for a Heston forward PDE solve.

    # C++ parity: ``class FdmHestonGreensFct``.
    """

    #: # C++ parity: the nested ``Algorithm`` enum.
    Algorithm = FdmHestonGreensFctAlgorithm

    __slots__ = ("_l0", "_mesher", "_process", "_trafo_type")

    def __init__(
        self,
        mesher: FdmMesher,
        process: HestonProcess,
        trafo_type: TransformationType,
        l0: float = 1.0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._process: HestonProcess = process
        self._trafo_type: TransformationType = trafo_type
        self._l0: float = l0

    def get(self, t: float, algorithm: FdmHestonGreensFctAlgorithm) -> Array:
        """Density on the grid at horizon ``t``.

        # C++ parity: ``FdmHestonGreensFct::get``.
        """
        p = self._process
        r = p.risk_free_rate().forward_rate(0.0, t, Compounding.Continuous).rate()
        q = p.dividend_yield().forward_rate(0.0, t, Compounding.Continuous).rate()

        s0 = p.s0().value()
        v0 = p.v0
        l0 = self._l0
        x0 = math.log(s0) + (r - q - 0.5 * v0 * l0 * l0) * t

        rho = p.rho
        theta = p.theta
        kappa = p.kappa
        sigma = p.sigma

        layout = self._mesher.layout()
        out = np.zeros(layout.size(), dtype=np.float64)

        sd_x = l0 * math.sqrt(v0 * t)
        sqrt_rnd = SquareRootProcessRNDCalculator(v0, kappa, theta, sigma)
        sd_v = sigma * math.sqrt(v0 * t)
        z0 = v0 + kappa * (theta - v0) * t
        power_exponent = 1.0 - 2.0 * kappa * theta / (sigma * sigma)

        for it in layout.iter():
            x = self._mesher.location(it, 0)
            v_raw = self._mesher.location(it, 1)
            v = math.exp(v_raw) if self._trafo_type == TransformationType.Log else v_raw

            if algorithm == FdmHestonGreensFctAlgorithm.ZeroCorrelation:
                p_x = _ONE_OVER_SQRT_TWO_PI / sd_x * math.exp(-0.5 * ((x - x0) / sd_x) ** 2)
                ret = sqrt_rnd.pdf(v, t) * p_x
            elif algorithm == FdmHestonGreensFctAlgorithm.SemiAnalytical:
                ret = p.pdf(x, v, t, 1e-4)
            elif algorithm == FdmHestonGreensFctAlgorithm.Gaussian:
                ret = (
                    1.0
                    / (_TWO_PI * sd_x * sd_v * math.sqrt(1.0 - rho * rho))
                    * math.exp(
                        -(
                            ((x - x0) / sd_x) ** 2
                            + ((v - z0) / sd_v) ** 2
                            - 2.0 * rho * (x - x0) * (v - z0) / (sd_x * sd_v)
                        )
                        / (2.0 * (1.0 - rho * rho))
                    )
                )
            else:
                raise LibraryException("unknown algorithm")

            if self._trafo_type == TransformationType.Log:
                ret *= v
            elif self._trafo_type == TransformationType.Power:
                ret *= math.pow(v, power_exponent)
            elif self._trafo_type != TransformationType.Plain:
                raise LibraryException("unknown transformation type")

            out[it.index] = ret

        return out


__all__ = ["FdmHestonGreensFct", "FdmHestonGreensFctAlgorithm"]
