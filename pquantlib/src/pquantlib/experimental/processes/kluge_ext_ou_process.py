"""KlugeExtOUProcess — correlated Kluge + extended OU two-asset process.

# C++ parity: ql/experimental/processes/klugeextouprocess.{hpp,cpp}
# (v1.42.1).

Three-factor model with two correlated asset-spot processes
(electricity ``P`` modelled as Kluge ``ExtOUWithJumps``; gas ``G``
modelled as ``ExtendedOU``):

    P_t = exp(p_t + X_t + Y_t)    G_t = exp(g_t + U_t)
    dX_t = -a X_t dt + sigma_x dW_t^x
    dY_t = -b Y_t dt + J_t dN_t       omega(J) = eta * exp(-eta * J)
    dU_t = -k U_t dt + sigma_u dW_t^u
    rho = corr(dW_t^x, dW_t^u)

This W5-A port exposes only the accessors needed by
``FdmKlugeExtOUOp``:

* ``rho`` (correlation).
* ``get_kluge_process()`` — the embedded ExtOUWithJumps.
* ``get_ext_ou_process()`` — the embedded ExtendedOU.

The multi-D ``initialValues`` / ``drift`` / ``diffusion`` / ``evolve``
surface for Monte-Carlo sampling is deferred.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
)


@final
class KlugeExtOUProcess:
    """Two-asset Kluge + extended-OU correlated process.

    # C++ parity: ``class KlugeExtOUProcess : public StochasticProcess``
    # in klugeextouprocess.hpp:56.

    Parameters
    ----------
    rho
        Correlation between the two Brownian motions.
    kluge
        ExtOUWithJumps process for the electricity factor ``P``.
    ext_ou
        ExtendedOU process for the gas factor ``G``.
    """

    __slots__ = ("_ext_ou", "_kluge", "_rho")

    def __init__(
        self,
        rho: float,
        kluge: ExtOUWithJumpsProcess,
        ext_ou: ExtendedOrnsteinUhlenbeckProcess,
    ) -> None:
        qassert.require(
            -1.0 <= rho <= 1.0, f"correlation rho must be in [-1, 1]: {rho}"
        )
        self._rho: float = float(rho)
        self._kluge: ExtOUWithJumpsProcess = kluge
        self._ext_ou: ExtendedOrnsteinUhlenbeckProcess = ext_ou

    def rho(self) -> float:
        # C++ parity: rho().
        return self._rho

    def get_kluge_process(self) -> ExtOUWithJumpsProcess:
        # C++ parity: getKlugeProcess().
        return self._kluge

    def get_ext_ou_process(self) -> ExtendedOrnsteinUhlenbeckProcess:
        # C++ parity: getExtOUProcess().
        return self._ext_ou


__all__ = ["KlugeExtOUProcess"]
