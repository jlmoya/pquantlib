"""HybridHestonHullWhiteProcess — 3-factor Heston + Hull-White process.

# C++ parity: ql/processes/hybridhestonhullwhiteprocess.{hpp,cpp} (v1.43) —
# ``class HybridHestonHullWhiteProcess : public StochasticProcess``.

Joins a 2-factor :class:`~pquantlib.processes.heston_process.HestonProcess`
(spot + variance) with a 1-factor
:class:`~pquantlib.processes.hull_white_forward_process.HullWhiteForwardProcess`
(short rate) under a single correlation ``corrEquityShortRate`` between the
equity and short-rate Brownians, giving the state vector ``(S, v, r)``.

``evolve`` has two discretizations and they give different numbers:

``BSMHullWhite`` (the C++ default)
    integrates the equity leg with the *terminal* variance
    ``v1 = eta^2 dt + (HW variance contribution) + (cross term)`` and a
    terminal correlation ``rhoT`` clamped into ``[-maxRho, maxRho]``.

``Euler``
    uses the instantaneous ``eta sqrt(dt) dw[0]`` and the raw correlation.

Both share the same drift ``mu = m1 + m2 + m3 + m4 + m5``, whose five terms are
transcribed verbatim from C++ so the ordering of the floating-point operations
matches.

The C++ class carries a ``\\bug This class was not tested enough to guarantee
its functionality... work in progress`` note. That is upstream's own caveat and
is preserved here rather than silently cleaned up.
"""

from __future__ import annotations

import math
import sys
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException

if TYPE_CHECKING:
    from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

_QL_EPSILON = sys.float_info.epsilon


class Discretization(IntEnum):
    """Discretization tags for :meth:`HybridHestonHullWhiteProcess.evolve`.

    # C++ parity: ``enum Discretization { Euler, BSMHullWhite }``
    # (hybridhestonhullwhiteprocess.hpp:44); the values match the C++
    # declaration order.

    An ``IntEnum`` rather than a plain namespace of ints: callers in this port
    pass both the enum member and a bare ``int``, and only ``IntEnum`` types
    correctly for both. C++'s enumerators are unscoped and implicitly convert
    to int, so this is also the closer reading of the declaration.
    """

    Euler = 0
    BSMHullWhite = 1


class HybridHestonHullWhiteProcess(StochasticProcess):
    """Three-factor Heston Hull-White process over ``(S, v, r)``.

    # C++ parity: ``HybridHestonHullWhiteProcess``
    # (hybridhestonhullwhiteprocess.hpp:42-83).
    """

    # C++ scopes this enum inside the class (hybridhestonhullwhiteprocess.hpp:44),
    # and callers in this port reach it both ways — nested as
    # ``HybridHestonHullWhiteProcess.Discretization`` and as a module-level
    # import. Binding the one enum here gives both paths the SAME object,
    # rather than leaving two definitions that could drift apart.
    Discretization = Discretization

    __slots__ = (
        "_corr_equity_short_rate",
        "_disc_scheme",
        "_end_discount",
        "_heston_process",
        "_hull_white_model",
        "_hull_white_process",
        "_max_rho",
        "_t_forward",
    )

    def __init__(
        self,
        heston_process: HestonProcess,
        hull_white_process: HullWhiteForwardProcess,
        corr_equity_short_rate: float,
        discretization: Discretization = Discretization.BSMHullWhite,
    ) -> None:
        """# C++ parity: the constructor (hybridhestonhullwhiteprocess.cpp:30-56)."""
        super().__init__()
        self._heston_process: HestonProcess = heston_process
        self._hull_white_process: HullWhiteForwardProcess = hull_white_process
        # The model is what computes P(t, T); C++ builds it from the Heston
        # curve and the Hull-White process parameters, not from the process.
        # Imported here rather than at module scope: hull_white transitively
        # imports back into processes, and a module-level import makes the two
        # partially-initialised. Only the constructor needs it at runtime.
        from pquantlib.models.shortrate.onefactor.hull_white import (  # noqa: PLC0415
            HullWhite,
        )

        self._hull_white_model: HullWhite = HullWhite(
            heston_process.risk_free_rate(),
            hull_white_process.a(),
            hull_white_process.sigma(),
        )
        self._corr_equity_short_rate: float = corr_equity_short_rate
        # Named ``_disc_scheme`` rather than ``_discretization`` because
        # ``StochasticProcess`` already owns the latter for its discretization
        # *object*; reusing the name would replace it with an int.
        self._disc_scheme: Discretization = discretization
        rho = heston_process.rho
        # "reserve for rounding errors" -- C++ comment.
        self._max_rho: float = math.sqrt(1 - rho * rho) - math.sqrt(_QL_EPSILON)
        self._t_forward: float = hull_white_process.get_forward_measure_time()
        self._end_discount: float = heston_process.risk_free_rate().discount(
            self._t_forward
        )

        qassert.require(
            corr_equity_short_rate * corr_equity_short_rate + rho * rho <= 1.0,
            "correlation matrix is not positive definite",
        )
        qassert.require(
            hull_white_process.sigma() > 0.0,
            "positive vol of Hull White process is required",
        )

    # --- inspectors -------------------------------------------------------

    def size(self) -> int:
        """# C++ parity: ``size`` (hybridhestonhullwhiteprocess.cpp:58-60)."""
        return 3

    def factors(self) -> int:
        """Three independent Brownians.

        # C++ parity: inherited ``StochasticProcess::factors`` = ``size()``
        # for this process, since ``diffusion`` is 3x3.
        """
        return 3

    def heston_process(self) -> HestonProcess:
        """# C++ parity: ``hestonProcess()`` (hybridhestonhullwhiteprocess.cpp:212-215)."""
        return self._heston_process

    def hull_white_process(self) -> HullWhiteForwardProcess:
        """# C++ parity: ``hullWhiteProcess()`` (hybridhestonhullwhiteprocess.cpp:217-220)."""
        return self._hull_white_process

    def eta(self) -> float:
        """The equity/short-rate correlation.

        # C++ parity: ``eta()`` (hybridhestonhullwhiteprocess.cpp:208-210) --
        # note the name: it returns ``corrEquityShortRate_``, not a volatility.
        """
        return self._corr_equity_short_rate

    def discretization(self) -> Discretization:
        """# C++ parity: ``discretization()`` (hybridhestonhullwhiteprocess.cpp:222-225)."""
        return self._disc_scheme

    def time(self, date: Date) -> float:
        """# C++ parity: ``time`` (hybridhestonhullwhiteprocess.cpp:227-229)."""
        return self._heston_process.time(date)

    def update(self) -> None:
        """# C++ parity: ``update`` (hybridhestonhullwhiteprocess.cpp:231-233)."""
        self._end_discount = self._heston_process.risk_free_rate().discount(
            self._t_forward
        )

    # --- process surface --------------------------------------------------

    def initial_values(self) -> npt.NDArray[np.float64]:
        """# C++ parity: ``initialValues`` (hybridhestonhullwhiteprocess.cpp:62-68)."""
        heston = self._heston_process
        return np.array(
            [heston.s0().value(), heston.v0, self._hull_white_process.x0()],
            dtype=np.float64,
        )

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """# C++ parity: ``drift`` (hybridhestonhullwhiteprocess.cpp:70-79)."""
        y0 = self._heston_process.drift(t, np.array([x[0], x[1]], dtype=np.float64))
        return np.array(
            [y0[0], y0[1], self._hull_white_process.drift_1d(t, float(x[2]))],
            dtype=np.float64,
        )

    def apply(
        self, x0: npt.NDArray[np.float64], dx: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """# C++ parity: ``apply`` (hybridhestonhullwhiteprocess.cpp:81-90)."""
        yt = self._heston_process.apply(
            np.array([x0[0], x0[1]], dtype=np.float64),
            np.array([dx[0], dx[1]], dtype=np.float64),
        )
        return np.array(
            [
                yt[0],
                yt[1],
                self._hull_white_process.apply_1d(float(x0[2]), float(dx[2])),
            ],
            dtype=np.float64,
        )

    def diffusion(
        self, t: float, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """# C++ parity: ``diffusion`` (hybridhestonhullwhiteprocess.cpp:92-109).

        The third row is built so that row 2 has total variance ``sigma^2`` and
        correlation ``corrEquityShortRate`` with the equity Brownian: entry
        ``[2][1]`` cancels the Heston cross-term and ``[2][2]`` takes up the
        remaining variance.
        """
        ret = np.zeros((3, 3), dtype=np.float64)
        m = self._heston_process.diffusion(
            t, np.array([x[0], x[1]], dtype=np.float64)
        )
        ret[0][0] = m[0][0]
        ret[1][0] = m[1][0]
        ret[1][1] = m[1][1]

        sigma = self._hull_white_process.sigma()
        ret[2][0] = self._corr_equity_short_rate * sigma
        ret[2][1] = -ret[2][0] * ret[1][0] / ret[1][1]
        ret[2][2] = math.sqrt(
            sigma * sigma - ret[2][1] * ret[2][1] - ret[2][0] * ret[2][0]
        )
        return ret

    def evolve(  # noqa: PLR0915 — verbatim transcription of the C++ formula
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """One step of ``(S, v, r)``.

        # C++ parity: ``evolve`` (hybridhestonhullwhiteprocess.cpp:111-196).
        """
        heston = self._heston_process
        hw = self._hull_white_process

        r = float(x0[2])
        a = hw.a()
        sigma = hw.sigma()
        rho = self._corr_equity_short_rate
        xi = heston.rho
        v0 = float(x0[1])
        eta = math.sqrt(v0) if v0 > 0.0 else 0.0
        s = t0
        t = t0 + dt
        t_fwd = self._t_forward

        dy = heston.dividend_yield().forward_rate(
            s, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()
        df = math.log(
            heston.risk_free_rate().discount(t) / heston.risk_free_rate().discount(s)
        )

        eat_fwd = math.exp(-a * t_fwd)
        eat = math.exp(-a * t)
        eas = math.exp(-a * s)
        iat = 1.0 / eat
        ias = 1.0 / eas

        m1 = -(dy + 0.5 * eta * eta) * dt - df
        m2 = -rho * sigma * eta / a * (dt - 1 / a * eat_fwd * (iat - ias))
        m3 = (r - hw.alpha(s)) * hw.B(s, t)
        m4 = (
            sigma
            * sigma
            / (2 * a * a)
            * (dt + 2 / a * (eat - eas) - 1 / (2 * a) * (eat * eat - eas * eas))
        )
        m5 = (
            -sigma
            * sigma
            / (a * a)
            * (
                dt
                - 1 / a * (1 - eat * ias)
                - 1 / (2 * a) * eat_fwd * (iat - 2 * ias + eat * ias * ias)
            )
        )
        mu = m1 + m2 + m3 + m4 + m5

        ret = np.empty(3, dtype=np.float64)
        eta2 = heston.sigma * eta
        nu = heston.kappa * (heston.theta - eta * eta)
        dw0 = float(dw[0])
        dw1 = float(dw[1])
        dw2_raw = float(dw[2])

        ret[1] = (
            v0
            + nu * dt
            + eta2 * math.sqrt(dt) * (xi * dw0 + math.sqrt(1 - xi * xi) * dw1)
        )

        if self._disc_scheme == Discretization.BSMHullWhite:
            v1 = (
                eta * eta * dt
                + sigma
                * sigma
                / (a * a)
                * (
                    dt
                    - 2 / a * (1 - eat * ias)
                    + 1 / (2 * a) * (1 - eat * eat * ias * ias)
                )
                + 2 * sigma * eta / a * rho * (dt - 1 / a * (1 - eat * ias))
            )
            v2 = hw.variance_1d(t0, r, dt)
            v12 = (1 - eat * ias) * (
                sigma * eta / a * rho + sigma * sigma / (a * a)
            ) - sigma * sigma / (2 * a * a) * (1 - eat * eat * ias * ias)

            qassert.require(v1 > 0.0 and v2 > 0.0, "zero or negative variance given")

            # terminal rho must be between -maxRho and +maxRho
            rho_t = min(self._max_rho, max(-self._max_rho, v12 / math.sqrt(v1 * v2)))
            qassert.require(
                rho_t <= 1.0
                and rho_t >= -1.0
                and 1 - rho_t * rho_t / (1 - xi * xi) >= 0.0,
                "invalid terminal correlation",
            )

            dw_2 = (
                rho_t * dw0
                - rho_t * xi / math.sqrt(1 - xi * xi) * dw1
                + math.sqrt(1 - rho_t * rho_t / (1 - xi * xi)) * dw2_raw
            )
            ret[2] = hw.evolve_1d(t0, r, dt, dw_2)
            vol = math.sqrt(v1) * dw0
            ret[0] = float(x0[0]) * math.exp(mu + vol)
        elif self._disc_scheme == Discretization.Euler:
            dw_2 = (
                rho * dw0
                - rho * xi / math.sqrt(1 - xi * xi) * dw1
                + math.sqrt(1 - rho * rho / (1 - xi * xi)) * dw2_raw
            )
            ret[2] = hw.evolve_1d(t0, r, dt, dw_2)
            vol = eta * math.sqrt(dt) * dw0
            ret[0] = float(x0[0]) * math.exp(mu + vol)
        else:
            # C++ parity: ``QL_FAIL("unknown discretization scheme")``.
            raise LibraryException("unknown discretization scheme")

        return ret

    def numeraire(self, t: float, x: npt.NDArray[np.float64]) -> float:
        """Forward-measure numeraire ``P(t, T, r) / P(0, T)``.

        # C++ parity: ``numeraire`` (hybridhestonhullwhiteprocess.cpp:198-206).
        """
        return (
            self._hull_white_model.discount_bond_scalar(t, self._t_forward, float(x[2]))
            / self._end_discount
        )

__all__ = ["Discretization", "HybridHestonHullWhiteProcess"]
