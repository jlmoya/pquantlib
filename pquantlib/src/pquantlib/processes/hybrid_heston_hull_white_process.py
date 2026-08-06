"""HybridHestonHullWhiteProcess — Heston equity + Hull-White short rate.

# C++ parity: ql/processes/hybridhestonhullwhiteprocess.{hpp,cpp} (v1.43) —
# ``class HybridHestonHullWhiteProcess : public StochasticProcess``
# (hybridhestonhullwhiteprocess.hpp:42-82,
#  hybridhestonhullwhiteprocess.cpp:30-230).

A three-factor model: the two Heston factors (spot, variance) plus the
Hull-White short rate, correlated through a single extra parameter
``corrEquityShortRate``. The Hull-White leg is a ``HullWhiteForwardProcess``
(so the equity is priced under the T-forward measure) and a separate
``HullWhite`` MODEL instance is held purely to evaluate ``P(t, T)`` inside
:meth:`numeraire`.

Note what the class inherits and never fills in: no discretization is passed
to the base, so ``expectation`` / ``std_deviation`` / ``covariance`` are NOT
available — only ``drift`` / ``diffusion`` / ``apply`` / ``evolve`` /
``numeraire``. That is the C++ shape, not an omission here.

Two discretizations are offered and they take genuinely different branches
of :meth:`evolve`:

* ``BSMHullWhite`` (the default) integrates the equity variance over the step
  analytically (``v1``), derives a terminal correlation ``rhoT`` clamped to
  ``+/-maxRho``, and steps the short rate with a correlated normal;
* ``Euler`` uses the instantaneous ``eta*sqrt(dt)`` and the raw correlation.

Divergences from C++:

* # C++ parity divergence: C++ takes ``Handle<YieldTermStructure>`` inside
  the constituent processes and ``ext::shared_ptr`` for the processes
  themselves; this port threads the objects directly.
* ``update()`` refreshes ``endDiscount_`` and — deliberately — does NOT chain
  to ``StochasticProcess::update()``, so observers are not notified
  (hybridhestonhullwhiteprocess.cpp:228-230). That looks like a C++ bug but
  it is v1.43 behaviour and is reproduced rather than corrected.
* The ``HullWhite`` model import is done lazily inside ``__init__``:
  ``pquantlib.models.shortrate.onefactor.hull_white`` itself imports from
  ``pquantlib.processes``, so a module-level import here would close an
  import cycle.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.models.shortrate.onefactor.hull_white import HullWhite


class HybridHestonHullWhiteProcess(StochasticProcess):
    """Three-factor Heston / Hull-White hybrid process.

    # C++ parity: ``class HybridHestonHullWhiteProcess : public
    # StochasticProcess``.
    """

    class Discretization(IntEnum):
        """Evolution scheme.

        # C++ parity: ``HybridHestonHullWhiteProcess::Discretization``
        # (hybridhestonhullwhiteprocess.hpp:44) — ``{ Euler, BSMHullWhite }``,
        # so ``Euler == 0`` and ``BSMHullWhite == 1``.
        """

        Euler = 0
        BSMHullWhite = 1

    def __init__(
        self,
        heston_process: HestonProcess,
        hull_white_process: HullWhiteForwardProcess,
        corr_equity_short_rate: float,
        discretization: Discretization = Discretization.BSMHullWhite,
    ) -> None:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:30-54.
        super().__init__()
        # Lazy import: see the module docstring (import-cycle note).
        from pquantlib.models.shortrate.onefactor.hull_white import (  # noqa: PLC0415
            HullWhite as _HullWhite,
        )

        self._heston_process: HestonProcess = heston_process
        self._hull_white_process: HullWhiteForwardProcess = hull_white_process
        self._hull_white_model: HullWhite = _HullWhite(
            heston_process.risk_free_rate(),
            hull_white_process.a(),
            hull_white_process.sigma(),
        )
        self._corr_equity_short_rate: float = float(corr_equity_short_rate)
        # NOT ``_discretization``: that name is taken by the base
        # ``StochasticProcess``'s discretization slot, and shadowing it with
        # the enum would silently break ``expectation``/``covariance``.
        self._discretization_scheme: HybridHestonHullWhiteProcess.Discretization = (
            discretization
        )
        rho = heston_process.rho
        # "reserve for rounding errors" — hybridhestonhullwhiteprocess.cpp:42-43
        self._max_rho: float = math.sqrt(1.0 - rho * rho) - math.sqrt(QL_EPSILON)
        self._T: float = hull_white_process.get_forward_measure_time()
        self._end_discount: float = heston_process.risk_free_rate().discount(self._T)

        qassert.require(
            corr_equity_short_rate * corr_equity_short_rate + rho * rho <= 1.0,
            "correlation matrix is not positive definite",
        )
        qassert.require(
            hull_white_process.sigma() > 0.0,
            "positive vol of Hull White process is required",
        )

    # --- inspectors --------------------------------------------------------

    def heston_process(self) -> HestonProcess:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:209-212.
        return self._heston_process

    def hull_white_process(self) -> HullWhiteForwardProcess:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:214-217.
        return self._hull_white_process

    def eta(self) -> float:
        """The equity / short-rate correlation.

        # C++ parity: hybridhestonhullwhiteprocess.cpp:205-207 — the accessor
        # is named ``eta()`` but returns ``corrEquityShortRate_``, NOT the
        # local volatility ``eta`` used inside ``evolve``.
        """
        return self._corr_equity_short_rate

    def discretization(self) -> HybridHestonHullWhiteProcess.Discretization:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:219-222.
        return self._discretization_scheme

    # --- StochasticProcess interface --------------------------------------

    def size(self) -> int:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:56-58.
        return 3

    def initial_values(self) -> npt.NDArray[np.float64]:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:60-66.
        return np.array(
            [
                self._heston_process.s0().value(),
                self._heston_process.v0,
                self._hull_white_process.x0(),
            ],
            dtype=np.float64,
        )

    def drift(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:68-77.
        y0 = self._heston_process.drift(t, np.array([x[0], x[1]], dtype=np.float64))
        return np.array(
            [y0[0], y0[1], self._hull_white_process.drift_1d(t, float(x[2]))],
            dtype=np.float64,
        )

    def apply(
        self, x0: npt.NDArray[np.float64], dx: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:79-88.
        yt = self._heston_process.apply(
            np.array([x0[0], x0[1]], dtype=np.float64),
            np.array([dx[0], dx[1]], dtype=np.float64),
        )
        return np.array(
            [yt[0], yt[1], self._hull_white_process.apply_1d(float(x0[2]), float(dx[2]))],
            dtype=np.float64,
        )

    def diffusion(self, t: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """3x3 lower-triangular diffusion.

        # C++ parity: hybridhestonhullwhiteprocess.cpp:90-105. Row 2 is built
        # so that the row norm equals the Hull-White ``sigma`` while its
        # projection onto row 1 realises ``corrEquityShortRate``.
        """
        ret = np.zeros((3, 3), dtype=np.float64)
        m = self._heston_process.diffusion(t, np.array([x[0], x[1]], dtype=np.float64))
        ret[0, 0] = m[0, 0]
        ret[1, 0] = m[1, 0]
        ret[1, 1] = m[1, 1]

        sigma = self._hull_white_process.sigma()
        ret[2, 0] = self._corr_equity_short_rate * sigma
        ret[2, 1] = -ret[2, 0] * ret[1, 0] / ret[1, 1]
        ret[2, 2] = math.sqrt(sigma * sigma - ret[2, 1] * ret[2, 1] - ret[2, 0] * ret[2, 0])
        return ret

    def evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Advance (spot, variance, short rate) by ``dt``.

        # C++ parity: hybridhestonhullwhiteprocess.cpp:107-197. Transcribed
        # term by term, including the m1..m5 decomposition of the log-spot
        # drift and the ``eta = sqrt(v)`` full-truncation guard.
        """
        heston = self._heston_process
        hwp = self._hull_white_process

        r = float(x0[2])
        a = hwp.a()
        sigma = hwp.sigma()
        rho = self._corr_equity_short_rate
        xi = heston.rho
        v0 = float(x0[1])
        eta = math.sqrt(v0) if v0 > 0.0 else 0.0
        s = t0
        t = t0 + dt
        big_t = self._T
        dy = heston.dividend_yield().forward_rate(
            s, t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()

        df = math.log(heston.risk_free_rate().discount(t) / heston.risk_free_rate().discount(s))

        eat_big = math.exp(-a * big_t)
        eat = math.exp(-a * t)
        eas = math.exp(-a * s)
        iat = 1.0 / eat
        ias = 1.0 / eas

        m1 = -(dy + 0.5 * eta * eta) * dt - df
        m2 = -rho * sigma * eta / a * (dt - 1.0 / a * eat_big * (iat - ias))
        m3 = (r - hwp.alpha(s)) * hwp.B(s, t)
        m4 = (
            sigma
            * sigma
            / (2.0 * a * a)
            * (dt + 2.0 / a * (eat - eas) - 1.0 / (2.0 * a) * (eat * eat - eas * eas))
        )
        m5 = (
            -sigma
            * sigma
            / (a * a)
            * (
                dt
                - 1.0 / a * (1.0 - eat * ias)
                - 1.0 / (2.0 * a) * eat_big * (iat - 2.0 * ias + eat * ias * ias)
            )
        )
        mu = m1 + m2 + m3 + m4 + m5

        ret = np.empty(3, dtype=np.float64)

        eta2 = heston.sigma * eta
        nu = heston.kappa * (heston.theta - eta * eta)

        ret[1] = (
            v0
            + nu * dt
            + eta2
            * math.sqrt(dt)
            * (xi * float(dw[0]) + math.sqrt(1.0 - xi * xi) * float(dw[1]))
        )

        if (
            self._discretization_scheme
            is HybridHestonHullWhiteProcess.Discretization.BSMHullWhite
        ):
            v1 = (
                eta * eta * dt
                + sigma
                * sigma
                / (a * a)
                * (
                    dt
                    - 2.0 / a * (1.0 - eat * ias)
                    + 1.0 / (2.0 * a) * (1.0 - eat * eat * ias * ias)
                )
                + 2.0 * sigma * eta / a * rho * (dt - 1.0 / a * (1.0 - eat * ias))
            )
            v2 = hwp.variance_1d(t0, r, dt)
            v12 = (1.0 - eat * ias) * (sigma * eta / a * rho + sigma * sigma / (a * a)) - sigma * sigma / (
                2.0 * a * a
            ) * (1.0 - eat * eat * ias * ias)

            qassert.require(v1 > 0.0 and v2 > 0.0, "zero or negative variance given")

            # terminal rho must be between -maxRho and +maxRho
            rho_t = min(self._max_rho, max(-self._max_rho, v12 / math.sqrt(v1 * v2)))
            qassert.require(
                rho_t <= 1.0 and rho_t >= -1.0 and 1.0 - rho_t * rho_t / (1.0 - xi * xi) >= 0.0,
                "invalid terminal correlation",
            )

            dw_0 = float(dw[0])
            dw_2 = (
                rho_t * float(dw[0])
                - rho_t * xi / math.sqrt(1.0 - xi * xi) * float(dw[1])
                + math.sqrt(1.0 - rho_t * rho_t / (1.0 - xi * xi)) * float(dw[2])
            )

            ret[2] = hwp.evolve_1d(t0, r, dt, dw_2)

            vol = math.sqrt(v1) * dw_0
            ret[0] = float(x0[0]) * math.exp(mu + vol)
        elif (
            self._discretization_scheme
            is HybridHestonHullWhiteProcess.Discretization.Euler
        ):
            dw_2 = (
                rho * float(dw[0])
                - rho * xi / math.sqrt(1.0 - xi * xi) * float(dw[1])
                + math.sqrt(1.0 - rho * rho / (1.0 - xi * xi)) * float(dw[2])
            )

            ret[2] = hwp.evolve_1d(t0, r, dt, dw_2)

            vol = eta * math.sqrt(dt) * float(dw[0])
            ret[0] = float(x0[0]) * math.exp(mu + vol)
        else:
            qassert.fail("unknown discretization scheme")

        return ret

    def numeraire(self, t: float, x: npt.NDArray[np.float64]) -> float:
        """``P(t, T; r=x[2]) / P(0, T)``.

        # C++ parity: hybridhestonhullwhiteprocess.cpp:199-203.
        """
        return self._hull_white_model.discount_bond_scalar(t, self._T, float(x[2])) / (
            self._end_discount
        )

    def time(self, date: Date) -> float:
        # C++ parity: hybridhestonhullwhiteprocess.cpp:224-226.
        return self._heston_process.time(date)

    def update(self) -> None:
        """Refresh the terminal discount factor.

        # C++ parity: hybridhestonhullwhiteprocess.cpp:228-230 — note it does
        # NOT call ``StochasticProcess::update()``, so observers are not
        # notified. Reproduced deliberately; see the module docstring.
        """
        self._end_discount = self._heston_process.risk_free_rate().discount(self._T)


__all__ = ["HybridHestonHullWhiteProcess"]
