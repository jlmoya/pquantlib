"""LocalVolRNDCalculator — terminal density implied by a local-volatility surface.

# C++ parity: ql/methods/finitedifferences/utilities/localvolrndcalculator.{hpp,cpp}
# (v1.43).

Solves the Fokker-Planck (forward) equation for ``x = ln S`` under a local
volatility surface, on a per-step ``Concentrating1dMesher`` that is *widened
and re-interpolated* whenever probability mass reaches the edge of the grid.
``pdf`` then interpolates in time between the two bracketing grid slices;
``cdf`` integrates ``pdf`` adaptively, extending the integration bound until
the density there is negligible; ``invcdf`` warm-starts Brent from the mean of
the nearest slice.

The class is a ``LazyObject``: everything above happens in
``_perform_calculations`` on first use, and again after any observed input
changes.
"""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING, final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.integrals.discrete_integrals import DiscreteSimpsonIntegral
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.predefined_1d_mesher import (
    Predefined1dMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_local_vol_fwd_op import (
    FdmLocalVolFwdOp,
)
from pquantlib.methods.finitedifferences.schemes.douglas_scheme import DouglasScheme
from pquantlib.methods.finitedifferences.utilities.risk_neutral_density_calculator import (
    InvCDFHelper,
    RiskNeutralDensityCalculator,
)
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
        LocalVolTermStructure,
    )
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure

_INV_CUM_NORMAL = InverseCumulativeNormal()
_SIMPSON = DiscreteSimpsonIntegral()


@final
class LocalVolRNDCalculator(RiskNeutralDensityCalculator, LazyObject):
    """Terminal risk-neutral density under a local-volatility surface.

    # C++ parity: ``class LocalVolRNDCalculator : public
    # RiskNeutralDensityCalculator, public LazyObject``.
    """

    def __init__(
        self,
        spot: Quote,
        r_ts: YieldTermStructure,
        q_ts: YieldTermStructure,
        local_vol: LocalVolTermStructure,
        x_grid: int = 101,
        t_grid: int = 51,
        x0_density: float = 0.1,
        local_vol_prob_eps: float = 1e-6,
        max_iter: int = 10000,
        gaussian_step_size: float | None = None,
        time_grid: TimeGrid | None = None,
    ) -> None:
        """Build from ``t_grid`` steps, or from an explicit ``time_grid``.

        # C++ parity: the two constructors. C++ overloads on whether a
        # ``TimeGrid`` is supplied; Python folds that into one optional
        # argument, and ``t_grid`` is then derived as ``timeGrid->size() - 1``
        # exactly as the second C++ constructor does.
        """
        LazyObject.__init__(self)
        self._x_grid: int = x_grid
        self._x0_density: float = x0_density
        self._local_vol_prob_eps: float = local_vol_prob_eps
        self._max_iter: int = max_iter
        #: C++ default is ``-Null<Time>()``, i.e. a large negative sentinel that
        #: fails the ``> 0.0`` test; ``None`` is the Python analogue.
        self._gaussian_step_size: float | None = gaussian_step_size
        self._spot: Quote = spot
        self._local_vol: LocalVolTermStructure = local_vol
        self._r_ts: YieldTermStructure = r_ts
        self._q_ts: YieldTermStructure = q_ts

        if time_grid is not None:
            self._time_grid: TimeGrid = time_grid
            self._t_grid: int = time_grid.size() - 1
        else:
            self._time_grid = TimeGrid.regular(local_vol.max_time(), t_grid)
            self._t_grid = t_grid

        self._xm: list[Fdm1dMesher] = [Predefined1dMesher([0.0, 1.0])] * self._t_grid
        self._pm: Array = np.zeros((self._t_grid, self._x_grid), dtype=np.float64)
        self._rescale_time_steps: list[int] = []
        self._p_fct: list[CubicNaturalSpline] = []

        for observable in (spot, r_ts, q_ts, local_vol):
            observable.register_with(self)

    # --- accessors -------------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """# C++ parity: ``LocalVolRNDCalculator::timeGrid``."""
        return self._time_grid

    def mesher(self, t: float) -> Fdm1dMesher:
        """# C++ parity: ``LocalVolRNDCalculator::mesher``."""
        self.calculate()
        idx = self._time_grid.index(t)
        qassert.require(idx <= len(self._xm), f"inconsistent time {t} given")
        if idx > 0:
            return self._xm[idx - 1]
        return Predefined1dMesher([math.log(self._spot.value())] * self._x_grid)

    def rescale_time_steps(self) -> list[int]:
        """# C++ parity: ``LocalVolRNDCalculator::rescaleTimeSteps``."""
        self.calculate()
        return list(self._rescale_time_steps)

    # --- RiskNeutralDensityCalculator ------------------------------------

    def pdf(self, x: float, t: float, /) -> float:
        """# C++ parity: ``LocalVolRNDCalculator::pdf``."""
        self.calculate()
        qassert.require(t > 0, "positive time expected")
        qassert.require(
            t <= self._time_grid.back(), "given time exceeds local vol time grid"
        )

        t_min = min(self._time_grid.at(1), 1.0 / 365.0)

        if t <= t_min:
            vol = self._local_vol.local_vol_at_time(0.0, self._spot.value())
            std_dev = vol * math.sqrt(t)
            xm = -0.5 * std_dev * std_dev + math.log(
                self._spot.value() * self._q_ts.discount(t) / self._r_ts.discount(t)
            )
            return NormalDistribution(xm, std_dev)(x)

        if t <= self._time_grid.at(1):
            vol = self._local_vol.local_vol_at_time(0.0, self._spot.value())
            std_dev = vol * math.sqrt(t_min)
            xm = -0.5 * std_dev * std_dev + math.log(
                self._spot.value()
                * self._q_ts.discount(t_min)
                / self._r_ts.discount(t_min)
            )
            gaussian_pdf = NormalDistribution(xm, std_dev)
            delta_t = self._time_grid.at(1) - t_min
            return gaussian_pdf(x) * (self._time_grid.at(1) - t) / delta_t + (
                self._probability_interpolation(0, x) * (t - t_min) / delta_t
            )

        times: list[float] = list(self._time_grid.times)
        lb = bisect.bisect_left(times, t)
        idx = lb - 1
        delta_t = times[lb] - times[lb - 1]
        return self._probability_interpolation(idx - 1, x) * (times[lb] - t) / delta_t + (
            self._probability_interpolation(idx, x) * (t - times[lb - 1]) / delta_t
        )

    def cdf(self, x: float, t: float, /) -> float:
        """# C++ parity: ``LocalVolRNDCalculator::cdf``."""
        self.calculate()

        tc = self._time_grid.closest_time(t)
        idx = (
            self._time_grid.index(tc) - 1
            if tc > t
            else min(len(self._xm) - 1, self._time_grid.index(tc))
        )

        locations = self._xm[idx].locations()
        xl = float(locations[0])
        xr = float(locations[-1])

        if x < xl:
            return 0.0
        if x > xr:
            return 1.0

        addition = 0.1 * (xr - xl)

        if x > 0.5 * (xr + xl):
            while self.pdf(xr, t) > 0.01 * self._local_vol_prob_eps:
                addition *= 1.1
                xr += addition
            return 1.0 - GaussLobattoIntegral(
                self._max_iter, 0.1 * self._local_vol_prob_eps
            )(lambda z: self.pdf(z, t), x, xr)

        while self.pdf(xl, t) > 0.01 * self._local_vol_prob_eps:
            addition *= 1.1
            xl -= addition
        return GaussLobattoIntegral(self._max_iter, 0.1 * self._local_vol_prob_eps)(
            lambda z: self.pdf(z, t), xl, x
        )

    def invcdf(self, p: float, t: float, /) -> float:
        """# C++ parity: ``LocalVolRNDCalculator::invcdf``."""
        self.calculate()

        close_grid_time = self._time_grid.closest_time(t)

        if close_grid_time == 0.0:
            locations = self._xm[0].locations()
            step_size = 0.02 * (float(locations[-1]) - float(locations[0]))
            return InvCDFHelper(
                self,
                math.log(self._spot.value()),
                0.1 * self._local_vol_prob_eps,
                self._max_iter,
                step_size,
            ).inverse_cdf(p, t)

        idx = self._time_grid.index(close_grid_time) - 1
        x = np.asarray(self._xm[idx].locations(), dtype=np.float64)
        step_size = 0.005 * (float(x[-1]) - float(x[0]))
        xm = _SIMPSON(x, x * self._pm[idx])
        return InvCDFHelper(
            self, xm, 0.1 * self._local_vol_prob_eps, self._max_iter, step_size
        ).inverse_cdf(p, t)

    # --- LazyObject -------------------------------------------------------

    def _perform_calculations(self) -> None:  # noqa: PLR0915 — one C++ function
        """# C++ parity: ``LocalVolRNDCalculator::performCalculations``."""
        self._rescale_time_steps = []

        s_t = self._time_grid.at(1)
        gss = self._gaussian_step_size
        t = min(s_t, gss if (gss is not None and gss > 0.0) else 0.5 * s_t)
        vol = self._local_vol.local_vol_at_time(0.0, self._spot.value())

        std_dev = vol * math.sqrt(t)
        xm = -0.5 * std_dev * std_dev + math.log(
            self._spot.value() * self._q_ts.discount(t) / self._r_ts.discount(t)
        )

        std_dev_of_first_step = vol * math.sqrt(s_t)
        norm_inv_eps = _INV_CUM_NORMAL(1.0 - self._local_vol_prob_eps)

        s_lower_bound = xm - norm_inv_eps * std_dev_of_first_step
        s_upper_bound = xm + norm_inv_eps * std_dev_of_first_step

        mesher: Fdm1dMesher = Concentrating1dMesher(
            s_lower_bound, s_upper_bound, self._x_grid, (xm, self._x0_density), True
        )

        x = np.asarray(mesher.locations(), dtype=np.float64).copy()
        gaussian_pdf = NormalDistribution(xm, vol * math.sqrt(t))
        p = np.array([gaussian_pdf(float(xi)) for xi in x], dtype=np.float64)
        p = self._rescale_pdf(x, p)

        qassert.require(x.size > 10, "x grid is too small. Minimum size is greater than 10")

        b = max(1, int(x.size * 0.04))

        evolver = DouglasScheme(
            0.5,
            FdmLocalVolFwdOp(
                FdmMesherComposite(mesher),
                self._spot,
                self._r_ts,
                self._q_ts,
                self._local_vol,
            ),
        )

        self._p_fct = [None] * self._t_grid  # pyright: ignore[reportAttributeAccessIssue]
        self._xm = [mesher] * self._t_grid
        self._pm = np.zeros((self._t_grid, self._x_grid), dtype=np.float64)

        for i in range(1, self._t_grid + 1):
            dt = self._time_grid.at(i) - t

            # Leaking probability mass?
            head = p[:b]
            tail = p[-b:]
            max_left_value = max(abs(float(head.min())), abs(float(head.max())))
            max_right_value = max(abs(float(tail.min())), abs(float(tail.max())))

            if max(max_left_value, max_right_value) > self._local_vol_prob_eps:
                self._rescale_time_steps.append(i)

                old_lower_bound = s_lower_bound
                old_upper_bound = s_upper_bound

                xm = _SIMPSON(x, x * p)
                vols = np.array(
                    [
                        self._local_vol.local_vol_at_time(t + dt, math.exp(float(xj)))
                        for xj in x
                    ],
                    dtype=np.float64,
                )
                vm = _SIMPSON(x, vols) / (float(x[-1]) - float(x[0]))
                scaling_factor = vm * math.sqrt(0.5 * self._time_grid.back())

                if max_left_value > self._local_vol_prob_eps:
                    s_lower_bound -= scaling_factor * (old_upper_bound - old_lower_bound)
                if max_right_value > self._local_vol_prob_eps:
                    s_upper_bound += scaling_factor * (old_upper_bound - old_lower_bound)

                mesher = Concentrating1dMesher(
                    s_lower_bound, s_upper_bound, self._x_grid, (xm, 0.1), False
                )

                p_spline = CubicNaturalSpline(x, p)
                xn = np.asarray(mesher.locations(), dtype=np.float64).copy()
                pn = np.zeros(xn.size, dtype=np.float64)
                for j in range(xn.size):
                    if old_lower_bound <= float(xn[j]) <= old_upper_bound:
                        pn[j] = p_spline(float(xn[j]))

                x = xn
                p = self._rescale_pdf(xn, pn)

                evolver = DouglasScheme(
                    0.5,
                    FdmLocalVolFwdOp(
                        FdmMesherComposite(mesher),
                        self._spot,
                        self._r_ts,
                        self._q_ts,
                        self._local_vol,
                    ),
                )

            evolver.set_step(dt)
            t += dt

            if dt > QL_EPSILON:
                p = evolver.step(p, t)
                p = self._rescale_pdf(x, p)

            self._xm[i - 1] = mesher
            self._pm[i - 1, :] = p
            self._p_fct[i - 1] = CubicNaturalSpline(
                np.asarray(mesher.locations(), dtype=np.float64), self._pm[i - 1]
            )

    # --- helpers ----------------------------------------------------------

    def _probability_interpolation(self, idx: int, x: float) -> float:
        """# C++ parity: ``LocalVolRNDCalculator::probabilityInterpolation``."""
        self.calculate()
        locations = self._xm[idx].locations()
        if x < float(locations[0]) or x > float(locations[-1]):
            return 0.0
        return self._p_fct[idx](x)

    @staticmethod
    def _rescale_pdf(x: Array, p: Array) -> Array:
        """# C++ parity: ``LocalVolRNDCalculator::rescalePDF`` — normalise to mass 1."""
        return p / _SIMPSON(x, p)


__all__ = ["LocalVolRNDCalculator"]
