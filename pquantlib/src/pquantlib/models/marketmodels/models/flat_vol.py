"""FlatVol / FlatVolFactory — flat-per-rate volatility MarketModel.

# C++ parity: ql/models/marketmodels/models/flatvol.{hpp,cpp} (v1.43).

The workhorse concrete ``MarketModel``: each forward rate has a single flat
instantaneous volatility ``vols[i]``, and the per-step covariance matrix is
built by integrating the flat-vol covariance over each correlation sub-interval
(weighted by a ``PiecewiseConstantCorrelation``), then taking the spectral
rank-reduced pseudo-square-root.

``FlatVolFactory`` is the ``YieldTermStructure``-driven factory: it reads the
initial forward rates off the curve, displaces the linearly-interpolated
volatility term structure to match, builds an exponential time-homogeneous
forward correlation, and hands the lot to ``FlatVol``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    exponential_correlations,
)
from pquantlib.models.marketmodels.correlations.time_homogeneous_forward_correlation import (
    TimeHomogeneousForwardCorrelation,
)
from pquantlib.models.marketmodels.market_model import MarketModel, MarketModelFactory
from pquantlib.models.marketmodels.models.pseudo_sqrt import (
    SalvagingAlgorithm,
    rank_reduced_sqrt,
)
from pquantlib.patterns.observer import Observable
from pquantlib.time.compounding import Compounding

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
    from pquantlib.models.marketmodels.piecewise_constant_correlation import (
        PiecewiseConstantCorrelation,
    )
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure


def flat_vol_covariance(
    t1: float,
    t2: float,
    t: float,
    s: float,
    v1: float,
    v2: float,
) -> float:
    """Integral of the flat-vol covariance over ``[t1, t2]``.

    # C++ parity: flatvol.cpp ``flatVolCovariance``. ``T`` (=``t``) and ``S``
    (=``s``) are the two rate fixing times; ``v1``/``v2`` their flat vols.
    """
    qassert.require(
        t1 <= t2,
        f"integrations bounds ({t1},{t2}) are in reverse order",
    )
    cut_off = min(s, t)
    if t1 >= cut_off:
        return 0.0
    cut_off = min(t2, cut_off)
    return (cut_off - t1) * v1 * v2


class FlatVol(MarketModel):
    """Flat-per-rate volatility market model.

    # C++ parity: flatvol.hpp/.cpp FlatVol.
    """

    def __init__(
        self,
        volatilities: list[float],
        correlations: PiecewiseConstantCorrelation,
        evolution: EvolutionDescription,
        number_of_factors: int,
        initial_rates: list[float],
        displacements: list[float],
    ) -> None:
        super().__init__()
        self._number_of_factors = number_of_factors
        self._number_of_rates = len(initial_rates)
        self._number_of_steps = len(evolution.evolution_times())
        self._initial_rates = list(initial_rates)
        self._displacements = list(displacements)
        self._evolution = evolution

        n = self._number_of_rates
        rate_times = evolution.rate_times()
        qassert.require(
            n == len(rate_times) - 1,
            f"mismatch between number of rates ({n}) and rate times",
        )
        qassert.require(
            n == len(displacements),
            f"mismatch between number of rates ({n}) and displacements "
            f"({len(displacements)})",
        )
        qassert.require(
            n == len(volatilities),
            f"mismatch between number of rates ({n}) and vols ({len(volatilities)})",
        )
        qassert.require(
            n <= self._number_of_factors * self._number_of_steps,
            f"number of rates ({n}) greater than number of factors "
            f"({self._number_of_factors}) times number of steps "
            f"({self._number_of_steps})",
        )
        qassert.require(
            number_of_factors <= n,
            f"number of factors ({number_of_factors}) cannot be greater than "
            f"numberOfRates ({n})",
        )
        qassert.require(
            number_of_factors > 0,
            f"number of factors ({number_of_factors}) must be greater than zero",
        )

        self._pseudo_roots: list[Matrix] = []
        eff_stop_time = 0.0
        corr_times = correlations.times()
        evol_times = evolution.evolution_times()
        # C++ parity: ``kk`` is a single running index across ALL steps,
        # declared with ``k`` in the outer for-loop (``for (k=0, kk=0; ...)``)
        # and advanced past correlation times <= evolTimes[k] at each step end.
        kk = 0
        for k in range(self._number_of_steps):
            # one covariance per evolution step
            covariance = np.zeros((n, n), dtype=np.float64)
            # there might be more than one correlation matrix in a single step
            while corr_times[kk] < evol_times[k]:
                eff_start_time = eff_stop_time
                eff_stop_time = corr_times[kk]
                corr_matrix = correlations.correlation(kk)
                self._accumulate(
                    covariance, eff_start_time, eff_stop_time, rate_times,
                    volatilities, corr_matrix, n,
                )
                kk += 1
            # last part in the evolution step
            eff_start_time = eff_stop_time
            eff_stop_time = evol_times[k]
            corr_matrix = correlations.correlation(kk)
            self._accumulate(
                covariance, eff_start_time, eff_stop_time, rate_times,
                volatilities, corr_matrix, n,
            )
            # no more use for the kk-th correlation matrix
            while kk < len(corr_times) and corr_times[kk] <= evol_times[k]:
                kk += 1

            # make it symmetric (we only filled the upper triangle)
            for i in range(n):
                for j in range(i + 1, n):
                    covariance[j, i] = covariance[i, j]

            pseudo_root = rank_reduced_sqrt(
                covariance, number_of_factors, 1.0, SalvagingAlgorithm.NONE
            )
            qassert.require(
                pseudo_root.shape[0] == n,
                f"step {k} flat vol wrong number of rows: {pseudo_root.shape[0]} "
                f"instead of {n}",
            )
            qassert.require(
                pseudo_root.shape[1] == number_of_factors,
                f"step {k} flat vol wrong number of columns: "
                f"{pseudo_root.shape[1]} instead of {self._number_of_factors}",
            )
            self._pseudo_roots.append(pseudo_root)

    @staticmethod
    def _accumulate(
        covariance: Matrix,
        eff_start_time: float,
        eff_stop_time: float,
        rate_times: list[float],
        vols: list[float],
        corr_matrix: Matrix,
        n: int,
    ) -> None:
        """Add the flat-vol covariance over ``[start, stop]`` (upper triangle).

        # C++ parity: the inner i/j double-loop body in flatvol.cpp.
        """
        for i in range(n):
            for j in range(i, n):
                cov = flat_vol_covariance(
                    eff_start_time, eff_stop_time,
                    rate_times[i], rate_times[j], vols[i], vols[j],
                )
                covariance[i, j] += cov * corr_matrix[i, j]

    def initial_rates(self) -> list[float]:
        """The initial forward rates."""
        return self._initial_rates

    def displacements(self) -> list[float]:
        """The displacement (shift) of each rate."""
        return self._displacements

    def evolution(self) -> EvolutionDescription:
        """The evolution description."""
        return self._evolution

    def number_of_rates(self) -> int:
        """Number of forward rates."""
        return len(self._initial_rates)

    def number_of_factors(self) -> int:
        """Number of driving factors."""
        return self._number_of_factors

    def number_of_steps(self) -> int:
        """Number of evolution steps."""
        return self._number_of_steps

    def pseudo_root(self, i: int) -> Matrix:
        """Pseudo-square-root of the covariance matrix for step ``i``.

        # C++ parity: FlatVol::pseudoRoot.
        """
        qassert.require(
            i < self._number_of_steps,
            f"the index {i} is invalid: it must be less than number of steps "
            f"({self._number_of_steps})",
        )
        return self._pseudo_roots[i]


class FlatVolFactory(MarketModelFactory, Observable):
    """Builds ``FlatVol`` models off a yield curve + a volatility term structure.

    # C++ parity: flatvol.hpp/.cpp FlatVolFactory.

    ``times`` / ``vols`` define a linearly-interpolated volatility curve (the
    C++ comment marks this as a stand-in for a real volatility structure);
    ``displacement`` is applied uniformly to every rate, and the raw vols are
    rescaled to displaced vols by ``f * vol / (f + displacement)``.

    # C++ parity divergence: C++ takes ``Handle<YieldTermStructure>``. PQuantLib
    does not implement ``Handle``/``RelinkableHandle`` (see
    ``cashflows/cms_coupon_pricer.py``), so the term structure is threaded
    directly. The observer wiring is kept: the factory registers with the curve
    and re-broadcasts its ``update()``.
    """

    def __init__(
        self,
        long_term_correlation: float,
        beta: float,
        times: list[float],
        vols: list[float],
        yield_curve: YieldTermStructure,
        displacement: float,
    ) -> None:
        # C++ parity: FlatVolFactory ctor.
        Observable.__init__(self)
        self._long_term_correlation = long_term_correlation
        self._beta = beta
        self._times = list(times)
        self._vols = list(vols)
        self._yield_curve = yield_curve
        self._displacement = displacement
        # C++ builds a LinearInterpolation over (times_, vols_) then calls
        # update(); the pquantlib ctor already runs update().
        self._volatility = LinearInterpolation(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._vols, dtype=np.float64),
        )
        yield_curve.register_with(self)

    def create(
        self, evolution: EvolutionDescription, number_of_factors: int
    ) -> MarketModel:
        """Build a ``FlatVol`` for the given evolution + factor count.

        # C++ parity: FlatVolFactory::create.
        """
        rate_times = evolution.rate_times()
        number_of_rates = len(rate_times) - 1

        initial_rates = [
            self._yield_curve.forward_rate(
                rate_times[i], rate_times[i + 1], Compounding.Simple
            ).rate()
            for i in range(number_of_rates)
        ]

        displaced_volatilities = [
            initial_rates[i]
            * self._volatility(rate_times[i])
            / (initial_rates[i] + self._displacement)
            for i in range(number_of_rates)
        ]

        displacements = [self._displacement] * number_of_rates

        correlations = exponential_correlations(
            evolution.rate_times(), self._long_term_correlation, self._beta
        )
        corr = TimeHomogeneousForwardCorrelation(correlations, rate_times)
        return FlatVol(
            displaced_volatilities,
            corr,
            evolution,
            number_of_factors,
            initial_rates,
            displacements,
        )

    def update(self) -> None:
        """Re-broadcast the yield curve's notification.

        # C++ parity: FlatVolFactory::update -> notifyObservers().
        """
        self.notify_observers()
