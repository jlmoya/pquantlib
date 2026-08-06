"""VariancePathPricer — realised variance of one simulated path.

# C++ parity: ql/pricingengines/forward/mcvarianceswapengine.hpp (v1.43) —
# ``class VariancePathPricer : public PathPricer<Path>`` and the
# ``detail::Integrand`` functor it integrates.

The C++ body is four lines and every one of them is load-bearing::

    Real VariancePathPricer::operator()(const Path& path) const {
        QL_REQUIRE(!path.empty(), "the path cannot be empty");
        Time t0 = path.timeGrid().front();
        Time t  = path.timeGrid().back();
        Time dt = path.timeGrid().dt(0);
        SegmentIntegral integrator(static_cast<Size>(t/dt));
        detail::Integrand f(path, process_);
        return integrator(f, t0, t)/t;
    }

with::

    class Integrand {
        Real operator()(Time u) const {
            Size i = static_cast<Size>(u/path_.timeGrid().dt(0));
            Real sigma = process_->diffusion(u, path_[i]);
            return sigma*sigma;
        }
    };

Three things a port gets wrong, none of which show up under a constant
volatility (where the integrand is the constant ``sigma**2`` and *any*
quadrature reproduces it):

1. **The path index truncates.** ``i = int(u / dt)`` is a floor, so the
   integrand reads a piecewise-constant left-endpoint sample of the path. It
   does **not** interpolate between nodes, and it does not round.
2. **The quadrature is a midpoint rule** over ``[t0, t]`` with
   ``int(t / dt)`` intervals, so the integrand is evaluated at cell *centres*
   ``t0 + (k + 1/2) * h``. Composed with (1) that reads ``path[k]`` for cell
   ``k`` whenever the integration grid and the path grid coincide — but the
   two grids only coincide when ``t0 == 0``, and ``SegmentIntegral`` divides
   ``[t0, t]`` rather than ``[0, t]``. A port that assumes they always
   coincide is right by accident on the common case and wrong on a path whose
   grid does not start at zero.
3. **The normalisation is ``/ t``, not ``/ (t - t0)``.** The integral runs
   from ``t0`` but is divided by the *end* time.

The diffusion is taken from the process, i.e.
``GeneralizedBlackScholesProcess.diffusion(u, x) = localVolatility().local_vol(u, x)`` —
the **local** volatility, not the Black volatility. For a term structure of
Black volatilities those differ by the Dupire step, which is exactly where a
port can be plausibly wrong without the price ever looking odd.

Cross-validated against ``migration-harness/references/v143/pe/mcforward.json``
(cases ``vs_pathpricer_direct`` and ``vs_pathpricer_direct_termvol``).
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.math.integrals.segment import SegmentIntegral
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class Integrand:
    """Squared local volatility along a sampled path, as a function of time.

    # C++ parity: ``detail::Integrand`` in mcvarianceswapengine.hpp.

    Kept as a named class rather than folded into a closure because C++ gives
    it an identity and because its indexing convention — ``int(u / dt)``, a
    truncation — is the single most portable-looking, most easily mis-ported
    line in the engine. Naming it makes it directly testable.
    """

    __slots__ = ("_dt", "_path", "_process")

    def __init__(self, path: Path, process: GeneralizedBlackScholesProcess) -> None:
        self._path: Path = path
        self._process: GeneralizedBlackScholesProcess = process
        # C++ caches nothing, but dt(0) is a constant of the grid.
        self._dt: float = path.time_grid.dt(0)

    def __call__(self, u: float, /) -> float:
        """``sigma(u, path[int(u / dt)]) ** 2``.

        # C++ parity: ``detail::Integrand::operator()(Time t)``.
        """
        i = int(u / self._dt)
        sigma = self._process.diffusion_1d(u, self._path[i])
        return sigma * sigma


@final
class VariancePathPricer(PathPricer[Path]):
    """Annualised realised variance of a single path.

    # C++ parity: ``class VariancePathPricer : public PathPricer<Path>``.
    """

    __slots__ = ("_process",)

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        self._process: GeneralizedBlackScholesProcess = process

    def __call__(self, path: Path) -> float:
        """Integrate the squared local vol along ``path`` and divide by ``t``.

        # C++ parity: ``VariancePathPricer::operator()``.
        """
        qassert.require(not path.empty(), "the path cannot be empty")
        grid = path.time_grid
        t0 = grid.front()
        t = grid.back()
        dt = grid.dt(0)
        # C++: SegmentIntegral(static_cast<Size>(t/dt)) — truncation, and it is
        # t (the end time) over dt, not the span (t - t0) over dt.
        integrator = SegmentIntegral(int(t / dt))
        f = Integrand(path, self._process)
        return integrator(f, t0, t) / t


__all__ = ["Integrand", "VariancePathPricer"]
