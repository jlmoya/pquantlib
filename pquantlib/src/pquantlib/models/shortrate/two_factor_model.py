"""TwoFactorModel — abstract two-factor short-rate model.

# C++ parity: ql/models/shortrate/twofactormodel.{hpp,cpp} (v1.42.1).

The model describes a short-rate as a deterministic function of two
state variables ``x_t``, ``y_t`` driven by correlated 1-D processes:

* ``dx_t = mu_x(t, x_t) dt + sigma_x(t, x_t) dW^x_t``
* ``dy_t = mu_y(t, y_t) dt + sigma_y(t, y_t) dW^y_t``
* ``dW^x dW^y = rho dt``

Subclasses MUST implement ``dynamics()`` returning a
``ShortRateDynamics`` instance that wraps the two 1-D processes plus
the correlation and the ``r_t = f(t, x, y)`` mapping.

Nested ``ShortRateDynamics`` is preserved as a standalone class (not
inlined) to mirror the C++ nested-class structure and because G2's
``Dynamics`` subclass adds a fitting parameter.

Deferred from L4-D: ``ShortRateTree`` (TreeLattice2D-based) — the
Python port has not landed Lattice2D / TrinomialTree yet (per L4
carve-outs). ``tree()`` raises ``LibraryException`` with a clear
deferral message.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib.models.shortrate.short_rate_model import (
    ShortRateModel,
    _tree_not_implemented,
)

if TYPE_CHECKING:
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
    from pquantlib.time.time_grid import TimeGrid


class ShortRateDynamics:
    """Joint dynamics of the two state variables.

    # C++ parity: nested ``TwoFactorModel::ShortRateDynamics`` in
    # twofactormodel.hpp:72-104 (v1.42.1).

    Subclasses MUST implement ``short_rate(t, x, y)`` returning the
    short-rate as a function of time and the two state variables.

    The two 1-D processes and the correlation between their Brownian
    drivers are exposed via ``x_process`` / ``y_process`` /
    ``correlation`` accessors (matches C++ naming).
    """

    __slots__ = ("_correlation", "_x_process", "_y_process")

    def __init__(
        self,
        x_process: StochasticProcess1D,
        y_process: StochasticProcess1D,
        correlation: float,
    ) -> None:
        self._x_process: StochasticProcess1D = x_process
        self._y_process: StochasticProcess1D = y_process
        self._correlation: float = float(correlation)

    @property
    def x_process(self) -> StochasticProcess1D:
        """Risk-neutral dynamics of ``x_t``.

        # C++ parity: ``ShortRateDynamics::xProcess``.
        """
        return self._x_process

    @property
    def y_process(self) -> StochasticProcess1D:
        """Risk-neutral dynamics of ``y_t``.

        # C++ parity: ``ShortRateDynamics::yProcess``.
        """
        return self._y_process

    @property
    def correlation(self) -> float:
        """Correlation between the two Brownians.

        # C++ parity: ``ShortRateDynamics::correlation``.
        """
        return self._correlation

    @abstractmethod
    def short_rate(self, t: float, x: float, y: float) -> float:
        """Short-rate at time ``t`` given ``(x_t, y_t)``.

        # C++ parity: pure-virtual ``shortRate(Time, Real, Real)``.
        """

    # C++ ``process()`` returns a StochasticProcessArray with the
    # 2x2 correlation matrix. The ``StochasticProcessArray`` class is
    # deferred per L4 carve-out; if a caller needs the joint
    # multi-D representation, point them at ``G2Process`` directly.


class TwoFactorModel(ShortRateModel):
    """Abstract two-factor short-rate model.

    # C++ parity: ``class TwoFactorModel`` in
    # ql/models/shortrate/twofactormodel.hpp:37-49 (v1.42.1).
    """

    def __init__(self, n_params: int) -> None:
        # C++ parity: twofactormodel.cpp:26-27 — forwards nArguments.
        super().__init__(n_arguments=n_params)

    @abstractmethod
    def dynamics(self) -> ShortRateDynamics:
        """Return the joint short-rate dynamics.

        # C++ parity: pure-virtual ``TwoFactorModel::dynamics``.
        """

    def tree(self, grid: TimeGrid) -> object:
        # C++ parity: twofactormodel.cpp:29-40 — builds two trinomial
        # trees and wraps them in a TreeLattice2D. Deferred per L4
        # carve-out (TrinomialTree / TreeLattice2D not ported).
        raise _tree_not_implemented(type(self).__name__)


__all__ = ["ShortRateDynamics", "TwoFactorModel"]
