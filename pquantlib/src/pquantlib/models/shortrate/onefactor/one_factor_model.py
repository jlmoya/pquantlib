"""OneFactorModel abstract base + ShortRateDynamics.

# C++ parity: ql/models/shortrate/onefactormodel.{hpp,cpp} (v1.42.1).

The abstract one-factor short-rate model. Concrete subclasses
implement ``dynamics()`` (returning a ``ShortRateDynamics`` over a
1-D stochastic process) and inherit ``tree(grid)`` from the parent
``ShortRateModel``.

Trees: the C++ ``tree(grid)`` builds a TrinomialTree over the process
and wraps it in a ``ShortRateTree`` lattice. Both classes depend on
the TreeLattice / TrinomialTree machinery that is deferred to L4-F
(or later). The base class declares ``tree(grid)`` and the concrete
short-rate models override it to raise ``LibraryException`` with a
clear "tree not implemented" message.

Dynamics: ``ShortRateDynamics`` is a thin pair ``(variable(t, r),
short_rate(t, x))`` plus a backing ``StochasticProcess1D``. It
encapsulates the change of variables between the modelled state ``x``
and the short rate ``r``. Subclasses (Vasicek/HullWhite/CIR/ECIR)
provide their own ShortRateDynamics with the appropriate process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib.exceptions import LibraryException
from pquantlib.models.model import ShortRateModel

if TYPE_CHECKING:
    from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class ShortRateDynamics(ABC):
    """Change of variables between short-rate and state-variable.

    # C++ parity: nested ``OneFactorModel::ShortRateDynamics`` in
    # ql/models/shortrate/onefactormodel.hpp:54-72 (v1.42.1).

    Holds a 1-D stochastic process for the state ``x``, and a pair of
    abstract methods to map between ``x`` and the short rate ``r``:

    - ``variable(t, r) -> x``
    - ``short_rate(t, x) -> r``
    """

    __slots__ = ("_process",)

    def __init__(self, process: StochasticProcess1D) -> None:
        self._process: StochasticProcess1D = process

    @abstractmethod
    def variable(self, t: float, r: float) -> float:
        """Compute state variable ``x`` from short rate ``r`` at time ``t``.

        # C++ parity: ``ShortRateDynamics::variable(Time, Rate)``.
        """
        ...

    @abstractmethod
    def short_rate(self, t: float, variable: float) -> float:
        """Compute short rate ``r`` from state variable ``x`` at time ``t``.

        # C++ parity: ``ShortRateDynamics::shortRate(Time, Real)``.
        """
        ...

    @property
    def process(self) -> StochasticProcess1D:
        """The backing 1-D stochastic process for the state.

        # C++ parity: ``ShortRateDynamics::process``.
        """
        return self._process


class OneFactorModel(ShortRateModel):
    """Abstract single-factor short-rate model.

    # C++ parity: ``class OneFactorModel : public ShortRateModel`` in
    # ql/models/shortrate/onefactormodel.hpp:38-51 (v1.42.1).

    Adds the abstract ``dynamics()`` method (return the
    ``ShortRateDynamics`` describing this model's short-rate variable).

    ``tree(grid)`` is declared as abstract on ``ShortRateModel``. In
    the C++ port it builds a TrinomialTree from the dynamics' process
    and wraps it in a ``ShortRateTree`` lattice. We override with a
    raising stub here — concrete subclasses re-raise with their own
    "not implemented in pquantlib" message. The lattice carry-over is
    documented in the L4-B completion notes.
    """

    @abstractmethod
    def dynamics(self) -> ShortRateDynamics:
        """Return the model's short-rate dynamics.

        # C++ parity: ``OneFactorModel::dynamics``.
        """
        ...

    def tree(self, grid: object) -> object:
        """Build a recombining trinomial tree lattice over ``grid``.

        # C++ parity: ``OneFactorModel::tree(const TimeGrid&)`` in
        # onefactormodel.cpp:89-95 — builds ``TrinomialTree`` over
        # ``dynamics()->process()`` and wraps in a ShortRateTree.

        DEFERRED in pquantlib L4-B per cluster scope: TrinomialTree /
        TreeLattice1D / ShortRateTree are L4-F (or later) work. Callers
        currently use ``discount`` / ``discount_bond`` / ``discount_bond_option``
        analytical paths instead.
        """
        raise LibraryException(
            "OneFactorModel.tree: TrinomialTree lattice not yet ported (L4-B carve-out)"
        )
