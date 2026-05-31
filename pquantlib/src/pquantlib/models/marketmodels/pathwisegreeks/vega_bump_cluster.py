"""VegaBumpCluster + VegaBumpCollection — pseudo-root bump groupings.

# C++ parity: ql/models/marketmodels/pathwisegreeks/vegabumpcluster.{hpp,cpp}
# (v1.42.1).

Bumping every pseudo-root element individually is excessive, so elements are
coupled into clusters. A ``VegaBumpCluster`` is a rectangular block in
``(factor, rate, step)`` space; a ``VegaBumpCollection`` is a set of clusters
covering the pseudo-root grid for a given ``MarketModel``.

Divergences from C++:

- ``isFull()`` and ``isNonOverlapping()`` reproduce the C++ behaviour exactly,
  including the historical quirk that both *return* ``numberFailures > 0``
  (i.e. they return ``True`` when there *is* a coverage/overlap failure, which
  reads opposite to their names). ``isSensible()`` likewise returns ``True``
  whenever the collection was constructed via the model ctor (the ``checked_``
  short-circuit), and otherwise ``isNonOverlapping() and isFull()``. These are
  faithful ports of v1.42.1, not corrections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.market_model import MarketModel


class VegaBumpCluster:
    """A rectangular ``(factor, rate, step)`` block of pseudo-root elements.

    # C++ parity: VegaBumpCluster.
    """

    def __init__(
        self,
        factor_begin: int,
        factor_end: int,
        rate_begin: int,
        rate_end: int,
        step_begin: int,
        step_end: int,
    ) -> None:
        # C++ parity: VegaBumpCluster ctor.
        self._factor_begin = factor_begin
        self._factor_end = factor_end
        self._rate_begin = rate_begin
        self._rate_end = rate_end
        self._step_begin = step_begin
        self._step_end = step_end

        qassert.require(
            factor_begin < factor_end,
            "must have factorBegin_ < factorEnd_ in VegaBumpCluster ",
        )
        qassert.require(
            rate_begin < rate_end,
            "must have rateBegin_ < rateEnd_ in VegaBumpCluster ",
        )
        qassert.require(
            step_begin < step_end,
            "must have stepBegin_ < stepEnd_ in VegaBumpCluster ",
        )

    def does_intersect(self, comparee: VegaBumpCluster) -> bool:
        # C++ parity: VegaBumpCluster::doesIntersect.
        if self._factor_end <= comparee._factor_begin:
            return False
        if self._rate_end <= comparee._rate_begin:
            return False
        if self._step_end <= comparee._step_begin:
            return False
        if comparee._factor_end <= self._factor_begin:
            return False
        if comparee._rate_end <= self._rate_begin:
            return False
        return not comparee._step_end <= self._step_begin

    def is_compatible(self, vol_structure: MarketModel) -> bool:
        # C++ parity: VegaBumpCluster::isCompatible.
        if self._rate_end > vol_structure.number_of_rates():
            return False
        if self._step_end > vol_structure.number_of_steps():
            return False
        if self._factor_end > vol_structure.number_of_factors():
            return False
        first_alive_rate = vol_structure.evolution().first_alive_rate()[
            self._step_end - 1
        ]
        # if the rate has reset after the beginning of the last step of the bump
        return self._rate_begin >= first_alive_rate

    def factor_begin(self) -> int:
        return self._factor_begin

    def factor_end(self) -> int:
        return self._factor_end

    def rate_begin(self) -> int:
        return self._rate_begin

    def rate_end(self) -> int:
        return self._rate_end

    def step_begin(self) -> int:
        return self._step_begin

    def step_end(self) -> int:
        return self._step_end


class VegaBumpCollection:
    """A collection of ``VegaBumpCluster``s over a ``MarketModel``'s grid.

    # C++ parity: VegaBumpCollection.

    Two constructors (mirrored as classmethods / overloaded ctor):

    - ``VegaBumpCollection(model, allow_factorwise_bumping=True)`` builds one
      cluster per alive ``(rate, step)`` (and per factor if factor-wise),
      pre-marking the collection ``checked_ = full = nonOverlapped = True``.
    - ``VegaBumpCollection.from_clusters(all_bumps, model)`` wraps a given list
      of clusters (validating compatibility), with ``checked_ = False``.
    """

    def __init__(
        self,
        vol_structure: MarketModel,
        allow_factorwise_bumping: bool = True,
    ) -> None:
        # C++ parity: VegaBumpCollection(model, factorwiseBumping).
        self._associated_vol_structure = vol_structure
        self._all_bumps: list[VegaBumpCluster] = []

        steps = vol_structure.number_of_steps()
        rates = vol_structure.number_of_rates()
        factors = vol_structure.number_of_factors()
        first_alive = vol_structure.evolution().first_alive_rate()

        for s in range(steps):
            for r in range(first_alive[s], rates):
                if allow_factorwise_bumping:
                    for f in range(factors):
                        self._all_bumps.append(
                            VegaBumpCluster(f, f + 1, r, r + 1, s, s + 1)
                        )
                else:
                    self._all_bumps.append(
                        VegaBumpCluster(0, factors, r, r + 1, s, s + 1)
                    )

        self._checked = True
        self._full = True
        self._non_overlapped = True

    @classmethod
    def from_clusters(
        cls,
        all_bumps: list[VegaBumpCluster],
        vol_structure: MarketModel,
    ) -> VegaBumpCollection:
        """Build from an explicit list of clusters (unchecked).

        # C++ parity: VegaBumpCollection(std::vector<VegaBumpCluster>, model).
        """
        self = cls.__new__(cls)
        self._all_bumps = list(all_bumps)
        self._associated_vol_structure = vol_structure
        self._checked = False
        self._full = False
        self._non_overlapped = False
        for bump in self._all_bumps:
            qassert.require(
                bump.is_compatible(vol_structure),
                "incompatible bumps passed to VegaBumpCollection",
            )
        return self

    def associated_model(self) -> MarketModel:
        # C++ parity: VegaBumpCollection::associatedModel.
        return self._associated_vol_structure

    def all_bumps(self) -> list[VegaBumpCluster]:
        # C++ parity: VegaBumpCollection::allBumps.
        return self._all_bumps

    def number_bumps(self) -> int:
        # C++ parity: VegaBumpCollection::numberBumps.
        return len(self._all_bumps)

    def _coverage_grid(self) -> list[list[list[bool]]]:
        """``[step][rate][factor]`` boolean coverage grid, all ``False``."""
        model = self._associated_vol_structure
        factors = model.number_of_factors()
        rates = model.number_of_rates()
        steps = model.number_of_steps()
        return [
            [[False] * factors for _ in range(rates)] for _ in range(steps)
        ]

    def is_full(self) -> bool:
        """Is every alive pseudo-root element bumped at least once?

        # C++ parity: VegaBumpCollection::isFull. NOTE: faithfully reproduces
        # the v1.42.1 return of ``numberFailures > 0`` (reads inverted vs the
        # name), and the ``checked_`` short-circuit.
        """
        if self._checked:
            return self._full

        model = self._associated_vol_structure
        v = self._coverage_grid()
        for bump in self._all_bumps:
            for f in range(bump.factor_begin(), bump.factor_end()):
                for r in range(bump.rate_begin(), bump.rate_end()):
                    for s in range(bump.step_begin(), bump.step_end()):
                        v[s][r][f] = True

        first_alive = model.evolution().first_alive_rate()
        number_failures = 0
        for s in range(model.number_of_steps()):
            for f in range(model.number_of_factors()):
                for r in range(first_alive[s], model.number_of_rates()):
                    if not v[s][r][f]:
                        number_failures += 1
        return number_failures > 0

    def is_non_overlapping(self) -> bool:
        """Is every alive pseudo-root element bumped at most once?

        # C++ parity: VegaBumpCollection::isNonOverlapping. NOTE: faithfully
        # reproduces the v1.42.1 return of ``numberFailures > 0`` and the
        # ``checked_`` short-circuit.
        """
        if self._checked:
            return self._non_overlapped

        v = self._coverage_grid()
        number_failures = 0
        for bump in self._all_bumps:
            for f in range(bump.factor_begin(), bump.factor_end()):
                for r in range(bump.rate_begin(), bump.rate_end()):
                    for s in range(bump.step_begin(), bump.step_end()):
                        if v[s][r][f]:
                            number_failures += 1
                        v[s][r][f] = True
        return number_failures > 0

    def is_sensible(self) -> bool:
        """Is every alive pseudo-root element bumped precisely once?

        # C++ parity: VegaBumpCollection::isSensible.
        """
        if self._checked:
            return True
        return self.is_non_overlapping() and self.is_full()
