"""Convergence-table decorator for a statistics class.

# C++ parity: ql/math/statistics/convergencestatistics.hpp (v1.43).

C++::

    template <class T, class U = DoublingConvergenceSteps>
    class ConvergenceStatistics : public T;

so, as with the other "inherit from the template parameter" classes in this
package, :class:`ConvergenceStatistics` is a **mixin** and the underlying
statistics class is supplied by multiple inheritance, mixin first::

    class ConvergingStatistics(ConvergenceStatistics, Statistics): ...

The sampling policy ``U`` *is* expressible directly, as a constructor
argument: pass ``sampling_rule=`` any object with ``initial_samples()`` and
``next_samples(current)``. It defaults to
:class:`DoublingConvergenceSteps`, exactly as in C++.

The C++ ``ConvergenceStatistics(const T& stats, const U& rule)`` overload —
build a decorated accumulator from an already-populated one — is not
ported: it relies on T's copy constructor, and PQuantLib's statistics
classes expose no copy protocol. The default-construct-then-add path is
the one every caller uses.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol


class ConvergenceSteps(Protocol):
    """Sampling policy: at which sample counts the mean is recorded.

    # C++ parity: the informal ``U`` concept documented at
    # convergencestatistics.hpp:44-53.
    """

    def initial_samples(self) -> int: ...
    def next_samples(self, current: int) -> int: ...


class DoublingConvergenceSteps:
    """Record the mean at 1, 3, 7, 15, … samples.

    # C++ parity: ``class DoublingConvergenceSteps``
    # (convergencestatistics.hpp:33-37).
    """

    __slots__ = ()

    def initial_samples(self) -> int:
        """First sample count at which the mean is recorded."""
        return 1

    def next_samples(self, current: int) -> int:
        """Next sample count after ``current``."""
        return 2 * current + 1


class ConvergenceStatistics:
    """Adds a convergence table tracking the mean to a statistics class.

    # C++ parity: ``template <class T, class U> class ConvergenceStatistics``
    # (convergencestatistics.hpp:57-84).
    """

    # Deliberately *not* slotted: a mixin with a non-empty ``__slots__``
    # cannot be combined with a slotted statistics class ("multiple bases
    # have instance lay-out conflict"), and every underlying statistics
    # class in this package is slotted.
    _sampling_rule: ConvergenceSteps
    _table: list[tuple[int, float]]
    _next_sample_size: int

    if TYPE_CHECKING:
        # Supplied by the class this mixin is combined with — the C++ ``T``
        # template parameter.
        def samples(self) -> int: ...
        def mean(self) -> float: ...

    def __init__(
        self, *args: Any, sampling_rule: ConvergenceSteps | None = None, **kwargs: Any
    ) -> None:
        """Forward ``*args``/``**kwargs`` to the underlying statistics class.

        # C++ parity: convergencestatistics.hpp:96-100.
        """
        # The decorator's own state is set up *before* the underlying class is
        # constructed: some statistics classes call ``self.reset()`` from their
        # own ``__init__``, which Python dispatches to the override below.
        # (C++ has no such hazard — ``T::reset()`` inside ``T``'s constructor
        # never reaches the derived override.)
        self._sampling_rule = (
            DoublingConvergenceSteps() if sampling_rule is None else sampling_rule
        )
        self._table = []
        self._next_sample_size = self._sampling_rule.initial_samples()
        super().__init__(*args, **kwargs)
        self.reset()

    def add(self, value: float, weight: float = 1.0) -> None:
        """Add a datum and, at a sampling step, record the running mean.

        # C++ parity: convergencestatistics.hpp:103-112.
        """
        super().add(value, weight)  # type: ignore[misc]  # supplied by the mixed-in class
        if self.samples() == self._next_sample_size:
            self._table.append((self.samples(), self.mean()))
            self._next_sample_size = self._sampling_rule.next_samples(self._next_sample_size)

    def add_sequence(
        self, values: Iterable[float], weights: Iterable[float] | None = None
    ) -> None:
        """Add a sequence of data through :meth:`add`, so steps are recorded.

        # C++ parity: convergencestatistics.hpp:66-76 — redeclared on the
        # decorator precisely so it routes through the overriding ``add``.
        """
        if weights is None:
            for v in values:
                self.add(v)
        else:
            for v, w in zip(values, weights, strict=False):
                self.add(v, w)

    def reset(self) -> None:
        """Drop every sample and clear the convergence table.

        # C++ parity: convergencestatistics.hpp:115-120.
        """
        super().reset()  # type: ignore[misc]  # supplied by the mixed-in class
        self._next_sample_size = self._sampling_rule.initial_samples()
        self._table.clear()

    def convergence_table(self) -> list[tuple[int, float]]:
        """The recorded ``(sample count, mean)`` pairs.

        # C++ parity: convergencestatistics.hpp:122-126.
        """
        return self._table
