"""PathPricer — abstract base for single-path option pricers.

# C++ parity: ql/methods/montecarlo/pathpricer.hpp (v1.42.1).

C++ uses a templated abstract class
``template <class PathType, class ValueType=Real> class PathPricer``
with a pure-virtual ``operator()(const PathType&)``.  The Python port
collapses both type parameters: ``PathType`` becomes a PEP 695
generic on the class (``PathPricer[PathT]``) and ``ValueType`` is
always ``float`` (no QuantLib pricer in v1.42.1 returns anything
else).  PEP 695 syntax keeps callers concise:

    class EuropeanPathPricer(PathPricer[Path]):
        def __call__(self, path: Path) -> float: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class PathPricer[PathT](ABC):
    """Abstract single-path pricer.

    # C++ parity: ``PathPricer<PathType, ValueType=Real>``.  Subclasses
    # implement ``__call__(self, path) -> float``.
    """

    @abstractmethod
    def __call__(self, path: PathT) -> float:
        """Value the option on a single sampled path."""


__all__ = ["PathPricer"]
