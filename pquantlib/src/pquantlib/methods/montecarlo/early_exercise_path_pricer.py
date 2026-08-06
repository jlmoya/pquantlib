"""EarlyExercisePathPricer — abstract base for early-exercise pricers.

# C++ parity: ql/methods/montecarlo/earlyexercisepathpricer.hpp (v1.42.1).

C++ uses a templated abstract class
``template <class PathType, class TimeType=Size, class ValueType=Real>
class EarlyExercisePathPricer``
with three pure-virtual methods:

* ``operator()(path, t)`` — exercise value of the option at time-grid
  index ``t``.
* ``state(path, t)`` — state used as input to the regression at index
  ``t``. For ``Path`` this is just ``Real`` (typically a scaled
  underlying price); for ``MultiPath`` it's an ``Array`` of asset
  prices.
* ``basisSystem()`` — list of basis callables for the regression.
  Their codomain matches the path type's state.

The Python port keeps the same role but flattens the type parameters:

* ``PathType`` becomes a PEP 695 generic on the class.
* ``StateType`` becomes a second generic parameter (since for
  ``Path`` it's ``float`` and for ``MultiPath`` it's
  ``ndarray[float64]``).
* ``TimeType`` is always ``int`` (no other QL pricer in v1.42.1 uses a
  non-int time index).
* ``ValueType`` is always ``float``.

The basis system is just a list of callables `StateType -> float`
(or ``Iterable[Callable[[StateType], float]]``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from pquantlib.exceptions import LibraryException
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer


class EarlyExercisePathPricer[PathT, StateT](ABC):
    """Abstract early-exercise path pricer.

    # C++ parity: ``EarlyExercisePathPricer<PathType, TimeType=Size, ValueType=Real>``
    # (earlyexercisepathpricer.hpp:62-76).

    Subclasses must implement:

    * :meth:`__call__(path, t) -> float` — exercise value at index ``t``.
    * :meth:`state(path, t) -> StateT` — regression state at index ``t``.
    * :meth:`basis_system() -> list[Callable[[StateT], float]]` — basis
      functions used by ``LongstaffSchwartzPathPricer`` for the
      least-squares regression.
    """

    @abstractmethod
    def __call__(self, path: PathT, t: int) -> float:
        """Exercise value of the option on ``path`` at grid index ``t``."""

    @abstractmethod
    def state(self, path: PathT, t: int) -> StateT:
        """Regression state for ``path`` at grid index ``t``."""

    @abstractmethod
    def basis_system(self) -> list[Callable[[StateT], float]]:
        """Basis functions for the LSM regression."""


class EarlyExerciseTraits:
    """Path-shape traits used by the early-exercise machinery.

    # C++ parity: ``template <class PathType> class EarlyExerciseTraits``
    # (earlyexercisepathpricer.hpp:35-58) — a primary template with a
    # deliberately unusable body plus two explicit specialisations, for
    # ``Path`` and ``MultiPath``.

    The two specialisations carry one behavioural difference each: the
    ``StateType`` (``Real`` vs ``Array``) and how the path length is read
    (``path.length()`` vs ``path.pathSize()``). Python folds the dispatch into
    a single runtime lookup, and keeps the C++ "dummy definition, will not
    work" arm as an explicit raise for an unrecognised path type.
    """

    @staticmethod
    def path_length(path: object) -> int:
        """# C++ parity: ``EarlyExerciseTraits<...>::pathLength``."""
        if isinstance(path, MultiPath):
            return path.path_size()
        if isinstance(path, Path):
            return path.length()
        raise LibraryException(
            f"EarlyExerciseTraits has no specialisation for {type(path).__name__}"
        )

    @staticmethod
    def state_type(path: object) -> type[float] | type[np.ndarray[Any, Any]]:
        """# C++ parity: ``EarlyExerciseTraits<...>::StateType``.

        ``Real`` for a single ``Path``, ``Array`` for a ``MultiPath``.
        """
        if isinstance(path, MultiPath):
            return np.ndarray
        if isinstance(path, Path):
            return float
        raise LibraryException(
            f"EarlyExerciseTraits has no specialisation for {type(path).__name__}"
        )


# ``PathPricer`` is re-exported here for convenience — the
# ``LongstaffSchwartzPathPricer`` is *both* a ``PathPricer[PathT]`` (so the
# MC simulation can drive it via the standard contract) *and* delegates
# to an ``EarlyExercisePathPricer[PathT, StateT]`` internally.

__all__ = ["EarlyExercisePathPricer", "EarlyExerciseTraits", "PathPricer"]
