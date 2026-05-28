"""DiscretizedAsset — base for tree/FD-discretized financial assets.

# C++ parity: ql/discretizedasset.{hpp,cpp} (v1.42.1) — ``class
#             DiscretizedAsset``. Logical home is ``ql/methods/`` even
#             though the C++ source file is at the root of ``ql/``.

A ``DiscretizedAsset`` holds:

- ``time`` — current rollback time (mutable; moves backward as the
  lattice rolls back).
- ``values`` — ``Array`` of node values at the current time slice.
- ``method`` — the ``Lattice`` driving the rollback.

The high-level ``initialize`` / ``rollback`` / ``partial_rollback`` /
``present_value`` calls all delegate to the held ``Lattice``. C++
inlines these four in the header; the Python port does too.

The low-level contract that subclasses must implement:

- ``reset(size)`` — produce the initial ``values`` array (typically
  constant, e.g. 1.0 for a discount bond, 0.0 for an option).
- ``mandatory_times()`` — times at which rollback must stop (payment
  dates, exercise dates, etc.).

The ``pre_adjust_values`` and ``post_adjust_values`` template methods
guard re-invocation with the C++ "already-at-this-time" cache (via
``close_enough``); subclasses override ``_pre_adjust_values_impl`` and
``_post_adjust_values_impl`` only.

The ``is_on_time(t)`` helper checks whether ``t`` lies on the held
``TimeGrid`` exactly — used by ``DiscretizedOption`` to decide when
to apply an exercise condition.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.closeness import close_enough

if TYPE_CHECKING:
    from pquantlib.methods.lattices.lattice import Lattice

# C++ ``QL_MAX_REAL`` (the sentinel ``DiscretizedAsset`` uses to mean
# "no adjustment has happened yet"). Python ``sys.float_info.max`` is
# the IEEE-754 ``DBL_MAX``, the same constant.
_QL_MAX_REAL: float = sys.float_info.max


class DiscretizedAsset(ABC):
    """Discretized asset used by lattice/FD numerical methods.

    # C++ parity: ``class DiscretizedAsset`` (discretizedasset.hpp:36-143).

    Subclasses populate ``values_`` via ``reset(size)`` and surface
    their stopping times via ``mandatory_times()``. The template
    methods ``pre_adjust_values`` and ``post_adjust_values`` cache
    the last-adjustment time so the lattice never re-applies an
    adjustment at a time it already saw.
    """

    def __init__(self) -> None:
        # C++ parity: ctor initializes ``latestPreAdjustment_`` and
        # ``latestPostAdjustment_`` to ``QL_MAX_REAL`` (a finite double
        # sentinel — close_enough never matches it for typical times).
        self._latest_pre_adjustment: float = _QL_MAX_REAL
        self._latest_post_adjustment: float = _QL_MAX_REAL
        self._time: float = 0.0
        self._values: Array = np.empty(0, dtype=np.float64)
        # ``_method`` is the Lattice driving the rollback; populated by
        # ``initialize``. ``None`` until then.
        self._method: Lattice | None = None

    # -- inspectors -------------------------------------------------------

    @property
    def time(self) -> float:
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        self._time = value

    @property
    def values(self) -> Array:
        return self._values

    @values.setter
    def values(self, value: Array) -> None:
        self._values = value

    def method(self) -> Lattice | None:
        """Return the held Lattice, or ``None`` if not yet initialized.

        # C++ parity: ``DiscretizedAsset::method`` (discretizedasset.hpp:51).
        """
        return self._method

    # -- C++ "set/get" helpers — Python attribute properties handle the
    # getters; the C++ ``time()``/``values()`` setters are reproduced
    # via free methods below for callers that prefer the C++ shape.

    def set_time(self, t: float) -> None:
        """Setter helper (mirrors C++ ``Time& time()`` writable accessor)."""
        self._time = t

    def set_values(self, v: Array) -> None:
        """Setter helper (mirrors C++ ``Array& values()`` writable accessor)."""
        self._values = v

    # -- high-level interface --------------------------------------------

    def initialize(self, method: Lattice, t: float) -> None:
        """Initialize via the given ``Lattice`` at time ``t``.

        # C++ parity: ``DiscretizedAsset::initialize``
        # (discretizedasset.hpp:182-187 inline definition).
        """
        self._method = method
        method.initialize(self, t)

    def rollback(self, to_t: float) -> None:
        """Roll back via the held lattice with final adjustment.

        # C++ parity: ``DiscretizedAsset::rollback``
        # (discretizedasset.hpp:189-191).
        """
        self._require_method().rollback(self, to_t)

    def partial_rollback(self, to_t: float) -> None:
        """Roll back without the final adjustment.

        # C++ parity: ``DiscretizedAsset::partialRollback``
        # (discretizedasset.hpp:193-195).
        """
        self._require_method().partial_rollback(self, to_t)

    def present_value(self) -> float:
        """Present value via the held lattice.

        # C++ parity: ``DiscretizedAsset::presentValue``
        # (discretizedasset.hpp:197-199).
        """
        return self._require_method().present_value(self)

    # -- low-level interface (subclasses override) -----------------------

    @abstractmethod
    def reset(self, size: int) -> None:
        """Populate ``values`` to a fresh ``Array`` of size ``size``.

        # C++ parity: ``DiscretizedAsset::reset(Size size) = 0``
        # (discretizedasset.hpp:89).
        """

    @abstractmethod
    def mandatory_times(self) -> list[float]:
        """Times at which rollback must stop (not necessarily sorted).

        # C++ parity: ``DiscretizedAsset::mandatoryTimes`` pure virtual
        # (discretizedasset.hpp:124).
        """

    # -- adjustment template methods --------------------------------------

    def pre_adjust_values(self) -> None:
        """Run ``_pre_adjust_values_impl`` unless already done at this time.

        # C++ parity: ``DiscretizedAsset::preAdjustValues`` inline
        # (discretizedasset.hpp:201-206).
        """
        if not close_enough(self._time, self._latest_pre_adjustment):
            self._pre_adjust_values_impl()
            self._latest_pre_adjustment = self._time

    def post_adjust_values(self) -> None:
        """Run ``_post_adjust_values_impl`` unless already done at this time.

        # C++ parity: ``DiscretizedAsset::postAdjustValues`` inline
        # (discretizedasset.hpp:208-213).
        """
        if not close_enough(self._time, self._latest_post_adjustment):
            self._post_adjust_values_impl()
            self._latest_post_adjustment = self._time

    def adjust_values(self) -> None:
        """Run both pre- and post-adjustment.

        # C++ parity: ``DiscretizedAsset::adjustValues`` (inline,
        # discretizedasset.hpp:113-116).
        """
        self.pre_adjust_values()
        self.post_adjust_values()

    def is_on_time(self, t: float) -> bool:
        """True iff ``t`` snaps to the same grid point as the held time.

        # C++ parity: ``DiscretizedAsset::isOnTime``
        # (discretizedasset.hpp:215-218 inline).
        """
        m = self._require_method()
        grid = m.time_grid()
        return close_enough(grid[grid.index(t)], self._time)

    # -- subclass extension points ----------------------------------------

    def _pre_adjust_values_impl(self) -> None:  # noqa: B027 — default no-op
        """Hook for subclass-specific pre-adjustment (default no-op).

        # C++ parity: ``DiscretizedAsset::preAdjustValuesImpl`` virtual
        # (discretizedasset.hpp:134) — default empty.
        """

    def _post_adjust_values_impl(self) -> None:  # noqa: B027 — default no-op
        """Hook for subclass-specific post-adjustment (default no-op).

        # C++ parity: ``DiscretizedAsset::postAdjustValuesImpl`` virtual
        # (discretizedasset.hpp:136) — default empty.
        """

    # -- helpers ----------------------------------------------------------

    def _require_method(self) -> Lattice:
        if self._method is None:
            raise RuntimeError(
                "DiscretizedAsset method is not set; "
                "call initialize(method, t) first",
            )
        return self._method

    @staticmethod
    def _is_sentinel(value: float) -> bool:
        # Helper for tests/inspection — checks the QL_MAX_REAL sentinel.
        return not math.isfinite(value) or value >= _QL_MAX_REAL
