"""CPI namespace utilities — observation-interpolation types for inflation swaps.

# C++ parity: ql/indexes/inflationindex.hpp (v1.42.1) — the ``struct CPI``
   block at line 39ff.

Defines the ``InterpolationType`` enum used by inflation swaps + cashflows
to specify how an inflation index observation is interpolated between two
monthly fixings:

* ``AsIndex`` — use whichever mode the index's curve uses.
* ``Flat`` — use the previous month's fixing unchanged within the month.
* ``Linear`` — linearly interpolate between bracketing fixings.

``is_interpolated`` mirrors the C++ free function ``detail::CPI::isInterpolated``.

# C++ parity: ``detail::CPI::effectiveInterpolationType`` collapses ``AsIndex``
  to ``Flat`` because, in v1.42.1, the index-side ``interpolated`` flag is
  always false (see L7-A inflation_index docstring). We carry the same
  collapsing behaviour to keep CPI swap consistency checks aligned with C++.
"""

from __future__ import annotations

from enum import IntEnum


class InterpolationType(IntEnum):
    """How an inflation observation is interpolated between two month fixings.

    # C++ parity: ``CPI::InterpolationType`` (inflationindex.hpp:42-46).
    """

    AsIndex = 0
    Flat = 1
    Linear = 2


def is_interpolated(interp: InterpolationType) -> bool:
    """True iff the interpolation is *Linear*.

    # C++ parity: ``detail::CPI::isInterpolated``. ``AsIndex`` resolves to
    # the index's own ``interpolated()`` flag, which in v1.42.1 is always
    # ``False`` for ``ZeroInflationIndex`` (see L7-A docstring); we therefore
    # report ``False`` for ``AsIndex`` here. Concrete consumers that need the
    # full index-aware resolution can call ``effective_interpolation_type``
    # below.
    """
    return interp == InterpolationType.Linear


def effective_interpolation_type(interp: InterpolationType) -> InterpolationType:
    """Collapse ``AsIndex`` to ``Flat``.

    # C++ parity: ``detail::CPI::effectiveInterpolationType`` in
    # ``inflationindex.cpp``. v1.42.1 treats ``AsIndex`` as ``Flat`` because
    # the index-side ``interpolated`` flag is fixed false. We mirror that.
    """
    if interp == InterpolationType.AsIndex:
        return InterpolationType.Flat
    return interp


__all__ = [
    "InterpolationType",
    "effective_interpolation_type",
    "is_interpolated",
]
