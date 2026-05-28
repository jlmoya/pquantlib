"""CPI.InterpolationType + helper tests.

Verifies the enum values and the ``effective_interpolation_type`` collapse
that the L7-D inflation swap path relies on.
"""

from __future__ import annotations

from pquantlib.indexes.inflation.cpi import (
    InterpolationType,
    effective_interpolation_type,
    is_interpolated,
)


def test_interpolation_type_values_match_cpp() -> None:
    # C++ parity: ql/indexes/inflationindex.hpp:42-46 — AsIndex=0, Flat=1, Linear=2.
    assert int(InterpolationType.AsIndex) == 0
    assert int(InterpolationType.Flat) == 1
    assert int(InterpolationType.Linear) == 2


def test_is_interpolated_true_only_for_linear() -> None:
    assert not is_interpolated(InterpolationType.AsIndex)
    assert not is_interpolated(InterpolationType.Flat)
    assert is_interpolated(InterpolationType.Linear)


def test_effective_interpolation_type_collapses_asindex_to_flat() -> None:
    assert effective_interpolation_type(InterpolationType.AsIndex) == InterpolationType.Flat
    assert effective_interpolation_type(InterpolationType.Flat) == InterpolationType.Flat
    assert effective_interpolation_type(InterpolationType.Linear) == InterpolationType.Linear
