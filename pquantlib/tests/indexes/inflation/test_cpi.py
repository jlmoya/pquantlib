"""CPI.InterpolationType + helper tests.

Verifies the enum values and the ``effective_interpolation_type`` collapse
that the L7-D inflation swap path relies on.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.indexes.inflation.cpi import (
    CPI,
    InterpolationType,
    effective_interpolation_type,
    is_interpolated,
    lagged_fixing,
    lagged_yoy_rate,
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


def test_cpi_namespace_mirrors_the_cpp_qualified_names() -> None:
    """``struct CPI`` is a namespace carrier in C++; the port exposes the same names."""
    assert CPI.InterpolationType is InterpolationType
    assert CPI.Flat is InterpolationType.Flat
    assert CPI.Linear is InterpolationType.Linear
    assert CPI.AsIndex is InterpolationType.AsIndex
    assert CPI.lagged_fixing is lagged_fixing
    assert CPI.lagged_yoy_rate is lagged_yoy_rate


def test_cpi_namespace_is_not_instantiable() -> None:
    with pytest.raises(LibraryException, match="namespace"):
        CPI()
