"""Cross-validate ConvolvedStudentT (Behrens-Fisher) cumulative + inverse.

Probe source: migration-harness/cpp/probes/cluster_w6c/probe.cpp
Reference:    migration-harness/references/cluster/w6c.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.convolved_student_t import (
    ConvolvedStudentT,
    CumulativeBehrensFisher,
    InverseCumulativeBehrensFisher,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w6c")


def test_cumulative_sum_two_t3(cpp_ref: dict[str, Any]) -> None:
    cbf = CumulativeBehrensFisher([3, 3], [1.0, 1.0])
    tolerance.tight(cbf(0.0), cpp_ref["convolved_t3_t3_cum_at_0"])
    tolerance.tight(cbf(1.0), cpp_ref["convolved_t3_t3_cum_at_1"])
    tolerance.tight(cbf(2.5), cpp_ref["convolved_t3_t3_cum_at_2_5"])


def test_density_sum_two_t3(cpp_ref: dict[str, Any]) -> None:
    cbf = CumulativeBehrensFisher([3, 3], [1.0, 1.0])
    tolerance.tight(cbf.density(0.0), cpp_ref["convolved_t3_t3_dens_at_0"])
    tolerance.tight(cbf.density(1.0), cpp_ref["convolved_t3_t3_dens_at_1"])


def test_single_t5(cpp_ref: dict[str, Any]) -> None:
    # A single odd-order T is the degenerate one-term convolution.
    cbf = CumulativeBehrensFisher([5], [1.0])
    tolerance.tight(cbf(1.5), cpp_ref["convolved_t5_cum_at_1_5"])


def test_public_alias_is_cumulative() -> None:
    assert ConvolvedStudentT is CumulativeBehrensFisher


def test_inverse_round_trip(cpp_ref: dict[str, Any]) -> None:
    cbf = CumulativeBehrensFisher([3, 3], [0.5, 0.5])
    tolerance.tight(cbf(1.0), cpp_ref["convolved_t3_t3_half_cum_at_1"])
    inv = InverseCumulativeBehrensFisher([3, 3], [0.5, 0.5])
    # The Brent solve has 1e-6 accuracy; recovers x=1.0 to within that band.
    tolerance.loose(
        inv(cbf(1.0)),
        cpp_ref["convolved_t3_t3_half_inv_of_cum_at_1"],
        reason="Brent root solve with 1e-6 accuracy (C++ default).",
    )


def test_inverse_symmetry() -> None:
    inv = InverseCumulativeBehrensFisher([3, 3], [0.5, 0.5])
    assert inv(0.5) == 0.0
    # symmetric distribution: inv(q) == -inv(1-q)
    tolerance.loose(inv(0.7), -inv(0.3), reason="distribution is symmetric.")


def test_even_order_rejected() -> None:
    with pytest.raises(LibraryException, match="Even degree"):
        CumulativeBehrensFisher([2], [1.0])


def test_mismatched_sizes_rejected() -> None:
    with pytest.raises(LibraryException, match="Incompatible sizes"):
        CumulativeBehrensFisher([3, 3], [1.0])
