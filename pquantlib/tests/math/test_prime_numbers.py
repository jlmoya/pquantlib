"""Cross-validate PrimeNumbers against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/tail.json`` —
``prime_numbers``. Indices span the 15-entry hard-coded table and run well
past it (up to the 1001st prime), where the trial-division loop takes over.
"""

from __future__ import annotations

from typing import Any

from pquantlib.math.prime_numbers import PrimeNumbers
from pquantlib.testing import tolerance


def test_matches_cpp_exactly(v143_tail: dict[str, Any]) -> None:
    for index, expected in v143_tail["prime_numbers"]:
        assert PrimeNumbers.get(index) == expected


def test_covers_both_sides_of_the_precomputed_table(v143_tail: dict[str, Any]) -> None:
    """The table has 15 entries; index 15 onwards is computed."""
    indices = [i for i, _ in v143_tail["prime_numbers"]]
    assert min(indices) == 0
    assert any(i < 15 for i in indices)
    assert any(i >= 15 for i in indices)


def test_out_of_order_access_is_consistent(v143_tail: dict[str, Any]) -> None:
    """The C++ table grows monotonically and is shared across calls, so asking
    for a high index first must not change what a lower index returns.
    """
    high = PrimeNumbers.get(2000)
    for index, expected in v143_tail["prime_numbers"]:
        assert PrimeNumbers.get(index) == expected
    tolerance.exact(float(PrimeNumbers.get(2000)), float(high))


def test_sequence_is_strictly_increasing_and_prime() -> None:
    previous = 0
    for i in range(200):
        p = PrimeNumbers.get(i)
        assert p > previous
        assert all(p % q != 0 for q in range(2, int(p**0.5) + 1))
        previous = p
