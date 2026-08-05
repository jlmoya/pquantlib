"""Cross-validate the binomial family against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``binomial_coefficient``, ``binomial`` and ``peizer_pratt``. ``p`` includes
both endpoints (which take dedicated C++ branches keyed on a ``-QL_MAX_REAL``
sentinel) and 1e-6 either side of them; ``k`` runs one past ``n`` so the
out-of-support branch is pinned too.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.binomial_distribution import (
    BinomialDistribution,
    CumulativeBinomialDistribution,
    binomial_coefficient,
    binomial_coefficient_ln,
    peizer_pratt_method2_inversion,
)
from pquantlib.testing import tolerance


def test_binomial_coefficient_ln_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for case in v143["binomial_coefficient"]:
        tolerance.tight(
            binomial_coefficient_ln(case["n"], case["k"]),
            case["ln"],
            reason=f"n={case['n']}, k={case['k']}",
        )


def test_binomial_coefficient_matches_cpp_exactly(v143: dict[str, Any]) -> None:
    """EXACT: ``floor(0.5 + exp(ln C))`` lands on an integer-valued double."""
    for case in v143["binomial_coefficient"]:
        tolerance.exact(binomial_coefficient(case["n"], case["k"]), case["value"])


def test_binomial_coefficient_rejects_k_greater_than_n() -> None:
    with pytest.raises(LibraryException, match="n<k not allowed"):
        binomial_coefficient_ln(3, 5)


def test_pdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["binomial"]:
        pdf = BinomialDistribution(block["p"], block["n"])
        for k, expected in block["pdf"]:
            tolerance.tight(pdf(k), expected, reason=f"p={block['p']}, n={block['n']}, k={k}")


def test_cdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["binomial"]:
        cdf = CumulativeBinomialDistribution(block["p"], block["n"])
        for k, expected in block["cdf"]:
            tolerance.tight(cdf(k), expected, reason=f"p={block['p']}, n={block['n']}, k={k}")


def test_degenerate_p_is_a_point_mass(v143: dict[str, Any]) -> None:
    """p == 0 and p == 1 dispatch on an exact-zero log sentinel: must be bit-exact."""
    for block in v143["binomial"]:
        if block["p"] not in (0.0, 1.0):
            continue
        pdf = BinomialDistribution(block["p"], block["n"])
        for k, expected in block["pdf"]:
            tolerance.exact(pdf(k), expected)


def test_peizer_pratt_matches_cpp_exactly(v143: dict[str, Any]) -> None:
    for z, n, expected in v143["peizer_pratt"]:
        tolerance.exact(peizer_pratt_method2_inversion(z, n), expected, reason=f"z={z}, n={n}")


def test_peizer_pratt_rejects_even_n() -> None:
    with pytest.raises(LibraryException, match="must be an odd number"):
        peizer_pratt_method2_inversion(0.5, 50)
