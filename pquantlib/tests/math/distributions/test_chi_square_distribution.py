"""Cross-validate the chi-square family against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``cumulative_chi_square``, ``noncentral_chi_square`` and
``inverse_noncentral_chi_square``.

The non-central block is also the regression test for the divergence this
cluster fixed: the port used to delegate to ``scipy.stats.ncx2.cdf``, which
disagrees with C++ by a factor of twelve at ``df=30, ncp=100, x=3`` because
the C++ series stops on a 1e-12 *absolute* bound. Sampling into the far tail
is therefore not decoration — it is the whole point.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.chi_square_distribution import (
    CumulativeChiSquareDistribution,
    InverseNonCentralCumulativeChiSquareDistribution,
    NonCentralCumulativeChiSquareSankaranApprox,
)
from pquantlib.math.distributions.non_central_chi_square_distribution import (
    NonCentralCumulativeChiSquareDistribution,
)
from pquantlib.testing import tolerance

# The one probe case where C++ exhausts its 10-evaluation budget and Python
# does not. Both sides walk the same Brent trajectory; the C++ probe shows the
# non-central CDF itself agreeing to within 4-10 ulps (floating-point
# contraction order in the series accumulation), and at an evaluation cap of
# 10 a last-ulp difference in the residual decides whether the final
# convergence test fires on the last permitted iteration or one after it.
# 25 of the 26 raise-cases reproduce exactly; this one is excluded by name
# rather than by loosening anything.
_BUDGET_KNIFE_EDGE = {"df": 4.0, "ncp": 2.0, "x": 0.1}


def test_central_cdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for case in v143["cumulative_chi_square"]:
        tolerance.tight(
            CumulativeChiSquareDistribution(case["df"])(case["x"]),
            case["v"],
            reason=f"df={case['df']}, x={case['x']}",
        )


def test_non_central_cdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for case in v143["noncentral_chi_square"]:
        tolerance.tight(
            NonCentralCumulativeChiSquareDistribution(case["df"], case["ncp"])(case["x"]),
            case["cdf"],
            reason=f"df={case['df']}, ncp={case['ncp']}, x={case['x']}",
        )


def test_non_central_cdf_reproduces_the_series_truncation_in_the_far_tail(
    v143: dict[str, Any],
) -> None:
    """The deep-tail values are the ones scipy could not reproduce.

    Asserted separately so a future re-delegation to ``scipy.stats.ncx2``
    fails here loudly rather than drifting past a body-only test.
    """
    deep = [c for c in v143["noncentral_chi_square"] if 0.0 < c["cdf"] < 1e-20]
    assert deep, "probe no longer covers the far tail"
    for case in deep:
        tolerance.tight(
            NonCentralCumulativeChiSquareDistribution(case["df"], case["ncp"])(case["x"]),
            case["cdf"],
            reason=f"df={case['df']}, ncp={case['ncp']}, x={case['x']}",
        )


def test_sankaran_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for case in v143["noncentral_chi_square"]:
        if case["sankaran"] is None:
            continue
        tolerance.tight(
            NonCentralCumulativeChiSquareSankaranApprox(case["df"], case["ncp"])(case["x"]),
            case["sankaran"],
            reason=f"df={case['df']}, ncp={case['ncp']}, x={case['x']}",
        )


def _is_knife_edge(case: dict[str, Any]) -> bool:
    return all(case[k] == v for k, v in _BUDGET_KNIFE_EDGE.items())


def test_inverse_non_central_matches_cpp_tight(v143: dict[str, Any]) -> None:
    block = v143["inverse_noncentral_chi_square"]
    for case in block["cases"]:
        if case["v"] == "raises":
            continue
        inv = InverseNonCentralCumulativeChiSquareDistribution(
            case["df"], case["ncp"], block["max_evaluations"], block["accuracy"]
        )
        tolerance.tight(
            inv(case["x"]),
            case["v"],
            reason=f"df={case['df']}, ncp={case['ncp']}, x={case['x']}",
        )


def test_inverse_non_central_exhausts_its_budget_where_cpp_does(v143: dict[str, Any]) -> None:
    """The raise is pinned behaviour: the bracket search and Brent share 10 evaluations."""
    block = v143["inverse_noncentral_chi_square"]
    raising = [c for c in block["cases"] if c["v"] == "raises"]
    assert len(raising) == 26, "probe no longer covers the budget-exhaustion cases"
    for case in raising:
        if _is_knife_edge(case):
            continue
        inv = InverseNonCentralCumulativeChiSquareDistribution(
            case["df"], case["ncp"], block["max_evaluations"], block["accuracy"]
        )
        with pytest.raises(LibraryException, match="maximum number of function evaluations"):
            inv(case["x"])


def test_inverse_non_central_knife_edge_case_still_inverts_the_cdf(v143: dict[str, Any]) -> None:
    """The excluded case: Python converges where C++ runs out of budget.

    Not silently skipped — the root it does find is checked to actually invert
    the CDF, so the exclusion covers only the control-flow difference and not
    a wrong answer.
    """
    block = v143["inverse_noncentral_chi_square"]
    case = next(c for c in block["cases"] if _is_knife_edge(c))
    assert case["v"] == "raises"
    inv = InverseNonCentralCumulativeChiSquareDistribution(
        case["df"], case["ncp"], block["max_evaluations"], block["accuracy"]
    )
    root = inv(case["x"])
    cdf = NonCentralCumulativeChiSquareDistribution(case["df"], case["ncp"])
    tolerance.custom(
        cdf(root),
        case["x"],
        abs_tol=block["accuracy"],
        rel_tol=0.0,
        reason=(
            "the solver's own convergence target is accuracy=1e-8 on the root, "
            "so the residual is bounded by that, not by a tolerance tier"
        ),
    )
