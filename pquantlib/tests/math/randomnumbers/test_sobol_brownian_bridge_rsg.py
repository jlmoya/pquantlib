"""Cross-validate the Sobol Brownian-bridge RSGs against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json`` —
sections ``sobol_brownian_bridge_rsg``,
``burley2020_sobol_brownian_bridge_rsg`` and ``brownian_ordered_indices``.

The dimension ordering is the answer here, so it is pinned twice: directly, as
the ``factors x steps`` index matrix out of ``SobolBrownianGenerator``, and
indirectly through the bridged paths. The matrix is integers, hence EXACT; the
paths have been through ``InverseCumulativeNormal`` and a Brownian bridge, so
they get a derived tolerance.

.. rubric:: Tolerance derivation

The uniform Sobol stream underneath is bit-exact (see ``test_sobol_rsg``).
``InverseCumulativeNormal`` differs from C++ only by FMA contraction in the
Release build; measured over these cases the discrepancy is at most ~5e-15
relative, and the deviates here have magnitude < 4, so bound the per-variate
absolute error by 1e-13. The bridge forms, for each output coordinate, a
weighted combination of the ``steps`` variates of one factor with weights in
[0, 1] and a unit standard deviation, so absolute errors add at worst
linearly: ``steps * 1e-13``. Relative error is then unbounded near a
coordinate that happens to land close to zero — which is exactly what
happens in the 3x5 Steps case, where one coordinate is 0.0074 and the
relative discrepancy reaches 6e-12. Hence the assertion is on the *absolute*
bound ``steps * 1e-13``, with TIGHT's 1e-12 relative kept alongside it.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.randomnumbers.sobol_brownian_bridge_rsg import (
    Burley2020SobolBrownianBridgeRsg,
    Ordering,
    SobolBrownianBridgeRsg,
)
from pquantlib.math.randomnumbers.sobol_rsg import DirectionIntegers
from pquantlib.models.marketmodels.browniangenerators.sobol_brownian_generator import (
    SobolBrownianGenerator,
)
from pquantlib.testing import reference_reader, tolerance

_ORDERINGS = {
    "Factors": Ordering.FACTORS,
    "Steps": Ordering.STEPS,
    "Diagonal": Ordering.DIAGONAL,
}

_REASON = (
    "bridged Gaussian: the uniform Sobol stream is bit-exact, but each variate "
    "carries <=1e-13 absolute FMA-contraction error from InverseCumulativeNormal "
    "and the bridge sums `steps` of them with weights in [0,1], so the absolute "
    "error bound is steps*1e-13; relative error is meaningless where a bridged "
    "coordinate lands near zero (one is 0.0074)"
)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


def _assert_path(actual: Any, expected: list[float], steps: int) -> None:
    for a, e in zip(actual, expected, strict=True):
        tolerance.custom(
            float(a), float(e), abs_tol=steps * 1e-13, rel_tol=1e-12, reason=_REASON
        )


def test_ordered_indices_exact(cpp: dict[str, Any]) -> None:
    """The Sobol-dimension-to-(factor, step) assignment, per ordering.

    ``Factors`` gives each factor a contiguous block; ``Steps`` interleaves so
    the best dimensions cover step 0 of every factor; ``Diagonal`` sweeps
    both. For 2x4 the Steps and Diagonal matrices coincide — a real property
    of the schemes at that shape, so the 3x5 case is what separates them.
    """
    for case in cpp["brownian_ordered_indices"]:
        gen = SobolBrownianGenerator(
            case["factors"],
            case["steps"],
            _ORDERINGS[case["ordering"]],
            42,
            DirectionIntegers.JoeKuoD7,
        )
        assert gen.ordered_indices() == case["ordered_indices"]


def test_orderings_are_not_all_the_same(cpp: dict[str, Any]) -> None:
    """Guard against every ordering silently collapsing to the same matrix."""
    by_key = {
        (c["factors"], c["steps"], c["ordering"]): c["ordered_indices"]
        for c in cpp["brownian_ordered_indices"]
    }
    assert by_key[(2, 4, "Factors")] != by_key[(2, 4, "Steps")]
    assert by_key[(3, 5, "Diagonal")] != by_key[(2, 4, "Diagonal")]


def test_sobol_bridge_paths(cpp: dict[str, Any]) -> None:
    """First four paths and one 1 000 deep, for all three orderings."""
    for case in cpp["sobol_brownian_bridge_rsg"]:
        g = SobolBrownianBridgeRsg(
            case["factors"],
            case["steps"],
            _ORDERINGS[case["ordering"]],
            case["seed"],
            DirectionIntegers[case["direction_integers"]],
        )
        assert g.dimension() == case["dimension"] == case["factors"] * case["steps"]
        for expected in case["sequences"]:
            _assert_path(g.next_sequence(), expected, case["steps"])
        for _ in range(1000):
            g.next_sequence()
        _assert_path(g.next_sequence(), case["after_1000"], case["steps"])


def test_first_path_is_all_zeros(cpp: dict[str, Any]) -> None:
    """Sobol's first point is the centre of the cube, so the first path is flat.

    Every coordinate of Sobol draw 1 is exactly 0.5, the inverse normal of
    which is 0, and a bridge of zeros is zeros. This is a sharp check that the
    generator has *not* silently skipped its first point (as a scipy-backed
    Sobol does, since scipy emits the origin first).
    """
    for case in cpp["sobol_brownian_bridge_rsg"]:
        assert all(v == 0.0 for v in case["sequences"][0])
        g = SobolBrownianBridgeRsg(
            case["factors"],
            case["steps"],
            _ORDERINGS[case["ordering"]],
            case["seed"],
            DirectionIntegers[case["direction_integers"]],
        )
        assert all(float(v) == 0.0 for v in g.next_sequence())


def test_burley_bridge_paths(cpp: dict[str, Any]) -> None:
    """The Owen-scrambled variant, same two phases."""
    for case in cpp["burley2020_sobol_brownian_bridge_rsg"]:
        g = Burley2020SobolBrownianBridgeRsg(
            case["factors"],
            case["steps"],
            _ORDERINGS[case["ordering"]],
            case["seed"],
            DirectionIntegers[case["direction_integers"]],
            case["scramble_seed"],
        )
        assert g.dimension() == case["dimension"]
        for expected in case["sequences"]:
            _assert_path(g.next_sequence(), expected, case["steps"])
        for _ in range(1000):
            g.next_sequence()
        _assert_path(g.next_sequence(), case["after_1000"], case["steps"])


def test_burley_first_path_is_not_flat(cpp: dict[str, Any]) -> None:
    """Scrambling destroys the "first point is the centre" property.

    The contrast with ``test_first_path_is_all_zeros`` is the cheapest
    available proof that the scramble is actually applied.
    """
    for case in cpp["burley2020_sobol_brownian_bridge_rsg"]:
        assert any(v != 0.0 for v in case["sequences"][0])


def test_last_sequence_does_not_advance() -> None:
    """``last_sequence`` is a pure accessor."""
    g = SobolBrownianBridgeRsg(2, 4, Ordering.DIAGONAL, 0, DirectionIntegers.JoeKuoD7)
    g.next_sequence()
    second = g.next_sequence()
    for a, e in zip(g.last_sequence(), second, strict=True):
        tolerance.exact(float(a), float(e))
    for a, e in zip(g.last_sequence(), second, strict=True):
        tolerance.exact(float(a), float(e))
