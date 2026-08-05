"""Cross-validate Ranlux64UniformRng against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json``,
section ``ranlux64``.

EXACT tier: the output is a 48-bit integer divided by 2^48, so every value is
representable and the division is exact.

Thirty draws are pinned per case, not five. Ranlux3 (P=223) and Ranlux4
(P=389) share the same R=24, so the two luxury levels emit *identical* first
24 values and only diverge on draw 25, when the discard-block adaptor throws
away P-R base outputs. A five-draw test would pass with the luxury level
wired up wrong.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.randomnumbers.ranlux import (
    Ranlux3UniformRng,
    Ranlux4UniformRng,
    Ranlux64UniformRng,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


def test_ranlux_draws_exact(cpp: dict[str, Any]) -> None:
    """First 30 draws and a draw 10 000 deep, for both luxury levels."""
    for case in cpp["ranlux64"]:
        g = Ranlux64UniformRng(case["P"], case["R"], case["seed"])
        for expected in case["draws"]:
            tolerance.exact(g.next().value, expected)
        for _ in range(10000):
            g.next()
        tolerance.exact(g.next().value, case["after_10000"])


def test_seed_zero_is_the_standard_default_seed(cpp: dict[str, Any]) -> None:
    """``seed == 0`` selects ``default_seed`` (19780503), not 1.

    [rand.eng.sub] specifies the bootstrap LCG as
    ``e(value == 0u ? default_seed : value)``, and
    ``subtract_with_carry_engine::default_seed`` is 19780503. An earlier
    version of this port mapped 0 to 1 on the belief that this was the
    library's fallback; the probe settles it — the seed-0 and seed-19780503
    streams are the same stream.
    """
    by_label = {c["label"]: c for c in cpp["ranlux64"]}
    assert by_label["Ranlux3_seed0"]["draws"] == by_label["Ranlux3_default"]["draws"]
    zero = Ranlux3UniformRng(0)
    default = Ranlux3UniformRng(19780503)
    for _ in range(30):
        tolerance.exact(zero.next().value, default.next().value)


def test_luxury_levels_diverge_only_after_the_first_block(cpp: dict[str, Any]) -> None:
    """Ranlux3 and Ranlux4 share R=24, so they split exactly at draw 25."""
    by_label = {c["label"]: c for c in cpp["ranlux64"]}
    lux3 = by_label["Ranlux3_seed42"]["draws"]
    lux4 = by_label["Ranlux4_seed42"]["draws"]
    assert lux3[:24] == lux4[:24]
    assert lux3[24:] != lux4[24:]

    g3 = Ranlux3UniformRng(42)
    g4 = Ranlux4UniformRng(42)
    for _ in range(24):
        tolerance.exact(g3.next().value, g4.next().value)
    assert g3.next().value != g4.next().value


def test_typedefs_match_their_template_arguments() -> None:
    """# C++ parity: ranluxuniformrng.hpp:65-66."""
    for seed in (42, 19780503):
        named = Ranlux3UniformRng(seed)
        raw = Ranlux64UniformRng(223, 24, seed)
        for _ in range(50):
            tolerance.exact(named.next().value, raw.next().value)
        named4 = Ranlux4UniformRng(seed)
        raw4 = Ranlux64UniformRng(389, 24, seed)
        for _ in range(50):
            tolerance.exact(named4.next().value, raw4.next().value)


def test_draws_lie_in_the_unit_interval() -> None:
    """Structural: the 48-bit output over 2^48 can reach 0 but never 1."""
    g = Ranlux3UniformRng(42)
    for _ in range(1000):
        v = g.next().value
        assert 0.0 <= v < 1.0
