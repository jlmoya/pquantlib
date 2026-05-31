"""Tests for the W10-B LMM forward-rate evolvers (batch a).

Cross-validates against ``migration-harness/references/cluster/w10b.json``.

Cross-validation design (two ambiguities removed):

1. **Brownian stream.** Every evolver is driven by
   ``MTBrownianGeneratorFactory(seed)``, whose Gaussian stream pquantlib ports
   bit-identically from C++ (``RandomSequenceGenerator<MersenneTwisterUniformRng>``
   + ``InverseCumulativeNormal``). MT is used rather than Sobol because Sobol's
   stream (scipy Joe-Kuo) diverges from C++ (Jaeckel) beyond ~2 factors.

2. **Pseudo-root sign.** The spectral pseudo-root ``A`` of a covariance matrix
   is only unique up to an orthogonal rotation / sign of each eigenvector, so
   ``FlatVol``'s raw ``A`` differs between C++ (Jacobi) and pquantlib (LAPACK)
   for expired-rate (zero-row) steps even though ``A @ A.T`` agrees. Since the
   diffusion term ``A @ Z`` is sign-sensitive, both the C++ probe and these
   tests run the evolvers against a ``PseudoRootFacade`` built from the SAME
   pseudo-root matrices (emitted by the probe under ``pr5`` / ``pr2`` /
   ``cotswap_pr``). This isolates the drift+diffusion evolution math and makes
   the evolved rates match TIGHT.

C++ parity:
  ql/models/marketmodels/evolvers/lognormalfwdratepc.{hpp,cpp}
  ql/models/marketmodels/evolvers/lognormalfwdrateeuler.{hpp,cpp}
  ql/models/marketmodels/evolvers/lognormalfwdrateipc.{hpp,cpp}
  ql/models/marketmodels/evolvers/lognormalfwdrateeulerconstrained.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.models.marketmodels.browniangenerators.mt_brownian_generator import (
    MTBrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.evolution_description import (
    money_market_measure,
    terminal_measure,
)
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_euler import (
    LogNormalFwdRateEuler,
)
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_euler_constrained import (
    LogNormalFwdRateEulerConstrained,
)
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_ipc import (
    LogNormalFwdRateIpc,
)
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_pc import (
    LogNormalFwdRatePc,
)
from pquantlib.models.marketmodels.models.pseudo_root_facade import PseudoRootFacade
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w10b")


def rate_times6() -> list[float]:
    return [0.5 * (i + 1) for i in range(6)]  # 0.5..3.0


def make_facade(
    ref: dict[str, Any], prefix: str, n: int, factors: int
) -> PseudoRootFacade:
    """Rebuild a ``PseudoRootFacade`` from the probe-emitted pseudo-roots.

    Reads ``<prefix>_pr_<step>_<i>_<k>`` (the per-step ``A`` matrices) and
    ``<prefix>_init_<i>`` (the model's initial rates) so the Python evolver runs
    against byte-identical ``A`` matrices to the C++ probe.
    """
    steps = n  # PseudoRootFacade requires n pseudo-roots (one per rate/step)
    pseudo_roots: list[np.ndarray] = []
    for s in range(steps):
        a = np.zeros((n, factors), dtype=np.float64)
        for i in range(n):
            for k in range(factors):
                a[i, k] = ref[f"{prefix}_pr_{s}_{i}_{k}"]
        pseudo_roots.append(a)
    initial_rates = [ref[f"{prefix}_init_{i}"] for i in range(n)]
    return PseudoRootFacade(pseudo_roots, rate_times6(), initial_rates, [0.0] * n)


def _evolve_final_forwards(evolver: Any) -> list[float]:
    evolver.start_new_path()
    steps = len(evolver.numeraires())
    for _ in range(steps):
        evolver.advance_step()
    return list(evolver.current_state().forward_rates())


def test_pc_evolved_forwards(ref: dict[str, Any]) -> None:
    mm = make_facade(ref, "pr5", 5, 5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = LogNormalFwdRatePc(mm, gf, num, 0)
    f = _evolve_final_forwards(evolver)
    for i in range(5):
        tight(f[i], ref[f"pc_fwd_{i}"])


def test_euler_evolved_forwards(ref: dict[str, Any]) -> None:
    mm = make_facade(ref, "pr5", 5, 5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = LogNormalFwdRateEuler(mm, gf, num, 0)
    f = _evolve_final_forwards(evolver)
    for i in range(5):
        tight(f[i], ref[f"euler_fwd_{i}"])


def test_ipc_evolved_forwards(ref: dict[str, Any]) -> None:
    mm = make_facade(ref, "pr5", 5, 5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = LogNormalFwdRateIpc(mm, gf, num, 0)
    f = _evolve_final_forwards(evolver)
    for i in range(5):
        tight(f[i], ref[f"ipc_fwd_{i}"])


def test_pc_money_market_measure(ref: dict[str, Any]) -> None:
    # spot (money-market) measure exercises the non-terminal drift branch.
    mm = make_facade(ref, "pr5", 5, 5)
    gf = MTBrownianGeneratorFactory(7)
    num = money_market_measure(mm.evolution())
    evolver = LogNormalFwdRatePc(mm, gf, num, 0)
    f = _evolve_final_forwards(evolver)
    for i in range(5):
        tight(f[i], ref[f"pc_mm_fwd_{i}"])


def test_pc_two_factor_reduced(ref: dict[str, Any]) -> None:
    # 2-factor reduced model exercises the computeReduced drift path.
    mm = make_facade(ref, "pr2", 5, 2)
    gf = MTBrownianGeneratorFactory(99)
    num = terminal_measure(mm.evolution())
    evolver = LogNormalFwdRatePc(mm, gf, num, 0)
    f = _evolve_final_forwards(evolver)
    for i in range(5):
        tight(f[i], ref[f"pc_2f_fwd_{i}"])


def test_euler_constrained_inactive_matches_euler(ref: dict[str, Any]) -> None:
    # With no active constraints, the constrained Euler evolver must reproduce
    # the plain Euler evolver step for step (same RNG stream, weight 1.0).
    mm = make_facade(ref, "pr5", 5, 5)
    num = terminal_measure(mm.evolution())

    plain = LogNormalFwdRateEuler(mm, MTBrownianGeneratorFactory(42), num, 0)
    f_plain = _evolve_final_forwards(plain)

    constrained = LogNormalFwdRateEulerConstrained(
        mm, MTBrownianGeneratorFactory(42), num, 0
    )
    constrained.set_constraint_type(
        list(range(len(num))), [i + 1 for i in range(len(num))]
    )
    # dummy (inactive) constraint values must be positive (C++ takes their log
    # unconditionally); the inactive flag means they are never applied.
    constrained.set_this_constraint([0.05] * len(num), [False] * len(num))
    constrained.start_new_path()
    total_weight = 1.0
    for _ in range(len(num)):
        total_weight *= constrained.advance_step()
    f_con = list(constrained.current_state().forward_rates())

    assert total_weight == pytest.approx(1.0, abs=1e-15)
    for i in range(5):
        tight(f_con[i], f_plain[i])


def test_euler_constrained_active_constraint_hits_target(ref: dict[str, Any]) -> None:
    # An active single-forward-rate constraint must force that rate's evolved
    # value to the target (within numerical precision), and produce a non-unit
    # likelihood-ratio weight.
    mm = make_facade(ref, "pr5", 5, 5)
    num = terminal_measure(mm.evolution())
    constrained = LogNormalFwdRateEulerConstrained(
        mm, MTBrownianGeneratorFactory(42), num, 0
    )
    constrained.set_constraint_type(
        list(range(len(num))), [i + 1 for i in range(len(num))]
    )
    target = 0.05
    active = [False] * len(num)
    active[0] = True  # constrain rate 0 on step 0
    constrained.set_this_constraint([target] * len(num), active)
    constrained.start_new_path()
    w0 = constrained.advance_step()
    # after step 0, forward rate 0 is fixed at the target.
    f = list(constrained.current_state().forward_rates())
    tight(f[0], target)
    assert w0 != pytest.approx(1.0, abs=1e-9)
