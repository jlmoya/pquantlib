"""Tests for the W10-B Balland / Normal / swap-rate / SVD evolvers (batch b).

Cross-validates against ``migration-harness/references/cluster/w10b.json``.

Same cross-validation design as ``test_lmm_fwd_rate_evolvers``: bit-identical MT
Brownian stream + a ``PseudoRootFacade`` rebuilt from the probe-emitted
pseudo-roots (removing the spectral eigenvector-sign ambiguity), so the evolved
rates match C++ TIGHT.

C++ parity:
  ql/models/marketmodels/evolvers/lognormalfwdrateballand.{hpp,cpp}
  ql/models/marketmodels/evolvers/lognormalfwdrateiballand.{hpp,cpp}
  ql/models/marketmodels/evolvers/normalfwdratepc.{hpp,cpp}
  ql/models/marketmodels/evolvers/lognormalcotswapratepc.{hpp,cpp}
  ql/models/marketmodels/evolvers/lognormalcmswapratepc.{hpp,cpp}
  ql/models/marketmodels/evolvers/svddfwdratepc.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.models.marketmodels.browniangenerators.mt_brownian_generator import (
    MTBrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.evolution_description import terminal_measure
from pquantlib.models.marketmodels.evolvers.lognormal_cmswap_rate_pc import (
    LogNormalCmSwapRatePc,
)
from pquantlib.models.marketmodels.evolvers.lognormal_cotswap_rate_pc import (
    LogNormalCotSwapRatePc,
)
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_balland import (
    LogNormalFwdRateBalland,
)
from pquantlib.models.marketmodels.evolvers.lognormal_fwd_rate_iballand import (
    LogNormalFwdRateIBalland,
)
from pquantlib.models.marketmodels.evolvers.normal_fwd_rate_pc import NormalFwdRatePc
from pquantlib.models.marketmodels.evolvers.svdd_fwd_rate_pc import SVDDFwdRatePc
from pquantlib.models.marketmodels.evolvers.volprocesses.square_root_andersen import (
    SquareRootAndersen,
)
from pquantlib.models.marketmodels.models.pseudo_root_facade import PseudoRootFacade
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w10b")


def rate_times6() -> list[float]:
    return [0.5 * (i + 1) for i in range(6)]


def make_facade(
    ref: dict[str, Any], prefix: str, n: int, factors: int
) -> PseudoRootFacade:
    pseudo_roots: list[np.ndarray] = []
    for s in range(n):
        a = np.zeros((n, factors), dtype=np.float64)
        for i in range(n):
            for k in range(factors):
                a[i, k] = ref[f"{prefix}_pr_{s}_{i}_{k}"]
        pseudo_roots.append(a)
    initial_rates = [ref[f"{prefix}_init_{i}"] for i in range(n)]
    return PseudoRootFacade(pseudo_roots, rate_times6(), initial_rates, [0.0] * n)


def test_balland_evolved_forwards(ref: dict[str, Any]) -> None:
    mm = make_facade(ref, "pr5", 5, 5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = LogNormalFwdRateBalland(mm, gf, num, 0)
    evolver.start_new_path()
    for _ in range(len(num)):
        evolver.advance_step()
    f = list(evolver.current_state().forward_rates())
    for i in range(5):
        tight(f[i], ref[f"balland_fwd_{i}"])


def test_iballand_evolved_forwards(ref: dict[str, Any]) -> None:
    mm = make_facade(ref, "pr5", 5, 5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = LogNormalFwdRateIBalland(mm, gf, num, 0)
    evolver.start_new_path()
    for _ in range(len(num)):
        evolver.advance_step()
    f = list(evolver.current_state().forward_rates())
    for i in range(5):
        tight(f[i], ref[f"iballand_fwd_{i}"])


def test_normal_pc_evolved_forwards(ref: dict[str, Any]) -> None:
    mm = make_facade(ref, "pr5", 5, 5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = NormalFwdRatePc(mm, gf, num, 0)
    evolver.start_new_path()
    for _ in range(len(num)):
        evolver.advance_step()
    f = list(evolver.current_state().forward_rates())
    for i in range(5):
        tight(f[i], ref[f"normal_pc_fwd_{i}"])


def test_cotswap_pc_evolved_rates(ref: dict[str, Any]) -> None:
    # the cotswap_pr facade's initial rates are coterminal swap rates.
    mm = make_facade(ref, "cotswap_pr", 5, 5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = LogNormalCotSwapRatePc(mm, gf, num, 0)
    evolver.start_new_path()
    for _ in range(len(num)):
        evolver.advance_step()
    sr = list(evolver.current_state().coterminal_swap_rates())
    for i in range(5):
        tight(sr[i], ref[f"cotswap_sr_{i}"])
    f = list(evolver.current_state().forward_rates())
    for i in range(5):
        tight(f[i], ref[f"cotswap_fwd_{i}"])


def test_cmswap_pc_evolved_rates(ref: dict[str, Any]) -> None:
    span = 2
    mm = make_facade(ref, "pr5", 5, 5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = LogNormalCmSwapRatePc(span, mm, gf, num, 0)
    evolver.start_new_path()
    for _ in range(len(num)):
        evolver.advance_step()
    sr = list(evolver.current_state().cm_swap_rates(span))
    for i in range(5):
        tight(sr[i], ref[f"cmswap_sr_{i}"])
    f = list(evolver.current_state().forward_rates())
    for i in range(5):
        tight(f[i], ref[f"cmswap_fwd_{i}"])


def test_svdd_evolved_forwards(ref: dict[str, Any]) -> None:
    mm = make_facade(ref, "pr5", 5, 5)
    evol_times = mm.evolution().evolution_times()
    vol = SquareRootAndersen(1.0, 1.0, 0.3, 1.0, evol_times, 2, 0.5, 0.5, 1.5)
    gf = MTBrownianGeneratorFactory(42)
    num = terminal_measure(mm.evolution())
    evolver = SVDDFwdRatePc(mm, gf, vol, 5, 1, num, 0)
    evolver.start_new_path()
    for _ in range(len(num)):
        evolver.advance_step()
    f = list(evolver.current_state().forward_rates())
    for i in range(5):
        tight(f[i], ref[f"svdd_fwd_{i}"])
