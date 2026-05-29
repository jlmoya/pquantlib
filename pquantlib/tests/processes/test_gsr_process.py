"""Tests for the GSR stochastic process.

Cross-validates against ``migration-harness/references/cluster/l10b.json``.

C++ parity:
- ``ql/processes/gsrprocess.{hpp,cpp}`` @ v1.42.1
- ``ql/processes/gsrprocesscore.{hpp,cpp}`` @ v1.42.1
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.processes.gsr_process import GsrProcess
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l10b")


def _build_const_process() -> GsrProcess:
    """Single-piece GsrProcess: sigma=0.01, kappa=0.05, T=10."""
    return GsrProcess(
        times=np.array([], dtype=np.float64),
        vols=np.array([0.01], dtype=np.float64),
        reversions=np.array([0.05], dtype=np.float64),
        T=10.0,
    )


def test_gsr_process_inspectors(reference_data: dict[str, Any]) -> None:
    """``sigma(t)``, ``reversion(t)``, ``x0()`` are constant in the const piece."""
    ref = reference_data["gsr_process_const"]
    p = _build_const_process()
    tight(p.sigma(0.5), ref["sigma"])
    tight(p.reversion(0.5), ref["reversion"])
    tight(p.x0(), ref["x0"])


def test_gsr_process_diffusion(reference_data: dict[str, Any]) -> None:
    """``diffusion_1d(t, x) = sigma(t)`` — x-independent."""
    ref = reference_data["gsr_process_const"]
    p = _build_const_process()
    tight(p.diffusion_1d(0.5, 0.0), ref["diffusion_t_0_5"])


def test_gsr_process_variance(reference_data: dict[str, Any]) -> None:
    """Conditional variance over [0, dt]; matches affine-form OU limit (TIGHT)."""
    ref = reference_data["gsr_process_const"]
    p = _build_const_process()
    # TIGHT: with a single piece + constant reversion the GsrProcess
    # variance reduces to the closed-form OU variance — no piecewise
    # boundary effects, so the match is bit-true to ~13 digits.
    tight(p.variance_1d(0.0, 0.0, 1.0), ref["variance_0_1"])
    tight(p.variance_1d(0.0, 0.0, 5.0), ref["variance_0_5"])
    tight(p.std_deviation_1d(0.0, 0.0, 5.0), ref["std_deviation_0_5"])


def test_gsr_process_y(reference_data: dict[str, Any]) -> None:
    """``y(t)`` — accumulated quadratic forward-state covariance."""
    ref = reference_data["gsr_process_const"]
    p = _build_const_process()
    tight(p.y(0.0), ref["y_0"])
    tight(p.y(1.0), ref["y_1"])
    tight(p.y(5.0), ref["y_5"])


def test_gsr_process_G(reference_data: dict[str, Any]) -> None:  # noqa: N802 — G is the C++ name
    """``G(t, w)`` — Hull-White-style integral."""
    ref = reference_data["gsr_process_const"]
    p = _build_const_process()
    tight(p.G(0.0, 1.0, 0.0), ref["G_0_1"])
    tight(p.G(0.0, 5.0, 0.0), ref["G_0_5"])
    tight(p.G(1.0, 5.0, 0.0), ref["G_1_5"])


def test_gsr_process_expectation(reference_data: dict[str, Any]) -> None:
    """Conditional expectation incorporates the T-forward measure drift correction."""
    ref = reference_data["gsr_process_const"]
    p = _build_const_process()
    # Both expectations include x0 dep + rn part + tf part; the tf
    # correction is small (~1e-4 to 1e-3) — TIGHT match is fine.
    tight(p.expectation_1d(0.0, 0.0, 1.0), ref["expectation_0_0_1"])
    tight(p.expectation_1d(1.0, 0.0, 1.0), ref["expectation_1_0_2"])


def test_gsr_process_forward_measure_time_default() -> None:
    """Default ``T`` and explicit set/get round-trip."""
    p = _build_const_process()
    assert p.get_forward_measure_time() == 10.0
    p.set_forward_measure_time(20.0)
    assert p.get_forward_measure_time() == 20.0


def test_gsr_process_check_T_rejects_out_of_range() -> None:  # noqa: N802 — T is the C++ name
    """Past the forward measure horizon, the process raises."""
    p = _build_const_process()
    # T_fwd = 10, dt = 11 -> t0 + dt = 11 > T_fwd.
    with pytest.raises(LibraryException, match="must not be greater"):
        p.variance_1d(0.0, 0.0, 11.0)


def test_gsr_process_cache_hit_is_cheap() -> None:
    """A second call to ``y``/``G``/``variance`` shouldn't re-run the inner loops.

    We can't measure performance directly in a unit test, but we can
    check that flushing the cache makes the second call return the
    same value (which would also fail if the cache were ever stale).
    """
    p = _build_const_process()
    v1 = p.variance_1d(0.0, 0.0, 3.0)
    v2 = p.variance_1d(0.0, 0.0, 3.0)
    assert v1 == v2
    p.flush_cache()
    v3 = p.variance_1d(0.0, 0.0, 3.0)
    assert v1 == v3


def test_gsr_process_multi_piece_validates_sizes() -> None:
    """``len(times) == len(vols) - 1`` is enforced at construction."""
with pytest.raises(LibraryException, match="number of volatilities"):
        GsrProcess(
            times=np.array([1.0, 2.0]),
            vols=np.array([0.01, 0.02]),  # should be 3
            reversions=np.array([0.05]),
        )


def test_gsr_process_increasing_times() -> None:
    """Step-time grid must be strictly increasing."""
with pytest.raises(LibraryException, match="times must be increasing"):
        GsrProcess(
            times=np.array([2.0, 1.0]),
            vols=np.array([0.01, 0.02, 0.03]),
            reversions=np.array([0.05]),
        )


def test_gsr_process_reversion_size_validation() -> None:
    """Reversion size must be 1 or len(times)+1."""
with pytest.raises(LibraryException, match="number of reversions"):
        GsrProcess(
            times=np.array([1.0, 2.0]),
            vols=np.array([0.01, 0.02, 0.03]),
            reversions=np.array([0.05, 0.05]),  # should be 1 or 3
        )


def test_gsr_process_loose_ou_equivalence() -> None:
    """One-piece GsrProcess at constant sigma + reversion is OU under fwd measure.

    LOOSE: the GsrProcess's T-forward expectation differs from the
    pure-OU expectation by the ``G(t, T) sigma^2`` drift correction.
    But the std-deviation (which doesn't involve the correction) is
    bit-true OU.
    """
    sigma = 0.02
    kappa = 0.1
    horizon = 30.0
    p = GsrProcess(
        times=np.array([], dtype=np.float64),
        vols=np.array([sigma], dtype=np.float64),
        reversions=np.array([kappa], dtype=np.float64),
        T=horizon,
    )
    # OU variance over [0, 1]: sigma^2/(2*kappa) * (1 - exp(-2*kappa*1)).
    expected_var = sigma * sigma / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa))
    loose(p.variance_1d(0.0, 0.0, 1.0), expected_var)


def test_gsr_process_set_vols_then_explicit_flush() -> None:
    """``set_vols`` + ``flush_cache`` recomputes; ``set_vols`` alone may return cached.

    Matches the C++ contract (gsr.cpp:121-127 + gsrprocess.cpp:setVols):
    the GsrProcess setter only updates the underlying piecewise array;
    a flush is required for downstream cached quantities to be
    recomputed. The owning ``Gsr`` model orchestrates the flush in
    ``Gsr::updateVolatility``.
    """
    p = _build_const_process()
    _ = p.variance_1d(0.0, 0.0, 1.0)
    p.set_vols(np.array([0.02], dtype=np.float64))
    p.flush_cache()
    new_var = p.variance_1d(0.0, 0.0, 1.0)
    # The new variance should be roughly 4x larger (sigma doubled).
    expected_factor_min = 3.5
    expected_factor_max = 4.5
    initial = 0.01 * 0.01 / (2 * 0.05) * (1.0 - np.exp(-2 * 0.05))
    ratio = new_var / initial
    assert expected_factor_min < ratio < expected_factor_max
