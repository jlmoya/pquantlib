"""Tests for the abstract TwoFactorModel base + ShortRateModel.

# C++ parity: ql/models/shortrate/twofactormodel.{hpp,cpp} +
#             ql/models/model.hpp::ShortRateModel @ v1.42.1.
"""

from __future__ import annotations

import math

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.model import CalibratedModel
from pquantlib.models.shortrate.short_rate_model import ShortRateModel
from pquantlib.models.shortrate.two_factor_model import ShortRateDynamics, TwoFactorModel
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess


def test_short_rate_model_is_calibrated_model() -> None:
    """ShortRateModel inherits from CalibratedModel.

    # C++ parity: model.hpp:141 — ``class ShortRateModel : public CalibratedModel``.
    """
    assert issubclass(ShortRateModel, CalibratedModel)


def test_two_factor_model_is_short_rate_model() -> None:
    """TwoFactorModel inherits from ShortRateModel.

    # C++ parity: twofactormodel.hpp:37 — ``class TwoFactorModel : public ShortRateModel``.
    """
    assert issubclass(TwoFactorModel, ShortRateModel)


def test_short_rate_dynamics_exposes_processes_and_correlation() -> None:
    """ShortRateDynamics base exposes x/y processes + correlation accessors.

    # C++ parity: twofactormodel.hpp:72-104.
    """
    x_proc = OrnsteinUhlenbeckProcess(0.1, 0.01)
    y_proc = OrnsteinUhlenbeckProcess(0.2, 0.02)

    class _Concrete(ShortRateDynamics):
        def short_rate(self, t: float, x: float, y: float) -> float:
            return x + y

    dyn = _Concrete(x_proc, y_proc, -0.5)
    assert dyn.x_process is x_proc
    assert dyn.y_process is y_proc
    assert dyn.correlation == -0.5
    assert math.isclose(dyn.short_rate(0.0, 0.01, 0.02), 0.03, abs_tol=1e-14)


def test_tree_raises_deferred_exception() -> None:
    """tree(grid) on TwoFactorModel raises LibraryException pending Lattice2D port.

    # C++ parity: twofactormodel.cpp:29-40 — builds TreeLattice2D from
    # two TrinomialTrees. Both deferred per L4 carve-out.
    """
    # Minimal concrete subclass to exercise tree().
    class _Stub(TwoFactorModel):
        def dynamics(self) -> ShortRateDynamics:
            class _D(ShortRateDynamics):
                def short_rate(self, t: float, x: float, y: float) -> float:
                    return 0.0

            return _D(
                OrnsteinUhlenbeckProcess(0.1, 0.01),
                OrnsteinUhlenbeckProcess(0.1, 0.01),
                0.0,
            )

    stub = _Stub(n_params=0)
    with pytest.raises(LibraryException, match="Lattice"):
        stub.tree(grid=None)  # type: ignore[arg-type]
