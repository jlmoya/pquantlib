"""Tests for the Brigo-Mercurio G2++ Gaussian two-factor short-rate model.

Cross-validates against ``migration-harness/references/cluster/l4d.json``.

C++ parity: ql/models/shortrate/twofactormodels/g2.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.shortrate.two_factor_model import TwoFactorModel
from pquantlib.models.shortrate.twofactor.g2 import G2
from pquantlib.payoffs import OptionType
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l4d")


@pytest.fixture
def flat_5_curve() -> FlatForward:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    return FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)


@pytest.fixture
def g2(flat_5_curve: FlatForward) -> G2:
    return G2(flat_5_curve, 0.1, 0.01, 0.1, 0.01, -0.75)


class TestG2:
    """G2 ctor + closed-form discount / discount_bond / discount_bond_option."""

    def test_is_two_factor_model(self, g2: G2) -> None:
        assert isinstance(g2, TwoFactorModel)

    def test_inspectors(self, g2: G2, reference_data: dict[str, Any]) -> None:
        ref = reference_data["g2"]
        tight(g2.a(), ref["a"])
        tight(g2.sigma(), ref["sigma"])
        tight(g2.b(), ref["b"])
        tight(g2.eta(), ref["eta"])
        tight(g2.rho(), ref["rho"])

    def test_discount_curve_passthrough(
        self, g2: G2, reference_data: dict[str, Any]
    ) -> None:
        """G2.discount(t) just passes through to the term structure.

        # C++ parity: g2.hpp:85 — ``discount(t) = termStructure()->discount(t)``.
        """
        ref = reference_data["g2"]
        tight(g2.discount(1.0), ref["discount_t1"])
        tight(g2.discount(5.0), ref["discount_t5"])
        tight(g2.discount(10.0), ref["discount_t10"])

    def test_discount_bond_scalar(self, g2: G2, reference_data: dict[str, Any]) -> None:
        """discount_bond(now, T, x, y) — closed-form ``A * exp(-B*x - B*y)``.

        # C++ parity: g2.cpp:75-77.
        """
        ref = reference_data["g2"]
        tight(
            g2.discount_bond(0.5, 5.5, 0.01, -0.005),
            ref["discount_bond_0_5_5_5_x001_yneg0005"],
        )
        # At origin (x=y=0) the bond price collapses to the curve discount.
        tight(g2.discount_bond(0.0, 5.0, 0.0, 0.0), ref["discount_bond_0_5_origin"])

    def test_discount_bond_array_overload(
        self, g2: G2, reference_data: dict[str, Any]
    ) -> None:
        """The Array overload routes through to the scalar overload.

        # C++ parity: g2.hpp:67-72.
        """
        ref = reference_data["g2"]
        factors = np.array([0.005, -0.002])
        tight(g2.discount_bond(0.0, 3.0, factors), ref["discount_bond_factors_0_3_x_005_yneg002"])

    def test_discount_bond_array_too_small_rejected(self, g2: G2) -> None:
        with pytest.raises(Exception, match="needs two factors"):
            g2.discount_bond(0.0, 3.0, np.array([0.01]))

    def test_discount_bond_option(self, g2: G2, reference_data: dict[str, Any]) -> None:
        """Black-formula discount-bond option.

        # C++ parity: g2.cpp:79-87.
        """
        ref = reference_data["g2"]
        tight(g2.discount_bond_option(OptionType.Call, 0.95, 1.0, 3.0), ref["dbo_call_K0_95_T1_TB3"])
        tight(g2.discount_bond_option(OptionType.Put, 0.95, 1.0, 3.0), ref["dbo_put_K0_95_T1_TB3"])

    def test_dynamics_returns_short_rate(self, g2: G2) -> None:
        """The Dynamics object exposes x/y OU processes + analytical short-rate.

        # C++ parity: g2.cpp:48-51 and g2.hpp:118-130.
        """
        dyn = g2.dynamics()
        assert isinstance(dyn.x_process, OrnsteinUhlenbeckProcess)
        assert isinstance(dyn.y_process, OrnsteinUhlenbeckProcess)
        tight(dyn.correlation, -0.75)
        # phi(0) = forward(0,0) + drift correction. For x=y=0, r(0) = phi(0).
        # On the 5% flat-forward curve, phi(0) ~= 0.05 (the small correction
        # terms vanish at t=0 since sigma*(1-exp(0))/a = 0).
        tight(dyn.short_rate(0.0, 0.0, 0.0), 0.05000000000012957)

    def test_B_helper(self, g2: G2) -> None:  # noqa: N802 — math symbol
        """The closed-form ``B(x,t) = (1-exp(-x*t))/x``.

        # C++ parity: g2.cpp:107-109.
        """
        # At x=0.1, t=5: B = (1 - exp(-0.5))/0.1 = 3.93469...
        expected = (1.0 - math.exp(-0.5)) / 0.1
        tight(g2.B(0.1, 5.0), expected)

    def test_swaption(self, g2: G2, reference_data: dict[str, Any]) -> None:
        """5y x 5y annual payer at 5% — cross-validated against C++ G2.swaption.

        # C++ parity: ``G2::swaption`` in g2.cpp:218-246.

        Tolerance: TIGHT. Empirically the 200-segment trapezoid integral
        with an inner Brent root search matches C++ bit-exactly on this
        setup. The closed-form integrand is composed entirely of arithmetic
        + ``math.exp`` + ``CumulativeNormalDistribution`` (the latter also
        bit-exact vs C++ since L1-D), and the sum order matches the C++
        ``SegmentIntegral`` loop step by step.
        """
        ref = reference_data["g2_swaption"]
        price = g2.swaption(
            nominal=1.0,
            is_payer=True,
            start=5.0,
            fixed_pay_times=[6.0, 7.0, 8.0, 9.0, 10.0],
            fixed_rate=0.05,
            range_=5.0,
            intervals=200,
        )
        tight(price, ref["price"])

    def test_calibrated_model_params_round_trip(self, g2: G2) -> None:
        """params() returns the 5-vector (a, sigma, b, eta, rho); set_params restores.

        # C++ parity: CalibratedModel::params + setParams.
        """
        p = g2.params()
        assert p.shape == (5,)
        tight(p[0], 0.1)
        tight(p[1], 0.01)
        tight(p[2], 0.1)
        tight(p[3], 0.01)
        tight(p[4], -0.75)
        # Mutate one slot and round-trip
        new = p.copy()
        new[4] = -0.5
        g2.set_params(new)
        tight(g2.rho(), -0.5)
        # generate_arguments was triggered: discount_bond at new params
        # (with both ``now`` and ``maturity`` strictly positive — so
        # ``V(now)`` is non-trivial and rho-dependent) should differ
        # from the original-rho value.
        with_rho_neg05 = g2.discount_bond(0.5, 3.0, 0.005, -0.002)
        # Setting rho back should restore the original answer.
        g2.set_params(p)
        tight(g2.rho(), -0.75)
        with_rho_neg075 = g2.discount_bond(0.5, 3.0, 0.005, -0.002)
        # The two answers should differ by enough to be observable.
        assert abs(with_rho_neg05 - with_rho_neg075) > 1e-10
