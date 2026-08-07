"""Cross-validate SaddlePointLossModel against C++ QuantLib v1.43.

Probe source: migration-harness/cpp/probes/v143_experimental_creditloss/probe.cpp
Reference:    migration-harness/references/v143/experimental/creditloss.json

The saddlepoint model is a four-stage pipeline (cumulants -> Brent saddle
search -> high-order expansion -> market-factor quadrature), so the probe pins
each stage separately and so do these tests: a mismatch localises instead of
being smeared across the composition.

TOLERANCE TIERS
---------------
Everything is TIGHT except six arrays that depend on the located saddle point
*linearly*. The derivation is in :func:`_saddle_dependent_reason`.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.credit.saddlepoint_loss_model import (
    SaddlePointLossModel,
    remaining_probabilities,
)
from pquantlib.testing import tolerance
from tests.experimental.credit import _creditloss_fixture as fx


def _saddle_dependent_reason() -> str:
    """Why the saddle-derived quantities are LOOSE and not TIGHT.

    ``find_saddle`` runs Brent to the C++ default ``accuracy = 1e-3``
    (saddlepointlossmodel.hpp:222). The bracket endpoints and the objective
    ``K'(x) - target`` agree with C++ to 1.8e-14 relative (measured here by
    ``test_cumulant_first_derivative_conditional``), but Brent's iterate
    sequence branches on the *sign* of a residual, so a 1e-14 perturbation can
    send it down a different interpolation path; the returned root then differs
    by up to the solver's own tolerance scale. Measured: 5.2e-12 relative on
    ``saddle_found``.

    Quantities that are *stationary* in the saddle - the whole point of the
    method is that the leading term of the expansion has zero first derivative
    at s* - absorb that perturbation quadratically and stay at 1e-14
    (``prob_over_loss_portf_cond`` measures 4.9e-14). Quantities that use s*
    linearly inherit it linearly: ``prob_density_cond`` (exp(K0 - s*.l)),
    ``expected_shortfall_full_portfolio_cond`` (divides by s*),
    ``split_loss_cond`` (exp(lgd.s*/N)) and ``expected_shortfall_tranche_cond``
    (built on the previous two) measure 8.6e-13 .. 3.4e-12.

    LOOSE (1e-8 relative) therefore clears the observed residual by ~3.5 orders
    of magnitude while still being ~4 orders tighter than the Brent tolerance
    that causes it.
    """
    return (
        "linear in the Brent-located saddle point, whose own residual is "
        "5.2e-12 at the C++ default accuracy of 1e-3"
    )


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return fx.load_reference()


@pytest.fixture(scope="module")
def model_setup() -> tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]]:
    f = fx.build_fixture()
    model = SaddlePointLossModel(f.latent_model)
    f.basket.set_loss_model(model)
    probs = remaining_probabilities(f.basket, f.calc_date)
    inv = [
        f.latent_model.inverse_cumulative_y(probs[i], i) for i in range(len(probs))
    ]
    return f, model, inv


def _tight_all(actual: list[float], expected: list[float]) -> None:
    for a, e in zip(actual, expected, strict=True):
        tolerance.tight(a, e)


def _loose_all(actual: list[float], expected: list[float], reason: str) -> None:
    for a, e in zip(actual, expected, strict=True):
        tolerance.loose(a, e, reason=reason)


# -----------------------------------------------------------------------------
# Fixture agreement — everything downstream is a function of these numbers.
# -----------------------------------------------------------------------------


def test_fixture_matches_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    f, _model, _inv = model_setup
    assert f.today.serial_number() == cpp_ref["evaluation_date_serial"]
    assert f.calc_date.serial_number() == cpp_ref["saddle_calc_date_serial"]
    assert f.calc_date_2y.serial_number() == cpp_ref["saddle_calc_date2_serial"]
    tolerance.tight(f.basket.basket_notional(), cpp_ref["saddle_basket_notional"])
    tolerance.tight(f.basket.remaining_notional(), cpp_ref["saddle_remaining_notional"])
    tolerance.tight(f.basket.attachment_amount(), cpp_ref["saddle_attach_amount"])
    tolerance.tight(f.basket.detachment_amount(), cpp_ref["saddle_detach_amount"])
    tolerance.tight(f.basket.tranche_notional(), cpp_ref["saddle_tranche_notional"])
    _tight_all(fx.HAZARD_RATES, cpp_ref["saddle_hazard_rates"])
    _tight_all(fx.NOTIONALS, cpp_ref["saddle_notionals"])
    _tight_all(fx.RECOVERIES, cpp_ref["saddle_recoveries"])


def test_unconditional_default_probabilities_match_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    f, _model, inv = model_setup
    _tight_all(
        remaining_probabilities(f.basket, f.calc_date), cpp_ref["saddle_uncond_probs"]
    )
    _tight_all(inv, cpp_ref["saddle_inv_uncond_probs"])


# -----------------------------------------------------------------------------
# Block D — conditional cumulants at fixed (saddle, market factor).
# No root search involved, so a mismatch localises to the closed form.
# -----------------------------------------------------------------------------


def _cumulant_grid(
    model: SaddlePointLossModel, inv: list[float], cpp_ref: dict[str, Any]
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {k: [] for k in (
        "k0", "k1", "k2", "k3", "k4", "t0", "t2", "t3", "t4", "u0", "u2"
    )}
    for m in cpp_ref["saddle_mkt_grid"]:
        mkt = [m]
        for s in cpp_ref["saddle_grid"]:
            out["k0"].append(model.cumulant_generating_cond(inv, s, mkt))
            out["k1"].append(model.cum_gen_1st_derivative_cond(inv, s, mkt))
            out["k2"].append(model.cum_gen_2nd_derivative_cond(inv, s, mkt))
            out["k3"].append(model.cum_gen_3rd_derivative_cond(inv, s, mkt))
            out["k4"].append(model.cum_gen_4th_derivative_cond(inv, s, mkt))
            d0, d2, d3, d4 = model.cum_gen_0234_deriv_cond(inv, s, mkt)
            out["t0"].append(d0)
            out["t2"].append(d2)
            out["t3"].append(d3)
            out["t4"].append(d4)
            e0, e2 = model.cum_gen_02_deriv_cond(inv, s, mkt)
            out["u0"].append(e0)
            out["u2"].append(e2)
    return out


@pytest.fixture(scope="module")
def cumulants(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> dict[str, list[float]]:
    _f, model, inv = model_setup
    return _cumulant_grid(model, inv, cpp_ref)


@pytest.mark.parametrize(
    ("key", "ref_key"),
    [
        ("k0", "saddle_cgf_cond"),
        ("k1", "saddle_cgf1_cond"),
        ("k2", "saddle_cgf2_cond"),
        ("k3", "saddle_cgf3_cond"),
        ("k4", "saddle_cgf4_cond"),
    ],
)
def test_conditional_cumulants_match_cpp(
    cumulants: dict[str, list[float]],
    cpp_ref: dict[str, Any],
    key: str,
    ref_key: str,
) -> None:
    _tight_all(cumulants[key], cpp_ref[ref_key])


def test_cumulant_first_derivative_conditional(
    cumulants: dict[str, list[float]], cpp_ref: dict[str, Any]
) -> None:
    """Pinned separately: it is the objective the saddle search brackets."""
    _tight_all(cumulants["k1"], cpp_ref["saddle_cgf1_cond"])


@pytest.mark.parametrize(
    ("key", "ref_key"),
    [
        ("t0", "saddle_cgf0234_d0"),
        ("t2", "saddle_cgf0234_d2"),
        ("t3", "saddle_cgf0234_d3"),
        ("t4", "saddle_cgf0234_d4"),
        ("u0", "saddle_cgf02_d0"),
        ("u2", "saddle_cgf02_d2"),
    ],
)
def test_bundled_cumulant_derivatives_match_cpp(
    cumulants: dict[str, list[float]],
    cpp_ref: dict[str, Any],
    key: str,
    ref_key: str,
) -> None:
    """The 0234 / 02 bundles must agree with the one-at-a-time forms."""
    _tight_all(cumulants[key], cpp_ref[ref_key])


# -----------------------------------------------------------------------------
# Block E — the objective functors the root search consumes.
# -----------------------------------------------------------------------------


def test_saddle_objective_function_matches_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    _f, model, inv = model_setup
    values: list[float] = []
    derivatives: list[float] = []
    for m in cpp_ref["saddle_mkt_grid"]:
        mkt = [m]
        for target in cpp_ref["saddle_obj_targets"]:
            f_obj = SaddlePointLossModel.SaddleObjectiveFunction(
                model, target, inv, mkt
            )
            for x in cpp_ref["saddle_grid"]:
                values.append(f_obj(x))
                derivatives.append(f_obj.derivative(x))
    _tight_all(values, cpp_ref["saddle_obj_value"])
    _tight_all(derivatives, cpp_ref["saddle_obj_derivative"])


def test_saddle_perc_objective_function_matches_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    f, model, _inv = model_setup
    values = [
        SaddlePointLossModel.SaddlePercObjFunction(model, target, f.calc_date)(x)
        for target in cpp_ref["saddle_perc_obj_targets"]
        for x in cpp_ref["saddle_perc_obj_xs"]
    ]
    _tight_all(values, cpp_ref["saddle_perc_obj_value"])


# -----------------------------------------------------------------------------
# Block F — the located saddle point, including the two clamped regimes.
# -----------------------------------------------------------------------------


def test_find_saddle_matches_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    _f, model, inv = model_setup
    found = [
        model.find_saddle(inv, loss, [m])
        for m in cpp_ref["saddle_mkt_grid"]
        for loss in cpp_ref["saddle_loss_fractions"]
    ]
    _loose_all(found, cpp_ref["saddle_found"], _saddle_dependent_reason())


def test_find_saddle_tight_accuracy_matches_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    """Same points at accuracy 1e-12: shows how much is tolerance, not model."""
    _f, model, inv = model_setup
    found = [
        model.find_saddle(inv, loss, [m], 1e-12, 200)
        for m in cpp_ref["saddle_mkt_grid"]
        for loss in cpp_ref["saddle_loss_fractions"]
    ]
    _loose_all(found, cpp_ref["saddle_found_acc1e12"], _saddle_dependent_reason())


# -----------------------------------------------------------------------------
# Block G — conditional statistics built on the saddle point.
# -----------------------------------------------------------------------------


def test_conditional_probabilities_over_loss_match_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    _f, model, inv = model_setup
    high_order: list[float] = []
    first_order: list[float] = []
    density: list[float] = []
    esf_full: list[float] = []
    for m in cpp_ref["saddle_mkt_grid"]:
        mkt = [m]
        for loss in cpp_ref["saddle_abs_losses"]:
            high_order.append(model.prob_over_loss_portf_cond(inv, loss, mkt))
            first_order.append(
                model.prob_over_loss_portf_cond_1st_order(inv, loss, mkt)
            )
            density.append(model.prob_density_cond(inv, loss, mkt))
            esf_full.append(
                model.expected_shortfall_full_portfolio_cond(inv, loss, mkt)
            )
    # Stationary in the saddle -> TIGHT.
    _tight_all(high_order, cpp_ref["saddle_prob_over_loss_portf_cond"])
    _tight_all(first_order, cpp_ref["saddle_prob_over_loss_portf_cond_1st"])
    # Linear in the saddle -> LOOSE.
    _loose_all(density, cpp_ref["saddle_prob_density_cond"], _saddle_dependent_reason())
    _loose_all(
        esf_full,
        cpp_ref["saddle_esf_full_portfolio_cond"],
        _saddle_dependent_reason(),
    )


def test_conditional_tranche_probability_matches_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    _f, model, inv = model_setup
    values = [
        model.prob_over_loss_cond(inv, fract, [m])
        for m in cpp_ref["saddle_mkt_grid"]
        for fract in cpp_ref["saddle_tranche_fractions"]
    ]
    _tight_all(values, cpp_ref["saddle_prob_over_loss_cond"])


def test_conditional_expected_losses_match_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    _f, model, inv = model_setup
    el = [model.conditional_expected_loss(inv, [m]) for m in cpp_ref["saddle_mkt_grid"]]
    etl = [
        model.conditional_expected_tranche_loss(inv, [m])
        for m in cpp_ref["saddle_mkt_grid"]
    ]
    _tight_all(el, cpp_ref["saddle_conditional_expected_loss"])
    _tight_all(etl, cpp_ref["saddle_conditional_expected_tranche_loss"])


def test_conditional_splits_match_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    _f, model, inv = model_setup
    split: list[float] = []
    esf_split: list[float] = []
    esf_tranche: list[float] = []
    for m in cpp_ref["saddle_mkt_grid"]:
        mkt = [m]
        for loss in (5.0, 30.0, 150.0):
            split += model.split_loss_cond(inv, loss, mkt)
            esf_split += model.expected_shortfall_split_cond(inv, loss, mkt)
            esf_tranche.append(
                model.expected_shortfall_tranche_cond(inv, loss, 0.95, mkt)
            )
    _loose_all(split, cpp_ref["saddle_split_loss_cond"], _saddle_dependent_reason())
    # No saddle search at all in the ESF split — pure Gaussian moment matching.
    _tight_all(esf_split, cpp_ref["saddle_esf_split_cond"])
    _loose_all(
        esf_tranche, cpp_ref["saddle_esf_tranche_cond"], _saddle_dependent_reason()
    )


# -----------------------------------------------------------------------------
# Block H — integrated (unconditional) quantities: everything above plus the
# order-25 Gauss-Hermite quadrature over the market factor.
# -----------------------------------------------------------------------------


def test_unconditional_cumulants_match_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    f, model, _inv = model_setup
    d = f.calc_date
    grid = cpp_ref["saddle_uncond_s_grid"]
    _tight_all([model.cumulant_generating(d, s) for s in grid], cpp_ref["saddle_cgf_uncond"])
    _tight_all(
        [model.cum_gen_1st_derivative(d, s) for s in grid], cpp_ref["saddle_cgf1_uncond"]
    )
    _tight_all(
        [model.cum_gen_2nd_derivative(d, s) for s in grid], cpp_ref["saddle_cgf2_uncond"]
    )
    _tight_all(
        [model.cum_gen_3rd_derivative(d, s) for s in grid], cpp_ref["saddle_cgf3_uncond"]
    )
    _tight_all(
        [model.cum_gen_4th_derivative(d, s) for s in grid], cpp_ref["saddle_cgf4_uncond"]
    )


def test_integrated_probabilities_match_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    f, model, _inv = model_setup
    d = f.calc_date
    _tight_all(
        [model.prob_over_loss(d, x) for x in cpp_ref["saddle_tranche_fractions"]],
        cpp_ref["saddle_prob_over_loss"],
    )
    _tight_all(
        [model.prob_over_portf_loss(d, x) for x in cpp_ref["saddle_abs_losses"]],
        cpp_ref["saddle_prob_over_portf_loss"],
    )
    _tight_all(
        [model.prob_density(d, x) for x in cpp_ref["saddle_abs_losses"]],
        cpp_ref["saddle_prob_density"],
    )


def test_expected_tranche_loss_matches_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    f, model, _inv = model_setup
    tolerance.tight(
        model.expected_tranche_loss(f.calc_date),
        cpp_ref["saddle_expected_tranche_loss"],
    )
    # Second horizon, to catch anything that accidentally hard-codes the date.
    tolerance.tight(
        model.expected_tranche_loss(f.calc_date_2y),
        cpp_ref["saddle_expected_tranche_loss_2y"],
    )


def test_percentile_and_expected_shortfall_match_cpp(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
    cpp_ref: dict[str, Any],
) -> None:
    f, model, _inv = model_setup
    levels = cpp_ref["saddle_percentile_levels"]
    _tight_all(
        [model.percentile(f.calc_date, q) for q in levels], cpp_ref["saddle_percentile"]
    )
    _tight_all(
        [model.expected_shortfall(f.calc_date, q) for q in levels],
        cpp_ref["saddle_expected_shortfall"],
    )
    tolerance.tight(
        model.percentile(f.calc_date_2y, 0.95), cpp_ref["saddle_percentile_95_2y"]
    )


def test_percentile_before_evaluation_date_is_zero(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
) -> None:
    """# C++ parity: saddlepointlossmodel.hpp:854-856."""
    f, model, _inv = model_setup
    tolerance.exact(model.percentile(f.today, 0.95), 0.0)


def test_percentile_rejects_out_of_range_level(
    model_setup: tuple[fx.CreditLossFixture, SaddlePointLossModel, list[float]],
) -> None:
    """# C++ parity: saddlepointlossmodel.hpp:850-852."""
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    _f, model, _inv = model_setup
    with pytest.raises(LibraryException, match="Incorrect percentile value"):
        model.percentile(_f.calc_date, 1.5)
