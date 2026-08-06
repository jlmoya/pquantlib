"""Cross-validation of the LIBOR-market-model volatility models against C++ v1.43.

Covers ``LmVolatilityModel``, ``LmLinearExponentialVolatilityModel``,
``LmExtLinearExponentialVolModel``, ``LmFixedVolatilityModel`` and
``LmConstWrapperVolatilityModel``.

Every expected value comes from
``migration-harness/references/v143/legacy/lmm.json``, produced by
``migration-harness/cpp/probes/v143_legacy_lmm/probe.cpp``. None of the classes
here reads a date or the evaluation date, so no ``Settings`` pinning is needed
in this module (the process / model modules do pin it).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.legacy.libormarketmodels.lm_const_wrapper_vol_model import (
    LmConstWrapperVolatilityModel,
)
from pquantlib.legacy.libormarketmodels.lm_ext_lin_exp_vol_model import (
    LmExtLinearExponentialVolModel,
)
from pquantlib.legacy.libormarketmodels.lm_fixed_vol_model import LmFixedVolatilityModel
from pquantlib.legacy.libormarketmodels.lm_lin_exp_vol_model import (
    LmLinearExponentialVolatilityModel,
)
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.models.parameter import ConstantParameter
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight

# probe.cpp: kA / kB / kC / kD — mutually distinct, none 0 or 1.
A: Final[float] = 0.23
B: Final[float] = 0.17
C: Final[float] = 0.11
D: Final[float] = 0.29
SIZE: Final[int] = 6


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/legacy/lmm")


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    assert act.shape == exp.shape, f"{label}: shape {act.shape} != {exp.shape}"
    for idx, (a, e) in enumerate(zip(act.ravel(), exp.ravel(), strict=True)):
        tight(float(a), float(e), reason=f"{label}[{idx}]")


# --- LmLinearExponentialVolatilityModel --------------------------------------


@pytest.fixture(scope="module")
def linexp(cpp: dict[str, Any]) -> LmLinearExponentialVolatilityModel:
    return LmLinearExponentialVolatilityModel(
        cpp["lm_linexp_volatility"]["fixing_times"], A, B, C, D
    )


def test_linexp_size_and_params_come_from_the_constructor(
    cpp: dict[str, Any], linexp: LmLinearExponentialVolatilityModel
) -> None:
    """All four coefficients are distinct, so a swapped pair cannot hide."""
    ref = cpp["lm_linexp_volatility"]
    assert linexp.size() == ref["size"]
    assert linexp.fixing_times() == ref["fixing_times"]
    _assert_close([p(0.0) for p in linexp.params()], ref["params"], "linexp.params")
    assert [p(0.0) for p in linexp.params()] == [A, B, C, D]


def test_linexp_volatility_vector_matches_cpp(
    cpp: dict[str, Any], linexp: LmLinearExponentialVolatilityModel
) -> None:
    ref = cpp["lm_linexp_volatility"]
    got = [linexp.volatility(t) for t in ref["eval_times"]]
    _assert_close(got, ref["volatility_vector"], "linexp.volatility")


def test_linexp_volatility_scalar_matches_the_vector_form(
    cpp: dict[str, Any], linexp: LmLinearExponentialVolatilityModel
) -> None:
    ref = cpp["lm_linexp_volatility"]
    got = [
        [linexp.volatility_scalar(i, t) for i in range(SIZE)] for t in ref["eval_times"]
    ]
    _assert_close(got, ref["volatility_scalar"], "linexp.volatility_scalar")


@pytest.mark.parametrize(("u", "key"), [(1.7, "integrated_variance_u1_7"), (4.0, "integrated_variance_u4_0")])
def test_linexp_integrated_variance_matches_cpp(
    cpp: dict[str, Any], linexp: LmLinearExponentialVolatilityModel, u: float, key: str
) -> None:
    """The closed form is pinned on the FULL (i, j) grid at two horizons."""
    ref = cpp["lm_linexp_volatility"]
    got = [[linexp.integrated_variance(i, j, u) for j in range(SIZE)] for i in range(SIZE)]
    _assert_close(got, ref[key], f"linexp.integrated_variance[{u}]")


def test_linexp_set_params_takes_effect_immediately() -> None:
    """The (a, b, c, d) are re-read on every call, so setParams needs no cache flush."""
    model = LmLinearExponentialVolatilityModel([0.5, 1.0, 1.5], A, B, C, D)
    before = model.volatility(0.25).copy()
    model.set_params(
        [ConstantParameter(v, PositiveConstraint()) for v in (0.9, 0.4, 0.2, 0.7)]
    )
    after = model.volatility(0.25)
    assert not np.allclose(before, after)
    assert [p(0.0) for p in model.params()] == [0.9, 0.4, 0.2, 0.7]


# --- LmExtLinearExponentialVolModel ------------------------------------------


def test_ext_linexp_starts_with_unit_scalings_and_equals_its_base(
    cpp: dict[str, Any],
) -> None:
    ref = cpp["lm_ext_linexp_volatility"]
    model = LmExtLinearExponentialVolModel(
        cpp["lm_linexp_volatility"]["fixing_times"], A, B, C, D
    )
    assert model.size() == ref["size"]
    assert len(model.params()) == ref["n_params"] == 4 + SIZE
    _assert_close(
        [p(0.0) for p in model.params()], ref["params_default"], "extlinexp.params"
    )
    got = [model.volatility(t) for t in cpp["lm_linexp_volatility"]["eval_times"]]
    _assert_close(got, ref["volatility_vector_default"], "extlinexp.volatility_default")


@pytest.fixture(scope="module")
def ext_scaled(cpp: dict[str, Any]) -> LmExtLinearExponentialVolModel:
    """probe.cpp installs k_i = 0.8 + 0.05 i — all distinct, none equal to 1."""
    model = LmExtLinearExponentialVolModel(
        cpp["lm_linexp_volatility"]["fixing_times"], A, B, C, D
    )
    model.set_params(
        [ConstantParameter(v, PositiveConstraint()) for v in (A, B, C, D)]
        + [
            ConstantParameter(0.8 + 0.05 * i, PositiveConstraint())
            for i in range(SIZE)
        ]
    )
    return model


def test_ext_linexp_scaled_params_round_trip(
    cpp: dict[str, Any], ext_scaled: LmExtLinearExponentialVolModel
) -> None:
    ref = cpp["lm_ext_linexp_volatility"]
    _assert_close(
        [p(0.0) for p in ext_scaled.params()], ref["params_scaled"], "extlinexp.params_scaled"
    )


def test_ext_linexp_scaled_volatility_matches_cpp(
    cpp: dict[str, Any], ext_scaled: LmExtLinearExponentialVolModel
) -> None:
    ref = cpp["lm_ext_linexp_volatility"]
    times = cpp["lm_linexp_volatility"]["eval_times"]
    _assert_close(
        [ext_scaled.volatility(t) for t in times],
        ref["volatility_vector_scaled"],
        "extlinexp.volatility_scaled",
    )
    _assert_close(
        [[ext_scaled.volatility_scalar(i, t) for i in range(SIZE)] for t in times],
        ref["volatility_scalar_scaled"],
        "extlinexp.volatility_scalar_scaled",
    )


def test_ext_linexp_scaled_integrated_variance_matches_cpp(
    cpp: dict[str, Any], ext_scaled: LmExtLinearExponentialVolModel
) -> None:
    """Scaled by k_i k_j — an asymmetric bug would show as a non-symmetric grid."""
    ref = cpp["lm_ext_linexp_volatility"]
    got = [
        [ext_scaled.integrated_variance(i, j, 1.7) for j in range(SIZE)]
        for i in range(SIZE)
    ]
    _assert_close(got, ref["integrated_variance_scaled_u1_7"], "extlinexp.iv_scaled")


# --- LmFixedVolatilityModel ---------------------------------------------------


@pytest.fixture(scope="module")
def fixed_vol(cpp: dict[str, Any]) -> LmFixedVolatilityModel:
    ref = cpp["lm_fixed_volatility"]
    return LmFixedVolatilityModel(
        np.asarray(ref["volatilities"], dtype=np.float64), ref["start_times"]
    )


def test_fixed_volatility_inspectors_return_the_constructor_arguments(
    cpp: dict[str, Any], fixed_vol: LmFixedVolatilityModel
) -> None:
    ref = cpp["lm_fixed_volatility"]
    assert fixed_vol.size() == ref["size"]
    assert len(fixed_vol.params()) == ref["n_params"] == 0
    _assert_close(fixed_vol.volatilities(), ref["volatilities"], "fixedvol.volatilities")
    assert fixed_vol.start_times() == ref["start_times"]


def test_fixed_volatility_vector_matches_cpp_across_every_bucket(
    cpp: dict[str, Any], fixed_vol: LmFixedVolatilityModel
) -> None:
    """The probe's time grid straddles every bucket boundary and both endpoints."""
    ref = cpp["lm_fixed_volatility"]
    got = [fixed_vol.volatility(t) for t in ref["eval_times"]]
    _assert_close(got, ref["volatility_vector"], "fixedvol.volatility")


def test_fixed_volatility_scalar_matches_the_vector_form_where_defined(
    cpp: dict[str, Any], fixed_vol: LmFixedVolatilityModel
) -> None:
    ref = cpp["lm_fixed_volatility"]
    expected = np.asarray(ref["volatility_scalar_where_nonzero"], dtype=np.float64)
    for k, t in enumerate(ref["eval_times"]):
        vector = fixed_vol.volatility(t)
        for i in range(int(ref["size"])):
            if vector[i] != 0.0:
                tight(
                    fixed_vol.volatility_scalar(i, t),
                    float(expected[k, i]),
                    reason=f"fixedvol.volatility_scalar[{k}][{i}]",
                )


def test_fixed_volatility_scalar_rejects_an_already_fixed_forward(
    fixed_vol: LmFixedVolatilityModel,
) -> None:
    """C++ reads volatilities_[i - ti] with unsigned Size and no bounds check.

    For ``i < ti`` that is an out-of-bounds read (undefined behaviour). Python
    would silently index from the end instead, so the port raises. The probe
    deliberately never exercises the C++ side of this.
    """
    with pytest.raises(LibraryException, match="already fixed"):
        fixed_vol.volatility_scalar(0, 4.0)


@pytest.mark.parametrize("t", [0.24, 4.41])
def test_fixed_volatility_rejects_times_outside_the_start_time_range(
    fixed_vol: LmFixedVolatilityModel, t: float
) -> None:
    with pytest.raises(LibraryException, match="invalid time given"):
        fixed_vol.volatility(t)


def test_fixed_volatility_rejects_bad_constructor_arguments() -> None:
    with pytest.raises(LibraryException, match="too few dates"):
        LmFixedVolatilityModel(np.asarray([0.1]), [0.0])
    with pytest.raises(LibraryException, match="same size"):
        LmFixedVolatilityModel(np.asarray([0.1, 0.2]), [0.0, 1.0, 2.0])
    with pytest.raises(LibraryException, match="invalid time"):
        LmFixedVolatilityModel(np.asarray([0.1, 0.2, 0.3]), [0.0, 2.0, 1.0])


def test_fixed_volatility_has_no_analytic_integrated_variance(
    cpp: dict[str, Any], fixed_vol: LmFixedVolatilityModel
) -> None:
    """The base-class failure is what routes LfmCovarianceProxy to quadrature."""
    assert cpp["lm_fixed_volatility"]["integrated_variance_raises"] is True
    with pytest.raises(LibraryException, match="not supported"):
        fixed_vol.integrated_variance(0, 0, 1.0)


# --- LmConstWrapperVolatilityModel -------------------------------------------


def test_const_wrapper_volatility_forwards_everything_but_the_params(
    cpp: dict[str, Any], linexp: LmLinearExponentialVolatilityModel
) -> None:
    ref = cpp["lm_const_wrapper_volatility"]
    wrapper = LmConstWrapperVolatilityModel(linexp)

    assert wrapper.size() == ref["size"] == linexp.size()
    # The whole point of the wrapper: an EMPTY argument vector.
    assert len(wrapper.params()) == ref["n_params"] == 0
    assert wrapper.volatility_model() is linexp

    times = cpp["lm_linexp_volatility"]["eval_times"]
    _assert_close(
        [wrapper.volatility(t) for t in times], ref["volatility_vector"], "wrapvol.volatility"
    )
    _assert_close(
        [[wrapper.volatility_scalar(i, t) for i in range(SIZE)] for t in times],
        ref["volatility_scalar"],
        "wrapvol.volatility_scalar",
    )
    _assert_close(
        [[wrapper.integrated_variance(i, j, 1.7) for j in range(SIZE)] for i in range(SIZE)],
        ref["integrated_variance_u1_7"],
        "wrapvol.integrated_variance",
    )


def test_const_wrapper_volatility_set_params_cannot_move_the_wrapped_model() -> None:
    """``set_params([])`` is accepted and is a no-op, as in C++."""
    inner = LmLinearExponentialVolatilityModel([0.5, 1.0, 1.5], A, B, C, D)
    wrapper = LmConstWrapperVolatilityModel(inner)
    before = wrapper.volatility(0.25).copy()
    wrapper.set_params([])
    assert np.array_equal(wrapper.volatility(0.25), before)
    assert [p(0.0) for p in inner.params()] == [A, B, C, D]
