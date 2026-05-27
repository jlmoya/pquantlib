"""Model abstract bases — behavioral tests.

Cross-validates against the L4-A foundations probe for the
``calibrate()`` orchestration end-to-end (the toy model drives the
calibration error of three synthetic helpers to zero by adjusting
its single Parameter to the average of the market values).

No new C++ probe is needed — the calibration glue is composition of
already-probed pieces (LevenbergMarquardt, Simplex, Parameter,
Constraint). Tests focus on the orchestration semantics.
"""

from __future__ import annotations

from typing import override

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.models.model import (
    CalibratedModel,
    Model,
    TermStructureConsistentModel,
)
from pquantlib.models.parameter import ConstantParameter, PiecewiseConstantParameter
from pquantlib.testing import tolerance

# ---------------------------------------------------------------------
# Minimal toy model + helper for orchestration tests (CalibrationHelper-
# like duck type; the real ``CalibrationHelper`` ABC lands in Stage 4).
# ---------------------------------------------------------------------


class _ToyHelper:
    """Minimal CalibrationHelper-like object for orchestration tests.

    Duck-types ``calibration_error()``. The Stage 4 ``CalibrationHelper``
    ABC will formalize this protocol.
    """

    def __init__(self, model: _ToyModel, market_value: float) -> None:
        self._model: _ToyModel = model
        self._market: float = market_value

    def calibration_error(self) -> float:
        return self._market - self._model.r()


class _ToyModel(CalibratedModel):
    """Single-parameter model whose value is just ``arguments[0]`` at t=0.

    ``calibrate`` to a list of toy helpers should drive ``r`` to the
    mean of their market values (since the optimizer minimizes
    sum((market_i - r)^2), the minimum is the arithmetic mean).
    """

    def __init__(self, r0: float = 0.0) -> None:
        super().__init__(1)
        self.arguments[0] = ConstantParameter(r0, NoConstraint())

    def r(self) -> float:
        return self.arguments[0](0.0)


class _TwoArgModel(CalibratedModel):
    """Two-argument model — Constant + PiecewiseConstant, used as a fixture."""

    def __init__(self) -> None:
        super().__init__(2)
        self.arguments[0] = ConstantParameter(0.1, NoConstraint())
        self.arguments[1] = PiecewiseConstantParameter([1.0, 2.0])
        self.arguments[1].set_param(0, 0.2)
        self.arguments[1].set_param(1, 0.3)
        self.arguments[1].set_param(2, 0.4)


class _TwoArgZeroModel(CalibratedModel):
    """Two-argument model with both arguments zero-initialized."""

    def __init__(self) -> None:
        super().__init__(2)
        self.arguments[0] = ConstantParameter(0.0, NoConstraint())
        self.arguments[1] = PiecewiseConstantParameter([1.0])


# ---------------------------------------------------------------------
# CalibratedModel
# ---------------------------------------------------------------------


def test_calibrated_model_params_flatten() -> None:
    m = _TwoArgModel()
    p = m.params()
    assert p.shape == (4,)
    tolerance.tight(float(p[0]), 0.1)
    tolerance.tight(float(p[1]), 0.2)
    tolerance.tight(float(p[2]), 0.3)
    tolerance.tight(float(p[3]), 0.4)


def test_calibrated_model_set_params_round_trip() -> None:
    m = _TwoArgZeroModel()
    new_params = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    m.set_params(new_params)
    out = m.params()
    tolerance.exact(float(out[0]), 0.5)
    tolerance.exact(float(out[1]), 0.6)
    tolerance.exact(float(out[2]), 0.7)


def test_calibrated_model_set_params_size_mismatch_raises() -> None:
    m = _ToyModel(r0=0.0)
    with pytest.raises(LibraryException, match="too big"):
        m.set_params(np.array([0.1, 0.2], dtype=np.float64))


def test_calibrate_drives_toy_model_to_mean() -> None:
    # Three helpers with market values [0.10, 0.20, 0.30]; the LS
    # minimum is the mean = 0.20.
    m = _ToyModel(r0=0.0)
    helpers = [_ToyHelper(m, 0.10), _ToyHelper(m, 0.20), _ToyHelper(m, 0.30)]
    method = LevenbergMarquardt()
    ec = EndCriteria(1000, 100, 1e-12, 1e-12, 1e-12)
    m.calibrate(helpers, method, ec)
    tolerance.loose(m.r(), 0.20)
    assert m.end_criteria not in (Type.MaxIterations, Type.Unknown)


def test_calibrate_function_evaluation_counter() -> None:
    m = _ToyModel(r0=0.0)
    helpers = [_ToyHelper(m, 0.10), _ToyHelper(m, 0.20)]
    method = LevenbergMarquardt()
    ec = EndCriteria(1000, 100, 1e-12, 1e-12, 1e-12)
    m.calibrate(helpers, method, ec)
    assert m.function_evaluation > 0


def test_calibrate_problem_values_at_optimum() -> None:
    m = _ToyModel(r0=0.0)
    helpers = [_ToyHelper(m, 0.10), _ToyHelper(m, 0.20), _ToyHelper(m, 0.30)]
    method = LevenbergMarquardt()
    ec = EndCriteria(1000, 100, 1e-12, 1e-12, 1e-12)
    m.calibrate(helpers, method, ec)
    # At r=0.20, residuals are [-0.10, 0.00, +0.10].
    pv = m.problem_values
    assert pv.shape == (3,)
    tolerance.loose(float(pv[0]), -0.10)
    tolerance.loose(float(pv[1]), 0.00)
    tolerance.loose(float(pv[2]), 0.10)


def test_calibrate_with_weights() -> None:
    m = _ToyModel(r0=0.0)
    helpers = [_ToyHelper(m, 0.10), _ToyHelper(m, 0.30)]
    method = LevenbergMarquardt()
    ec = EndCriteria(1000, 100, 1e-12, 1e-12, 1e-12)
    # Weights {3, 1}: minimum is at the weighted mean
    # (3*0.10 + 1*0.30) / 4 = 0.15.
    m.calibrate(helpers, method, ec, weights=[3.0, 1.0])
    tolerance.loose(m.r(), 0.15)


def test_calibrate_no_instruments_raises() -> None:
    m = _ToyModel()
    method = LevenbergMarquardt()
    ec = EndCriteria(1000, 100, 1e-12, 1e-12, 1e-12)
    with pytest.raises(LibraryException, match="no instruments"):
        m.calibrate([], method, ec)


def test_calibrate_weights_mismatch_raises() -> None:
    m = _ToyModel()
    helpers = [_ToyHelper(m, 0.10), _ToyHelper(m, 0.20)]
    method = LevenbergMarquardt()
    ec = EndCriteria(1000, 100, 1e-12, 1e-12, 1e-12)
    with pytest.raises(LibraryException, match=r"mismatch.*weights"):
        m.calibrate(helpers, method, ec, weights=[1.0, 2.0, 3.0])


# ---------------------------------------------------------------------
# TermStructureConsistentModel
# ---------------------------------------------------------------------


def test_term_structure_consistent_model_holds_curve() -> None:
    class _DummyCurve:
        pass

    curve = _DummyCurve()
    m = TermStructureConsistentModel(curve)  # type: ignore[arg-type]
    assert m.term_structure is curve


# ---------------------------------------------------------------------
# Model abstract base
# ---------------------------------------------------------------------


def test_model_is_abstract() -> None:
    with pytest.raises(TypeError, match="Can't instantiate"):
        Model()  # type: ignore[abstract]


def test_calibrated_model_implements_model() -> None:
    m = _ToyModel(r0=0.0)
    assert isinstance(m, Model)


def test_calibrated_model_update_notifies_observers() -> None:
    calls: list[int] = []

    class _Observer:
        def update(self) -> None:
            calls.append(1)

    obs = _Observer()
    m = _ToyModel(r0=0.0)
    m.register_with(obs)
    m.update()
    assert len(calls) == 1


def test_generate_arguments_is_default_noop() -> None:
    class _CustomModel(CalibratedModel):
        def __init__(self) -> None:
            super().__init__(1)
            self.arguments[0] = ConstantParameter(0.0, NoConstraint())
            self.generate_args_called = 0

        @override
        def generate_arguments(self) -> None:
            self.generate_args_called += 1

    m = _CustomModel()
    m.update()
    assert m.generate_args_called == 1


def test_calibrated_model_value_evaluates_rms_error() -> None:
    m = _ToyModel(r0=0.20)
    helpers = [_ToyHelper(m, 0.10), _ToyHelper(m, 0.20), _ToyHelper(m, 0.30)]
    # At r=0.20: residuals [-0.10, 0.00, 0.10]; RMS = sqrt(0.02) ~ 0.1414.
    v = m.value(np.array([0.20], dtype=np.float64), helpers)
    tolerance.loose(v, float(np.sqrt(0.02)))


class _TwoParamModel(CalibratedModel):
    """Model with two scalar parameters whose ``sum()`` is the calibrated value."""

    def __init__(self) -> None:
        super().__init__(2)
        self.arguments[0] = ConstantParameter(0.10, NoConstraint())
        self.arguments[1] = ConstantParameter(0.00, NoConstraint())

    def sum_(self) -> float:
        return self.arguments[0](0.0) + self.arguments[1](0.0)


class _SumHelper:
    def __init__(self, model: _TwoParamModel, target: float) -> None:
        self._m: _TwoParamModel = model
        self._t: float = target

    def calibration_error(self) -> float:
        return self._t - self._m.sum_()


def test_calibrate_with_fix_parameters() -> None:
    # Two-param model: r and offset. Fix r=0.10 (free=offset only);
    # helpers want r+offset = market => offset = market - 0.10 each.
    m = _TwoParamModel()
    helpers = [_SumHelper(m, 0.20), _SumHelper(m, 0.20)]
    method = LevenbergMarquardt()
    ec = EndCriteria(1000, 100, 1e-12, 1e-12, 1e-12)
    # Fix first param; optimize second only.
    m.calibrate(helpers, method, ec, fix_parameters=[True, False])
    # r should remain 0.10; offset should become 0.10.
    tolerance.exact(m.arguments[0](0.0), 0.10)
    tolerance.loose(m.arguments[1](0.0), 0.10)
