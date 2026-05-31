"""AlphaFinder — solve for the alpha parameter matching a calibration target.

# C++ parity: ql/models/marketmodels/models/alphafinder.{hpp,cpp} (v1.42.1).

Given a rate-one vol vector, a rate-two homogeneous vol vector and their
correlations (plus the swap-rate weights ``w0, w1``), AlphaFinder solves for the
single alpha-form parameter (and the consequent multipliers ``a, b``) so that
the modified rate-two vols give a swap variance equal to ``targetVariance``.

The C++ class is built on a handful of templated helper minimisers
(``Bisection``, ``FindHighestOK``, ``FindLowestOK``, ``Minimize`` — golden
section) that take member-function pointers. They are ported here as small
module-level helpers taking a Python callable; the algorithm (interval
bisection / golden-section search) is reproduced verbatim so the deterministic
root path matches C++ exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from pquantlib.math.quadratic import Quadratic

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.models.alpha_form import AlphaForm

# golden-section weight W = 0.5*(3 - sqrt(5))
# C++ parity: Real W = 0.5*(3.0-std::sqrt(5.0)) (alphafinder.cpp Minimize).
_GOLDEN_W = 0.5 * (3.0 - math.sqrt(5.0))


def _bisection(
    target: float,
    low: float,
    high: float,
    tolerance: float,
    value: Callable[[float], float],
) -> float:
    # C++ parity: template Bisection<T> in alphafinder.cpp (anonymous ns).
    x = 0.5 * (low + high)
    y = value(x)
    while True:
        if y < target:
            low = x
        elif y > target:
            high = x
        x = 0.5 * (low + high)
        y = value(x)
        if math.fabs(high - low) <= tolerance:
            break
    return x


def _find_highest_ok(
    low: float,
    high: float,
    tolerance: float,
    value: Callable[[float], bool],
) -> float:
    # C++ parity: template FindHighestOK<T> in alphafinder.cpp.
    x = 0.5 * (low + high)
    ok = value(x)
    while True:
        if ok:
            low = x
        else:
            high = x
        x = 0.5 * (low + high)
        ok = value(x)
        if math.fabs(high - low) <= tolerance:
            break
    return x


def _find_lowest_ok(
    low: float,
    high: float,
    tolerance: float,
    value: Callable[[float], bool],
) -> float:
    # C++ parity: template FindLowestOK<T> in alphafinder.cpp.
    x = 0.5 * (low + high)
    ok = value(x)
    while True:
        if ok:
            high = x
        else:
            low = x
        x = 0.5 * (low + high)
        ok = value(x)
        if math.fabs(high - low) <= tolerance:
            break
    return x


def _minimize(
    low: float,
    high: float,
    tolerance: float,
    value: Callable[[float], float],
    condition: Callable[[float], bool],
) -> tuple[float, bool]:
    """Golden-section minimise ``value`` subject to ``condition`` staying true.

    Returns ``(x, failed)``.

    # C++ parity: template Minimize<T> in alphafinder.cpp (returns x, writes
    ``failed`` through reference).
    """
    w = _GOLDEN_W
    left_value = value(low)
    right_value = value(high)
    x = w * low + (1 - w) * high
    mid_value = value(x)
    failed = True

    while high - low > tolerance:
        if x - low > high - x:  # left interval is bigger
            tentative_new_mid = w * low + (1 - w) * x
            tentative_new_mid_value = value(tentative_new_mid)
            conditioner = condition(tentative_new_mid_value)
            if not conditioner:
                if condition(x):
                    return x, failed
                return (low, failed) if left_value < right_value else (high, failed)
            if tentative_new_mid_value < mid_value:  # go left
                high = x
                right_value = mid_value
                x = tentative_new_mid
                mid_value = tentative_new_mid_value
            else:  # go right
                low = tentative_new_mid
                left_value = tentative_new_mid_value
        else:
            tentative_new_mid = w * x + (1 - w) * high
            tentative_new_mid_value = value(tentative_new_mid)
            conditioner = condition(tentative_new_mid_value)
            if not conditioner:
                if condition(x):
                    return x, failed
                return (low, failed) if left_value < right_value else (high, failed)
            if tentative_new_mid_value < mid_value:  # go right
                low = x
                left_value = mid_value
                x = tentative_new_mid
                mid_value = tentative_new_mid_value
            else:  # go left
                high = tentative_new_mid
                right_value = tentative_new_mid_value

    failed = False
    return x, failed


class AlphaFinder:
    """Solve for the alpha-form parameter matching a target swap variance.

    # C++ parity: AlphaFinder.
    """

    def __init__(self, parametric_form: AlphaForm) -> None:
        # C++ parity: AlphaFinder(ext::shared_ptr<AlphaForm> parametricform).
        self._parametric_form = parametric_form
        # working state set by solve()/solveWithMaxHomogeneity()
        self._stepindex = 0
        self._rateonevols: list[float] = []
        self._ratetwohomogeneousvols: list[float] = []
        self._putativevols: list[float] = []
        self._correlations: list[float] = []
        self._w0 = 0.0
        self._w1 = 0.0
        self._constant_part = 0.0
        self._linear_part = 0.0
        self._quadratic_part = 0.0
        self._total_var = 0.0
        self._target_variance = 0.0

    # --- private value helpers (member-fn pointers in C++) -----------------

    def _compute_linear_part(self, alpha: float) -> float:
        # C++ parity: AlphaFinder::computeLinearPart.
        cov = 0.0
        self._parametric_form.set_alpha(alpha)
        for i in range(self._stepindex + 1):
            vol1 = self._ratetwohomogeneousvols[i] * self._parametric_form(i)
            cov += vol1 * self._rateonevols[i] * self._correlations[i]
        cov *= 2 * self._w0 * self._w1
        return cov

    def _compute_quadratic_part(self, alpha: float) -> float:
        # C++ parity: AlphaFinder::computeQuadraticPart.
        var = 0.0
        self._parametric_form.set_alpha(alpha)
        for i in range(self._stepindex + 1):
            vol = self._ratetwohomogeneousvols[i] * self._parametric_form(i)
            var += vol * vol
        var *= self._w1 * self._w1
        return var

    def _homogeneity_failure(self, alpha: float) -> float:
        # C++ parity: AlphaFinder::homogeneityfailure.
        self._final_part(
            alpha,
            self._stepindex,
            self._ratetwohomogeneousvols,
            self._compute_quadratic_part(alpha),
            self._compute_linear_part(alpha),
            self._constant_part,
            self._putativevols,
        )
        result = 0.0
        for i in range(self._stepindex + 2):
            val = self._putativevols[i] - self._ratetwohomogeneousvols[i]
            result += val * val
        return result

    def _final_part(
        self,
        alpha_found: float,
        stepindex: int,
        ratetwohomogeneousvols: list[float],
        quadratic_part: float,
        linear_part: float,
        constant_part: float,
        ratetwovols: list[float],
    ) -> tuple[float, float, float, bool]:
        """Resolve ``a`` from the quadratic, distribute the remaining variance
        into ``b`` and fill ``ratetwovols`` in place.

        Returns ``(alpha, a, b, success)`` (C++ writes alpha/a/b through refs).

        # C++ parity: AlphaFinder::finalPart.
        """
        alpha = alpha_found
        q2 = Quadratic(quadratic_part, linear_part, constant_part - self._target_variance)
        self._parametric_form.set_alpha(alpha)
        a, _y, _real = q2.roots()  # C++ parity: q2.roots(a, y)

        var_so_far = 0.0
        for i in range(stepindex + 1):
            ratetwovols[i] = ratetwohomogeneousvols[i] * self._parametric_form(i) * a
            var_so_far += ratetwovols[i] * ratetwovols[i]

        var_to_find = self._total_var - var_so_far
        if var_to_find < 0:
            return alpha, a, 0.0, False
        required_sd = math.sqrt(var_to_find)
        b = required_sd / (  # C++ parity: `b`
            ratetwohomogeneousvols[stepindex + 1] * self._parametric_form(stepindex)
        )
        ratetwovols[stepindex + 1] = required_sd
        return alpha, a, b, True

    def _value_at_turning_point(self, alpha: float) -> float:
        # C++ parity: AlphaFinder::valueAtTurningPoint (caches linear/quadratic).
        self._linear_part = self._compute_linear_part(alpha)
        self._quadratic_part = self._compute_quadratic_part(alpha)
        q = Quadratic(self._quadratic_part, self._linear_part, self._constant_part)
        return q.value_at_turning_point()

    def _minus_value_at_turning_point(self, alpha: float) -> float:
        # C++ parity: AlphaFinder::minusValueAtTurningPoint.
        return -self._value_at_turning_point(alpha)

    def _test_if_solution_exists(self, alpha: float) -> bool:
        # C++ parity: AlphaFinder::testIfSolutionExists.
        a_exists = self._value_at_turning_point(alpha) < self._target_variance
        if not a_exists:
            return False
        _alpha, _a, _b, success = self._final_part(
            alpha,
            self._stepindex,
            self._ratetwohomogeneousvols,
            self._compute_quadratic_part(alpha),
            self._compute_linear_part(alpha),
            self._constant_part,
            self._putativevols,
        )
        return success

    # --- public solvers ----------------------------------------------------

    def solve(
        self,
        alpha0: float,
        stepindex: int,
        rateonevols: list[float],
        ratetwohomogeneousvols: list[float],
        correlations: list[float],
        w0: float,
        w1: float,
        target_variance: float,
        tolerance: float,
        alpha_max: float,
        alpha_min: float,
        steps: int,
        ratetwovols: list[float],
    ) -> tuple[bool, float, float, float]:
        """Solve for alpha; returns ``(success, alpha, a, b)`` and fills
        ``ratetwovols`` in place.

        # C++ parity: AlphaFinder::solve (alpha/a/b returned by ref in C++).
        """
        self._stepindex = stepindex
        self._rateonevols = rateonevols
        self._ratetwohomogeneousvols = ratetwohomogeneousvols
        self._correlations = correlations
        self._w0 = w0
        self._w1 = w1
        self._total_var = 0.0
        for i in range(stepindex + 2):
            self._total_var += ratetwohomogeneousvols[i] * ratetwohomogeneousvols[i]
        self._target_variance = target_variance

        # constant part does not depend on alpha
        self._constant_part = 0.0
        for i in range(stepindex + 1):
            self._constant_part += rateonevols[i] * rateonevols[i]
        self._constant_part *= w0 * w0

        value_at_tp = self._value_at_turning_point(alpha0)

        if value_at_tp <= target_variance:
            _alpha, a, b, _ok = self._final_part(
                alpha0,
                stepindex,
                ratetwohomogeneousvols,
                self._quadratic_part,
                self._linear_part,
                self._constant_part,
                ratetwovols,
            )
            return True, alpha0, a, b

        # we now have to solve
        bottom_value = self._value_at_turning_point(alpha_min)
        bottom_alpha = alpha_min
        top_value = self._value_at_turning_point(alpha_max)
        top_alpha = alpha_max
        bilimit = alpha0

        if bottom_value > target_variance and top_value > target_variance:
            i = 1
            while i < steps and top_value > target_variance:
                top_alpha = alpha0 + (alpha_max - alpha0) * (i + 0.0) / (steps + 0.0)
                top_value = self._value_at_turning_point(top_alpha)
                i += 1
            if top_value <= target_variance:
                bilimit = alpha0 + (top_alpha - alpha0) * (i - 2.0) / (steps + 0.0)

        if bottom_value > target_variance and top_value > target_variance:
            i = 1
            while i < steps and top_value > target_variance:
                bottom_alpha = alpha0 + (alpha_min - alpha0) * (i + 0.0) / (steps + 0.0)
                bottom_value = self._value_at_turning_point(bottom_alpha)
                i += 1
            if bottom_value <= target_variance:
                bilimit = alpha0 + (bottom_alpha - alpha0) * (i - 2.0) / (steps + 0.0)

        if bottom_value > target_variance and top_value > target_variance:
            return False, 0.0, 0.0, 0.0

        if bottom_value <= target_variance:
            # find root of (as if) increasing function
            alpha = _bisection(
                target_variance,
                bottom_alpha,
                bilimit,
                tolerance,
                self._value_at_turning_point,
            )
        else:
            # find root of (as if) decreasing function
            alpha = _bisection(
                -target_variance,
                bilimit,
                top_alpha,
                tolerance,
                self._minus_value_at_turning_point,
            )

        _alpha, a, b, _ok = self._final_part(
            alpha,
            stepindex,
            ratetwohomogeneousvols,
            self._quadratic_part,
            self._linear_part,
            self._constant_part,
            ratetwovols,
        )
        return True, alpha, a, b

    def solve_with_max_homogeneity(  # noqa: PLR0915
        self,
        alpha0: float,
        stepindex: int,
        rateonevols: list[float],
        ratetwohomogeneousvols: list[float],
        correlations: list[float],
        w0: float,
        w1: float,
        target_variance: float,
        tolerance: float,
        alpha_max: float,
        alpha_min: float,
        steps: int,
        ratetwovols: list[float],
    ) -> tuple[bool, float, float, float]:
        """Like :meth:`solve` but minimise the homogeneity-failure metric over
        the feasible alpha interval. Returns ``(success, alpha, a, b)``.

        # C++ parity: AlphaFinder::solveWithMaxHomogeneity.
        """
        self._stepindex = stepindex
        self._rateonevols = rateonevols
        self._ratetwohomogeneousvols = ratetwohomogeneousvols
        self._putativevols = [0.0] * len(ratetwohomogeneousvols)
        self._correlations = correlations
        self._w0 = w0
        self._w1 = w1
        self._total_var = 0.0
        for i in range(stepindex + 2):
            self._total_var += ratetwohomogeneousvols[i] * ratetwohomogeneousvols[i]
        self._target_variance = target_variance

        self._constant_part = 0.0
        for i in range(stepindex + 1):
            self._constant_part += rateonevols[i] * rateonevols[i]
        self._constant_part *= w0 * w0

        alpha1 = alpha_min
        alpha2 = alpha_max

        alpha0_ok = self._test_if_solution_exists(alpha0)
        alpha_max_ok = self._test_if_solution_exists(alpha_max)
        alpha_min_ok = self._test_if_solution_exists(alpha_min)

        found_ok_point = alpha0_ok or alpha_max_ok or alpha_min_ok

        if found_ok_point:
            if not alpha_min_ok:
                if alpha0_ok:
                    alpha1 = _find_lowest_ok(
                        alpha_min, alpha0, tolerance, self._test_if_solution_exists
                    )
                else:
                    # alpha_max_ok must be true to get here
                    alpha1 = _find_lowest_ok(
                        alpha0, alpha_max, tolerance, self._test_if_solution_exists
                    )
            if not alpha_max_ok:
                alpha2 = _find_highest_ok(
                    alpha1, alpha_max, tolerance, self._test_if_solution_exists
                )
            else:
                alpha2 = alpha_max
        else:
            # search a grid for a feasible alpha
            found_up_ok = False
            found_down_ok = False
            alpha_up = alpha0
            alpha_down = alpha0
            step_size = (alpha_max - alpha0) / steps
            j = 0
            while j < steps and not found_up_ok and not found_down_ok:
                alpha_up = alpha0 + j * step_size
                found_up_ok = self._test_if_solution_exists(alpha_up)
                alpha_down = alpha0 - j * step_size
                found_down_ok = self._test_if_solution_exists(alpha_down)
                j += 1
            found_ok_point = found_up_ok or found_down_ok
            if not found_ok_point:
                return False, 0.0, 0.0, 0.0
            if found_up_ok:
                alpha1 = alpha_up
                alpha2 = _find_highest_ok(
                    alpha1, alpha_max, tolerance, self._test_if_solution_exists
                )
            else:
                alpha2 = alpha_down
                alpha1 = _find_lowest_ok(
                    alpha_min, alpha2, tolerance, self._test_if_solution_exists
                )

        # minimise homogeneity failure within [alpha1, alpha2]
        alpha, _failed = _minimize(
            alpha1,
            alpha2,
            tolerance,
            self._homogeneity_failure,
            self._test_if_solution_exists,
        )

        _alpha, a, b, _ok = self._final_part(
            alpha,
            stepindex,
            ratetwohomogeneousvols,
            self._compute_quadratic_part(alpha),
            self._compute_linear_part(alpha),
            self._constant_part,
            ratetwovols,
        )
        return True, alpha, a, b
