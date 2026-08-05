"""Linear regression built on :mod:`pquantlib.math.general_linear_least_squares`.

# C++ parity: ql/math/linearleastsquaresregression.hpp (v1.43) — header-only
#             template.

``LinearRegression`` is nothing but a basis-function factory in front of
``GeneralLinearLeastSquares``: it assembles ``[intercept, x]`` for a scalar
regressor or ``[intercept, x_0, ..., x_{m-1}]`` for a vector one, and hands
that to the SVD fit.

The one behaviour worth spelling out: ``intercept`` is a *value*, not a flag.
``intercept=1.0`` gives the usual constant basis function; ``intercept=2.5``
gives a constant-2.5 basis function, so the fitted coefficient is the usual one
divided by 2.5; and ``intercept=0.0`` drops the constant term entirely rather
than fitting it to zero, which changes ``dim()``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pquantlib.math.general_linear_least_squares import (
    GeneralLinearLeastSquares,
    RegressionArgument,
)

# One basis function. The argument is a Real for a scalar regressor and a
# sequence of Reals for a vector one, matching the C++ ``ArgumentType``.
type BasisFunction = Callable[..., float]


class LinearFct:
    """Project a vector-valued regressor onto its ``i``-th component.

    # C++ parity: ``details::LinearFct`` —
    # linearleastsquaresregression.hpp:35-47.
    """

    __slots__ = ("_i",)

    def __init__(self, i: int) -> None:
        self._i: int = i

    def __call__(self, x: Sequence[float]) -> float:
        return x[self._i]


class _Constant:
    """The intercept basis function. C++ spells it as a capturing lambda."""

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        self._value: float = value

    def __call__(self, _x: RegressionArgument) -> float:
        # The C++ lambda captures the intercept and ignores its argument.
        return self._value


class _Identity:
    """The scalar regressor itself. C++ spells it as a lambda."""

    __slots__ = ()

    def __call__(self, x: float) -> float:
        return x


class LinearFcts:
    """Assemble the basis for :class:`LinearRegression`.

    # C++ parity: ``details::LinearFcts`` —
    # linearleastsquaresregression.hpp:49-71. The C++ dispatch is
    # ``if constexpr (std::is_arithmetic_v<ArgumentType>)``; Python decides at
    # runtime on the first element instead, which is the same decision.
    """

    __slots__ = ("_v",)

    def __init__(self, x: Sequence[RegressionArgument], intercept: float) -> None:
        v: list[BasisFunction] = []
        if intercept != 0.0:
            v.append(_Constant(intercept))
        first = x[0]
        if isinstance(first, (int, float)):
            v.append(_Identity())
        else:
            for i in range(len(first)):
                v.append(LinearFct(i))
        self._v: list[BasisFunction] = v

    def fcts(self) -> list[BasisFunction]:
        # C++ parity: linearleastsquaresregression.hpp:66-68.
        return self._v


class LinearRegression(GeneralLinearLeastSquares):
    """``y_i = a_0 + a_1 x_0 + ... + a_n x_{n-1} + eps``.

    # C++ parity: ``class LinearRegression`` —
    # linearleastsquaresregression.hpp:75-88, ctors at :91-105.
    """

    __slots__ = ()

    def __init__(
        self,
        x: Sequence[RegressionArgument],
        y: Sequence[float],
        intercept: float = 1.0,
    ) -> None:
        super().__init__(x, y, LinearFcts(x, intercept).fcts())


class LinearLeastSquaresRegression(GeneralLinearLeastSquares):
    """Explicit-basis regression; kept for backward compatibility.

    # C++ parity: ``class LinearLeastSquaresRegression`` —
    # linearleastsquaresregression.hpp:107-118. The C++ comment says to use
    # ``GeneralLinearLeastSquares`` directly; the name is kept because the
    # published API has it.
    """

    __slots__ = ()


__all__ = [
    "BasisFunction",
    "LinearFct",
    "LinearFcts",
    "LinearLeastSquaresRegression",
    "LinearRegression",
]
