"""Kernel functions in the statistical sense.

# C++ parity: ql/math/kernelfunctions.hpp (v1.43).

A kernel function is a non-negative, symmetric, real-valued function that
integrates to one. C++ declares the concept as an abstract functor base
(``KernelFunction`` with a pure-virtual ``operator()``) and ships one
concrete implementation, ``GaussianKernel``.

The abstract base is *not* what the interpolations bind against: both
``KernelInterpolation`` and ``KernelInterpolation2D`` take the kernel as a
template parameter and only require ``Real operator()(Real)``, so a plain
function pointer works too (the C++ test-suite passes
``&epanechnikovKernel``). The Python port keeps both doors open:
:class:`KernelFunction` is an ABC for classes that want the contract
spelled out, and the interpolations accept any ``Callable[[float], float]``.

``GaussianKernel`` is a :class:`~pquantlib.math.distributions.normal_distribution.NormalDistribution`
scaled by ``sqrt(2*pi)``, which makes ``GaussianKernel(0, sigma)(x)``
equal to ``exp(-x^2 / (2 sigma^2)) / sigma`` — i.e. the *un-normalised*
Gaussian bump divided by sigma, not a density. Getting that factor wrong
is invisible in ``KernelInterpolation`` (the gamma normalisation divides
it straight back out) but not in ``primitive``/``derivative``, so it is
pinned by the probe on both tails.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution

# C++ ``normFact_ = M_SQRT2 * M_SQRTPI`` == sqrt(2*pi).  Both factors are
# spelled here as the DECIMAL LITERALS from ql/mathconstants.hpp:73-94 rather
# than as ``math.sqrt(...)``: the literal for M_SQRTPI rounds to a double one
# ULP away from ``math.sqrt(math.pi)``.  (Their product happens to agree in
# this case, but the constants are what C++ multiplies, so they are what we
# multiply.)
_M_SQRT2: Final[float] = 1.41421356237309504880
_M_SQRTPI: Final[float] = 1.77245385090551602792981
_NORM_FACT: Final[float] = _M_SQRT2 * _M_SQRTPI


class KernelFunction(ABC):
    """Abstract kernel functor.

    # C++ parity: ``class KernelFunction`` (kernelfunctions.hpp:36-40).

    Subclasses implement ``__call__(x) -> float``. C++ derived classes
    "serve as functors" via ``operator()``; the Python equivalent is
    ``__call__``, so any :class:`KernelFunction` is already a
    ``Callable[[float], float]``.
    """

    __slots__ = ()

    @abstractmethod
    def __call__(self, x: float) -> float:
        """Evaluate the kernel at ``x``."""


class GaussianKernel(KernelFunction):
    """Gaussian kernel — a normal pdf scaled back up by ``sqrt(2*pi)``.

    # C++ parity: ``class GaussianKernel`` (kernelfunctions.hpp:44-64).

    Args:
        average: the mean of the underlying normal distribution.
        sigma: the standard deviation of the underlying normal
            distribution; must be strictly positive.
    """

    __slots__ = ("_cnd", "_nd", "_norm_fact")

    def __init__(self, average: float, sigma: float) -> None:
        self._nd: NormalDistribution = NormalDistribution(average, sigma)
        self._cnd: CumulativeNormalDistribution = CumulativeNormalDistribution(average, sigma)
        self._norm_fact: float = _NORM_FACT

    def __call__(self, x: float) -> float:
        # C++ parity: kernelfunctions.hpp:50.
        return self._nd(x) * self._norm_fact

    def derivative(self, x: float) -> float:
        """First derivative of the kernel at ``x``.

        # C++ parity: kernelfunctions.hpp:52-54.
        """
        return self._nd.derivative(x) * self._norm_fact

    def primitive(self, x: float) -> float:
        """Antiderivative of the kernel at ``x``.

        # C++ parity: kernelfunctions.hpp:56-58.
        """
        return self._cnd(x) * self._norm_fact


__all__ = ["GaussianKernel", "KernelFunction"]
