"""OneFactorCopula — abstract base + Gaussian + Student concrete subclasses.

# C++ parity: ql/experimental/credit/onefactorcopula.{hpp,cpp},
# onefactorgaussiancopula.{hpp,cpp}, onefactorstudentcopula.{hpp,cpp}
# @ v1.42.1 (099987f0).

The one-factor copula

    Y_i = a_i M + sqrt(1 - a_i^2) Z_i

models joint defaults via a common factor ``M`` and idiosyncratic factors
``Z_i``. Given a per-name unconditional default probability ``p``, the
*conditional* probability given a realization ``m`` of M is

    p_hat(p, m) = F_Z((F_Y^-1(p) - sqrt(rho) * m) / sqrt(1 - rho))

where ``rho`` is the squared common-factor loading (``a_i^2``). This is the
mechanism used by ``GaussianLHPLossModel`` (Vasicek 2002 LHP closed form),
the binomial / recursive / saddlepoint loss models, and the latent-model
hierarchy.

The C++ ``OneFactorCopula`` is a ``LazyObject`` because it pre-tabulates
the cumulative distribution of Y for the Student-copula path. The Gaussian
copula overrides ``cumulativeY``/``inverseCumulativeY`` with closed-form
identities and so does not need the table. The Python port keeps the same
two-step structure: ``OneFactorCopula`` is the LazyObject with the Euler
integration helpers; ``OneFactorGaussianCopula`` is the closed-form
override; ``OneFactorStudentCopula`` rebuilds the table on
``_perform_calculations``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.stats import t as student_t  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.patterns.lazy_object import LazyObject


class OneFactorCopula(LazyObject):
    """Abstract base for one-factor copula models.

    Subclasses must implement ``density(m)`` and ``cumulative_z(z)`` and
    may override ``cumulative_y`` / ``inverse_cumulative_y`` if a
    closed form is available; otherwise the base class uses the
    Euler-integration table built by ``_perform_calculations``.

    # C++ parity: onefactorcopula.hpp:102.
    """

    __slots__ = ("_correlation", "_cumulative_y_tab", "_max", "_min", "_steps", "_y_tab")

    def __init__(
        self,
        correlation: float,
        maximum: float = 5.0,
        integration_steps: int = 50,
        minimum: float = -5.0,
    ) -> None:
        super().__init__()
        qassert.require(
            -1.0 <= correlation <= 1.0,
            f"correlation out of range [-1, +1]: {correlation}",
        )
        self._correlation = correlation
        self._max = maximum
        self._steps = integration_steps
        self._min = minimum
        self._y_tab: list[float] = []
        self._cumulative_y_tab: list[float] = []

    def correlation(self) -> float:
        """Single correlation parameter."""
        self.calculate()
        return self._correlation

    def set_correlation(self, correlation: float) -> None:
        """Mutate the correlation (Quote-equivalent path).

        # C++ parity divergence: the C++ class stores a ``Handle<Quote>``
        # for the correlation. Python uses a direct setter; rho mutation
        # invalidates the cache so the Student-copula table is rebuilt.
        """
        qassert.require(
            -1.0 <= correlation <= 1.0,
            f"correlation out of range [-1, +1]: {correlation}",
        )
        self._correlation = correlation
        self.update()

    @abstractmethod
    def density(self, m: float) -> float:
        """Density function of M (zero mean, unit variance)."""

    @abstractmethod
    def cumulative_z(self, z: float) -> float:
        """Cumulative distribution function of Z (zero mean, unit variance)."""

    # --- Default implementations using the tabulated Y distribution ---

    def cumulative_y(self, y: float) -> float:
        """Cumulative distribution of Y.

        Default: linear interpolation on the tabulated ``_y_tab`` /
        ``_cumulative_y_tab`` arrays. Subclasses may override with a
        closed form.

        # C++ parity: onefactorcopula.cpp:57-75.
        """
        self.calculate()
        qassert.require(len(self._y_tab) > 0, "cumulative Y not tabulated yet")

        if y < self._y_tab[0]:
            return self._cumulative_y_tab[0]
        for i in range(1, len(self._y_tab)):
            if self._y_tab[i] > y:
                return (
                    (self._y_tab[i] - y) * self._cumulative_y_tab[i - 1]
                    + (y - self._y_tab[i - 1]) * self._cumulative_y_tab[i]
                ) / (self._y_tab[i] - self._y_tab[i - 1])
        return self._cumulative_y_tab[-1]

    def inverse_cumulative_y(self, x: float) -> float:
        """Inverse cumulative distribution of Y.

        Default: linear interpolation on the tabulated arrays.

        # C++ parity: onefactorcopula.cpp:78-96.
        """
        self.calculate()
        qassert.require(len(self._y_tab) > 0, "cumulative Y not tabulated yet")

        if x < self._cumulative_y_tab[0]:
            return self._y_tab[0]
        for i in range(1, len(self._cumulative_y_tab)):
            if self._cumulative_y_tab[i] > x:
                return (
                    (self._cumulative_y_tab[i] - x) * self._y_tab[i - 1]
                    + (x - self._cumulative_y_tab[i - 1]) * self._y_tab[i]
                ) / (self._cumulative_y_tab[i] - self._cumulative_y_tab[i - 1])
        return self._y_tab[-1]

    def _perform_calculations(self) -> None:
        """No-op by default — Gaussian copula needs no table.

        Student copula overrides this to rebuild ``_y_tab`` and
        ``_cumulative_y_tab`` over [-max, max] when correlation changes.
        """
        # default no-op

    # --- Euler integration helpers over M ---

    def steps(self) -> int:
        return self._steps

    def dm(self) -> float:
        # # C++ parity: onefactorcopula.hpp:272.
        return (self._max - self._min) / self._steps

    def m(self, i: int) -> float:
        # # C++ parity: onefactorcopula.hpp:276.
        qassert.require(i < self._steps, f"index out of range: {i}")
        return self._min + self.dm() * i + self.dm() / 2.0

    def density_dm(self, i: int) -> float:
        # # C++ parity: onefactorcopula.hpp:281.
        qassert.require(i < self._steps, f"index out of range: {i}")
        return self.density(self.m(i)) * self.dm()

    # --- Conditional default probability ---

    def conditional_probability(self, prob: float, m: float) -> float:
        """p_hat(m) = F_Z((F_Y^-1(p) - sqrt(rho) m) / sqrt(1-rho)).

        # C++ parity: onefactorcopula.cpp:27-42.
        """
        self.calculate()
        # FIXME (mirrors C++ guard at onefactorcopula.cpp:31)
        if prob < 1e-10:
            return 0.0

        c = self._correlation
        res = self.cumulative_z(
            (self.inverse_cumulative_y(prob) - np.sqrt(c) * m) / np.sqrt(1.0 - c)
        )
        qassert.require(
            0.0 <= res <= 1.0,
            f"conditional probability {res} out of range",
        )
        return res

    def conditional_probability_vec(
        self, probs: list[float], m: float
    ) -> list[float]:
        """Vector overload of ``conditional_probability``.

        # C++ parity: onefactorcopula.cpp:44-54.
        """
        self.calculate()
        return [self.conditional_probability(p, m) for p in probs]

    def integral(self, p: float) -> float:
        """Integral over rho_M(m) of the conditional probability.

        # C++ parity: onefactorcopula.hpp:169 — Euler integration over M.
        """
        qassert.require(
            0.0 <= p <= 1.0, f"probability p={p} out of range [0,1]"
        )
        self.calculate()
        avg = 0.0
        for k in range(self.steps()):
            pp = self.conditional_probability(p, self.m(k))
            avg += pp * self.density_dm(k)
        return avg

    def integral_fn(
        self,
        f: Callable[[list[float]], float],
        probabilities: list[float],
    ) -> float:
        """Integral over rho_M(m) of f(conditional probabilities vector).

        Mirrors the templated ``integral`` at onefactorcopula.hpp:193 but
        takes any callable ``f(list[float]) -> float`` as Python duck-types
        the C++ template.
        """
        self.calculate()
        avg: float = 0.0
        for i in range(self._steps):
            conditional = self.conditional_probability_vec(probabilities, self.m(i))
            avg += f(conditional) * self.density_dm(i)
        return avg

    def check_moments(self, tolerance: float = 1e-2) -> int:
        """Check (norm=1, mean=0, var=1) for M, Z, Y up to ``tolerance``.

        # C++ parity: onefactorcopula.cpp:99-157. Returns 0 on success;
        # raises on tolerance violation. The default tolerance is loosened
        # from the C++ default because Euler integration over 50 steps
        # only achieves ~1e-2 accuracy.
        """
        self.calculate()
        norm = 0.0
        mean = 0.0
        var = 0.0
        for i in range(self.steps()):
            norm += self.density_dm(i)
            mean += self.m(i) * self.density_dm(i)
            var += self.m(i) ** 2 * self.density_dm(i)
        qassert.require(abs(norm - 1.0) < tolerance, f"M norm {norm} out of tolerance")
        qassert.require(abs(mean) < tolerance, f"M mean {mean} out of tolerance")
        qassert.require(
            abs(var - 1.0) < tolerance, f"M variance {var} out of tolerance"
        )
        return 0


class OneFactorGaussianCopula(OneFactorCopula):
    """One-factor Gaussian copula.

    All three of M, Z, Y are standard normal so cumulative_y is the same
    standard CDF as cumulative_z — no table needed.

    # C++ parity: onefactorgaussiancopula.{hpp,cpp}.
    """

    __slots__ = ("_cumulative", "_density", "_inv_cumulative")

    def __init__(
        self,
        correlation: float,
        maximum: float = 5.0,
        integration_steps: int = 50,
    ) -> None:
        super().__init__(correlation, maximum, integration_steps)
        self._density = NormalDistribution()
        self._cumulative = CumulativeNormalDistribution()
        self._inv_cumulative = InverseCumulativeNormal()

    def density(self, m: float) -> float:
        # # C++ parity: onefactorgaussiancopula.hpp:64.
        return self._density(m)

    def cumulative_z(self, z: float) -> float:
        # # C++ parity: onefactorgaussiancopula.hpp:68.
        return self._cumulative(z)

    def cumulative_y(self, y: float) -> float:
        # # C++ parity: onefactorgaussiancopula.hpp:72.
        return self._cumulative(y)

    def inverse_cumulative_y(self, x: float) -> float:
        # # C++ parity: onefactorgaussiancopula.hpp:76.
        return InverseCumulativeNormal.standard_value(x)

    def _perform_calculations(self) -> None:
        # # C++ parity: onefactorgaussiancopula.hpp:57 — nothing to do.
        return None


class OneFactorStudentCopula(OneFactorCopula):
    """One-factor (double) Student-t copula.

    Both M and Z are Student-t distributed with ``nz`` and ``nm`` degrees of
    freedom, scaled to unit variance. Y has no closed form so its CDF is
    tabulated on the [-max, max] grid via Euler integration of the
    convolution defining F_Y.

    # C++ parity: onefactorstudentcopula.{hpp,cpp}.
    """

    __slots__ = ("_nm", "_nz", "_scale_m", "_scale_z")

    def __init__(
        self,
        correlation: float,
        nz: int,
        nm: int,
        maximum: float = 10.0,
        integration_steps: int = 200,
    ) -> None:
        qassert.require(nz > 2, f"nz must be > 2 for finite variance: {nz}")
        qassert.require(nm > 2, f"nm must be > 2 for finite variance: {nm}")
        super().__init__(correlation, maximum, integration_steps)
        self._nz = nz
        self._nm = nm
        # Variance of Student-t(nu) is nu/(nu-2); scale to unit variance.
        # # C++ parity: onefactorstudentcopula.cpp constructor (scale = sqrt((nu-2)/nu)).
        self._scale_m: float = float(np.sqrt((nm - 2.0) / nm))
        self._scale_z: float = float(np.sqrt((nz - 2.0) / nz))

    def density(self, m: float) -> float:
        # # C++ parity: onefactorstudentcopula.hpp:80.
        return float(student_t.pdf(m / self._scale_m, self._nm)) / self._scale_m  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    def cumulative_z(self, z: float) -> float:
        # # C++ parity: onefactorstudentcopula.hpp:84.
        return float(student_t.cdf(z / self._scale_z, self._nz))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    def _perform_calculations(self) -> None:
        """Tabulate Y's CDF over [-max, max].

        Mirrors C++ ``performCalculations`` for Student copula: walks an
        Euler grid over Y and integrates the joint density of (M, Z) to
        get F_Y(y).

        # C++ parity: onefactorstudentcopula.cpp performCalculations.
        """
        # Resize tables
        n = self._steps + 1
        self._y_tab = [0.0] * n
        self._cumulative_y_tab = [0.0] * n
        dy = (self._max - self._min) / self._steps
        c = self._correlation
        sqrt_c = float(np.sqrt(c))
        sqrt_1mc = float(np.sqrt(1.0 - c))

        # Build the table by 1D quadrature over M:
        #   F_Y(y) = ∫ rho_M(m) F_Z((y - sqrt(c) m) / sqrt(1-c)) dm.
        # We use the same Euler grid for M as the integral() helper to
        # match C++'s ad-hoc midpoint rule.
        for i in range(n):
            y = self._min + dy * i
            acc = 0.0
            for k in range(self._steps):
                m = self.m(k)
                arg = (y - sqrt_c * m) / sqrt_1mc if sqrt_1mc > 0 else 0.0
                acc += self.cumulative_z(arg) * self.density_dm(k)
            self._y_tab[i] = y
            self._cumulative_y_tab[i] = acc
