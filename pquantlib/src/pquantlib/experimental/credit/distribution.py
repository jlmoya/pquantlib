"""Distribution — discretised probability density + cumulative.

# C++ parity: ql/experimental/credit/distribution.{hpp,cpp} (v1.43).

Bucket-based discretisation of a continuous distribution on a finite
interval [xmin, xmax]. Buckets have equal width (except the last one,
which is adjusted to land exactly on xmax). Three quantities are
maintained per bucket:

- ``density[i]`` — probability density (height of the histogram bar)
- ``cumulative[i]`` — integrated density from ``xmin`` up to bucket end
- ``excess[i]`` — probability of exceeding ``x[i]``
- ``cumulative_excess[i]`` — integrated excess from ``xmin`` to bucket
- ``average[i]`` — average value of samples landing in bucket i

Samples are added via either:

  - ``add(value)`` — increments the bucket's count + running average sum.
  - ``add_density(bucket, value)`` + ``add_average(bucket, value)`` —
    directly pushes into the density / average accumulator.

After all samples are added, ``normalize()`` converts counts to density
and the cumulative arrays. ``normalize()`` is idempotent.

# C++ parity divergence: the C++ class uses raw int sizes and friend-
# class access for the ``ManipulateDistribution::convolve`` helper. The
# Python port keeps :class:`ManipulateDistribution` with the same static
# ``convolve``, but the private-field access it needs goes through a
# single ``Distribution.bind_internals()`` accessor rather than a friend
# declaration. ``convolve_distributions`` is the module-level entry point
# both share.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pquantlib import qassert


@dataclass(slots=True)
class _DistributionInternals:
    """Friend-style accessor bundle for ``convolve_distributions``.

    The C++ implementation of ``ManipulateDistribution::convolve`` is a
    friend class with raw access to ``Distribution``'s private vectors.
    Python avoids the protected-attribute access warnings by exposing a
    single ``bind_internals()`` getter that returns this bundle (the
    lists are shared by reference, so mutations through it write back to
    the underlying Distribution).
    """

    density: list[float]
    cumulative_density: list[float]
    excess: list[float]
    dx: list[float]
    xmin: float
    xmax: float


class Distribution:
    """Discretised probability density + cumulative + excess + average."""

    __slots__ = (
        "_average",
        "_count",
        "_cumulative_density",
        "_cumulative_excess",
        "_density",
        "_dx",
        "_excess",
        "_is_normalized",
        "_overflow",
        "_size",
        "_underflow",
        "_x",
        "_xmax",
        "_xmin",
    )

    def __init__(self, n_buckets: int, xmin: float, xmax: float) -> None:
        self._size: int = n_buckets
        self._xmin: float = xmin
        self._xmax: float = xmax
        self._count: list[int] = [0] * n_buckets
        self._x: list[float] = [0.0] * n_buckets
        self._dx: list[float] = [0.0] * n_buckets
        self._density: list[float] = [0.0] * n_buckets
        self._cumulative_density: list[float] = [0.0] * n_buckets
        self._excess: list[float] = [0.0] * n_buckets
        self._cumulative_excess: list[float] = [0.0] * n_buckets
        self._average: list[float] = [0.0] * n_buckets
        self._overflow: int = 0
        self._underflow: int = 0
        self._is_normalized: bool = False

        if n_buckets > 0:
            base_dx = (xmax - xmin) / n_buckets
            for i in range(n_buckets):
                self._dx[i] = base_dx
                self._x[i] = xmin if i == 0 else self._x[i - 1] + self._dx[i - 1]
            # Final bucket is exact-match to xmax to avoid locate() round-off.
            self._dx[-1] = xmax - self._x[-1]

    # ----- accessors -----------------------------------------------------------

    def size(self) -> int:
        return self._size

    def x(self, k: int) -> float:
        return self._x[k]

    def dx(self, k: int) -> float:
        return self._dx[k]

    def dx_at(self, x: float) -> float:
        """Width of the bucket containing ``x``.

        # C++ parity: Distribution::dx(Real) at distribution.cpp:71-75.
        """
        return self._dx[self.locate(x)]

    def density(self, k: int) -> float:
        self.normalize()
        return self._density[k]

    def cumulative(self, k: int) -> float:
        self.normalize()
        return self._cumulative_density[k]

    def excess(self, k: int) -> float:
        self.normalize()
        return self._excess[k]

    def cumulative_excess(self, k: int) -> float:
        self.normalize()
        return self._cumulative_excess[k]

    def average(self, k: int) -> float:
        return self._average[k]

    def overflow(self) -> int:
        return self._overflow

    def underflow(self) -> int:
        return self._underflow

    def bind_internals(self) -> _DistributionInternals:
        """Return shared-reference bundle of internal arrays.

        Used by ``convolve_distributions``. Mutations through the
        returned bundle write back to this distribution.
        """
        return _DistributionInternals(
            density=self._density,
            cumulative_density=self._cumulative_density,
            excess=self._excess,
            dx=self._dx,
            xmin=self._xmin,
            xmax=self._xmax,
        )

    # ----- locate --------------------------------------------------------------

    def locate(self, x: float) -> int:
        """Return the index of the bucket whose left edge is at or below ``x``.

        # C++ parity: distribution.cpp:55-68.
        """
        x_first = self._x[0]
        x_last = self._x[-1]
        dx_last = self._dx[-1]
        # We can't import math.isclose semantics directly — C++ uses
        # ``close()`` from comparison.hpp which is a relative-tol check.
        # Mirror it with the same tolerance pquantlib already uses
        # (1e-15 relative is the C++ default).
        eps_low = 1e-15 * max(1.0, abs(x_first))
        eps_high = 1e-15 * max(1.0, abs(x_last + dx_last))
        qassert.require(
            (x >= x_first - eps_low) and (x <= x_last + dx_last + eps_high),
            f"coordinate {x} out of range [{x_first}; {x_last + dx_last}]",
        )
        for i in range(len(self._x)):
            if self._x[i] > x:
                return i - 1
        return len(self._x) - 1

    # ----- mutators ------------------------------------------------------------

    def add(self, value: float) -> None:
        """Add a sample at ``value``.

        Bumps the bucket count + the running average sum. ``normalize()``
        will divide later.

        # C++ parity: distribution.cpp:78-92.
        """
        self._is_normalized = False
        if value < self._x[0]:
            self._underflow += 1
            return
        for i in range(len(self._count)):
            if self._x[i] + self._dx[i] > value:
                self._count[i] += 1
                self._average[i] += value
                return
        self._overflow += 1

    def add_density(self, bucket: int, value: float) -> None:
        qassert.require(0 <= bucket < self._size, "bucket out of range")
        self._is_normalized = False
        self._density[bucket] += value

    def add_average(self, bucket: int, value: float) -> None:
        qassert.require(0 <= bucket < self._size, "bucket out of range")
        self._is_normalized = False
        self._average[bucket] += value

    def normalize(self) -> None:
        """Convert raw counts to densities + populate cumulative + excess arrays.

        Idempotent: subsequent calls are no-ops until a new ``add*`` call
        flips the ``_is_normalized`` flag.

        # C++ parity: distribution.cpp:111-148.
        """
        if self._is_normalized:
            return

        count = self._underflow + self._overflow + sum(self._count)
        self._excess[0] = 1.0
        self._cumulative_excess[0] = 0.0
        for i in range(self._size):
            if count > 0:
                self._density[i] = (1.0 / self._dx[i]) * self._count[i] / count
                if self._count[i] > 0:
                    self._average[i] /= self._count[i]
            if self._density[i] == 0.0:
                self._average[i] = self._x[i] + self._dx[i] / 2.0
            self._cumulative_density[i] = self._density[i] * self._dx[i]
            if i > 0:
                self._cumulative_density[i] += self._cumulative_density[i - 1]
                self._excess[i] = 1.0 - self._cumulative_density[i - 1]
                self._cumulative_excess[i] = (
                    self._excess[i - 1] * self._dx[i - 1]
                    + self._cumulative_excess[i - 1]
                )

        self._is_normalized = True

    # ----- queries -------------------------------------------------------------

    def confidence_level(self, quantil: float) -> float:
        """Smallest ``x`` such that ``cumulative(x) > quantil``.

        # C++ parity: distribution.cpp:151-159.
        """
        self.normalize()
        for i in range(self._size):
            if self._cumulative_density[i] > quantil:
                return self._x[i] + self._dx[i]
        return self._x[-1] + self._dx[-1]

    def expected_value(self) -> float:
        """Mean of the distribution.

        # C++ parity: distribution.cpp:162-171.
        """
        self.normalize()
        expected = 0.0
        for i in range(self._size):
            mid = self._x[i] + self._dx[i] / 2.0
            expected += mid * self._dx[i] * self._density[i]
        return expected

    def expected_value_of(self, f: Callable[[float], float]) -> float:
        """``E[f(X)]`` over the bucketed density, sampling ``f`` at bucket mids.

        # C++ parity: the member template ``Distribution::expectedValue(F& f)``
        # at distribution.hpp:80-89.
        """
        self.normalize()
        expected = 0.0
        for i in range(self._size):
            x = self._x[i] + self._dx[i] / 2.0
            expected += f(x) * self._dx[i] * self._density[i]
        return expected

    def tranche_expected_value(self, a: float, d: float) -> float:
        """Expected value of the tranche [a, d].

        # C++ parity: distribution.cpp:174-190.
        """
        self.normalize()
        expected = 0.0
        for i in range(self._size):
            mid = self._x[i] + self._dx[i] / 2.0
            if mid < a:
                continue
            if mid > d:
                break
            expected += (mid - a) * self._dx[i] * self._density[i]
        expected += (d - a) * (1.0 - self.cumulative_density(d))
        return expected

    def cumulative_excess_probability(self, a: float, b: float) -> float:
        """Integral of excess probability between ``a`` and ``b``.

        # C++ parity: distribution.cpp:204-217.
        """
        self.normalize()
        qassert.require(
            b <= self._xmax,
            f"end of interval {b} out of range [{self._xmin}, {self._xmax}]",
        )
        qassert.require(
            a >= self._xmin,
            f"start of interval {a} out of range [{self._xmin}, {self._xmax}]",
        )
        i = self.locate(a)
        j = self.locate(b)
        return self._cumulative_excess[j] - self._cumulative_excess[i]

    def cumulative_density(self, x: float) -> float:
        """Linearly-interpolated cumulative density at ``x``.

        # C++ parity: distribution.cpp:220-232.
        """
        tiny = self._dx[-1] * 1e-3
        qassert.require(x > 0, "x must be positive")
        self.normalize()
        for i in range(self._size):
            if self._x[i] + self._dx[i] + tiny >= x:
                # ``_cum_density_before`` carries the i == 0 out-of-bounds
                # read; see its docstring.
                return (
                    (x - self._x[i]) * self._cumulative_density[i]
                    + (self._x[i] + self._dx[i] - x) * self._cum_density_before(i)
                ) / self._dx[i]
        qassert.fail(
            f"x = {x} beyond distribution cutoff {self._x[-1] + self._dx[-1]}"
        )

    def tranche(self, attachment_point: float, detachment_point: float) -> None:
        """Transform this loss distribution into the tranche loss distribution.

        For losses ``L_T = min(L, D) - min(L, A)``, i.e. shift left by ``A``,
        cut off at ``D - A`` and pile the residual mass onto ``L_T = 0``.

        Destructive; the C++ comment says it plainly: "Dangerous to perform
        calls to members after this; transform and clone?".

        # C++ parity: distribution.cpp:234-285.
        """
        qassert.require(
            attachment_point < detachment_point,
            "attachment >= detachment point",
        )
        qassert.require(
            self._x[-1] > attachment_point
            and self._x[-1] + self._dx[-1] >= detachment_point,
            "attachment or detachment too large",
        )

        self.normalize()

        # shift — drop every bucket entirely below the attachment point.
        while self._x[0] < attachment_point:
            del self._x[0]
            del self._dx[0]
            del self._count[0]
            del self._density[0]
            del self._cumulative_density[0]
            del self._excess[0]

        # remove losses over the detachment point
        detach_posit = next(
            (i for i, xi in enumerate(self._x) if xi > detachment_point), None
        )
        if detach_posit is not None:
            del self._x[detach_posit + 1 :]

        self._size = len(self._x)
        del self._cumulative_density[self._size :]
        self._cumulative_density[-1] = 1.0
        del self._count[self._size :]
        del self._dx[self._size :]

        # truncate
        for i in range(len(self._x)):
            self._x[i] = min(
                max(self._x[i] - attachment_point, 0.0),
                detachment_point - attachment_point,
            )

        self._density.clear()
        self._excess.clear()
        self._cumulative_excess.clear()  # ? reuse?
        self._density.append((self._cumulative_density[0] - 0.0) / self._dx[0])
        self._excess.append(1.0)
        for i in range(1, self._size - 1):
            self._excess.append(1.0 - self._cumulative_density[i - 1])
            self._density.append(
                (self._cumulative_density[i] - self._cumulative_density[i - 1])
                / self._dx[i]
            )
        self._excess.append(1.0 - self._cumulative_density[-1])
        self._density.append((1.0 - self._cumulative_density[-1]) / self._dx[-1])

    def expected_shortfall(self, perc_value: float) -> float:
        """Expected value above the quantile at ``perc_value``.

        # C++ parity: distribution.cpp:327-342.
        """
        qassert.require(0.0 <= perc_value <= 1.0, "Incorrect percentile")
        self.normalize()
        i_val = self.locate(self.confidence_level(perc_value))

        if i_val == self._size - 1:
            return self._x[-1]

        expected = 0.0
        for i in range(i_val, self._size):
            expected += self._x[i] * (
                self._cumulative_density[i] - self._cum_density_before(i)
            )
        return expected / (1.0 - self._cumulative_density[i_val])

    def _cum_density_before(self, i: int) -> float:
        """``cumulativeDensity_[i-1]``, with C++'s out-of-bounds read at i == 0.

        # C++ parity note (DEFECT, reproduced verbatim): both
        # ``Distribution::expectedShortfall`` (distribution.cpp:338-340) and
        # ``Distribution::cumulativeDensity`` (distribution.cpp:225-229) index
        # ``cumulativeDensity_[i-1]`` inside a loop that starts at ``i == 0``::
        #
        #     for (int i = iVal; i < size_; i++)
        #         expected += x_[i] *
        #             (cumulativeDensity_[i] - cumulativeDensity_[i-1]);
        #
        # ``std::vector::operator[]`` does no bounds checking, so at ``i == 0``
        # this reads the double immediately before the buffer — undefined
        # behaviour. The value observed on arm64/libc++ is 0.0, solved for from
        # the probe's ``homog_expected_shortfall_5y`` (a tranched distribution
        # whose ``iVal`` is 0): only G == 0.0 reproduces 14.616955566384418;
        # G == 1.0, which is what Python's negative-index wraparound would
        # silently supply, gives 13.2. 0.0 is also the mathematically correct
        # convention — there is no probability mass below the first bucket —
        # so this port returns 0.0 rather than reading past the array.
        """
        return self._cumulative_density[i - 1] if i > 0 else 0.0


class ManipulateDistribution:
    """Namespace class holding the ``Distribution`` convolution.

    # C++ parity: ``class ManipulateDistribution`` at distribution.hpp:129-133
    # — a class with a single static member, declared a ``friend`` of
    # ``Distribution`` (distribution.hpp:36 + :39) so ``convolve`` can reach
    # the private vectors of both operands.
    """

    __slots__ = ()

    @staticmethod
    def convolve(d1: Distribution, d2: Distribution) -> Distribution:
        """Convolve two distributions.

        # C++ parity: ``ManipulateDistribution::convolve`` at
        # distribution.cpp:287-324.
        """
        return convolve_distributions(d1, d2)


def convolve_distributions(d1: Distribution, d2: Distribution) -> Distribution:
    """Convolve two distributions (assumes equal-width buckets + xmin = 0).

    # C++ parity: ManipulateDistribution::convolve at distribution.cpp:288-324.
    The C++ version is a friend class with raw access to ``Distribution``'s
    private vectors. Python achieves equivalent low-level access via a
    single ``Distribution.bind_internals`` accessor (used here only).
    """
    qassert.require(
        d1.dx(0) == d2.dx(0),
        "bucket sizes differ in d1 and d2",
    )
    for i in range(1, d1.size()):
        qassert.require(d1.dx(i) == d1.dx(i - 1), "bucket size varies in d1")
    for i in range(1, d2.size()):
        qassert.require(d2.dx(i) == d2.dx(i - 1), "bucket size varies in d2")
    d1_state = d1.bind_internals()
    d2_state = d2.bind_internals()
    qassert.require(
        d1_state.xmin == 0.0 and d2_state.xmin == 0.0,
        "distributions offset larger than 0",
    )

    out = Distribution(
        d1.size() + d2.size() - 1, 0.0, d1_state.xmax + d2_state.xmax
    )
    out_state = out.bind_internals()
    for i1 in range(d1.size()):
        dx = d1_state.dx[i1]
        for i2 in range(d2.size()):
            out_state.density[i1 + i2] = (
                d1_state.density[i1] * d2_state.density[i2] * dx
            )

    out_state.excess[0] = 1.0
    for i in range(out.size()):
        out_state.cumulative_density[i] = out_state.density[i] * out_state.dx[i]
        if i > 0:
            out_state.cumulative_density[i] += out_state.cumulative_density[i - 1]
            out_state.excess[i] = (
                out_state.excess[i - 1]
                - out_state.density[i - 1] * out_state.dx[i - 1]
            )

    return out
