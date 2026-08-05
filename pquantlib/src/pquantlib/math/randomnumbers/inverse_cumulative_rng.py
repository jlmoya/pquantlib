"""Inverse-cumulative (scalar) random number generator.

# C++ parity: ql/math/randomnumbers/inversecumulativerng.hpp (v1.43) —
# ``template <class RNG, class IC> class InverseCumulativeRng``.

Takes a uniform deviate in (0, 1) as a cumulative-probability value and maps
it through an inverse cumulative distribution. The weight passes through
untouched.
"""

from __future__ import annotations

from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.randomnumbers.random_number_generator import (
    InverseCumulative,
    RandomNumberGenerator,
    Sample,
)


class InverseCumulativeRng[
    Rng: RandomNumberGenerator, Ic: InverseCumulative = InverseCumulativeNormal
]:
    """Uniform deviates mapped through an inverse cumulative distribution.

    # C++ parity: ``InverseCumulativeRng<RNG, IC>``
    # (inversecumulativerng.hpp:53-64).

    C++ default-constructs its ``IC`` member (``IC ICND_;``) and gives no way
    to configure it, so the only reachable configuration of e.g.
    ``InverseCumulativeRng<..., InverseCumulativePoisson>`` is lambda = 1.
    Python cannot default-construct an arbitrary type parameter, so the
    inverse cumulative is passed in; omit it and you get
    ``InverseCumulativeNormal()``, which is what every QuantLib trait
    instantiates.
    """

    __slots__ = ("_icnd", "_uniform_generator")

    def __init__(self, uniform_generator: Rng, inverse_cumulative: Ic | None = None) -> None:
        # C++ parity: inversecumulativerng.hpp:67-69.
        self._uniform_generator: Rng = uniform_generator
        self._icnd: InverseCumulative = (
            InverseCumulativeNormal() if inverse_cumulative is None else inverse_cumulative
        )

    def next(self) -> Sample:
        """One sample from the target distribution.

        # C++ parity: ``next`` (inversecumulativerng.hpp:71-77).
        """
        sample = self._uniform_generator.next()
        return Sample(value=self._icnd(sample.value), weight=sample.weight)

    def dimension(self) -> int:
        """Scalar generator — dimension is always 1."""
        return 1


__all__ = ["InverseCumulativeRng"]
