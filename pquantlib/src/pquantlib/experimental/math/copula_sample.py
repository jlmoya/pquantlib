"""VectorSample — a weighted vector-valued Monte Carlo sample.

# C++ parity: ql/methods/montecarlo/sample.hpp ``Sample<std::vector<Real>>``.

The copula RNGs in :mod:`pquantlib.experimental.math` return a *vector*-valued
sample (a 2-vector ``[u1, u2]`` of correlated uniforms). The library's existing
:class:`pquantlib.math.randomnumbers.random_number_generator.Sample` is typed
``value: float``; this companion dataclass is the vector-valued analogue, kept
local to the experimental copula machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class VectorSample:
    """A weighted vector sample: ``value`` is a list of reals."""

    value: list[float]
    weight: float = field(default=1.0)
