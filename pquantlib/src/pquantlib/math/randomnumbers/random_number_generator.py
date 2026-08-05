"""Random number generator Protocol + sample dataclass.

# C++ parity: ql/methods/montecarlo/sample.hpp (Sample<Real>) and the
# implicit interface that the templated RNG classes in
# ql/math/randomnumbers/* satisfy via their ``next()`` and (optionally)
# ``dimension()`` member functions, v1.42.1.

C++ uses ``template <class T> struct Sample { T value; Real weight; };``
together with template-typed RNGs (no runtime polymorphism — RNGs are
duck-typed at compile time). Python's structural typing collapses that
into a ``runtime_checkable`` Protocol whose ``next()`` returns a
``Sample`` instance.

``dimension()`` is defined on the Protocol but is OPTIONAL for non-
sequence (scalar) RNGs: it defaults to ``1``. Sequence RNGs (e.g.
``SobolRsg``, when ported in a later cluster) override it with their
problem dimension. This matches the C++ convention where scalar RNGs
do not expose ``dimension()`` while sequence RNGs do — Python's lack
of compile-time template duck-typing means we expose it uniformly with
a sensible default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pquantlib.math.array import Array


@dataclass(frozen=True, slots=True)
class Sample:
    """One draw from an RNG.

    # C++ parity: ql/methods/montecarlo/sample.hpp ``Sample<Real>``.

    ``weight`` defaults to 1.0 (the only value that uniform/Gaussian
    RNGs in v1.42.1 emit). Importance-sampling RNGs may set a different
    weight.
    """

    value: float
    weight: float = 1.0


@runtime_checkable
class RandomNumberGenerator(Protocol):
    """Structural type for uniform/Gaussian random number generators.

    # C++ parity: the duck-typed interface satisfied by
    # ``MersenneTwisterUniformRng``, ``KnuthUniformRng``,
    # ``LecuyerUniformRng``, ``Ranlux64UniformRng``,
    # ``Xoshiro256StarStarUniformRng``, and ``BoxMullerGaussianRng`` in
    # v1.42.1.

    Concrete scalar RNGs only need to implement ``next``. ``dimension``
    is OPTIONAL: its default returns 1 (scalar). Sequence RNGs
    (multi-dimensional, e.g. SobolRsg in a future cluster) override
    ``dimension`` to report the problem dimension.
    """

    def next(self) -> Sample: ...

    def dimension(self) -> int:
        """Default scalar dimension (1)."""
        return 1


@dataclass(frozen=True, slots=True, eq=False)
class SequenceSample:
    """One draw from a sequence generator.

    # C++ parity: ``Sample<std::vector<Real>>``
    # (ql/methods/montecarlo/sample.hpp).

    ``eq=False`` because ``value`` is a numpy array: the generated
    ``__eq__`` would return an array and blow up in a boolean context.
    """

    value: Array
    weight: float = 1.0


#: What a uniform sequence generator's ``next_sequence`` may return.
#:
#: The pure low-discrepancy generators in this package (``SobolRsg``,
#: ``Burley2020SobolRsg``) return a bare ``Array``: their C++ counterparts
#: always set ``weight = 1``, and the repo-wide convention is that a vector of
#: reals is an ``Array``. The weight-carrying generators
#: (``RandomSequenceGenerator``, ``InverseCumulativeRsg``, ``RandomizedLDS``)
#: return a ``SequenceSample``. Consumers call :func:`sequence_parts` rather
#: than caring which.
type SequenceValue = SequenceSample | Array


def sequence_parts(sample: SequenceValue) -> tuple[Array, float]:
    """Split either sequence representation into ``(values, weight)``."""
    if isinstance(sample, SequenceSample):
        return sample.value, sample.weight
    return sample, 1.0


@runtime_checkable
class UniformSequenceGenerator(Protocol):
    """Structural type for multi-dimensional (sequence) generators.

    # C++ parity: the duck-typed ``USG`` template parameter of
    # ``InverseCumulativeRsg`` and the ``LDS``/``PRS`` parameters of
    # ``RandomizedLDS``:
    #     USG::sample_type USG::nextSequence() const;
    #     Size USG::dimension() const;
    """

    def next_sequence(self) -> SequenceValue: ...

    def dimension(self) -> int: ...


@runtime_checkable
class InverseCumulative(Protocol):
    """Structural type for an inverse cumulative distribution function.

    # C++ parity: the duck-typed ``IC`` template parameter of
    # ``InverseCumulativeRng`` / ``InverseCumulativeRsg``:
    #     IC::IC(); Real IC::operator()(Real) const;
    """

    def __call__(self, x: float, /) -> float: ...
