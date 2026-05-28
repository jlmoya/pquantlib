"""Sobol low-discrepancy sequence generator.

# C++ parity: ql/math/randomnumbers/sobolrsg.{hpp,cpp} (v1.42.1)
#             ``class SobolRsg``.

C++ ``SobolRsg`` implements Sobol' sequences from scratch using a
Gray-code counter and bitwise operations, with eight different
direction-integer families baked in (Unit, Jaeckel, SobolLevitan,
SobolLevitanLemieux, JoeKuoD5/D6/D7, Kuo/Kuo2/Kuo3). The default is
Jaeckel.

The Python port wraps ``scipy.stats.qmc.Sobol`` (SciPy 1.7+). scipy
internally uses **Joe-Kuo** direction numbers (the most widely used
modern family). For the L5-A pilot we accept that the underlying
direction-integer family is fixed at Joe-Kuo — the C++ alternative
sets are explicitly carved out as deferred work.

A second port-relevant divergence: scipy's ``Sobol`` emits the **origin
``(0, 0, ..., 0)`` as its first vector** while C++ skips it (its
Gray-code counter starts at 1, not 0). To match the C++ probe's
"first-vector" expectations the Python wrapper transparently advances
past the origin at construction. The deterministic Sobol sequence is
independent of the (random) seed argument unless ``scramble=True``; the
``seed`` parameter is forwarded to scipy and only matters for
``Burley2020SobolRsg``.

Observation when comparing to the C++ probe at (dim=2, seed=42): the
first 5 C++ vectors (Jaeckel default) are::

    (0.5, 0.5), (0.75, 0.25), (0.25, 0.75), (0.375, 0.375), (0.875, 0.875)

For dim<=2 the first ~5 outputs coincide across all of the C++
direction-integer families and scipy's Joe-Kuo (the low-order Sobol
generators are largely determined by the leading polynomial mod 2).
The cross-validation tests assert exact agreement on this prefix.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from scipy.stats import qmc  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.array import Array

# Maximum supported dimension. C++ defines PPMT_MAX_DIM = 21200 via the
# primitive polynomials table. scipy's maximum is 21201 (Joe-Kuo).
# Cap at the smaller of the two for safety.
PPMT_MAX_DIM: Final[int] = 21200


class SobolRsg:
    """Sobol' low-discrepancy sequence over ``[0, 1)^d``.

    # C++ parity: ``SobolRsg`` (sobolrsg.hpp:110-151).

    Parameters
    ----------
    dimensionality:
        Output vector dimension. Must be ``1 <= d <= 21200``.
    seed:
        Forwarded to scipy. Has no effect on the (deterministic)
        unscrambled Sobol sequence; matters only when the subclass
        ``Burley2020SobolRsg`` enables scrambling.
    direction_integers:
        Accepted for API symmetry with C++ (carve-out — scipy uses
        Joe-Kuo unconditionally). Pass-through only.
    """

    def __init__(
        self,
        dimensionality: int,
        seed: int = 0,
        direction_integers: str | None = None,
    ) -> None:
        qassert.require(
            dimensionality >= 1,
            f"SobolRsg dimensionality must be >= 1, got {dimensionality}",
        )
        qassert.require(
            dimensionality <= PPMT_MAX_DIM,
            f"SobolRsg dimensionality must be <= {PPMT_MAX_DIM}, got {dimensionality}",
        )
        self._dim: int = dimensionality
        self._seed: int = seed
        self._direction_integers: str | None = direction_integers
        # scipy's Sobol is internally stateful; we configure it once and
        # iterate. ``scramble=False`` for the base class (a Burley2020
        # subclass overrides via __init__).
        self._engine: Any = self._make_engine(scramble=False)
        # scipy emits the origin (0,...,0) as the first draw, while C++
        # skips it. Pre-consume one draw so ``next_sequence`` returns
        # the same first vector as C++.
        self._engine.fast_forward(1)
        self._counter: int = 1  # number of draws actually consumed past origin
        # ``_last`` mirrors C++ ``sequence_`` (kept around for the
        # ``last_sequence`` accessor).
        self._last: Array | None = None

    # -- engine construction (subclass hook) ------------------------------

    def _make_engine(self, *, scramble: bool) -> Any:
        # scipy 1.16 renamed ``seed`` -> ``rng`` (numpy 2.0 random-API
        # migration). ``rng`` accepts the same integer seed values.
        return qmc.Sobol(d=self._dim, scramble=scramble, rng=self._seed)

    # -- C++ public API ----------------------------------------------------

    def next_sequence(self) -> Array:
        """Return the next ``d``-vector, components in ``[0, 1)``.

        # C++ parity: ``SobolRsg::nextSequence()`` returns ``Sample
        # <vector<Real>>``; the Python port returns a numpy float64
        # 1-D array (the ``Array`` alias). The C++ ``weight`` field
        # is always 1 for Sobol and is not exposed.
        """
        v = np.asarray(self._engine.random(1)[0], dtype=np.float64)
        self._last = v
        self._counter += 1
        return v

    def last_sequence(self) -> Array:
        """Return the most-recent vector emitted by ``next_sequence``.

        # C++ parity: ``SobolRsg::lastSequence()``. Raises if no draw
        # has been taken yet (mirrors the undefined-behavior of the C++
        # version reading ``sequence_`` before any draw, but we choose
        # a clear LibraryException over silent garbage).
        """
        last = self._last
        qassert.require(
            last is not None,
            "SobolRsg.last_sequence called before next_sequence",
        )
        # Local binding above so type-narrowing carries through the
        # ``qassert.require`` truthiness gate.
        assert last is not None  # pyright narrowing aid
        return last

    def skip_to(self, n: int) -> None:
        """Advance internal counter so the next ``next_sequence`` call
        returns the C++[n] vector (0-indexed, post-origin).

        # C++ parity: ``SobolRsg::skipTo(uint32 n)`` returns vector at
        # position ``n+1`` of the C++ Gray-code-counted sequence.
        # The Python port aligns ``skip_to(n) -> next_sequence()``
        # with ``C++[n]`` (i.e. ``skip_to(0)`` makes the next draw
        # the first non-origin vector, == 0.5, 0.5 in dim 2).
        #
        # scipy's ``fast_forward(k)`` consumes k draws starting from
        # the origin. After ``reset(); fast_forward(n+1)``, scipy
        # has emitted the origin + first n vectors, so the next draw
        # is ``C++[n]``.
        """
        qassert.require(n >= 0, f"SobolRsg.skip_to requires n >= 0, got {n}")
        self._engine.reset()
        # Skip origin + n post-origin vectors; the next draw is C++[n].
        self._engine.fast_forward(n + 1)
        self._counter = n + 1
        self._last = None

    def reset(self) -> None:
        """Reset the generator to its initial state (just past the origin)."""
        self._engine.reset()
        # Skip past the origin to match C++ Gray-code-start behavior.
        self._engine.fast_forward(1)
        self._counter = 1
        self._last = None

    def dimension(self) -> int:
        """Output vector dimension.

        # C++ parity: ``SobolRsg::dimension()``.
        """
        return self._dim
