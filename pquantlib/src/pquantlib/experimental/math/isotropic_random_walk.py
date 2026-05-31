"""Isotropic random-walk step generator.

# C++ parity: ql/experimental/math/isotropicrandomwalk.hpp (v1.42.1)
# (Copyright 2015 Andres Hernandez).

A draw from the supplied ``distribution`` gives the *radius* of a step
on a ``dim``-dimensional sphere; the direction is chosen isotropically
(uniformly over the sphere surface) using a separate Mersenne-Twister
angle stream. Per-dimension ``weights`` rescale the sphere into an
ellipsoid (used by the firefly / PSO Lévy-flight inertia to respect
parameter-box aspect ratios).

The spherical parametrisation follows the C++ recurrence exactly:
for ``dim > 1`` the first ``dim-2`` coordinates peel off
``radius*cos(phi)`` while ``radius`` accumulates ``*= sin(phi)`` and a
fresh ``phi = pi*u`` is drawn each step; the final two coordinates use
``cos(2 phi)`` / ``sin(2 phi)``. For ``dim == 1`` a single uniform
picks the sign.

# C++ parity note: in the final ``sin`` term, C++ reads ``*weight``
# (the last advanced iterator) for *both* the penultimate ``cos`` and
# the final ``sin`` coordinate — the weight iterator is **not** advanced
# for the final component. PQuantLib reproduces this (``weights[widx]``
# is reused for ``out[widx + 1]``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng

if TYPE_CHECKING:
    from collections.abc import Callable


class _RadiusEngine(Protocol):
    """Minimal uniform-engine interface a radius distribution consumes."""

    def next_real(self) -> float: ...


class IsotropicRandomWalk:
    """Isotropic random walk on a (possibly weighted) sphere.

    # C++ parity: ``template <Distribution, Engine> class
    # IsotropicRandomWalk`` in
    # ql/experimental/math/isotropicrandomwalk.hpp:38-122 (v1.42.1).

    Parameters
    ----------
    engine:
        Uniform engine consumed by ``distribution`` to draw the radius
        (exposes ``next_real()``).
    distribution:
        Callable ``distribution(engine) -> radius`` (e.g.
        ``LevyFlightDistribution``).
    dim:
        Dimension of the walk.
    weights:
        Optional per-dimension weights (default all-ones). If supplied,
        length must equal ``dim``.
    seed:
        Seed for the internal angle Mersenne-Twister stream.
    """

    __slots__ = ("_dim", "_distribution", "_engine", "_rng", "_weights")

    def __init__(
        self,
        engine: _RadiusEngine,
        distribution: Callable[[_RadiusEngine], float],
        dim: int,
        weights: npt.NDArray[np.float64] | None = None,
        seed: int = 1,
    ) -> None:
        self._engine: _RadiusEngine = engine
        self._distribution: Callable[[_RadiusEngine], float] = distribution
        self._rng: MersenneTwisterUniformRng = MersenneTwisterUniformRng(seed)
        self._dim: int = dim
        if weights is None or weights.size == 0:
            self._weights: npt.NDArray[np.float64] = np.ones(dim, dtype=np.float64)
        else:
            qassert.require(dim == weights.size, "Invalid weights")
            self._weights = weights.astype(np.float64, copy=True)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Per-dimension weights (read-only view of the current state).

        Not in the C++ API (the field is private there) — exposed in
        PQuantLib as a read-only inspector for the box-aspect ellipsoid
        rescaling done by ``set_dimension_bounded``.
        """
        return self._weights

    @property
    def dimension(self) -> int:
        """Current walk dimension (read-only inspector; not in C++ API)."""
        return self._dim

    def next_real(self, out: npt.NDArray[np.float64]) -> None:
        """Write the next ``dim``-dimensional step into ``out`` (in place).

        # C++ parity: ``template<InputIterator> nextReal(first)``
        # isotropicrandomwalk.hpp:62-83 — the iterator-target ``first``
        # becomes the numpy buffer ``out`` (length ``dim``).
        """
        radius = self._distribution(self._engine)
        weights = self._weights
        if self._dim > 1:
            widx = 0
            # Isotropic random direction.
            phi = math.pi * self._rng.next_real()
            for _ in range(self._dim - 2):
                out[widx] = radius * math.cos(phi) * weights[widx]
                widx += 1
                radius *= math.sin(phi)
                phi = math.pi * self._rng.next_real()
            out[widx] = radius * math.cos(2.0 * phi) * weights[widx]
            # C++ parity: final sin term reuses weights[widx] (iterator
            # not advanced for the last component).
            out[widx + 1] = radius * math.sin(2.0 * phi) * weights[widx]
        elif self._rng.next_real() < 0.5:
            out[0] = -radius * weights[0]
        else:
            out[0] = radius * weights[0]

    def set_dimension(self, dim: int, weights: npt.NDArray[np.float64] | None = None) -> None:
        """Reset ``dim`` (and weights), all-ones if ``weights`` omitted.

        # C++ parity: the two-arg overloads
        # ``setDimension(Size)`` / ``setDimension(Size, const Array&)``
        # isotropicrandomwalk.hpp:84-94.
        """
        if weights is None:
            self._dim = dim
            self._weights = np.ones(dim, dtype=np.float64)
        else:
            qassert.require(dim == weights.size, "Invalid weights")
            self._dim = dim
            self._weights = weights.astype(np.float64, copy=True)

    def set_dimension_bounded(
        self,
        dim: int,
        lower_bound: npt.NDArray[np.float64],
        upper_bound: npt.NDArray[np.float64],
    ) -> None:
        """Reset ``dim`` with weights derived from a parameter box.

        # C++ parity: ``setDimension(Size, lowerBound, upperBound)``
        # isotropicrandomwalk.hpp:100-117. Weights are the per-dimension
        # box widths normalised by the largest width, turning the sphere
        # into a box-aspect ellipsoid.
        """
        qassert.require(dim == lower_bound.size, "Incompatible dimension and lower bound")
        qassert.require(dim == upper_bound.size, "Incompatible dimension and upper bound")
        bounds = (upper_bound - lower_bound).astype(np.float64, copy=True)
        max_bound = float(np.max(bounds))
        max_bound = 1.0 / max_bound
        bounds *= max_bound
        self.set_dimension(dim, bounds)
