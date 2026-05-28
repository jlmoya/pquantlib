"""FdmLinearOpLayout — memory layout for a multi-D FD operator.

# C++ parity: ql/methods/finitedifferences/operators/fdmlinearoplayout.{hpp,cpp}
# + ql/methods/finitedifferences/operators/fdmlinearopiterator.hpp (v1.42.1).

The layout owns a ``dim`` tuple (sizes per direction) and computes a
``spacing`` tuple (row-major strides) so a multi-D coordinate
``(c_0, c_1, ..., c_{k-1})`` maps to a flat index via
``inner_product(coords, spacing)``.

Two helpers are exposed:

* ``index(coords)`` — flat index from coordinates.
* ``iter()`` — yields ``FdmLinearOpIterator`` instances over the
  layout in C++ row-major order (``i0`` varies fastest).

The 1-D case is the only one exercised by L5-D; the layout is kept
fully multi-D so multi-asset FD can reuse it in Phase 6 unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import final


@final
class FdmLinearOpIterator:
    """Iterator yielding (index, coordinates) over the layout.

    # C++ parity: ``FdmLinearOpIterator`` in fdmlinearopiterator.hpp.

    The iterator stores an immutable ``dim`` tuple and walks
    ``coordinates`` in row-major order (first axis varies fastest).
    Each instance carries its own flat ``index`` (kept consistent
    with ``coordinates``).
    """

    __slots__ = ("_coordinates", "_dim", "_index")

    def __init__(self, dim: tuple[int, ...], coordinates: tuple[int, ...], index: int) -> None:
        self._dim: tuple[int, ...] = dim
        self._coordinates: tuple[int, ...] = coordinates
        self._index: int = index

    @property
    def index(self) -> int:
        """Flat row-major index."""
        return self._index

    @property
    def coordinates(self) -> tuple[int, ...]:
        """Per-direction coordinates."""
        return self._coordinates


@final
class FdmLinearOpLayout:
    """Multi-D index layout for a FD linear operator.

    # C++ parity: ``FdmLinearOpLayout``.
    """

    __slots__ = ("_dim", "_size", "_spacing")

    def __init__(self, dim: tuple[int, ...] | list[int]) -> None:
        dim_tuple: tuple[int, ...] = tuple(int(d) for d in dim)
        if len(dim_tuple) == 0:
            raise ValueError("dim must be non-empty")
        spacing: list[int] = [0] * len(dim_tuple)
        spacing[0] = 1
        # C++: partial_sum(dim.begin(), dim.end()-1, spacing.begin()+1, multiplies<>()).
        # i.e. spacing[i+1] = spacing[i] * dim[i] for i in 0..n-2.
        for i in range(len(dim_tuple) - 1):
            spacing[i + 1] = spacing[i] * dim_tuple[i]
        size: int = spacing[-1] * dim_tuple[-1]

        self._dim: tuple[int, ...] = dim_tuple
        self._spacing: tuple[int, ...] = tuple(spacing)
        self._size: int = size

    def dim(self) -> tuple[int, ...]:
        """Per-direction sizes."""
        return self._dim

    def spacing(self) -> tuple[int, ...]:
        """Per-direction strides for the flat-index layout."""
        return self._spacing

    def size(self) -> int:
        """Total number of grid points (product of dim)."""
        return self._size

    def index(self, coordinates: tuple[int, ...] | list[int]) -> int:
        """Flat index from per-direction coordinates.

        # C++ parity: ``FdmLinearOpLayout::index`` —
        # ``inner_product(coords, spacing)``.
        """
        coords = tuple(int(c) for c in coordinates)
        if len(coords) != len(self._dim):
            raise ValueError(f"coordinate dimension mismatch: got {len(coords)}, layout has {len(self._dim)}")
        idx = 0
        for c, s in zip(coords, self._spacing, strict=True):
            idx += c * s
        return idx

    def iter(self) -> Iterator[FdmLinearOpIterator]:
        """Yield ``FdmLinearOpIterator`` over the layout in row-major order.

        # C++ parity: ``FdmLinearOpLayout::begin/end`` / range-for-loop;
        # the C++ increment operator advances coordinates with the first
        # axis varying fastest. Python returns a generator over the
        # full range.
        """
        dim = self._dim
        n = len(dim)
        coords = [0] * n
        for index in range(self._size):
            yield FdmLinearOpIterator(dim, tuple(coords), index)
            # Increment coords with axis 0 fastest.
            for i in range(n):
                coords[i] += 1
                if coords[i] == dim[i]:
                    coords[i] = 0
                else:
                    break

    def neighbourhood(self, iterator: FdmLinearOpIterator, direction: int, offset: int) -> int:
        """Flat index of the neighbour offset by ``offset`` along ``direction``.

        # C++ parity: ``FdmLinearOpLayout::neighbourhood(iter, i, offset)``
        # — clamps at the boundary (i.e. returns the iter's own index
        # if going past the edge).
        """
        coords = list(iterator.coordinates)
        new_coord = coords[direction] + offset
        if new_coord < 0:
            new_coord = 0
        elif new_coord >= self._dim[direction]:
            new_coord = self._dim[direction] - 1
        coords[direction] = new_coord
        return self.index(tuple(coords))


__all__ = ["FdmLinearOpIterator", "FdmLinearOpLayout"]
