"""MultiPath — correlated multiple asset paths.

# C++ parity: ql/methods/montecarlo/multipath.hpp (v1.42.1) —
# ``class MultiPath``.

A list-of-``Path`` container.  Each ``Path`` shares the same time
grid; ``multipath[j]`` is the path followed by the j-th asset.

Python divergences:

* C++ ``std::vector<Path>`` becomes ``list[Path]``.
* No ``operator[](Size) &`` mutable element access exposed publicly
  (``__getitem__`` returns the immutable shared reference). The
  contained ``Path`` objects' ``values`` properties are mutable for
  ``MultiPathGenerator`` to write into.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.methods.montecarlo.path import Path
from pquantlib.time.time_grid import TimeGrid


class MultiPath:
    """Container of correlated single-asset paths.

    # C++ parity: ``MultiPath`` (multipath.hpp).

    Two construction modes (mirroring C++):

    * ``MultiPath(n_assets, time_grid)`` — n_assets fresh Path's on
      the same time grid.
    * ``MultiPath(paths_list)`` — direct construction from existing
      paths (all sharing the same time grid; not enforced — caller's
      responsibility).
    """

    __slots__ = ("_paths",)

    def __init__(self, paths: list[Path]) -> None:
        """Construct directly from a list of paths.

        # C++ parity: ``MultiPath(std::vector<Path>)``.
        """
        self._paths: list[Path] = paths

    @classmethod
    def from_assets_and_grid(cls, n_assets: int, time_grid: TimeGrid) -> MultiPath:
        """Construct ``n_assets`` paths on the same time grid.

        # C++ parity: ``MultiPath(Size, const TimeGrid&)``.
        """
        qassert.require(n_assets > 0, "number of asset must be positive")
        paths = [Path(time_grid) for _ in range(n_assets)]
        return cls(paths)

    # --- inspectors -------------------------------------------------------

    def asset_number(self) -> int:
        """Number of asset paths — C++ ``assetNumber``."""
        return len(self._paths)

    def path_size(self) -> int:
        """Number of points per asset path — C++ ``pathSize``."""
        return self._paths[0].length()

    def __len__(self) -> int:
        return self.asset_number()

    def __getitem__(self, j: int) -> Path:
        """Return the j-th asset path."""
        return self._paths[j]

    def at(self, j: int) -> Path:
        """Bounds-checked access — mirrors C++ ``MultiPath::at``."""
        qassert.require(0 <= j < len(self._paths), f"asset index {j} out of range")
        return self._paths[j]

    def __iter__(self):
        return iter(self._paths)


__all__ = ["MultiPath"]
