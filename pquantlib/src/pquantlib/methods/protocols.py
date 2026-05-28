"""Cross-cluster structural protocols for numerical methods.

# C++ parity: none — this module exists to give the Python type-checker
# structural (PEP 544) protocols so MC engines, FD engines, and lattice
# pricers can accept *any* asset / lattice / path-generator that
# matches the contract, without binding to the concrete inheritance
# tree.

Three protocols are defined:

- ``DiscretizedAssetProtocol`` — the structural shape of
  ``methods.lattices.DiscretizedAsset``. Used by lattice pricers that
  want to accept user-defined asset types (e.g. an exotic basket
  option) without forcing them to inherit from the concrete base.

- ``LatticeProtocol`` — the structural shape of
  ``methods.lattices.Lattice``. Used by ``DiscretizedAsset`` itself
  so its ``method_`` member can be type-checked against any tree/FD
  lattice implementation, not just the abstract base.

- ``PathGeneratorProtocol`` — the structural shape of an MC path
  generator. C++ uses templates (``template <class PathGenerator>``)
  for compile-time duck typing; Python's structural typing collapses
  that into a ``Protocol`` with ``next() -> path`` and ``dimension()
  -> int``. The exact return type of ``next()`` is left to the concrete
  generator (``MultiPath`` vs ``Path`` etc.) — declared as ``Any``
  here because the Path types live in a later cluster.

These protocols are *not* inherited from by the concrete classes
(they're structural). They are purely a type-checker convenience —
``isinstance(obj, DiscretizedAssetProtocol)`` would only work if we
marked them ``@runtime_checkable``, which we deliberately avoid for
the path generator (``Any`` return type makes runtime checking
meaningless) and only enable for the simpler protocols.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pquantlib.math.array import Array


@runtime_checkable
class DiscretizedAssetProtocol(Protocol):
    """Structural shape of a discretized asset.

    Mirrors :class:`pquantlib.methods.lattices.DiscretizedAsset`'s
    public surface — exactly the methods that ``Lattice`` calls on
    an asset during rollback. Protocols cannot enforce abstract
    methods, but ``isinstance`` checks against this Protocol verify
    that all the listed methods exist at runtime.
    """

    @property
    def time(self) -> float: ...

    @property
    def values(self) -> Array: ...

    def set_time(self, t: float) -> None: ...

    def set_values(self, v: Array) -> None: ...

    def reset(self, size: int) -> None: ...

    def mandatory_times(self) -> list[float]: ...

    def pre_adjust_values(self) -> None: ...

    def post_adjust_values(self) -> None: ...

    def adjust_values(self) -> None: ...

    def is_on_time(self, t: float) -> bool: ...


@runtime_checkable
class LatticeProtocol(Protocol):
    """Structural shape of a lattice-based numerical method.

    Mirrors :class:`pquantlib.methods.lattices.Lattice`'s public
    surface. The four high-level rollback methods, the tree-side
    ``size`` / ``descendant`` / ``probability`` accessors, and the
    ``time_grid()`` + ``grid(t)`` accessors are listed.

    The ``initialize`` / ``rollback`` / ``partial_rollback`` /
    ``present_value`` methods take ``object`` (rather than
    ``DiscretizedAssetProtocol``) for ergonomic reasons — pyright
    would otherwise require explicit narrowing at every call site;
    concrete lattices type-check at runtime via ``isinstance``.
    """

    def time_grid(self) -> Any: ...

    def columns(self) -> int: ...

    def size(self, i: int) -> int: ...

    def descendant(self, i: int, index: int, branch: int) -> int: ...

    def probability(self, i: int, index: int, branch: int) -> float: ...

    def initialize(self, asset: object, t: float) -> None: ...

    def rollback(self, asset: object, to_t: float) -> None: ...

    def partial_rollback(self, asset: object, to_t: float) -> None: ...

    def present_value(self, asset: object) -> float: ...

    def grid(self, t: float) -> Array: ...


class PathGeneratorProtocol(Protocol):
    """Structural shape of a Monte Carlo path generator.

    The exact path type varies — single-asset ``Path``, multi-asset
    ``MultiPath``, plus the various ``Sample[Path]`` wrappers. The
    Protocol uses ``Any`` for the path type so it accepts any of them.

    C++ uses templates here (e.g. ``MonteCarloModel<Path,
    PathGenerator, PathPricer>``); Python's structural typing makes
    this a Protocol used at type-check time. Not marked
    ``@runtime_checkable`` because the ``Any`` return type would make
    ``isinstance`` checks effectively meaningless.
    """

    def next(self) -> Any: ...

    def dimension(self) -> int: ...
