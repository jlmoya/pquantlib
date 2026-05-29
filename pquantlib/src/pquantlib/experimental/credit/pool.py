"""Pool — collection of Issuers backing a basket / CDO.

# C++ parity: ql/experimental/credit/pool.{hpp,cpp} (v1.42.1).

A Pool stores a set of Issuers keyed by name, the contract-trigger
DefaultProbKey under which each name enters the basket, and a
mutable per-name "time" attribute (used by simulation code).

The C++ class uses std::map for ordered lookups; Python's dict provides
the same O(log n)→O(1) lookup semantics. Insertion order is preserved
in ``names_`` (mirroring C++ which keeps an explicit vector of names).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.issuer import Issuer


class Pool:
    """A collection of ``Issuer`` instances keyed by name.

    Each entry carries its own contract-trigger DefaultProbKey + a mutable
    ``time`` attribute (defaults to 0.0).

    Mutations: ``add`` (no-op if name already present); ``set_time``
    overrides; ``clear`` empties the pool.
    """

    __slots__ = ("_data", "_default_keys", "_names", "_time")

    def __init__(self) -> None:
        self._data: dict[str, Issuer] = {}
        self._time: dict[str, float] = {}
        self._names: list[str] = []
        self._default_keys: dict[str, DefaultProbKey] = {}

    def size(self) -> int:
        """Number of issuers currently in the pool."""
        return len(self._names)

    def clear(self) -> None:
        """Drop all issuers."""
        self._data.clear()
        self._time.clear()
        self._names.clear()
        self._default_keys.clear()

    def has(self, name: str) -> bool:
        """True iff ``name`` is registered."""
        return name in self._data

    def add(
        self,
        name: str,
        issuer: Issuer,
        contract_trigger: DefaultProbKey | None = None,
    ) -> None:
        """Add an issuer under ``name``. No-op if name already present.

        # C++ parity: pool.cpp:44-52. Default contract_trigger is the
        # NorthAmericaCorpDefaultKey on an empty currency + SeniorSec;
        # the Python port keeps this opt-in (caller passes one
        # explicitly or we use an empty DefaultProbKey as sentinel).
        """
        if not self.has(name):
            self._data[name] = issuer
            self._time[name] = 0.0
            self._names.append(name)
            self._default_keys[name] = (
                contract_trigger if contract_trigger is not None else DefaultProbKey()
            )

    def get(self, name: str) -> Issuer:
        """Return the issuer registered under ``name`` or raise."""
        qassert.require(self.has(name), f"{name} not found")
        return self._data[name]

    def default_key(self, name: str) -> DefaultProbKey:
        """Return the contract-trigger key for ``name`` or raise."""
        qassert.require(self.has(name), f"{name} not found")
        return self._default_keys[name]

    def set_time(self, name: str, time: float) -> None:
        """Override the per-name time attribute.

        # C++ parity: pool.cpp:69-71 — Note that C++ allows ``setTime``
        # on a non-existent name (silently creates the entry in the map).
        # The Python port preserves that lax behaviour.
        """
        self._time[name] = time

    def get_time(self, name: str) -> float:
        """Return the per-name time attribute or raise."""
        qassert.require(self.has(name), f"{name} not found")
        return self._time[name]

    def names(self) -> list[str]:
        """Return the insertion-ordered list of names."""
        return list(self._names)

    def default_keys(self) -> list[DefaultProbKey]:
        """Return the contract-trigger keys (one per name), in dict-order.

        # C++ parity: pool.cpp:77-83 — note C++ iterates the std::map
        # which is sorted-by-name. Python's dict preserves insertion order;
        # if you need name-sorted output, sort externally.
        """
        return list(self._default_keys.values())
