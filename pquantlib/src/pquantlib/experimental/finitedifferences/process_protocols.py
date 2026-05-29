"""Process protocols for cross-cluster wiring (W5-B).

These :class:`typing.Protocol` types describe the *structural*
interface that the W5-B FD engines require from their underlying
stochastic processes. The concrete process classes will be ported by
the W5-A subagent (Extended-OU + Kluge processes); until then these
Protocols let W5-B's engines compile, type-check, and be wired to
mocks for unit tests.

# C++ parity:
# * ExtendedOrnsteinUhlenbeckProcess: ql/experimental/processes/extendedornsteinuhlenbeckprocess.hpp
# * ExtOUWithJumpsProcess: ql/experimental/processes/extouwithjumpsprocess.hpp
# * KlugeExtOUProcess: ql/experimental/processes/klugeextouprocess.hpp
# (v1.42.1)

Each protocol exposes the minimum surface used by the engines'
constructors — typically ``x0()`` (initial state) and ``factors()``
(dimension count). Engine ``calculate()`` paths will need a richer
surface (drift / diffusion / FD discretization) — those bindings are
deferred to W5-A's full implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExtendedOrnsteinUhlenbeckProcessProtocol(Protocol):
    """Minimal interface for Extended Ornstein-Uhlenbeck processes.

    # C++ parity: ``class ExtendedOrnsteinUhlenbeckProcess``.
    Used by :class:`FdSimpleExtOUStorageEngine`.
    """

    def x0(self) -> float:
        """Initial state (the C++ ``x0()`` accessor)."""
        ...


@runtime_checkable
class ExtOUWithJumpsProcessProtocol(Protocol):
    """Minimal interface for Extended-OU + jumps processes.

    # C++ parity: ``class ExtOUWithJumpsProcess``.
    Used by :class:`FdSimpleExtOUJumpSwingEngine`.
    """

    def factors(self) -> int:
        """Number of stochastic factors driving the process."""
        ...


@runtime_checkable
class KlugeExtOUProcessProtocol(Protocol):
    """Minimal interface for Kluge + Extended-OU combo processes.

    # C++ parity: ``class KlugeExtOUProcess``.
    Used by :class:`FdSimpleKlugeExtOUVPPEngine`.
    """

    def factors(self) -> int:
        """Number of stochastic factors (typically 3: power LN-OU, jump, gas)."""
        ...


__all__ = [
    "ExtOUWithJumpsProcessProtocol",
    "ExtendedOrnsteinUhlenbeckProcessProtocol",
    "KlugeExtOUProcessProtocol",
]
