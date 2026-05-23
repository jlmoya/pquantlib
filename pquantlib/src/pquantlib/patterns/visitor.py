"""Visitor pattern as runtime-checkable Protocols.

# C++ parity: ql/patterns/visitor.hpp (v1.42.1) — class Visitor<T>, AcyclicVisitor.

Python's structural typing collapses the C++/Java separate
Visitor / Visitable / PolymorphicVisitor / PolymorphicVisitable family
into one ``Visitor`` Protocol + one ``Visitable`` Protocol. The C++
template type-parameter is dropped because structural matching at the
``.visit`` call site is what the runtime actually checks; type-narrowing
on a specific target type is done by the visit-method's own signature
in concrete Visitor classes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Visitor(Protocol):
    def visit(self, target: object) -> None: ...


@runtime_checkable
class Visitable(Protocol):
    def accept(self, visitor: Visitor) -> None: ...
