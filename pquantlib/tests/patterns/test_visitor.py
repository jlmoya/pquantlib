"""Tests for pquantlib.patterns.visitor."""

from __future__ import annotations

from typing import cast

from pquantlib.patterns.visitor import Visitable, Visitor


class _DoubleVisitor:
    def __init__(self) -> None:
        self.last_visited: float | None = None

    def visit(self, target: float) -> None:
        self.last_visited = target * 2


class _Number:
    def __init__(self, value: float) -> None:
        self.value: float = value

    def accept(self, visitor: Visitor) -> None:  # type: ignore[type-arg]
        visitor.visit(self.value)


def test_visitor_protocol_runtime_check() -> None:
    v = _DoubleVisitor()
    # runtime_checkable Protocol membership
    assert isinstance(v, Visitor)


def test_visitable_protocol_runtime_check() -> None:
    n = _Number(3.0)
    assert isinstance(n, Visitable)


def test_visitor_visits_target() -> None:
    n = _Number(3.0)
    v = _DoubleVisitor()
    n.accept(cast("Visitor", v))
    assert v.last_visited == 6.0
