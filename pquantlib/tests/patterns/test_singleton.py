"""Tests for pquantlib.patterns.singleton."""

from __future__ import annotations

from pquantlib.patterns.singleton import Singleton


class _Settings(Singleton):
    def __init__(self) -> None:
        self.value: int = 0


class _OtherSettings(Singleton):
    def __init__(self) -> None:
        self.value: int = 99


def test_singleton_same_instance_per_subclass() -> None:
    a = _Settings()
    b = _Settings()
    assert a is b


def test_singleton_distinct_instances_across_subclasses() -> None:
    a = _Settings()
    b = _OtherSettings()
    assert a is not b
    assert isinstance(a, _Settings)
    assert isinstance(b, _OtherSettings)


def test_singleton_state_persists_across_lookups() -> None:
    a = _Settings()
    a.value = 42
    b = _Settings()
    assert b.value == 42
    # Reset for test isolation; later tests assume default 0 might NOT hold,
    # so this test documents the invariant rather than relying on it.
