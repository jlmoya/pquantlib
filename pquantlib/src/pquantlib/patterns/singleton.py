"""Singleton metaclass-based base.

# C++ parity: ql/patterns/singleton.hpp (v1.42.1) — class Singleton<T>.

Subclass ``Singleton`` to get one-instance-per-class semantics:

    class Settings(Singleton):
        ...

    Settings() is Settings()  # True

Distinct subclasses get distinct instances.
"""

from __future__ import annotations

from typing import Any, ClassVar


class _SingletonMeta(type):
    """Metaclass: returns the cached instance per class on every call.

    The instance map is keyed by ``Any`` rather than ``type`` because pyright
    sees ``cls`` inside ``__call__`` as ``Self@_SingletonMeta`` (a metaclass
    type), not as ``type`` — even though it is a ``type`` at runtime.
    """

    _instances: ClassVar[dict[Any, Any]] = {}

    def __call__(cls, *args: object, **kwargs: object) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(metaclass=_SingletonMeta):
    """Subclass to get singleton semantics."""
