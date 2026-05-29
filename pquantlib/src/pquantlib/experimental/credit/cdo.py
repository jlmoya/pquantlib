"""CDO — generic copula-based CDO instrument (carved out in W3-C).

# C++ parity: ql/experimental/credit/cdo.{hpp,cpp} (v1.42.1).

The C++ ``CDO`` class is a CDO instrument with explicit copula
parameterization (it takes a ``OneFactorCopula`` argument and uses
``LossDistBucketing`` for inline loss-distribution construction).
It predates the ``SyntheticCDO`` + ``DefaultLossModel`` architecture and
overlaps almost entirely with the latter; in practice the QuantLib test
suite uses ``SyntheticCDO`` (cluster_w3c covers that path).

# Carve-out: The ``CDO`` class itself is structurally identical to
# ``SyntheticCDO`` but takes ``Handle<OneFactorCopula>`` directly. Since
# ``OneFactorCopula`` is in cluster W3-D (one-factor copula models), the
# concrete ``CDO`` is deferred until both W3-D and the implicit-correlation
# Brent solver are in place. Importing this module gives access to a
# documentation-only stub that raises on instantiation.
"""

from __future__ import annotations

from pquantlib import qassert


class CDO:
    """Stub for the C++ generic-copula CDO class.

    # Carve-out: see module docstring. Use ``SyntheticCDO`` instead.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        qassert.fail(
            "CDO is not implemented in W3-C; use SyntheticCDO "
            "(ql/experimental/credit/syntheticcdo.hpp parity). The C++ "
            "CDO class is a deferred carve-out — it requires OneFactorCopula "
            "from cluster W3-D + an implicit-correlation Brent solver.",
        )


__all__ = ["CDO"]
