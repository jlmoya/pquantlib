"""Experimental barrier-option instruments and engines.

# C++ parity: ql/experimental/barrieroption/* + ql/instruments/
#   {partialtimebarrieroption,softbarrieroption}.hpp (v1.42.1).

Hosts the barrier-option family that lives in ``ql/experimental`` plus
two instruments (``PartialTimeBarrierOption`` and ``SoftBarrierOption``)
that QuantLib promoted into ``ql/instruments`` while keeping their
analytic engines under ``ql/pricingengines/barrier``. The Python port
collects all of these under one ``experimental.barrieroption`` package
because they share a single problem domain (specialty barrier monitoring
regimes) and naturally co-locate.
"""
