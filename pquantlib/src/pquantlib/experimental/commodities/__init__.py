"""Commodity experimental foundation types.

# C++ parity: ql/experimental/commodities/* (v1.42.1).

This sub-package hosts the commodity-domain foundation types: units of
measure (and petroleum UOM concretes), UOM conversions + a singleton
conversion-factor registry, commodity-type and payment-term flyweights,
typed commodity quantities with UOM-converting arithmetic, exchange-
contract specs, date intervals + pricing periods, the ``Commodity``
instrument base, and the static commodity-pricing helpers.

Downstream W7-C commodity instruments (EnergyCommodity / EnergySwap /
EnergyFuture / CommodityIndex / CommodityCurve) layer on top of these.
"""
