"""Monte Carlo multi-asset (basket) framework (experimental.mcbasket).

# C++ parity: ql/experimental/mcbasket/* (v1.43).

* payoff surface — :mod:`path_payoff`, :mod:`adapted_path_payoff`;
* instrument — :mod:`path_multi_asset_option`;
* European engine — :mod:`mc_path_basket_engine` (``MCPathBasketEngine``,
  ``EuropeanPathMultiPathPricer``, ``MakeMCPathBasketEngine``);
* early-exercise engines — :mod:`longstaff_schwartz_multi_path_pricer`,
  :mod:`mc_longstaff_schwartz_path_engine` (the abstract driver) and
  :mod:`mc_american_path_engine` (``MCAmericanPathEngine`` +
  ``MakeMCAmericanPathEngine``).

Like C++, the modules are imported directly — this package re-exports nothing.
"""
