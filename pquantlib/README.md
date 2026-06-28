# PQuantLib

[![PyPI](https://img.shields.io/pypi/v/pquantlib)](https://pypi.org/project/pquantlib/)
[![Python](https://img.shields.io/pypi/pyversions/pquantlib)](https://pypi.org/project/pquantlib/)
[![License](https://img.shields.io/pypi/l/pquantlib)](https://github.com/jlmoya/pquantlib/blob/main/pquantlib/LICENSE.TXT)

A pure-Python port of the industry-standard **[QuantLib](https://github.com/lballabio/QuantLib)**
quantitative-finance library, tracking C++ QuantLib **v1.42.1** (pinned commit `099987f0`) as
the ground truth. Every ported class is cross-validated against the C++ reference across
**4000+ tests** using tiered tolerances (exact / tight / loose).

The library is fully typed (`py.typed`) and delegates the heavy numerics to the scientific
Python stack (numpy / scipy / mpmath) rather than re-implementing them.

## Install

```bash
uv add pquantlib          # or: pip install pquantlib
```

**Requires Python ≥ 3.14.** Runtime dependencies (`numpy`, `scipy`, `mpmath`) are pulled in
automatically.

## Quick start — price a European option

```python
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.processes.generalized_black_scholes_process import GeneralizedBlackScholesProcess
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.pricingengines.vanilla.analytic_european_engine import AnalyticEuropeanEngine

dc = Actual365Fixed()
ref = Date.from_ymd(15, Month.June, 2026)

process = GeneralizedBlackScholesProcess(
    x0=SimpleQuote(100.0),
    dividend_ts=FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc),
    risk_free_ts=FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc),
    black_vol_ts=BlackConstantVol(
        reference_date=ref, calendar=NullCalendar(), day_counter=dc, volatility=0.20
    ),
)

option = EuropeanOption(PlainVanillaPayoff(OptionType.Call, 100.0), EuropeanExercise(ref + 365))
option.set_pricing_engine(AnalyticEuropeanEngine(process))

print(f"NPV={option.npv():.4f}  delta={option.delta():.4f}  vega={option.vega():.4f}")
```

The public API is reached through submodule imports (e.g. `from pquantlib.time.date import Date`);
the top-level package exposes only `__version__`.

## What's covered

- **Time** — `Date`, calendars, day-count conventions, schedules, `Settings.evaluation_date`
- **Term structures** — flat & bootstrapped yield curves, Black/local volatility surfaces
- **Indexes & cashflows** — Ibor/overnight indexes, fixed & floating legs, CMS/capped-floored/digital coupons
- **Instruments** — bonds, vanilla & overnight swaps, vanilla/exotic options (barrier, double-barrier, Asian, basket, lookback, cliquet, digital), convertibles
- **Pricing engines** — analytic, binomial tree, Monte Carlo (incl. Longstaff-Schwartz American), finite differences
- **Models** — short-rate (Vasicek / Hull-White / CIR / G2++ / Black-Karasinski / Gaussian1d), Heston & Bates, SABR/ZABR, with Levenberg-Marquardt / Simplex calibration
- **Market models** — the full LMM/BGM stack (models, evolvers, products, callability, pathwise greeks)
- plus inflation, credit (CDS / default curves), and the `experimental/*` surface

See the [project repository](https://github.com/jlmoya/pquantlib) for the full migration history,
design docs, and carve-out documentation.

## Interactive showcase

A companion **Streamlit** app, [`pquantlib-showcase`](https://github.com/jlmoya/pquantlib/tree/main/pquantlib-showcase),
drives this library live — yield curves, bonds, swaps, vanilla options across all four
pricing engines, Greeks, exotics, and the Heston volatility smile + calibration, each
recomputed as you move a slider. From a clone of the repository:

```bash
uv run streamlit run pquantlib-showcase/app.py
```

It is a separate workspace member, so it is **not** bundled with the `pquantlib` wheel.

## License

BSD-3-Clause — same spirit as upstream QuantLib.
