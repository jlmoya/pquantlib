"""PQuantLib Showcase — landing page.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import pquantlib
import streamlit as st

from pquantlib_showcase import ui
from pquantlib_showcase.quant import options

ui.setup_page("Showcase", icon="📈")
ui.about_sidebar()

ui.hero(
    "PQuantLib in action",
    "A pure-Python port of QuantLib v1.42.1 — quantitative finance, live in your browser.",
    pills=[f"v{pquantlib.__version__}", "4048 tests", "functional 1:1 with C++ QuantLib", "numpy · scipy"],
)

st.markdown(
    """
PQuantLib is a faithful Python translation of the industry-standard **QuantLib** C++
library (v1.42.1). This app drives the library directly — every price, curve, Greek and
calibration you see below is computed **live** by PQuantLib when you move a slider.
"""
)

# A live "hello world": price an option right on the landing page.
st.subheader("Live proof of life")
c1, c2 = st.columns([1, 1.3])
with c1:
    spot = st.slider("Spot", 60.0, 140.0, 100.0, 1.0)
    strike = st.slider("Strike", 60.0, 140.0, 100.0, 1.0)
    vol = st.slider("Volatility", 0.05, 0.60, 0.20, 0.01)
    t = st.slider("Maturity (years)", 0.1, 3.0, 1.0, 0.1)
res = options.price_vanilla("Call", spot, strike, 0.05, 0.0, vol, t)
with c2:
    st.markdown("**European call**, priced four independent ways — they agree:")
    ui.metric_cards(
        [
            ("Closed-form", f"{res.analytic:.4f}", None),
            ("Binomial tree", f"{res.binomial:.4f}", f"{res.binomial - res.analytic:+.4f}"),
            ("Monte Carlo", f"{res.mc:.4f}", f"±{res.mc_error:.3f}"),
            ("Finite diff.", f"{res.fd:.4f}", f"{res.fd - res.analytic:+.4f}"),
        ]
    )
    st.caption(
        f"Δ {res.delta:.4f}  ·  Γ {res.gamma:.4f}  ·  ν {res.vega:.4f}  "
        f"·  Θ {res.theta:.4f}  ·  ρ {res.rho:.4f}"
    )

st.divider()
st.subheader("What you can explore")
left, mid, right = st.columns(3)
with left:
    st.markdown(
        "**📅 Time & Market Data**\nCalendars, day-count conventions, schedules.\n\n"
        "**📉 Yield Curves**\nFlat & bootstrapped curves; discount/zero/forward.\n\n"
        "**🏦 Bonds**\nClean/dirty price, yield, accrued, cashflows."
    )
with mid:
    st.markdown(
        "**🔁 Interest-Rate Swaps**\nVanilla & OIS: NPV, fair rate, DV01.\n\n"
        "**🎯 Vanilla Options**\nFour pricing engines side by side.\n\n"
        "**📐 Greeks & Risk**\nDelta/gamma/vega/theta profiles."
    )
with right:
    st.markdown(
        "**🚧 Exotic Options**\nBarrier, double-barrier, Asian.\n\n"
        "**🌊 Heston Model**\nStochastic-vol smile vs Black-Scholes.\n\n"
        "**🛠 Heston Calibration**\nFit 5 params to a market smile."
    )

ui.note("Open any page from the sidebar menu. Nothing is mocked — it is all PQuantLib.")
