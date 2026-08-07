// migration-harness/cpp/probes/v143_ts_optionletstripper/probe.cpp
//
// Pins the abstract intermediate `OptionletStripper`
// (ql/termstructures/volatility/optionlet/optionletstripper.{hpp,cpp} @ v1.43).
//
// `OptionletStripper` is the layer between `StrippedOptionletBase` and the two
// concrete strippers. It owns ALL the shared state and implements the whole
// read interface; subclasses supply only `performCalculations`. Its own
// closed-form contribution is:
//
//   * the optionlet-tenor / cap-floor-length walk, driven by
//     `optionletFrequency_ ? *optionletFrequency_ : iborIndex_->tenor()`
//     (optionletstripper.cpp:58-73) — NOT by the term-vol surface's own
//     option tenors, which only supply the upper bound;
//   * the sizing of every mutable vector (optionletstripper.cpp:75-84);
//   * the metadata forwards to the term-vol surface — dayCounter/calendar/
//     settlementDays/businessDayConvention (optionletstripper.cpp:140-154);
//   * the `ext::optional<Period>` optionletFrequency accessor
//     (optionletstripper.cpp:173-175). `nullopt` is NOT a zero Period, so it is
//     emitted as JSON `null` and pinned separately from a real Period.
//
// NOTE for anyone reading the reference: in v1.43 the base ctor does NOT
// compute the date grid; it only sizes it. `optionletDates_` /
// `optionletTimes_` / `optionletPaymentDates_` / `optionletAccrualPeriods_` /
// `atmOptionletRate_` are filled by `OptionletStripper1::performCalculations`
// (optionletstripper1.cpp:61-84) and copied wholesale from stripper1 by
// `OptionletStripper2::performCalculations` (optionletstripper2.cpp:60-68).
// Every base accessor for those calls `calculate()` first, so this probe
// NEVER reads them before a calculation has run.
//
// Dates are emitted BOTH as serial number and as ISO string so a date bug
// cannot hide behind a plausible serial.
//
// Emits ONE JSON object on stdout and nothing else.

#include <ql/indexes/ibor/euribor.hpp>
#include <ql/indexes/ibor/eonia.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>
#include <ql/termstructures/volatility/capfloor/capfloortermvolcurve.hpp>
#include <ql/termstructures/volatility/capfloor/capfloortermvolsurface.hpp>
#include <ql/termstructures/volatility/optionlet/optionletstripper1.hpp>
#include <ql/termstructures/volatility/optionlet/optionletstripper2.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/utilities/dataformatters.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace QuantLib;

namespace {

// --- tiny JSON emitters -----------------------------------------------------

std::string periodToString(const Period& p) {
    std::ostringstream os;
    os << p;
    return os.str();
}

std::string isoDate(const Date& d) {
    std::ostringstream os;
    os << io::iso_date(d);
    return os.str();
}

void emitDateVector(const std::string& name,
                    const std::vector<Date>& dates,
                    const std::string& indent) {
    std::cout << indent << "\"" << name << "_serial\": [";
    for (Size i = 0; i < dates.size(); ++i)
        std::cout << (i ? ", " : "") << dates[i].serialNumber();
    std::cout << "],\n";
    std::cout << indent << "\"" << name << "_iso\": [";
    for (Size i = 0; i < dates.size(); ++i)
        std::cout << (i ? ", " : "") << "\"" << isoDate(dates[i]) << "\"";
    std::cout << "],\n";
}

void emitRealVector(const std::string& name,
                    const std::vector<Real>& v,
                    const std::string& indent,
                    bool trailingComma = true) {
    std::cout << indent << "\"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << v[i];
    std::cout << "]" << (trailingComma ? ",\n" : "\n");
}

void emitPeriodVector(const std::string& name,
                      const std::vector<Period>& v,
                      const std::string& indent) {
    std::cout << indent << "\"" << name << "\": [";
    for (Size i = 0; i < v.size(); ++i)
        std::cout << (i ? ", " : "") << "\"" << periodToString(v[i]) << "\"";
    std::cout << "],\n";
}

void emitMatrix(const std::string& name,
                const std::vector<std::vector<Real> >& m,
                const std::string& indent,
                bool trailingComma = true) {
    std::cout << indent << "\"" << name << "\": [";
    for (Size i = 0; i < m.size(); ++i) {
        std::cout << (i ? ", " : "") << "[";
        for (Size j = 0; j < m[i].size(); ++j)
            std::cout << (j ? ", " : "") << m[i][j];
        std::cout << "]";
    }
    std::cout << "]" << (trailingComma ? ",\n" : "\n");
}

// --- the pinned read interface of OptionletStripper -------------------------

// `settlementDays()` forwards to CapFloorTermVolatilityStructure, which
// forwards to TermStructure::settlementDays(); that one QL_REQUIREs
// `settlementDays_ != Null<Natural>()` (termstructure.hpp:127-131), i.e. it
// THROWS for a fixed-reference-date surface. Both outcomes are pinned.
void emitBaseInterface(const OptionletStripper& s, const std::string& indent) {
    std::cout << indent << "\"optionlet_maturities\": " << s.optionletMaturities() << ",\n";
    emitPeriodVector("optionlet_fixing_tenors", s.optionletFixingTenors(), indent);

    // optionletstripper.cpp:173-175 — ext::optional<Period>: null vs a Period.
    std::cout << indent << "\"optionlet_frequency\": ";
    if (s.optionletFrequency())
        std::cout << "\"" << periodToString(*s.optionletFrequency()) << "\"";
    else
        std::cout << "null";
    std::cout << ",\n";

    // optionletstripper.cpp:140-154 — metadata forwards.
    std::cout << indent << "\"day_counter\": \"" << s.dayCounter().name() << "\",\n";
    std::cout << indent << "\"calendar\": \"" << s.calendar().name() << "\",\n";
    std::cout << indent << "\"business_day_convention\": " << int(s.businessDayConvention())
              << ",\n";
    std::cout << indent << "\"business_day_convention_name\": \"" << s.businessDayConvention()
              << "\",\n";
    try {
        Natural sd = s.settlementDays();
        std::cout << indent << "\"settlement_days\": " << sd << ",\n";
        std::cout << indent << "\"settlement_days_throws\": false,\n";
    } catch (const std::exception&) {
        std::cout << indent << "\"settlement_days\": null,\n";
        std::cout << indent << "\"settlement_days_throws\": true,\n";
    }

    std::cout << indent << "\"displacement\": " << s.displacement() << ",\n";
    std::cout << indent << "\"volatility_type\": " << int(s.volatilityType()) << ",\n";
    std::cout << indent << "\"volatility_type_name\": \"" << s.volatilityType() << "\",\n";

    // Every accessor below calls calculate() first (optionletstripper.cpp:87-137),
    // so these are populated, never uninitialised storage.
    emitDateVector("optionlet_fixing_dates", s.optionletFixingDates(), indent);
    emitRealVector("optionlet_fixing_times", s.optionletFixingTimes(), indent);
    emitDateVector("optionlet_payment_dates", s.optionletPaymentDates(), indent);
    emitRealVector("optionlet_accrual_periods", s.optionletAccrualPeriods(), indent);
    emitRealVector("atm_optionlet_rates", s.atmOptionletRates(), indent);

    std::vector<std::vector<Real> > strikes, vols;
    for (Size i = 0; i < s.optionletMaturities(); ++i) {
        strikes.push_back(s.optionletStrikes(i));
        vols.push_back(s.optionletVolatilities(i));
    }
    emitMatrix("optionlet_strikes", strikes, indent);
    emitMatrix("optionlet_volatilities", vols, indent, /*trailingComma=*/false);
}

const Date kEvalA(15, January, 2024);  // Monday, TARGET business day.
const Date kEvalHoliday(1, May, 2024); // Wednesday, TARGET Labour Day holiday.

Handle<YieldTermStructure> flatCurve(const Date& refDate, Rate r) {
    Handle<Quote> q(ext::make_shared<SimpleQuote>(r));
    return Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(refDate, q, Actual365Fixed()));
}

} // namespace

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // ========================================================================
    // A) Euribor3M, fixed-reference surface at a business-day eval date.
    //    Same market data as references/cluster/l8c.json so the two pins agree.
    // ========================================================================
    {
        Settings::instance().evaluationDate() = kEvalA;
        Handle<YieldTermStructure> curve = flatCurve(kEvalA, 0.03);
        auto index = ext::make_shared<Euribor>(Period(3, Months), curve);

        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                      Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        Matrix vols(4, 3, 0.18);
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            kEvalA, TARGET(), ModifiedFollowing, tenors, strikes, vols, Actual365Fixed());

        OptionletStripper1 stripper(surface, index);

        std::cout << "  \"euribor3m_flat18\": {\n";
        std::cout << "    \"eval_date_serial\": " << kEvalA.serialNumber() << ",\n";
        std::cout << "    \"eval_date_iso\": \"" << isoDate(kEvalA) << "\",\n";
        std::cout << "    \"reference_date_serial\": " << surface->referenceDate().serialNumber()
                  << ",\n";
        std::cout << "    \"reference_date_iso\": \"" << isoDate(surface->referenceDate())
                  << "\",\n";
        std::cout << "    \"n_strikes\": " << surface->strikes().size() << ",\n";
        std::cout << "    \"switch_strike\": " << stripper.switchStrike() << ",\n";
        emitBaseInterface(stripper, "    ");
        std::cout << "  },\n";
    }

    // ========================================================================
    // B) Euribor6M, Actual360 surface, evaluation date ON A TARGET HOLIDAY
    //    (1 May 2024, Labour Day), non-flat vols, wider tenor grid.
    // ========================================================================
    {
        Settings::instance().evaluationDate() = kEvalHoliday;
        Handle<YieldTermStructure> curve = flatCurve(kEvalHoliday, 0.025);
        auto index = ext::make_shared<Euribor>(Period(6, Months), curve);

        std::vector<Period> tenors = {Period(1, Years),  Period(2, Years), Period(3, Years),
                                      Period(5, Years),  Period(7, Years), Period(10, Years)};
        std::vector<Rate> strikes = {0.01, 0.03, 0.05};
        Matrix vols(6, 3);
        for (Size i = 0; i < 6; ++i)
            for (Size j = 0; j < 3; ++j)
                vols[i][j] = 0.22 - 0.006 * Real(i) + 0.011 * Real(j);
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            kEvalHoliday, TARGET(), Following, tenors, strikes, vols, Actual360());

        OptionletStripper1 stripper(surface, index);

        std::cout << "  \"euribor6m_holiday_evaldate\": {\n";
        std::cout << "    \"eval_date_serial\": " << kEvalHoliday.serialNumber() << ",\n";
        std::cout << "    \"eval_date_iso\": \"" << isoDate(kEvalHoliday) << "\",\n";
        std::cout << "    \"eval_date_is_target_holiday\": "
                  << (TARGET().isBusinessDay(kEvalHoliday) ? "false" : "true") << ",\n";
        std::cout << "    \"reference_date_serial\": " << surface->referenceDate().serialNumber()
                  << ",\n";
        std::cout << "    \"reference_date_iso\": \"" << isoDate(surface->referenceDate())
                  << "\",\n";
        std::cout << "    \"n_strikes\": " << surface->strikes().size() << ",\n";
        std::cout << "    \"switch_strike\": " << stripper.switchStrike() << ",\n";
        // The per-cap term vols the stripper reads back off the surface
        // (optionletstripper1.cpp:128-129). capFloorLengths_[i] = (i+2)*6M for
        // a 6M index with no explicit optionlet frequency. Emitted so that a
        // divergence in the STRIPPED vols can be attributed to either the
        // surface's (time, strike) interpolation or the stripping itself.
        {
            std::vector<std::vector<Real> > termVols;
            for (Size i = 0; i < 19; ++i) {
                Period len(Integer((i + 2) * 6), Months);
                std::vector<Real> row;
                for (Size j = 0; j < strikes.size(); ++j)
                    row.push_back(surface->volatility(len, strikes[j], true));
                termVols.push_back(row);
            }
            emitMatrix("cap_floor_term_vols", termVols, "    ");
        }
        emitBaseInterface(stripper, "    ");
        std::cout << "  },\n";
    }

    // ========================================================================
    // C) Euribor3M with an explicit optionletFrequency of 6M.
    //    The tenor walk now steps by 6M, so optionletTenors_/capFloorLengths_
    //    differ from (A) even though the index and the surface are unchanged.
    //    ext::optional<Period> is engaged here; (A) pins the nullopt branch.
    // ========================================================================
    {
        Settings::instance().evaluationDate() = kEvalA;
        Handle<YieldTermStructure> curve = flatCurve(kEvalA, 0.03);
        auto index = ext::make_shared<Euribor>(Period(3, Months), curve);

        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                      Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        Matrix vols(4, 3, 0.18);
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            kEvalA, TARGET(), ModifiedFollowing, tenors, strikes, vols, Actual365Fixed());

        // optionletstripper1.hpp — (surface, index, switchStrike, accuracy,
        // maxIter, discount, type, displacement, dontThrow, optionletFrequency)
        OptionletStripper1 stripper(surface, index, Null<Rate>(), 1.0e-6, 100,
                                    Handle<YieldTermStructure>(), ShiftedLognormal, 0.0, false,
                                    Period(6, Months));

        std::cout << "  \"euribor3m_freq6m\": {\n";
        std::cout << "    \"eval_date_serial\": " << kEvalA.serialNumber() << ",\n";
        std::cout << "    \"eval_date_iso\": \"" << isoDate(kEvalA) << "\",\n";
        std::cout << "    \"n_strikes\": " << surface->strikes().size() << ",\n";
        emitBaseInterface(stripper, "    ");
        std::cout << "  },\n";
    }

    // ========================================================================
    // D) Moving-reference-date surface (settlementDays = 2), eval date on a
    //    TARGET holiday. This is the ONLY configuration in which
    //    `settlementDays()` returns instead of throwing.
    // ========================================================================
    {
        Settings::instance().evaluationDate() = kEvalHoliday;
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(2, TARGET(),
                                          Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)),
                                          Actual365Fixed()));
        auto index = ext::make_shared<Euribor>(Period(3, Months), curve);

        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                      Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        Matrix vols(4, 3, 0.18);
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            Natural(2), TARGET(), ModifiedFollowing, tenors, strikes, vols, Actual365Fixed());

        OptionletStripper1 stripper(surface, index);

        std::cout << "  \"euribor3m_moving_settlement\": {\n";
        std::cout << "    \"eval_date_serial\": " << kEvalHoliday.serialNumber() << ",\n";
        std::cout << "    \"eval_date_iso\": \"" << isoDate(kEvalHoliday) << "\",\n";
        std::cout << "    \"reference_date_serial\": " << surface->referenceDate().serialNumber()
                  << ",\n";
        std::cout << "    \"reference_date_iso\": \"" << isoDate(surface->referenceDate())
                  << "\",\n";
        std::cout << "    \"n_strikes\": " << surface->strikes().size() << ",\n";
        std::cout << "    \"switch_strike\": " << stripper.switchStrike() << ",\n";
        emitBaseInterface(stripper, "    ");
        std::cout << "  },\n";
    }

    // ========================================================================
    // E) Normal volatility type with displacement 0 (the only legal Normal
    //    combination — optionletstripper.cpp:43-46 rejects a non-null
    //    displacement under Normal).
    // ========================================================================
    {
        Settings::instance().evaluationDate() = kEvalA;
        Handle<YieldTermStructure> curve = flatCurve(kEvalA, 0.03);
        auto index = ext::make_shared<Euribor>(Period(3, Months), curve);

        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                      Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        Matrix vols(4, 3, 0.006); // normal vols are absolute, ~60bp
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            kEvalA, TARGET(), ModifiedFollowing, tenors, strikes, vols, Actual365Fixed());

        OptionletStripper1 stripper(surface, index, Null<Rate>(), 1.0e-6, 100,
                                    Handle<YieldTermStructure>(), Normal, 0.0);

        std::cout << "  \"euribor3m_normal\": {\n";
        std::cout << "    \"eval_date_serial\": " << kEvalA.serialNumber() << ",\n";
        std::cout << "    \"n_strikes\": " << surface->strikes().size() << ",\n";
        emitBaseInterface(stripper, "    ");
        std::cout << "  },\n";
    }

    // ========================================================================
    // F) Shifted-lognormal with a non-zero displacement, to pin that the
    //    (displacement, volatilityType) pair rides through the base intact.
    // ========================================================================
    {
        Settings::instance().evaluationDate() = kEvalA;
        Handle<YieldTermStructure> curve = flatCurve(kEvalA, 0.03);
        auto index = ext::make_shared<Euribor>(Period(3, Months), curve);

        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                      Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        Matrix vols(4, 3, 0.18);
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            kEvalA, TARGET(), ModifiedFollowing, tenors, strikes, vols, Actual365Fixed());

        OptionletStripper1 stripper(surface, index, Null<Rate>(), 1.0e-6, 100,
                                    Handle<YieldTermStructure>(), ShiftedLognormal, 0.01);

        std::cout << "  \"euribor3m_shifted\": {\n";
        std::cout << "    \"eval_date_serial\": " << kEvalA.serialNumber() << ",\n";
        std::cout << "    \"n_strikes\": " << surface->strikes().size() << ",\n";
        emitBaseInterface(stripper, "    ");
        std::cout << "  },\n";
    }

    // ========================================================================
    // G) OptionletStripper2 on top of a stripper1 + an ATM term-vol curve.
    //    Its base is re-built from stripper1's own surface/index/type/
    //    displacement/frequency (optionletstripper2.cpp:39-44), so the tenor
    //    grid must come out identical to a same-input stripper1, while the
    //    strike/vol rows are augmented with one ATM column per (expiry, row).
    //
    //    The surface MUST be moving-reference-date here. OptionletStripper2::
    //    performCalculations builds a `StrippedOptionletAdapter(stripper1_)`
    //    (optionletstripper2.cpp:94) whose ctor reads `s->settlementDays()`
    //    (strippedoptionletadapter.cpp:35-39); on a fixed-reference-date
    //    surface that QL_REQUIRE fails and OptionletStripper2 cannot be
    //    calculated AT ALL. The fixed-reference variant is pinned as a throw
    //    below so a port cannot quietly make it "work".
    // ========================================================================
    {
        Settings::instance().evaluationDate() = kEvalA;
        Handle<YieldTermStructure> curve(
            ext::make_shared<FlatForward>(2, TARGET(),
                                          Handle<Quote>(ext::make_shared<SimpleQuote>(0.03)),
                                          Actual365Fixed()));
        auto index = ext::make_shared<Euribor>(Period(3, Months), curve);

        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                      Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        Matrix vols(4, 3, 0.18);
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            Natural(2), TARGET(), ModifiedFollowing, tenors, strikes, vols, Actual365Fixed());

        auto stripper1 = ext::make_shared<OptionletStripper1>(surface, index);

        std::vector<Period> atmTenors = {Period(1, Years), Period(2, Years), Period(3, Years)};
        std::vector<Volatility> atmVols = {0.18, 0.18, 0.18};
        Handle<CapFloorTermVolCurve> atmCurve(ext::make_shared<CapFloorTermVolCurve>(
            Natural(2), TARGET(), ModifiedFollowing, atmTenors, atmVols, Actual365Fixed()));

        OptionletStripper2 stripper2(stripper1, atmCurve);

        std::cout << "  \"stripper2_euribor3m_flat18\": {\n";
        std::cout << "    \"eval_date_serial\": " << kEvalA.serialNumber() << ",\n";
        std::cout << "    \"eval_date_iso\": \"" << isoDate(kEvalA) << "\",\n";
        std::cout << "    \"surface_settlement_days\": 2,\n";
        std::cout << "    \"reference_date_serial\": " << surface->referenceDate().serialNumber()
                  << ",\n";
        std::cout << "    \"reference_date_iso\": \"" << isoDate(surface->referenceDate())
                  << "\",\n";
        std::cout << "    \"atm_option_tenors\": [\"1Y\", \"2Y\", \"3Y\"],\n";
        std::cout << "    \"atm_curve_vol\": 0.18,\n";
        emitRealVector("atm_cap_floor_strikes", stripper2.atmCapFloorStrikes(), "    ");
        emitRealVector("atm_cap_floor_prices", stripper2.atmCapFloorPrices(), "    ");
        emitRealVector("spreads_vol", stripper2.spreadsVol(), "    ");
        std::cout << "    \"row_widths\": [";
        for (Size i = 0; i < stripper2.optionletMaturities(); ++i)
            std::cout << (i ? ", " : "") << stripper2.optionletStrikes(i).size();
        std::cout << "],\n";
        // Same inputs, fixed reference date -> C++ cannot calculate at all.
        bool fixedReferenceThrows = false;
        try {
            Handle<YieldTermStructure> fixedCurve = flatCurve(kEvalA, 0.03);
            auto fixedIndex = ext::make_shared<Euribor>(Period(3, Months), fixedCurve);
            auto fixedSurface = ext::make_shared<CapFloorTermVolSurface>(
                kEvalA, TARGET(), ModifiedFollowing, tenors, strikes, vols, Actual365Fixed());
            auto fixedStripper1 = ext::make_shared<OptionletStripper1>(fixedSurface, fixedIndex);
            Handle<CapFloorTermVolCurve> fixedAtm(ext::make_shared<CapFloorTermVolCurve>(
                kEvalA, TARGET(), ModifiedFollowing, atmTenors, atmVols, Actual365Fixed()));
            OptionletStripper2 bad(fixedStripper1, fixedAtm);
            (void)bad.atmCapFloorStrikes();
        } catch (const std::exception&) {
            fixedReferenceThrows = true;
        }
        std::cout << "    \"fixed_reference_date_surface_throws\": "
                  << (fixedReferenceThrows ? "true" : "false") << ",\n";
        emitBaseInterface(stripper2, "    ");
        std::cout << "  },\n";
    }

    // ========================================================================
    // H) Guard rails enforced by the base constructor
    //    (optionletstripper.cpp:43-51, 64-66).
    // ========================================================================
    {
        Settings::instance().evaluationDate() = kEvalA;
        Handle<YieldTermStructure> curve = flatCurve(kEvalA, 0.03);
        auto index = ext::make_shared<Euribor>(Period(3, Months), curve);
        auto onIndex = ext::make_shared<Eonia>(curve);

        std::vector<Period> tenors = {Period(1, Years), Period(2, Years),
                                      Period(3, Years), Period(5, Years)};
        std::vector<Rate> strikes = {0.02, 0.04, 0.06};
        Matrix vols(4, 3, 0.18);
        auto surface = ext::make_shared<CapFloorTermVolSurface>(
            kEvalA, TARGET(), ModifiedFollowing, tenors, strikes, vols, Actual365Fixed());

        // Normal + non-zero displacement -> rejected (optionletstripper.cpp:43-46).
        bool normalWithDisplacementThrows = false;
        try {
            OptionletStripper1 bad(surface, index, Null<Rate>(), 1.0e-6, 100,
                                   Handle<YieldTermStructure>(), Normal, 0.01);
            (void)bad.optionletMaturities();
        } catch (const std::exception&) {
            normalWithDisplacementThrows = true;
        }

        // OvernightIndex without an optionletFrequency -> rejected
        // (optionletstripper.cpp:48-51).
        bool overnightWithoutFrequencyThrows = false;
        try {
            OptionletStripper1 bad(surface, onIndex);
            (void)bad.optionletMaturities();
        } catch (const std::exception&) {
            overnightWithoutFrequencyThrows = true;
        }

        // OvernightIndex WITH an optionletFrequency -> accepted; the tenor walk
        // steps by the frequency, not by the index's 1D tenor.
        bool overnightWithFrequencyThrows = false;
        Size overnightMaturities = 0;
        std::vector<Period> overnightTenors;
        try {
            OptionletStripper1 ok(surface, onIndex, Null<Rate>(), 1.0e-6, 100,
                                  Handle<YieldTermStructure>(), ShiftedLognormal, 0.0, false,
                                  Period(1, Years));
            overnightMaturities = ok.optionletMaturities();
            overnightTenors = ok.optionletFixingTenors();
        } catch (const std::exception&) {
            overnightWithFrequencyThrows = true;
        }

        // Surface too short for even one cap-floor length
        // (optionletstripper.cpp:64-66): a 1Y-max surface cannot host a 3M
        // index, whose first capFloorLength is 6M... but a 1D index whose
        // first capFloorLength is 2Y under a 1Y frequency can't fit either.
        bool tooShortSurfaceThrows = false;
        try {
            std::vector<Period> shortTenors = {Period(6, Months), Period(1, Years)};
            Matrix shortVols(2, 3, 0.18);
            auto shortSurface = ext::make_shared<CapFloorTermVolSurface>(
                kEvalA, TARGET(), ModifiedFollowing, shortTenors, strikes, shortVols,
                Actual365Fixed());
            OptionletStripper1 bad(shortSurface, index, Null<Rate>(), 1.0e-6, 100,
                                   Handle<YieldTermStructure>(), ShiftedLognormal, 0.0, false,
                                   Period(1, Years));
            (void)bad.optionletMaturities();
        } catch (const std::exception&) {
            tooShortSurfaceThrows = true;
        }

        std::cout << "  \"guards\": {\n";
        std::cout << "    \"normal_with_displacement_throws\": "
                  << (normalWithDisplacementThrows ? "true" : "false") << ",\n";
        std::cout << "    \"overnight_without_frequency_throws\": "
                  << (overnightWithoutFrequencyThrows ? "true" : "false") << ",\n";
        std::cout << "    \"overnight_with_frequency_throws\": "
                  << (overnightWithFrequencyThrows ? "true" : "false") << ",\n";
        std::cout << "    \"overnight_with_frequency_maturities\": " << overnightMaturities
                  << ",\n";
        emitPeriodVector("overnight_with_frequency_tenors", overnightTenors, "    ");
        std::cout << "    \"too_short_surface_throws\": "
                  << (tooShortSurfaceThrows ? "true" : "false") << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
