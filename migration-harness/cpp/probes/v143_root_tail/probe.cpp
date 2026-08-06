// migration-harness/cpp/probes/v143_root_tail/probe.cpp
//
// Reference values for the root-level (ql/*.hpp) coverage tail:
//
//   * ql/prices.{hpp,cpp}          — midEquivalent, midSafe, IntervalPrice
//   * ql/event.{hpp,cpp}           — Event::hasOccurred
//   * ql/cashflow.{hpp,cpp}        — CashFlow::hasOccurred, tradingExCoupon
//   * ql/settings.{hpp,cpp}        — Settings + SavedSettings restore semantics
//   * ql/rebatedexercise.{hpp,cpp} — RebatedExercise
//
// What is pinned and why:
//
//   * midEquivalent over the FULL availability lattice. The C++ branch
//     structure short-circuits: a valid bid returns the bid even when a valid
//     `last` and `close` exist, and an invalid bid does NOT fall through to
//     `last` when the ask is valid. A port that reordered the branches would
//     still reproduce the "all four available" case, so every one of the 16
//     available/unavailable combinations is emitted, plus the non-positive
//     variants (0.0 and -1.0) which C++ treats as unavailable but which are
//     NOT Null<Real>() — a port that only tested `is None` would pass the
//     16-case lattice and fail here.
//
//   * IntervalPrice::value / setValue for each of the four Type members
//     separately, from a bar whose four fields are pairwise distinct, so a
//     switch that mapped High→Low is caught. The ctor argument ORDER is
//     (open, close, high, low) — deliberately not conventional OHLC — and the
//     probe passes four distinct values positionally so a port that reordered
//     them is caught.
//
//   * extractValues / extractComponent on a series whose dates are inserted
//     OUT of chronological order, because both C++ helpers iterate a
//     std::map (key-ordered) and a port backed by an insertion-ordered dict
//     would otherwise silently return a differently-ordered vector.
//
//   * Event::hasOccurred as a truth table over
//     (refDate < / == / > eventDate) x (includeRefDate unset / true / false)
//     x (evaluationDate used as fallback). The `refDate == Date()` rows are
//     the ones that pin the Settings fallback, which is exactly the path a
//     port is tempted to shortcut to `return false`.
//
//   * CashFlow::hasOccurred separately from Event::hasOccurred, because the
//     override adds the includeTodaysCashFlows() override and only applies it
//     when refDate is null or equal to the evaluation date.
//
//   * tradingExCoupon with an ex-coupon date, at refDate null (Settings
//     fallback) and at explicit dates either side.
//
//   * SavedSettings: the state observed AFTER the guard goes out of scope,
//     including the fact that the C++ ctor snapshots the RESOLVED evaluation
//     date (DateProxy::operator Date() maps an unset date to todaysDate()), so
//     an initially-unset date comes back ANCHORED, not unset.
//
//   * RebatedExercise: rebate(i) for the broadcast (scalar) ctor and for the
//     vector ctor, and rebatePaymentDate(i) with a NON-ZERO settlement lag, a
//     real calendar (TARGET) and a non-default convention (ModifiedPreceding)
//     applied to exercise dates that fall on/next to TARGET holidays — with a
//     zero-lag row alongside, so an accepted-then-discarded lag is visible in
//     the reference itself.
//
// Emits JSON on stdout; redirect to references/v143/root/tail.json.

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ql/cashflow.hpp>
#include <ql/cashflows/simplecashflow.hpp>
#include <ql/errors.hpp>
#include <ql/event.hpp>
#include <ql/exercise.hpp>
#include <ql/prices.hpp>
#include <ql/rebatedexercise.hpp>
#include <ql/settings.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/date.hpp>
#include <ql/timeseries.hpp>

using namespace QuantLib;

namespace {

    std::ostream& out = std::cout;

    void num(Real x) {
        if (x == Null<Real>())
            out << "null";
        else
            out << std::setprecision(17) << x;
    }

    void optnum(const ext::optional<Real>& x) {
        if (!x)
            out << "null";
        else
            num(*x);
    }

    std::string boolstr(bool b) { return b ? "true" : "false"; }

    // --- mid-price lattice --------------------------------------------------

    void emitMidEquivalent() {
        // Each input is drawn from: Null (unavailable), 0.0 and -1.0
        // (present but non-positive -> C++ treats as unavailable), and a
        // distinct positive value per slot so the returned number identifies
        // WHICH slot won.
        const Real N = Null<Real>();
        const std::vector<Real> bids{N, 0.0, -1.0, 100.0};
        const std::vector<Real> asks{N, 0.0, -1.0, 200.0};
        const std::vector<Real> lasts{N, 0.0, -1.0, 300.0};
        const std::vector<Real> closes{N, 0.0, -1.0, 400.0};

        out << "  \"mid_equivalent\": [\n";
        bool first = true;
        for (Real b : bids)
            for (Real a : asks)
                for (Real l : lasts)
                    for (Real c : closes) {
                        if (!first)
                            out << ",\n";
                        first = false;
                        out << "    {\"bid\": ";
                        num(b);
                        out << ", \"ask\": ";
                        num(a);
                        out << ", \"last\": ";
                        num(l);
                        out << ", \"close\": ";
                        num(c);
                        out << ", ";
                        try {
                            Real v = midEquivalent(b, a, l, c);
                            out << "\"value\": ";
                            num(v);
                            out << ", \"throws\": false}";
                        } catch (const std::exception&) {
                            out << "\"value\": null, \"throws\": true}";
                        }
                    }
        out << "\n  ],\n";
    }

    void emitMidSafe() {
        const Real N = Null<Real>();
        const std::vector<Real> bids{N, 0.0, -1.0, 100.0};
        const std::vector<Real> asks{N, 0.0, -1.0, 200.0};
        out << "  \"mid_safe\": [\n";
        bool first = true;
        for (Real b : bids)
            for (Real a : asks) {
                if (!first)
                    out << ",\n";
                first = false;
                out << "    {\"bid\": ";
                num(b);
                out << ", \"ask\": ";
                num(a);
                out << ", ";
                try {
                    Real v = midSafe(b, a);
                    out << "\"value\": ";
                    num(v);
                    out << ", \"throws\": false}";
                } catch (const std::exception&) {
                    out << "\"value\": null, \"throws\": true}";
                }
            }
        out << "\n  ],\n";
    }

    // --- IntervalPrice ------------------------------------------------------

    void emitIntervalPrice() {
        out << "  \"interval_price\": {\n";

        IntervalPrice def;
        out << "    \"default_ctor\": {\"open\": ";
        num(def.open());
        out << ", \"close\": ";
        num(def.close());
        out << ", \"high\": ";
        num(def.high());
        out << ", \"low\": ";
        num(def.low());
        out << "},\n";

        // ctor argument order is (open, close, high, low) — four distinct
        // values so a reordering port is caught.
        IntervalPrice p(11.0, 22.0, 33.0, 44.0);
        out << "    \"ctor_open_close_high_low\": {\"open\": ";
        num(p.open());
        out << ", \"close\": ";
        num(p.close());
        out << ", \"high\": ";
        num(p.high());
        out << ", \"low\": ";
        num(p.low());
        out << "},\n";

        const IntervalPrice::Type types[4] = {IntervalPrice::Open, IntervalPrice::Close,
                                              IntervalPrice::High, IntervalPrice::Low};
        const char* typeNames[4] = {"Open", "Close", "High", "Low"};

        out << "    \"type_values\": {\"Open\": " << static_cast<int>(IntervalPrice::Open)
            << ", \"Close\": " << static_cast<int>(IntervalPrice::Close)
            << ", \"High\": " << static_cast<int>(IntervalPrice::High)
            << ", \"Low\": " << static_cast<int>(IntervalPrice::Low) << "},\n";

        out << "    \"value_by_type\": {";
        for (int i = 0; i < 4; ++i) {
            if (i)
                out << ", ";
            out << "\"" << typeNames[i] << "\": ";
            num(p.value(types[i]));
        }
        out << "},\n";

        // setValue on each slot independently, starting from a fresh bar each
        // time so cross-talk between slots is visible.
        out << "    \"set_value\": [";
        for (int i = 0; i < 4; ++i) {
            if (i)
                out << ", ";
            IntervalPrice q(11.0, 22.0, 33.0, 44.0);
            q.setValue(99.0, types[i]);
            out << "{\"set\": \"" << typeNames[i] << "\", \"open\": ";
            num(q.open());
            out << ", \"close\": ";
            num(q.close());
            out << ", \"high\": ";
            num(q.high());
            out << ", \"low\": ";
            num(q.low());
            out << "}";
        }
        out << "],\n";

        IntervalPrice r(1.0, 2.0, 3.0, 4.0);
        r.setValues(51.0, 52.0, 53.0, 54.0);
        out << "    \"set_values\": {\"open\": ";
        num(r.open());
        out << ", \"close\": ";
        num(r.close());
        out << ", \"high\": ";
        num(r.high());
        out << ", \"low\": ";
        num(r.low());
        out << "},\n";

        // Dates deliberately NOT in chronological order.
        std::vector<Date> dates{Date(15, March, 2025), Date(3, January, 2025),
                                Date(28, February, 2025)};
        std::vector<Real> opens{101.0, 201.0, 301.0};
        std::vector<Real> closes{102.0, 202.0, 302.0};
        std::vector<Real> highs{103.0, 203.0, 303.0};
        std::vector<Real> lows{104.0, 204.0, 304.0};

        TimeSeries<IntervalPrice> ts =
            IntervalPrice::makeSeries(dates, opens, closes, highs, lows);

        out << "    \"series_input_dates\": [";
        for (Size i = 0; i < dates.size(); ++i) {
            if (i)
                out << ", ";
            out << dates[i].serialNumber();
        }
        out << "],\n";

        std::vector<Date> sd = ts.dates();
        out << "    \"series_dates\": [";
        for (Size i = 0; i < sd.size(); ++i) {
            if (i)
                out << ", ";
            out << sd[i].serialNumber();
        }
        out << "],\n";

        out << "    \"extract_values\": {";
        for (int i = 0; i < 4; ++i) {
            if (i)
                out << ", ";
            out << "\"" << typeNames[i] << "\": [";
            std::vector<Real> v = IntervalPrice::extractValues(ts, types[i]);
            for (Size j = 0; j < v.size(); ++j) {
                if (j)
                    out << ", ";
                num(v[j]);
            }
            out << "]";
        }
        out << "},\n";

        out << "    \"extract_component_close\": [";
        TimeSeries<Real> comp = IntervalPrice::extractComponent(ts, IntervalPrice::Close);
        std::vector<Date> cd = comp.dates();
        std::vector<Real> cv = comp.values();
        for (Size j = 0; j < cd.size(); ++j) {
            if (j)
                out << ", ";
            out << "{\"date\": " << cd[j].serialNumber() << ", \"value\": ";
            num(cv[j]);
            out << "}";
        }
        out << "],\n";

        // size-mismatch guard
        out << "    \"make_series_size_mismatch_throws\": ";
        try {
            std::vector<Real> shortVec{1.0};
            IntervalPrice::makeSeries(dates, shortVec, closes, highs, lows);
            out << "false";
        } catch (const std::exception&) {
            out << "true";
        }
        out << "\n  },\n";
    }

    // --- Event / CashFlow ---------------------------------------------------

    const Date EVAL_DATE(15, June, 2025);
    const Date CF_DATE(20, June, 2025);

    void emitHasOccurred() {
        Settings::instance().evaluationDate() = EVAL_DATE;

        // refDates spanning strictly-before / equal / strictly-after the
        // cash-flow date, plus the null date (Settings fallback) and the
        // evaluation date itself (which triggers the includeTodaysCashFlows
        // override inside the CashFlow specialisation).
        struct Row {
            const char* name;
            Date refDate;
        };
        const Row refs[] = {{"null", Date()},
                            {"eval_date", EVAL_DATE},
                            {"before", Date(19, June, 2025)},
                            {"equal", CF_DATE},
                            {"after", Date(21, June, 2025)}};

        SimpleCashFlow cf(1000.0, CF_DATE);
        const Event& ev = cf;

        out << "  \"has_occurred\": {\n";
        out << "    \"eval_date_serial\": " << EVAL_DATE.serialNumber() << ",\n";
        out << "    \"cf_date_serial\": " << CF_DATE.serialNumber() << ",\n";
        out << "    \"rows\": [\n";

        bool first = true;
        // include_reference_date_events: both settings values.
        for (bool irde : {false, true}) {
            Settings::instance().includeReferenceDateEvents() = irde;
            // includeTodaysCashFlows: unset / true / false.
            for (int itcf = -1; itcf <= 1; ++itcf) {
                if (itcf < 0)
                    Settings::instance().includeTodaysCashFlows() = ext::nullopt;
                else
                    Settings::instance().includeTodaysCashFlows() = (itcf == 1);
                for (const Row& r : refs) {
                    // includeRefDate argument: unset / true / false.
                    for (int arg = -1; arg <= 1; ++arg) {
                        ext::optional<bool> a;
                        if (arg >= 0)
                            a = (arg == 1);
                        if (!first)
                            out << ",\n";
                        first = false;
                        out << "      {\"settings_include_ref_date_events\": " << boolstr(irde)
                            << ", \"settings_include_todays_cash_flows\": "
                            << (itcf < 0 ? "null" : boolstr(itcf == 1))
                            << ", \"ref\": \"" << r.name << "\""
                            << ", \"arg_include_ref_date\": "
                            << (arg < 0 ? "null" : boolstr(arg == 1))
                            << ", \"event\": " << boolstr(ev.Event::hasOccurred(r.refDate, a))
                            << ", \"cashflow\": " << boolstr(cf.hasOccurred(r.refDate, a)) << "}";
                    }
                }
            }
        }
        out << "\n    ]\n  },\n";

        // restore for the next section
        Settings::instance().includeReferenceDateEvents() = false;
        Settings::instance().includeTodaysCashFlows() = ext::nullopt;
    }

    // A CashFlow with a real ex-coupon date; SimpleCashFlow has none.
    class ExCouponFlow : public CashFlow {
      public:
        ExCouponFlow(Real amount, Date date, Date exCoupon)
        : amount_(amount), date_(date), exCoupon_(exCoupon) {}
        Date date() const override { return date_; }
        Real amount() const override { return amount_; }
        Date exCouponDate() const override { return exCoupon_; }

      private:
        Real amount_;
        Date date_, exCoupon_;
    };

    void emitTradingExCoupon() {
        Settings::instance().evaluationDate() = EVAL_DATE;
        const Date EX_DATE(14, June, 2025);  // one day before the evaluation date

        ExCouponFlow withEx(1000.0, CF_DATE, EX_DATE);
        SimpleCashFlow withoutEx(1000.0, CF_DATE);

        out << "  \"trading_ex_coupon\": {\n";
        out << "    \"ex_date_serial\": " << EX_DATE.serialNumber() << ",\n";
        out << "    \"no_ex_coupon_date_null_ref\": " << boolstr(withoutEx.tradingExCoupon(Date()))
            << ",\n";
        out << "    \"null_ref\": " << boolstr(withEx.tradingExCoupon(Date())) << ",\n";
        out << "    \"before_ex\": " << boolstr(withEx.tradingExCoupon(Date(13, June, 2025)))
            << ",\n";
        out << "    \"on_ex\": " << boolstr(withEx.tradingExCoupon(EX_DATE)) << ",\n";
        out << "    \"after_ex\": " << boolstr(withEx.tradingExCoupon(Date(16, June, 2025)))
            << "\n  },\n";
    }

    // --- SavedSettings ------------------------------------------------------

    void emitSavedSettings() {
        out << "  \"saved_settings\": {\n";

        // Case 1: evaluation date pinned on entry; every field mutated inside.
        Settings::instance().evaluationDate() = EVAL_DATE;
        Settings::instance().includeReferenceDateEvents() = false;
        Settings::instance().includeTodaysCashFlows() = ext::nullopt;
        Settings::instance().enforcesTodaysHistoricFixings() = false;
        {
            SavedSettings guard;
            Settings::instance().evaluationDate() = Date(1, December, 2030);
            Settings::instance().includeReferenceDateEvents() = true;
            Settings::instance().includeTodaysCashFlows() = true;
            Settings::instance().enforcesTodaysHistoricFixings() = true;
        }
        out << "    \"pinned_restore\": {\"evaluation_date\": "
            << Date(Settings::instance().evaluationDate()).serialNumber()
            << ", \"include_reference_date_events\": "
            << boolstr(Settings::instance().includeReferenceDateEvents())
            << ", \"include_todays_cash_flows\": "
            << (Settings::instance().includeTodaysCashFlows()
                    ? boolstr(*Settings::instance().includeTodaysCashFlows())
                    : "null")
            << ", \"enforces_todays_historic_fixings\": "
            << boolstr(Settings::instance().enforcesTodaysHistoricFixings()) << "},\n";

        // Case 2: includeTodaysCashFlows set on entry, cleared inside — proves
        // the optional's UNSET state is restored, not coerced to false.
        Settings::instance().includeTodaysCashFlows() = false;
        {
            SavedSettings guard;
            Settings::instance().includeTodaysCashFlows() = ext::nullopt;
        }
        out << "    \"optional_false_restore\": "
            << (Settings::instance().includeTodaysCashFlows()
                    ? boolstr(*Settings::instance().includeTodaysCashFlows())
                    : "null")
            << ",\n";

        Settings::instance().includeTodaysCashFlows() = ext::nullopt;
        {
            SavedSettings guard;
            Settings::instance().includeTodaysCashFlows() = true;
        }
        out << "    \"optional_unset_restore\": "
            << (Settings::instance().includeTodaysCashFlows()
                    ? boolstr(*Settings::instance().includeTodaysCashFlows())
                    : "null")
            << ",\n";

        // Case 3: evaluation date UNSET on entry. The C++ ctor snapshots the
        // RESOLVED date (DateProxy::operator Date() -> todaysDate()), so it
        // comes back ANCHORED rather than unset. Pinned as a boolean because
        // "today" is not reproducible in a reference file.
        Settings::instance().resetEvaluationDate();
        const Date todayAtEntry = Date::todaysDate();
        {
            SavedSettings guard;
            Settings::instance().evaluationDate() = Date(1, December, 2030);
        }
        out << "    \"unset_entry_restores_to_today\": "
            << boolstr(Date(Settings::instance().evaluationDate()) == todayAtEntry) << "\n";
        out << "  },\n";

        // leave the global state clean for the next section
        Settings::instance().evaluationDate() = EVAL_DATE;
        Settings::instance().includeReferenceDateEvents() = false;
        Settings::instance().includeTodaysCashFlows() = ext::nullopt;
        Settings::instance().enforcesTodaysHistoricFixings() = false;
    }

    // --- RebatedExercise ----------------------------------------------------

    void emitRebatedExercise() {
        TARGET cal;
        // 2025-04-18 is Good Friday (TARGET holiday), 2025-04-21 Easter Monday,
        // 2025-05-01 Labour Day, 2025-12-25/26 Christmas — exercise dates are
        // chosen so the settlement roll actually has to move.
        std::vector<Date> berDates{Date(17, April, 2025), Date(30, April, 2025),
                                   Date(24, December, 2025)};
        BermudanExercise ber(berDates);
        EuropeanExercise eur(Date(30, April, 2025));
        AmericanExercise amer(Date(1, April, 2025), Date(30, April, 2025));

        out << "  \"rebated_exercise\": {\n";
        out << "    \"bermudan_date_serials\": [";
        for (Size i = 0; i < berDates.size(); ++i) {
            if (i)
                out << ", ";
            out << ber.dates()[i].serialNumber();
        }
        out << "],\n";

        // Scalar ctor broadcast + non-zero settlement lag + non-default
        // convention.
        RebatedExercise scalarRe(ber, -12.5, 3, cal, ModifiedPreceding);
        out << "    \"scalar\": {\"rebates\": [";
        for (Size i = 0; i < berDates.size(); ++i) {
            if (i)
                out << ", ";
            num(scalarRe.rebate(i));
        }
        out << "], \"payment_dates\": [";
        for (Size i = 0; i < berDates.size(); ++i) {
            if (i)
                out << ", ";
            out << scalarRe.rebatePaymentDate(i).serialNumber();
        }
        out << "], \"type\": " << static_cast<int>(scalarRe.type()) << "},\n";

        // Same exercise, ZERO lag — the delta against "scalar" is what makes an
        // accepted-then-discarded settlement-days argument visible.
        RebatedExercise zeroLag(ber, -12.5, 0, cal, ModifiedPreceding);
        out << "    \"scalar_zero_lag_payment_dates\": [";
        for (Size i = 0; i < berDates.size(); ++i) {
            if (i)
                out << ", ";
            out << zeroLag.rebatePaymentDate(i).serialNumber();
        }
        out << "],\n";

        // Same exercise + lag, NullCalendar/Following — the delta against
        // "scalar" makes an accepted-then-discarded calendar/convention visible.
        RebatedExercise nullCal(ber, -12.5, 3, NullCalendar(), Following);
        out << "    \"scalar_null_calendar_payment_dates\": [";
        for (Size i = 0; i < berDates.size(); ++i) {
            if (i)
                out << ", ";
            out << nullCal.rebatePaymentDate(i).serialNumber();
        }
        out << "],\n";

        std::vector<Real> vec{1.5, -2.5, 3.5};
        RebatedExercise vecRe(ber, vec, 2, cal, Preceding);
        out << "    \"vector\": {\"rebates\": [";
        for (Size i = 0; i < vec.size(); ++i) {
            if (i)
                out << ", ";
            num(vecRe.rebate(i));
        }
        out << "], \"payment_dates\": [";
        for (Size i = 0; i < vec.size(); ++i) {
            if (i)
                out << ", ";
            out << vecRe.rebatePaymentDate(i).serialNumber();
        }
        out << "]},\n";

        RebatedExercise eurRe(eur, 7.25, 1, cal, Following);
        out << "    \"european\": {\"n_rebates\": " << eurRe.rebates().size() << ", \"rebate0\": ";
        num(eurRe.rebate(0));
        out << ", \"payment_date0\": " << eurRe.rebatePaymentDate(0).serialNumber()
            << ", \"type\": " << static_cast<int>(eurRe.type()) << "},\n";

        // Guard rails.
        out << "    \"vector_on_european_throws\": ";
        try {
            RebatedExercise bad(eur, std::vector<Real>{1.0}, 0, cal, Following);
            out << "false";
        } catch (const std::exception&) {
            out << "true";
        }
        out << ",\n";

        out << "    \"vector_wrong_size_throws\": ";
        try {
            RebatedExercise bad(ber, std::vector<Real>{1.0, 2.0}, 0, cal, Following);
            out << "false";
        } catch (const std::exception&) {
            out << "true";
        }
        out << ",\n";

        out << "    \"american_payment_date_throws\": ";
        try {
            RebatedExercise amerRe(amer, 1.0, 0, cal, Following);
            amerRe.rebatePaymentDate(0);
            out << "false";
        } catch (const std::exception&) {
            out << "true";
        }
        out << ",\n";

        out << "    \"rebate_out_of_range_throws\": ";
        try {
            scalarRe.rebate(berDates.size());
            out << "false";
        } catch (const std::exception&) {
            out << "true";
        }
        out << "\n  }\n";
    }

}

int main() {
    try {
        out << "{\n";
        emitMidEquivalent();
        emitMidSafe();
        emitIntervalPrice();
        emitHasOccurred();
        emitTradingExCoupon();
        emitSavedSettings();
        emitRebatedExercise();
        out << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "probe failed: " << e.what() << std::endl;
        return 1;
    }
}
