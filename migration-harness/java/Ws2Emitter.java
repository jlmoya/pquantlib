/**
 * Ws2Emitter — emit reference values for W-S2 (dividend-option compat layer).
 *
 * Harness artifact: not a port, not shipped in pquantlib.
 *
 * This is the PRIMARY cross-validation gate for W-S2. It prices the exact
 * test scenario from CRRDividendOptionTest / FDDividendOptionTest using the
 * JQuantLib helper classes (which internally drive
 * BinomialDividendVanillaEngine / FDDividend*Engine), and emits NPV + greeks
 * as JSON. The Python compat layer must reproduce these CRR values
 * step-for-step (TIGHT / EXACT), since pquantlib-helpers' own
 * BinomialDividendVanillaEngine is the line-for-line port of the Java one.
 *
 * The FD values are emitted for completeness (W-S3 builds FD helpers); the
 * W-S2 Python test only asserts the CRR (binomial) values against this file.
 *
 * Build (paths relative to migration-harness/java/):
 *   cd /Users/josemoya/eclipse-workspace/jquantlib/jquantlib-helpers
 *   mvn -q -o dependency:build-classpath -Dmdep.outputFile=/tmp/ws2_cp.txt
 *   JQ=/Users/josemoya/eclipse-workspace/jquantlib
 *   CP="$(cat /tmp/ws2_cp.txt):$JQ/jquantlib/target/classes:$JQ/jquantlib-helpers/target/classes"
 *   javac -cp "$CP" -d /tmp/ws2 migration-harness/java/Ws2Emitter.java
 *   java  -cp "$CP:/tmp/ws2" Ws2Emitter > migration-harness/references/cluster/ws2_java.json
 *
 * Scenario (identical to the JQuantLib helper test suite):
 *   type=Put, S=36, K=40, r=0.06, q=0.00, vol=0.20,
 *   today=15-May-1998, settlement=17-May-1998, maturity=17-May-1999,
 *   dayCounter=Actual365Fixed, calendar=Target,
 *   3 dividends of 2.06 each at today + i*3 months + 15 days (i=1,2,3).
 *   timeSteps = (maturity - settlement) * 3  (CRRDividendOptionHelper default).
 */
import java.util.ArrayList;
import java.util.List;

import org.jquantlib.Settings;
import org.jquantlib.daycounters.Actual365Fixed;
import org.jquantlib.daycounters.DayCounter;
import org.jquantlib.helpers.CRRAmericanDividendOptionHelper;
import org.jquantlib.helpers.CRREuropeanDividendOptionHelper;
import org.jquantlib.helpers.FDAmericanDividendOptionHelper;
import org.jquantlib.helpers.FDEuropeanDividendOptionHelper;
import org.jquantlib.instruments.Option;
import org.jquantlib.time.Calendar;
import org.jquantlib.time.Date;
import org.jquantlib.time.Month;
import org.jquantlib.time.Period;
import org.jquantlib.time.TimeUnit;
import org.jquantlib.time.calendars.Target;

public class Ws2Emitter {

    static String bits(final double d) {
        // Emit the raw IEEE-754 bit pattern (signed long) so Python can
        // reconstruct the exact double for EXACT-tier checks where the
        // arithmetic is deterministic across JVM <-> CPython.
        return Long.toString(Double.doubleToLongBits(d));
    }

    public static void main(final String[] args) {
        final Calendar calendar = new Target();
        final Date today = new Date(15, Month.May, 1998);
        final Date settlementDate = new Date(17, Month.May, 1998);
        final Option.Type type = Option.Type.Put;
        final double strike = 40.0;
        final double underlying = 36.0;
        final double riskFreeRate = 0.06;
        final double volatility = 0.2;
        final double dividendYield = 0.00;
        final Date maturityDate = new Date(17, Month.May, 1999);
        final DayCounter dc = new Actual365Fixed();

        final List<Date> divDates = new ArrayList<Date>();
        final List<Double> divAmounts = new ArrayList<Double>();
        for (int i = 1; i <= 3; i++) {
            final Date divDate = today.add(new Period(i * 3, TimeUnit.Months)).add(new Period(15, TimeUnit.Days));
            divDates.add(divDate);
            divAmounts.add(2.06);
        }

        // The helper default timeSteps = (maturity - settlement)*3.
        final int timeSteps = (int) (maturityDate.sub(settlementDate) * 3);

        final StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"scenario\": {\n");
        sb.append("    \"type\": \"Put\",\n");
        sb.append("    \"underlying\": ").append(underlying).append(",\n");
        sb.append("    \"strike\": ").append(strike).append(",\n");
        sb.append("    \"risk_free_rate\": ").append(riskFreeRate).append(",\n");
        sb.append("    \"dividend_yield\": ").append(dividendYield).append(",\n");
        sb.append("    \"volatility\": ").append(volatility).append(",\n");
        sb.append("    \"today_serial\": ").append(today.serialNumber()).append(",\n");
        sb.append("    \"settlement_serial\": ").append(settlementDate.serialNumber()).append(",\n");
        sb.append("    \"maturity_serial\": ").append(maturityDate.serialNumber()).append(",\n");
        sb.append("    \"day_counter\": \"Actual365Fixed\",\n");
        sb.append("    \"calendar\": \"Target\",\n");
        sb.append("    \"dividend_amount\": 2.06,\n");
        sb.append("    \"dividend_date_serials\": [");
        for (int i = 0; i < divDates.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(divDates.get(i).serialNumber());
        }
        sb.append("],\n");
        sb.append("    \"time_steps\": ").append(timeSteps).append("\n");
        sb.append("  },\n");

        // --- CRR European (PRIMARY GATE for W-S2) ---
        new Settings().setEvaluationDate(today);
        {
            final CRREuropeanDividendOptionHelper opt = new CRREuropeanDividendOptionHelper(
                    type, underlying, strike, riskFreeRate, dividendYield, volatility,
                    settlementDate, maturityDate, divDates, divAmounts, calendar, dc);
            emit(sb, "crr_european", opt.NPV(), opt.delta(), opt.gamma(), opt.theta(), true);
        }

        // --- CRR American (PRIMARY GATE for W-S2) ---
        new Settings().setEvaluationDate(today);
        {
            final CRRAmericanDividendOptionHelper opt = new CRRAmericanDividendOptionHelper(
                    type, underlying, strike, riskFreeRate, dividendYield, volatility,
                    settlementDate, maturityDate, divDates, divAmounts, calendar, dc);
            emit(sb, "crr_american", opt.NPV(), opt.delta(), opt.gamma(), opt.theta(), true);
        }

        // --- FD European (for W-S3; emitted for completeness) ---
        new Settings().setEvaluationDate(today);
        {
            final FDEuropeanDividendOptionHelper opt = new FDEuropeanDividendOptionHelper(
                    type, underlying, strike, riskFreeRate, dividendYield, volatility,
                    settlementDate, maturityDate, divDates, divAmounts, calendar, dc);
            emit(sb, "fd_european", opt.NPV(), opt.delta(), opt.gamma(), opt.theta(), true);
        }

        // --- FD American (for W-S3; emitted for completeness) ---
        new Settings().setEvaluationDate(today);
        {
            final FDAmericanDividendOptionHelper opt = new FDAmericanDividendOptionHelper(
                    type, underlying, strike, riskFreeRate, dividendYield, volatility,
                    settlementDate, maturityDate, divDates, divAmounts, calendar, dc);
            emit(sb, "fd_american", opt.NPV(), opt.delta(), opt.gamma(), opt.theta(), false);
        }

        sb.append("}\n");
        System.out.print(sb.toString());
    }

    static void emit(final StringBuilder sb, final String key,
                     final double npv, final double delta, final double gamma, final double theta,
                     final boolean trailingComma) {
        sb.append("  \"").append(key).append("\": {\n");
        sb.append("    \"npv\": ").append(npv).append(",\n");
        sb.append("    \"npv_bits\": ").append(bits(npv)).append(",\n");
        sb.append("    \"delta\": ").append(delta).append(",\n");
        sb.append("    \"delta_bits\": ").append(bits(delta)).append(",\n");
        sb.append("    \"gamma\": ").append(gamma).append(",\n");
        sb.append("    \"gamma_bits\": ").append(bits(gamma)).append(",\n");
        sb.append("    \"theta\": ").append(theta).append(",\n");
        sb.append("    \"theta_bits\": ").append(bits(theta)).append("\n");
        sb.append("  }").append(trailingComma ? "," : "").append("\n");
    }
}
