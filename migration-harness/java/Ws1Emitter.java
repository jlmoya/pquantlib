/**
 * Ws1Emitter — emit reference values for W-S1 (XorShiftRandom cross-validation).
 *
 * Harness artifact: not a port, not shipped in pquantlib.
 * Run with:
 *   javac -d /tmp/ws1 \
 *       ../../jquantlib-contrib/src/main/java/org/jquantlib/math/randomnumbers/XorShiftRandom.java \
 *       Ws1Emitter.java
 *   java -cp /tmp/ws1 Ws1Emitter > ../references/cluster/ws1.json
 *
 * (Adjust paths relative to migration-harness/java/)
 *
 * Output JSON schema:
 *   {
 *     "seed": 42,
 *     "next_long": [ ... 16 signed 64-bit longs ... ],
 *     "next_double_bits": [ ... 16 longs from Double.doubleToLongBits(d) ... ]
 *   }
 *
 * next_double_bits allows EXACT-tier reconstruction in Python via:
 *   struct.unpack('!d', struct.pack('!Q', bits))[0]
 * where bits is treated as unsigned (hence '!Q' not '!q').
 */
import org.jquantlib.math.randomnumbers.XorShiftRandom;

public class Ws1Emitter {
    public static void main(String[] args) {
        final long SEED = 42L;
        final int N = 16;

        // --- next_long sequence ---
        XorShiftRandom rng1 = new XorShiftRandom(SEED);
        long[] longs = new long[N];
        for (int i = 0; i < N; i++) {
            longs[i] = rng1.nextLong();
        }

        // --- next_double sequence (fresh seed) ---
        XorShiftRandom rng2 = new XorShiftRandom(SEED);
        long[] doubleBits = new long[N];
        for (int i = 0; i < N; i++) {
            double d = rng2.nextDouble();
            doubleBits[i] = Double.doubleToLongBits(d);
        }

        // --- emit JSON manually (no external deps) ---
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"seed\": ").append(SEED).append(",\n");

        // next_long array
        sb.append("  \"next_long\": [");
        for (int i = 0; i < N; i++) {
            sb.append(longs[i]);
            if (i < N - 1) sb.append(", ");
        }
        sb.append("],\n");

        // next_double_bits array
        sb.append("  \"next_double_bits\": [");
        for (int i = 0; i < N; i++) {
            // emit as unsigned decimal so Python can use struct.pack('!Q', bits)
            // Java doubleToLongBits returns a signed long; we emit the signed value.
            // Python will load it as int and use struct.pack('!q', bits) or handle sign.
            // Emit as signed long (may be negative for doubles with high exponent bits).
            sb.append(doubleBits[i]);
            if (i < N - 1) sb.append(", ");
        }
        sb.append("]\n");

        sb.append("}\n");
        System.out.print(sb.toString());
    }
}
