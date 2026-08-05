// migration-harness/cpp/probes/currencies_all/probe.cpp
//
// Reference data for EVERY ISO-4217 currency C++ QuantLib v1.43 ships, across
// all six per-continent headers (africa / america / asia / europe / oceania /
// crypto) — 111 classes in total.
//
// A currency is pure static data, so what needs pinning is exactly the payload
// of its Currency::Data: name, ISO code, ISO numeric code, symbol, fraction
// symbol, fractions per unit, the rounding convention, the triangulation
// currency (non-empty only for the pre-euro legacy currencies) and the minor
// unit codes. Every one of those is trivial to transcribe wrong and invisible
// until a conversion or a printed amount is off, which is why they are read
// off the C++ objects rather than off a data sheet.
//
// String encoding: a handful of symbols are Latin-1 byte escapes in the C++
// sources (USD "\xA2", GBP/CYP "\xA3", JPY "\xA5"), so the emitted strings are
// NOT valid UTF-8 byte-for-byte. Every byte >= 0x80 is therefore emitted as a
// \u00XX escape — i.e. the byte is read as a Latin-1 code point, which is what
// the sources mean and what the Python string literals hold. Output is pure
// ASCII and so is unambiguously loadable as UTF-8 JSON.
//
// Emits JSON on stdout; generate-references.sh redirects it to
// references/currencies/all.json.

#include <cstdio>
#include <iostream>
#include <string>

#include <ql/currencies/africa.hpp>
#include <ql/currencies/america.hpp>
#include <ql/currencies/asia.hpp>
#include <ql/currencies/crypto.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/currencies/oceania.hpp>
#include <ql/version.hpp>

using namespace QuantLib;

namespace {

// JSON-escape, mapping any byte >= 0x80 to its Latin-1 code point.
std::string esc(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char ch : s) {
        switch (ch) {
          case '"':  out += "\\\""; break;
          case '\\': out += "\\\\"; break;
          case '\b': out += "\\b";  break;
          case '\f': out += "\\f";  break;
          case '\n': out += "\\n";  break;
          case '\r': out += "\\r";  break;
          case '\t': out += "\\t";  break;
          default:
            if (ch < 0x20 || ch >= 0x80) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(ch));
                out += buf;
            } else {
                out += static_cast<char>(ch);
            }
        }
    }
    return out;
}

bool first = true;

void emit(const Currency& c) {
    if (!first)
        std::cout << ",\n";
    first = false;

    const Rounding& r = c.rounding();
    const Currency& tri = c.triangulationCurrency();

    std::cout << "    \"" << esc(c.code()) << "\": {\n"
              << "      \"name\": \"" << esc(c.name()) << "\",\n"
              << "      \"code\": \"" << esc(c.code()) << "\",\n"
              << "      \"numeric_code\": " << c.numericCode() << ",\n"
              << "      \"symbol\": \"" << esc(c.symbol()) << "\",\n"
              << "      \"fraction_symbol\": \"" << esc(c.fractionSymbol()) << "\",\n"
              << "      \"fractions_per_unit\": " << c.fractionsPerUnit() << ",\n"
              << "      \"rounding_type\": " << static_cast<int>(r.type()) << ",\n";
    // Rounding's default ctor is `= default` over `Integer precision_;` and
    // `Integer digit_;` with no initialiser — only type_ is `= None`. So on a
    // default-constructed Rounding those two members are indeterminate, and
    // emitting them would pin whatever this build happened to leave on the
    // stack. They are also never read: operator() returns early on type None.
    // Emit null for them instead, so the reference stays reproducible.
    if (r.type() == Rounding::None) {
        std::cout << "      \"rounding_precision\": null,\n"
                  << "      \"rounding_digit\": null,\n";
    } else {
        std::cout << "      \"rounding_precision\": " << r.precision() << ",\n"
                  << "      \"rounding_digit\": " << r.roundingDigit() << ",\n";
    }
    std::cout << "      \"triangulation_code\": "
              << (tri.empty() ? std::string("null") : "\"" + esc(tri.code()) + "\"") << ",\n"
              << "      \"minor_unit_codes\": [";
    bool firstMinor = true;
    for (const std::string& m : c.minorUnitCodes()) {
        if (!firstMinor)
            std::cout << ", ";
        firstMinor = false;
        std::cout << "\"" << esc(m) << "\"";
    }
    std::cout << "]\n"
              << "    }";
}

} // namespace

int main() {
    std::cout << "{\n";
    std::cout << "  \"quantlib_version\": \"" << QL_VERSION << "\",\n";
    std::cout << "  \"currencies\": {\n";

    // --- africa.cpp (14) ---
    emit(AOACurrency());
    emit(BWPCurrency());
    emit(EGPCurrency());
    emit(ETBCurrency());
    emit(GHSCurrency());
    emit(KESCurrency());
    emit(MADCurrency());
    emit(MURCurrency());
    emit(NGNCurrency());
    emit(TNDCurrency());
    emit(UGXCurrency());
    emit(XOFCurrency());
    emit(ZARCurrency());
    emit(ZMWCurrency());

    // --- america.cpp (16) ---
    emit(ARSCurrency());
    emit(BRLCurrency());
    emit(CADCurrency());
    emit(CLPCurrency());
    emit(COPCurrency());
    emit(MXNCurrency());
    emit(PENCurrency());
    emit(PEICurrency());
    emit(PEHCurrency());
    emit(TTDCurrency());
    emit(USDCurrency());
    emit(VEBCurrency());
    emit(MXVCurrency());
    emit(COUCurrency());
    emit(CLFCurrency());
    emit(UYUCurrency());

    // --- asia.cpp (29) ---
    emit(BDTCurrency());
    emit(CNYCurrency());
    emit(HKDCurrency());
    emit(IDRCurrency());
    emit(ILSCurrency());
    emit(INRCurrency());
    emit(IQDCurrency());
    emit(IRRCurrency());
    emit(JPYCurrency());
    emit(KRWCurrency());
    emit(KWDCurrency());
    emit(KZTCurrency());
    emit(MYRCurrency());
    emit(NPRCurrency());
    emit(PKRCurrency());
    emit(SARCurrency());
    emit(SGDCurrency());
    emit(THBCurrency());
    emit(TWDCurrency());
    emit(VNDCurrency());
    emit(QARCurrency());
    emit(BHDCurrency());
    emit(OMRCurrency());
    emit(JODCurrency());
    emit(AEDCurrency());
    emit(PHPCurrency());
    emit(CNHCurrency());
    emit(LKRCurrency());
    emit(UZSCurrency());

    // --- europe.cpp (42) ---
    emit(BGLCurrency());
    emit(BYRCurrency());
    emit(CHFCurrency());
    emit(CYPCurrency());
    emit(CZKCurrency());
    emit(DKKCurrency());
    emit(EEKCurrency());
    emit(EURCurrency());
    emit(GBPCurrency());
    emit(HUFCurrency());
    emit(ISKCurrency());
    emit(LTLCurrency());
    emit(LVLCurrency());
    emit(MKDCurrency());
    emit(NOKCurrency());
    emit(PLNCurrency());
    emit(ROLCurrency());
    emit(RONCurrency());
    emit(RUBCurrency());
    emit(SEKCurrency());
    emit(SITCurrency());
    emit(TRLCurrency());
    emit(TRYCurrency());
    // pre-euro legacy: these carry EUR as triangulation currency
    emit(ATSCurrency());
    emit(BEFCurrency());
    emit(DEMCurrency());
    emit(ESPCurrency());
    emit(FIMCurrency());
    emit(FRFCurrency());
    emit(GRDCurrency());
    emit(IEPCurrency());
    emit(ITLCurrency());
    emit(LUFCurrency());
    emit(MTLCurrency());
    emit(NLGCurrency());
    emit(PTECurrency());
    emit(SKKCurrency());
    emit(UAHCurrency());
    emit(RSDCurrency());
    emit(HRKCurrency());
    emit(BGNCurrency());
    emit(GELCurrency());

    // --- oceania.cpp (2) ---
    emit(AUDCurrency());
    emit(NZDCurrency());

    // --- crypto.cpp (8) ---
    emit(BTCCurrency());
    emit(ETHCurrency());
    emit(ETCCurrency());
    emit(BCHCurrency());
    emit(XRPCurrency());
    emit(LTCCurrency());
    emit(DASHCurrency());
    emit(ZECCurrency());

    std::cout << "\n  }\n}\n";
    return 0;
}
