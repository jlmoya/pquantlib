// migration-harness/cpp/probes/v143_trinomial_g2process/common.hpp
//
// Minimal ReferenceWriter shim.
//
// The probe body in this directory is shared verbatim with JQuantLib's
// harness, which writes its references through a `ReferenceWriter` that
// serialises a {test_group, cpp_version, cpp_commit, cases:[{name, inputs,
// expected}]} document straight to a file. PQuantLib's convention is
// different: probes print to stdout, generate-references.sh redirects, and the
// document is a flat object keyed by case name.
//
// Rather than fork the 600-line probe body — the thing that actually has to
// stay identical between the two ports — only the output layer is adapted.
// The API (`ReferenceWriter(test_group, version, generated_by)`, `addCase`,
// `write`) matches JQuantLib's so the shared body compiles unchanged.

#ifndef PQUANTLIB_HARNESS_V143_TRINOMIAL_G2PROCESS_COMMON_HPP
#define PQUANTLIB_HARNESS_V143_TRINOMIAL_G2PROCESS_COMMON_HPP

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace jqml_harness {

using json = nlohmann::json;

class ReferenceWriter {
  public:
    ReferenceWriter(std::string test_group, std::string cpp_version, std::string generated_by)
    : test_group_(std::move(test_group)), cpp_version_(std::move(cpp_version)),
      generated_by_(std::move(generated_by)) {}

    void addCase(const std::string& name, json inputs, json expected) {
        cases_.push_back(Case{name, std::move(inputs), std::move(expected)});
    }

    void write() const {
        json doc = json::object();
        for (const auto& c : cases_)
            doc[c.name] = json{{"inputs", c.inputs}, {"expected", c.expected}};
        // Metadata under a reserved key so it cannot collide with a case name.
        doc["_meta"] = json{{"test_group", test_group_},
                            {"cpp_version", cpp_version_},
                            {"generated_by", generated_by_}};
        std::cout << doc.dump(2) << std::endl;
    }

  private:
    struct Case {
        std::string name;
        json inputs;
        json expected;
    };
    std::string test_group_;
    std::string cpp_version_;
    std::string generated_by_;
    std::vector<Case> cases_;
};

}  // namespace jqml_harness

#endif
