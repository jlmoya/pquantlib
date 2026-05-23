# migration-harness

C++ ground-truth infrastructure for PQuantLib.

## Layout

```
migration-harness/
├── cpp/
│   ├── quantlib/          # git submodule → QuantLib v1.42.1 @ 099987f0
│   ├── probes/            # one-off C++ probes emitting reference values
│   └── build/             # cmake build directory (gitignored)
├── references/            # JSON reference value files (committed)
├── build-cpp.sh           # one-time C++ submodule + probes build
└── generate-references.sh # rebuild + run all probes, regenerate JSONs
```

## One-time setup

```bash
# 1. Initialize the QuantLib submodule (pinned to v1.42.1 @ 099987f0)
cd /Users/josemoya/Projects/PycharmProjects/pquantlib
git submodule add https://github.com/lballabio/QuantLib.git migration-harness/cpp/quantlib
cd migration-harness/cpp/quantlib
git checkout 099987f0ca2c11c505dc4348cdb9ce01a598e1e5
cd ../../..
git submodule update --init --recursive

# 2. Build QuantLib + all probes
./migration-harness/build-cpp.sh
```

## Per-phase workflow

When porting a class that needs cross-validation:

1. Write a small C++ probe under `migration-harness/cpp/probes/<topic>/<class>_probe.cpp`
2. Add it to `migration-harness/cpp/probes/CMakeLists.txt` (one line per probe)
3. Run `./migration-harness/generate-references.sh <topic>/<class>_probe`
4. The probe emits a JSON to `migration-harness/references/<topic>/<class>.json`
5. The pytest test loads the JSON via `pquantlib.testing.reference_reader.load("<topic>/<class>")` and compares via tolerance helpers

## Probe template

```cpp
// migration-harness/cpp/probes/math/copulas/clayton_copula_probe.cpp
#include <ql/math/copulas/claytoncopula.hpp>
#include <iostream>
#include <iomanip>

using namespace QuantLib;

int main() {
    // High precision for cross-platform reproducibility
    std::cout << std::setprecision(17);

    // Emit JSON to stdout (caller redirects to references/<topic>/<class>.json)
    std::cout << "{\n";
    std::cout << "  \"clayton_theta_0.5_at_(0.3,0.6)\": "
              << ClaytonCopula(0.5)(0.3, 0.6) << ",\n";
    std::cout << "  \"clayton_theta_2.0_at_(0.5,0.5)\": "
              << ClaytonCopula(2.0)(0.5, 0.5) << "\n";
    std::cout << "}\n";
    return 0;
}
```

## Notes

- All probe binaries are built with the same C++ standard library and optimization flags as QuantLib v1.42.1's recommended build.
- Reference JSONs are committed (small) so tests don't require the C++ build at runtime.
- A C++ rebuild is only required when adding NEW probes or updating the QuantLib submodule pin.
