cmake_minimum_required(VERSION 3.20)
project(pquantlib_probes CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED QUANTLIB_SRC_DIR)
    message(FATAL_ERROR "QUANTLIB_SRC_DIR must be passed via -D")
endif()
if(NOT DEFINED QUANTLIB_BUILD_DIR)
    message(FATAL_ERROR "QUANTLIB_BUILD_DIR must be passed via -D")
endif()

# QuantLib transitively #include's <boost/config.hpp> via ql/qldefines.hpp,
# so probe TUs need Boost on the include path even though we don't link
# against any Boost libraries directly.
find_package(Boost REQUIRED)

include_directories(${QUANTLIB_SRC_DIR} ${Boost_INCLUDE_DIRS})
link_directories(${QUANTLIB_BUILD_DIR}/ql)

# Find dynamic library (handles .dylib on macOS, .so on Linux)
find_library(QUANTLIB_LIBRARY
    NAMES QuantLib libQuantLib
    PATHS ${QUANTLIB_BUILD_DIR}/ql
    NO_DEFAULT_PATH)
if(NOT QUANTLIB_LIBRARY)
    message(FATAL_ERROR "Could not find QuantLib library in ${QUANTLIB_BUILD_DIR}/ql")
endif()

# Add one add_executable + target_link_libraries per probe.
# Convention: probe at probes/<topic>/<class>_probe.cpp → executable name <topic>_<class>_probe
# The harness/generate-references.sh script turns the executable name into the
# JSON output path by translating underscores to slashes:
#   harness_sentinel_probe → references/harness/sentinel.json

# === harness self-check ===
add_executable(harness_sentinel_probe harness/sentinel_probe.cpp)
target_link_libraries(harness_sentinel_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / enums ===
add_executable(time_enums_probe time/enums_probe.cpp)
target_link_libraries(time_enums_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / period ===
add_executable(time_period_probe time/period_probe.cpp)
target_link_libraries(time_period_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / date ===
add_executable(time_date_probe time/date_probe.cpp)
target_link_libraries(time_date_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / parsers ===
add_executable(time_parsers_probe time/parsers_probe.cpp)
target_link_libraries(time_parsers_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / calendar ===
add_executable(time_calendar_probe time/calendar_probe.cpp)
target_link_libraries(time_calendar_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / imm ===
add_executable(time_imm_probe time/imm_probe.cpp)
target_link_libraries(time_imm_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / schedule ===
add_executable(time_schedule_probe time/schedule_probe.cpp)
target_link_libraries(time_schedule_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / asx + ecb ===
add_executable(time_asx_ecb_probe time/asx_ecb_probe.cpp)
target_link_libraries(time_asx_ecb_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / timegrid + timeseries ===
add_executable(time_timegrid_probe time/timegrid_probe.cpp)
target_link_libraries(time_timegrid_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / one ===
add_executable(daycounters_one_day_counter_probe daycounters/one_day_counter_probe.cpp)
target_link_libraries(daycounters_one_day_counter_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / actual family (360 / 364 / 365 / 36525 / 366) ===
add_executable(daycounters_actual_family_probe daycounters/actual_family_probe.cpp)
target_link_libraries(daycounters_actual_family_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / thirty family (Thirty360 9 conventions + Thirty365) ===
add_executable(daycounters_thirty_family_probe daycounters/thirty_family_probe.cpp)
target_link_libraries(daycounters_thirty_family_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / simple + business252 ===
add_executable(daycounters_simple_business_probe daycounters/simple_business_probe.cpp)
target_link_libraries(daycounters_simple_business_probe PRIVATE ${QUANTLIB_LIBRARY})

# === daycounters / actualactual ===
add_executable(daycounters_actualactual_probe daycounters/actualactual_probe.cpp)
target_link_libraries(daycounters_actualactual_probe PRIVATE ${QUANTLIB_LIBRARY})

# === time / calendars (all 41 sovereign/exchange, default-market) ===
add_executable(time_calendars_all_probe time/calendars_all_probe.cpp)
target_link_libraries(time_calendars_all_probe PRIVATE ${QUANTLIB_LIBRARY})

# === math / first batch (rounding, factorial, errorfunction, beta, ...) ===
add_executable(math_first_batch_probe math/first_batch_probe.cpp)
target_link_libraries(math_first_batch_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L1-B cluster: copulas + simple distributions + statistics + currencies ===
add_executable(cluster_b_probe cluster_b/probe.cpp)
target_link_libraries(cluster_b_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L1-C cluster: Solver1D family + simple integrals ===
add_executable(cluster_c_probe cluster_c/probe.cpp)
target_link_libraries(cluster_c_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L1-D cluster: deterministic RNGs ===
add_executable(cluster_d_probe cluster_d/probe.cpp)
target_link_libraries(cluster_d_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L1-E cluster: simple interpolations + matrix + math root ===
add_executable(cluster_e_probe cluster_e/probe.cpp)
target_link_libraries(cluster_e_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L2-A: foundations (quotes — SimpleQuote / DerivedQuote / CompositeQuote) ===
add_executable(quotes_foundations_probe quotes/foundations_probe.cpp)
target_link_libraries(quotes_foundations_probe PRIVATE ${QUANTLIB_LIBRARY})

# === L2-C cluster: indexes (Euribor/USDLibor/Sofr/...) + rate helpers ===
add_executable(cluster_l2c_probe cluster_l2c/probe.cpp)
target_link_libraries(cluster_l2c_probe PRIVATE ${QUANTLIB_LIBRARY})
