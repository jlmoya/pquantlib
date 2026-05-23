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

include_directories(${QUANTLIB_SRC_DIR})
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
#
# Example (uncomment + adapt when adding the first real probe):
#
# add_executable(math_copulas_clayton_probe math/copulas/clayton_probe.cpp)
# target_link_libraries(math_copulas_clayton_probe PRIVATE ${QUANTLIB_LIBRARY})

# Placeholder so cmake doesn't fail on an empty project
add_custom_target(noop_placeholder ALL
    COMMAND ${CMAKE_COMMAND} -E echo "No probes registered yet. See README.")
