// Emit TimeGrid + TimeSeries reference values.
//
// TimeGrid: three constructors (regular, mandatory-only, mandatory+steps),
// index() / closest_index() / closest_time(), dt(i) and front/back.
// TimeSeries: insert (via map-style assignment), first/last date, size,
// lookup hits and misses.

#include <ql/time/date.hpp>
#include <ql/timegrid.hpp>
#include <ql/timeseries.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace QuantLib;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "{\n";

    // --- TimeGrid: regular -------------------------------------------------
    {
        TimeGrid tg(1.0, 4);  // [0.0, 0.25, 0.5, 0.75, 1.0]
        std::cout << "  \"regular\": {\n";
        std::cout << "    \"size\": " << tg.size() << ",\n";
        std::cout << "    \"times\": [";
        for (size_t i = 0; i < tg.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << tg[i];
        }
        std::cout << "],\n";
        std::cout << "    \"dt\": [";
        for (size_t i = 0; i < tg.size() - 1; ++i) {
            if (i) std::cout << ", ";
            std::cout << tg.dt(i);
        }
        std::cout << "],\n";
        std::cout << "    \"front\": " << tg.front() << ",\n";
        std::cout << "    \"back\": " << tg.back() << "\n";
        std::cout << "  },\n";
    }

    // --- TimeGrid: mandatory-only ----------------------------------------
    {
        std::vector<Time> mt = {0.5, 1.0, 1.5, 2.0};
        TimeGrid tg(mt.begin(), mt.end());
        // Should prepend 0.0 → [0, 0.5, 1, 1.5, 2]
        std::cout << "  \"mandatory_only\": {\n";
        std::cout << "    \"size\": " << tg.size() << ",\n";
        std::cout << "    \"times\": [";
        for (size_t i = 0; i < tg.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << tg[i];
        }
        std::cout << "]\n";
        std::cout << "  },\n";
    }

    // --- TimeGrid: mandatory + steps -------------------------------------
    {
        std::vector<Time> mt = {1.0, 2.0};
        TimeGrid tg(mt.begin(), mt.end(), 4);  // dtMax = 2/4 = 0.5
        std::cout << "  \"mandatory_with_steps\": {\n";
        std::cout << "    \"size\": " << tg.size() << ",\n";
        std::cout << "    \"times\": [";
        for (size_t i = 0; i < tg.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << tg[i];
        }
        std::cout << "]\n";
        std::cout << "  },\n";
    }

    // --- TimeGrid: index / closest_index ---------------------------------
    {
        TimeGrid tg(1.0, 4);  // [0.0, 0.25, 0.5, 0.75, 1.0]
        std::cout << "  \"lookups\": {\n";
        std::cout << "    \"index_at_0_5\": " << tg.index(0.5) << ",\n";
        std::cout << "    \"closest_index_at_0_4\": " << tg.closestIndex(0.4) << ",\n";
        std::cout << "    \"closest_index_at_0_6\": " << tg.closestIndex(0.6) << ",\n";
        std::cout << "    \"closest_index_at_neg\": " << tg.closestIndex(-1.0) << ",\n";
        std::cout << "    \"closest_index_at_big\": " << tg.closestIndex(100.0) << ",\n";
        std::cout << "    \"closest_time_at_0_4\": " << tg.closestTime(0.4) << "\n";
        std::cout << "  },\n";
    }

    // --- TimeSeries<double> ----------------------------------------------
    {
        TimeSeries<double> ts;
        ts[Date(15, March, 2024)] = 1.0;
        ts[Date(15, April, 2024)] = 2.0;
        ts[Date(15, May, 2024)]   = 3.0;

        std::cout << "  \"timeseries\": {\n";
        std::cout << "    \"size\": " << ts.size() << ",\n";
        std::cout << "    \"first_date_serial\": " << ts.firstDate().serialNumber() << ",\n";
        std::cout << "    \"last_date_serial\": " << ts.lastDate().serialNumber() << ",\n";
        std::cout << "    \"value_at_april\": " << ts[Date(15, April, 2024)] << "\n";
        std::cout << "  }\n";
    }

    std::cout << "}\n";
    return 0;
}
