"""MonteCarloModel — sample-collection driver.

# C++ parity: ql/methods/montecarlo/montecarlomodel.hpp (v1.42.1) —
# ``template <template <class> class MC, class RNG, class S>
#  class MonteCarloModel``.

C++ parameterizes on the MC trait (Single- vs Multi-variate), the
RNG family (PseudoRandom / LowDiscrepancy), and the statistics
accumulator (default ``Statistics`` = GeneralStatistics).  Python
collapses these into runtime polymorphism: ``MonteCarloModel``
takes a path generator, a path pricer, a statistics accumulator,
plus optional antithetic / control-variate inputs.

The orchestrator method is ``add_samples(N)`` which runs N MC
iterations, each one:

* drawing a sample path from ``path_generator.next()``,
* pricing it via ``path_pricer(path)``,
* optionally adding the geometric-average (or any other) control-
  variate adjustment ``cv_option_value - cv_path_pricer(path)``,
* optionally averaging with an antithetic-path sample,
* accumulating the result into the statistics aggregator.
"""

from __future__ import annotations

from typing import Protocol

from pquantlib import qassert
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.path_pricer import PathPricer


class PathSampleProtocol[PathT](Protocol):
    """Structural shape of a (path, weight) sample.

    # C++ parity: ``Sample<PathT>`` — see ``PathSample`` and
    # ``MultiPathSample``.
    """

    @property
    def value(self) -> PathT: ...

    @property
    def weight(self) -> float: ...


class PathGeneratorTypeProtocol[PathT](Protocol):
    """Structural shape of a path generator emitting ``PathT`` samples.

    # C++ parity: ``PathGenerator<GSG>`` and ``MultiPathGenerator<GSG>``
    # — both have ``next()`` and ``antithetic()`` returning
    # ``Sample<PathT>``.
    """

    def next(self) -> PathSampleProtocol[PathT]: ...

    def antithetic(self) -> PathSampleProtocol[PathT]: ...


class MonteCarloModel[PathT]:
    """Generic MC sample-collection driver.

    # C++ parity: ``MonteCarloModel<MC, RNG, S>`` (montecarlomodel.hpp).
    """

    __slots__ = (
        "_cv_option_value",
        "_cv_path_generator",
        "_cv_path_pricer",
        "_is_antithetic",
        "_is_control_variate",
        "_path_generator",
        "_path_pricer",
        "_sample_accumulator",
    )

    def __init__(
        self,
        path_generator: PathGeneratorTypeProtocol[PathT],
        path_pricer: PathPricer[PathT],
        sample_accumulator: GeneralStatistics,
        antithetic_variate: bool,
        cv_path_pricer: PathPricer[PathT] | None = None,
        cv_option_value: float = 0.0,
        cv_path_generator: PathGeneratorTypeProtocol[PathT] | None = None,
    ) -> None:
        """Wire the generator + pricer + accumulator.

        # C++ parity: ``MonteCarloModel(pathGenerator, pathPricer,
        # stats, antithetic, cvPathPricer=nullptr, cvValue=0,
        # cvPathGenerator=nullptr)``.
        """
        self._path_generator: PathGeneratorTypeProtocol[PathT] = path_generator
        self._path_pricer: PathPricer[PathT] = path_pricer
        self._sample_accumulator: GeneralStatistics = sample_accumulator
        self._is_antithetic: bool = antithetic_variate
        self._cv_path_pricer: PathPricer[PathT] | None = cv_path_pricer
        self._cv_option_value: float = cv_option_value
        self._cv_path_generator: PathGeneratorTypeProtocol[PathT] | None = cv_path_generator
        self._is_control_variate: bool = cv_path_pricer is not None

    def add_samples(self, samples: int) -> None:
        """Run ``samples`` MC iterations into the accumulator.

        # C++ parity: ``MonteCarloModel::addSamples`` (montecarlomodel.hpp:91-125).
        """
        for _ in range(samples):
            path_sample = self._path_generator.next()
            price = self._path_pricer(path_sample.value)

            if self._is_control_variate:
                qassert.require(self._cv_path_pricer is not None, "CV pricer required")
                assert self._cv_path_pricer is not None
                if self._cv_path_generator is None:
                    price += self._cv_option_value - self._cv_path_pricer(path_sample.value)
                else:
                    cv_path = self._cv_path_generator.next()
                    price += self._cv_option_value - self._cv_path_pricer(cv_path.value)

            if self._is_antithetic:
                anti_sample = self._path_generator.antithetic()
                price2 = self._path_pricer(anti_sample.value)
                if self._is_control_variate:
                    assert self._cv_path_pricer is not None
                    if self._cv_path_generator is None:
                        price2 += self._cv_option_value - self._cv_path_pricer(
                            anti_sample.value
                        )
                    else:
                        cv_anti = self._cv_path_generator.antithetic()
                        price2 += self._cv_option_value - self._cv_path_pricer(
                            cv_anti.value
                        )
                self._sample_accumulator.add((price + price2) / 2.0, path_sample.weight)
            else:
                self._sample_accumulator.add(price, path_sample.weight)

    def sample_accumulator(self) -> GeneralStatistics:
        """Return the underlying statistics accumulator.

        # C++ parity: ``MonteCarloModel::sampleAccumulator``.
        """
        return self._sample_accumulator


__all__ = ["MonteCarloModel", "PathGeneratorTypeProtocol", "PathSampleProtocol"]
