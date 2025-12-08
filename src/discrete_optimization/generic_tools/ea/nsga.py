#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional, Union

import numpy as np
from deap import algorithms, tools

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.ea.base import (
    BaseGa,
    DeapCrossover,
    DeapMutation,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class Nsga(BaseGa):
    """NSGA

    Args:
        problem:
            the problem to solve
        encoding:
            name (str) of an encoding registered in the register solution of Problem
            or a dictionary of the form {'type': TypeAttribute, 'n': int} where type refers to a TypeAttribute and n
             to the dimension of the problem in this encoding (e.g. length of the vector)
            by default, the first encoding in the problem register_solution will be used.

    """

    allowed_objective_handling = [ObjectiveHandling.MULTI_OBJ]

    initial_solution: Optional[Solution] = None
    """Initial solution used for warm start."""

    def __init__(
        self,
        problem: Problem,
        objectives: Optional[Union[str, list[str]]] = None,
        mutation: Optional[Union[Mutation, DeapMutation]] = None,
        crossover: Optional[DeapCrossover] = None,
        encoding: Optional[Union[str, dict[str, Any]]] = None,
        objective_weights: Optional[list[float]] = None,
        pop_size: int = 100,
        max_evals: Optional[int] = None,
        mut_rate: float = 0.1,
        crossover_rate: float = 0.9,
        deap_verbose: bool = True,
        initial_population: Optional[list[list[Any]]] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            objectives=objectives,
            mutation=mutation,
            crossover=crossover,
            encoding=encoding,
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            objective_weights=objective_weights,
            pop_size=pop_size,
            max_evals=max_evals,
            mut_rate=mut_rate,
            crossover_rate=crossover_rate,
            deap_verbose=deap_verbose,
            initial_population=initial_population,
            params_objective_function=params_objective_function,
            **kwargs,
        )

        # No choice of selection: In NSGA, only 1 selection: Non Dominated Sorted Selection
        ref_points = tools.uniform_reference_points(nobj=len(self._objectives))
        self._toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    def solve(self, callbacks: list[Callback] = None, **kwargs: Any) -> ResultStorage:
        callback = CallbackList(callbacks)
        callback.on_solve_start(self)

        #  Define the statistics to collect at each generation
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Initialize population
        population = self.generate_initial_population()

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Compile statistics about the population
        record = stats.compile(population)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        logger.debug(logbook.stream)

        # Begin the generational process
        ngen = int(self._max_evals / self._pop_size)
        logger.debug(f"ngen: {ngen}")
        for gen in range(1, ngen):
            offspring = algorithms.varAnd(
                population, self._toolbox, self._crossover_rate, self._mut_rate
            )

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            population = self._toolbox.select(population + offspring, self._pop_size)

            # Compile statistics about the new population
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            logger.debug(logbook.stream)

        solfits = []
        for s in population:
            pure_int_sol = [i for i in s]
            problem_sol = self.problem.build_solution_from_encoding(
                int_vector=pure_int_sol, encoding_name=self._attribute_name
            )
            solfits.append((problem_sol, self.aggreg_from_sol(problem_sol)))
        result_storage = self.create_result_storage(solfits)

        callback.on_solve_end(result_storage, self)

        return result_storage
