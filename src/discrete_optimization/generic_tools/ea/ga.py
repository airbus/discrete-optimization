#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any

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
)
from discrete_optimization.generic_tools.ea.base import (
    BaseGa,
    DeapCrossover,
    DeapMutation,
)
from discrete_optimization.generic_tools.encoding_register import (
    AttributeType,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class DeapSelection(Enum):
    SEL_TOURNAMENT = 0
    SEL_RANDOM = 1
    SEL_BEST = 2
    SEL_ROULETTE = 4
    SEL_WORST = 5
    SEL_STOCHASTIC_UNIVERSAL_SAMPLING = 6


class Ga(BaseGa):
    """Single objective GA

    Args:
        problem:
            the problem to solve
        encoding:
            name (str) of an encoding registered in the register solution of Problem
            or a dictionary of the form {'type': TypeAttribute, 'n': int} where type refers to a TypeAttribute and n
             to the dimension of the problem in this encoding (e.g. length of the vector)
            by default, the first encoding in the problem register_solution will be used.

    """

    hyperparameters = BaseGa.hyperparameters + [
        EnumHyperparameter(
            name="selection", enum=DeapSelection, default=DeapSelection.SEL_TOURNAMENT
        ),
        FloatHyperparameter(
            name="tournament_size",
            low=0,
            high=1,
            default=0.2,
            depends_on=("selection", [DeapSelection.SEL_TOURNAMENT]),
        ),
    ]

    allowed_objective_handling = [ObjectiveHandling.SINGLE, ObjectiveHandling.AGGREGATE]

    def __init__(
        self,
        problem: Problem,
        objectives: str | list[str] | None = None,
        mutation: Mutation | DeapMutation | None = None,
        crossover: DeapCrossover | None = None,
        selection: DeapSelection = DeapSelection.SEL_TOURNAMENT,
        encoding: str | tuple[str, AttributeType] | None = None,
        objective_handling: ObjectiveHandling = ObjectiveHandling.AGGREGATE,
        objective_weights: list[float] | None = None,
        pop_size: int = 100,
        max_evals: int | None = None,
        mut_rate: float = 0.1,
        crossover_rate: float = 0.9,
        tournament_size: float = 0.2,  # as a percentage of the population
        deap_verbose: bool = True,
        initial_population: list[list[Any]] | None = None,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            objectives=objectives,
            mutation=mutation,
            crossover=crossover,
            encoding=encoding,
            objective_handling=objective_handling,
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
        self._tournament_size = tournament_size
        self._selection_type = selection

        # Choice of selection
        if self._selection_type == DeapSelection.SEL_TOURNAMENT:
            self._toolbox.register(
                "select",
                tools.selTournament,
                tournsize=int(self._tournament_size * self._pop_size),
            )
        elif self._selection_type == DeapSelection.SEL_RANDOM:
            self._toolbox.register("select", tools.selRandom)
        elif self._selection_type == DeapSelection.SEL_BEST:
            self._toolbox.register("select", tools.selBest)
        elif self._selection_type == DeapSelection.SEL_ROULETTE:
            self._toolbox.register("select", tools.selRoulette)
        elif self._selection_type == DeapSelection.SEL_WORST:
            self._toolbox.register("select", tools.selWorst)
        elif self._selection_type == DeapSelection.SEL_STOCHASTIC_UNIVERSAL_SAMPLING:
            self._toolbox.register("select", tools.selStochasticUniversalSampling)

    def solve(self, callbacks: list[Callback] = None, **kwargs: Any) -> ResultStorage:
        callback = CallbackList(callbacks)
        callback.on_solve_start(self)

        #  Define the statistics to collect at each generation
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(1)

        # Initialize population
        population = self.generate_initial_population()

        fits = self._toolbox.map(self._toolbox.evaluate, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit

        # Run the GA: final population and statistics logbook are created
        pop_vector, logbook = algorithms.eaSimple(
            population=population,
            toolbox=self._toolbox,
            cxpb=self._crossover_rate,
            mutpb=self._mut_rate,
            ngen=int(self._max_evals / self._pop_size),
            stats=stats,
            halloffame=hof,
            verbose=self._deap_verbose,
        )

        best_vector = hof[0]

        pure_int_sol = [i for i in best_vector]
        problem_sol = self.problem.build_solution_from_encoding(
            int_vector=pure_int_sol, encoding_name=self._attribute_name
        )

        result_storage = self.create_result_storage(
            [(problem_sol, self.aggreg_from_sol(problem_sol))],
        )
        callback.on_solve_end(result_storage, self)
        return result_storage
