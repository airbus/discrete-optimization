#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Optional, Union

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.generic_tools.encoding_register import AttributeType
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class AlternatingGa(SolverDO, WarmstartMixin):
    """Multi-encoding single objective GA

    Args:
        problem:
            the problem to solve
        encoding:
            name (str) of an encoding registered in the register solution of Problem
            or a dictionary of the form {'type': TypeAttribute, 'n': int} where type refers to a TypeAttribute and n
             to the dimension of the problem in this encoding (e.g. length of the vector)
            by default, the first encoding in the problem register_solution will be used.

    """

    initial_solution: Optional[Solution] = None
    """Initial solution used for warm start."""

    def __init__(
        self,
        problem: Problem,
        objectives: str | list[str] | None,
        encodings: Optional[Union[list[str], list[tuple[str, AttributeType]]]] = None,
        mutations: list[Mutation | DeapMutation | None] | None = None,
        crossovers: list[DeapCrossover | None] | None = None,
        selections: list[DeapSelection | None] | None = None,
        objective_handling: Optional[ObjectiveHandling] = None,
        objective_weights: list[float] | None = None,
        pop_size: int = 100,
        max_evals: int = 10000,
        sub_evals: list[int] | None = None,
        mut_rate: float | None = None,
        crossover_rate: float | None = None,
        tournament_size: float | None = None,  # as a percentage of the population
        deap_verbose: bool = False,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.mutations = mutations
        self.crossovers = crossovers
        self.selections = selections
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.objective_weights = objective_weights
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.deap_verbose = deap_verbose
        self.user_defined_params_objective_function = (
            params_objective_function is not None
        )
        if encodings is None:
            self.encodings = list(self.problem.get_attribute_register())
        else:
            self.encodings = encodings
        if sub_evals is None:
            logger.warning(
                "No value specified for sub_evals. Using the default 100*pop_size - This should really be set carefully"
            )
            self.sub_evals = [100 * pop_size] * len(self.encodings)
        else:
            self.sub_evals = sub_evals

    def set_warm_start(self, solution: Solution) -> None:
        """Make the solver warm start from the given solution.

        Will be ignored if arg `initial_variable` is set and not None in call to `solve()`.

        """
        self.initial_solution = solution

    def solve(self, **kwargs: Any) -> ResultStorage:
        # Initialise the population (here at random)
        count_evals = 0
        current_encoding_index = 0

        if self.initial_solution is None:
            start_solution = self.problem.get_dummy_solution()
        else:
            start_solution = self.initial_solution

        for attribute_name in self.encodings:
            self.problem.set_fixed_attributes(
                attribute_name=attribute_name,
                solution=start_solution,
            )
        tmp_sol = None
        while count_evals < self.max_evals:
            kwargs_ga: dict[str, Any] = {}
            kwargs_ga["encoding"] = self.encodings[current_encoding_index]
            kwargs_ga["max_evals"] = self.sub_evals[current_encoding_index]
            kwargs_ga["pop_size"] = self.pop_size
            if self.mutations is not None:
                kwargs_ga["mutation"] = self.mutations[current_encoding_index]
            if self.crossovers is not None:
                kwargs_ga["crossover"] = self.crossovers[current_encoding_index]
            if self.selections is not None:
                kwargs_ga["selection"] = self.selections[current_encoding_index]
            if self.objective_handling is not None:
                kwargs_ga["objective_handling"] = self.objective_handling
            if self.mut_rate is not None:
                kwargs_ga["mut_rate"] = self.mut_rate
            if self.crossover_rate is not None:
                kwargs_ga["crossover_rate"] = self.crossover_rate
            if self.tournament_size is not None:
                kwargs_ga["tournament_size"] = self.tournament_size
            if self.user_defined_params_objective_function:
                kwargs_ga["params_objective_function"] = self.params_objective_function

            ga_solver = Ga(
                problem=self.problem,
                objectives=self.objectives,
                objective_weights=self.objective_weights,
                deap_verbose=self.deap_verbose,
                **kwargs_ga,
            )
            # manage warm_start for first step
            if count_evals == 0 and self.initial_solution is not None:
                ga_solver.set_warm_start(self.initial_solution)

            tmp_sol = ga_solver.solve().get_best_solution()
            count_evals += self.sub_evals[current_encoding_index]  # type: ignore
            if self.encodings is not None:
                self.problem.set_fixed_attributes(  # type: ignore
                    self.encodings[current_encoding_index], tmp_sol
                )
            current_encoding_index += 1
            current_encoding_index = current_encoding_index % len(self.encodings)
        if tmp_sol is None:
            raise RuntimeError(
                "ga_solver.solve().get_best_solution() should not be None!"
            )
        problem_sol = tmp_sol
        result_storage = self.create_result_storage(
            [(problem_sol, self.aggreg_from_sol(problem_sol))],
        )
        return result_storage
