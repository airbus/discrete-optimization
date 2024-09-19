#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

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
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


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
        objectives: Union[str, list[str]],
        encodings: Optional[Union[list[str], list[dict[str, Any]]]] = None,
        mutations: Optional[Union[list[Mutation], list[DeapMutation]]] = None,
        crossovers: Optional[list[DeapCrossover]] = None,
        selections: Optional[list[DeapSelection]] = None,
        objective_handling: Optional[ObjectiveHandling] = None,
        objective_weights: Optional[list[float]] = None,
        pop_size: Optional[int] = None,
        max_evals: int = 10000,
        sub_evals: Optional[list[int]] = None,
        mut_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        tournament_size: Optional[float] = None,
        deap_verbose: bool = False,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.encodings = encodings
        self.mutations = mutations
        self.crossovers = crossovers
        self.selections = selections
        self.objective_handling = objective_handling
        self.objectives = objectives
        self.objective_weights = objective_weights
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.sub_evals = sub_evals
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.deap_verbose = deap_verbose

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

        if self.encodings is not None:
            for i in range(len(self.encodings)):
                self.problem.set_fixed_attributes(  # type: ignore
                    self.encodings[i], start_solution  # type: ignore
                )
        while count_evals < self.max_evals:
            kwargs_ga: dict[str, Any] = {}
            if self.mutations is not None:
                kwargs_ga["mutation"] = self.mutations[current_encoding_index]
            if self.encodings is not None:
                kwargs_ga["encoding"] = self.encodings[current_encoding_index]
            if self.crossovers is not None:
                kwargs_ga["crossover"] = self.crossovers[current_encoding_index]
            if self.selections is not None:
                kwargs_ga["selection"] = self.selections[current_encoding_index]
            if self.sub_evals is not None:
                kwargs_ga["max_evals"] = self.sub_evals[current_encoding_index]
            if self.objective_handling is not None:
                kwargs_ga["objective_handling"] = self.objective_handling
            if self.pop_size is not None:
                kwargs_ga["pop_size"] = self.pop_size
            if self.mut_rate is not None:
                kwargs_ga["mut_rate"] = self.mut_rate
            if self.crossover_rate is not None:
                kwargs_ga["crossover_rate"] = self.crossover_rate
            if self.tournament_size is not None:
                kwargs_ga["tournament_size"] = self.tournament_size

            ga_solver = Ga(
                problem=self.problem,
                objectives=self.objectives,
                objective_weights=self.objective_weights,
                deap_verbose=self.deap_verbose,
                **kwargs_ga
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
