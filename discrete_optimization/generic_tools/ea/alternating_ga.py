#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Union

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    Problem,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class AlternatingGa:
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

    def __init__(
        self,
        problem: Problem,
        objectives: Union[str, List[str]],
        encodings: Optional[Union[List[str], List[Dict[str, Any]]]] = None,
        mutations: Optional[Union[List[Mutation], List[DeapMutation]]] = None,
        crossovers: Optional[List[DeapCrossover]] = None,
        selections: Optional[List[DeapSelection]] = None,
        objective_handling: Optional[ObjectiveHandling] = None,
        objective_weights: Optional[List[float]] = None,
        pop_size: Optional[int] = None,
        max_evals: int = 10000,
        sub_evals: Optional[List[int]] = None,
        mut_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        tournament_size: Optional[float] = None,
        deap_verbose: bool = False,
    ):
        self.problem = problem
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

        (
            self.aggreg_from_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem, params_objective_function=None
        )

    def solve(self, **kwargs: Any) -> ResultStorage:
        # Initialise the population (here at random)
        count_evals = 0
        current_encoding_index = 0
        if self.encodings is not None:
            for i in range(len(self.encodings)):
                self.problem.set_fixed_attributes(  # type: ignore
                    self.encodings[i], self.problem.get_dummy_solution()  # type: ignore
                )
        while count_evals < self.max_evals:
            kwargs_ga: Dict[str, Any] = {}
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
            tmp_sol = ga_solver.solve().get_best_solution()
            count_evals += self.sub_evals[current_encoding_index]  # type: ignore
            if self.encodings is not None:
                self.problem.set_fixed_attributes(  # type: ignore
                    self.encodings[current_encoding_index], tmp_sol
                )
        if tmp_sol is None:
            raise RuntimeError(
                "ga_solver.solve().get_best_solution() should not be None!"
            )
        problem_sol = tmp_sol
        result_storage = ResultStorage(
            list_solution_fits=[(problem_sol, self.aggreg_from_sol(problem_sol))],
            best_solution=problem_sol,
            mode_optim=self.params_objective_function.sense_function,
        )
        return result_storage
