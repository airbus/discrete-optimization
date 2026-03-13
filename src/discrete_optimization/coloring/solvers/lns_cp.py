#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Easy Large neighborhood search solver for coloring."""

import random
from enum import Enum
from typing import Any, Iterable, Optional

from ortools.sat.python.cp_model import Constraint

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.solvers.cpsat import CpSatColoringSolver
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import (
    OrtoolsCpSatConstraintHandler,
)
from discrete_optimization.generic_tools.lns_tools import (
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class InitialColoringMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialColoring(InitialSolution):
    """Initial solution provider for lns algorithm.

    Attributes:
        problem (ColoringProblem): input coloring problem
        initial_method (InitialColoringMethod): the method to use to provide the initial solution.
    """

    hyperparameters = [
        EnumHyperparameter(
            "initial_method",
            enum=InitialColoringMethod,
            default=InitialColoringMethod.GREEDY,
        )
    ]

    def __init__(
        self,
        problem: ColoringProblem,
        initial_method: InitialColoringMethod,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        self.initial_method = initial_method
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem, params_objective_function=params_objective_function
        )

    def get_starting_solution(self) -> ResultStorage:
        """Compute initial solution via greedy methods.

        Returns: initial solution storage
        """
        if self.initial_method == InitialColoringMethod.DUMMY:
            sol = self.problem.get_dummy_solution()
            fit = self.aggreg_from_sol(sol)
            return ResultStorage(
                mode_optim=self.params_objective_function.sense_function,
                list_solution_fits=[(sol, fit)],
            )
        else:
            solver = GreedyColoringSolver(
                problem=self.problem,
                params_objective_function=self.params_objective_function,
            )
            return solver.solve(strategy=NxGreedyColoringMethod.largest_first)


class FixColorsCpSatConstraintHandler(OrtoolsCpSatConstraintHandler):
    """Constraint builder for LNS coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of nodes color.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    hyperparameters = [
        FloatHyperparameter("fraction_to_fix", low=0.0, high=1.0, default=0.9),
    ]

    def __init__(self, problem: ColoringProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix

    def adding_constraint_from_results_store(
        self,
        solver: CpSatColoringSolver,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Constraint]:
        """Include constraint that fix decision on a subset of nodes, according to current solutions found.

        Args:
            solver: a coloring CpSolver
            result_storage: current pool of solutions
            result_storage_last_iteration: pool of solutions found in previous LNS iteration
            **kwargs:

        Returns:
            list of added constraints

        """
        range_node = range(self.problem.number_of_nodes)
        current_solution = self.extract_best_solution_from_last_iteration(
            result_storage=result_storage,
            result_storage_last_iteration=result_storage_last_iteration,
        )
        if current_solution is None:
            raise ValueError("result_storage.get_best_solution() should not be None.")
        if not isinstance(current_solution, ColoringSolution):
            raise ValueError(
                "result_storage.get_best_solution() should be a ColoringSolution."
            )
        if current_solution.colors is None:
            raise ValueError(
                "result_storage.get_best_solution().colors should not be None."
            )
        subpart_color = set(
            random.sample(
                range_node, int(self.fraction_to_fix * self.problem.number_of_nodes)
            )
        )
        dict_color = {
            i: current_solution.colors[i] for i in range(self.problem.number_of_nodes)
        }
        current_nb_color = max(dict_color.values())
        constraints = []
        for i in range_node:
            if i in subpart_color and dict_color[i] < current_nb_color:
                constraints.append(
                    solver.cp_model.Add(solver.variables["colors"][i] == dict_color[i])
                )
            constraints.append(
                solver.cp_model.Add(solver.variables["colors"][i] <= current_nb_color)
            )
        return constraints


class PostProcessSolutionColoring(PostProcessSolution):
    """Post process class for coloring problem.

     It transforms the color vector to have colors between 0 and nb_colors-1

    Attributes:
        problem (ColoringProblem): coloring instance
        params_objective_function (ParamsObjectiveFunction): params of the objective function
    """

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        self.params_objective_function = params_objective_function
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        bs = result_storage.get_best_solution()
        if bs is None:
            raise ValueError("result_storage.get_best_solution() should not be None.")
        if not isinstance(bs, ColoringSolution):
            raise ValueError(
                "result_storage.get_best_solution() should be a ColoringSolution."
            )
        if bs.colors is None:
            raise ValueError(
                "result_storage.get_best_solution().colors should not be None."
            )
        colors = bs.colors
        set_colors = sorted(set(colors))
        nb_colors = len(set(set_colors))
        new_color_dict = {set_colors[i]: i for i in range(nb_colors)}
        new_solution = ColoringSolution(
            problem=self.problem,
            colors=[new_color_dict[colors[i]] for i in range(len(colors))],
        )
        fit = self.aggreg_from_sol(new_solution)
        result_storage.append((new_solution, fit))
        return result_storage
