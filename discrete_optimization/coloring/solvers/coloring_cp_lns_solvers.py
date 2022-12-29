"""Large neighborhood search + Constraint programming toolbox for coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from enum import Enum
from typing import Any, Iterable, Optional

from minizinc import Instance

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.coloring.solvers.greedy_coloring import (
    GreedyColoring,
    NXGreedyColoringMethod,
)
from discrete_optimization.generic_tools.cp_tools import CPSolver
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_cp import (
    ConstraintHandler,
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

    def __init__(
        self,
        problem: ColoringProblem,
        initial_method: InitialColoringMethod,
        params_objective_function: ParamsObjectiveFunction,
    ):
        self.problem = problem
        self.initial_method = initial_method
        (
            self.aggreg_sol,
            self.aggreg_dict,
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
            fit = self.aggreg_sol(sol)
            return ResultStorage(
                list_solution_fits=[(sol, fit)],
                best_solution=sol,
                mode_optim=self.params_objective_function.sense_function,
            )
        else:
            solver = GreedyColoring(
                coloring_model=self.problem,
                params_objective_function=self.params_objective_function,
            )
            return solver.solve(strategy=NXGreedyColoringMethod.largest_first)


class ConstraintHandlerFixColorsCP(ConstraintHandler):
    """Constraint builder for LNS coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of nodes color.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    def __init__(self, problem: ColoringProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CPSolver,
        child_instance: Instance,
        previous_constraints: Iterable[Any],
    ) -> None:
        pass

    def adding_constraint_from_results_store(
        self,
        cp_solver: CPSolver,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        """Include constraint that fix decision on a subset of nodes, according to current solutions found.

        Args:
            cp_solver (CPSolver): a coloring CPSolver
            child_instance: minizinc instance where to include the constraint
            result_storage: current pool of solutions
            last_result_store: pool of solutions found in previous LNS iteration (optional)

        Returns: an empty list, unused.

        """
        range_node = range(1, self.problem.number_of_nodes + 1)
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, ColoringSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a ColoringSolution."
            )
        if current_solution.colors is None:
            raise ValueError(
                "result_storage.get_best_solution().colors " "should not be None."
            )
        subpart_color = set(
            random.sample(
                range_node, int(self.fraction_to_fix * self.problem.number_of_nodes)
            )
        )
        dict_color = {
            i + 1: current_solution.colors[i] + 1
            for i in range(self.problem.number_of_nodes)
        }
        current_nb_color = max(dict_color.values())
        list_strings = []
        for i in range_node:
            if i in subpart_color and dict_color[i] < current_nb_color:
                str1 = (
                    "constraint color_graph["
                    + str(i)
                    + "] == "
                    + str(dict_color[i])
                    + ";\n"
                )
                child_instance.add_string(str1)
                list_strings.append(str1)
            str1 = (
                "constraint color_graph["
                + str(i)
                + "] <= "
                + str(current_nb_color)
                + ";\n"
            )
            child_instance.add_string(str1)
            list_strings.append(str1)
        return list_strings


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
        params_objective_function: ParamsObjectiveFunction,
    ):
        self.problem = problem
        self.params_objective_function = params_objective_function
        (
            self.aggreg_from_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        bs = result_storage.get_best_solution()
        if bs is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(bs, ColoringSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a ColoringSolution."
            )
        if bs.colors is None:
            raise ValueError(
                "result_storage.get_best_solution().colors " "should not be None."
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
        result_storage.add_solution(new_solution, fit)
        return result_storage
