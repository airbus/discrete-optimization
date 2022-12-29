"""Large neighborhood search + Linear programming toolbox for coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from enum import Enum
from typing import Any, Dict, Hashable, Mapping

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_lp_solvers import (
    ColoringLP,
    ColoringLP_MIP,
)
from discrete_optimization.coloring.solvers.greedy_coloring import GreedyColoring
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_mip import (
    ConstraintHandler,
    InitialSolution,
)
from discrete_optimization.generic_tools.lp_tools import MilpSolver, MilpSolverName
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
            return solver.solve()


class ConstraintHandlerFixColorsGrb(ConstraintHandler):
    """Constraint builder used in LNS+LP (using gurobi solver) for coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of nodes color.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    def __init__(self, problem: ColoringProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix

    def adding_constraint_from_results_store(
        self, milp_solver: MilpSolver, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:
        if not isinstance(milp_solver, ColoringLP):
            raise ValueError("milp_solver must a ColoringLP for this constraint.")
        if milp_solver.model is None:
            milp_solver.init_model()
            if milp_solver.model is None:
                raise RuntimeError(
                    "milp_solver.model must be not None after calling milp_solver.init_model()"
                )
        subpart_color = set(
            random.sample(
                milp_solver.nodes_name,
                int(self.fraction_to_fix * milp_solver.number_of_nodes),
            )
        )
        dict_color_fixed = {}
        dict_color_start = {}
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
        max_color = max(current_solution.colors)
        for n in milp_solver.nodes_name:
            dict_color_start[n] = current_solution.colors[
                milp_solver.index_nodes_name[n]
            ]
            if n in subpart_color and dict_color_start[n] <= max_color - 1:
                dict_color_fixed[n] = dict_color_start[n]
        colors_var = milp_solver.variable_decision["colors_var"]
        lns_constraint: Dict[Hashable, Any] = {}
        for key in colors_var:
            n, c = key
            if c == dict_color_start[n]:
                colors_var[n, c].start = 1
                colors_var[n, c].varhintval = 1
            else:
                colors_var[n, c].start = 0
                colors_var[n, c].varhintval = 0
            if n in dict_color_fixed:
                if c == dict_color_fixed[n]:
                    lns_constraint[(n, c)] = milp_solver.model.addConstr(
                        colors_var[key] == 1, name=str((n, c))
                    )
                else:
                    lns_constraint[(n, c)] = milp_solver.model.addConstr(
                        colors_var[key] == 0, name=str((n, c))
                    )
        return lns_constraint

    def remove_constraints_from_previous_iteration(
        self, milp_solver: MilpSolver, previous_constraints: Mapping[Hashable, Any]
    ) -> None:
        if not isinstance(milp_solver, ColoringLP):
            raise ValueError("milp_solver must a ColoringLP for this constraint.")
        if milp_solver.model is None:
            milp_solver.init_model()
            if milp_solver.model is None:
                raise RuntimeError(
                    "milp_solver.model must be not None after calling milp_solver.init_model()"
                )
        milp_solver.model.remove(list(previous_constraints.values()))
        milp_solver.model.update()


class ConstraintHandlerFixColorsPyMip(ConstraintHandler):
    """Constraint builder used in LNS+ LP (using pymip library) for coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of nodes color.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    def __init__(self, problem: ColoringProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix

    def adding_constraint_from_results_store(
        self, milp_solver: MilpSolver, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:
        if not isinstance(milp_solver, ColoringLP_MIP):
            raise ValueError("milp_solver must a ColoringLP for this constraint.")
        if milp_solver.model is None:
            milp_solver.init_model()
            if milp_solver.model is None:
                raise RuntimeError(
                    "milp_solver.model must be not None after calling milp_solver.init_model()"
                )
        subpart_color = set(
            random.sample(
                milp_solver.nodes_name,
                int(self.fraction_to_fix * milp_solver.number_of_nodes),
            )
        )

        dict_color_fixed = {}
        dict_color_start = {}
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
        max_color = max(current_solution.colors)
        for n in milp_solver.nodes_name:
            dict_color_start[n] = current_solution.colors[
                milp_solver.index_nodes_name[n]
            ]
            if n in subpart_color and dict_color_start[n] <= max_color - 1:
                dict_color_fixed[n] = dict_color_start[n]
        colors_var = milp_solver.variable_decision["colors_var"]
        lns_constraint: Dict[Hashable, Any] = {}
        start = []
        for key in colors_var:
            n, c = key
            if c == dict_color_start[n]:
                start += [(colors_var[n, c], 1)]
            else:
                start += [(colors_var[n, c], 0)]
            if n in dict_color_fixed:
                if c == dict_color_fixed[n]:
                    lns_constraint[(n, c)] = milp_solver.model.add_constr(
                        colors_var[key] == 1, name=str((n, c))
                    )
                else:
                    lns_constraint[(n, c)] = milp_solver.model.add_constr(
                        colors_var[key] == 0, name=str((n, c))
                    )
        milp_solver.model.start = start
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
        return lns_constraint

    def remove_constraints_from_previous_iteration(
        self, milp_solver: MilpSolver, previous_constraints: Mapping[Hashable, Any]
    ) -> None:
        if not isinstance(milp_solver, ColoringLP_MIP):
            raise ValueError("milp_solver must a ColoringLP for this constraint.")
        if milp_solver.model is None:
            milp_solver.init_model()
            if milp_solver.model is None:
                raise RuntimeError(
                    "milp_solver.model must be not None after calling milp_solver.init_model()"
                )
        milp_solver.model.remove(list(previous_constraints.values()))
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
