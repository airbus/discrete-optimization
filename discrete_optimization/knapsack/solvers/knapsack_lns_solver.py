#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from enum import Enum
from typing import Any, Dict, Hashable, Mapping

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_mip import (
    ConstraintHandler,
    InitialSolution,
)
from discrete_optimization.generic_tools.lp_tools import MilpSolver, MilpSolverName
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.greedy_solvers import (
    GreedyBest,
    ResultStorage,
)
from discrete_optimization.knapsack.solvers.lp_solvers import LPKnapsack


class InitialKnapsackMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialKnapsackSolution(InitialSolution):
    def __init__(
        self,
        problem: KnapsackModel,
        initial_method: InitialKnapsackMethod,
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
        if self.initial_method == InitialKnapsackMethod.GREEDY:
            greedy_solver = GreedyBest(
                self.problem, params_objective_function=self.params_objective_function
            )
            return greedy_solver.solve()
        else:
            solution = self.problem.get_dummy_solution()
            fit = self.aggreg_sol(solution)
            return ResultStorage(
                list_solution_fits=[(solution, fit)],
                best_solution=solution,
                mode_optim=self.params_objective_function.sense_function,
            )


class ConstraintHandlerKnapsack(ConstraintHandler):
    def __init__(self, problem: KnapsackModel, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self, milp_solver: MilpSolver, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:
        if not isinstance(milp_solver, LPKnapsack):
            raise ValueError("milp_solver must a LPKnapsack for this constraint.")
        if milp_solver.model is None:
            milp_solver.init_model()
            if milp_solver.model is None:
                raise RuntimeError(
                    "milp_solver.model must be not None after calling milp_solver.init_model()"
                )
        subpart_item = set(
            random.sample(
                range(self.problem.nb_items),
                int(self.fraction_to_fix * self.problem.nb_items),
            )
        )
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, KnapsackSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a KnapsackSolution."
            )
        dict_f_fixed = {}
        dict_f_start = {}
        start = []
        for c in range(self.problem.nb_items):
            dict_f_start[c] = current_solution.list_taken[c]
            if c in subpart_item:
                dict_f_fixed[c] = dict_f_start[c]
        x_var = milp_solver.variable_decision["x"]
        lns_constraint: Dict[Hashable, Any] = {}
        for key in x_var:
            start += [(x_var[key], dict_f_start[key])]
            if key in subpart_item:
                lns_constraint[key] = milp_solver.model.add_constr(
                    x_var[key] == dict_f_start[key], name=str(key)
                )
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
        milp_solver.model.start = start
        return lns_constraint

    def remove_constraints_from_previous_iteration(
        self, milp_solver: MilpSolver, previous_constraints: Mapping[Hashable, Any]
    ) -> None:
        if not isinstance(milp_solver, LPKnapsack):
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
