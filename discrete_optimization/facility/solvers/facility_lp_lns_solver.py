#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from enum import Enum
from typing import Any, Dict, Hashable, Mapping

import mip

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.solvers.facility_lp_solver import (
    LP_Facility_Solver_PyMip,
    MilpSolver,
    MilpSolverName,
)
from discrete_optimization.facility.solvers.greedy_solvers import (
    GreedySolverFacility,
    ResultStorage,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_mip import (
    ConstraintHandler,
    InitialSolution,
)

logger = logging.getLogger(__name__)


class InitialFacilityMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialFacilitySolution(InitialSolution):
    """Initial solution provider for lns algorithm.

    Attributes:
        problem (FacilityProblem): input coloring problem
        initial_method (InitialFacilityMethod): the method to use to provide the initial solution.
    """

    def __init__(
        self,
        problem: FacilityProblem,
        initial_method: InitialFacilityMethod,
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
        if self.initial_method == InitialFacilityMethod.GREEDY:
            greedy_solver = GreedySolverFacility(
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


class ConstraintHandlerFacility(ConstraintHandler):
    """Constraint builder used in LNS+LP for coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of allocation of customer to
    facility.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    def __init__(
        self,
        problem: FacilityProblem,
        fraction_to_fix: float = 0.9,
        skip_first_iter: bool = True,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0
        self.skip_first_iter = skip_first_iter

    def adding_constraint_from_results_store(
        self, milp_solver: MilpSolver, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:
        if not isinstance(milp_solver, LP_Facility_Solver_PyMip):
            raise ValueError(
                "milp_solver must a LP_Facility_Solver_PyMip for this constraint."
            )
        if milp_solver.model is None:  # for mypy
            milp_solver.init_model()
            if milp_solver.model is None:
                raise RuntimeError(
                    "milp_solver.model must be not None after calling milp_solver.init_model()."
                )
        if self.iter == 0 and self.skip_first_iter:
            logger.debug(
                f"Dummy : {self.problem.evaluate(result_storage.get_best_solution_fit()[0])}"  # type: ignore
            )
            self.iter += 1
            return {}
        subpart_customer = set(
            random.sample(
                range(self.problem.customer_count),
                int(self.fraction_to_fix * self.problem.customer_count),
            )
        )
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, FacilitySolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a FacilitySolution."
            )
        dict_f_fixed = {}
        dict_f_start = {}
        start = []
        for c in range(self.problem.customer_count):
            dict_f_start[c] = current_solution.facility_for_customers[c]
            if c in subpart_customer:
                dict_f_fixed[c] = dict_f_start[c]
        x_var = milp_solver.variable_decision["x"]
        lns_constraint: Dict[Hashable, Any] = {}
        for key in x_var:
            f, c = key
            if f == dict_f_start[c]:
                if isinstance(x_var[f, c], mip.Var):
                    start += [(x_var[f, c], 1)]
            else:
                if isinstance(x_var[f, c], mip.Var):
                    start += [(x_var[f, c], 0)]
            if c in dict_f_fixed:
                if f == dict_f_fixed[c]:
                    if isinstance(x_var[f, c], mip.Var):
                        lns_constraint[(f, c)] = milp_solver.model.add_constr(
                            x_var[key] == 1, name=str((f, c))
                        )
                else:
                    if isinstance(x_var[f, c], mip.Var):
                        lns_constraint[(f, c)] = milp_solver.model.add_constr(
                            x_var[key] == 0, name=str((f, c))
                        )
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
        milp_solver.model.start = start
        return lns_constraint

    def remove_constraints_from_previous_iteration(
        self, milp_solver: MilpSolver, previous_constraints: Mapping[Hashable, Any]
    ) -> None:
        if not isinstance(milp_solver, LP_Facility_Solver_PyMip):
            raise ValueError(
                "milp_solver must a LP_Facility_Solver_PyMip for this constraint."
            )
        if milp_solver.model is None:  # for mypy
            milp_solver.init_model()
            if milp_solver.model is None:
                raise RuntimeError(
                    "milp_solver.model must be not None after calling milp_solver.init_model()."
                )
        milp_solver.model.remove(list(previous_constraints.values()))
        if milp_solver.milp_solver_name == MilpSolverName.GRB:
            milp_solver.model.solver.update()
