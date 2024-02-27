#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Dict, List, Optional

from ortools.sat.python.cp_model import (
    CpModel,
    CpSolver,
    IntVar,
    VarArrayAndObjectiveSolutionPrinter,
)

from discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack
from discrete_optimization.rcpsp.solver.cpsat_solver import cpstatus_to_dostatus

logger = logging.getLogger(__name__)


class CPSatKnapsackSolver(CPSolver, SolverKnapsack):
    def __init__(
        self,
        problem: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem, params_objective_function=params_objective_function
        )
        self.model: Optional[CpModel] = None
        self.variables: Dict[str, List[IntVar]] = {}

    def init_model(self, **args: Any) -> None:
        model = CpModel()
        variables = [
            model.NewBoolVar(name=f"x_{i}") for i in range(self.problem.nb_items)
        ]
        model.Add(
            sum(
                [
                    variables[i] * self.problem.list_items[i].weight
                    for i in range(self.problem.nb_items)
                ]
            )
            <= self.problem.max_capacity
        )
        model.Maximize(
            sum(
                [
                    variables[i] * self.problem.list_items[i].value
                    for i in range(self.problem.nb_items)
                ]
            )
        )
        self.model = model
        self.variables["taken"] = variables

    def retrieve_solution(self, solver: CpSolver) -> ResultStorage:
        taken = [int(solver.Value(var)) for var in self.variables["taken"]]
        sol = KnapsackSolution(problem=self.problem, list_taken=taken)
        fit = self.aggreg_sol(sol)
        return ResultStorage(
            [(sol, fit)], mode_optim=self.params_objective_function.sense_function
        )

    def solve(
        self, parameters_cp: Optional[ParametersCP] = None, **args: Any
    ) -> ResultStorage:
        if self.model is None:
            self.init_model(**args)
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = parameters_cp.time_limit
        solver.parameters.num_workers = parameters_cp.nb_process
        callback = VarArrayAndObjectiveSolutionPrinter(variables=[])
        status = solver.Solve(self.model, callback)
        self.status_solver = cpstatus_to_dostatus(status_from_cpsat=status)
        logger.info(
            f"Solver finished, status={solver.StatusName(status)}, objective = {solver.ObjectiveValue()},"
            f"best obj bound = {solver.BestObjectiveBound()}"
        )
        return self.retrieve_solution(solver=solver)
