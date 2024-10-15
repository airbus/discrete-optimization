#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Union

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers_map import solve
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    VariantMultiskillRcpspProblem,
    VariantMultiskillRcpspSolution,
)


class RcpspBasedMultiskillRcpspSolver(SolverDO):
    problem: Union[MultiskillRcpspProblem, VariantMultiskillRcpspProblem]

    def __init__(
        self,
        problem: Union[MultiskillRcpspProblem, VariantMultiskillRcpspProblem],
        method,
        params_objective_function: ParamsObjectiveFunction = None,
        **args
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.problem_rcpsp = problem.build_multimode_rcpsp_calendar_representative()
        self.method = method
        self.args_solve = args
        self.args_solve["params_objective_function"] = self.params_objective_function

    def solve(self, **kwargs):
        res_storage = solve(
            method=self.method, problem=self.problem_rcpsp, **self.args_solve
        )
        list_solution_fits = []
        for s, fit in res_storage:
            sol: RcpspSolution = s
            mode = sol.rcpsp_modes
            modes = {i + 2: mode[i] for i in range(len(mode))}
            modes[self.problem.source_task] = 1
            modes[self.problem.sink_task] = 1
            ms_rcpsp_solution = VariantMultiskillRcpspSolution(
                problem=self.problem,
                priority_list_task=sol.rcpsp_permutation,
                modes_vector=sol.rcpsp_modes,
                priority_worker_per_task=[
                    [w for w in self.problem.employees]
                    for i in range(self.problem.n_jobs_non_dummy)
                ],
            )
            list_solution_fits += [
                (ms_rcpsp_solution, self.aggreg_from_sol(ms_rcpsp_solution))
            ]
        return self.create_result_storage(
            list_solution_fits,
        )
