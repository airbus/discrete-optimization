#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_evaluate_function_aggregated,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_mip import InitialSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.lns_lp import (
    InitialRcpspMethod,
    InitialRcpspSolution,
)
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspProblem


class InitialMultiskillRcpspSolution(InitialSolution):
    def __init__(
        self,
        problem: MultiskillRcpspProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        initial_method: InitialRcpspMethod = InitialRcpspMethod.PILE,
    ):
        self.problem = problem
        self.params_objective_function = params_objective_function
        if self.params_objective_function is None:
            self.params_objective_function = get_default_objective_setup(
                problem=self.problem
            )
        self.aggreg, _ = build_evaluate_function_aggregated(
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )
        self.initial_method = initial_method

    def get_starting_solution(self) -> ResultStorage:
        multi_skill_rcpsp = self.problem.build_multimode_rcpsp_calendar_representative()
        init_solution = InitialRcpspSolution(
            problem=multi_skill_rcpsp,
            params_objective_function=self.params_objective_function,
            initial_method=self.initial_method,
        )
        s = init_solution.get_starting_solution()
        list_solution_fits = []
        class_solution = self.problem.get_solution_type()
        if class_solution is None:
            class_solution = self.problem.to_variant_model().get_solution_type()
        for sol, fit in s:
            sol: RcpspSolution = sol
            mode = sol.rcpsp_modes
            modes = {
                multi_skill_rcpsp.tasks_list_non_dummy[i]: mode[i]
                for i in range(len(mode))
            }
            modes[self.problem.source_task] = 1
            modes[self.problem.sink_task] = 1
            ms_rcpsp_solution = class_solution(
                problem=self.problem,
                priority_list_task=sol.rcpsp_permutation,
                modes_vector=sol.rcpsp_modes,
                priority_worker_per_task=[
                    [w for w in self.problem.employees]
                    for i in range(self.problem.n_jobs_non_dummy)
                ],
            )
            list_solution_fits += [(ms_rcpsp_solution, self.aggreg(ms_rcpsp_solution))]
        return ResultStorage(
            mode_optim=self.params_objective_function.sense_function,
            list_solution_fits=list_solution_fits,
        )
