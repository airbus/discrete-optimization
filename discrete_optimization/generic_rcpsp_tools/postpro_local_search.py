#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_rcpsp_tools.ls_solver import (
    LS_SOLVER,
    LS_RCPSP_Solver,
)
from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_cp import PostProcessSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPSolution_Preemptive,
    schedule_solution_preemptive_to_variant,
)


class PostProLocalSearch(PostProcessSolution):
    def __init__(
        self,
        problem: ANY_RCPSP,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs
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
        self.dict_params = kwargs

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        solver = LS_RCPSP_Solver(problem=self.problem, ls_solver=LS_SOLVER.SA)
        s = result_storage.get_best_solution().copy()
        if isinstance(s, MS_RCPSPSolution_Preemptive):
            s = schedule_solution_preemptive_to_variant(s)
        if self.problem != s.problem:
            s.change_problem(self.problem)
        result_store = solver.solve(
            nb_iteration_max=self.dict_params.get("nb_iteration_max", 200),
            init_solution=s,
        )
        solution, f = result_store.get_last_best_solution()
        result_storage.list_solution_fits += [
            (solution, self.aggreg_from_sol(solution))
        ]
        return result_storage
