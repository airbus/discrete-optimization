#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Iterable

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_evaluate_function_aggregated,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_mip import (
    InitialSolution,
    PymipConstraintHandler,
)
from discrete_optimization.generic_tools.lp_tools import MilpSolverName
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import (
    InitialMethodRCPSP,
    InitialSolutionRCPSP,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution,
)
from discrete_optimization.rcpsp_multiskill.solvers.lp_model import LP_Solver_MRSCPSP


class InitialSolutionMS_RCPSP(InitialSolution):
    def __init__(
        self,
        problem: MS_RCPSPModel,
        params_objective_function: ParamsObjectiveFunction = None,
        initial_method: InitialMethodRCPSP = InitialMethodRCPSP.PILE,
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
        init_solution = InitialSolutionRCPSP(
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
            sol: RCPSPSolution = sol
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


class ConstraintHandlerFixStartTime(PymipConstraintHandler):
    def __init__(self, problem: MS_RCPSPModel, fraction_fix_start_time: float = 0.9):
        self.problem = problem
        self.fraction_fix_start_time = fraction_fix_start_time

    def adding_constraint_from_results_store(
        self, solver: LP_Solver_MRSCPSP, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:

        nb_jobs = self.problem.nb_tasks
        lns_constraints = []
        current_solution, fit = result_storage.get_best_solution_fit()

        start = []
        for j in current_solution.schedule:
            start_time_j = current_solution.schedule[j]["start_time"]
            mode = current_solution.modes[j]
            start += [(solver.start_times_task[j], start_time_j)]
            start += [(solver.modes[j][mode], 1)]
            for m in solver.modes[j]:
                start += [(solver.modes[j][m], 1 if mode == m else 0)]
        solver.model.start = start
        # Fix start time for a subset of task.
        jobs_to_fix = set(
            random.sample(
                list(current_solution.rcpsp_schedule),
                int(self.fraction_fix_start_time * nb_jobs),
            )
        )
        for job_to_fix in jobs_to_fix:
            lns_constraints.append(
                solver.model.add_constr(
                    solver.start_times_task[job_to_fix]
                    - current_solution.schedule[job_to_fix]["start_time"]
                    == 0
                )
            )
        if solver.lp_solver == MilpSolverName.GRB:
            solver.model.solver.update()
        return lns_constraints


class ConstraintHandlerStartTimeIntervalMRCPSP(PymipConstraintHandler):
    def __init__(
        self,
        problem: MS_RCPSPModel,
        fraction_to_fix: float = 0.9,
        minus_delta: int = 2,
        plus_delta: int = 2,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta

    def adding_constraint_from_results_store(
        self, solver: LP_Solver_MRSCPSP, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:
        current_solution: MS_RCPSPSolution = result_storage.get_best_solution()
        start = []
        for j in current_solution.schedule:
            start_time_j = current_solution.schedule[j]["start_time"]
            mode = current_solution.modes[j]
            start += [(solver.start_times_task[j], start_time_j)]
            start += [(solver.modes[j][mode], 1)]
            for m in solver.modes[j]:
                start += [(solver.modes[j][m], 1 if mode == m else 0)]
        solver.model.start = start
        lns_constraints = []
        max_time = max(
            [
                current_solution.schedule[x]["end_time"]
                for x in current_solution.schedule
            ]
        )
        last_jobs = [
            x
            for x in current_solution.schedule
            if current_solution.schedule[x]["end_time"] >= max_time - 5
        ]
        nb_jobs = self.problem.nb_tasks
        jobs_to_fix = set(
            random.sample(
                list(current_solution.schedule), int(self.fraction_to_fix * nb_jobs)
            )
        )
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        for job in jobs_to_fix:
            start_time_j = current_solution.schedule[job]["start_time"]
            min_st = max(start_time_j - self.minus_delta, 0)
            max_st = min(start_time_j + self.plus_delta, max_time)
            lns_constraints.append(
                solver.model.add_constr(solver.start_times_task[job] <= max_st)
            )
            lns_constraints.append(
                solver.model.add_constr(solver.start_times_task[job] >= min_st)
            )
        if solver.lp_solver == MilpSolverName.GRB:
            solver.model.solver.update()
        return lns_constraints
