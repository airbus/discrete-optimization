#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Union

import numpy as np

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_preemptive import (
    PreemptiveRcpspProblem,
    PreemptiveRcpspSolution,
)
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp_multiskill.multiskill_to_rcpsp import MultiSkillToRcpsp
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    PreemptiveMultiskillRcpspSolution,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import (
    CpMultiskillRcpspSolver,
    CpPreemptiveMultiskillRcpspSolver,
    CpSolverName,
    stick_to_solution,
    stick_to_solution_preemptive,
)

logger = logging.getLogger(__name__)


class MultimodeTranspositionMultiskillRcpspSolver(SolverDO):
    problem: MultiskillRcpspProblem

    def __init__(
        self,
        problem: MultiskillRcpspProblem,
        multimode_problem: Union[RcpspProblem, PreemptiveRcpspProblem] = None,
        worker_type_to_worker: dict[str, set[Union[str, int]]] = None,
        params_objective_function: ParamsObjectiveFunction = None,
        solver_multimode_rcpsp: SolverDO = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.multimode_problem = multimode_problem
        self.worker_type_to_worker = worker_type_to_worker
        self.solver_multimode_rcpsp = solver_multimode_rcpsp

    def solve(self, **kwargs) -> ResultStorage:
        if self.multimode_problem is None or self.worker_type_to_worker is None:
            algo = MultiSkillToRcpsp(self.problem)
            rcpsp_problem = algo.construct_rcpsp_by_worker_type(
                limit_number_of_mode_per_task=True,
                max_number_of_mode=3,
                check_resource_compliance=True,
            )
            self.multimode_problem = rcpsp_problem
            self.worker_type_to_worker = algo.worker_type_to_worker
        result_store = self.solver_multimode_rcpsp.solve(**kwargs)
        solution, fit = result_store.get_best_solution_fit()
        solution: PreemptiveRcpspSolution = solution
        res = rebuild_multiskill_solution_cp_based(
            multiskill_rcpsp_problem=self.problem,
            multimode_rcpsp_problem=self.multimode_problem,
            worker_type_to_worker=self.worker_type_to_worker,
            solution_rcpsp=solution,
        )
        return res


def rebuild_multiskill_solution(
    multiskill_rcpsp_problem: MultiskillRcpspProblem,
    multimode_rcpsp_problem: Union[RcpspProblem, PreemptiveRcpspProblem],
    worker_type_to_worker: dict[str, set[Union[str, int]]],
    solution_rcpsp: Union[RcpspSolution, PreemptiveRcpspSolution],
):
    new_horizon = multimode_rcpsp_problem.horizon
    resource_avail_in_time = {}
    for res in multimode_rcpsp_problem.resources_list:
        if multimode_rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = multimode_rcpsp_problem.resources[res][
                : new_horizon + 1
            ]
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, multimode_rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    worker_avail_in_time = {}
    for i in multiskill_rcpsp_problem.employees:
        worker_avail_in_time[i] = np.array(
            multiskill_rcpsp_problem.employees[i].calendar_employee[: new_horizon + 1],
            dtype=np.bool_,
        )
    rcpsp_schedule = solution_rcpsp.rcpsp_schedule
    employee_usage = {}
    modes_dict = multimode_rcpsp_problem.build_mode_dict(solution_rcpsp.rcpsp_modes)
    sorted_tasks = sorted(rcpsp_schedule, key=lambda x: solution_rcpsp.get_end_time(x))
    for task in sorted_tasks:
        employee_usage[task] = {}
        ressource_requirements = multimode_rcpsp_problem.mode_details[task][
            modes_dict[task]
        ]
        non_zeros_ressource_requirements = set(
            [
                k
                for k in ressource_requirements
                if k in worker_type_to_worker and ressource_requirements[k] > 0
            ]
        )
        if len(non_zeros_ressource_requirements) >= 1:
            active_times = solution_rcpsp.get_active_time(task)
            for k in non_zeros_ressource_requirements:
                number_worker = ressource_requirements[k]
                workers_available = [
                    w
                    for w in worker_type_to_worker[k]
                    if all(worker_avail_in_time[w][i] for i in active_times)
                ]
                if len(workers_available) >= number_worker:
                    wavail = workers_available[:number_worker]
                    skills_needed_by_task = [
                        s
                        for s in multiskill_rcpsp_problem.mode_details[task][1]
                        if s in multiskill_rcpsp_problem.skills_set
                        and multiskill_rcpsp_problem.mode_details[task][1][s] > 0
                    ]
                    non_zeros = multiskill_rcpsp_problem.employees[
                        wavail[0]
                    ].get_non_zero_skills()
                    skills_interest = [
                        s for s in non_zeros if s in skills_needed_by_task
                    ]
                    employee_usage[task].update(
                        {emp: set(skills_interest) for emp in wavail}
                    )
                    for i in active_times:
                        for w in wavail:
                            worker_avail_in_time[w][i] = False
                else:
                    if isinstance(solution_rcpsp, PreemptiveRcpspSolution):
                        for s, e in zip(
                            solution_rcpsp.rcpsp_schedule[task]["starts"],
                            solution_rcpsp.rcpsp_schedule[task]["ends"],
                        ):
                            at = range(s, e)
                            workers_available = [
                                w
                                for w in worker_type_to_worker[k]
                                if all(worker_avail_in_time[w][i] for i in at)
                            ]
                            if len(workers_available) >= number_worker:
                                wavail = workers_available[:number_worker]
                                skills_needed_by_task = [
                                    s
                                    for s in multiskill_rcpsp_problem.mode_details[
                                        task
                                    ][1]
                                    if s in multiskill_rcpsp_problem.skills_set
                                    and multiskill_rcpsp_problem.mode_details[task][1][
                                        s
                                    ]
                                    > 0
                                ]
                                non_zeros = multiskill_rcpsp_problem.employees[
                                    wavail[0]
                                ].get_non_zero_skills()
                                skills_interest = [
                                    s for s in non_zeros if s in skills_needed_by_task
                                ]
                                for emp in wavail:
                                    if emp not in employee_usage[task]:
                                        employee_usage[task][emp] = {
                                            "skills": [],
                                            "times": [],
                                        }
                                    employee_usage[task][emp]["skills"] += [
                                        set(skills_interest)
                                    ]
                                    employee_usage[task][emp]["times"] += [(s, e)]
                                for i in at:
                                    for w in wavail:
                                        worker_avail_in_time[w][i] = False
                            else:
                                logger.warning("Problem finding a worker")

    if isinstance(solution_rcpsp, PreemptiveRcpspSolution):
        return PreemptiveMultiskillRcpspSolution(
            problem=multiskill_rcpsp_problem,
            modes={task: 1 for task in multiskill_rcpsp_problem.tasks_list},
            employee_usage=employee_usage,
            schedule=rcpsp_schedule,
        )
    else:
        return MultiskillRcpspSolution(
            problem=multiskill_rcpsp_problem,
            modes={task: 1 for task in multiskill_rcpsp_problem.tasks_list},
            employee_usage=employee_usage,
            schedule=rcpsp_schedule,
        )


def rebuild_multiskill_solution_cp_based(
    multiskill_rcpsp_problem: MultiskillRcpspProblem,
    multimode_rcpsp_problem: Union[RcpspProblem, PreemptiveRcpspProblem],
    worker_type_to_worker: dict[str, set[Union[str, int]]],
    solution_rcpsp: Union[RcpspSolution, PreemptiveRcpspSolution],
):
    if isinstance(solution_rcpsp, RcpspSolution):
        model = CpMultiskillRcpspSolver(
            problem=multiskill_rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED
        )
        model.init_model(
            add_calendar_constraint_unit=False,
            fake_tasks=True,
            one_ressource_per_task=False,
            exact_skills_need=False,
            output_type=True,
        )
        strings = stick_to_solution(solution_rcpsp, model)
        for s in strings:
            model.instance.add_string(s)
    else:
        model = CpPreemptiveMultiskillRcpspSolver(
            problem=multiskill_rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED
        )
        model.init_model(
            add_calendar_constraint_unit=False,
            fake_tasks=True,
            exact_skills_need=False,
            one_ressource_per_task=False,
            output_type=True,
            nb_preemptive=10,
            unit_usage_preemptive=True,
            max_preempted=100,
        )
        strings = stick_to_solution_preemptive(solution_rcpsp, model)
        for s in strings:
            model.instance.add_string(s)
    result_store = model.solve(time_limit=3600)
    return result_store
