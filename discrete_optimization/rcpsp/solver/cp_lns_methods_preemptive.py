#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import math
import random
from enum import Enum
from typing import Any, Hashable, Iterable, List, Optional, Set, Union

import networkx as nx
import numpy as np
from minizinc import Instance

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    GraphRCPSP,
    GraphRCPSPSpecialConstraints,
)
from discrete_optimization.generic_tools.cp_tools import CPSolver, SignEnum
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_cp import ConstraintHandler
from discrete_optimization.generic_tools.lns_mip import PostProcessSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    PartialSolutionPreemptive,
    RCPSPModelPreemptive,
    RCPSPSolutionPreemptive,
)
from discrete_optimization.rcpsp.rcpsp_solution_utils import (
    get_max_time_solution,
    get_tasks_ending_between_two_times,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN_PREEMMPTIVE,
    CP_RCPSP_MZN,
    CP_RCPSP_MZN_PREEMMPTIVE,
)
from discrete_optimization.rcpsp.solver.ls_solver import LS_SOLVER, LS_RCPSP_Solver
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraints,
    RCPSPModelSpecialConstraintsPreemptive,
    RCPSPSolutionSpecialPreemptive,
    compute_constraints_details,
)

logger = logging.getLogger(__name__)


def last_opti_solution(last_result_store: ResultStorage):
    current_solution, fit = next(
        (
            last_result_store.list_solution_fits[j]
            for j in range(len(last_result_store.list_solution_fits))
            if "opti_from_cp"
            in last_result_store.list_solution_fits[j][0].__dict__.keys()
        ),
        (None, None),
    )
    if current_solution is None or fit != last_result_store.get_best_solution_fit()[1]:
        current_solution, fit = last_result_store.get_last_best_solution()
    return current_solution


class PostProLeftShift(PostProcessSolution):
    def __init__(
        self,
        problem: RCPSPModelPreemptive,
        params_objective_function: ParamsObjectiveFunction = None,
        do_ls: bool = False,
        **kwargs
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
        self.graph = self.problem.compute_graph()
        if isinstance(
            problem,
            (RCPSPModelSpecialConstraintsPreemptive, RCPSPModelSpecialConstraints),
        ):
            self.graph_rcpsp = GraphRCPSPSpecialConstraints(problem=self.problem)
            self.special_constraints = True
        else:
            self.graph_rcpsp = GraphRCPSP(problem=self.problem)
            self.special_constraints = False
        self.successors = {
            n: nx.algorithms.descendants(self.graph.graph_nx, n)
            for n in self.graph.graph_nx.nodes()
        }
        self.predecessors = {
            n: nx.algorithms.descendants(self.graph.graph_nx, n)
            for n in self.graph.graph_nx.nodes()
        }
        self.immediate_predecessors = {
            n: self.graph.get_predecessors(n) for n in self.graph.nodes_name
        }
        self.do_ls = do_ls
        self.dict_params = kwargs

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        new_solution = sgs_variant(
            solution=last_opti_solution(result_storage),
            problem=self.problem,
            predecessors_dict=self.immediate_predecessors,
        )
        fit = self.aggreg_from_sol(new_solution)
        result_storage.add_solution(new_solution, fit)
        if self.do_ls:
            solver = LS_RCPSP_Solver(model=self.problem, ls_solver=LS_SOLVER.SA)
            s = result_storage.get_best_solution().copy()
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


def sgs_variant(
    solution: RCPSPSolutionPreemptive, problem: RCPSPModelPreemptive, predecessors_dict
):
    new_proposed_schedule = {}
    new_horizon = min(solution.get_end_time(problem.sink_task) * 3, problem.horizon)
    resource_avail_in_time = {}
    modes_dict = problem.build_mode_dict(solution.rcpsp_modes)
    for r in problem.resources_list:
        if problem.is_varying_resource():
            resource_avail_in_time[r] = np.copy(problem.resources[r][:new_horizon])
        else:
            resource_avail_in_time[r] = problem.resources[r] * np.ones(new_horizon)
    sorted_tasks = sorted(
        solution.rcpsp_schedule.keys(), key=lambda x: (solution.get_start_time(x), x)
    )
    for task in list(sorted_tasks):
        if len(solution.rcpsp_schedule[task]["starts"]) > 1:
            new_proposed_schedule[task] = {
                "starts": solution.rcpsp_schedule[task]["starts"],
                "ends": solution.rcpsp_schedule[task]["ends"],
            }
            for s, e in zip(
                new_proposed_schedule[task]["starts"],
                new_proposed_schedule[task]["ends"],
            ):
                for res in problem.resources_list:
                    resource_avail_in_time[res][s:e] -= problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
            sorted_tasks.remove(task)

    def is_startable(tt):
        if isinstance(problem, RCPSPModelSpecialConstraintsPreemptive):
            return (
                all(
                    x in new_proposed_schedule
                    for x in problem.special_constraints.dict_start_after_nunit_reverse.get(
                        tt, {}
                    )
                )
                and all(
                    x in new_proposed_schedule
                    for x in problem.special_constraints.dict_start_at_end_reverse.get(
                        tt, {}
                    )
                )
                and all(
                    x in new_proposed_schedule
                    for x in problem.special_constraints.dict_start_at_end_offset_reverse.get(
                        tt, {}
                    )
                )
                and all(t in new_proposed_schedule for t in predecessors_dict[tt])
            )
        return all(t in new_proposed_schedule for t in predecessors_dict[tt])

    while True:
        task = next((t for t in sorted_tasks if is_startable(t)))
        sorted_tasks.remove(task)
        if len(solution.rcpsp_schedule[task]["starts"]) > 1:
            continue
        times_predecessors = [
            new_proposed_schedule[t]["ends"][-1]
            if t in new_proposed_schedule
            else solution.rcpsp_schedule[t]["ends"][-1]
            for t in predecessors_dict[task]
        ]
        if isinstance(problem, RCPSPModelSpecialConstraintsPreemptive):
            times_predecessors += [
                new_proposed_schedule[t]["starts"][0]
                if t in new_proposed_schedule
                else solution.rcpsp_schedule[t]["starts"][0]
                for t in problem.special_constraints.dict_start_together.get(
                    task, set()
                )
            ]
            times_predecessors += [
                new_proposed_schedule[t]["starts"][0]
                + problem.special_constraints.dict_start_after_nunit_reverse.get(task)[
                    t
                ]
                if t in new_proposed_schedule
                else solution.rcpsp_schedule[t]["starts"][0]
                + problem.special_constraints.dict_start_after_nunit_reverse.get(task)[
                    t
                ]
                for t in problem.special_constraints.dict_start_after_nunit_reverse.get(
                    task, {}
                )
            ]
            times_predecessors += [
                new_proposed_schedule[t]["ends"][-1]
                if t in new_proposed_schedule
                else solution.rcpsp_schedule[t]["ends"][-1]
                for t in problem.special_constraints.dict_start_at_end_reverse.get(
                    task, {}
                )
            ]
            times_predecessors += [
                new_proposed_schedule[t]["ends"][-1]
                + problem.special_constraints.dict_start_at_end_offset_reverse.get(
                    task
                )[t]
                if t in new_proposed_schedule
                else solution.rcpsp_schedule[t]["ends"][-1]
                + problem.special_constraints.dict_start_at_end_offset_reverse.get(
                    task
                )[t]
                for t in problem.special_constraints.dict_start_at_end_offset_reverse.get(
                    task, {}
                )
            ]
        if len(times_predecessors) > 0:
            min_time = max(times_predecessors)
        else:
            min_time = 0
        if solution.get_start_time(task) == solution.get_end_time(task):
            new_proposed_schedule[task] = {"starts": [min_time], "ends": [min_time]}
        if len(solution.rcpsp_schedule[task]["starts"]) > 1:
            new_proposed_schedule[task] = {
                "starts": solution.rcpsp_schedule[task]["starts"],
                "ends": solution.rcpsp_schedule[task]["ends"],
            }
        else:
            for t in range(min_time, problem.horizon):
                if all(
                    resource_avail_in_time[res][time]
                    >= problem.mode_details[task][modes_dict[task]].get(res, 0)
                    for res in problem.resources_list
                    for time in range(
                        t, t + problem.mode_details[task][modes_dict[task]]["duration"]
                    )
                ):
                    new_starting_time = t
                    break
            new_proposed_schedule[task] = {
                "starts": [new_starting_time],
                "ends": [
                    new_starting_time
                    + problem.mode_details[task][modes_dict[task]]["duration"]
                ],
            }
            for s, e in zip(
                new_proposed_schedule[task]["starts"],
                new_proposed_schedule[task]["ends"],
            ):
                for res in problem.resources_list:
                    resource_avail_in_time[res][s:e] -= problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
        if len(sorted_tasks) == 0:
            break
    new_solution = RCPSPSolutionSpecialPreemptive(
        problem=problem,
        rcpsp_schedule=new_proposed_schedule,
        rcpsp_schedule_feasible=True,
        rcpsp_modes=solution.rcpsp_modes,
    )
    logger.debug(
        ("New : ", problem.evaluate(new_solution), problem.satisfy(new_solution))
    )
    logger.debug(("Old : ", problem.evaluate(solution), problem.satisfy(solution)))
    return new_solution


def constraints_strings(
    current_solution: RCPSPSolutionPreemptive,
    subtasks: Set[Hashable],
    minus_delta: int,
    plus_delta: int,
    jobs_to_fix: Set[Hashable],
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE],
    constraint_max_time=True,
    minus_delta_2=0,
    plus_delta_2=0,
):
    max_time = get_max_time_solution(solution=current_solution)
    list_strings = []
    for job in subtasks:
        for j in range(len(current_solution.rcpsp_schedule[job]["starts"])):
            start_time_j = current_solution.rcpsp_schedule[job]["starts"][j]
            end_time_j = current_solution.rcpsp_schedule[job]["ends"][j]
            duration_j = end_time_j - start_time_j
            string1Start = cp_solver.constraint_start_time_string_preemptive_i(
                task=job,
                start_time=max(0, start_time_j - minus_delta),
                sign=SignEnum.UEQ,
                part_id=j + 1,
            )
            string2Start = cp_solver.constraint_start_time_string_preemptive_i(
                task=job,
                start_time=min(max_time, start_time_j + plus_delta)
                if constraint_max_time
                else start_time_j + plus_delta,
                sign=SignEnum.LEQ,
                part_id=j + 1,
            )
            string1Dur = cp_solver.constraint_duration_string_preemptive_i(
                task=job,
                duration=max(duration_j - 5, 0),
                sign=SignEnum.UEQ,
                part_id=j + 1,
            )
            string2Dur = cp_solver.constraint_duration_string_preemptive_i(
                task=job, duration=duration_j + 5, sign=SignEnum.LEQ, part_id=j + 1
            )
            list_strings += [string1Dur, string1Start, string2Dur, string2Start]
        for k in range(
            len(current_solution.rcpsp_schedule[job]["starts"]), cp_solver.nb_preemptive
        ):
            string1Dur = cp_solver.constraint_duration_string_preemptive_i(
                task=job, duration=0, sign=SignEnum.EQUAL, part_id=k + 1
            )
            list_strings += [string1Dur]
    for job in jobs_to_fix:
        is_paused = len(current_solution.rcpsp_schedule[job]["starts"]) > 1
        is_paused_str = "true" if is_paused else "false"
        list_strings += [
            "constraint is_paused["
            + str(cp_solver.index_in_minizinc[job])
            + "]=="
            + is_paused_str
            + ";\n"
        ]
    for job in jobs_to_fix:
        if job in subtasks:
            continue
        for j in range(len(current_solution.rcpsp_schedule[job]["starts"])):
            start_time_j = current_solution.rcpsp_schedule[job]["starts"][j]
            end_time_j = current_solution.rcpsp_schedule[job]["ends"][j]
            duration_j = end_time_j - start_time_j
            if minus_delta_2 == 0 and plus_delta_2 == 0:
                string1Start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=start_time_j,
                    sign=SignEnum.EQUAL,
                    part_id=j + 1,
                )
                string1Dur = cp_solver.constraint_duration_string_preemptive_i(
                    task=job, duration=duration_j, sign=SignEnum.EQUAL, part_id=j + 1
                )
                list_strings += [string1Dur, string1Start]
            else:
                string1Start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=max(0, start_time_j - minus_delta_2),
                    sign=SignEnum.UEQ,
                    part_id=j + 1,
                )
                string2Start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=min(max_time, start_time_j + plus_delta_2)
                    if constraint_max_time
                    else start_time_j + plus_delta_2,
                    sign=SignEnum.LEQ,
                    part_id=j + 1,
                )
                string1Dur = cp_solver.constraint_duration_string_preemptive_i(
                    task=job,
                    duration=max(duration_j - 5, 0),
                    sign=SignEnum.UEQ,
                    part_id=j + 1,
                )
                string2Dur = cp_solver.constraint_duration_string_preemptive_i(
                    task=job, duration=duration_j + 5, sign=SignEnum.LEQ, part_id=j + 1
                )
                list_strings += [string1Dur, string1Start, string2Dur, string2Start]
        for k in range(
            len(current_solution.rcpsp_schedule[job]["starts"]), cp_solver.nb_preemptive
        ):
            if minus_delta_2 == 0 and plus_delta_2 == 0:
                string1Start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=current_solution.rcpsp_schedule[job]["ends"][-1],
                    sign=SignEnum.EQUAL,
                    part_id=k + 1,
                )
                list_strings += [string1Start]
            else:
                string1Start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=current_solution.rcpsp_schedule[job]["ends"][-1]
                    - minus_delta_2,
                    sign=SignEnum.UEQ,
                    part_id=k + 1,
                )
                string2Start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=min(
                        max_time,
                        current_solution.rcpsp_schedule[job]["ends"][-1] + plus_delta_2,
                    )
                    if constraint_max_time
                    else current_solution.rcpsp_schedule[job]["ends"][-1]
                    + plus_delta_2,
                    sign=SignEnum.LEQ,
                    part_id=k + 1,
                )
                list_strings += [string2Start, string1Start]
            string1Dur = cp_solver.constraint_duration_string_preemptive_i(
                task=job, duration=0, sign=SignEnum.EQUAL, part_id=k + 1
            )
            list_strings += [string1Dur]

    return list_strings


class NeighborFixStart(ConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModelPreemptive,
        fraction_to_fix: float = 0.9,
        delta_time_from_makepan_to_not_fix: int = 5,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.delta_time_from_makepan_to_not_fix = delta_time_from_makepan_to_not_fix

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        current_solution, fit = result_storage.get_best_solution_fit()
        current_solution: RCPSPSolutionPreemptive = current_solution
        max_time = get_max_time_solution(solution=current_solution)
        nb_jobs = self.problem.n_jobs
        jobs_to_fix = set(
            random.sample(
                current_solution.rcpsp_schedule.keys(),
                int(self.fraction_to_fix * nb_jobs),
            )
        )
        subtasks = set(
            [s for s in current_solution.rcpsp_schedule if s not in jobs_to_fix]
        )
        list_strings = constraints_strings(
            current_solution=current_solution,
            subtasks=subtasks,
            minus_delta=max_time,
            plus_delta=max_time,
            jobs_to_fix=jobs_to_fix,
            cp_solver=cp_solver,
        )
        for s in list_strings:
            child_instance.add_string(s)
        return list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


class MethodSubproblem(Enum):
    BLOCK_TIME = 0
    BLOCK_TIME_AND_PREDECESSORS = 1


def intersect(i1, i2):
    if i2[0] >= i1[1] or i1[0] >= i2[1]:
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]


class NeighborFixStartSubproblem(ConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModelPreemptive,
        nb_cut_part: int = 20,
        fraction_size_subproblem: float = 0.1,
        method: MethodSubproblem = MethodSubproblem.BLOCK_TIME,
    ):
        self.problem = problem
        self.fraction_size_subproblem = fraction_size_subproblem
        self.nb_jobs_subproblem = int(
            self.fraction_size_subproblem * self.problem.n_jobs
        )
        self.method = method
        self.graph = self.problem.compute_graph()
        self.graph_nx = self.graph.graph_nx
        if isinstance(
            problem,
            (RCPSPModelSpecialConstraintsPreemptive, RCPSPModelSpecialConstraints),
        ):
            self.graph_rcpsp = GraphRCPSPSpecialConstraints(problem=self.problem)
            self.special_constraints = True
        else:
            self.graph_rcpsp = GraphRCPSP(problem=self.problem)
            self.special_constraints = False
        self.nb_cut_part = nb_cut_part
        self.current_sub_part = 0

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is not None:
            current_solution, fit = next(
                (
                    last_result_store.list_solution_fits[j]
                    for j in range(len(last_result_store.list_solution_fits))
                    if "opti_from_cp"
                    in last_result_store.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        else:
            current_solution, fit = next(
                (
                    result_storage.list_solution_fits[j]
                    for j in range(len(result_storage.list_solution_fits))
                    if "opti_from_cp"
                    in result_storage.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        if current_solution is None or fit != result_storage.get_best_solution_fit()[1]:
            current_solution, fit = result_storage.get_last_best_solution()
        current_solution: RCPSPSolutionPreemptive = current_solution
        method = random.choices([0, 1, 2], weights=[0.5, 0.1, 0.4], k=1)[0]
        if method == 0:
            subtasks = create_subproblem_cut_time(
                current_solution=current_solution, neighbor_fix_problem=self
            )
        elif method == 1:
            subtasks = create_subproblems_random_and_predecessors(
                current_solution=current_solution, neighbor_fix_problem=self
            )
        elif method == 2:
            subtasks = create_subproblems_problems(
                current_solution=current_solution, neighbor_fix_problem=self
            )
        subtasks = set(subtasks)
        evaluation = self.problem.evaluate(current_solution)
        if evaluation["constraint_penalty"] == 0:
            list_strings = constraints_strings(
                current_solution=current_solution,
                subtasks=subtasks,
                plus_delta=6000,
                minus_delta=6000,
                plus_delta_2=1,
                minus_delta_2=1,
                jobs_to_fix=set(self.problem.tasks_list),
                cp_solver=cp_solver,
                constraint_max_time=True,
            )
        else:
            list_strings = constraints_strings(
                current_solution=current_solution,
                subtasks=subtasks,
                plus_delta=6000,
                minus_delta=6000,
                plus_delta_2=400,
                minus_delta_2=400,
                jobs_to_fix=set(self.problem.tasks_list),
                cp_solver=cp_solver,
                constraint_max_time=False,
            )
        for s in list_strings:
            child_instance.add_string(s)
        child_instance.add_string(
            "constraint sec_objective<="
            + str(100 * evaluation["constraint_penalty"])
            + ";\n"
        )
        if evaluation["constraint_penalty"] > 0:
            strings = cp_solver.constraint_objective_max_time_set_of_jobs(
                [self.problem.sink_task]
            )
        else:
            strings = cp_solver.constraint_objective_max_time_set_of_jobs(subtasks)
        for s in strings:
            child_instance.add_string(s)
            list_strings += [s]
        self.current_sub_part = self.current_sub_part % self.nb_cut_part
        return list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


def create_subproblem_cut_time(
    current_solution, neighbor_fix_problem: NeighborFixStartSubproblem, subtasks=None
):
    nb_job_sub = math.ceil(
        neighbor_fix_problem.problem.n_jobs / neighbor_fix_problem.nb_cut_part
    )
    task_of_interest = sorted(
        current_solution.rcpsp_schedule,
        key=lambda x: current_solution.rcpsp_schedule[x]["ends"][-1],
    )
    task_of_interest = task_of_interest[
        neighbor_fix_problem.current_sub_part
        * nb_job_sub : (neighbor_fix_problem.current_sub_part + 1)
        * nb_job_sub
    ]
    if subtasks is None:
        subtasks = task_of_interest
    else:
        subtasks.update(task_of_interest)
    neighbor_fix_problem.current_sub_part = neighbor_fix_problem.current_sub_part + 1
    return subtasks


def create_subproblems_random_and_predecessors(
    current_solution, neighbor_fix_problem: NeighborFixStartSubproblem, subtasks=None
):
    if subtasks is None:
        subtasks = set()
        len_subtask = 0
    else:
        len_subtask = len(subtasks)
    while len_subtask < neighbor_fix_problem.nb_jobs_subproblem:
        random_pick = random.choice(neighbor_fix_problem.problem.tasks_list)
        interval = (
            current_solution.rcpsp_schedule[random_pick]["starts"][0],
            current_solution.rcpsp_schedule[random_pick]["ends"][0],
        )
        task_intersect = [
            t
            for t in current_solution.rcpsp_schedule
            if intersect(
                interval,
                (
                    current_solution.rcpsp_schedule[t]["starts"][0],
                    current_solution.rcpsp_schedule[t]["ends"][0],
                ),
            )
            is not None
        ]
        for k in set(task_intersect):
            task_intersect += list(neighbor_fix_problem.graph.get_predecessors(k)) + [
                l for l in neighbor_fix_problem.graph.get_neighbors(k)
            ]
            if isinstance(
                neighbor_fix_problem.problem, RCPSPModelSpecialConstraintsPreemptive
            ):
                task_intersect += list(
                    neighbor_fix_problem.problem.special_constraints.dict_start_at_end.get(
                        k, {}
                    )
                )
                task_intersect += list(
                    neighbor_fix_problem.problem.special_constraints.dict_start_at_end_reverse.get(
                        k, {}
                    )
                )
        subtasks.update(task_intersect)
        len_subtask = len(subtasks)
    return subtasks


def create_subproblems_problems(
    current_solution, neighbor_fix_problem: NeighborFixStartSubproblem
):
    if neighbor_fix_problem.special_constraints:
        details_constraints = compute_constraints_details(
            solution=current_solution,
            constraints=neighbor_fix_problem.problem.special_constraints,
        )
        sorted_constraints = sorted(details_constraints, key=lambda x: -x[-1])
        random.shuffle(sorted_constraints)
        subtasks = set()
        len_subtasks = 0
        j = 0
        while (
            j <= len(sorted_constraints) - 1
            and len_subtasks < 4 * neighbor_fix_problem.nb_jobs_subproblem
        ):
            t1, t2 = sorted_constraints[j][1], sorted_constraints[j][2]
            subtasks.add(t1)
            subtasks.add(t2)
            subtasks.update(
                neighbor_fix_problem.graph_rcpsp.components_graph_constraints[
                    neighbor_fix_problem.graph_rcpsp.index_components[t1]
                ]
            )
            for c in set(
                neighbor_fix_problem.graph_rcpsp.components_graph_constraints[
                    neighbor_fix_problem.graph_rcpsp.index_components[t1]
                ]
            ):
                subtasks.update(neighbor_fix_problem.graph_rcpsp.get_next_activities(c))
                subtasks.update(
                    neighbor_fix_problem.graph_rcpsp.get_descendants_activities(c)
                )
                subtasks.update(
                    neighbor_fix_problem.graph_rcpsp.get_ancestors_activities(c)
                )
                subtasks.update(neighbor_fix_problem.graph_rcpsp.get_pred_activities(c))
            len_subtasks = len(subtasks)
            j += 1
        if len_subtasks < neighbor_fix_problem.nb_jobs_subproblem:
            subtasks = create_subproblem_cut_time(
                current_solution, neighbor_fix_problem, subtasks
            )
        return subtasks
    else:
        return create_subproblem_cut_time(current_solution, neighbor_fix_problem)


class NeighborFlexibleStart(ConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModelPreemptive,
        fraction_to_fix: float = 0.9,
        minus_delta: int = 2,
        plus_delta: int = 2,
        delta_time_from_makepan_to_not_fix: int = 5,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta
        self.delta_time_from_makepan_to_not_fix = delta_time_from_makepan_to_not_fix

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is not None:
            current_solution, fit = next(
                (
                    last_result_store.list_solution_fits[j]
                    for j in range(len(last_result_store.list_solution_fits))
                    if "opti_from_cp"
                    in last_result_store.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        else:
            current_solution, fit = next(
                (
                    result_storage.list_solution_fits[j]
                    for j in range(len(result_storage.list_solution_fits))
                    if "opti_from_cp"
                    in result_storage.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        if current_solution is None or fit != result_storage.get_best_solution_fit()[1]:
            current_solution, fit = result_storage.get_last_best_solution()
        current_solution: RCPSPSolutionPreemptive = current_solution
        max_time = get_max_time_solution(current_solution)
        last_jobs = get_tasks_ending_between_two_times(
            solution=current_solution,
            time_1=max_time - self.delta_time_from_makepan_to_not_fix,
            time_2=max_time,
        )
        nb_jobs = self.problem.n_jobs
        jobs_to_fix = set(
            random.sample(
                current_solution.rcpsp_schedule.keys(),
                int(self.fraction_to_fix * nb_jobs),
            )
        )
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        list_strings = constraints_strings(
            current_solution,
            subtasks=jobs_to_fix,
            minus_delta=self.minus_delta,
            plus_delta=self.plus_delta,
            jobs_to_fix=set(),
            cp_solver=cp_solver,
        )
        for s in list_strings:
            child_instance.add_string(s)
        return list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


def get_ressource_breaks(
    problem_calendar: RCPSPModelPreemptive, solution: RCPSPSolutionPreemptive
):
    ressources = problem_calendar.resources_list
    ressource_arrays = {}
    ressource_arrays_usage = {}
    makespan = get_max_time_solution(solution=solution)
    for r in ressources:
        ressource_arrays[r] = np.zeros(makespan)
        ressource_arrays_usage[r] = np.zeros((makespan, len(solution.rcpsp_schedule)))
    sorted_keys_schedule = problem_calendar.tasks_list
    modes = problem_calendar.build_mode_dict(solution.rcpsp_modes)
    for ji in range(len(sorted_keys_schedule)):
        j = sorted_keys_schedule[ji]
        for r in problem_calendar.resources_list:
            if problem_calendar.mode_details[j][modes[j]].get(r, 0) == 0:
                continue
            if (
                solution.rcpsp_schedule[j]["starts"][0]
                == solution.rcpsp_schedule[j]["ends"][-1]
            ):
                continue
            for s, e, index in zip(
                solution.rcpsp_schedule[j]["starts"],
                solution.rcpsp_schedule[j]["ends"],
                range(len(solution.rcpsp_schedule[j]["starts"])),
            ):
                ressource_arrays_usage[r][s:e, ji] = index + 1
                ressource_arrays[r][s:e] += problem_calendar.mode_details[j][modes[j]][
                    r
                ]
    index_ressource = {}
    task_concerned = {}
    constraints = {}
    for r in ressource_arrays:
        if problem_calendar.is_varying_resource():
            index = np.argwhere(
                ressource_arrays[r] > problem_calendar.resources[r][:makespan]
            )
        else:
            index = np.argwhere(ressource_arrays[r] > problem_calendar.resources[r])
        index_ressource[r] = index
        task_concerned[r] = [
            j
            for j in range(ressource_arrays_usage[r].shape[1])
            if any(
                ressource_arrays_usage[r][ind[0], j] >= 1
                for ind in index
                if problem_calendar.get_resource_available(r, ind[0]) == 0
            )
        ]
        task_concerned[r] = [sorted_keys_schedule[rr] for rr in task_concerned[r]]
        constraints[r] = {}
        for t in task_concerned[r]:
            current_start = solution.rcpsp_schedule[t]["starts"][0]
            first_possible_start_future = next(
                (
                    st
                    for st in range(current_start, problem_calendar.horizon)
                    if problem_calendar.get_resource_available(r, st)
                    >= problem_calendar.mode_details[t][modes[t]][r]
                ),
                None,
            )
            first_possible_start_before = next(
                (
                    st - problem_calendar.mode_details[t][modes[t]]["duration"] + 1
                    for st in range(solution.rcpsp_schedule[t]["ends"][-1], -1, -1)
                    if problem_calendar.get_resource_available(r, st - 1)
                    >= problem_calendar.mode_details[t][modes[t]][r]
                    and problem_calendar.get_resource_available(
                        r,
                        max(
                            0,
                            st
                            - problem_calendar.mode_details[t][modes[t]]["duration"]
                            + 1,
                        ),
                    )
                    >= problem_calendar.mode_details[t][1][r]
                ),
                None,
            )
            constraints[r][t] = (
                first_possible_start_before,
                first_possible_start_future,
            )
    return index_ressource, constraints


class PostProcessSolutionNonFeasible(PostProcessSolution):
    def __init__(
        self,
        problem_calendar: RCPSPModelPreemptive,
        problem_no_calendar: RCPSPModelPreemptive,
        partial_solution: PartialSolutionPreemptive = None,
        params_objective_function: ParamsObjectiveFunction = None,
        do_ls=True,
        **kwargs
    ):
        self.problem_calendar = problem_calendar
        self.problem_no_calendar = problem_no_calendar
        self.partial_solution = partial_solution
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem_calendar,
            params_objective_function=params_objective_function,
        )
        if self.partial_solution is None:

            def check_solution(problem, solution):
                return True

        else:

            def check_solution(problem, solution):
                start_together = partial_solution.start_together
                start_at_end = partial_solution.start_at_end
                start_at_end_plus_offset = partial_solution.start_at_end_plus_offset
                start_after_nunit = partial_solution.start_after_nunit
                for (t1, t2) in start_together:
                    b = (
                        solution.rcpsp_schedule[t1]["starts"][0]
                        == solution.rcpsp_schedule[t2]["starts"][0]
                    )
                    if not b:
                        return False
                for (t1, t2) in start_at_end:
                    b = (
                        solution.rcpsp_schedule[t2]["starts"][0]
                        == solution.rcpsp_schedule[t1]["ends"][-1]
                    )
                    if not b:
                        return False
                for (t1, t2, off) in start_at_end_plus_offset:
                    b = (
                        solution.rcpsp_schedule[t2]["starts"][0]
                        >= solution.rcpsp_schedule[t1]["ends"][-1] + off
                    )
                    if not b:
                        return False
                for (t1, t2, off) in start_after_nunit:
                    b = (
                        solution.rcpsp_schedule[t2]["starts"][0]
                        >= solution.rcpsp_schedule[t1]["starts"][0] + off
                    )
                    if not b:
                        return False
                return True

        self.check_sol = check_solution
        self.do_ls = do_ls
        self.dict_params = kwargs

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        for sol in list(result_storage.list_solution_fits):
            solution: RCPSPSolutionPreemptive = sol[0]
            if "satisfy" not in solution.__dict__.keys():
                rb, constraints = get_ressource_breaks(self.problem_calendar, solution)
                solution.satisfy = not (any(len(rb[r]) > 0 for r in rb))
                solution.constraints = constraints
            if self.partial_solution is None:
                solution_p = RCPSPSolutionPreemptive(
                    problem=self.problem_calendar,
                    rcpsp_permutation=solution.rcpsp_permutation,
                    rcpsp_modes=solution.rcpsp_modes,
                )
                solution_p.satisfy = self.check_sol(self.problem_calendar, solution_p)
                result_storage.list_solution_fits += [
                    (solution_p, -self.aggreg_from_sol(solution_p))
                ]
        if self.do_ls:
            solver = LS_RCPSP_Solver(
                model=self.problem_calendar, ls_solver=LS_SOLVER.SA
            )
            satisfiable = [
                (s, f) for s, f in result_storage.list_solution_fits if s.satisfy
            ]
            if len(satisfiable) > 0:
                s: RCPSPSolutionPreemptive = max(satisfiable, key=lambda x: x[1])[
                    0
                ].copy()
            else:
                s = result_storage.get_best_solution().copy()
            if self.problem_calendar != s.problem:
                s.change_problem(self.problem_calendar)
            result_store = solver.solve(
                nb_iteration_max=self.dict_params.get("nb_iteration_max", 200),
                init_solution=s,
            )
            for solution, f in result_store.list_solution_fits:
                solution.satisfy = self.check_sol(self.problem_calendar, solution)
                result_storage.list_solution_fits += [
                    (solution, -self.aggreg_from_sol(solution))
                ]
        return result_storage


class ConstraintHandlerAddCalendarConstraint(ConstraintHandler):
    def __init__(
        self,
        problem_calendar: RCPSPModelPreemptive,
        problem_no_calendar: RCPSPModelPreemptive,
        other_constraint: ConstraintHandler,
    ):
        self.problem_calendar = problem_calendar
        self.problem_no_calendar = problem_no_calendar
        self.other_constraint = other_constraint
        self.store_constraints = set()

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is None:
            raise ValueError("This constraint need last_result_store to be not None.")
        list_strings = []
        r = random.random()
        if r <= 0.2:
            solution, fit = last_result_store.get_last_best_solution()
        elif r <= 0.4:
            solution, fit = last_result_store.get_best_solution_fit()
        elif r <= 0.99:
            solution, fit = last_result_store.get_random_best_solution()
        else:
            solution, fit = last_result_store.get_random_solution()
        for s in self.store_constraints:  # we keep trace of already
            child_instance.add_string(s)
        if "satisfy" in solution.__dict__.keys() and solution.satisfy:
            return self.other_constraint.adding_constraint_from_results_store(
                cp_solver, child_instance, result_storage, last_result_store
            )
        ressource_breaks, constraints = get_ressource_breaks(
            self.problem_calendar, solution
        )
        for r in ressource_breaks:
            if len(ressource_breaks[r]) == 0:
                continue
            prev_time = None
            for index_range in range(len(ressource_breaks[r])):
                ind = ressource_breaks[r][index_range][0]
                if (prev_time is None or ind >= prev_time + 20) or (
                    index_range >= 1
                    and ind > ressource_breaks[r][index_range - 1][0] + 1
                ):
                    prev_time = ind
                else:
                    continue
                rq = self.problem_calendar.get_resource_available(r, ind)
                s = cp_solver.constraint_ressource_requirement_at_time_t(
                    time=ind, ressource=r, ressource_number=rq, sign=SignEnum.UEQ
                )
                child_instance.add_string(s)
                list_strings += [s]
        satisfiable = [
            (s, f)
            for s, f in last_result_store.list_solution_fits
            if "satisfy" in s.__dict__.keys() and s.satisfy
        ]
        if len(satisfiable) > 0:
            res = ResultStorage(
                list_solution_fits=satisfiable, mode_optim=result_storage.mode_optim
            )
            self.other_constraint.adding_constraint_from_results_store(
                cp_solver, child_instance, res, last_result_store
            )
        self.store_constraints.update(list_strings)
        return self.store_constraints

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


class OptionNeighbor(Enum):
    MIX_ALL = 0
    MIX_FAST = 1
    MIX_LARGE_NEIGH = 2
    LARGE = 4
    DEBUG = 3


class Params:
    fraction_to_fix: float
    minus_delta: int
    plus_delta: int
    delta_time_from_makepan_to_not_fix: int

    def __init__(
        self,
        fraction_to_fix: float = 0.9,
        minus_delta: int = 2,
        plus_delta: int = 2,
        delta_time_from_makepan_to_not_fix: int = 5,
    ):
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta
        self.delta_time_from_makepan_to_not_fix = delta_time_from_makepan_to_not_fix


class ConstraintHandlerMix(ConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModelPreemptive,
        list_params: List[Params],
        list_proba: List[float],
    ):
        self.problem = problem
        self.list_params = list_params
        self.list_proba = list_proba
        if isinstance(self.list_proba, list):
            self.list_proba = np.array(self.list_proba)
        self.list_proba = self.list_proba / np.sum(self.list_proba)
        self.index_np = np.array(range(len(self.list_proba)), dtype=np.int)
        self.current_iteration = 0
        self.status = {
            i: {"nb_usage": 0, "nb_improvement": 0}
            for i in range(len(self.list_params))
        }
        self.last_index_param = None
        self.last_fitness = None

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN_PREEMMPTIVE],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        new_fitness = result_storage.get_best_solution_fit()[1]
        if self.last_index_param is not None:
            if new_fitness != self.last_fitness:
                self.status[self.last_index_param]["nb_improvement"] += 1
                self.last_fitness = new_fitness
                self.list_proba[self.last_index_param] *= 1.05
                self.list_proba = self.list_proba / np.sum(self.list_proba)
            else:
                self.list_proba[self.last_index_param] *= 0.95
                self.list_proba = self.list_proba / np.sum(self.list_proba)
        else:
            self.last_fitness = new_fitness
        if random.random() <= 0.95:
            choice = np.random.choice(self.index_np, size=1, p=self.list_proba)[0]
        else:
            max_improvement = max(
                [
                    self.status[x]["nb_improvement"]
                    / max(self.status[x]["nb_usage"], 1)
                    for x in self.status
                ]
            )
            choice = random.choice(
                [
                    x
                    for x in self.status
                    if self.status[x]["nb_improvement"]
                    / max(self.status[x]["nb_usage"], 1)
                    == max_improvement
                ]
            )
        d_params = {
            key: getattr(self.list_params[int(choice)], key)
            for key in self.list_params[0].__dict__.keys()
        }
        ch = NeighborFlexibleStart(problem=self.problem, **d_params)
        self.current_iteration += 1
        self.last_index_param = choice
        self.status[self.last_index_param]["nb_usage"] += 1
        return ch.adding_constraint_from_results_store(
            cp_solver, child_instance, result_storage, last_result_store
        )

    def remove_constraints_from_previous_iteration(
        self, cp_solver: CPSolver, child_instance, previous_constraints: Iterable[Any]
    ):
        pass


def build_neighbor_operator(option_neighbor: OptionNeighbor, rcpsp_model):
    params_om = [Params(fraction_to_fix=0.75, minus_delta=100, plus_delta=100)]
    params_all = [
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.85, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
        Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
        Params(fraction_to_fix=0.92, minus_delta=10, plus_delta=0),
        Params(fraction_to_fix=0.88, minus_delta=0, plus_delta=10),
        Params(fraction_to_fix=0.9, minus_delta=10, plus_delta=0),
        Params(fraction_to_fix=0.8, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=15, plus_delta=15),
        Params(fraction_to_fix=0.9, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=1.0, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.85, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.98, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.9, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.98, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.98, minus_delta=8, plus_delta=8),
        Params(fraction_to_fix=0.98, minus_delta=10, plus_delta=10),
    ]
    params_fast = [
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.8, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.8, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.9, minus_delta=1, plus_delta=1),
        Params(fraction_to_fix=0.92, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.98, minus_delta=7, plus_delta=7),
        Params(fraction_to_fix=0.95, minus_delta=5, plus_delta=5),
    ]
    params_debug = [Params(fraction_to_fix=1.0, minus_delta=0, plus_delta=0)]
    params_large = [
        Params(fraction_to_fix=0.9, minus_delta=12, plus_delta=12),
        Params(fraction_to_fix=0.8, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.7, minus_delta=12, plus_delta=12),
        Params(fraction_to_fix=0.7, minus_delta=5, plus_delta=5),
        Params(fraction_to_fix=0.6, minus_delta=3, plus_delta=3),
        Params(fraction_to_fix=0.4, minus_delta=2, plus_delta=2),
        Params(fraction_to_fix=0.9, minus_delta=4, plus_delta=4),
        Params(fraction_to_fix=0.7, minus_delta=4, plus_delta=4),
        Params(fraction_to_fix=0.8, minus_delta=5, plus_delta=5),
    ]
    params = None
    if option_neighbor.name == OptionNeighbor.MIX_ALL.name:
        params = params_all
    if option_neighbor.name == OptionNeighbor.MIX_FAST.name:
        params = params_fast
    if option_neighbor.name == OptionNeighbor.MIX_LARGE_NEIGH.name:
        params = params_large
    if option_neighbor.name == OptionNeighbor.DEBUG.name:
        params = params_debug
    if option_neighbor.name == OptionNeighbor.LARGE.name:
        params = params_om
    probas = [1 / len(params)] * len(params)
    constraint_handler = ConstraintHandlerMix(
        problem=rcpsp_model, list_params=params, list_proba=probas
    )
    return constraint_handler
