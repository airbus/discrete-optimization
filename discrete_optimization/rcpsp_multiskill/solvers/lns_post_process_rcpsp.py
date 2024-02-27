#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Union

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_mip import PostProcessSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Preemptive,
    MS_RCPSPSolution_Preemptive_Variant,
)


class PostProMSRCPSP(PostProcessSolution):
    def __init__(
        self,
        problem: MS_RCPSPModel,
        params_objective_function: ParamsObjectiveFunction = None,
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
        self.graph = self.problem.compute_graph()

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

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        new_solution = sgs_variant(
            solution=result_storage.get_best_solution(),
            problem=self.problem,
            predecessors_dict=self.immediate_predecessors,
        )
        fit = self.aggreg_from_sol(new_solution)
        result_storage.add_solution(new_solution, fit)

        for s in random.sample(
            result_storage.list_solution_fits,
            min(len(result_storage.list_solution_fits), 1000),
        ):
            new_solution = sgs_variant(
                solution=s[0],
                problem=self.problem,
                predecessors_dict=self.immediate_predecessors,
            )
            fit = self.aggreg_from_sol(new_solution)
            result_storage.add_solution(new_solution, fit)
        return result_storage


class PostProMSRCPSPPreemptive(PostProcessSolution):
    def __init__(
        self,
        problem: MS_RCPSPModel,
        params_objective_function: ParamsObjectiveFunction = None,
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
        self.graph = self.problem.compute_graph()
        self.immediate_predecessors = {
            n: self.graph.get_predecessors(n) for n in self.graph.nodes_name
        }

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        new_solution = shift_left_method(
            solution=result_storage.get_last_best_solution()[0],
            problem=self.problem,
            predecessors_dict=self.immediate_predecessors,
        )
        fit = self.aggreg_from_sol(new_solution)
        result_storage.list_solution_fits += [(new_solution, fit)]

        for s in random.sample(
            result_storage.list_solution_fits,
            min(len(result_storage.list_solution_fits), 10),
        ):
            if len(s[0].schedule) == self.problem.nb_tasks:
                new_solution = shift_left_method(
                    solution=s[0],
                    problem=self.problem,
                    predecessors_dict=self.immediate_predecessors,
                )
                fit = self.aggreg_from_sol(new_solution)
                result_storage.list_solution_fits += [(new_solution, fit)]
        return result_storage


def sgs_variant(solution: MS_RCPSPSolution, problem: MS_RCPSPModel, predecessors_dict):
    new_proposed_schedule = {}
    resource_avail_in_time = {}
    new_horizon = min(solution.get_end_time(problem.sink_task), problem.horizon)
    for res in problem.resources_set:
        resource_avail_in_time[res] = np.copy(
            problem.resources_availability[res][: new_horizon + 1]
        )
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = np.array(problem.employees[i].calendar_employee)
    sorted_tasks = sorted(
        solution.schedule,
        key=lambda x: (solution.schedule[x]["start_time"], problem.index_task[x]),
    )
    for task in sorted_tasks:
        employee_used = [
            emp
            for emp in solution.employee_usage.get(task, {})
            if len(solution.employee_usage[task][emp]) > 0
        ]
        times_predecessors = [
            new_proposed_schedule[t]["end_time"]
            if t in new_proposed_schedule
            else solution.schedule[t]["end_time"]
            for t in predecessors_dict[task]
        ]
        if len(times_predecessors) > 0:
            min_time = max(times_predecessors)
        else:
            min_time = 0
        if (
            problem.do_special_constraints
            and task in problem.special_constraints.start_times_window
        ):
            if problem.special_constraints.start_times_window[task][0] is not None:
                min_time = max(
                    min_time, problem.special_constraints.start_times_window[task][0]
                )
        new_starting_time = solution.schedule[task]["start_time"]
        if solution.schedule[task]["start_time"] == solution.schedule[task]["end_time"]:
            new_starting_time = min_time
        else:
            for t in range(min_time, solution.schedule[task]["start_time"] + 1):
                if all(
                    all(
                        worker_avail_in_time[emp][time]
                        for time in range(
                            t,
                            t
                            + solution.schedule[task]["end_time"]
                            - solution.schedule[task]["start_time"],
                        )
                    )
                    for emp in employee_used
                ) and all(
                    resource_avail_in_time[res][time]
                    >= problem.mode_details[task][solution.modes[task]].get(res, 0)
                    for res in problem.resources_list
                    for time in range(
                        t,
                        t
                        + solution.schedule[task]["end_time"]
                        - solution.schedule[task]["start_time"],
                    )
                ):
                    new_starting_time = t
                    break
        duration = (
            solution.schedule[task]["end_time"] - solution.schedule[task]["start_time"]
        )
        new_proposed_schedule[task] = {
            "start_time": new_starting_time,
            "end_time": new_starting_time + duration,
        }
        for res in problem.resources_list:
            resource_avail_in_time[res][
                new_starting_time : new_starting_time + duration
            ] -= problem.mode_details[task][solution.modes[task]].get(res, 0)
        for t in range(new_starting_time, new_starting_time + duration):
            for emp in employee_used:
                worker_avail_in_time[emp][t] = False
    new_solution = MS_RCPSPSolution(
        problem=problem,
        modes=solution.modes,
        schedule=new_proposed_schedule,
        employee_usage=solution.employee_usage,
    )
    return new_solution


def sgs_variant_preemptive(
    solution: MS_RCPSPSolution_Preemptive, problem: MS_RCPSPModel, predecessors_dict
):
    new_proposed_schedule = {}
    new_proposed_schedule = {}
    resource_avail_in_time = {}
    modes_dict = solution.modes
    new_horizon = min(solution.get_end_time(problem.sink_task), problem.horizon)
    for res in problem.resources_set:
        resource_avail_in_time[res] = np.copy(
            problem.resources_availability[res][: new_horizon + 1]
        )
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = np.array(problem.employees[i].calendar_employee)
    sorted_tasks = sorted(
        solution.schedule.keys(), key=lambda x: (solution.get_start_time(x))
    )
    for task in list(sorted_tasks):
        if len(solution.schedule[task]["starts"]) > 1:
            new_proposed_schedule[task] = {
                "starts": solution.schedule[task]["starts"],
                "ends": solution.schedule[task]["ends"],
            }
            employees = solution.employee_used(task)
            for i in range(len(new_proposed_schedule[task]["starts"])):
                employee_used = employees[i]
                s = new_proposed_schedule[task]["starts"][i]
                e = new_proposed_schedule[task]["ends"][i]
                for res in problem.resources_list:
                    resource_avail_in_time[res][s:e] -= problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
                for t in range(s, e):
                    for emp in employee_used:
                        worker_avail_in_time[emp][t] = False
            sorted_tasks.remove(task)

    def is_startable(tt):
        if "special_constraints" in problem.__dict__.keys():
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
        if len(solution.schedule[task]["starts"]) > 1:
            continue
        times_predecessors = [
            new_proposed_schedule[t]["ends"][-1]
            if t in new_proposed_schedule
            else solution.schedule[t]["ends"][-1]
            for t in predecessors_dict[task]
        ]
        if isinstance(problem, MS_RCPSPSolution_Preemptive_Variant):
            times_predecessors += [
                new_proposed_schedule[t]["starts"][0]
                if t in new_proposed_schedule
                else solution.schedule[t]["starts"][0]
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
                else solution.schedule[t]["starts"][0]
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
                else solution.schedule[t]["ends"][-1]
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
                else solution.schedule[t]["ends"][-1]
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
        if len(solution.schedule[task]["starts"]) > 1:
            new_proposed_schedule[task] = {
                "starts": solution.schedule[task]["starts"],
                "ends": solution.schedule[task]["ends"],
            }
        else:
            emp = solution.employee_used(task)[0]
            for t in range(min_time, problem.horizon):
                if all(
                    resource_avail_in_time[res][time]
                    >= problem.mode_details[task][modes_dict[task]].get(res, 0)
                    for res in problem.resources_list
                    for time in range(
                        t, t + problem.mode_details[task][modes_dict[task]]["duration"]
                    )
                ) and all(worker_avail_in_time[e][t] >= 1 for e in emp):

                    new_starting_time = t
                    break
            new_proposed_schedule[task] = {
                "starts": [new_starting_time],
                "ends": [
                    new_starting_time
                    + problem.mode_details[task][modes_dict[task]]["duration"]
                ],
            }
            employee_used = solution.employee_used(task)
            for s, e, emp in zip(
                new_proposed_schedule[task]["starts"],
                new_proposed_schedule[task]["ends"],
                employee_used,
            ):
                for res in problem.resources_list:
                    resource_avail_in_time[res][s:e] -= problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
                for em in emp:
                    worker_avail_in_time[em][s:e] = 0

        if len(sorted_tasks) == 0:
            break

    new_solution = MS_RCPSPSolution_Preemptive(
        problem=problem,
        schedule=new_proposed_schedule,
        modes=solution.modes,
        employee_usage=solution.employee_usage,
    )
    return new_solution


def compute_schedule_per_employees(
    solution: Union[MS_RCPSPSolution_Preemptive, MS_RCPSPSolution],
    problem: MS_RCPSPModel,
    predecessors_dict,
):
    employees = {e: [] for e in problem.employees}
    employees["none"] = []
    for t in problem.tasks_list:
        starts = solution.get_start_times_list(t)
        ends = solution.get_start_times_list(t)
        employee_usages = solution.employee_used(t)
        for i in range(len(starts)):
            for emp in employee_usages[i]:
                employees[emp] += [(starts[i], ends[i], t)]
    for emp in employees:
        employees[emp] = sorted(employees[emp], key=lambda x: (x[0], x[1]))
    return employees


def shift_left_method(
    solution: Union[MS_RCPSPSolution_Preemptive, MS_RCPSPSolution],
    problem: MS_RCPSPModel,
    predecessors_dict,
):
    allparts_to_schedule = []
    for t in problem.tasks_list:
        starts = solution.get_start_times_list(t)
        ends = solution.get_end_times_list(t)
        employees = solution.employee_used(t)
        allparts_to_schedule += [
            (s, e, emp, t) for s, e, emp in zip(starts, ends, employees)
        ]

    sorted_piece_to_schedule = sorted(allparts_to_schedule, key=lambda x: (x[0], x[1]))
    duration_done = {t: 0 for t in problem.tasks_list}

    new_proposed_schedule = {}
    task_that_be_shifted_left = []
    new_proposed_schedule = {}
    resource_avail_in_time = {}
    modes_dict = solution.modes
    new_horizon = min(solution.get_end_time(problem.sink_task), problem.horizon)
    for res in problem.resources_set:
        resource_avail_in_time[res] = np.copy(
            problem.resources_availability[res][: new_horizon + 30]
        )
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = np.copy(problem.employees[i].calendar_employee)
    dones = set()

    def is_startable(tt):
        if "special_constraints" in problem.__dict__.keys():
            return (
                all(
                    x in dones
                    for x in problem.special_constraints.dict_start_after_nunit_reverse.get(
                        tt, {}
                    )
                )
                and all(
                    x in dones
                    for x in problem.special_constraints.dict_start_at_end_reverse.get(
                        tt, {}
                    )
                )
                and all(
                    x in dones
                    for x in problem.special_constraints.dict_start_at_end_offset_reverse.get(
                        tt, {}
                    )
                )
                and all(t in dones for t in predecessors_dict[tt])
            )
        return all(t in dones for t in predecessors_dict[tt])

    def get_ressource_available(res, time, task):
        return resource_avail_in_time[res][time] - sum(
            [
                problem.mode_details[tasks][solution.modes[tasks]].get(res, 0)
                for tasks in new_proposed_schedule
                if tasks not in dones
                and tasks != task
                and not problem.partial_preemption_data[tasks][solution.modes[tasks]][
                    res
                ]
                and time >= new_proposed_schedule[tasks]["starts"][0]
            ]
        )

    while True:
        X = next((x for x in sorted_piece_to_schedule if is_startable(x[-1])))
        sorted_piece_to_schedule.remove(X)
        start, end, employees_used, task = X
        duration_this_task = end - start
        duration_done[task] += duration_this_task
        if (
            duration_done[task]
            >= problem.mode_details[task][solution.modes[task]]["duration"] - 0.0001
        ):
            dones.add(task)
        if task not in new_proposed_schedule:
            times_predecessors = [
                new_proposed_schedule[t]["ends"][-1]
                if t in new_proposed_schedule
                else solution.schedule[t]["ends"][-1]
                for t in predecessors_dict[task]
            ]
            if task in problem.special_constraints.start_times_window:
                if problem.special_constraints.start_times_window[task][0] is not None:
                    times_predecessors += [
                        problem.special_constraints.start_times_window[task][0]
                    ]
            if isinstance(solution, MS_RCPSPSolution_Preemptive_Variant):
                times_predecessors += [
                    new_proposed_schedule[t]["starts"][0]
                    if t in new_proposed_schedule
                    else solution.schedule[t]["starts"][0]
                    for t in problem.special_constraints.dict_start_together.get(
                        task, set()
                    )
                ]
                times_predecessors += [
                    new_proposed_schedule[t]["starts"][0]
                    + problem.special_constraints.dict_start_after_nunit_reverse.get(
                        task
                    )[t]
                    if t in new_proposed_schedule
                    else solution.schedule[t]["starts"][0]
                    + problem.special_constraints.dict_start_after_nunit_reverse.get(
                        task
                    )[t]
                    for t in problem.special_constraints.dict_start_after_nunit_reverse.get(
                        task, {}
                    )
                ]
                times_predecessors += [
                    new_proposed_schedule[t]["ends"][-1]
                    if t in new_proposed_schedule
                    else solution.schedule[t]["ends"][-1]
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
                    else solution.schedule[t]["ends"][-1]
                    + problem.special_constraints.dict_start_at_end_offset_reverse.get(
                        task
                    )[t]
                    for t in problem.special_constraints.dict_start_at_end_offset_reverse.get(
                        task, {}
                    )
                ]
        else:
            times_predecessors = [new_proposed_schedule[task]["ends"][-1]]
        if len(times_predecessors) > 0:
            min_time = max(times_predecessors)
        else:
            min_time = 0
        emp = employees_used
        new_starting_time = None
        for t in range(min_time, problem.horizon):
            if all(
                get_ressource_available(res, time, task)
                >= problem.mode_details[task][modes_dict[task]].get(res, 0)
                for res in problem.resources_list
                for time in range(t, t + duration_this_task)
            ) and all(
                worker_avail_in_time[e][time] >= 1
                for e in emp
                for time in range(t, t + duration_this_task)
            ):
                new_starting_time = t
                break
        if task not in new_proposed_schedule:
            new_proposed_schedule[task] = {
                "starts": [new_starting_time],
                "ends": [new_starting_time + duration_this_task],
            }
        else:
            new_proposed_schedule[task]["starts"] += [new_starting_time]
            new_proposed_schedule[task]["ends"] += [
                new_starting_time + duration_this_task
            ]
        employee_used = employees_used

        for s, e, emp in zip(
            [new_proposed_schedule[task]["starts"][-1]],
            [new_proposed_schedule[task]["ends"][-1]],
            [employee_used],
        ):
            for res in problem.resources_list:
                releasable = problem.partial_preemption_data[task][
                    solution.modes[task]
                ][res]
                if releasable:
                    resource_avail_in_time[res][s:e] -= problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
            for em in emp:
                worker_avail_in_time[em][s:e] = 0
        if task in dones:
            for res in problem.resources_list:
                releasable = problem.partial_preemption_data[task][
                    solution.modes[task]
                ][res]
                if not releasable:
                    resource_avail_in_time[res][
                        new_proposed_schedule[task]["starts"][
                            0
                        ] : new_proposed_schedule[task]["ends"][-1]
                    ] -= problem.mode_details[task][modes_dict[task]].get(res, 0)
        if len(sorted_piece_to_schedule) == 0:
            break
    if isinstance(solution, MS_RCPSPSolution_Preemptive):
        employee_usage = {}
        for t in solution.employee_usage:
            employee_usage[t] = []
            for i in range(len(solution.employee_usage[t])):
                employee_usage[t] += [
                    {
                        s: solution.employee_usage[t][i][s]
                        for s in solution.employee_usage[t][i]
                        if len(solution.employee_usage[t][i][s]) > 0
                    }
                ]
        new_solution = MS_RCPSPSolution_Preemptive(
            problem=problem,
            schedule=new_proposed_schedule,
            modes=solution.modes,
            employee_usage=employee_usage,
        )
    else:
        schedule = {
            t: {
                "start_time": new_proposed_schedule[t]["starts"][0],
                "end_time": new_proposed_schedule[t]["ends"][0],
            }
            for t in new_proposed_schedule
        }
        new_solution = MS_RCPSPSolution(
            problem=problem,
            schedule=schedule,
            modes=solution.modes,
            employee_usage=solution.employee_usage,
        )
    return new_solution
