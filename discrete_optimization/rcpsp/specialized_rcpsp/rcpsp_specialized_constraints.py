#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from copy import deepcopy
from functools import partial
from typing import Dict, Hashable, List, Optional, Set, Type, Union

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)
from discrete_optimization.rcpsp.fast_function_rcpsp import (
    compute_mean_ressource,
    sgs_fast_partial_schedule_preemptive,
    sgs_fast_partial_schedule_preemptive_minduration,
    sgs_fast_preemptive_minduration,
    sgs_fast_preemptive_some_special_constraints,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    RCPSPModelPreemptive,
    RCPSPSolutionPreemptive,
)
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import intersect
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)

logger = logging.getLogger(__name__)


class RCPSPSolutionSpecialPreemptive(RCPSPSolutionPreemptive):
    def change_problem(self, new_problem: Problem):
        self.__init__(
            problem=new_problem,
            rcpsp_permutation=self.rcpsp_permutation,
            rcpsp_modes=self.rcpsp_modes,
        )

    def __setattr__(self, key, value):
        super.__setattr__(self, key, value)
        if key == "rcpsp_permutation":
            self._schedule_to_recompute = True

    def copy(self):
        return RCPSPSolutionSpecialPreemptive(
            problem=self.problem,
            rcpsp_permutation=deepcopy(self.rcpsp_permutation),
            rcpsp_modes=deepcopy(self.rcpsp_modes),
            rcpsp_schedule=deepcopy(self.rcpsp_schedule),
            rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
            standardised_permutation=self.standardised_permutation,
        )

    def lazy_copy(self):
        return RCPSPSolutionSpecialPreemptive(
            problem=self.problem,
            rcpsp_permutation=self.rcpsp_permutation,
            rcpsp_modes=self.rcpsp_modes,
            rcpsp_schedule=self.rcpsp_schedule,
            rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
            standardised_permutation=self.standardised_permutation,
        )

    def __str__(self):
        if self.rcpsp_schedule is None:
            sched_str = "None"
        else:
            sched_str = str(self.rcpsp_schedule)
        val = "RCPSP solution (rcpsp_schedule): " + sched_str
        return val

    def generate_schedule_from_permutation_serial_sgs(self, do_fast=True):
        if do_fast:
            super().generate_schedule_from_permutation_serial_sgs(do_fast=True)
        else:
            schedule, feasible = self.problem.sgs_func(
                solution=self, rcpsp_problem=self.problem
            )
            self.rcpsp_schedule = schedule
            self.rcpsp_schedule_feasible = feasible
            self._schedule_to_recompute = False

    def generate_schedule_from_permutation_serial_sgs_2(
        self,
        current_t: int = 0,
        completed_tasks: Optional[Set[Hashable]] = None,
        partial_schedule: Optional[Dict[Hashable, Dict[str, List[int]]]] = None,
        do_fast: bool = True,
    ):
        if do_fast:
            super().generate_schedule_from_permutation_serial_sgs_2(
                current_t=current_t,
                completed_tasks=completed_tasks,
                partial_schedule=partial_schedule,
                do_fast=do_fast,
            )
        else:
            if completed_tasks is None:
                completed_tasks = {}
            if partial_schedule is None:
                partial_schedule = partial_schedule
            schedule, feasible = self.problem.sgs_func_partial(
                solution=self,
                current_t=current_t,
                partial_schedule=partial_schedule,
                completed_tasks=completed_tasks,
                rcpsp_problem=self.problem,
            )
            self.rcpsp_schedule = schedule
            self.rcpsp_schedule_feasible = not feasible
            self._schedule_to_recompute = False


class RCPSPModelSpecialConstraintsPreemptive(RCPSPModelPreemptive):
    def __init__(
        self,
        resources: Union[Dict[str, int], Dict[str, List[int]]],
        non_renewable_resources: List[str],
        mode_details: Dict[Hashable, Dict[Union[str, int], Dict[str, int]]],
        successors: Dict[Union[int, str], List[Union[str, int]]],
        horizon,
        special_constraints: SpecialConstraintsDescription = None,
        preemptive_indicator: Dict[Hashable, bool] = None,
        relax_the_start_at_end: bool = True,
        tasks_list: List[Union[int, str]] = None,
        source_task=None,
        sink_task=None,
        name_task: Dict[int, str] = None,
        **kwargs
    ):
        super().__init__(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            tasks_list=tasks_list,
            source_task=source_task,
            sink_task=sink_task,
            name_task=name_task,
            preemptive_indicator=preemptive_indicator,
        )

        self.special_constraints = special_constraints
        self.do_special_constraints = special_constraints is not None
        if self.special_constraints is None:
            self.special_constraints = SpecialConstraintsDescription()
        self.predecessors_dict = {task: [] for task in self.tasks_list}
        for task in self.successors:
            for stask in self.successors[task]:
                self.predecessors_dict[stask] += [task]
        if self.do_special_constraints:
            for t1, t2 in self.special_constraints.start_at_end:
                if t2 not in self.successors[t1]:
                    self.successors[t1].append(t2)
            for t1, t2, off in self.special_constraints.start_at_end_plus_offset:
                if t2 not in self.successors[t1]:
                    self.successors[t1].append(t2)
            for t1, t2 in self.special_constraints.start_together:
                for predt1 in self.predecessors_dict[t1]:
                    if t2 not in self.successors[predt1]:
                        self.successors[predt1] += [t2]
                for predt2 in self.predecessors_dict[t2]:
                    if t1 not in self.successors[predt2]:
                        self.successors[predt2] += [t1]
        self.graph = self.compute_graph()
        self.predecessors = self.graph.predecessors_dict
        self.sgs_func = generate_schedule_from_permutation_serial_sgs_preemptive
        self.sgs_func_partial = (
            generate_schedule_from_permutation_serial_sgs_partial_schedule_preempptive
        )
        self.relax_the_start_at_end = relax_the_start_at_end
        (
            self.func_sgs,
            self.func_sgs_2,
            self.compute_mean_resource,
        ) = create_np_data_and_jit_functions(self)

    def is_preemptive(self):
        return True

    def has_special_constraints(self):
        return self.do_special_constraints

    def update_function(self):
        (
            self.func_sgs,
            self.func_sgs_2,
            self.compute_mean_resource,
        ) = create_np_data_and_jit_functions(self)

    def update_functions(self):
        self.update_function()

    def copy(self):
        return RCPSPModelSpecialConstraintsPreemptive(
            resources=deepcopy(self.resources),
            non_renewable_resources=deepcopy(self.non_renewable_resources),
            mode_details=deepcopy(self.mode_details),
            successors=deepcopy(self.successors),
            horizon=self.horizon,
            special_constraints=deepcopy(self.special_constraints),
            preemptive_indicator=deepcopy(self.preemptive_indicator),
            relax_the_start_at_end=self.relax_the_start_at_end,
            tasks_list=deepcopy(self.tasks_list),
            source_task=self.source_task,
            sink_task=self.sink_task,
            name_task=deepcopy(self.name_task),
        )

    def lazy_copy(self):
        return RCPSPModelSpecialConstraintsPreemptive(
            resources=self.resources,
            non_renewable_resources=self.non_renewable_resources,
            mode_details=self.mode_details,
            successors=self.successors,
            horizon=self.horizon,
            special_constraints=self.special_constraints,
            preemptive_indicator=self.preemptive_indicator,
            relax_the_start_at_end=self.relax_the_start_at_end,
            tasks_list=self.tasks_list,
            source_task=self.source_task,
            sink_task=self.sink_task,
            name_task=self.name_task,
        )

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == "rcpsp_permutation":
            single_mode_list = [1 for i in range(self.n_jobs_non_dummy)]
            rcpsp_sol = RCPSPSolutionSpecialPreemptive(
                problem=self, rcpsp_permutation=int_vector, rcpsp_modes=single_mode_list
            )
            objectives = self.evaluate(rcpsp_sol)
            return objectives
        return None

    def evaluate_function(self, rcpsp_sol: RCPSPSolutionPreemptive):
        if rcpsp_sol._schedule_to_recompute:
            rcpsp_sol.generate_schedule_from_permutation_serial_sgs()
        makespan = rcpsp_sol.get_end_time(task=self.sink_task)
        if rcpsp_sol.rcpsp_schedule_feasible:
            penalty = evaluate_constraints(
                solution=rcpsp_sol, constraints=self.special_constraints
            )
        else:
            penalty = 0
        return makespan, penalty

    def evaluate(self, rcpsp_sol: RCPSPSolutionPreemptive) -> Dict[str, float]:
        obj_makespan, penalty = self.evaluate_function(rcpsp_sol)
        return {"makespan": obj_makespan, "constraint_penalty": penalty}

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0),
            "constraint_penalty": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100.0
            ),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def satisfy(self, rcpsp_sol: RCPSPSolutionPreemptive):
        s = check_solution(
            problem=self,
            solution=rcpsp_sol,
            relax_the_start_at_end=self.relax_the_start_at_end,
        )
        if not s:
            return s
        return super().satisfy(rcpsp_sol)

    def get_dummy_solution(self, random_perm: bool = False):
        rcpsp_permutation = list(range(self.n_jobs_non_dummy))
        if random_perm:
            random.shuffle(rcpsp_permutation)
        sol = RCPSPSolutionSpecialPreemptive(
            problem=self,
            rcpsp_permutation=rcpsp_permutation,
            rcpsp_modes=[1 for i in range(self.n_jobs_non_dummy)],
        )
        return sol

    def get_solution_type(self) -> Type[Solution]:
        return RCPSPSolutionSpecialPreemptive


def evaluate_constraints(
    solution: Union[RCPSPSolution, RCPSPSolutionPreemptive],
    constraints: SpecialConstraintsDescription,
):
    list_constraints_not_respected = compute_constraints_details(solution, constraints)
    return sum([x[-1] for x in list_constraints_not_respected])


def compute_constraints_details(
    solution: Union[RCPSPSolution, RCPSPSolutionPreemptive],
    constraints: SpecialConstraintsDescription,
):
    if (
        "rcpsp_schedule_feasible" in solution.__dict__.keys()
        and not solution.rcpsp_schedule_feasible
    ):
        return []
    start_together = constraints.start_together
    start_at_end = constraints.start_at_end
    start_at_end_plus_offset = constraints.start_at_end_plus_offset
    start_after_nunit = constraints.start_after_nunit
    disjunctive = constraints.disjunctive_tasks
    list_constraints_not_respected = []
    for (t1, t2) in start_together:
        time1 = solution.get_start_time(t1)
        time2 = solution.get_start_time(t2)
        b = time1 == time2
        if not b:
            list_constraints_not_respected += [
                ("start_together", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2) in start_at_end:
        time1 = solution.get_end_time(t1)
        time2 = solution.get_start_time(t2)
        b = time1 == time2
        if not b:
            list_constraints_not_respected += [
                ("start_at_end", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2, off) in start_at_end_plus_offset:
        time1 = solution.get_end_time(t1) + off
        time2 = solution.get_start_time(t2)
        b = time2 >= time1
        if not b:
            list_constraints_not_respected += [
                ("start_at_end_plus_offset", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2, off) in start_after_nunit:
        time1 = solution.get_start_time(t1) + off
        time2 = solution.get_start_time(t2)
        b = time2 >= time1
        if not b:
            list_constraints_not_respected += [
                ("start_after_nunit", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for t1, t2 in disjunctive:
        b = intersect(
            [solution.get_start_time(t1), solution.get_end_time(t1)],
            [solution.get_start_time(t2), solution.get_end_time(t2)],
        )
        if b is not None:
            list_constraints_not_respected += [
                ("disjunctive", t1, t2, None, None, b[1] - b[0])
            ]
    for t in constraints.start_times_window:
        if constraints.start_times_window[t][0] is not None:
            if solution.get_start_time(t) < constraints.start_times_window[t][0]:
                list_constraints_not_respected += [
                    (
                        "start_window_0",
                        t,
                        t,
                        None,
                        None,
                        constraints.start_times_window[t][0]
                        - solution.get_start_time(t),
                    )
                ]

        if constraints.start_times_window[t][1] is not None:
            if solution.get_start_time(t) > constraints.start_times_window[t][1]:
                list_constraints_not_respected += [
                    (
                        "start_window_1",
                        t,
                        t,
                        None,
                        None,
                        -constraints.start_times_window[t][1]
                        + solution.get_start_time(t),
                    )
                ]

    for t in constraints.end_times_window:
        if constraints.end_times_window[t][0] is not None:
            if solution.get_end_time(t) < constraints.end_times_window[t][0]:
                list_constraints_not_respected += [
                    (
                        "end_window_0",
                        t,
                        t,
                        None,
                        None,
                        constraints.end_times_window[t][0] - solution.get_end_time(t),
                    )
                ]

        if constraints.end_times_window[t][1] is not None:
            if solution.get_end_time(t) > constraints.end_times_window[t][1]:
                list_constraints_not_respected += [
                    (
                        "end_window_1",
                        t,
                        t,
                        None,
                        None,
                        -constraints.end_times_window[t][1] + solution.get_end_time(t),
                    )
                ]
    return list_constraints_not_respected


def check_solution(
    problem: RCPSPModelSpecialConstraintsPreemptive,
    solution: Union[
        RCPSPSolutionSpecialPreemptive,
        RCPSPSolution,
        RCPSPSolutionPreemptive,
    ],
    relax_the_start_at_end: bool = True,
):
    if not solution.rcpsp_schedule_feasible:
        return False
    start_together = problem.special_constraints.start_together
    start_at_end = problem.special_constraints.start_at_end
    start_at_end_plus_offset = problem.special_constraints.start_at_end_plus_offset
    start_after_nunit = problem.special_constraints.start_after_nunit
    disjunctive = problem.special_constraints.disjunctive_tasks
    for (t1, t2) in start_together:
        if not relax_the_start_at_end:
            b = solution.get_start_time(t1) == solution.get_start_time(t2)
            if not b:
                return False
    for (t1, t2) in start_at_end:
        if relax_the_start_at_end:
            b = solution.get_start_time(t2) >= solution.get_end_time(t1)
        else:
            b = solution.get_start_time(t2) == solution.get_end_time(t1)
        if not b:
            return False
    for (t1, t2, off) in start_at_end_plus_offset:
        b = solution.get_start_time(t2) >= solution.get_end_time(t1) + off
        if not b:
            logger.debug(("start_at_end_plus_offset NOT respected: ", t1, t2, off))
            logger.debug(
                (
                    solution.get_start_time(t2),
                    " >= ",
                    solution.get_end_time(t1),
                    "+",
                    off,
                )
            )
            return False
    for (t1, t2, off) in start_after_nunit:
        b = solution.get_start_time(t2) >= solution.get_start_time(t1) + off
        if not b:
            logger.debug(("start_after_nunit NOT respected: ", t1, t2, off))
            return False
    for t1, t2 in disjunctive:
        b = intersect(
            [solution.get_start_time(t1), solution.get_end_time(t1)],
            [solution.get_start_time(t2), solution.get_end_time(t2)],
        )
        if b is not None:
            return False
    for t in problem.special_constraints.start_times_window:
        if problem.special_constraints.start_times_window[t][0] is not None:
            if (
                solution.get_start_time(t)
                < problem.special_constraints.start_times_window[t][0]
            ):
                logger.debug(
                    (
                        "start time 0, ",
                        t,
                        solution.get_start_time(t),
                        problem.special_constraints.start_times_window[t][0],
                    )
                )
                return False
        if problem.special_constraints.start_times_window[t][1] is not None:
            if (
                solution.get_start_time(t)
                > problem.special_constraints.start_times_window[t][1]
            ):
                logger.debug(
                    (
                        "start time 1, ",
                        t,
                        solution.get_start_time(t),
                        problem.special_constraints.start_times_window[t][1],
                    )
                )
                return False
    for t in problem.special_constraints.end_times_window:
        if problem.special_constraints.end_times_window[t][0] is not None:
            if (
                solution.get_end_time(t)
                < problem.special_constraints.end_times_window[t][0]
            ):
                logger.debug(
                    (
                        "end time 0, ",
                        t,
                        solution.get_end_time(t),
                        problem.special_constraints.end_times_window[t][0],
                    )
                )
                return False
        if problem.special_constraints.end_times_window[t][1] is not None:
            if (
                solution.get_end_time(t)
                > problem.special_constraints.end_times_window[t][1]
            ):
                logger.debug(
                    (
                        "end time 1, ",
                        t,
                        solution.get_end_time(t),
                        problem.special_constraints.end_times_window[t][1],
                    )
                )
                return False
    return True


def generate_schedule_from_permutation_serial_sgs_preemptive(
    solution, rcpsp_problem: RCPSPModelSpecialConstraintsPreemptive
):
    activity_end_times = {}

    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon

    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = np.copy(
                rcpsp_problem.resources[res][: new_horizon + 1]
            )
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        minimum_starting_time[act] = 0
        if rcpsp_problem.do_special_constraints:
            if act in rcpsp_problem.special_constraints.start_times_window:
                minimum_starting_time[act] = (
                    rcpsp_problem.special_constraints.start_times_window[act][0]
                    if rcpsp_problem.special_constraints.start_times_window[act][0]
                    is not None
                    else 0
                )

    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)
    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)

    for k in modes_dict:
        if modes_dict[k] not in rcpsp_problem.mode_details[k]:
            modes_dict[k] = 1
    expected_durations_task = {
        k: rcpsp_problem.mode_details[k][modes_dict[k]]["duration"] for k in modes_dict
    }
    schedules = {}

    def ressource_consumption(res, task, duration, mode):
        dur = rcpsp_problem.mode_details[task][mode]["duration"]
        if duration > dur:
            return 0
        return rcpsp_problem.mode_details[task][mode].get(res, 0)

    def look_for_task(perm, ignore_sc=False):
        act_ids = []
        for task_id in perm:
            respected = True
            # Check all kind of precedence constraints....
            for pred in rcpsp_problem.predecessors.get(task_id, {}):
                if pred in perm_extended:
                    respected = False
                    break
            if not ignore_sc:
                for (
                    pred
                ) in rcpsp_problem.special_constraints.dict_start_at_end_reverse.get(
                    task_id, {}
                ):
                    if pred in perm_extended:
                        respected = False
                        break
                for (
                    pred
                ) in rcpsp_problem.special_constraints.dict_start_at_end_offset_reverse.get(
                    task_id, {}
                ):
                    if pred in perm_extended:
                        respected = False
                        break
                for (
                    pred
                ) in rcpsp_problem.special_constraints.dict_start_after_nunit_reverse.get(
                    task_id, {}
                ):
                    if pred in perm_extended:
                        respected = False
                        break
            task_to_start_too = set()
            if respected:
                task_to_start_too = (
                    rcpsp_problem.special_constraints.dict_start_together.get(
                        task_id, set()
                    )
                )
                if not ignore_sc:
                    if len(task_to_start_too) > 0:
                        if not all(
                            s not in perm_extended
                            for t in task_to_start_too
                            for s in rcpsp_problem.predecessors[t]
                        ):
                            respected = False
                        if not all(
                            s not in perm_extended
                            for t in task_to_start_too
                            for s in rcpsp_problem.special_constraints.dict_start_at_end_reverse.get(
                                t, {}
                            )
                        ):
                            respected = False
                        if not all(
                            s not in perm_extended
                            for t in task_to_start_too
                            for s in rcpsp_problem.special_constraints.dict_start_at_end_offset_reverse.get(
                                t, {}
                            )
                        ):
                            respected = False
                        if not all(
                            s not in perm_extended
                            for t in task_to_start_too
                            for s in rcpsp_problem.special_constraints.dict_start_after_nunit_reverse.get(
                                t, {}
                            )
                        ):
                            respected = False
            if respected:
                act_ids = [task_id] + list(task_to_start_too)
                break
        return act_ids

    unfeasible = False
    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
        act_ids = look_for_task(
            [
                k
                for k in rcpsp_problem.special_constraints.dict_start_at_end_reverse
                if k in perm_extended
            ]
        )
        if len(act_ids) == 0:
            act_ids = look_for_task(perm_extended)
        if (
            len(act_ids) == 0
        ):  # The constraints in the model are not necessarly trustable, leading to this problem.
            act_ids = look_for_task(perm_extended, ignore_sc=True)
        current_min_time = max([minimum_starting_time[act_id] for act_id in act_ids])
        starts = {act_id: [] for act_id in act_ids}
        ends = {act_id: [] for act_id in act_ids}
        cur_duration = {act_id: 0 for act_id in act_ids}
        valid = False
        first_step = (
            False  # we force the starting of all act_id to be the same current time
        )
        while not valid:
            if all(expected_durations_task[act_id] for act_id in act_ids) == 0:
                for act_id in act_ids:
                    starts[act_id] += [current_min_time]
                    ends[act_id] += [current_min_time]
                    cur_duration[act_id] += ends[act_id][-1] - starts[act_id][-1]
            else:
                reached_end = True
                if not first_step:
                    current_min_time = next(
                        (
                            t
                            for t in range(current_min_time, new_horizon)
                            if all(
                                resource_avail_in_time[res][t]
                                >= sum(
                                    [
                                        ressource_consumption(
                                            res=res,
                                            task=ac,
                                            mode=modes_dict[ac],
                                            duration=cur_duration[ac],
                                        )
                                        for ac in act_ids
                                    ]
                                )
                                for res in rcpsp_problem.resources_list
                            )
                        ),
                        None,
                    )
                    if current_min_time is None:
                        unfeasible = True
                        break
                    current_min_time_dict = {ac: current_min_time for ac in act_ids}
                    first_step = True
                reached_dict = {}
                for ac in act_ids:
                    reached_t = None
                    for t in range(
                        current_min_time_dict[ac],
                        current_min_time_dict[ac]
                        + expected_durations_task[ac]
                        - cur_duration[ac],
                    ):
                        if t >= new_horizon:
                            reached_end = False
                            unfeasible_non_renewable_resources = True
                            break
                        if any(
                            resource_avail_in_time[res][t]
                            < rcpsp_problem.mode_details[ac][modes_dict[ac]].get(res, 0)
                            for res in rcpsp_problem.resources_list
                        ):
                            reached_end = False
                            break
                        else:
                            reached_t = t
                    reached_dict[ac] = reached_t
                    if reached_t is not None and rcpsp_problem.can_be_preempted(ac):
                        starts[ac] += [current_min_time_dict[ac]]
                        ends[ac] += [reached_dict[ac] + 1]
                        cur_duration[ac] += ends[ac][-1] - starts[ac][-1]
                        for res in rcpsp_problem.resources_list:
                            for t in range(starts[ac][-1], ends[ac][-1]):
                                resource_avail_in_time[res][
                                    t
                                ] -= rcpsp_problem.mode_details[ac][modes_dict[ac]].get(
                                    res, 0
                                )
                                if resource_avail_in_time[res][t] < 0:
                                    logger.warning(
                                        "Resources available should not be negative"
                                    )
                    if (
                        reached_end
                        and reached_dict[ac] is not None
                        and not rcpsp_problem.can_be_preempted(ac)
                    ):
                        starts[ac] += [current_min_time_dict[ac]]
                        ends[ac] += [reached_dict[ac] + 1]
                        cur_duration[ac] += ends[ac][-1] - starts[ac][-1]
                        for res in rcpsp_problem.resources_list:
                            for t in range(starts[ac][-1], ends[ac][-1]):
                                resource_avail_in_time[res][
                                    t
                                ] -= rcpsp_problem.mode_details[ac][modes_dict[ac]].get(
                                    res, 0
                                )
                                if resource_avail_in_time[res][t] < 0:
                                    logger.warning(
                                        "Resources available should not be negative"
                                    )
                                if (
                                    res in rcpsp_problem.non_renewable_resources
                                    and t == ends[ac][-1] - 1
                                ):
                                    for tt in range(t + 1, new_horizon):
                                        resource_avail_in_time[res][
                                            tt
                                        ] -= rcpsp_problem.mode_details[ac][
                                            modes_dict[ac]
                                        ].get(
                                            res, 0
                                        )
                                        if resource_avail_in_time[res][tt] < 0:
                                            unfeasible_non_renewable_resources = True
            valid = all(
                cur_duration[ac] == expected_durations_task[ac] for ac in act_ids
            )
            if not valid:
                current_min_time_dict = {
                    ac: next(
                        (
                            t
                            for t in range(
                                reached_dict[ac] + 2
                                if reached_dict[ac] is not None
                                else current_min_time_dict[ac] + 1,
                                new_horizon,
                            )
                            if all(
                                resource_avail_in_time[res][t]
                                >= sum(
                                    [
                                        ressource_consumption(
                                            res=res,
                                            task=ac,
                                            mode=modes_dict[ac],
                                            duration=cur_duration[ac] + 1,
                                        )
                                    ]
                                )
                                for res in rcpsp_problem.resources_list
                            )
                        ),
                        None,
                    )
                    for ac in act_ids
                }
                if any(
                    current_min_time_dict[ac] is None for ac in current_min_time_dict
                ):
                    unfeasible = True
                    break
        if not unfeasible_non_renewable_resources and not unfeasible:
            for ac in starts:
                activity_end_times[ac] = ends[ac][-1]
                schedules[ac] = (starts[ac], ends[ac])
                perm_extended.remove(ac)
                for s in rcpsp_problem.successors[ac]:
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s], activity_end_times[ac]
                    )
                for s in rcpsp_problem.special_constraints.dict_start_at_end.get(
                    ac, {}
                ):
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s], activity_end_times[ac]
                    )
                for s in rcpsp_problem.special_constraints.dict_start_after_nunit.get(
                    ac, {}
                ):
                    minimum_starting_time[s] = max(
                        starts[ac][0]
                        + rcpsp_problem.special_constraints.dict_start_after_nunit[ac][
                            s
                        ],
                        minimum_starting_time[s],
                    )
                for s in rcpsp_problem.special_constraints.dict_start_at_end_offset.get(
                    ac, {}
                ):
                    minimum_starting_time[s] = max(
                        activity_end_times[ac]
                        + rcpsp_problem.special_constraints.dict_start_at_end_offset[
                            ac
                        ][s],
                        minimum_starting_time[s],
                    )
        else:
            break
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["starts"] = schedules[act_id][0]
        rcpsp_schedule[act_id]["ends"] = schedules[act_id][1]
    if unfeasible_non_renewable_resources or unfeasible:
        logger.debug(
            (
                "unfeasible: ",
                unfeasible,
                "unfeasible_non_renewable_resources: ",
                unfeasible_non_renewable_resources,
            )
        )
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["starts"] = [9999999]
            rcpsp_schedule[last_act_id]["ends"] = [9999999]
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def generate_schedule_from_permutation_serial_sgs_partial_schedule_preempptive(
    solution,
    rcpsp_problem,
    partial_schedule: Dict[Hashable, Dict[str, List[int]]],
    current_t,
    completed_tasks,
):
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon
    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = list(
                rcpsp_problem.resources[res][: new_horizon + 1]
            )
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        minimum_starting_time[act] = current_t

        if rcpsp_problem.do_special_constraints:
            if act in rcpsp_problem.special_constraints.start_times_window:
                minimum_starting_time[act] = (
                    max(
                        rcpsp_problem.special_constraints.start_times_window[act][0],
                        minimum_starting_time[act],
                    )
                    if rcpsp_problem.special_constraints.start_times_window[act][0]
                    is not None
                    else minimum_starting_time[act]
                )
    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)

    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
    for k in modes_dict:
        if modes_dict[k] not in rcpsp_problem.mode_details[k]:
            modes_dict[k] = 1
    expected_durations_task = {
        k: rcpsp_problem.mode_details[k][modes_dict[k]]["duration"] for k in modes_dict
    }
    done_duration_task = {k: 0 for k in modes_dict}
    schedules = deepcopy(partial_schedule)
    # Update current resource usage by the scheduled task (ongoing task, in practice)
    for task in partial_schedule:
        starts = partial_schedule[task]["starts"]
        ends = partial_schedule[task]["ends"]
        done_duration_task[task] = sum(
            [ends[i] - starts[i] for i in range(len(starts))]
        )
        end_t = ends[-1]
        for s, e in zip(starts, ends):
            for t in range(s, e):
                for res in resource_avail_in_time:
                    resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
                    if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                        for tt in range(end_t, new_horizon):
                            resource_avail_in_time[res][
                                tt
                            ] -= rcpsp_problem.mode_details[task][modes_dict[task]].get(
                                res, 0
                            )
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
        if done_duration_task[task] == expected_durations_task[task]:
            activity_end_times[task] = end_t
            perm_extended.remove(task)
            for s in rcpsp_problem.successors[task]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[task]
                )
        else:
            minimum_starting_time[task] = ends[-1]
    perm_extended = [x for x in perm_extended if x not in completed_tasks]
    # fix modes in case specified mode not in mode details for the activites
    for ac in modes_dict:
        if modes_dict[ac] not in rcpsp_problem.mode_details[ac]:
            modes_dict[ac] = 1

    def ressource_consumption(res, task, duration, mode):
        dur = rcpsp_problem.mode_details[task][mode]["duration"]
        if duration > dur:
            return 0
        return rcpsp_problem.mode_details[task][mode].get(res, 0)

    def look_for_task(perm):
        act_ids = []
        for task_id in perm:
            respected = True
            # Check all kind of precedence constraints....
            for pred in rcpsp_problem.predecessors.get(task_id, {}):
                if pred in perm_extended:
                    respected = False
                    break
            for pred in rcpsp_problem.special_constraints.dict_start_at_end_reverse.get(
                task_id, {}
            ):
                if pred in perm_extended:
                    respected = False
                    break
            for (
                pred
            ) in rcpsp_problem.special_constraints.dict_start_at_end_offset_reverse.get(
                task_id, {}
            ):
                if pred in perm_extended:
                    respected = False
                    break
            for (
                pred
            ) in rcpsp_problem.special_constraints.dict_start_after_nunit_reverse.get(
                task_id, {}
            ):
                if pred in perm_extended:
                    respected = False
                    break
            task_to_start_too = set()
            if respected:
                task_to_start_too = (
                    rcpsp_problem.special_constraints.dict_start_together.get(
                        task_id, set()
                    )
                )
                if len(task_to_start_too) > 0:
                    if not all(
                        s not in perm_extended
                        for t in task_to_start_too
                        for s in rcpsp_problem.predecessors[t]
                    ):
                        respected = False
                    if not all(
                        s not in perm_extended
                        for t in task_to_start_too
                        for s in rcpsp_problem.special_constraints.dict_start_at_end_reverse.get(
                            t, {}
                        )
                    ):
                        respected = False
                    if not all(
                        s not in perm_extended
                        for t in task_to_start_too
                        for s in rcpsp_problem.special_constraints.dict_start_at_end_offset_reverse.get(
                            t, {}
                        )
                    ):
                        respected = False
                    if not all(
                        s not in perm_extended
                        for t in task_to_start_too
                        for s in rcpsp_problem.special_constraints.dict_start_after_nunit_reverse.get(
                            t, {}
                        )
                    ):
                        respected = False
            if respected:
                act_ids = [task_id] + list(task_to_start_too)
                break
        return act_ids

    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
        act_ids = look_for_task(
            [
                k
                for k in rcpsp_problem.special_constraints.dict_start_at_end_reverse
                if k in perm_extended
            ]
        )
        if len(act_ids) == 0:
            act_ids = look_for_task(perm_extended)
        current_min_time = max([minimum_starting_time[act_id] for act_id in act_ids])
        starts = {act_id: [] for act_id in act_ids}
        ends = {act_id: [] for act_id in act_ids}
        cur_duration = {act_id: 0 for act_id in act_ids}
        valid = False
        first_step = (
            False  # we force the starting of all act_id to be the same current time
        )
        while not valid:
            if all(expected_durations_task[act_id] for act_id in act_ids) == 0:
                for act_id in act_ids:
                    starts[act_id] += [current_min_time]
                    ends[act_id] += [current_min_time]
                    cur_duration[act_id] += ends[act_id][-1] - starts[act_id][-1]
            else:
                reached_end = True
                if not first_step:
                    current_min_time = next(
                        t
                        for t in range(current_min_time, new_horizon)
                        if all(
                            resource_avail_in_time[res][t]
                            >= sum(
                                [
                                    ressource_consumption(
                                        res=res,
                                        task=ac,
                                        mode=modes_dict[ac],
                                        duration=cur_duration[ac],
                                    )
                                    for ac in act_ids
                                ]
                            )
                            for res in rcpsp_problem.resources_list
                        )
                    )
                    current_min_time_dict = {ac: current_min_time for ac in act_ids}
                    first_step = True
                reached_dict = {}
                for ac in act_ids:
                    reached_t = None
                    for t in range(
                        current_min_time_dict[ac],
                        current_min_time_dict[ac]
                        + expected_durations_task[ac]
                        - cur_duration[ac],
                    ):
                        if t >= new_horizon:
                            reached_end = False
                            unfeasible_non_renewable_resources = True
                            break
                        if any(
                            resource_avail_in_time[res][t]
                            < rcpsp_problem.mode_details[ac][modes_dict[ac]].get(res, 0)
                            for res in rcpsp_problem.resources_list
                        ):
                            reached_end = False
                            break
                        else:
                            reached_t = t
                    reached_dict[ac] = reached_t
                    if reached_t is not None and rcpsp_problem.can_be_preempted(ac):
                        starts[ac] += [current_min_time_dict[ac]]
                        ends[ac] += [reached_dict[ac] + 1]
                        cur_duration[ac] += ends[ac][-1] - starts[ac][-1]
                        for res in rcpsp_problem.resources_list:
                            for t in range(starts[ac][-1], ends[ac][-1]):
                                resource_avail_in_time[res][
                                    t
                                ] -= rcpsp_problem.mode_details[ac][modes_dict[ac]][res]
                                if resource_avail_in_time[res][t] < 0:
                                    logger.warning(
                                        "Resources available should not be negative"
                                    )
                    if (
                        reached_end
                        and reached_dict[ac] is not None
                        and not rcpsp_problem.can_be_preempted(ac)
                    ):
                        starts[ac] += [current_min_time_dict[ac]]
                        ends[ac] += [reached_dict[ac] + 1]
                        cur_duration[ac] += ends[ac][-1] - starts[ac][-1]
                        for res in rcpsp_problem.resources_list:
                            for t in range(starts[ac][-1], ends[ac][-1]):
                                resource_avail_in_time[res][
                                    t
                                ] -= rcpsp_problem.mode_details[ac][modes_dict[ac]][res]
                                if resource_avail_in_time[res][t] < 0:
                                    logger.warning(
                                        "Resources available should not be negative"
                                    )
                                if (
                                    res in rcpsp_problem.non_renewable_resources
                                    and t == ends[ac][-1] - 1
                                ):
                                    for tt in range(t + 1, new_horizon):
                                        resource_avail_in_time[res][
                                            tt
                                        ] -= rcpsp_problem.mode_details[ac][
                                            modes_dict[ac]
                                        ][
                                            res
                                        ]
                                        if resource_avail_in_time[res][tt] < 0:
                                            unfeasible_non_renewable_resources = True
            valid = all(
                cur_duration[ac] == expected_durations_task[ac] for ac in act_ids
            )
            if not valid:
                current_min_time_dict = {
                    ac: next(
                        t
                        for t in range(
                            reached_dict[ac] + 2
                            if reached_dict[ac] is not None
                            else current_min_time_dict[ac] + 1,
                            new_horizon,
                        )
                        if all(
                            resource_avail_in_time[res][t]
                            >= sum(
                                [
                                    ressource_consumption(
                                        res=res,
                                        task=ac,
                                        mode=modes_dict[ac],
                                        duration=cur_duration[ac] + 1,
                                    )
                                ]
                            )
                            for res in rcpsp_problem.resources_list
                        )
                    )
                    for ac in act_ids
                }
        if not unfeasible_non_renewable_resources:
            for ac in starts:
                activity_end_times[ac] = ends[ac][-1]
                schedules[ac] = (starts[ac], ends[ac])
                perm_extended.remove(ac)
                for s in rcpsp_problem.successors[ac]:
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s], activity_end_times[ac]
                    )
                for s in rcpsp_problem.special_constraints.dict_start_at_end.get(
                    ac, {}
                ):
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s], activity_end_times[ac]
                    )
                for s in rcpsp_problem.special_constraints.dict_start_after_nunit.get(
                    ac, {}
                ):
                    minimum_starting_time[s] = max(
                        starts[ac][0]
                        + rcpsp_problem.special_constraints.dict_start_after_nunit[ac][
                            s
                        ],
                        minimum_starting_time[s],
                    )
                for s in rcpsp_problem.special_constraints.dict_start_at_end_offset.get(
                    ac, {}
                ):
                    minimum_starting_time[s] = max(
                        activity_end_times[ac]
                        + rcpsp_problem.special_constraints.dict_start_at_end_offset[
                            ac
                        ][s],
                        minimum_starting_time[s],
                    )
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = schedules[act_id]
    for act_id in completed_tasks:
        rcpsp_schedule[act_id] = partial_schedule[act_id]
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["starts"] = [99999999]
            rcpsp_schedule[last_act_id]["ends"] = [9999999]
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def create_np_data_and_jit_functions(
    rcpsp_problem: Union[RCPSPModelSpecialConstraintsPreemptive],
):
    consumption_array = np.zeros(
        (
            rcpsp_problem.n_jobs,
            rcpsp_problem.max_number_of_mode,
            len(rcpsp_problem.resources_list),
        ),
        dtype=np.int_,
    )
    duration_array = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.max_number_of_mode), dtype=np.int_
    )
    predecessors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int_)
    successors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int_)
    preemptive_tag = np.zeros(rcpsp_problem.n_jobs, dtype=np.bool_)
    horizon = rcpsp_problem.horizon
    ressource_available = np.zeros(
        (len(rcpsp_problem.resources_list), horizon), dtype=np.int_
    )
    ressource_renewable = np.ones((len(rcpsp_problem.resources_list)), dtype=bool)
    min_duration_preemptive_bool = np.zeros(rcpsp_problem.n_jobs, dtype=bool)
    min_duration_preemptive = np.zeros(rcpsp_problem.n_jobs, dtype=np.int_)
    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        min_duration_preemptive_bool[i] = rcpsp_problem.duration_subtask[task][0]
        min_duration_preemptive[i] = rcpsp_problem.duration_subtask[task][1]

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        preemptive_tag[i] = rcpsp_problem.can_be_preempted(task)
        index_mode = 0
        for mode in sorted(
            rcpsp_problem.mode_details[rcpsp_problem.tasks_list[i]].keys()
        ):
            for k in range(len(rcpsp_problem.resources_list)):
                consumption_array[i, index_mode, k] = rcpsp_problem.mode_details[task][
                    mode
                ].get(rcpsp_problem.resources_list[k], 0)
            duration_array[i, index_mode] = rcpsp_problem.mode_details[task][mode][
                "duration"
            ]
            index_mode += 1

    task_index = {rcpsp_problem.tasks_list[i]: i for i in range(rcpsp_problem.n_jobs)}
    for k in range(len(rcpsp_problem.resources_list)):
        if rcpsp_problem.is_varying_resource():
            ressource_available[k, :] = rcpsp_problem.resources[
                rcpsp_problem.resources_list[k]
            ][: ressource_available.shape[1]]
        else:
            ressource_available[k, :] = np.full(
                ressource_available.shape[1],
                rcpsp_problem.resources[rcpsp_problem.resources_list[k]],
                dtype=np.int_,
            )
        if rcpsp_problem.resources_list[k] in rcpsp_problem.non_renewable_resources:
            ressource_renewable[k] = False

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        for s in rcpsp_problem.successors[task]:
            index_s = task_index[s]
            predecessors[index_s, i] = 1
            successors[i, index_s] = 1
    minimum_starting_time_array = np.zeros(rcpsp_problem.n_jobs, dtype=np.int_)
    for t in rcpsp_problem.special_constraints.start_times_window:
        if rcpsp_problem.special_constraints.start_times_window[t][0] is not None:
            minimum_starting_time_array[
                rcpsp_problem.index_task[t]
            ] = rcpsp_problem.special_constraints.start_times_window[t][0]

    start_at_end_plus_offset = np.zeros(
        (len(rcpsp_problem.special_constraints.start_at_end_plus_offset), 3),
        dtype=np.int_,
    )
    start_after_nunit = np.zeros(
        (len(rcpsp_problem.special_constraints.start_after_nunit), 3), dtype=np.int_
    )
    j = 0
    for t1, t2, off in rcpsp_problem.special_constraints.start_at_end_plus_offset:
        start_at_end_plus_offset[j, 0] = rcpsp_problem.index_task[t1]
        start_at_end_plus_offset[j, 1] = rcpsp_problem.index_task[t2]
        start_at_end_plus_offset[j, 2] = off
        j += 1
    j = 0
    for t1, t2, off in rcpsp_problem.special_constraints.start_after_nunit:
        start_after_nunit[j, 0] = rcpsp_problem.index_task[t1]
        start_after_nunit[j, 1] = rcpsp_problem.index_task[t2]
        start_after_nunit[j, 2] = off
        j += 1
    if not rcpsp_problem.is_duration_minimum_preemption():
        func_sgs = partial(
            sgs_fast_preemptive_some_special_constraints,
            consumption_array=consumption_array,
            preemptive_tag=preemptive_tag,
            start_after_nunit=start_after_nunit,
            start_at_end_plus_offset=start_at_end_plus_offset,
            minimum_starting_time_array=minimum_starting_time_array,
            duration_array=duration_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
        )
        func_sgs_2 = partial(
            sgs_fast_partial_schedule_preemptive,
            consumption_array=consumption_array,
            preemptive_tag=preemptive_tag,
            duration_array=duration_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
        )
    else:
        func_sgs = partial(
            sgs_fast_preemptive_minduration,
            consumption_array=consumption_array,
            preemptive_tag=preemptive_tag,
            duration_array=duration_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            min_duration_preemptive=min_duration_preemptive,
            min_duration_preemptive_bool=min_duration_preemptive_bool,
        )
        func_sgs_2 = partial(
            sgs_fast_partial_schedule_preemptive_minduration,
            consumption_array=consumption_array,
            preemptive_tag=preemptive_tag,
            duration_array=duration_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            min_duration_preemptive=min_duration_preemptive,
            min_duration_preemptive_bool=min_duration_preemptive_bool,
        )
    func_compute_mean_resource = partial(
        compute_mean_ressource,
        consumption_array=consumption_array,
        ressource_available=ressource_available,
        ressource_renewable=ressource_renewable,
    )
    return func_sgs, func_sgs_2, func_compute_mean_resource
