#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import re
from enum import Enum
from typing import Any

import numpy as np

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver, dp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution
from discrete_optimization.rcpsp.solvers import RcpspSolver
from discrete_optimization.rcpsp.solvers.cpm import CpmRcpspSolver

logger = logging.getLogger(__name__)


class DpRcpspModeling(Enum):
    TASK_AND_TIME = 0
    TASK_ORIGINAL = 1
    TASK_AVAIL_TASK_UPD = 2
    TASK_MULTIMODE = 3


def create_resource_consumption_from_calendar(
    calendar_availability: np.ndarray,
) -> list[dict[str, int]]:
    max_capacity = np.max(calendar_availability)
    fake_tasks: list[dict[str, int]] = []
    delta = calendar_availability[:-1] - calendar_availability[1:]
    index_non_zero = np.nonzero(delta)[0]
    if calendar_availability[0] < max_capacity:
        consume = {
            "value": int(max_capacity - calendar_availability[0]),
            "duration": int(index_non_zero[0] + 1),
            "start": 0,
        }
        fake_tasks += [consume]
    for j in range(len(index_non_zero) - 1):
        ind = index_non_zero[j]
        value = calendar_availability[ind + 1]
        if value != max_capacity:
            consume = {
                "value": int(max_capacity - value),
                "duration": int(index_non_zero[j + 1] - ind),
                "start": int(ind + 1),
            }
            fake_tasks += [consume]
    return fake_tasks


class DpRcpspSolver(DpSolver, RcpspSolver, WarmstartMixin):
    hyperparameters = DpSolver.hyperparameters + [
        EnumHyperparameter(
            name="modeling",
            enum=DpRcpspModeling,
            default=DpRcpspModeling.TASK_MULTIMODE,
        ),
        CategoricalHyperparameter(
            name="dual_bound", choices=[True, False], default=False
        ),
    ]
    problem: RcpspProblem
    modeling: DpRcpspModeling
    transitions: dict

    def __init__(self, problem: RcpspProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        cpm = CpmRcpspSolver(problem=self.problem)
        cpm.run_classic_cpm()
        result = cpm.map_node
        self.remaining_per_task = {
            n: result[self.problem.sink_task]._EFD - result[n]._EFD for n in result
        }
        self.result_cpm = result

    def init_model(
        self,
        **kwargs: Any,
    ):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        modeling: DpRcpspModeling = kwargs["modeling"]
        if modeling == DpRcpspModeling.TASK_AND_TIME:
            if self.problem.is_varying_resource() or self.problem.is_rcpsp_multimode():
                self.init_model_task_time_calendar(**kwargs)
            else:
                self.init_model_task_time(**kwargs)
        if modeling == DpRcpspModeling.TASK_ORIGINAL:
            self.init_model_task(**kwargs)
        if modeling == DpRcpspModeling.TASK_AVAIL_TASK_UPD:
            self.init_model_task_upd(**kwargs)
        if modeling == DpRcpspModeling.TASK_MULTIMODE:
            if self.problem.is_varying_resource():
                self.init_model_multimode_calendar(**kwargs)
            else:
                self.init_model_multimode(**kwargs)
        self.modeling = modeling

    def init_model_task_time(self, **kwargs: Any) -> None:
        model = dp.Model()
        task_object = model.add_object_type(number=self.problem.n_jobs)
        unscheduled = model.add_set_var(
            object_type=task_object, target=range(self.problem.n_jobs)
        )
        scheduled = model.add_set_var(object_type=task_object, target=set())
        nb_resource = len(self.problem.resources_list)
        res_available = [
            [
                model.add_int_var(target=self.problem.get_max_resource_capacity(res))
                for i in range(self.problem.horizon)
            ]
            for res in self.problem.resources_list
        ]
        starts = [model.add_int_var(target=0) for i in range(self.problem.n_jobs)]
        durs = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        consumption = [
            [
                self.problem.mode_details[t][1].get(res)
                for res in self.problem.resources_list
            ]
            for t in self.problem.tasks_list
        ]
        prec_ = [set() for t in self.problem.tasks_list]
        for task in self.problem.successors:
            for succ in self.problem.successors[task]:
                prec_[self.problem.index_task[succ]].add(self.problem.index_task[task])
        consumption_table = model.add_int_table(consumption)
        predecessors = model.add_set_table(prec_, object_type=task_object)
        model.add_base_case([unscheduled.is_empty()])
        for i_res in range(len(res_available)):
            for t_res in range(len(res_available[i_res])):
                model.add_state_constr(res_available[i_res][t_res] >= 0)
        current_time = model.add_int_var(target=0)
        self.transitions = {}
        for i in range(self.problem.n_jobs):
            n_z = [
                (j, consumption[i][j])
                for j in range(nb_resource)
                if consumption[i][j] > 0
            ]
            dur = self.problem.mode_details[self.problem.tasks_list[i]][1]["duration"]
            for time in range(self.problem.horizon - dur):
                transition = dp.Transition(
                    name=f"schedule_{i}_{time}",
                    cost=dp.max(current_time, time + dur)
                    - current_time
                    + dp.IntExpr.state_cost(),
                    effects=[
                        (res_available[j][time + k], res_available[j][time + k] - conso)
                        for j, conso in n_z
                        for k in range(dur)
                    ]
                    + [
                        (current_time, dp.max(current_time, time + dur)),
                        # (current_sched_time, time),
                        (unscheduled, unscheduled.remove(i)),
                        (scheduled, scheduled.add(i)),
                        (starts[i], time),
                    ],
                    preconditions=[
                        unscheduled.contains(i),
                        predecessors[i].issubset(scheduled)  # ,
                        # time >= current_sched_time
                    ]
                    + [time >= starts[p] + durs[p] for p in prec_[i]],
                )
                model.add_transition(transition)
                self.transitions[("sched", i, 1, time)] = transition
        model.add_dual_bound(consumption_table[unscheduled, 0])
        self.model = model

    def init_model_task(self, **kwargs: Any) -> None:
        model = dp.Model()
        task_object = model.add_object_type(number=self.problem.n_jobs)
        unscheduled = model.add_set_var(
            object_type=task_object, target=range(self.problem.n_jobs)
        )
        scheduled = model.add_set_var(object_type=task_object, target=set())
        capacities = [
            self.problem.get_max_resource_capacity(r)
            for r in self.problem.resources_list
        ]
        nb_resource = len(self.problem.resources_list)
        starts = [model.add_int_var(target=0) for i in range(self.problem.n_jobs)]
        durs = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        consumption = [
            [
                self.problem.mode_details[t][1].get(res, 0)
                for res in self.problem.resources_list
            ]
            for t in self.problem.tasks_list
        ]
        prec_ = [set() for t in self.problem.tasks_list]
        for task in self.problem.successors:
            for succ in self.problem.successors[task]:
                prec_[self.problem.index_task[succ]].add(self.problem.index_task[task])
        predecessors = model.add_set_table(prec_, object_type=task_object)
        current_time = model.add_int_var(target=0)
        current_resource_consumption = [
            model.add_int_resource_var(target=0, less_is_better=True)
            for i in range(nb_resource)
        ]
        ongoing = model.add_set_var(object_type=task_object, target=set())
        finished = model.add_set_var(object_type=task_object, target=set())
        cur_makespan = model.add_int_var(target=0)
        model.add_base_case([unscheduled.is_empty(), ongoing.is_empty()])
        for j in range(nb_resource):
            model.add_state_constr(current_resource_consumption[j] <= capacities[j])
        self.transitions = {}
        for i in range(self.problem.n_jobs):
            n_z = [
                (j, consumption[i][j])
                for j in range(nb_resource)
                if consumption[i][j] > 0
            ]
            dur = self.problem.mode_details[self.problem.tasks_list[i]][1]["duration"]
            if dur > 0:
                start_task = dp.Transition(
                    name=f"start_{i}",
                    cost=dp.IntExpr.state_cost(),
                    effects=[
                        (ongoing, ongoing.add(i)),
                        (scheduled, scheduled.add(i)),
                        (unscheduled, unscheduled.remove(i)),
                        (starts[i], current_time),
                    ]
                    + [
                        (
                            current_resource_consumption[j],
                            current_resource_consumption[j] + cons,
                        )
                        for j, cons in n_z
                    ],
                    preconditions=[
                        unscheduled.contains(i),
                        predecessors[i].issubset(finished),
                    ]
                    + [current_time >= starts[p] + durs[p] for p in prec_[i]]
                    + [
                        current_resource_consumption[j] + cons
                        <= capacities[j]  # put it also here
                        for j, cons in n_z
                    ],
                )
                model.add_transition(start_task)
                self.transitions[("start", i, 1)] = start_task
            else:
                start_dummy_task = dp.Transition(
                    name=f"start_{i}",
                    cost=dp.IntExpr.state_cost(),
                    effects=[
                        (scheduled, scheduled.add(i)),
                        (finished, finished.add(i)),
                        (unscheduled, unscheduled.remove(i)),
                        (starts[i], current_time),
                    ],
                    preconditions=[
                        unscheduled.contains(i),
                        predecessors[i].issubset(finished),
                    ]
                    + [current_time >= starts[p] + durs[p] for p in prec_[i]],
                )
                model.add_transition(start_dummy_task, forced=True)
                self.transitions[("start", i, 1)] = start_dummy_task
            if dur > 0:
                ending_task = dp.Transition(
                    name=f"end_{i}",
                    cost=(dp.max(starts[i] + durs[i], cur_makespan) - cur_makespan)
                    + dp.IntExpr.state_cost(),
                    # cost=starts[i] + durs[i] + dp.IntExpr.state_cost(),
                    effects=[
                        (current_time, starts[i] + durs[i]),
                        (ongoing, ongoing.remove(i)),
                        (finished, finished.add(i)),
                        (cur_makespan, dp.max(starts[i] + durs[i], cur_makespan)),
                    ]
                    + [
                        (
                            current_resource_consumption[j],
                            current_resource_consumption[j] - cons,
                        )
                        for j, cons in n_z
                    ],
                    preconditions=[
                        ongoing.contains(i),
                        starts[i] + durs[i] >= current_time,
                    ],
                )
                model.add_transition(ending_task)
                self.transitions[("end", i, 1)] = ending_task
        # remaining = model.add_int_table([self.remaining_per_task[t] for t in self.problem.tasks_list])
        # model.add_dual_bound((~unscheduled.is_empty()).if_then_else(remaining.max(unscheduled), 0))
        # model.add_dual_bound(((unscheduled.len() < 20) & ~unscheduled.is_empty())
        #                      .if_then_else(remaining.max(unscheduled), 0))

        self.starts = starts
        self.model = model

    def init_model_task_upd(self, **kwargs: Any) -> None:
        model = dp.Model()
        lfd = [self.result_cpm[n]._LFD for n in self.problem.tasks_list]
        # lsf_table = model.add_int_table(lsf)
        task_object = model.add_object_type(number=self.problem.n_jobs)
        time_object = model.add_object_type(number=self.problem.horizon)
        delays = model.add_set_var(object_type=time_object, target=set())
        unscheduled = model.add_set_var(
            object_type=task_object, target=range(self.problem.n_jobs)
        )
        scheduled = model.add_set_var(object_type=task_object, target=set())
        # res_to_index = {self.problem.resources_list[i]: i
        #                 for i in range(len(self.problem.resources_list))}
        capacities = [
            self.problem.get_max_resource_capacity(r)
            for r in self.problem.resources_list
        ]
        nb_resource = len(self.problem.resources_list)
        starts = [model.add_int_var(target=0) for i in range(self.problem.n_jobs)]
        durs = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        consumption = [
            [
                self.problem.mode_details[t][1].get(res)
                for res in self.problem.resources_list
            ]
            for t in self.problem.tasks_list
        ]
        prec_ = [set() for t in self.problem.tasks_list]
        for task in self.problem.successors:
            for succ in self.problem.successors[task]:
                prec_[self.problem.index_task[succ]].add(self.problem.index_task[task])
        predecessors = model.add_set_table(prec_, object_type=task_object)
        current_time = model.add_int_var(target=0)
        current_resource_consumption = [
            model.add_int_resource_var(target=0, less_is_better=True)
            for i in range(nb_resource)
        ]
        ongoing = model.add_set_var(object_type=task_object, target=set())
        finished = model.add_set_var(object_type=task_object, target=set())
        cur_makespan = model.add_int_var(target=0)
        max_delay = model.add_int_resource_var(target=0, less_is_better=False)
        # model.add_base_case([unscheduled.is_empty(), ongoing.is_empty()])
        model.add_base_case([unscheduled.is_empty()])
        to_update = model.add_int_var(target=0)
        # for j in range(nb_resource):
        #     model.add_state_constr(current_resource_consumption[j] <= capacities[j])
        can_start = [model.add_int_var(target=0) for i in range(self.problem.n_jobs)]
        self.transitions = {}
        for i in range(self.problem.n_jobs):
            n_z = [
                (j, consumption[i][j])
                for j in range(nb_resource)
                if consumption[i][j] > 0
            ]
            dur = self.problem.mode_details[self.problem.tasks_list[i]][1]["duration"]
            if dur > 0:
                start_task = dp.Transition(
                    name=f"start_{i}",
                    cost=dp.IntExpr.state_cost(),
                    effects=[
                        (ongoing, ongoing.add(i)),
                        (scheduled, scheduled.add(i)),
                        (unscheduled, unscheduled.remove(i)),
                        (max_delay, dp.max(max_delay, current_time + durs[i] - lfd[i])),
                        (starts[i], current_time),
                        (to_update, 1),
                    ]
                    + [
                        (
                            current_resource_consumption[j],
                            current_resource_consumption[j] + cons,
                        )
                        for j, cons in n_z
                    ],
                    preconditions=[unscheduled.contains(i), can_start[i] == 1]
                    + [current_time >= starts[p] + durs[p] for p in prec_[i]]
                    + [
                        current_resource_consumption[j] + cons <= capacities[j]
                        for j, cons in n_z
                    ],
                )
                model.add_transition(start_task)
                self.transitions[("start", i, 1)] = start_task
            else:
                start_dummy_task = dp.Transition(
                    name=f"start_{i}",
                    cost=dp.IntExpr.state_cost(),
                    effects=[
                        (scheduled, scheduled.add(i)),
                        (finished, finished.add(i)),
                        (unscheduled, unscheduled.remove(i)),
                        (starts[i], current_time),
                        (to_update, 1),
                    ],
                    preconditions=[
                        unscheduled.contains(i),
                        predecessors[i].issubset(finished),
                    ]
                    + [current_time >= starts[p] + durs[p] for p in prec_[i]],
                )
                model.add_transition(start_dummy_task, forced=True)
                self.transitions[("start", i, 1)] = start_dummy_task
            if dur > 0:
                ending_task = dp.Transition(
                    name=f"end_{i}",
                    cost=5 * (dp.max(starts[i] + durs[i], cur_makespan) - cur_makespan)
                    + dp.IntExpr.state_cost(),
                    effects=[
                        (current_time, starts[i] + durs[i]),
                        (ongoing, ongoing.remove(i)),
                        (finished, finished.add(i)),
                        (to_update, 1),
                        (cur_makespan, dp.max(starts[i] + durs[i], cur_makespan)),
                    ]
                    + [
                        (
                            current_resource_consumption[j],
                            current_resource_consumption[j] - cons,
                        )
                        for j, cons in n_z
                    ],
                    preconditions=[
                        ongoing.contains(i),
                        starts[i] + durs[i] >= current_time,
                    ],
                )
                model.add_transition(ending_task)
                self.transitions[("end", i, 1)] = ending_task
        compute_avail = dp.Transition(
            name="update_avail",
            cost=-2
            * (
                sum(
                    predecessors[i].issubset(finished).if_then_else(1, 0) - can_start[i]
                    for i in range(self.problem.n_jobs)
                )
            )
            + dp.IntExpr.state_cost(),
            effects=[
                (can_start[i], predecessors[i].issubset(finished).if_then_else(1, 0))
                for i in range(self.problem.n_jobs)
            ]
            + [(to_update, 0)],
            preconditions=[to_update == 1],
        )
        model.add_transition(compute_avail)  # forced=True)
        self.transitions["compute_avail"] = compute_avail
        # remaining = model.add_int_table([self.remaining_per_task[t] for t in self.problem.tasks_list])
        # model.add_dual_bound((~unscheduled.is_empty()).if_then_else(remaining.max(unscheduled), 0))
        self.starts = starts
        self.max_delay = max_delay
        self.current_time = current_time
        self.model = model

    def init_model_multimode(self, **kwargs: Any) -> None:
        model = dp.Model()
        task_object = model.add_object_type(number=self.problem.n_jobs)
        unscheduled = model.add_set_var(
            object_type=task_object, target=range(self.problem.n_jobs)
        )
        scheduled = model.add_set_var(object_type=task_object, target=set())
        nb_resource = len(self.problem.resources_list)
        nb_tasks = self.problem.n_jobs
        capacities = [
            self.problem.get_max_resource_capacity(r)
            for r in self.problem.resources_list
        ]
        is_renewable = [
            r not in self.problem.non_renewable_resources
            for r in self.problem.resources_list
        ]
        index_non_renewable = set(
            [j for j in range(nb_resource) if not is_renewable[j]]
        )
        starts = [model.add_int_var(target=0) for i in range(nb_tasks)]
        tasks_list = self.problem.tasks_list
        modes_per_task = [
            [m for m in sorted(self.problem.mode_details[t])] for t in tasks_list
        ]
        durs = [
            [
                self.problem.mode_details[tasks_list[i]][m]["duration"]
                for m in modes_per_task[i]
            ]
            for i in range(nb_tasks)
        ]
        consumption = [
            [
                [
                    self.problem.mode_details[tasks_list[i]][m].get(res, 0)
                    for res in self.problem.resources_list
                ]
                for m in modes_per_task[i]
            ]
            for i in range(nb_tasks)
        ]
        prec_ = [set() for t in self.problem.tasks_list]
        for task in self.problem.successors:
            for succ in self.problem.successors[task]:
                prec_[self.problem.index_task[succ]].add(self.problem.index_task[task])
        predecessors = model.add_set_table(prec_, object_type=task_object)
        current_time = model.add_int_var(target=0)
        current_resource_consumption = [
            model.add_int_resource_var(target=0, less_is_better=True)
            for i in range(nb_resource)
        ]
        ongoing = model.add_set_var(object_type=task_object, target=set())
        finished = model.add_set_var(object_type=task_object, target=set())
        modes_var = [model.add_int_var(target=0) for i in range(nb_tasks)]
        cur_makespan = model.add_int_var(target=0)
        model.add_base_case([unscheduled.is_empty(), ongoing.is_empty()])
        for j in range(nb_resource):
            model.add_state_constr(current_resource_consumption[j] <= capacities[j])
        self.transitions = {}
        for i in range(self.problem.n_jobs):
            for j_mode in range(len(modes_per_task[i])):
                n_z = [
                    (j, consumption[i][j_mode][j])
                    for j in range(nb_resource)
                    if consumption[i][j_mode][j] > 0
                ]
                dur = durs[i][j_mode]
                if dur > 0:
                    start_task = dp.Transition(
                        name=f"start_{i}_mode_{j_mode}",
                        cost=dp.IntExpr.state_cost(),
                        effects=[
                            (ongoing, ongoing.add(i)),
                            (scheduled, scheduled.add(i)),
                            (modes_var[i], j_mode),
                            (unscheduled, unscheduled.remove(i)),
                            (starts[i], current_time),
                        ]
                        + [
                            (
                                current_resource_consumption[j],
                                current_resource_consumption[j] + cons,
                            )
                            for j, cons in n_z
                        ],
                        preconditions=[
                            unscheduled.contains(i),
                            predecessors[i].issubset(finished),
                        ]
                        + [
                            current_resource_consumption[j] + cons <= capacities[j]
                            for j, cons in n_z
                        ],
                    )
                    model.add_transition(start_task)
                    self.transitions[
                        ("start", i, modes_per_task[i][j_mode])
                    ] = start_task
                else:
                    start_dummy_task = dp.Transition(
                        name=f"start_{i}_mode_{j_mode}",
                        cost=dp.IntExpr.state_cost(),
                        effects=[
                            (scheduled, scheduled.add(i)),
                            (finished, finished.add(i)),
                            (unscheduled, unscheduled.remove(i)),
                            (starts[i], current_time),
                        ],
                        preconditions=[
                            unscheduled.contains(i),
                            predecessors[i].issubset(finished),
                        ],
                    )
                    model.add_transition(start_dummy_task, forced=True)
                    self.transitions[
                        ("start", i, modes_per_task[i][j_mode])
                    ] = start_dummy_task
                if dur > 0:
                    ending_task = dp.Transition(
                        name=f"end_{i}_mode_{j_mode}",
                        cost=(dp.max(starts[i] + dur, cur_makespan) - cur_makespan)
                        + dp.IntExpr.state_cost(),
                        # cost=starts[i] + durs[i] + dp.IntExpr.state_cost(),
                        effects=[
                            (current_time, starts[i] + dur),
                            (ongoing, ongoing.remove(i)),
                            (finished, finished.add(i)),
                            (cur_makespan, dp.max(starts[i] + dur, cur_makespan)),
                        ]
                        + [
                            (
                                current_resource_consumption[j],
                                current_resource_consumption[j] - cons,
                            )
                            for j, cons in n_z
                            if is_renewable[j]
                        ],
                        preconditions=[
                            ongoing.contains(i),
                            modes_var[i] == j_mode,
                            starts[i] + dur >= current_time,
                        ],
                    )
                    model.add_transition(ending_task)
                    self.transitions[
                        ("end", i, modes_per_task[i][j_mode])
                    ] = ending_task

        remaining = model.add_int_table(
            [self.remaining_per_task[t] for t in self.problem.tasks_list]
        )
        model.add_dual_bound(
            (~unscheduled.is_empty()).if_then_else(remaining.max(unscheduled), 0)
        )
        self.starts = starts
        self.modes_per_task = modes_per_task
        self.model = model

    def init_model_multimode_calendar(self, **kwargs: Any) -> None:
        nb_resource = len(self.problem.resources_list)
        merged_calendars = []
        for res in self.problem.resources_list:
            cld = create_resource_consumption_from_calendar(
                self.problem.get_resource_availability_array(res)
            )
            for t in cld:
                t.update({res: t["value"]})
                merged_calendars.append(t)
        merged_calendars = sorted(merged_calendars, key=lambda x: x["start"])
        starts_calendar = [x["start"] for x in merged_calendars]
        durs_calendar = [x["duration"] for x in merged_calendars]
        ends_calendar = [x["start"] + x["duration"] for x in merged_calendars]
        consumption_calendar = [
            [x.get(res, 0) for res in self.problem.resources_list]
            for x in merged_calendars
        ]
        nb_task_calendar = len(starts_calendar)
        model = dp.Model()
        calendars_res = [
            model.add_int_table(self.problem.get_resource_availability_array(res))
            for res in self.problem.resources_list
        ]
        task_cal_object = model.add_object_type(number=nb_task_calendar)
        task_object = model.add_object_type(number=self.problem.n_jobs)
        ongoing_calendar = model.add_set_var(object_type=task_cal_object, target=set())
        unscheduled_calendar = model.add_set_var(
            object_type=task_cal_object, target=range(nb_task_calendar)
        )
        finished_calendar = model.add_set_var(object_type=task_cal_object, target=set())
        unscheduled = model.add_set_var(
            object_type=task_object, target=range(self.problem.n_jobs)
        )
        scheduled = model.add_set_var(object_type=task_object, target=set())
        nb_tasks = self.problem.n_jobs
        capacities = [
            self.problem.get_max_resource_capacity(r)
            for r in self.problem.resources_list
        ]
        is_renewable = [
            r not in self.problem.non_renewable_resources
            for r in self.problem.resources_list
        ]
        time_object = model.add_object_type(number=self.problem.horizon)
        starts = [model.add_int_var(target=0) for i in range(nb_tasks)]
        starts_element = [
            model.add_element_var(object_type=time_object, target=0)
            for i in range(nb_tasks)
        ]
        tasks_list = self.problem.tasks_list
        modes_per_task = [
            [m for m in sorted(self.problem.mode_details[t])] for t in tasks_list
        ]
        durs = [
            [
                self.problem.mode_details[tasks_list[i]][m]["duration"]
                for m in modes_per_task[i]
            ]
            for i in range(nb_tasks)
        ]
        consumption = [
            [
                [
                    self.problem.mode_details[tasks_list[i]][m].get(res)
                    for res in self.problem.resources_list
                ]
                for m in modes_per_task[i]
            ]
            for i in range(nb_tasks)
        ]
        prec_ = [set() for t in self.problem.tasks_list]
        for task in self.problem.successors:
            for succ in self.problem.successors[task]:
                prec_[self.problem.index_task[succ]].add(self.problem.index_task[task])
        predecessors = model.add_set_table(prec_, object_type=task_object)
        current_time = model.add_int_var(target=0)
        current_time_element = model.add_element_var(object_type=time_object, target=0)
        current_resource_consumption = [
            model.add_int_resource_var(target=0, less_is_better=True)
            for i in range(nb_resource)
        ]
        ongoing = model.add_set_var(object_type=task_object, target=set())
        finished = model.add_set_var(object_type=task_object, target=set())
        modes_var = [model.add_int_var(target=0) for i in range(nb_tasks)]
        cur_makespan = model.add_int_var(target=0)
        model.add_base_case(
            [
                unscheduled.is_empty(),
                ongoing.is_empty(),
                unscheduled_calendar.is_empty(),
            ]
        )
        for j in range(nb_resource):
            model.add_state_constr(current_resource_consumption[j] <= capacities[j])

        # for j in range(nb_resource):
        #     model.add_state_constr(current_resource_consumption[j] <= calendars_res[j][current_time_element])

        for i in range(nb_task_calendar):
            model.add_state_constr(
                (current_time <= starts_calendar[i])
                | (~unscheduled_calendar.contains(i))
            )
            model.add_state_constr(
                (current_time <= ends_calendar[i] + 1) | (finished_calendar.contains(i))
            )

        for i in range(self.problem.n_jobs):
            for j_mode in range(len(modes_per_task[i])):
                n_z = [
                    (j, consumption[i][j_mode][j])
                    for j in range(nb_resource)
                    if consumption[i][j_mode][j] > 0
                ]
                dur = durs[i][j_mode]
                if dur > 0:
                    start_task = dp.Transition(
                        name=f"start_{i}_mode_{j_mode}",
                        cost=dp.IntExpr.state_cost(),
                        effects=[
                            (ongoing, ongoing.add(i)),
                            (scheduled, scheduled.add(i)),
                            (modes_var[i], j_mode),
                            (unscheduled, unscheduled.remove(i)),
                            (starts[i], current_time),
                            (starts_element[i], current_time_element),
                        ]
                        + [
                            (
                                current_resource_consumption[j],
                                current_resource_consumption[j] + cons,
                            )
                            for j, cons in n_z
                        ],
                        preconditions=[
                            current_time_element + dur < self.problem.horizon,
                            unscheduled.contains(i),
                            predecessors[i].issubset(finished),
                        ]
                        + [
                            current_resource_consumption[j] + cons <= capacities[j]
                            for j, cons in n_z
                        ]
                        + [
                            cons <= calendars_res[j][current_time_element + d]
                            for j, cons in n_z
                            for d in range(dur)
                        ],  # Avoid to schedule it !!
                    )
                    model.add_transition(start_task)
                else:
                    start_dummy_task = dp.Transition(
                        name=f"start_{i}_mode_{j_mode}",
                        cost=dp.IntExpr.state_cost(),
                        effects=[
                            (scheduled, scheduled.add(i)),
                            (finished, finished.add(i)),
                            (unscheduled, unscheduled.remove(i)),
                            (starts[i], current_time),
                            (starts_element[i], current_time_element),
                        ],
                        preconditions=[
                            unscheduled.contains(i),
                            predecessors[i].issubset(finished),
                        ],
                    )
                    model.add_transition(start_dummy_task, forced=True)
                if dur > 0:
                    ending_task = dp.Transition(
                        name=f"end_{i}_mode_{j_mode}",
                        cost=(dp.max(starts[i] + dur, cur_makespan) - cur_makespan)
                        + dp.IntExpr.state_cost(),
                        # cost=starts[i] + durs[i] + dp.IntExpr.state_cost(),
                        effects=[
                            (current_time, starts[i] + dur),
                            (current_time_element, starts_element[i] + dur),
                            (ongoing, ongoing.remove(i)),
                            (finished, finished.add(i)),
                            (cur_makespan, dp.max(starts[i] + dur, cur_makespan)),
                        ]
                        + [
                            (
                                current_resource_consumption[j],
                                current_resource_consumption[j] - cons,
                            )
                            for j, cons in n_z
                            if is_renewable[j]
                        ],
                        preconditions=[
                            ongoing.contains(i),
                            modes_var[i] == j_mode,
                            starts[i] + dur >= current_time,
                        ],
                    )
                    model.add_transition(ending_task)

        for i in range(nb_task_calendar):
            st = starts_calendar[i]
            end = ends_calendar[i]
            n_z = [(j, consumption_calendar[i][j]) for j in range(nb_resource)]
            tr_calendar = dp.Transition(
                name=f"calendar_{i}",
                cost=dp.IntExpr.state_cost(),
                effects=[
                    (current_time, st),
                    (current_time_element, st),
                    (ongoing_calendar, ongoing_calendar.add(i)),
                    (unscheduled_calendar, unscheduled_calendar.remove(i)),
                ]
                + [
                    (
                        current_resource_consumption[j],
                        current_resource_consumption[j] + cons,
                    )
                    for j, cons in n_z
                ],
                preconditions=[unscheduled_calendar.contains(i), current_time <= st]
                + [
                    current_resource_consumption[j] + cons <= capacities[j]
                    for j, cons in n_z
                ],
            )
            model.add_transition(tr_calendar)
            tr_stop_0 = dp.Transition(
                name=f"calendar_stop_{i}",
                cost=(dp.max(end, cur_makespan) - cur_makespan)
                + dp.IntExpr.state_cost(),
                effects=[
                    (current_time, end),
                    (current_time_element, end),
                    (ongoing_calendar, ongoing_calendar.remove(i)),
                    (finished_calendar, finished_calendar.add(i)),
                    (cur_makespan, dp.max(end, cur_makespan)),
                ]
                + [
                    (
                        current_resource_consumption[j],
                        current_resource_consumption[j] - cons,
                    )
                    for j, cons in n_z
                ],
                preconditions=[
                    current_time >= st,
                    current_time <= end,
                    ongoing_calendar.contains(i),
                    ~unscheduled.is_empty(),
                ],
            )
            model.add_transition(tr_stop_0)
            tr_stop_1 = dp.Transition(
                name=f"calendar_stop_dummy_{i}",
                cost=-5 + dp.IntExpr.state_cost(),
                effects=[
                    (current_time, end),
                    (current_time_element, end),
                    (ongoing_calendar, ongoing_calendar.remove(i)),
                    (finished_calendar, finished_calendar.add(i)),
                ]
                + [
                    (
                        current_resource_consumption[j],
                        current_resource_consumption[j] - cons,
                    )
                    for j, cons in n_z
                ],
                preconditions=[
                    current_time >= st,
                    current_time <= end,
                    ongoing_calendar.contains(i),
                    unscheduled.is_empty(),
                ],
            )
            model.add_transition(tr_stop_1)

        if kwargs["dual_bound"]:
            remaining = model.add_int_table(
                [self.remaining_per_task[t] for t in self.problem.tasks_list]
            )
            model.add_dual_bound(
                (~unscheduled.is_empty()).if_then_else(remaining.max(unscheduled), 0)
            )
        # else:
        #     model.add_dual_bound(0)
        self.starts = starts
        self.modes_per_task = modes_per_task
        self.model = model

    def init_model_task_time_calendar(self, **kwargs: Any) -> None:
        model = dp.Model()
        nb_resource = len(self.problem.resources_list)
        nb_tasks = self.problem.n_jobs
        res_available = [
            [
                model.add_int_var(target=x)
                for x in self.problem.get_resource_availability_array(res)
            ]
            for res in self.problem.resources_list
        ]
        tasks_list = self.problem.tasks_list
        modes_per_task = [
            [m for m in sorted(self.problem.mode_details[t])] for t in tasks_list
        ]
        durs = [
            [
                self.problem.mode_details[tasks_list[i]][m]["duration"]
                for m in modes_per_task[i]
            ]
            for i in range(nb_tasks)
        ]
        consumption = [
            [
                [
                    self.problem.mode_details[tasks_list[i]][m].get(res)
                    for res in self.problem.resources_list
                ]
                for m in modes_per_task[i]
            ]
            for i in range(nb_tasks)
        ]
        prec_ = [set() for t in self.problem.tasks_list]
        for task in self.problem.successors:
            for succ in self.problem.successors[task]:
                prec_[self.problem.index_task[succ]].add(self.problem.index_task[task])

        starts = [model.add_int_var(target=0) for i in range(self.problem.n_jobs)]
        durs_var = [model.add_int_var(target=0) for i in range(self.problem.n_jobs)]
        task_object = model.add_object_type(number=self.problem.n_jobs)
        unscheduled = model.add_set_var(
            object_type=task_object, target=range(self.problem.n_jobs)
        )
        scheduled = model.add_set_var(object_type=task_object, target=set())
        consumption_table = model.add_int_table(consumption)
        predecessors = model.add_set_table(prec_, object_type=task_object)
        model.add_base_case([unscheduled.is_empty()])
        self.transitions = {}
        for i_res in range(len(res_available)):
            for t_res in range(len(res_available[i_res])):
                model.add_state_constr(res_available[i_res][t_res] >= 0)
        current_time = model.add_int_var(target=0)
        for i in range(self.problem.n_jobs):
            for j_mode in range(len(modes_per_task[i])):
                n_z = [
                    (j, consumption[i][j_mode][j])
                    for j in range(nb_resource)
                    if consumption[i][j_mode][j] > 0
                ]
                dur = durs[i][j_mode]
                for time in range(self.problem.horizon - dur):
                    transition = dp.Transition(
                        name=f"schedule_{i}_{time}_{j_mode}",
                        cost=dp.max(current_time, time)
                        - current_time
                        + dp.IntExpr.state_cost(),
                        effects=[
                            (
                                res_available[j][time + k],
                                res_available[j][time + k] - conso,
                            )
                            for j, conso in n_z
                            for k in range(dur)
                        ]
                        + [
                            (current_time, dp.max(current_time, time)),
                            (unscheduled, unscheduled.remove(i)),
                            (scheduled, scheduled.add(i)),
                            (durs_var[i], dur),
                            (starts[i], time),
                        ],
                        preconditions=[
                            unscheduled.contains(i),
                            time >= current_time,
                            predecessors[i].issubset(scheduled),
                        ]
                        + [time >= starts[p] + durs_var[p] for p in prec_[i]]
                        + [
                            res_available[j][time + k] >= conso
                            for j, conso in n_z
                            for k in range(dur)
                        ],
                    )
                    model.add_transition(transition)
                    self.transitions[
                        ("sched", i, modes_per_task[i][j_mode], time)
                    ] = transition
        # model.add_dual_bound(consumption_table[unscheduled, 0])
        self.modes_per_task = modes_per_task
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        if self.modeling == DpRcpspModeling.TASK_AND_TIME:
            return self.retrieve_solution_task_time(sol)
        if self.modeling in {
            DpRcpspModeling.TASK_ORIGINAL,
            DpRcpspModeling.TASK_AVAIL_TASK_UPD,
        }:
            return self.retrieve_solution_task(sol)
        if self.modeling in {DpRcpspModeling.TASK_MULTIMODE}:
            return self.retrieve_solution_multimode(sol)

    def retrieve_solution_task_time(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        schedule = {}
        modes_dict = {}
        for t in sol.transitions:
            x = extract_ints(t.name)
            if len(x) == 3:
                index_task, time, j_mode = x
                task = self.problem.tasks_list[index_task]
                modes_dict[task] = self.modes_per_task[index_task][j_mode]
            else:
                index_task, time = x
                task = self.problem.tasks_list[index_task]
                modes_dict[task] = 1
            dur = self.problem.mode_details[task][modes_dict[task]]["duration"]
            schedule[task] = {"start_time": time, "end_time": time + dur}
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=schedule,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
        )

    def retrieve_solution_task(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        durs = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        schedule = {}
        state = self.model.target_state
        for t in sol.transitions:
            state = t.apply(state, self.model)
            if "start" in t.name:
                index_task = extract_ints(t.name)[0]
                task = self.problem.tasks_list[index_task]
                start_time = state[self.starts[index_task]]
                schedule[task] = {
                    "start_time": start_time,
                    "end_time": start_time + durs[index_task],
                }
        return RcpspSolution(problem=self.problem, rcpsp_schedule=schedule)

    def retrieve_solution_multimode(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        schedule = {}
        state = self.model.target_state
        modes_dict = {}
        for t in sol.transitions:
            state = t.apply(state, self.model)
            logger.debug(f"{t.name}")
            if "start" in t.name:
                index_task, mode_task = extract_ints(t.name)
                task = self.problem.tasks_list[index_task]
                mode = self.modes_per_task[index_task][mode_task]
                modes_dict[task] = mode
                dur = self.problem.mode_details[task][mode]["duration"]
                start_time = state[self.starts[index_task]]
                schedule[task] = {
                    "start_time": start_time,
                    "end_time": start_time + dur,
                }
        return RcpspSolution(
            problem=self.problem,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
            rcpsp_schedule=schedule,
        )

    def set_warm_start(self, solution: RcpspSolution) -> None:
        if self.modeling in {DpRcpspModeling.TASK_AND_TIME}:
            initial_solution = []
            sorted_tasks = sorted(
                solution.rcpsp_schedule,
                key=lambda x: (solution.get_start_time(x), solution.get_end_time(x)),
            )
            for t in sorted_tasks:
                if t in self.problem.tasks_list_non_dummy:
                    m = solution.get_mode(t)
                else:
                    m = 1
                i_t = self.problem.index_task[t]
                initial_solution.append(
                    self.transitions[("sched", i_t, m, solution.get_start_time(t))]
                )
            self.initial_solution = initial_solution
        if self.modeling in {
            DpRcpspModeling.TASK_ORIGINAL,
            DpRcpspModeling.TASK_AVAIL_TASK_UPD,
            DpRcpspModeling.TASK_MULTIMODE,
        }:
            initial_solution = []
            sorted_tasks = sorted(
                solution.rcpsp_schedule,
                key=lambda x: (solution.get_start_time(x), solution.get_end_time(x)),
            )
            flattened_schedule = []
            for t in sorted_tasks:
                st, end = solution.get_start_time(t), solution.get_end_time(t)
                if st == end:
                    continue
                flattened_schedule.append(("start", self.problem.index_task[t], st))
                flattened_schedule.append(("end", self.problem.index_task[t], end))
            flattened_schedule = sorted(flattened_schedule, key=lambda x: x[2])
            flattened_schedule = (
                [
                    (
                        "start",
                        self.problem.index_task[self.problem.source_task],
                        solution.get_start_time(self.problem.source_task),
                    )
                ]
                + flattened_schedule
                + [
                    (
                        "start",
                        self.problem.index_task[self.problem.sink_task],
                        solution.get_start_time(self.problem.sink_task),
                    )
                ]
            )
            for tag, t, time_ in flattened_schedule:
                task = self.problem.tasks_list[t]
                if task in self.problem.tasks_list_non_dummy:
                    m = solution.get_mode(task)
                else:
                    m = 1
                i_t = t
                if tag == "start":
                    initial_solution.append(self.transitions[("start", i_t, m)])
                else:
                    initial_solution.append(self.transitions[("end", i_t, m)])
        if (
            self.modeling == DpRcpspModeling.TASK_MULTIMODE
            and self.problem.is_varying_resource()
        ):
            ##  TODO.. this is boring.
            pass

        pass
