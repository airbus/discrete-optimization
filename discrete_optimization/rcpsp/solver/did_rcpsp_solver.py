#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import re
from enum import Enum
from typing import Any

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.dyn_prog_tools import DidSolver, dp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.solver.cpm import CPM, CPMObject
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP


class DidRCPSPModeling(Enum):
    TASK_AND_TIME = 0
    TASK_ORIGINAL = 1
    TASK_AVAIL_TASK_UPD = 2
    TASK_MULTIMODE = 3


class DidRCPSPSolver(DidSolver, SolverRCPSP):
    hyperparameters = DidSolver.hyperparameters
    problem: RCPSPModel
    modeling: DidRCPSPModeling

    def __init__(self, problem: RCPSPModel, **kwargs: Any):
        super().__init__(problem, **kwargs)
        cpm = CPM(problem=self.problem)
        cpm.run_classic_cpm()
        result = cpm.map_node
        self.remaining_per_task = {
            n: result[self.problem.sink_task]._EFD - result[n]._EFD for n in result
        }
        self.result_cpm = result

    def init_model(
        self,
        modeling: DidRCPSPModeling = DidRCPSPModeling.TASK_MULTIMODE,
        **kwargs: Any,
    ):

        if modeling == DidRCPSPModeling.TASK_AND_TIME:
            self.init_model_task_time(**kwargs)
        if modeling == DidRCPSPModeling.TASK_ORIGINAL:
            self.init_model_task(**kwargs)
        if modeling == DidRCPSPModeling.TASK_AVAIL_TASK_UPD:
            self.init_model_task_upd(**kwargs)
        if modeling == DidRCPSPModeling.TASK_MULTIMODE:
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
        # current_sched_time = model.add_int_var(target=0)
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
        model.add_dual_bound(consumption_table[unscheduled, 0])
        self.model = model

    def init_model_task(self, **kwargs: Any) -> None:
        model = dp.Model()
        task_object = model.add_object_type(number=self.problem.n_jobs)
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
        # advance_time = dp.Transition(name="advance",
        #                              cost=dp.IntExpr.state_cost(),
        #                              effects=[(current_time, current_time+1)],
        #                              preconditions=[sum(can_start) == 0])
        # model.add_transition(advance_time)
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
        remaining = model.add_int_table(
            [self.remaining_per_task[t] for t in self.problem.tasks_list]
        )
        model.add_dual_bound(
            (~unscheduled.is_empty()).if_then_else(remaining.max(unscheduled), 0)
        )
        self.starts = starts
        self.modes_per_task = modes_per_task
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        if self.modeling == DidRCPSPModeling.TASK_AND_TIME:
            return self.retrieve_solution_task_time(sol)
        if self.modeling in {
            DidRCPSPModeling.TASK_ORIGINAL,
            DidRCPSPModeling.TASK_AVAIL_TASK_UPD,
        }:
            return self.retrieve_solution_task(sol)
        if self.modeling in {DidRCPSPModeling.TASK_MULTIMODE}:
            return self.retrieve_solution_multimode(sol)

    def retrieve_solution_task_time(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        durs = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        schedule = {}
        for t in sol.transitions:
            index_task, time = extract_ints(t.name)
            task = self.problem.tasks_list[index_task]
            schedule[task] = {"start_time": time, "end_time": time + durs[index_task]}
        return RCPSPSolution(problem=self.problem, rcpsp_schedule=schedule)

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
        return RCPSPSolution(problem=self.problem, rcpsp_schedule=schedule)

    def retrieve_solution_multimode(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        schedule = {}
        state = self.model.target_state
        modes_dict = {}
        for t in sol.transitions:
            state = t.apply(state, self.model)
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
        return RCPSPSolution(
            problem=self.problem,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
            rcpsp_schedule=schedule,
        )
