#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import dataclasses
from typing import Any, Dict, Iterable, Optional, Union

try:
    import optalcp as cp
except ImportError:
    cp = None
    optalcp_available = False
else:
    optalcp_available = True
import logging

import numpy as np

from discrete_optimization.flex_scheduling.fsp_utils import (
    compute_duration_function_time_cluster,
    get_lb_ub_start_end_date,
    get_lb_ub_start_end_date_group_of_task,
)
from discrete_optimization.flex_scheduling.problem import (
    RESOURCE_KEY,
    FlexProblem,
    GroupType,
    ObjectiveParamEarliness,
    ObjectiveParamResource,
    ObjectiveParamTardiness,
    ObjectivesEnum,
    ResourceData,
    ScheduleSolution,
    TaskData,
)
from discrete_optimization.flex_scheduling.solvers.cpsat import (
    build_multiple_cumulative_constraints_inputs,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConstraintIncludingParams:
    include_calendar_and_duration: bool = dataclasses.field(default=True)
    include_non_released_resource: bool = dataclasses.field(default=True)
    include_group_variables: bool = dataclasses.field(default=True)
    include_constraint_precedence_on_groups: bool = dataclasses.field(default=True)
    include_constraints_on_groups: bool = dataclasses.field(default=True)
    include_generalized_time_constraints: bool = dataclasses.field(default=True)
    include_variable_resource: bool = dataclasses.field(default=True)
    add_precedence_non_release_variables: bool = dataclasses.field(default=True)
    include_cumulative_constraint: bool = dataclasses.field(default=True)
    include_reservoir_constraint_non_release: bool = dataclasses.field(default=False)
    max_length_non_release: int | None = dataclasses.field(default=None)
    synchro_instead_of_non_release: bool = dataclasses.field(default=False)


def compute_duration_tasks_function_time_and_resource_calendars(
    problem: FlexProblem,
) -> tuple[Any, dict[tuple, np.ndarray], dict[tuple[int, int], tuple]]:
    method = compute_duration_function_time_cluster
    resource_calendar_dict = {
        problem.resources[i].id: problem.resources[i].calendar_availability > 0
        for i in range(len(problem.resources))
    }
    cumulative_calendar_dict = {
        r: np.cumsum(resource_calendar_dict[r]) for r in resource_calendar_dict
    }
    durations = {
        (i, m): None for i in range(problem.nb_tasks) for m in problem.tasks[i].modes
    }
    task_mode_to_calendar = {}
    for i in range(problem.nb_tasks):
        for m in problem.tasks[i].modes:
            task_data: TaskData = problem.tasks[i].modes[m]
            resource_non_zeros = [
                r
                for r in task_data.resource_consumption
                if task_data.resource_consumption[r] > 0
            ]
            if len(resource_non_zeros) == 0:
                durations[i, m] = ([], {task_data.duration: [[0, problem.horizon]]})
            elif len(resource_non_zeros) == 1:
                # One resource pool is used.
                orig_duration = task_data.duration
                res_consumption = task_data.resource_consumption[resource_non_zeros[0]]
                c = (
                    problem.resources[
                        problem.resource_id_to_index[resource_non_zeros[0]]
                    ].calendar_availability
                    >= res_consumption
                )
                durations[i, m] = method(
                    orig_duration=orig_duration,
                    resource_calendar=c,  # resource_calendar_dict[resource_non_zeros[0]],
                    cumulative_resource_calendar=np.cumsum(c),
                    # cumulative_calendar_dict[
                    #     resource_non_zeros[0]
                    # ],
                )
                resource_calendar_dict[(resource_non_zeros[0], res_consumption)] = c
                task_mode_to_calendar[i, m] = (resource_non_zeros[0], res_consumption)
            else:
                orig_duration = task_data.duration
                tuple_res = tuple(
                    [(r, task_data.resource_consumption[r]) for r in resource_non_zeros]
                )
                if tuple_res not in resource_calendar_dict:
                    # For the first resource in the tuple, b  "availability >= consumption"
                    first_res_id, first_consumption = tuple_res[0]
                    b = (
                        problem.resources[
                            problem.resource_id_to_index[first_res_id]
                        ].calendar_availability
                        >= first_consumption
                    )

                    for res_id, cons in tuple_res[1:]:
                        b &= (
                            problem.resources[
                                problem.resource_id_to_index[res_id]
                            ].calendar_availability
                            >= cons
                        )
                    resource_calendar_dict[tuple_res] = b
                    cumulative_calendar_dict[tuple_res] = np.cumsum(
                        resource_calendar_dict[tuple_res]
                    )
                durations[i, m] = method(
                    orig_duration=orig_duration,
                    resource_calendar=resource_calendar_dict[tuple_res],
                    cumulative_resource_calendar=cumulative_calendar_dict[tuple_res],
                )
                task_mode_to_calendar[i, m] = tuple_res
    return durations, resource_calendar_dict, task_mode_to_calendar


class OptalFlexProblemSolver(OptalCpSolver):
    problem: FlexProblem
    current_objective: str

    def __init__(
        self,
        problem: FlexProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **args,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function, **args
        )
        self.variables_dict = {}
        self.index_to_id = self.problem.index_to_task_id
        self.id_to_index = self.problem.task_id_to_index
        self.group_id_to_index = self.problem.group_id_to_index
        self.nb_tasks = self.problem.nb_tasks
        (
            self.min_start_time,
            self.max_start_time,
            self.min_end_time,
            self.max_end_time,
        ) = get_lb_ub_start_end_date(problem=self.problem)

        # self.min_start_time = {i: 0 for i in self.min_start_time}
        # self.min_end_time = {i: 0 for i in self.min_start_time}
        # self.max_start_time = {i: self.problem.horizon for i in self.min_start_time}
        # self.max_end_time = {i: self.problem.horizon for i in self.min_start_time}
        # (
        #     self.min_start_time,
        #     self.max_start_time,
        #     self.min_end_time,
        #     self.max_end_time,
        # ) = get_lb_ub_start_end_date(problem=self.problem)
        self.durations, self.resource_calendar_dict, self.task_mode_to_calendar = (
            compute_duration_tasks_function_time_and_resource_calendars(self.problem)
        )
        self.cur_sol = None
        self.current_objective = None

    def set_warm_start_from_previous_run(self):
        """Set warm start from previous run of the solver."""
        if self.status_solver is None:
            return
        self.cur_sol: cp.Solution
        sol = self.retrieve_sol_from_solver_solution(self.cur_sol)
        self.cur_sol.set_objective(sol._intern_obj[self.current_objective])
        print("Current objective", self.current_objective, self.cur_sol.get_objective())
        self.warm_start_solution = self.cur_sol
        self.use_warm_start = True

    def set_warm_start(self, solution: ScheduleSolution) -> None:
        """
        Creates an OptalCP Solution object from a discrete-optimization ScheduleSolution
        and registers it as a warm start for the solver.
        """
        sol_cp = cp.Solution()
        evaluation = self.problem.evaluate(solution)

        # 1. Set Main Intervals
        for i in range(self.nb_tasks):
            if i in self.variables_dict["main_interval"]:
                start = int(solution.schedule[i, 0])
                end = int(solution.schedule[i, 1])
                sol_cp.set_value(self.variables_dict["main_interval"][i], start, end)

        # 2. Set Optional Intervals
        for i in range(self.nb_tasks):
            if i not in self.variables_dict["opt_interval"]:
                continue
            selected_mode = solution.modes[i]
            modes = list(self.problem.tasks[i].modes.keys())
            for m in modes:
                var = self.variables_dict["opt_interval"][i][m]
                if m == selected_mode:
                    start = int(solution.schedule[i, 0])
                    end = int(solution.schedule[i, 1])
                    sol_cp.set_value(var, start, end)
                elif len(modes) > 1:
                    sol_cp.set_absent(var)

        # 3. Set Group Intervals
        if "group_interval_per_id" in self.variables_dict:
            for g_id, var in self.variables_dict["group_interval_per_id"].items():
                g_idx = self.group_id_to_index[g_id]
                group = self.problem.tasks_group[g_idx]
                t_indices = [self.id_to_index[t] for t in group.tasks_group]
                if t_indices:
                    g_start = int(np.min(solution.schedule[t_indices, 0]))
                    g_end = int(np.max(solution.schedule[t_indices, 1]))
                    sol_cp.set_value(var, g_start, g_end)

        # 4. Set Intervals of Non-Released Resource (The Fix)
        if "non_release_intervals_map" in self.variables_dict:
            for key, itv in self.variables_dict["non_release_intervals_map"].items():
                type_constraint = key[0]
                is_present = True
                start_val, end_val = 0, 0

                if type_constraint == "simple":
                    _, t1, t2 = key
                    i1 = self.id_to_index[t1]
                    i2 = self.id_to_index[t2]
                    start_val = int(solution.schedule[i1, 1])  # End of t1
                    end_val = int(solution.schedule[i2, 0])  # Start of t2

                elif type_constraint == "mode":
                    _, t1, mode, t2 = key
                    i1 = self.id_to_index[t1]
                    i2 = self.id_to_index[t2]
                    if solution.modes[i1] == mode:
                        start_val = int(solution.schedule[i1, 1])
                        end_val = int(solution.schedule[i2, 0])
                    else:
                        is_present = False

                elif type_constraint == "generic":
                    _, (tag1, tag2) = key

                    # Compute start (end of tag1)
                    if tag1[0] == "task":
                        i1 = self.id_to_index[tag1[1]]
                        start_val = int(solution.schedule[i1, 1])
                    else:  # group
                        g_idx = self.group_id_to_index[tag1[1]]
                        group = self.problem.tasks_group[g_idx]
                        t_indices = [self.id_to_index[t] for t in group.tasks_group]
                        start_val = int(np.max(solution.schedule[t_indices, 1]))

                    # Compute end (start of tag2)
                    if tag2[0] == "task":
                        i2 = self.id_to_index[tag2[1]]
                        end_val = int(solution.schedule[i2, 0])
                    else:  # group
                        g_idx = self.group_id_to_index[tag2[1]]
                        group = self.problem.tasks_group[g_idx]
                        t_indices = [self.id_to_index[t] for t in group.tasks_group]
                        end_val = int(np.min(solution.schedule[t_indices, 0]))

                if is_present:
                    sol_cp.set_value(itv, start_val, end_val)
                else:
                    sol_cp.set_absent(itv)

        # 5. Set Resource Capacities
        if "resource_capacity_variables" in self.variables_dict:
            res_consumptions = evaluation.get("resource_consumption", {})
            for r_id, var in self.variables_dict["resource_capacity_variables"].items():
                sol_cp.set_value(
                    var,
                    int(
                        np.max(
                            res_consumptions[self.problem.resource_id_to_index[r_id], :]
                        )
                    ),
                )

        # 6. Objectives
        if "obj_data" in self.variables_dict:
            _, weights, names = self.variables_dict["obj_data"]
            total_obj = 0.0
            for name, weight in zip(names, weights):
                val = evaluation.get(name, 0)
                total_obj += val * weight
            sol_cp.set_objective(int(total_obj))
            for i in range(len(self.variables_dict["obj_data"][2])):
                sol_cp.set_value(
                    self.variables_dict["obj_var"][i],
                    int(evaluation.get(self.variables_dict["obj_data"][2][i], 0)),
                )

        if "artificial_var" in self.variables_dict:
            sol_cp.set_value(
                self.variables_dict["artificial_var"], cp.IntervalMin, cp.IntervalMax
            )
        sol_cp.set_objective(int(evaluation[self.current_objective]))
        self.warm_start_solution = sol_cp
        self.use_warm_start = True

    def retrieve_sol_from_solver_solution(self, solution: "cp.Solution"):
        schedule = np.zeros((self.problem.nb_tasks, 2))
        allocation = np.zeros(self.problem.nb_tasks)
        schedule_l = {}
        for i in range(self.problem.nb_tasks):
            schedule_l[i] = []
            schedule[i, 0] = solution.get_start(self.variables_dict["main_interval"][i])
            schedule[i, 1] = solution.get_end(self.variables_dict["main_interval"][i])
            for m in self.variables_dict["opt_interval"][i]:
                if solution.is_present(self.variables_dict["opt_interval"][i][m]):
                    allocation[i] = m
        sol = ScheduleSolution(
            problem=self.problem, schedule=schedule, modes=allocation
        )
        details_subobj = {}
        for i, name in enumerate(self.variables_dict["obj_data"][2]):
            # logger.debug(f"{name} : {kpis[name]}")
            details_subobj[name] = solution.get_value(self.variables_dict["obj_var"][i])
            # details_subobj[name] = int(kpis[name])
        sol._intern_obj = details_subobj
        self.cur_sol = solution
        return sol

    def retrieve_solution(self, result: "cp.SolveResult") -> ScheduleSolution:
        # return ScheduleSolution(problem=self.problem,
        #                        schedule = np.zeros((self.problem.nb_tasks, 2)),
        #                        modes = np.ones(self.problem.nb_tasks))
        return self.retrieve_sol_from_solver_solution(result.solution)

    def init_model(self, params: ConstraintIncludingParams = None, **args: Any) -> None:
        if params is None:
            params = ConstraintIncludingParams()
        self.cp_model = cp.Model()
        self.create_base_intervals()
        self.create_opt_intervals()
        self.alternative_modes()
        self.create_calendar_step_function()
        if not params.include_calendar_and_duration:
            self.add_duration_constraint_no_calendar()
        else:
            self.add_duration_constraint_integral()
            # self.add_duration_constraint_element()
        self.constraint_precedence()
        if params.include_variable_resource:
            self.init_resource_variables()
        if params.include_group_variables:
            self.init_group_variables()
            self.constraint_non_overlap_group()
            self.constraint_precedence_on_groups()
        if params.include_non_released_resource:
            self.init_intervals_of_non_released_resource(params)
            if params.include_reservoir_constraint_non_release:
                for r in self.problem.resources:
                    self.constraint_reservoir_non_release_res(resource=r)
            self.constraint_group_non_release_resource()
        if params.synchro_instead_of_non_release:
            self.create_synchro_of_non_released_resource(params)
            self.constraint_group_non_release_resource()
        if params.include_cumulative_constraint:
            self.constraint_cumulative(params=params)
        self.create_objectives(params)
        # self.cp_model.minimize(self.cp_model.max([self.cp_model.end(self.variables_dict["main_interval"][i])
        #                                           for i in self.variables_dict["main_interval"]]))

    def create_base_intervals(self):
        itv_dict = {}
        for i in range(self.nb_tasks):
            possible_duration = [
                self.problem.tasks[i].modes[m].duration
                for m in self.problem.tasks[i].modes
            ]
            length_tuple = (min(possible_duration), None)
            if min(possible_duration) == max(possible_duration) == 0:
                length_tuple = 0
            itv_dict[i] = self.cp_model.interval_var(
                start=(int(self.min_start_time[i]), int(self.max_start_time[i])),
                end=(
                    int(self.min_end_time[i]),
                    int(self.max_end_time[i])
                    if not self.problem.tasks[i].soft_max_end_date
                    else int(self.problem.horizon),
                ),
                length=length_tuple,
                optional=False,
                name=f"task_{i}",
            )
        self.variables_dict["main_interval"] = itv_dict

    def create_opt_intervals(self):
        opt_itv_dict = {i: {} for i in range(self.nb_tasks)}
        for i in range(self.nb_tasks):
            task = self.problem.tasks[i]
            modes = list(task.modes.keys())
            if len(modes) == 1:
                opt_itv_dict[i][modes[0]] = self.variables_dict["main_interval"][i]
            else:
                for m in modes:
                    opt_itv_dict[i][m] = self.cp_model.interval_var(
                        start=(
                            int(self.min_start_time[i]),
                            int(self.max_start_time[i]),
                        ),
                        end=(
                            int(self.min_end_time[i]),
                            int(self.max_end_time[i])
                            if not self.problem.tasks[i].soft_max_end_date
                            else int(self.problem.horizon),
                        ),
                        length=None,
                        optional=True,
                        name=f"task_{i}_{m}",
                    )
        self.variables_dict["opt_interval"] = opt_itv_dict

    def alternative_modes(self):
        for i in range(self.nb_tasks):
            if len(self.variables_dict["opt_interval"][i]) > 1:
                self.cp_model.alternative(
                    self.variables_dict["main_interval"][i],
                    [
                        self.variables_dict["opt_interval"][i][m]
                        for m in self.variables_dict["opt_interval"][i]
                    ],
                )

    def create_calendar_step_function(self):
        self.calendar_step_functions = {}
        for i, m in self.task_mode_to_calendar:
            key = self.task_mode_to_calendar[i, m]
            if key in self.calendar_step_functions:
                continue
            array = self.resource_calendar_dict[key]
            initial_value = array[0]
            list_val = [(0, int(initial_value))]
            for t in range(1, array.shape[0]):
                if array[t] != array[t - 1]:
                    list_val.append((t, int(array[t])))
            self.calendar_step_functions[key] = self.cp_model.step_function(list_val)

    def add_duration_constraint_no_calendar(self):
        for i, m in self.task_mode_to_calendar:
            key = self.task_mode_to_calendar[i, m]
            duration = self.problem.tasks[i].modes[m].duration
            nb_modes = len(self.problem.tasks[i].modes)
            if nb_modes == 1:
                itv = self.variables_dict["main_interval"][i]
            else:
                itv: cp.IntervalVar = self.variables_dict["opt_interval"][i][m]
            self.cp_model.enforce(self.cp_model.length(itv) == duration)

    def add_duration_constraint_integral(self):
        for i, m in self.task_mode_to_calendar:
            key = self.task_mode_to_calendar[i, m]
            duration = self.problem.tasks[i].modes[m].duration
            nb_modes = len(self.problem.tasks[i].modes)
            if nb_modes == 1:
                itv = self.variables_dict["main_interval"][i]
                self.cp_model.forbid_start(itv, self.calendar_step_functions[key])
                # self.cp_model.forbid_end(itv, self.calendar_step_functions[key])
                # not really because of convention what we want
                self.cp_model.enforce(
                    self.cp_model.eval(self.calendar_step_functions[key], itv.end() - 1)
                    != 0
                )
                # Avoid to artificially finish later than needed
                self.cp_model.enforce(
                    self.cp_model.integral(self.calendar_step_functions[key], itv)
                    == int(duration)
                )
                continue
            itv: cp.IntervalVar = self.variables_dict["opt_interval"][i][m]
            presence_of_mode = itv.presence()
            self.cp_model.forbid_start(itv, self.calendar_step_functions[key])
            # self.cp_model.forbid_end(itv, self.calendar_step_functions[key])
            # not really because of convention what we want
            self.cp_model.enforce(
                self.cp_model.eval(self.calendar_step_functions[key], itv.end() - 1)
                != 0
            )
            # Avoid to artificially start/finish later than needed
            self.cp_model.enforce(
                self.cp_model.implies(
                    presence_of_mode,
                    self.cp_model.integral(self.calendar_step_functions[key], itv)
                    == int(duration),
                )
            )

    def add_duration_constraint_element(self):
        duration_step_function = {}
        for i, m in self.task_mode_to_calendar:
            key = self.task_mode_to_calendar[i, m]
            duration = self.problem.tasks[i].modes[m].duration
            nb_modes = len(self.problem.tasks[i].modes)
            if duration == 0:
                if nb_modes == 1:
                    itv = self.variables_dict["main_interval"][i]
                else:
                    itv: cp.IntervalVar = self.variables_dict["opt_interval"][i][m]
                self.cp_model.enforce(self.cp_model.length(itv) == 0)
                continue
            dur_array = self.durations[(i, m)][0]
            list_val = [(0, int(dur_array[0]))]
            for t in range(1, dur_array.shape[0]):
                if dur_array[t] != dur_array[t - 1]:
                    list_val.append((t, int(dur_array[t])))
            duration_step_function[(i, m)] = self.cp_model.step_function(list_val)
            if nb_modes == 1:
                itv = self.variables_dict["main_interval"][i]
                self.cp_model.forbid_start(itv, self.calendar_step_functions[key])
                length = itv.length()
                start = itv.start()
                self.cp_model.enforce(
                    self.cp_model.eval(duration_step_function[(i, m)], start) == length
                )
            itv: cp.IntervalVar = self.variables_dict["opt_interval"][i][m]
            presence_of_mode = itv.presence()
            self.cp_model.forbid_start(itv, self.calendar_step_functions[key])
            length = itv.length()
            start = itv.start()
            self.cp_model.enforce(
                self.cp_model.implies(
                    presence_of_mode,
                    self.cp_model.eval(duration_step_function[(i, m)], start) == length,
                )
            )

    def init_group_variables(self):
        if len(self.problem.tasks_group) == 0:
            return
        (
            gmin_start_time,
            gmax_start_time,
            gmin_end_time,
            gmax_end_time,
        ) = get_lb_ub_start_end_date_group_of_task(self.problem)
        group_interval_per_id = {}
        for group in self.problem.tasks_group:
            group_id = group.id
            ft = group.first_task_if_any
            lt = group.last_task_if_any
            index_task_in_group = {
                self.problem.task_id_to_index[t] for t in group.tasks_group
            }
            interval_group = self.cp_model.interval_var(
                start=(int(gmin_start_time[group_id]), int(gmax_start_time[group_id])),
                end=(int(gmin_end_time[group_id]), int(gmax_end_time[group_id])),
                length=None,
                optional=False,
                name=f"group_{group_id}",
            )
            if ft is not None:
                self.cp_model.start_at_start(
                    self.variables_dict["main_interval"][
                        self.problem.task_id_to_index[ft]
                    ],
                    interval_group,
                )
            if lt is not None:
                self.cp_model.end_at_end(
                    self.variables_dict["main_interval"][
                        self.problem.task_id_to_index[lt]
                    ],
                    interval_group,
                )
            self.cp_model.span(
                interval_group,
                [self.variables_dict["main_interval"][x] for x in index_task_in_group],
            )
            group_interval_per_id[group_id] = interval_group
        self.variables_dict["group_interval_per_id"] = group_interval_per_id

    def init_intervals_of_non_released_resource(
        self, params: ConstraintIncludingParams
    ):
        intervals_non_release = {}
        # New: Store map to retrieve these variables for warm start
        self.variables_dict["non_release_intervals_map"] = {}

        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor
            is not None
        ):
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor
            for t1, t2, d_res in data:
                if t1 in self.problem.task_id_to_index:
                    i1 = self.problem.task_id_to_index[t1]
                    i2 = self.problem.task_id_to_index[t2]
                    itv = self.cp_model.interval_var(
                        name=f"interval_end_{i1}_start_{i2}",
                    )
                    self.cp_model.start_at_end(
                        itv, self.variables_dict["main_interval"][i1]
                    )
                    self.cp_model.end_at_start(
                        itv, self.variables_dict["main_interval"][i2]
                    )

                    # Store mapping
                    self.variables_dict["non_release_intervals_map"][
                        ("simple", t1, t2)
                    ] = itv
                    if params.add_precedence_non_release_variables:
                        self.cp_model.end_before_start(
                            self.variables_dict["main_interval"][i1],
                            self.variables_dict["main_interval"][i2],
                            itv.length(),
                        )
                    for res in d_res:
                        if res not in intervals_non_release:
                            intervals_non_release[res] = []
                        intervals_non_release[res].append((itv, d_res[res]))
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            is not None
        ):
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            for (t1, mode), t2, d_res in data:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]

                itv = self.cp_model.interval_var(
                    optional=True, name=f"interval_end_{i1}_{mode}_start_{i2}"
                )
                self.cp_model.enforce(
                    itv.presence()
                    == self.variables_dict["opt_interval"][i1][mode].presence()
                )
                self.cp_model.start_at_end(
                    itv, self.variables_dict["opt_interval"][i1][mode]
                )
                self.cp_model.end_at_start(
                    itv, self.variables_dict["main_interval"][i2]
                )

                # Store mapping
                self.variables_dict["non_release_intervals_map"][
                    ("mode", t1, mode, t2)
                ] = itv
                if params.add_precedence_non_release_variables:
                    self.cp_model.end_before_start(
                        self.variables_dict["opt_interval"][i1][mode],
                        self.variables_dict["main_interval"][i2],
                        itv.length(),
                    )

                for res in d_res:
                    if res not in intervals_non_release:
                        intervals_non_release[res] = []
                    intervals_non_release[res].append((itv, d_res[res]))
        if (
            self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            is not None
        ):
            data = self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            for t1, t2, d_res in data:
                tag = []
                if t1.is_a_task:
                    i1 = self.problem.task_id_to_index[t1.task_id]
                    itv1 = self.variables_dict["main_interval"][i1]
                    min_end_time = self.min_end_time[i1]
                    tag.append(("task", t1.task_id))
                else:
                    group = t1.group_id
                    gr = self.problem.tasks_group[self.problem.group_id_to_index[group]]
                    min_end_time = max(
                        [
                            self.min_end_time[self.problem.task_id_to_index[i]]
                            for i in gr.tasks_group
                        ]
                    )
                    itv1 = self.variables_dict["group_interval_per_id"][group]
                    tag.append(("group", group))
                if t2.is_a_task:
                    i2 = self.problem.task_id_to_index[t2.task_id]
                    max_start_time = self.max_start_time[i2]
                    itv2 = self.variables_dict["main_interval"][i2]
                    tag.append(("task", t2.task_id))
                else:
                    group = t2.group_id
                    gr = self.problem.tasks_group[self.problem.group_id_to_index[group]]
                    max_start_time = min(
                        [
                            self.max_start_time[self.problem.task_id_to_index[i]]
                            for i in gr.tasks_group
                        ]
                    )
                    itv2 = self.variables_dict["group_interval_per_id"][group]
                    tag.append(("group", group))

                tag = tuple(tag)
                itv = self.cp_model.interval_var(
                    start=(min_end_time, max_start_time),
                    end=(min_end_time, max_start_time),
                    optional=False,
                    length=(0, params.max_length_non_release),
                    # int(max_start_time - min_end_time)),
                    name=f"interval_end_{tag[0]}_start_{tag[1]}",
                )
                self.cp_model.start_at_end(itv, itv1)
                self.cp_model.end_at_start(itv, itv2)
                if params.add_precedence_non_release_variables:
                    self.cp_model.end_before_start(itv1, itv2)
                #                                   itv.length())
                # Store mapping
                self.variables_dict["non_release_intervals_map"][("generic", tag)] = itv

                for res in d_res:
                    if res not in intervals_non_release:
                        intervals_non_release[res] = []
                    intervals_non_release[res].append((itv, d_res[res]))

        # for ind_group, group in enumerate(self.problem.tasks_group):
        #     if group.type_of_group == GroupType.GROUP_TASK_NON_RELEASED_RESOURCE:
        #         if group.res_not_released is not None:
        #             for res in group.res_not_released:
        #                 qty = group.res_not_released[res]
        #                 if res not in intervals_non_release:
        #                     intervals_non_release[res] = []
        #                 intervals_non_release[res].append(
        #                     (self.variables_dict["group_interval_per_id"][group.id],
        #                      qty)
        #                 )
        self.variables_dict["intervals_non_release"]: Dict[
            RESOURCE_KEY, list[tuple[cp.IntervalVar, int]]
        ] = intervals_non_release

    def create_synchro_of_non_released_resource(
        self, params: ConstraintIncludingParams
    ):
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor
            is not None
        ):
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor
            for t1, t2, d_res in data:
                if t1 in self.problem.task_id_to_index:
                    i1 = self.problem.task_id_to_index[t1]
                    i2 = self.problem.task_id_to_index[t2]
                    self.cp_model.start_at_end(
                        self.variables_dict["main_interval"][i2],
                        self.variables_dict["main_interval"][i1],
                    )
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            is not None
        ):
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            for (t1, mode), t2, d_res in data:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                self.cp_model.start_at_end(
                    self.variables_dict["main_interval"][i2],
                    self.variables_dict["opt_interval"][i1][mode],
                )

        if (
            self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            is not None
        ):
            data = self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            for t1, t2, d_res in data:
                tag = []
                if t1.is_a_task:
                    i1 = self.problem.task_id_to_index[t1.task_id]
                    itv1 = self.variables_dict["main_interval"][i1]
                    tag.append(("task", t1.task_id))
                else:
                    group = t1.group_id
                    gr = self.problem.tasks_group[self.problem.group_id_to_index[group]]
                    itv1 = self.variables_dict["group_interval_per_id"][group]
                    tag.append(("group", group))
                if t2.is_a_task:
                    i2 = self.problem.task_id_to_index[t2.task_id]
                    itv2 = self.variables_dict["main_interval"][i2]
                    tag.append(("task", t2.task_id))
                else:
                    group = t2.group_id
                    gr = self.problem.tasks_group[self.problem.group_id_to_index[group]]
                    itv2 = self.variables_dict["group_interval_per_id"][group]
                    tag.append(("group", group))
                self.cp_model.start_at_end(itv2, itv1)

    def constraint_reservoir_non_release_res(self, resource: ResourceData):
        res_id = resource.id
        list_time_level = []
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor
            is not None
        ):
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor
            for t1, t2, d_res in data:
                if t1 in self.problem.task_id_to_index and res_id in d_res:
                    i1 = self.problem.task_id_to_index[t1]
                    i2 = self.problem.task_id_to_index[t2]

                    list_time_level += [
                        (self.variables_dict["main_interval"][i1], "end", d_res[res_id])
                    ]
                    list_time_level += [
                        (
                            self.variables_dict["main_interval"][i2],
                            "start",
                            -d_res[res_id],
                        )
                    ]
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            is not None
        ):
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            for (t1, mode), t2, d_res in data:
                if res_id in d_res:
                    i1 = self.problem.task_id_to_index[t1]
                    i2 = self.problem.task_id_to_index[t2]
                    list_time_level += [
                        (
                            self.variables_dict["opt_interval"][i1][mode],
                            "end",
                            d_res[res_id],
                        )
                    ]
                    list_time_level += [
                        (
                            self.variables_dict["main_interval"][i2],
                            "start",
                            -d_res[res_id],
                        )
                    ]
        if (
            self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            is not None
        ):
            data = self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            for t1, t2, d_res in data:
                if res_id not in d_res:
                    continue
                tag = []
                if t1.is_a_task:
                    i1 = self.problem.task_id_to_index[t1.task_id]
                    itv1 = self.variables_dict["main_interval"][i1]
                    tag.append(("task", t1.task_id))
                else:
                    group = t1.group_id
                    itv1 = self.variables_dict["group_interval_per_id"][group]
                    tag.append(("group", group))
                if t2.is_a_task:
                    i2 = self.problem.task_id_to_index[t2.task_id]
                    itv2 = self.variables_dict["main_interval"][i2]
                    tag.append(("task", t2.task_id))
                else:
                    group = t2.group_id
                    itv2 = self.variables_dict["group_interval_per_id"][group]
                    tag.append(("group", group))

                tag = tuple(tag)
                list_time_level += [(itv1, "end", d_res[res_id])]
                list_time_level += [(itv2, "start", -d_res[res_id])]

        groups_non_release_resource = [
            g
            for g in self.problem.tasks_group
            if g.type_of_group == GroupType.GROUP_TASK_NON_RELEASED_RESOURCE
            and g.res_not_released.get(res_id, 0) > 0
        ]
        tasks_covered_in_group = set()
        if True:
            for g in groups_non_release_resource:
                tasks_covered_in_group.update(g.tasks_group)
                list_time_level += [
                    (
                        self.variables_dict["group_interval_per_id"][g.id],
                        "start",
                        g.res_not_released[res_id],
                    ),
                    (
                        self.variables_dict["group_interval_per_id"][g.id],
                        "end",
                        -g.res_not_released[res_id],
                    ),
                ]
        task_mode_consume = [
            (
                index_task,
                mode,
                self.problem.tasks[index_task]
                .modes[mode]
                .get_res_consumption(resource.id),
            )
            for index_task in range(self.problem.nb_tasks)
            for mode in self.problem.tasks[index_task].modes
            if self.problem.tasks[index_task]
            .modes[mode]
            .get_res_consumption(resource.id)
            > 0
            and self.problem.index_to_task_id[index_task] not in tasks_covered_in_group
        ]
        for i, m, cons in task_mode_consume:
            list_time_level.append(
                (self.variables_dict["opt_interval"][i][m], "start", cons)
            )
            list_time_level.append(
                (self.variables_dict["opt_interval"][i][m], "end", -cons)
            )

        max_capa = int(resource.max_capacity)
        steps = []
        for var, tag, level in list_time_level:
            if tag == "start":
                steps.append(self.cp_model.step_at_start(var, level))
            if tag == "end":
                steps.append(self.cp_model.step_at_end(var, level))
        self.cp_model.enforce(self.cp_model.sum(steps) <= max_capa)

    def constraint_precedence(self):
        """
        Basic precedence constraint
        """
        for t_id in self.problem.constraints.successors:
            index = self.problem.task_id_to_index[t_id]
            for succ in self.problem.constraints.successors[t_id]:
                succ_index = self.problem.task_id_to_index[succ]
                self.cp_model.end_before_start(
                    self.variables_dict["main_interval"][index],
                    self.variables_dict["main_interval"][succ_index],
                )

    def constraint_precedence_on_groups(self):
        """
        Basic precedence constraint between group of tasks
        """
        if self.problem.constraints.successors_group_tasks is None:
            return
        for g_id in self.problem.constraints.successors_group_tasks:
            for g_succ_id in self.problem.constraints.successors_group_tasks[g_id]:
                self.cp_model.end_before_start(
                    self.variables_dict["group_interval_per_id"][g_id],
                    self.variables_dict["group_interval_per_id"][g_succ_id],
                )

    def constraint_cumulative(self, params: ConstraintIncludingParams):
        for r in self.problem.resources:
            if r.renewable:
                if params.include_variable_resource:
                    if r.id in self.variables_dict["resource_capacity_variables"]:
                        self.constraint_cumulative_resource(
                            resource=r,
                            variable_max_capacity=True,
                        )
                self.constraint_cumulative_resource(
                    resource=r,
                    variable_max_capacity=False,
                )
            else:
                if params.include_variable_resource:
                    if r.id in self.variables_dict["resource_capacity_variables"]:
                        self.constraint_non_renewable_resource(
                            resource=r, variable_max_capacity=True
                        )
                self.constraint_non_renewable_resource(
                    resource=r, variable_max_capacity=False
                )

    def init_resource_variables(self):
        resource_capacity_var = {}
        for resource in self.problem.resources:
            object_resource: ObjectiveParamResource = (
                self.problem.objective_params.params_obj[ObjectivesEnum.RESOURCE_COST]
            )
            if object_resource.weight_per_resource_unit.get(resource.id, 0) > 0:
                resource_capacity_var[resource.id] = self.cp_model.int_var(
                    min=0,
                    max=int(resource.max_capacity),
                    name=f"res_capacity_{resource.id}",
                )
        self.variables_dict["resource_capacity_variables"] = resource_capacity_var

    def constraint_non_renewable_resource(
        self, resource: ResourceData, variable_max_capacity: bool = False
    ):
        capa = resource.max_capacity
        id_resource = resource.id
        task_mode_consume = [
            self.cp_model.step_at_start(
                self.variables_dict["opt_interval"][i][mode],
                -int(
                    self.problem.tasks[i].modes[mode].get_res_consumption(id_resource)
                ),
            )
            for i in self.variables_dict["opt_interval"]
            for mode in self.variables_dict["opt_interval"][i]
            if self.problem.tasks[i].modes[mode].get_res_consumption(id_resource) > 0
        ]
        if not variable_max_capacity:
            task_mode_consume.append(self.cp_model.step_at(cp.IntervalMin, capa))
            self.cp_model.enforce(self.cp_model.sum(task_mode_consume) >= 0)
        else:
            task_mode_consume.append(
                self.cp_model.step_at(
                    cp.IntervalMin,
                    self.variables_dict["resource_capacity_variables"][resource.id],
                )
            )
            self.cp_model.enforce(self.cp_model.sum(task_mode_consume) >= 0)

    def constraint_cumulative_resource(
        self,
        resource: ResourceData,
        variable_max_capacity: bool = False,
    ):
        post_cumulative_constraints(
            problem=self.problem,
            resource=resource,
            solver=self,
            variable_max_capacity=variable_max_capacity,
            include_intervals_non_release=True,
        )

    def constraint_non_overlap_group(self):
        for group in self.problem.tasks_group:
            if group.no_overlap:
                intervals = [
                    self.variables_dict["main_interval"][
                        self.problem.task_id_to_index[i_t]
                    ]
                    for i_t in group.tasks_group
                ]
                self.cp_model.no_overlap(intervals)

    def create_objectives(self, params: ConstraintIncludingParams):
        objs = []
        weights = []
        names = []
        for obj_enum in self.problem.objective_params.params_obj:
            if obj_enum == ObjectivesEnum.MAKESPAN:
                var, name = self.create_makespan()
                objs.append(var)
                weights.append(self.problem.objective_params.params_obj[obj_enum])
                names.append(name)
            if obj_enum == ObjectivesEnum.RESOURCE_COST:
                if params.include_variable_resource:
                    var, name = self.create_resource_objective(
                        obj_params_resource=self.problem.objective_params.params_obj[
                            obj_enum
                        ]
                    )
                    objs.append(var)
                    weights.append(
                        self.problem.objective_params.params_obj[obj_enum].weight
                    )
                    names.append(name)
            if obj_enum == ObjectivesEnum.WORK_IN_PROGRESS:
                # WIP objective measures the number of concurrent groups in progress
                obj_params_wip = self.problem.objective_params.params_obj[obj_enum]
                if obj_params_wip.count_nb_group_in_progress:
                    # TODO: Implement concurrent groups metric for OptalCP
                    # For now, add a placeholder
                    logger.warning(
                        "WIP concurrent groups metric not yet implemented in OptalCP solver"
                    )
                    objs.append(0)
                    weights.append(obj_params_wip.coefficient_on_nb_group_in_progress)
                    names.append("wip_cost")

            if obj_enum == ObjectivesEnum.TARDINESS:
                var, name = self.create_tardiness_objective(
                    obj_tardiness=self.problem.objective_params.params_obj[obj_enum]
                )
                objs.append(var)
                weights.append(1)
                names.append(name)
            if obj_enum == ObjectivesEnum.EARLINESS:
                var, name = self.create_earliness_objective(
                    obj_earliness=self.problem.objective_params.params_obj[obj_enum]
                )
                objs.append(var)
                weights.append(1)
                names.append(name)
        self.variables_dict["obj_data"] = (objs, weights, names)
        self.variables_dict["obj_var"] = [
            self.cp_model.int_var(
                min=cp.IntVarMin, max=cp.IntVarMax, name=f"obj_{names[i]}"
            )
            for i in range(len(names))
        ]
        for i in range(len(names)):
            self.cp_model.enforce(self.variables_dict["obj_var"][i] == objs[i])
        # Convert weights to integers for OptalCP compatibility
        int_weights = [int(w) if isinstance(w, (int, float)) else w for w in weights]
        self.cp_model.minimize(
            sum([objs[i] * int_weights[i] for i in range(len(objs))])
        )

    def create_resource_release_cost(self):
        exprs = []
        if "intervals_non_release" in self.variables_dict:
            for r in self.variables_dict["intervals_non_release"]:
                for itv, val in self.variables_dict["intervals_non_release"][r]:
                    exprs.append(self.cp_model.length(itv))
            sum_ = self.cp_model.sum(exprs)
            self.cp_model.minimize(sum_)
            # for expr in exprs:
            #    self.cp_model.enforce(expr <= 10)
            return sum_

    def implements_lexico_api(self) -> bool:
        return True

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        ind_obj = next(
            (
                i
                for i in range(len(self.variables_dict["obj_data"][2]))
                if self.variables_dict["obj_data"][2][i] == obj
            ),
            None,
        )
        if ind_obj is not None:
            self.cp_model.enforce(self.variables_dict["obj_data"][0][ind_obj] <= value)
        else:
            logger.warning(f"{obj} objective is absent it seems")

    def get_objr_expr(self, obj: Union[str, tuple]):
        if isinstance(obj, tuple):
            nb_objective = len(obj) // 2
            objs = [obj[2 * i] for i in range(nb_objective)]
            weights = [obj[2 * i + 1] for i in range(nb_objective)]
            objs_expr = [self.get_objr_expr(ob) for ob in objs]
            # Convert weights to integers for OptalCP
            int_weights = [
                int(w) if isinstance(w, (int, float)) else w for w in weights
            ]
            return self.cp_model.sum(
                [int(w) * o for w, o in zip(int_weights, objs_expr)]
            )
        ind_obj = next(
            (
                i
                for i in range(len(self.variables_dict["obj_data"][2]))
                if self.variables_dict["obj_data"][2][i] == obj
            ),
            None,
        )
        return self.variables_dict["obj_data"][0][ind_obj]

    def set_lexico_objective(self, obj: str) -> None:
        expr = self.get_objr_expr(obj)
        if expr is not None:
            self.cp_model.minimize(expr)
        else:
            logger.warning(f"{obj} objective is absent it seems")
        self.current_objective = obj

    def get_lexico_objectives_available(self) -> list[str]:
        return self.variables_dict["obj_data"][2]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        sol = res[-1][0]
        return sol._intern_obj[obj]

    def create_makespan(self):
        self.variables_dict["makespan"] = self.cp_model.max(
            [
                self.cp_model.end(self.variables_dict["main_interval"][i])
                for i in self.variables_dict["main_interval"]
            ]
        )
        return self.variables_dict["makespan"], "makespan"

    def create_resource_objective(self, obj_params_resource: ObjectiveParamResource):
        resource_cost = [
            self.variables_dict["resource_capacity_variables"][r]
            * int(obj_params_resource.weight_per_resource_unit[r])
            for r in obj_params_resource.weight_per_resource_unit
            if obj_params_resource.weight_per_resource_unit[r] != 0
            and (
                (r not in obj_params_resource.consider_in_objectives)
                or (obj_params_resource.consider_in_objectives[r])
            )
        ]
        return self.cp_model.sum(resource_cost) if resource_cost else 0, "resource_cost"

    def create_earliness_objective(self, obj_earliness: ObjectiveParamEarliness):
        self.variables_dict["earliness"] = {"tasks": {}, "groups": {}}
        cost_list: list[tuple[cp.IntExpr, float]] = []
        for id_task in obj_earliness.weight_per_task:
            if obj_earliness.weight_per_task[id_task] > 0:
                index = self.problem.task_id_to_index[id_task]
                deadline = int(self.problem.task_id_dict[id_task].max_ending_date)
                if deadline is not None:
                    end = self.cp_model.end(self.variables_dict["main_interval"][index])
                    earliness = self.cp_model.max2(0, deadline - end)
                    # cost_expr = penalty * lateness + earliness
                    cost_expr = earliness
                    cost_list.append(
                        (cost_expr, obj_earliness.weight_per_task[id_task])
                    )
                    self.variables_dict["earliness"]["tasks"][id_task] = {
                        "earliness": earliness,
                    }
        for id_group in obj_earliness.weight_per_groups:
            if obj_earliness.weight_per_groups[id_group] > 0:
                index = self.problem.group_id_to_index[id_group]
                deadline = int(self.problem.tasks_group[index].max_ending_date)
                soft = self.problem.tasks_group[index].soft_max_end_date
                if deadline is not None:
                    end = self.cp_model.end(
                        self.variables_dict["group_interval_per_id"][id_group]
                    )
                    earliness = self.cp_model.max2(0, deadline - end)
                    # cost_expr = penalty * lateness + earliness
                    cost_expr = earliness

                    cost_list.append(
                        (cost_expr, obj_earliness.weight_per_groups[id_group])
                    )
                    self.variables_dict["earliness"]["groups"][id_group] = {
                        "earliness": earliness,
                    }
        return (
            self.cp_model.sum([x[0] * int(x[1]) for x in cost_list]),
            "earliness",
        )

    def create_tardiness_objective(self, obj_tardiness: ObjectiveParamTardiness):
        self.variables_dict["tardiness"] = {"tasks": {}, "groups": {}}
        cost_list: list[tuple[cp.IntExpr, float]] = []
        for id_task in obj_tardiness.weight_per_task:
            if obj_tardiness.weight_per_task[id_task] > 0:
                index = self.problem.task_id_to_index[id_task]
                deadline = int(self.problem.task_id_dict[id_task].max_ending_date)
                if deadline is not None and obj_tardiness.weight_per_task[id_task] != 0:
                    end = self.cp_model.end(self.variables_dict["main_interval"][index])
                    lateness = self.cp_model.max2(0, end - deadline)
                    cost_list.append((lateness, obj_tardiness.weight_per_task[id_task]))
                    self.variables_dict["tardiness"]["tasks"][id_task] = {
                        "tardiness": lateness,
                    }
        for id_group in obj_tardiness.weight_per_groups:
            if obj_tardiness.weight_per_groups[id_group] > 0:
                index = self.problem.group_id_to_index[id_group]
                deadline = int(self.problem.tasks_group[index].max_ending_date)
                soft = self.problem.tasks_group[index].soft_max_end_date
                if (
                    deadline is not None
                    and obj_tardiness.weight_per_groups[id_group] != 0
                ):
                    end = self.cp_model.end(
                        self.variables_dict["group_interval_per_id"][id_group]
                    )
                    lateness = self.cp_model.max2(0, end - deadline)

                    cost_list.append(
                        (lateness, obj_tardiness.weight_per_groups[id_group])
                    )

                    self.variables_dict["tardiness"]["groups"][id_group] = {
                        "tardiness": lateness,
                    }
        return self.cp_model.sum([x[0] * int(x[1]) for x in cost_list]), "tardiness"

    def constraint_group_non_release_resource(self):
        groups_non_release_resource = [
            g
            for g in self.problem.tasks_group
            if g.type_of_group == GroupType.GROUP_TASK_NON_RELEASED_RESOURCE
        ]
        all_resource_concerned = set()
        for g in groups_non_release_resource:
            all_resource_concerned.update(set([r for r in g.res_not_released]))
        for resource in all_resource_concerned:
            intervals_and_consumption = []
            tasks_covered_in_group = set()
            for g in groups_non_release_resource:
                if "group_interval_per_id" in self.variables_dict:
                    if (
                        resource in g.res_not_released
                        and g.res_not_released[resource] > 0
                    ):
                        intervals_and_consumption.append(
                            (
                                self.variables_dict["group_interval_per_id"][g.id],
                                g.res_not_released[resource],
                            )
                        )
                        tasks_covered_in_group.update(g.tasks_group)
            task_mode_consume = [
                (
                    self.variables_dict["opt_interval"][i][mode],
                    int(
                        self.problem.tasks[i].modes[mode].get_res_consumption(resource)
                    ),
                )
                for i in self.variables_dict["opt_interval"]
                for mode in self.variables_dict["opt_interval"][i]
                if self.problem.tasks[i].modes[mode].get_res_consumption(resource) > 0
                and self.problem.index_to_task_id[i] not in tasks_covered_in_group
            ]
            if "intervals_non_release" in self.variables_dict:
                task_non_release = self.variables_dict["intervals_non_release"].get(
                    resource, []
                )
            else:
                task_non_release = []
            if self.problem.resource_dict[resource].max_capacity == 1:
                self.cp_model.no_overlap(
                    [
                        x[0]
                        for x in intervals_and_consumption
                        + task_mode_consume
                        + task_non_release
                    ]
                )
            else:
                pulses = [
                    self.cp_model.pulse(x[0], x[1])
                    for x in intervals_and_consumption
                    + task_mode_consume
                    + task_non_release
                ]

                self.cp_model.enforce(
                    self.cp_model.sum(pulses)
                    <= int(self.problem.resource_dict[resource].max_capacity)
                )


def post_cumulative_constraints(
    problem: FlexProblem,
    resource: ResourceData,
    solver: OptalFlexProblemSolver,
    variable_max_capacity: bool,
    include_intervals_non_release: bool = True,
):
    inputs_constraint = build_multiple_cumulative_constraints_inputs(
        problem=problem, resource=resource
    )
    task_non_release = []
    if (
        "intervals_non_release" in solver.variables_dict
        and include_intervals_non_release
        and not variable_max_capacity
    ):
        if resource.id in solver.variables_dict["intervals_non_release"]:
            task_non_release = solver.variables_dict["intervals_non_release"][
                resource.id
            ]
    for input_data in inputs_constraint:
        val = input_data["val"]
        set_task_mode_conso = list(input_data["set_task_mode_conso"])
        intervals_ = [
            solver.variables_dict["opt_interval"][x[0]][x[1]]
            for x in set_task_mode_conso
        ]
        consos = [x[2] for x in set_task_mode_conso]
        other_intervals_c = [
            x for x in task_non_release if x[1] + val <= resource.max_capacity
        ]
        calendar_intervals = [
            (
                solver.cp_model.interval_var(
                    start=f["start"],
                    end=f["start"] + f["duration"],
                    length=f["duration"],
                    name=f"res_",
                ),
                f["value"],
            )
            for f in input_data["calendar_tasks"]
        ]
        if len(intervals_) + len(other_intervals_c) == 0:
            # Useless
            continue

        if not variable_max_capacity:
            max_cap = int(resource.max_capacity)
            if max_cap == 1:
                solver.cp_model.no_overlap(
                    intervals_
                    + [x[0] for x in other_intervals_c]
                    + [x[0] for x in calendar_intervals]
                )
            else:
                cumulative = [
                    solver.cp_model.pulse(itv, height)
                    for itv, height in zip(intervals_, consos)
                ]
                cumulative.extend(
                    [solver.cp_model.pulse(x[0], x[1]) for x in other_intervals_c]
                )
                cumulative.extend(
                    [solver.cp_model.pulse(x[0], x[1]) for x in calendar_intervals]
                )
                solver.cp_model.enforce(solver.cp_model.sum(cumulative) <= max_cap)
        else:
            # Use of reservoir when the capacity is variable ??
            if "resource_capacity_variables" in solver.variables_dict:
                if (
                    resource.id
                    not in solver.variables_dict["resource_capacity_variables"]
                ):
                    continue
                # solver.cp_model.enforce(solver.cp_model.sum(cumulative) <=
                #                        solver.variables_dict["resource_capacity_variables"][resource.id])
                if True:
                    if "artificial_var" not in solver.variables_dict:
                        solver.variables_dict["artificial_var"] = (
                            solver.cp_model.interval_var(
                                start=cp.IntervalMin, end=cp.IntervalMax
                            )
                        )
                    cumulative = [
                        solver.cp_model.pulse(
                            solver.variables_dict["artificial_var"],
                            int(resource.max_capacity)
                            - solver.variables_dict["resource_capacity_variables"][
                                resource.id
                            ],
                        )
                    ]
                    cumulative.extend(
                        [
                            solver.cp_model.pulse(itv, height)
                            for itv, height in zip(intervals_, consos)
                        ]
                    )
                    cumulative.extend(
                        [solver.cp_model.pulse(x[0], x[1]) for x in other_intervals_c]
                    )
                    cumulative.extend(
                        [solver.cp_model.pulse(x[0], x[1]) for x in calendar_intervals]
                    )
                    solver.cp_model.enforce(
                        solver.cp_model.sum(cumulative) <= int(resource.max_capacity)
                    )
