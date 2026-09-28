#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  This is where the real cumulative constraint is happening
#  It is there to handle the resource blocking,
#  the calendar preemptive,
#  the classic ones.
from typing import Generic

from ortools.sat.python.cp_model import IntervalVar, LinearExprT

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_preemptive import (
    CalendarPreemptiveProblem,
)
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    Resource,
)
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    ResourceBlockingProblem,
)
from discrete_optimization.generic_tasks_tools.skill import NonSkillCumulativeResource
from discrete_optimization.generic_tasks_tools.solvers.cpsat.resource_blocking import (
    ResourceBlockingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)


class ProblemWithCalendarPreemptiveAndResourceBlocking(
    ResourceBlockingProblem[Task, NonSkillCumulativeResource, UnaryResource],
    CalendarPreemptiveProblem[Task, NonSkillCumulativeResource, UnaryResource],
    Generic[Task, NonSkillCumulativeResource, UnaryResource],
):
    pass


class CalendarResourceGenericCpSatSolver(
    ResourceBlockingCpSatSolver[Task, NonSkillCumulativeResource, UnaryResource],
    SchedulingCpSatSolver[Task],
    Generic[Task, NonSkillCumulativeResource, UnaryResource],
):
    problem: ProblemWithCalendarPreemptiveAndResourceBlocking[
        Task, NonSkillCumulativeResource, UnaryResource
    ]
    use_no_overlap_for_capa_1: bool = True
    """Flag to use rather no_overlap constraint when resource capacity is 1."""
    use_cumulative_for_capa_1: bool = False
    """Flag to use rather cumulative constraint when resource capacity is 1."""

    def get_resource_interval_and_consumption_for_task(
        self,
        resource: Resource,
        task: Task,
    ) -> tuple[IntervalVar, LinearExprT]:
        return (
            self.get_task_interval(task=task),
            self.get_cumulative_resource_demand_variable(task=task, resource=resource),
        )

    def get_resource_interval_and_consumption_for_task_and_mode(
        self, resource: Resource, task: Task, mode: int
    ) -> tuple[IntervalVar, LinearExprT]:
        if self.problem.is_cumulative_resource_task_mode_consumption_dependent(
            task, mode
        ):
            return (
                self.get_task_mode_interval(task=task, mode=mode),
                self.get_cumulative_resource_demand_variable(
                    task=task, resource=resource
                ),
            )
        return (
            self.get_task_mode_interval(task=task, mode=mode),
            self.problem.get_cumulative_resource_consumption(
                resource=resource, task=task, mode=mode
            ),
        )

    def create_calendar_resources_constraint(self, resource: Resource):
        if not self.problem.has_any_calendar_preempted():
            super().create_cumulative_constraint_including_blocking(resource)
            return
        decomposition = (
            self.problem.compute_calendar_break_and_task_for_cumulative_decomposition(
                resource
            )
        )
        capacity = self.problem.get_resource_max_capacity(resource)
        reservation_blocking, active_blocking = self.get_blocking_intervals_and_demands(
            resource
        )
        for decomp in decomposition:
            subset_tasks = set([x[0] for x in decomp["set_task_mode_conso"]])
            task_mode_of_interest = set(
                [(x[0], x[1]) for x in decomp["set_task_mode_conso"]]
            )
            task_to_include_as_one = set()
            task_mode_to_include = set()
            for t in subset_tasks:
                modes = self.problem.get_task_modes(t)
                if all((t, m) in task_mode_of_interest for m in modes):
                    task_to_include_as_one.add(t)
                else:
                    for m in modes:
                        if (t, m) in task_mode_of_interest:
                            task_mode_to_include.add((t, m))
            itvs = [
                self.get_resource_interval_and_consumption_for_task(
                    resource=resource, task=t
                )
                for t in task_to_include_as_one
            ]
            itvs.extend(
                [
                    self.get_resource_interval_and_consumption_for_task_and_mode(
                        resource=resource, task=x[0], mode=x[1]
                    )
                    for x in task_mode_to_include
                ]
            )
            calendar_intervals = [
                (
                    self.cp_model.new_fixed_size_interval_var(
                        start=f[0], size=f[1] - f[0], name=f"res_"
                    ),
                    f[2],
                )
                for f in decomp["calendar_tasks"]
            ]
            current_reservation = [
                x
                for x in reservation_blocking
                if x[1] + decomposition["val"] <= capacity
            ]
            all_intervals = []
            all_intervals.extend(itvs)
            all_intervals.extend(active_blocking)
            all_intervals.extend(current_reservation)
            all_intervals.extend(calendar_intervals)
            if itvs or active_blocking or current_reservation:
                if capacity == 1 and all(
                    isinstance(v[1], int) and v[1] == 1 for v in all_intervals
                ):
                    if (
                        self.use_no_overlap_for_capa_1
                        or not self.use_cumulative_for_capa_1
                    ):
                        self.cp_model.add_no_overlap([x[0] for x in all_intervals])
                    if self.use_cumulative_for_capa_1:
                        self.cp_model.add_cumulative(
                            intervals=[x[0] for x in all_intervals],
                            demands=[x[1] for x in all_intervals],
                            capacity=capacity,
                        )
                else:
                    self.cp_model.add_cumulative(
                        [x[0] for x in all_intervals],
                        [x[1] for x in all_intervals],
                        capacity,
                    )

        if self.problem.has_any_blocking():
            actual_tasks_intervals_n_consumptions = (
                self.get_resource_consumption_intervals(resource)
            )
            # MORE
            # CLASSICAL
            # CONSTRAINT
            intervals_no_calendar = []
            intervals_no_calendar.extend(actual_tasks_intervals_n_consumptions)
            intervals_no_calendar.extend(reservation_blocking)
            intervals_no_calendar.extend(active_blocking)
            intervals_1 = [
                interval
                for interval, demand in intervals_no_calendar
                if not isinstance(demand, int) or demand > 0
            ]
            demands_1 = [
                demand
                for interval, demand in intervals_no_calendar
                if not isinstance(demand, int) or demand > 0
            ]
            if len(intervals_1) > 0:
                if capacity == 1 and all(
                    isinstance(v, int) and v == 1 for v in demands_1
                ):
                    if (
                        self.use_no_overlap_for_capa_1
                        or not self.use_cumulative_for_capa_1
                    ):
                        self.cp_model.add_no_overlap(intervals_1)
                    if self.use_cumulative_for_capa_1:
                        self.cp_model.add_cumulative(
                            intervals=intervals_1, demands=demands_1, capacity=capacity
                        )
                else:
                    self.cp_model.add_cumulative(
                        intervals=intervals_1, demands=demands_1, capacity=capacity
                    )
            # Get fake tasks for calendar gaps
            fake_tasks_intervals = [
                (
                    self.cp_model.NewFixedSizeIntervalVar(
                        start=start,
                        size=end - start,
                        name=f"fake_task_{resource}_{i_task}",
                    ),
                    value,
                )
                for i_task, (start, end, value) in enumerate(
                    self.problem.get_fake_tasks(resource=resource)
                )
            ]
            if active_blocking or fake_tasks_intervals:
                intervals_with_calendar = []
                intervals_with_calendar.extend(
                    [
                        self.get_resource_interval_and_consumption_for_task(
                            resource, task
                        )
                        for task in self.problem.tasks_list
                        if not self.problem.is_task_calendar_preempted(task)
                    ]
                )
                intervals_with_calendar.extend(active_blocking)
                intervals_with_calendar.extend(fake_tasks_intervals)
                intervals_2 = [
                    interval
                    for interval, demand in intervals_with_calendar
                    if not isinstance(demand, int) or demand > 0
                ]
                demands_2 = [
                    demand
                    for interval, demand in intervals_with_calendar
                    if not isinstance(demand, int) or demand > 0
                ]

                if len(intervals_2) > 0:
                    if capacity == 1 and all(
                        isinstance(v, int) and v == 1 for v in demands_2
                    ):
                        if (
                            self.use_no_overlap_for_capa_1
                            or not self.use_cumulative_for_capa_1
                        ):
                            self.cp_model.add_no_overlap(intervals_2)
                        if self.use_cumulative_for_capa_1:
                            self.cp_model.add_cumulative(
                                intervals=intervals_2,
                                demands=demands_2,
                                capacity=capacity,
                            )
                    else:
                        self.cp_model.add_cumulative(
                            intervals=intervals_2, demands=demands_2, capacity=capacity
                        )
