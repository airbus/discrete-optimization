#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Generic

from ortools.sat.python.cp_model import IntervalVar, LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    CalendarResourceProblem,
    Resource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)


class CalendarResourceCpSatSolver(SchedulingCpSatSolver[Task], Generic[Task, Resource]):
    problem: CalendarResourceProblem[Task, Resource]

    use_no_overlap_for_capa_1: bool = True
    """Flag to use rather no_overlap constraint when resource capacity is 1."""
    use_cumulative_for_capa_1: bool = False
    """Flag to use rather cumulative constraint when resource capacity is 1."""

    def create_calendar_resources_constraint(self, resource: Resource):
        """Add the constraint for renewable resources with an availability calendar to the cpsat model.

        Constraint ensuring that the total demand on the given resource stay below its capacity.

        """
        actual_tasks_intervals_n_consumptions = self.get_resource_consumption_intervals(
            resource
        )
        fake_tasks_intervals_n_consumptions = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=start, size=end - start, name=f"fake_task_{resource}_{i_task}"
                ),
                value,
            )
            for i_task, (start, end, value) in enumerate(
                self.problem.get_fake_tasks(resource=resource)
            )
        ]
        all_tasks_intervals_n_consumptions = (
            actual_tasks_intervals_n_consumptions + fake_tasks_intervals_n_consumptions
        )
        intervals = [
            interval
            for interval, value in all_tasks_intervals_n_consumptions
            if not isinstance(value, int) or value > 0
        ]
        demands = [
            value
            for interval, value in all_tasks_intervals_n_consumptions
            if not isinstance(value, int) or value > 0
        ]
        capacity = self.problem.get_resource_max_capacity(resource)
        if len(intervals) > 0:
            if capacity == 1 and all(
                isinstance(value, int) and value == 1 for value in demands
            ):
                if self.use_no_overlap_for_capa_1 or not self.use_cumulative_for_capa_1:
                    self.cp_model.add_no_overlap(intervals)
                if self.use_cumulative_for_capa_1:
                    self.cp_model.add_cumulative(
                        intervals=intervals,
                        demands=demands,
                        capacity=capacity,
                    )
            else:
                self.cp_model.add_cumulative(
                    intervals=intervals,
                    demands=demands,
                    capacity=capacity,
                )

    @abstractmethod
    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, LinearExprT]]:
        """Get all intervals where a given resource is consumed by a task, and related consumption value.

        Args:
            resource:

        Returns: list of tuples (interval_var, consumption_value)

        """
        ...
