#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Generic

from ortools.sat.python.cp_model import IntervalVar

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.renewable_resource import (
    RenewableResourceProblem,
    Resource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)


class RenewableResourceCpSatSolver(
    SchedulingCpSatSolver[Task], Generic[Task, Resource]
):
    problem: RenewableResourceProblem[Task, Resource]

    def create_renewable_resources_constraint(self, resource: Resource):
        """Add the constraint for renewable resources to the cpsat model.

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
            if value > 0
        ]
        demands = [
            value for interval, value in all_tasks_intervals_n_consumptions if value > 0
        ]
        capacity = self.problem.get_resource_max_capacity(resource)
        if len(intervals) > 0:
            if capacity == 1 and all(value == 1 for value in demands):
                self.cp_model.add_no_overlap(intervals)
            else:
                self.cp_model.add_cumulative(
                    intervals=intervals,
                    demands=demands,
                    capacity=capacity,
                )

    @abstractmethod
    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, int]]:
        """Get all intervals where a given resource is consumed by a task, and related consumption value.

        Args:
            resource:

        Returns: list of tuples (interval_var, consumption_value)

        """
        ...
