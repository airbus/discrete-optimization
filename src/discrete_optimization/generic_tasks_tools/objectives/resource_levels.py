#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.calendar_resource import (
    CalendarResourceProblem,
    CalendarResourceSolution,
    Resource,
    Task,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
    NonRenewableResourceProblem,
    NonRenewableResourceSolution,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)


class CalendarRenewableResourceLevelObjectiveComputer(
    ObjectiveComputer[Task], Generic[Task, Resource]
):
    problem: CalendarResourceProblem[Task, Resource]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.CALENDAR_RESOURCES_LEVELS

    def __init__(
        self,
        problem: CalendarResourceProblem[Task, Resource],
        weight_objective: float = 1.0,
        weight_resource: dict[Resource, float] = None,
    ):
        super().__init__(problem, weight_objective)
        if weight_resource is None:
            self.weight_resource = {}
        else:
            self.weight_resource = weight_resource

    def get_weight_resource(self, resource: Resource) -> float:
        return self.weight_resource.get(resource, 0)

    def has_any_weight(self):
        return any(self.get_weight_resource(res) > 0 for res in self.weight_resource)

    def compute_objective(
        self, solution: CalendarResourceSolution[Task, Resource]
    ) -> float:
        return solution.compute_aggregated_calendar_resources_levels(
            weights={
                r: self.get_weight_resource(r)
                for r in self.problem.calendar_resources_list
            }
        )


class NonRenewableResourceLevelObjectiveComputer(
    ObjectiveComputer[Task], Generic[Task, NonRenewableResource]
):
    problem: NonRenewableResourceProblem[Task, NonRenewableResource]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.NON_RENEWABLE_RESOURCES_LEVELS

    def __init__(
        self,
        problem: NonRenewableResourceProblem[Task, Resource],
        weight_objective: float = 1.0,
        weight_resource: dict[Resource, float] = None,
    ):
        super().__init__(problem, weight_objective)
        if weight_resource is None:
            self.weight_resource = {}
        else:
            self.weight_resource = weight_resource

    def get_weight_resource(self, resource: Resource) -> float:
        return self.weight_resource.get(resource, 0)

    def has_any_weight(self):
        return any(self.get_weight_resource(res) > 0 for res in self.weight_resource)

    def compute_objective(
        self, solution: NonRenewableResourceSolution[Task, NonRenewableResource]
    ) -> float:
        return solution.compute_nb_non_renewable_resources_used(
            weights={
                r: self.get_weight_resource(r)
                for r in self.problem.non_renewable_resources_list
            }
        )
