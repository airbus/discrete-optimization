#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import TasksProblem
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    CalendarResourceProblem,
    CalendarResourceSolution,
    Resource,
    Task,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
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

logger = logging.getLogger(__name__)


class CalendarRenewableResourceLevelObjectiveComputer(
    ObjectiveComputer[Task], Generic[Task, Resource]
):
    """Objective computer aggregating calendar resource levels.

    It aggregates the max resource consumption over time. The resources are renewable,
    consumed by tasks and then released.

    Args:
        problem: problem on which it will be applied.
            can be None and specified later via `self.set_problem()`
        weight_objective: weight of this computer among list of all computers
        weight_resource: weight associated to each calendar resource.
            Default to 1 for non-skill cumulative resources and 0 for other (unary resources and skills)


    Notes:
        - For generic scheduling problem, calendar resources include
            - unary resources
            - skill resources
            - cumulative resources other than skills
          Usually, one want to use only non-skill cumulative resources.
          That is why the default value for the weights is 1 for non-skill cumulative resource and 0 else.
        - For some solver (e.g. optal), the weights need to be integers


    """

    problem: CalendarResourceProblem[Task, Resource]

    @staticmethod
    def get_objective_name() -> str:
        return Objective.CALENDAR_RESOURCES_LEVELS

    def __init__(
        self,
        problem: CalendarResourceProblem[Task, Resource] | None = None,
        weight_objective: float | int = 1,
        weight_resource: dict[Resource, float | int] | None = None,
    ):
        self.weight_resource = weight_resource
        super().__init__(problem, weight_objective)

    def set_problem(self, problem: TasksProblem[Task]):
        super().set_problem(problem)
        if self.weight_resource is None:
            if isinstance(self.problem, GenericSchedulingProblem):
                self.weight_resource = {
                    resource: 1
                    for resource in self.problem.non_skill_cumulative_resources_list
                }
            else:
                self.weight_resource = {}
                logger.warning(
                    "All resource weights are 0: the aggregated resource levels objective will be 0."
                )

    def get_weight_resource(self, resource: Resource) -> float | int:
        return self.weight_resource.get(resource, 0)

    def has_any_weight(self):
        return any(self.get_weight_resource(res) > 0 for res in self.weight_resource)

    def compute_objective(
        self, solution: CalendarResourceSolution[Task, Resource]
    ) -> float | int:
        return solution.compute_aggregated_calendar_resources_levels(
            weights={
                r: self.get_weight_resource(r)
                for r in self.problem.calendar_resources_list
            }
        )


class NonRenewableResourceLevelObjectiveComputer(
    ObjectiveComputer[Task], Generic[Task, NonRenewableResource, UnaryResource]
):
    """Objective computer aggregating non-renewable resource levels.

    It aggregates the resource consumptions.

    Args:
        problem: problem on which it will be applied.
            can be None and specified later via `self.set_problem()`
        weight_objective: weight of this computer among list of all computers
        weight_resource: weight associated to each resource.
            Default to 1 for each non-renewable resources


    Note:
        For some solver (e.g. optal), the weights need to be integers


    """

    problem: NonRenewableResourceProblem[Task, NonRenewableResource, UnaryResource]

    @staticmethod
    def get_objective_name() -> str:
        return Objective.NON_RENEWABLE_RESOURCES_LEVELS

    def __init__(
        self,
        problem: NonRenewableResourceProblem[Task, Resource, UnaryResource]
        | None = None,
        weight_objective: float | int = 1,
        weight_resource: dict[Resource, float | int] | None = None,
    ):
        self.weight_resource = weight_resource
        super().__init__(problem, weight_objective)

    def set_problem(self, problem: TasksProblem[Task]):
        super().set_problem(problem)
        if self.weight_resource is None:
            self.weight_resource = {
                resource: 1 for resource in self.problem.non_renewable_resources_list
            }

    def get_weight_resource(self, resource: Resource) -> float | int:
        return self.weight_resource.get(resource, 0)

    def has_any_weight(self):
        return any(self.get_weight_resource(res) > 0 for res in self.weight_resource)

    def compute_objective(
        self,
        solution: NonRenewableResourceSolution[
            Task, NonRenewableResource, UnaryResource
        ],
    ) -> float | int:
        return solution.compute_nb_non_renewable_resources_used(
            weights={
                r: self.get_weight_resource(r)
                for r in self.problem.non_renewable_resources_list
            }
        )
