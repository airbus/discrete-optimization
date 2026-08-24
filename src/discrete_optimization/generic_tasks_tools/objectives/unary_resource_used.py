#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)


class UnaryResourcesUsedComputer(ObjectiveComputer[Task], Generic[Task, UnaryResource]):
    problem: AllocationProblem[Task, UnaryResource]

    def __init__(
        self,
        problem: AllocationProblem[Task, UnaryResource] = None,
        weight_objective: float = 1.0,
        weight_per_unary_resource: dict[UnaryResource, float] = None,
    ) -> None:
        super().__init__(problem, weight_objective)
        if weight_per_unary_resource is None:
            self.weight_per_unary_resource = {
                ur: 1 for ur in self.problem.unary_resources_list
            }
        else:
            self.weight_per_unary_resource = weight_per_unary_resource

    def get_weight_per_unary_resource(self, ur: UnaryResource) -> float:
        return self.weight_per_unary_resource.get(ur, 0)

    def has_cost_on_unary_resource(self) -> bool:
        return any(
            self.get_weight_per_unary_resource(ur) != 0
            for ur in self.problem.unary_resources_list
        )

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.NB_UNARY_RESOURCES_USED

    def compute_objective(self, solution: AllocationSolution[Task, UnaryResource]):
        if not self.has_cost_on_unary_resource():
            return 0
        return sum(
            any(
                solution.is_allocated(task=task, unary_resource=unary_resource)
                for task in self.problem.tasks_list
            )
            * self.get_weight_per_unary_resource(unary_resource)
            for unary_resource in self.problem.unary_resources_list
        )
