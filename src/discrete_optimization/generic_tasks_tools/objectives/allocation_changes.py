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


# TODO : investigate, if not better to take into account via :
#  self.problem.get_unary_resource_cost ... !
class AllocationSwitchObjectiveComputer(
    ObjectiveComputer[Task], Generic[Task, UnaryResource]
):
    problem: AllocationProblem[Task, UnaryResource]
    base_allocation_solution: AllocationSolution[Task, UnaryResource]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.ALLOCATION_CHANGES

    def __init__(
        self,
        problem: AllocationProblem[Task, UnaryResource],
        base_allocation_solution: AllocationSolution[Task, UnaryResource] = None,
        weight_objective: float = 1.0,
        switch_on_cost: dict[Task, dict[UnaryResource, int]] = None,
        switch_off_cost: dict[Task, dict[UnaryResource, int]] = None,
    ):
        super().__init__(problem, weight_objective)
        self.base_allocation_solution = base_allocation_solution
        if switch_on_cost is None:
            self.switch_on_cost = {}
        else:
            self.switch_on_cost = switch_on_cost
        if switch_off_cost is None:
            self.switch_off_cost = {}
        else:
            self.switch_off_cost = switch_off_cost

    def set_base_allocation_solution(
        self, allocation_solution: AllocationSolution[Task, UnaryResource]
    ):
        self.base_allocation_solution = allocation_solution

    def get_switch_on_cost(self, task: Task, unary_resource: UnaryResource) -> int:
        # cost of assigning unary_resource to task, while it wasn't before.
        if task in self.switch_on_cost:
            return self.switch_on_cost[task].get(unary_resource, 0)
        return 0

    def get_switch_off_cost(self, task: Task, unary_resource: UnaryResource) -> int:
        # cost of disassigning unary_resource to task, while it was before.
        if task in self.switch_off_cost:
            return self.switch_off_cost[task].get(unary_resource, 0)
        return 0

    def has_switch_on_cost(self):
        return any(
            self.get_switch_on_cost(task, ur) > 0
            for task in self.switch_on_cost
            for ur in self.problem.unary_resources_list
        )

    def has_switch_off_cost(self):
        return any(
            self.get_switch_off_cost(task, ur) > 0
            for task in self.switch_off_cost
            for ur in self.problem.unary_resources_list
        )

    def has_switches_cost(self):
        return self.has_switch_on_cost() or self.has_switch_off_cost()

    def compute_objective(
        self, solution: AllocationSolution[Task, UnaryResource]
    ) -> float:
        cost = 0
        for task in self.problem.tasks_list:
            alloc_original = self.base_allocation_solution.get_task_allocation(task)
            new_alloc = solution.get_task_allocation(task)
            for u0 in alloc_original:
                if u0 not in new_alloc:
                    cost += self.get_switch_off_cost(task, u0)
            for u1 in new_alloc:
                if u1 not in alloc_original:
                    cost += self.get_switch_on_cost(task, u1)
        return cost
