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
from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeProblem,
    MultimodeSolution,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)


class AllocationCostComputer(ObjectiveComputer[Task], Generic[Task, UnaryResource]):
    problem: AllocationProblem[Task, UnaryResource]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.ALLOCATION_COST

    def __init__(
        self,
        problem: AllocationProblem[Task, UnaryResource] = None,
        weight_objective: float = 1.0,
        cost_allocation_resource_to_task: dict[Task, dict[UnaryResource, int]] = None,
    ):
        super().__init__(problem, weight_objective)
        if cost_allocation_resource_to_task is None:
            self._cost_allocation_resource_to_task = {}
        else:
            self._cost_allocation_resource_to_task = cost_allocation_resource_to_task

    def cost_allocation_resource_to_task(
        self, task: Task, unary_resource: UnaryResource
    ) -> float:
        if task in self._cost_allocation_resource_to_task:
            return self._cost_allocation_resource_to_task[task].get(unary_resource, 0)
        return 0

    def has_any_cost_allocation(self):
        return any(
            self._cost_allocation_resource_to_task[tm][u] != 0
            for tm in self._cost_allocation_resource_to_task
            for u in self._cost_allocation_resource_to_task[tm]
        )

    def compute_objective(
        self, solution: AllocationSolution[Task, UnaryResource]
    ) -> float:
        return sum(
            self.cost_allocation_resource_to_task(
                task=task, unary_resource=unary_resource
            )
            for task in self.problem.tasks_list
            for unary_resource in solution.get_task_allocation(task=task)
        )


class MultimodeAllocationProblem(
    AllocationProblem[Task, UnaryResource], MultimodeProblem[Task]
):
    pass


class MultimodeAllocationSolution(
    AllocationSolution[Task, UnaryResource], MultimodeSolution[Task]
):
    pass


class AllocationCostComputerMultimode(
    ObjectiveComputer[Task], Generic[Task, UnaryResource]
):
    problem: MultimodeAllocationProblem[Task, UnaryResource]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.ALLOCATION_COST

    def __init__(
        self,
        problem: MultimodeAllocationProblem[Task, UnaryResource] = None,
        weight_objective: float = 1.0,
        cost_allocation_resource_to_task_mode: dict[
            tuple[Task, int], dict[UnaryResource, int]
        ] = None,
    ):
        super().__init__(problem, weight_objective)
        if cost_allocation_resource_to_task_mode is None:
            self._cost_allocation_resource_to_task_mode = {}
        else:
            self._cost_allocation_resource_to_task_mode = (
                cost_allocation_resource_to_task_mode
            )

    def cost_allocation_resource_to_task_mode(
        self, task: Task, mode: int, unary_resource: UnaryResource
    ) -> float:
        if (task, mode) in self._cost_allocation_resource_to_task_mode:
            return self._cost_allocation_resource_to_task_mode[task, mode].get(
                unary_resource, 0
            )
        return 0

    def get_tasks_having_cost(self):
        return set(
            t
            for t, m in self._cost_allocation_resource_to_task_mode
            if self._cost_allocation_resource_to_task_mode[t, m] != 0
        )

    def has_any_cost_allocation(self):
        return any(
            self._cost_allocation_resource_to_task_mode[tm][u] != 0
            for tm in self._cost_allocation_resource_to_task_mode
            for u in self._cost_allocation_resource_to_task_mode[tm]
        )

    def compute_objective(
        self, solution: MultimodeAllocationSolution[Task, UnaryResource]
    ) -> float:
        return sum(
            self.cost_allocation_resource_to_task_mode(
                task=task, mode=solution.get_mode(task), unary_resource=unary_resource
            )
            for task in self.problem.tasks_list
            for unary_resource in solution.get_task_allocation(task=task)
        )
