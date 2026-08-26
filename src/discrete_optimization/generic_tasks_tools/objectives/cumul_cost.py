#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic, Hashable, TypeVar

from discrete_optimization.generic_tasks_tools.allocation import Task, UnaryResource
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
    NonRenewableResource,
    NonSkillCumulativeResource,
    Skill,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)

CUMUL_DIMENSIONS = TypeVar("CUMUL_DIMENSIONS", bound=Hashable)


class CumulCostComputer(
    ObjectiveComputer[Task],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
):
    problem: GenericSchedulingProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.CUMUL_COST

    def __init__(
        self,
        problem: GenericSchedulingProblem[
            Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
        ] = None,
        weight_objective: float = 1.0,
        cumul_dimensions: list[CUMUL_DIMENSIONS] = None,
        value_tasks: dict[CUMUL_DIMENSIONS, dict[Task, float]] = None,
        value_tasks_per_mode: dict[
            CUMUL_DIMENSIONS, dict[tuple[Task, int], float]
        ] = None,
    ):
        super().__init__(problem, weight_objective)
        if cumul_dimensions is not None:
            self.cumul_dimensions = cumul_dimensions
        else:
            self.cumul_dimensions = ["count"]
        if value_tasks is not None:
            self.value_tasks = value_tasks
        else:
            self.value_tasks = {"count": {t: 1 for t in self.problem.tasks_list}}
        if value_tasks_per_mode is not None:
            self.value_tasks_per_mode = value_tasks_per_mode
        else:
            self.value_tasks_per_mode = {}

    def depends_on_mode(self, dimension: CUMUL_DIMENSIONS, task: Task):
        if dimension in self.value_tasks:
            if task in self.value_tasks[dimension]:
                return False
        if dimension in self.value_tasks_per_mode:
            values = [
                self.get_value_dimension_task_mode(
                    dimension=dimension, task=task, mode=mode
                )
                for mode in self.problem.get_task_modes(task)
            ]
            if len(set(values)) > 1:
                return True
            return False
        return False

    def get_value_dimension_task_mode(
        self, dimension: CUMUL_DIMENSIONS, task: Task, mode: int
    ):
        if dimension in self.value_tasks:
            if task in self.value_tasks[dimension]:
                return self.value_tasks[dimension][task]
        if dimension in self.value_tasks_per_mode:
            if (task, mode) not in self.value_tasks_per_mode[dimension]:
                return self.value_tasks_per_mode[dimension][(task, mode)]
        return 0

    def compute_objective(
        self,
        solution: GenericSchedulingSolution[
            Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
        ],
    ) -> float:
        cumul_per_unary_resource = {
            ur: {dim: 0 for dim in self.cumul_dimensions}
            for ur in self.problem.unary_resources_list
        }
        for task in self.problem.tasks_list:
            allocated = solution.get_task_allocation(task)
            if len(allocated) == 0:
                continue
            mode = solution.get_mode(task)
            for dim in self.cumul_dimensions:
                value = self.get_value_dimension_task_mode(
                    dimension=dim, task=task, mode=mode
                )
                for ur in allocated:
                    cumul_per_unary_resource[ur][dim] += value
        cost = 0
        for dim in self.cumul_dimensions:
            nz = [
                val
                for ur in cumul_per_unary_resource
                if (val := cumul_per_unary_resource[ur][dim]) > 0
            ]
            if len(nz) == 0:
                continue
            cost += max(nz) - min(nz)
        return cost
