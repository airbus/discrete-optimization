#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from copy import deepcopy
from typing import Hashable, Iterable, Optional, Union

import numpy as np
import wrapt

from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
    WithoutAllocationProblem,
    WithoutAllocationSolution,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    Resource,
    convert_calendar_to_availability_intervals,
)
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    NoSkill,
    Skill,
    WithoutSkillProblem,
    WithoutSkillSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Solution,
    TypeObjective,
)


class RcpspResourceDependentSolution(
    GenericSchedulingSolution[
        Task, NoUnaryResource, NoSkill, NonSkillCumulativeResource, NonRenewableResource
    ],
    WithoutSkillSolution[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
    ],
    WithoutAllocationSolution[Task],
):
    def get_end_time(self, task: Task) -> int:
        return self.schedule[task][1]

    def get_start_time(self, task: Task) -> int:
        return self.schedule[task][0]

    def get_mode(self, task: Task) -> int:
        return self.modes[task]

    def copy(self) -> Solution:
        return RcpspResourceDependentSolution(
            problem=self.problem,
            schedule=deepcopy(self.schedule),
            modes=deepcopy(self.modes),
        )

    def __init__(
        self,
        problem: "RcpspResourceDependentProblem",
        schedule: dict[Task, tuple[int, int]],
        modes: dict[Task, int],
    ):
        super().__init__(problem)
        self.problem = problem
        self.schedule = schedule
        self.modes = modes


class RcpspResourceDependentProblem(
    GenericSchedulingProblem[
        Task, NoUnaryResource, NoSkill, NonSkillCumulativeResource, NonRenewableResource
    ],
    WithoutSkillProblem[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
    ],
    WithoutAllocationProblem[Task],
):
    @property
    def non_skill_cumulative_resources_list(self) -> list[Skill]:
        return [r for r in self.resources if r not in self.non_renewable_resources]

    def is_non_renewable_resource_task_mode_consumption_dependent(
        self, resource: NonRenewableResource, task: Task, mode: int
    ):
        if isinstance(self.mode_details[task][mode].get(resource, 0), int):
            return False
        if isinstance(self.mode_details[task][mode].get(resource, 0), dict):
            return True
        return None

    def is_cumulative_resource_task_mode_consumption_dependent(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> bool:
        # To be Overridden in child classes
        if isinstance(self.mode_details[task][mode].get(resource, 0), int):
            return False
        if isinstance(self.mode_details[task][mode].get(resource, 0), dict):
            return True
        return None

    def get_cumulative_resource_consumption_mapping(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> dict[frozenset[tuple[Task, int]], int]:
        # To be Overridden in child classes
        if self.is_cumulative_resource_task_mode_consumption_dependent(
            resource, task, mode
        ):
            return self.mode_details[task][mode][resource]
        return {
            frozenset([]): self.get_cumulative_resource_consumption(
                resource, task, mode
            )
        }

    def get_non_renewable_resource_consumption_mapping(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> dict[frozenset[tuple[Task, int]], int]:
        if self.is_non_renewable_resource_task_mode_consumption_dependent(
            resource, task, mode
        ):
            return self.mode_details[task][mode][resource]
        return {
            frozenset([]): self.get_non_renewable_resource_consumption(
                resource, task, mode
            )
        }

    def get_cumulative_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> int:
        return self.mode_details[task][mode].get(resource, 0)

    @wrapt.lru_cache(maxsize=None)
    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        return convert_calendar_to_availability_intervals(
            calendar=self.resources[resource], horizon=self.horizon
        )

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        return self.mode_details[task][mode]["duration"]

    @property
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        return self.non_renewable_resources

    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        capacity = self.resources[resource]
        if np.isscalar(capacity):
            return capacity
        else:
            return capacity[0]

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        return self.mode_details[task][mode].get(resource, 0)

    def get_precedence_constraints(self) -> dict[Task, Iterable[Task]]:
        return self.successors

    def get_no_overlap(self) -> set[frozenset[Task]]:
        return {}

    def get_makespan_upper_bound(self) -> int:
        return self.horizon

    def get_task_modes(self, task: Task) -> set[int]:
        return list(self.mode_details[task].keys())

    @property
    def tasks_list(self) -> list[Task]:
        return self._tasks_list

    def evaluate(self, variable: Solution) -> dict[str, float]:
        makespan = self.compute_subobjective(variable, objective=Objective.MAKESPAN)
        return {"makespan": makespan}

    def get_solution_type(self) -> type[Solution]:
        return RcpspResourceDependentSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.SINGLE,
            dict_objective_to_doc={
                "makespan": ObjectiveDoc(TypeObjective.OBJECTIVE, default_weight=1)
            },
        )

    def __init__(
        self,
        resources: dict[str, Union[int, list[int]]],
        non_renewable_resources: list[str],
        mode_details: dict[Hashable, dict[int, dict[str, int]]],
        successors: dict[Hashable, list[Hashable]],
        horizon: int,
        tasks_list: Optional[list[Hashable]] = None,
        source_task: Optional[Hashable] = None,
        sink_task: Optional[Hashable] = None,
    ):
        self.resources = resources
        self.non_renewable_resources = non_renewable_resources
        self.mode_details = mode_details
        self.successors = successors
        self.horizon = horizon
        self._tasks_list = tasks_list
        if tasks_list is None:
            self._tasks_list = list(self.mode_details.keys())
        self.source_task = source_task
        self.sink_task = sink_task
