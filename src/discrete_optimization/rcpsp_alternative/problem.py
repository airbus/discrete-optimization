#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Implementation of RCPSP with optional alternative subproblems
#  Between 2 mandatory task, different subpath of task should be accomplished (with or without precedence constraints).
#  This problem can represent different alternative physical path of task to accomplish on a shop floor.
import logging
from typing import Any, Hashable, Optional, Union

from discrete_optimization.generic_tasks_tools import AbsentValue
from discrete_optimization.generic_tasks_tools.alternative_subproblems import (
    AlternativeSchedulingSubProblem,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import (
    NonRenewableResource,
    RcpspSolution,
    Resource,
    Task,
)
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)

logger = logging.getLogger(__name__)


class RcpspWithAlternativePath(RcpspProblem):
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
        name_task: Optional[dict[Hashable, str]] = None,
        calendar_details: Optional[dict[str, list[list[int]]]] = None,
        special_constraints: Optional[SpecialConstraintsDescription] = None,
        fixed_permutation: Optional[list[int]] = None,
        fixed_modes: Optional[list[int]] = None,
        alternative_tasks: Optional[list[Hashable]] = None,
        list_alternative_subproblem: list[AlternativeSchedulingSubProblem] = None,
        **kwargs: Any,
    ):
        """
        Extension of RCPSPProblem, including
        :param alternative_tasks: tasks that are not mandatory
        :param alternative_tasks_data: data of the tasks when they are done (duration, resource usage),
         given per mode (like mode_details attribute)
        :param alternative_successors: successors of optional task (when active).
        The successors can be either optional or mandatory task.
        :param list_alternative_subproblem: list of alternative scheduling subproblem, describing the alternative paths.
        """
        self.alternative_tasks = alternative_tasks
        self.list_alternative_subproblem = list_alternative_subproblem
        super().__init__(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            tasks_list=tasks_list,
            source_task=source_task,
            sink_task=sink_task,
            name_task=name_task,
            calendar_details=calendar_details,
            special_constraints=special_constraints,
            fixed_permutation=fixed_permutation,
            fixed_modes=fixed_modes,
            **kwargs,
        )

    def is_optional(self, task: Task) -> bool:
        return task in self.alternative_tasks

    def get_alternative_scheduling_subproblem(
        self,
    ) -> list[AlternativeSchedulingSubProblem]:
        return self.list_alternative_subproblem

    def get_cumulative_resource_consumption(
        self, resource: Resource, task: Task, mode: int
    ) -> int:
        if mode is None or mode == AbsentValue.ABSENT:
            return 0
        if task in self.mode_details:
            return self.mode_details[task][mode].get(resource, 0)
        return 0

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        if mode is None or mode == AbsentValue.ABSENT:
            return 0
        mode_detail = self.mode_details[task][mode]
        return mode_detail.get(resource, 0)

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        if mode is None or mode == AbsentValue.ABSENT:
            return 0
        if task in self._tasks_list:
            return self.mode_details[task][mode]["duration"]
        return 0

    def get_task_modes(self, task: Task) -> set[int]:
        return set(self.mode_details[task])


def get_optional_tasks_done(sol: RcpspSolution, problem: RcpspWithAlternativePath):
    return [t for t in problem.alternative_tasks if sol.get_mode(t) is not None]
