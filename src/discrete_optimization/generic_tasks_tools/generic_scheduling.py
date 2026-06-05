#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from functools import cache
from typing import Generic, Optional

from discrete_optimization.generic_tasks_tools.allocation import (
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
    NonRenewableResourceProblem,
    NonRenewableResourceSolution,
)
from discrete_optimization.generic_tasks_tools.precedence_scheduling import (
    PrecedenceSchedulingProblem,
    PrecedenceSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
    SkillProblem,
    SkillSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpm import Cpm
from discrete_optimization.generic_tasks_tools.timelag import (
    TimelagProblem,
    TimelagSolution,
)
from discrete_optimization.generic_tasks_tools.timewindow import (
    TimewindowProblem,
    TimewindowSolution,
)

CumulativeResource = Skill | NonSkillCumulativeResource
Resource = CumulativeResource | UnaryResource


class GenericSchedulingProblem(
    SkillProblem[Task, UnaryResource, Skill, NonSkillCumulativeResource, UnaryResource],
    NonRenewableResourceProblem[Task, NonRenewableResource],
    PrecedenceSchedulingProblem[Task],
    TimelagProblem[Task],
    TimewindowProblem[Task],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
):
    """Scheduling problem with all optional features

    This class derives from other mixins to provide utilities that require that mix:
    - scheduling: tasks need to be scheduled
    - calendar: the renewable resources have their own calendar that will be used for constraining allocations and schedule
    - multimode: the tasks have several mode on which the duration depends
    - cumulative: the tasks consume cumulative resources according to the chosen mode
    - allocation: the tasks can have unary resources allocated to them
    - skill: some cumulative resource are skills that are brought to tasks by allocated unary resources
    - non-renewable: the tasks consume non-renewable resources according to the chosen mode
    - precedence: precedence constraints between tasks

    Even though this class is generic but encompasses also more specific cases:
    - singlemode: actually only one mode per task
    - no skills: if skills_list is empty
    - no allocation: unary_resources is empty
    - no cumulative ressources: if resources_list list only unary resources
    - no calendar: resource capacity can be given as a constant on [0, horizon)
    - no non-renewable ressources: if non_renewable_resources_list empty
    - no precedence constraints: precedence constraints empty

    We suppose that all renewable resources are
    - either cumulative ones
    - or unary resources

    This generic class is to be used to construct generic automatic solvers (e.g. ).

    """

    @property
    def calendar_resources_list(self) -> list[Resource]:
        return self.unary_resources_list + self.cumulative_resources_list

    def check_calendar_resources_list(self) -> None:
        """Check calendar resources list.

        Raises:
            AssertionError: if duplicates appear in the list

        Returns:

        """
        calendar_resources_list = (
            self.unary_resources_list + self.cumulative_resources_list
        )
        assert len(calendar_resources_list) == len(set(calendar_resources_list)), (
            "There are duplicates in calendar resources list, "
            "potentially because unary and cumulative resources intersect."
        )

    def update_resource_availabilities(self) -> None:
        super().update_resource_availabilities()
        self.check_calendar_resources_list()

    def is_unary_resource(self, resource: Resource) -> bool:
        """Check if given resource is a unary resource."""
        return resource in self.unary_resources_list

    @cache
    def get_task_start_or_end_tighter_lower_bound(
        self,
        task: Task,
        start_or_end: StartOrEnd,
        use_cpm: bool = False,
        horizon: Optional[int] = None,
    ) -> int:
        """Get a tighter lower bound on task start or end using possible durations.

        Args:
            use_cpm: whether to use CPM propagating bounds through precedence graph
            horizon: new horizon to take into account when computing tighter bounds,
                default to problem horizon.

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()
        if use_cpm:
            tasks_bounds = self.compute_tighter_task_bounds(
                use_cpm=True, horizon=horizon
            )
            start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound = (
                tasks_bounds[task]
            )
            match start_or_end:
                case StartOrEnd.START:
                    return start_lower_bound
                case _:
                    return end_lower_bound
        else:
            if start_or_end == StartOrEnd.START:
                return max(  # best bound between:
                    # default bound
                    0,
                    # initial pb bound
                    self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.START
                    ),
                    # bound deduced from initial end lower bound and max duration
                    self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.END
                    )
                    - max(
                        self.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.get_task_modes(task)
                    ),
                )
            else:
                return max(  # best bound between:
                    # default bound
                    0,
                    # initial pb bound
                    self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.END
                    ),
                    # bound deduced from initial start lower bound and min duration
                    max(
                        self.get_task_start_or_end_lower_bound(
                            task=task, start_or_end=StartOrEnd.START
                        ),
                        0,  # clip at 0
                    )
                    + min(
                        self.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.get_task_modes(task)
                    ),
                )

    @cache
    def get_task_start_or_end_tighter_upper_bound(
        self,
        task: Task,
        start_or_end: StartOrEnd,
        use_cpm: bool = False,
        horizon: Optional[int] = None,
    ) -> int:
        """Get a tighter upper bound on task start or end using possible durations.

        Args:
            use_cpm: whether to use CPM propagating bounds through precedence graph
            horizon: new horizon to take into account when computing tighter bounds,
                default to problem horizon.

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()
        if use_cpm:
            tasks_bounds = self.compute_tighter_task_bounds(
                use_cpm=True, horizon=horizon
            )
            start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound = (
                tasks_bounds[task]
            )
            match start_or_end:
                case StartOrEnd.START:
                    return start_upper_bound
                case _:
                    return end_upper_bound
        else:
            # no propagation, only using problem bounds + possible durations + new horizon
            if start_or_end == StartOrEnd.START:
                return min(  # best bound between:
                    # initial pb bound
                    self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.START
                    ),
                    # bound deduced from initial end upper bound (clipped at new horizon) and min duration
                    min(
                        self.get_task_start_or_end_upper_bound(
                            task=task, start_or_end=StartOrEnd.END
                        ),
                        horizon,  # new horizon
                    )
                    - min(  # min duration
                        self.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.get_task_modes(task)
                    ),
                )
            else:
                return min(  # best bound between:
                    # default bound: new horizon
                    horizon,
                    # initial pb bound
                    self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.END
                    ),
                    # bound deduced from initial start upper bound and max duration
                    self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.START
                    )
                    + max(
                        self.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.get_task_modes(task)
                    ),
                )

    def update_task_bounds(self) -> None:
        """Method to be called when problem time windows are updated.

        It clears necessary cache on computed tighter bounds.

        """
        self.get_task_start_or_end_tighter_upper_bound.cache_clear()
        self.get_task_start_or_end_tighter_lower_bound.cache_clear()
        self.compute_tighter_task_bounds.cache_clear()

    @cache
    def compute_tighter_task_bounds(
        self, use_cpm: bool = False, horizon: Optional[int] = None
    ) -> dict[Task, tuple[int, int, int, int]]:
        """Compute tighter task bounds from problem time windows and min-max tak durations.

        Args:
            use_cpm: whether to use CPM propagating bounds through precedence graph
            horizon: new horizon to take into account when computing tighter bounds,
                default to problem horizon.

        Returns:
            {task: (start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound)}

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()
        if use_cpm:
            cpm = Cpm(problem=self, horizon=horizon)
            cpm.compute_task_bounds()
            return cpm.get_task_bounds()
        else:
            return {
                task: (
                    self.get_task_start_or_end_tighter_lower_bound(
                        task=task,
                        start_or_end=StartOrEnd.START,
                        horizon=horizon,
                        use_cpm=False,
                    ),
                    self.get_task_start_or_end_tighter_lower_bound(
                        task=task,
                        start_or_end=StartOrEnd.END,
                        horizon=horizon,
                        use_cpm=False,
                    ),
                    self.get_task_start_or_end_tighter_upper_bound(
                        task=task,
                        start_or_end=StartOrEnd.START,
                        horizon=horizon,
                        use_cpm=False,
                    ),
                    self.get_task_start_or_end_tighter_upper_bound(
                        task=task,
                        start_or_end=StartOrEnd.END,
                        horizon=horizon,
                        use_cpm=False,
                    ),
                )
                for task in self.tasks_list
            }


class GenericSchedulingSolution(
    SkillSolution[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, UnaryResource
    ],
    NonRenewableResourceSolution[Task, NonRenewableResource],
    PrecedenceSchedulingSolution[Task],
    TimelagSolution[Task],
    TimewindowSolution[Task],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
):
    """Solution type associated to GenericSchedulingProblem."""

    problem: GenericSchedulingProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]

    def get_calendar_resource_consumption(self, resource: Resource, task: Task) -> int:
        """"""
        if self.problem.is_unary_resource(resource=resource):
            # unary resources: 0 (not allocated) or 1 (allocated)
            return int(self.is_allocated(task=task, unary_resource=resource))
        else:
            # cumulative resources
            return super().get_calendar_resource_consumption(
                resource=resource, task=task
            )
