#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Generic, Hashable

from discrete_optimization.generic_tasks_tools.allocation import (
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import (
    AbsentValue,
)
from discrete_optimization.generic_tasks_tools.exclusions import (
    ExclusionResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.precedence import (
    WithoutPrecedenceProblem,
    WithoutPrecedenceSolution,
)
from discrete_optimization.generic_tasks_tools.skill import (
    Skill,
)

# Hypothesis: No NonSkillCumulativeResource
NoNonSkillCumulativeResource = Hashable

CumulativeResource = Skill | NoNonSkillCumulativeResource
Resource = CumulativeResource | UnaryResource
AnyResource = NonRenewableResource | Resource


class GenericAllocationProblem(
    GenericSchedulingProblem[
        Task,
        UnaryResource,
        Skill,
        NoNonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
    WithoutPrecedenceProblem[Task],
    Generic[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Allocation problem with all optional features

    This class derives from other mixins to provide utilities that require that mix:
    - multimode: the tasks have several mode on which the duration depends
    - allocation: the tasks can have unary resources allocated to them
    - skill: some cumulative resource are skills that are brought to tasks by allocated unary resources
    - non-renewable: the tasks consume non-renewable resources according to the chosen mode  (same as cumulative resource when no time involved)
    - cost: the choice of a mode or of an allocation has a given cost

    Even though this class is generic but encompasses also more specific cases:
    - singlemode: actually only one mode per task
    - no skills: if skills_list is empty
    - no allocation: unary_resources is empty
    - no non-renewable ressources: if non_renewable_resources_list empty
    - no cost: cost = 0

    Note:
        We assume non non-skill cumulative resources which can be viewed as non-renewable resources in this context


    This generic class is to be used to construct generic automatic solvers.

    """

    def get_makespan_upper_bound(self) -> int:
        return 1

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        return 1

    @property
    def non_skill_cumulative_resources_list(self) -> list[NoNonSkillCumulativeResource]:
        """Get the cumulative resource list other that skill.

        Hypothesis: in pure allocation problem, we have only skills as cumulative resource.
        Other resources can be seen as non-renewable resources as task does not end (or even start).

        """
        return []

    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        raise RuntimeError(
            "Resource availability intervals have no meaning when no scheduling is taken into account.\n"
            "Indeed unary resources can be allocated to several tasks at once, their skills also , "
            "and cumulative resources should be converted to non-renewable resources to avoid confusion."
        )

    def satisfy_partial(
        self,
        variable: "GenericSchedulingSolution",
        non_renewable_capacity: bool = True,
        skill: bool = True,
        allocation: bool = True,
        mode_constraints: bool = True,
        optional_tasks: bool = True,
        alternative_scheduling: bool = True,
        exclusion_resources: bool = True,
        **kwargs,
    ) -> bool:
        """Partial checks on solution.

        One can switch off some checks by setting the corresponding parameter to False.

        """
        duration = False
        calendar = False
        precedence = False
        time_lags = False
        time_windows = False
        no_overlap = False
        forbidden_intervals = False
        resource_blocking = False

        return super().satisfy_partial(
            variable=variable,
            calendar=calendar,
            duration=duration,
            non_renewable_capacity=non_renewable_capacity,
            skill=skill,
            allocation=allocation,
            mode_constraints=mode_constraints,
            precedence=precedence,
            time_lags=time_lags,
            time_windows=time_windows,
            no_overlap=no_overlap,
            forbidden_intervals=forbidden_intervals,
            resource_blocking=resource_blocking,
            optional_tasks=optional_tasks,
            alternative_scheduling=alternative_scheduling,
            exclusion_resources=exclusion_resources,
        )


class GenericAllocationSolution(
    GenericSchedulingSolution[
        Task,
        UnaryResource,
        Skill,
        NoNonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
    WithoutPrecedenceSolution[Task],
    Generic[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Solution type associated to GenericAllocationProblem."""

    problem: GenericAllocationProblem[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ]

    def get_start_time(self, task: Task) -> int | AbsentValue:
        return 0

    def get_end_time(self, task: Task) -> int | AbsentValue:
        return 1
