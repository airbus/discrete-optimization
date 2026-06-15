#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Module containing mixins for skills.

A skill is a cumulative resource which is attached to a unary resource.

"""

import logging
from abc import abstractmethod
from collections.abc import Hashable
from typing import Generic, TypeVar

import wrapt

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    merge_resources_availability_intervals,
)
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResourceProblem,
    CumulativeResourceSolution,
    OtherCalendarResource,
)

logger = logging.getLogger(__name__)

Skill = TypeVar("Skill", bound=Hashable)
NonSkillCumulativeResource = TypeVar("NonSkillCumulativeResource", bound=Hashable)
CumulativeResource = Skill | NonSkillCumulativeResource
Resource = CumulativeResource | OtherCalendarResource


class SkillProblem(
    CumulativeResourceProblem[Task, CumulativeResource, OtherCalendarResource],
    AllocationProblem[Task, UnaryResource],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
):
    only_one_skill_per_task: bool = False
    """Only one skill from each unary resource allocated to a given task can be used for the task."""

    @property
    @abstractmethod
    def skills_list(self) -> list[Skill]:
        """List of skills needed by tasks and brought by unary resources."""
        ...

    @property
    @abstractmethod
    def non_skill_cumulative_resources_list(self) -> list[Skill]:
        """List of cumulative resources that are not skills."""
        ...

    @property
    def cumulative_resources_list(self) -> list[CumulativeResource]:
        return self.non_skill_cumulative_resources_list + self.skills_list

    @abstractmethod
    def get_unary_resource_skill_value(
        self, unary_resource: UnaryResource, skill: Skill
    ) -> int:
        """Skill value of given resource for given skill."""
        ...

    @wrapt.lru_cache(maxsize=None)
    def get_skills_of_task(self, task: Task) -> set[Skill]:
        return {
            skill
            for skill in self.skills_list
            if any(
                self.get_cumulative_resource_consumption(
                    task=task, resource=skill, mode=mode
                )
                > 0
                for mode in self.get_task_modes(task=task)
            )
        }

    @wrapt.lru_cache(maxsize=None)
    def get_unary_resource_with_skill(self, skill: Skill) -> set[UnaryResource]:
        return {
            unary_resource
            for unary_resource in self.unary_resources_list
            if self.get_unary_resource_skill_value(
                unary_resource=unary_resource, skill=skill
            )
            > 0
        }

    @wrapt.lru_cache(maxsize=None)
    def get_skills_of_unary_resource(self, unary_resource: UnaryResource) -> set[Skill]:
        return {
            skill
            for skill in self.skills_list
            if self.get_unary_resource_skill_value(
                unary_resource=unary_resource, skill=skill
            )
            > 0
        }

    def compute_skill_availabilities(self, skill: Skill) -> list[tuple[int, int, int]]:
        """Deduce skill availabilities from unary_resource availabilities and skill values."""
        return merge_resources_availability_intervals(
            intervals_per_resource=[
                [
                    (start, end, int(is_present) * skill_value)
                    for (start, end, is_present) in self.get_resource_availabilities(
                        resource=unary_resource
                    )
                ]
                for unary_resource in self.unary_resources_list
                if (
                    skill_value := self.get_unary_resource_skill_value(
                        unary_resource=unary_resource, skill=skill
                    )
                )
                > 0
            ],
            horizon=self.get_makespan_upper_bound(),
        )

    def update_skills(self):
        self.get_skills_of_task.cache_clear()
        self.get_unary_resource_with_skill.cache_clear()
        self.get_skills_of_unary_resource.cache_clear()


class SkillSolution(
    CumulativeResourceSolution[Task, CumulativeResource, OtherCalendarResource],
    AllocationSolution[Task, UnaryResource],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
):
    problem: SkillProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ]

    @abstractmethod
    def is_skill_used(
        self, task: Task, unary_resource: UnaryResource, skill: Skill
    ) -> bool:
        """Tell whether the given skill from given unary_resource is used in given task.

        If `True`, `self.is_allocated(task, unary_resource)` must also be `True`.
        If the skill is not needed by the task or not in unary_resource skills, should return False.

        """
        ...

    def get_skill_value_on_task(self, task: Task, skill: Skill) -> int:
        return sum(
            self.problem.get_unary_resource_skill_value(
                unary_resource=unary_resource, skill=skill
            )
            for unary_resource in self.problem.unary_resources_list
            if self.is_skill_used(task=task, unary_resource=unary_resource, skill=skill)
        )

    def check_skill_constraint(
        self, task: Task, skill: Skill, exact: bool = False, slack: int = 0
    ) -> bool:
        value_required = self.get_calendar_resource_consumption(
            task=task, resource=skill
        )
        if value_required == 0:
            return True
        else:
            value_from_unary_resources = self.get_skill_value_on_task(
                task=task, skill=skill
            )
            if value_from_unary_resources < value_required:
                logger.debug(
                    f"Violation of skill constraint for task {task} and skill {skill}"
                )
                return False
            if exact and value_from_unary_resources > value_required + slack:
                logger.debug(
                    f"Violation of exact skill constraint for task {task} and skill {skill}"
                )
                return False
            else:
                return True

    def check_skill_constraints(self, exact: bool = False, slack: int = 0) -> bool:
        return all(
            self.check_skill_constraint(
                task=task, skill=skill, exact=exact, slack=slack
            )
            for task in self.problem.tasks_list
            for skill in self.problem.skills_list
        )

    def check_skill_usage_and_allocation_consistency(self) -> bool:
        for task in self.problem.tasks_list:
            for unary_resource in self.problem.unary_resources_list:
                is_allocated = self.is_allocated(
                    task=task, unary_resource=unary_resource
                )
                for skill in self.problem.skills_list:
                    is_skill_used = self.is_skill_used(
                        task=task, unary_resource=unary_resource, skill=skill
                    )
                    if is_allocated < is_skill_used:
                        logger.debug(
                            f"Skill {skill} from unary_resource {unary_resource} is used for task {task}, "
                            "but the unary_resource is not allocated to the task."
                        )
                        return False
                    has_skill = (
                        self.problem.get_unary_resource_skill_value(
                            unary_resource=unary_resource, skill=skill
                        )
                        > 0
                    )
                    if is_skill_used > has_skill:
                        logger.debug(
                            f"Skill {skill} from unary_resource {unary_resource} is used for task {task}, "
                            "but the unary_resource has not this skill."
                        )
                        return False

        return True

    def check_only_one_skill_per_task_and_unary_resource(self):
        # only one skill per task ?
        if self.problem.only_one_skill_per_task:
            for task in self.problem.tasks_list:
                for unary_resource in self.problem.unary_resources_list:
                    if (
                        sum(
                            self.is_skill_used(
                                task=task, unary_resource=unary_resource, skill=skill
                            )
                            for skill in self.problem.get_skills_of_unary_resource(
                                unary_resource=unary_resource
                            )
                        )
                        > 1
                    ):
                        logger.debug(
                            f"The unary resource {unary_resource} is using more than one skill for task {task}."
                        )
                        return False

        return True


NoSkill = None


class WithoutSkillProblem(
    SkillProblem[
        Task, UnaryResource, NoSkill, NonSkillCumulativeResource, OtherCalendarResource
    ],
    Generic[Task, UnaryResource, NonSkillCumulativeResource, OtherCalendarResource],
):
    @property
    def skills_list(self) -> list[Skill]:
        return []

    def get_unary_resource_skill_value(
        self, unary_resource: UnaryResource, skill: Skill
    ) -> int:
        return 0


class WithoutSkillSolution(
    SkillSolution[
        Task, UnaryResource, NoSkill, NonSkillCumulativeResource, OtherCalendarResource
    ],
    Generic[Task, UnaryResource, NonSkillCumulativeResource, OtherCalendarResource],
):
    def is_skill_used(
        self, task: Task, unary_resource: UnaryResource, skill: Skill
    ) -> bool:
        return False

    def check_skill_constraint(
        self, task: Task, skill: NoSkill, exact: bool = False, slack: int = 0
    ) -> bool:
        return True

    def check_skill_constraints(self, exact: bool = False, slack: int = 0) -> bool:
        return True

    def check_skill_usage_and_allocation_consistency(self) -> bool:
        return True
