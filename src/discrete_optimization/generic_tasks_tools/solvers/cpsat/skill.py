#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import Generic

from ortools.sat.python.cp_model import LinearExprT

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.skill import (
    CumulativeResource,
    NonSkillCumulativeResource,
    NoSkill,
    Skill,
    SkillProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.allocation import (
    AllocationCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.cumulative_resource import (
    CumulativeResourceSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.utils import is_a_trivial_zero


class SkillSchedulingCpSatSolver(
    CumulativeResourceSchedulingCpSatSolver[
        Task, CumulativeResource, OtherCalendarResource
    ],
    AllocationCpSatSolver[Task, UnaryResource],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
):
    """Base class for cpsat solvers dealing with scheduling problems handling skills attached to unary resources."""

    use_exact_skill: bool = False
    """Allocate exactly the needed skill value to each task."""
    use_slack_for_skill: bool = False
    """Allow some additional slack on skill value, even when `use_exact_skill` is activated."""
    max_slack_for_skill: int = 5
    """Maximum slack for skill value."""
    use_only_skill_to_allocate: bool = False
    """Do not allocate a unary_resource if not contributing to a skill needed by a given task."""

    problem: SkillProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ]

    @abstractmethod
    def get_skill_variable(
        self, task: Task, unary_resource: UnaryResource, skill: Skill
    ) -> LinearExprT:
        """Get skill boolean variable telling if given skill is used by given unary resource for given task."""
        ...

    def create_fine_skill_constraints(self):
        """Create constraints on skills using variable on skill contribution of each unary resource."""
        for task in self.problem.tasks_list:
            for skill in self.problem.get_skills_of_task(task):
                skill_value_put_on_task = sum(
                    skill_value * skill_used_var
                    for unary_resource in self.problem.get_unary_resource_with_skill(
                        skill
                    )
                    if (
                        (
                            skill_value := self.problem.get_unary_resource_skill_value(
                                unary_resource=unary_resource, skill=skill
                            )
                        )
                        > 0
                        and not is_a_trivial_zero(
                            (
                                skill_used_var := self.get_skill_variable(
                                    task=task,
                                    unary_resource=unary_resource,
                                    skill=skill,
                                )
                            )
                        )
                    )
                )
                skill_value_needed_by_task = (
                    self.get_cumulative_resource_demand_variable(
                        task=task, resource=skill
                    )
                )
                if self.use_exact_skill:
                    if self.use_slack_for_skill:
                        self.cp_model.add(
                            skill_value_put_on_task >= skill_value_needed_by_task
                        )
                        self.cp_model.add(
                            skill_value_put_on_task
                            <= skill_value_needed_by_task + self.max_slack_for_skill
                        )
                    else:
                        self.cp_model.add(
                            skill_value_put_on_task == skill_value_needed_by_task
                        )
                else:
                    self.cp_model.add(
                        skill_value_put_on_task >= skill_value_needed_by_task
                    )

    def create_coarse_skill_constraints(self):
        """Create constraints on skills using only task allocation of each unary resource."""
        for task in self.problem.tasks_list:
            for skill in self.problem.get_skills_of_task(task):
                skill_value_put_on_task = sum(
                    skill_value * is_present_var
                    for unary_resource in self.problem.get_unary_resource_with_skill(
                        skill
                    )
                    if (
                        (
                            skill_value := self.problem.get_unary_resource_skill_value(
                                unary_resource=unary_resource, skill=skill
                            )
                        )
                        > 0
                        and not is_a_trivial_zero(
                            (
                                is_present_var
                                := self.get_task_unary_resource_is_present_variable(
                                    task=task, unary_resource=unary_resource
                                )
                            )
                        )
                    )
                )
                skill_value_needed_by_task = (
                    self.get_cumulative_resource_demand_variable(
                        task=task, resource=skill
                    )
                )
                self.cp_model.add(skill_value_put_on_task >= skill_value_needed_by_task)

    def is_compatible_task_unary_resource(
        self, task: Task, unary_resource: UnaryResource
    ) -> bool:
        if (
            self.at_most_one_unary_resource_per_task
            or self.exactly_one_unary_resource_per_task
        ):
            # will be the only resource allocated so must bring all needed skill
            if not any(
                all(
                    self.problem.get_unary_resource_skill_value(
                        unary_resource=unary_resource, skill=skill
                    )
                    >= self.problem.get_cumulative_resource_consumption(
                        task=task, mode=mode, resource=skill
                    )
                    for skill in self.problem.get_skills_of_task(task)
                )
                for mode in self.problem.get_task_modes(task)
            ):
                return False
        if self.use_only_skill_to_allocate:
            # the allocation must bring useful skill to the task
            common_skills = self.problem.get_skills_of_task(task).intersection(
                self.problem.get_skills_of_unary_resource(unary_resource)
            )
            if len(common_skills) == 0:
                return False
        return super().is_compatible_task_unary_resource(task, unary_resource)


class WithoutSkillSchedulingCpSatSolver(
    SkillSchedulingCpSatSolver[
        Task, UnaryResource, NoSkill, NonSkillCumulativeResource, OtherCalendarResource
    ],
    Generic[Task, UnaryResource, NonSkillCumulativeResource, OtherCalendarResource],
):
    """Mixin for solver on problems dealing with no skills."""

    def get_skill_variable(
        self, task: Task, unary_resource: UnaryResource, skill: Skill
    ) -> LinearExprT:
        return 0
