#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from typing import Generic, Optional

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.enums import AbsentValue, StartOrEnd
from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
    TaskVariable,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    Task,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.auto import (
    GenericSchedulingAutoCpSolver,
    SinglemodeGenericSchedulingAutoCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import (
    IntervalVar,
    LinearExpr,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.generic_scheduling import (
    GenericSchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    Solution,
)

try:
    import optalcp as cp
except ImportError:
    cp = None


logger = logging.getLogger(__name__)

AnyResource = NonRenewableResource | UnaryResource | Skill | NonSkillCumulativeResource


class GenericSchedulingAutoOptalSolver(
    GenericSchedulingOptalSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        AnyResource,
        ExclusionResource,
    ],
    GenericSchedulingAutoCpSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
    Generic[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Generic optal solver for scheduling problems (with or without allocation).

    The needed variables are automatically created, with common constraints (precedence, resource capacity).
    The objective is set by default to global makespan but can be changed by modifying `solver.default_objective` value.

    This solver class needs still to be derived to create a solution from the proper class.
    You will need at least to implement the conversion from the task variables to the actual solution object.

    If custom constraints are needed, override `init_model()`.
    If a custom objective is needed, set `solver.default_objective` to `Objective.CUSTOM` so that no objective is set
    by default and override `init_model()` to define your objective.

    """

    task_interval_variables: dict[Task, cp.IntervalVar]
    modes_intervals: dict[Task, dict[int, cp.IntervalVar]]
    allocation_intervals: dict[Task, dict[UnaryResource, cp.IntervalVar]]

    def _create_variables(self):
        self._create_interval_variables()
        self._create_interval_mode_variables()
        self._create_allocation_intervals()
        self._create_skill_variables()
        if self.use_energy_constraints:
            self._create_energy_variables()

    def _create_interval_variables(self):
        for task in self.problem.tasks_list:
            modes = self.problem.get_task_modes(task=task)
            possible_durations = set(
                self.problem.get_task_mode_duration(task=task, mode=mode)
                for mode in modes
            )
            self.task_interval_variables[task] = self.cp_model.interval_var(
                start=(
                    self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.START
                    ),
                    self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.START
                    ),
                ),
                end=(
                    self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.END
                    ),
                    self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.END
                    ),
                ),
                length=(min(possible_durations), max(possible_durations)),
                optional=self.create_present_task_variables_for_all_tasks
                or self.problem.is_optional(task),
                name=f"interval_{task}",
            )

    def _create_interval_mode_variables(self):
        for task in self.problem.tasks_list:
            task_interval = self.task_interval_variables[task]
            modes = self.problem.get_task_modes(task)
            self.modes_intervals[task] = {}
            if len(modes) == 1:
                mode = next(iter(modes))
                self.modes_intervals[task][mode] = task_interval
            else:
                for mode in modes:
                    self.modes_intervals[task][mode] = self.cp_model.interval_var(
                        start=(task_interval.start_min, task_interval.start_max),
                        end=(task_interval.end_min, task_interval.end_max),
                        length=self.problem.get_task_mode_duration(
                            task=task, mode=mode
                        ),
                        optional=True,
                        name=f"interval_mode_{task}_{mode}",
                    )
                # task interval must be one of the mode task intervals
                task_interval.alternative(self.modes_intervals[task].values())

    def _create_allocation_intervals(self):
        for task in self.problem.tasks_list:
            self.allocation_intervals[task] = {}
            task_interval = self.task_interval_variables[task]
            for unary_resource in self.problem.unary_resources_list:
                if self.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                ):
                    allocation_interval = self.cp_model.interval_var(
                        start=(task_interval.start_min, task_interval.start_max),
                        end=(task_interval.end_min, task_interval.end_max),
                        length=(task_interval.length_min, task_interval.length_max),
                        optional=True,
                        name=f"interval_allocated_{task}_{unary_resource}",
                    )
                    self.allocation_intervals[task][unary_resource] = (
                        allocation_interval
                    )
                    # synchro start and end
                    allocation_interval.start_at_start(task_interval)
                    allocation_interval.end_at_end(task_interval)
                    # cannot exist if the task is absent i.e. the main interval is absent
                    self.add_implication(
                        ~task_interval.presence(), ~allocation_interval.presence()
                    )
            # redundant alternative constraint if exactly one resource will be allocated
            if self.exactly_one_unary_resource_per_task or (
                # if use only skill to allocate, at_most_one_unary_resource_per_task is exactly one for each task needing a skill
                self.at_most_one_unary_resource_per_task
                and self.use_only_skill_to_allocate
            ):
                # main interval is one of the allocation intervals
                # (this is a redundant constraint but maybe useful to the solver)
                allocation_intervals = self.allocation_intervals[task].values()
                if len(allocation_intervals) > 0:
                    task_interval.alternative(allocation_intervals)

    def get_duration_variable(
        self, task: Task, task_interval_will_exist: Optional[bool] = None
    ) -> LinearExpr:
        return self.task_interval_variables[task].length()

    def get_task_interval(self, task: Task) -> cp.IntervalVar:
        return self.task_interval_variables[task]

    def get_task_is_present_variable(self, task: Task) -> LinearExpr:
        return self.task_interval_variables[task].presence()

    def get_task_unary_resource_interval(
        self, task: Task, unary_resource: UnaryResource
    ) -> IntervalVar:
        return self.allocation_intervals[task][unary_resource]

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExpr:
        try:
            return self.allocation_intervals[task][unary_resource].presence()
        except KeyError:
            return 0

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExpr:
        itv: cp.IntervalVar = self.get_task_interval(task)
        if start_or_end == StartOrEnd.START:
            return itv.start()
        else:
            return itv.end()

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExpr:
        return self.modes_intervals[task][mode].presence()

    def get_task_mode_interval(self, task: Task, mode: int) -> cp.IntervalVar:
        return self.modes_intervals[task][mode]

    def retrieve_solution(self, result: "cp.Solution") -> Solution:
        # construct generic tasks variables
        raw_sol = self.retrieve_tasks_variables_optal(solution=result)
        # convert to specific solution type
        return self.convert_task_variables_to_solution(raw_sol=raw_sol)

    def retrieve_tasks_variables_optal(
        self, solution: "cp.Solution"
    ) -> RawSolution[Task, UnaryResource, Skill]:
        """Construct each task variable from the optal internal solution.

        We cannot use the generic `retrieve_tasks_variables()` as the way to retrieve the value of a given expression
        or variable may vary. Often we need to go back to the interval variable it is derived from.

        This method will be called in `self.retrieve_solution()` before `self.convert_task_variables_to_solution()`.

        Args:
            solution:

        Returns:
            the task variables for the intermediate solution

        """
        task_variables = {}
        for task in self.problem.tasks_list:
            itv = self.get_task_interval(task)
            is_present = solution.is_present(itv)
            if not is_present:
                task_variables[task] = TaskVariable(
                    start=AbsentValue.ABSENT,
                    end=AbsentValue.ABSENT,
                    mode=AbsentValue.ABSENT,
                    allocated={},
                    is_present=False,
                )
            else:
                start, end = solution.get_value(itv)
                modes = self.problem.get_task_modes(task)
                mode = AbsentValue.ABSENT
                if len(modes) == 1:
                    mode = next(iter(modes))
                else:
                    for mode in modes:
                        if solution.is_present(
                            self.get_task_mode_interval(task=task, mode=mode)
                        ):
                            break

                def get_skill_used(
                    task: Task, unary_resource: UnaryResource
                ) -> set[Skill]:
                    """Get set of skills used by unary resource for the task

                    Hypothesis: the unary resource is allocated to the task. No need to check it again.

                    Args:
                        task:
                        unary_resource:

                    Returns:

                    """
                    try:
                        skill_variables = self.skill_variables[task][unary_resource]
                    except KeyError:
                        return set()
                    else:
                        skills = set()
                        for skill, skill_var in skill_variables.items():
                            if isinstance(skill_var, cp.BoolExpr):
                                # When skill_var is not a variable but an expression,
                                # we cannot use `solution.get_value()` on it (with optal 2026.11.0)
                                # It is the case where the use of the skill is directly
                                # defined by the fact that the unary resource is allocated to the task.
                                # As `get_skill_used()` is called only when `unary_resource` is allocated,
                                # we add the skill.
                                skills.add(skill)
                            else:  # bool variable
                                skill_used = solution.get_value(skill_variables[skill])
                                if skill_used > 0:
                                    skills.add(skill)
                        return skills

                allocated = {
                    unary_resource: get_skill_used(
                        task=task, unary_resource=unary_resource
                    )
                    for unary_resource, allocation_interval in self.allocation_intervals[
                        task
                    ].items()
                    if (solution.is_present(allocation_interval))
                }
                task_variables[task] = TaskVariable(
                    start=start,
                    end=end,
                    mode=mode,
                    allocated=allocated,
                    is_present=True,
                )
        metadata = {}
        # Stores other decision
        for c in (
            self.problem.get_flexible_gap_blocking_constraints()
            + self.problem.get_span_blocking_constraints()
        ):
            if c.has_a_choice():
                tag = c.metadata.name_choice
                choice = None
                for val in self._choices_blocking_constraint_vars[tag]:
                    if solution.get_value(
                        self._choices_blocking_constraint_vars[tag][val]
                    ):
                        choice = val
                        break
                metadata[tag] = choice

        return RawSolution(task_variables=task_variables, metadata=metadata)


class SinglemodeGenericSchedulingAutoOptalSolver(
    SinglemodeGenericSchedulingAutoCpSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
    GenericSchedulingAutoOptalSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Subclass of GenericSchedulingAutoOptalSolver for single mode problems.

    Give access to task intervals without dealing with modes.

    """

    ...
