#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import defaultdict
from typing import Generic, Optional

import wrapt

from discrete_optimization.generic_tasks_tools.allocation import (
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
    Penalty,
)
from discrete_optimization.generic_tasks_tools.no_overlap import (
    NoOverlapProblem,
    NoOverlapSolution,
)
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
    consolidate_min_time_lags,
)
from discrete_optimization.generic_tasks_tools.timewindow import (
    TimewindowProblem,
    TimewindowSolution,
)

CumulativeResource = Skill | NonSkillCumulativeResource
Resource = CumulativeResource | UnaryResource
AnyResource = NonRenewableResource | Resource


class GenericSchedulingProblem(
    SkillProblem[Task, UnaryResource, Skill, NonSkillCumulativeResource, UnaryResource],
    NonRenewableResourceProblem[Task, NonRenewableResource],
    PrecedenceSchedulingProblem[Task],
    TimelagProblem[Task],
    TimewindowProblem[Task],
    NoOverlapProblem[Task],
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
    - cost: the choice of a mode or of an allocation has a given cost

    Even though this class is generic but encompasses also more specific cases:
    - singlemode: actually only one mode per task
    - no skills: if skills_list is empty
    - no allocation: unary_resources is empty
    - no cumulative ressources: if resources_list list only unary resources
    - no calendar: resource capacity can be given as a constant on [0, horizon)
    - no non-renewable ressources: if non_renewable_resources_list empty
    - no precedence constraints: precedence constraints empty
    - no cost: cost = 0

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

    @wrapt.lru_cache(maxsize=None)
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

    @wrapt.lru_cache(maxsize=None)
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

    @wrapt.lru_cache(maxsize=None)
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

    def get_consolidated_time_lags(
        self,
        task1_start_or_end: StartOrEnd,
        task2_start_or_end: StartOrEnd,
        min_or_max: MinOrMax,
    ):
        """Get consolidated time lags.

        Same normalization as in `TimelagProblem` parent class.
        Also taking into account precedence constraints to enrich end to start min time lags.

        Args:
            task1_start_or_end:
            task2_start_or_end:
            min_or_max:

        Returns:

        """
        timelags = super().get_consolidated_time_lags(
            task1_start_or_end=task1_start_or_end,
            task2_start_or_end=task2_start_or_end,
            min_or_max=min_or_max,
        )
        if (task1_start_or_end, task2_start_or_end, min_or_max) == (
            StartOrEnd.END,
            StartOrEnd.START,
            MinOrMax.MIN,
        ):
            # end to start min time lags: we add precedence constraints and keep only max(resulting offsets)
            timelags = consolidate_min_time_lags(
                timelags
                + [
                    (task1, task2, 0)
                    for task1, next_tasks in self.get_precedence_constraints().items()
                    for task2 in next_tasks
                ]
            )
        return timelags

    def get_makespan_lower_bound(self) -> int:
        """Get a lower bound on global makespan.

        Computed tighter lower bounds on last tasks can be used to get a better makespan lower bound.

        """
        return max(
            self.get_task_start_or_end_tighter_lower_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            for task in self.get_last_tasks()
        )

    def get_makespan_tighter_lower_bound(
        self, use_cpm: bool = False, horizon: Optional[int] = None
    ) -> int:
        """Get a tighter lower bound on global makespan.

        Args:
            use_cpm: whether to use CPM bound propagation through precedence graph to improve tightness
            horizon: new horizon to take into account when computing tighter bounds,
                default to problem horizon.
                NB: The choice of horizon should not affect the result for the lower bound, but it could avoid
                CPM complete recomputation thanks to caching if it was already launched with same horizon.

        """
        if horizon is None:
            horizon = self.get_makespan_tighter_upper_bound()
        return max(
            self.get_task_start_or_end_tighter_lower_bound(
                task=task, start_or_end=StartOrEnd.END, use_cpm=use_cpm, horizon=horizon
            )
            for task in self.get_last_tasks()
        )

    def get_makespan_tighter_upper_bound(self) -> int:
        """Compute a tighter upper bound on makespan.

        The original makespan upper bound is used when computing tighter bounds for tasks starts and ends,
        via `self.compute_tighter_task_bounds()` or `self.get_task_start_or_end_tighter_upper_bound()`.
        From that tighter bounds, we can derive a new makespan upper bound.

        """
        return max(
            # do not use CPM (not necessary as last tasks horizon are just computed from horizon + time windows even in CPM)
            self.get_task_start_or_end_tighter_upper_bound(
                task=task, start_or_end=StartOrEnd.END, use_cpm=False
            )
            for task in self.get_last_tasks()
        )

    @wrapt.lru_cache(maxsize=None)
    def get_consolidated_precedence_constraints(self) -> dict[Task, set[Task]]:
        """Consolidate precedence constraints defined by problem.

        It takes into account time lags constraints.
        - end to start min constraint with non-negative offsets => precedence constraint
        - start synchronization => corresponding tasks should appear together in successors
        - end synchornization => corresponding tasks should share their successors

        """
        successors = defaultdict(set)

        # end to task min timelag with offset >=0  => precedence constraint
        # (original ones already included in consolidated time lags)
        for task1, task2, offset in self.get_consolidated_time_lags(
            task1_start_or_end=StartOrEnd.END,
            task2_start_or_end=StartOrEnd.START,
            min_or_max=MinOrMax.MIN,
        ):
            if offset >= 0:
                successors[task1].add(task2)

        # end together => same successors
        min_end_to_end_timelags_0_offset = [
            (t1, t2)
            for t1, t2, offset in self.get_consolidated_time_lags(
                task1_start_or_end=StartOrEnd.END,
                task2_start_or_end=StartOrEnd.END,
                min_or_max=MinOrMax.MIN,
            )
            if offset == 0
        ]
        max_end_to_end_timelags_0_offset = [
            (t1, t2) for t2, t1 in min_end_to_end_timelags_0_offset
        ]
        end_together = set(min_end_to_end_timelags_0_offset).intersection(
            max_end_to_end_timelags_0_offset
        )
        for task1, task2 in end_together:
            successors[task1].update(successors[task2])
            # the reverse will be done during the loop as (task2, task1) should also be in end_together

        # start together => same predecessors
        min_start_to_start_timelags_0_offset = [
            (t1, t2)
            for t1, t2, offset in self.get_consolidated_time_lags(
                task1_start_or_end=StartOrEnd.START,
                task2_start_or_end=StartOrEnd.START,
                min_or_max=MinOrMax.MIN,
            )
            if offset == 0
        ]
        max_start_to_start_timelags_0_offset = [
            (t1, t2) for t2, t1 in min_start_to_start_timelags_0_offset
        ]
        start_together = set(min_start_to_start_timelags_0_offset).intersection(
            max_start_to_start_timelags_0_offset
        )
        for task, next_tasks in successors.items():
            for task1, task2 in start_together:
                if task1 in next_tasks:
                    next_tasks.add(task2)

        return successors

    def update_time_lags(self) -> None:
        """Method to call when time lags have been updated.

        Clear cache from consolidated precedence constraints and time lags.

        Returns:

        """
        super().get_consolidated_time_lags.cache_clear()  # beware: parent class method also using cache !
        self.get_consolidated_precedence_constraints.cache_clear()

    def update_precedence_constraints(self) -> None:
        """Method to call when precedence constraints have been updated.

        Clear cache from consolidated precedence constraints and time lags.

        Returns:

        """
        self.update_time_lags()

    def compute_subobjective(
        self,
        variable: GenericSchedulingSolution,
        objective: Objective,
        resource_weights: Optional[dict[AnyResource, int]] = None,
    ) -> int:
        """Compute subobjective from given solution."""
        match objective:
            case Objective.MAKESPAN:
                return variable.get_max_end_time()
            case Objective.NB_TASKS_DONE:
                return variable.compute_nb_tasks_done()
            case Objective.NB_UNARY_RESOURCES_USED:
                return variable.compute_nb_unary_resources_used()
            case Objective.NB_RESOURCES_USED:
                return variable.compute_nb_calendar_resources_used(
                    weights=resource_weights
                ) + variable.compute_nb_non_renewable_resources_used(
                    weights=resource_weights
                )
            case Objective.RESOURCES_LEVELS:
                return variable.compute_aggregated_calendar_resources_levels(
                    weights=resource_weights
                ) + variable.compute_aggregated_non_renewable_resources_consumptions(
                    weights=resource_weights
                )
            case Objective.COST:
                return variable.compute_cost()
            case _:
                raise NotImplementedError()

    def compute_penalty(
        self, variable: GenericSchedulingSolution, penalty: Penalty
    ) -> int:
        """Compute penalty from given solution."""
        match penalty:
            case Penalty.TIME:
                penalty = 0
                # time windows
                for task in self.tasks_list:
                    start = variable.get_start_time(task)
                    end = variable.get_end_time(task)
                    start_lb = self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.START
                    )
                    end_lb = self.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.END
                    )
                    start_ub = self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.START
                    )
                    end_ub = self.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.END
                    )
                    penalty += max(0, start_lb - start)
                    penalty += max(0, end_lb - end)
                    penalty += max(0, start - start_ub)
                    penalty += max(0, end - end_ub)
                # time lags
                for task1_start_or_end in StartOrEnd:
                    for task2_start_or_end in StartOrEnd:
                        for min_or_max in MinOrMax:
                            for task1, task2, offset in self.get_original_time_lags(
                                task1_start_or_end=task1_start_or_end,
                                task2_start_or_end=task2_start_or_end,
                                min_or_max=min_or_max,
                            ):
                                t1 = variable.get_start_or_end_time(
                                    task=task1, start_or_end=task1_start_or_end
                                )
                                t2 = variable.get_start_or_end_time(
                                    task=task2, start_or_end=task2_start_or_end
                                )
                                if min_or_max == MinOrMax.MIN:
                                    penalty += max(0, t1 + offset - t2)
                                else:
                                    penalty += max(0, t2 - (t1 + offset))

            case _:
                raise NotImplementedError()

        return penalty

    def get_mode_cost(self, task: Task, mode: int) -> int:
        """Get cost of choosing given mode.

        Default to no cost. To be overridden in child classes with actual costs.

        Args:
            task:
            mode:

        Returns:

        """
        return 0

    def get_unary_resource_cost(
        self, task: Task, mode: int, unary_resource: UnaryResource
    ) -> int:
        """Get cost of allocating given unary resource.

        Default to no cost. To be overridden in child classes with actual costs.

        Args:
            task:
            mode:
            unary_resource:

        Returns:

        """
        return 0

    def satisfy(self, variable: "GenericSchedulingSolution") -> bool:
        return self.satisfy_partial(variable=variable)

    def satisfy_partial(
        self,
        variable: "GenericSchedulingSolution",
        duration: bool = True,
        calendar: bool = True,
        non_renewable_capacity: bool = True,
        precedence: bool = True,
        skill: bool = True,
        allocation: bool = True,
        time_lags: bool = True,
        time_windows: bool = True,
        no_overlap: bool = True,
        forbidden_intervals: bool = True,
    ) -> bool:
        """Partial checks on solution.

        One can switch off some checks by setting the corresponding parameter to False.

        Args:
            variable:
            duration:
            calendar:
            non_renewable_capacity:
            precedence:
            skill:
            allocation:
            time_lags:
            time_windows:
            no_overlap:
            forbidden_intervals:

        Returns:

        """
        return (
            # duration consistency
            (not duration or variable.check_duration_constraints())
            # calendar resources capacity violations (unary resources + skills + cumulative resources)
            and (
                not calendar
                or variable.check_all_calendar_resource_capacity_constraints()
            )
            # non-renewable resource violation
            and (
                not non_renewable_capacity
                or variable.check_all_non_renewable_resource_capacity_constraints()
            )
            # precedence relations
            and (not precedence or variable.check_precedence_constraints())
            # skill constraints
            and (
                not skill
                or (
                    variable.check_skill_constraints()
                    and variable.check_only_one_skill_per_task_and_unary_resource()
                    # Check consistency between compatibility/allocation/skill usage
                    and variable.check_skill_usage_and_allocation_consistency()
                )
            )
            and (not allocation or variable.check_allocation_consistency())
            # time lags
            and (not time_lags or variable.check_time_lags())
            # time window
            and (not time_windows or variable.check_time_windows())
            # no overlap
            and (not no_overlap or variable.check_no_overlap())
            # forbidden intervals
            and (not forbidden_intervals or variable.check_forbidden_intervals())
        )


class GenericSchedulingSolution(
    SkillSolution[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, UnaryResource
    ],
    NonRenewableResourceSolution[Task, NonRenewableResource],
    PrecedenceSchedulingSolution[Task],
    TimelagSolution[Task],
    TimewindowSolution[Task],
    NoOverlapSolution[Task],
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

    def compute_cost(self) -> int:
        return sum(
            (
                self.problem.get_mode_cost(
                    task=task, mode=(mode := self.get_mode(task=task))
                )
                + sum(
                    self.problem.get_unary_resource_cost(
                        task=task, mode=mode, unary_resource=unary_resource
                    )
                    for unary_resource in self.get_task_allocation(task=task)
                )
            )
            for task in self.problem.tasks_list
        )
