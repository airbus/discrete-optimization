#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable, Container, Hashable, Iterable
from copy import deepcopy
from typing import Optional

import numpy as np
import wrapt

from discrete_optimization.generic_tasks_tools.calendar_resource import (
    convert_availability_intervals_to_calendar,
    convert_calendar_to_availability_intervals,
)
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    OBJECTIVE_DEFAULT_WEIGHTS,
    PENALTY_DEFAULT_WEIGHTS,
    Objective,
    Penalty,
    RawSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Solution,
    TypeObjective,
)
from discrete_optimization.generic_tools.encoding_register import EncodingRegister

# types for annotations
Skill = str
NonSkillCumulativeResource = str
NonRenewableResource = str
UnaryResource = str
Task = Hashable
CumulativeResource = NonSkillCumulativeResource | Skill
Resource = CumulativeResource | UnaryResource  # calendar resources
AnyResource = NonRenewableResource | Resource
UnaryAvailabilityIntervals = list[tuple[int, int]]  # start, end
AvailabilityIntervals = list[tuple[int, int, int]]  # start, end, value


class GenericSchedulingImplProblem(
    GenericSchedulingProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]
):
    """Generic implementation of a scheduling problem.

    It implements the abstract class `GenericSchedulingProblem`.

    """

    def __init__(
        self,
        horizon: int,
        durations_per_mode: dict[Task, dict[int, int]],
        resource_consumptions: Optional[
            dict[Task, dict[int, dict[CumulativeResource | NonRenewableResource, int]]]
        ] = None,
        successors: Optional[dict[Task, Iterable[Task]]] = None,
        unary_resources: Optional[set[UnaryResource]] = None,
        unary_resources_skills: Optional[dict[UnaryResource, dict[Skill, int]]] = None,
        unary_resources_availabilities: Optional[
            dict[UnaryResource, UnaryAvailabilityIntervals]
        ] = None,
        unary_resources_task_compatibility: Optional[
            dict[Task, set[UnaryResource]]
        ] = None,
        skills: Optional[set[Skill]] = None,
        non_skill_cumulative_resources: Optional[
            dict[CumulativeResource, int | AvailabilityIntervals]
        ] = None,
        non_renewable_resources: Optional[dict[NonRenewableResource, int]] = None,
        time_windows: Optional[
            dict[Task, tuple[int | None, int | None, int | None, int | None]]
        ] = None,
        start_to_start_min_time_lags: Optional[list[tuple[Task, Task, int]]] = None,
        start_to_end_min_time_lags: Optional[list[tuple[Task, Task, int]]] = None,
        end_to_start_min_time_lags: Optional[list[tuple[Task, Task, int]]] = None,
        end_to_end_min_time_lags: Optional[list[tuple[Task, Task, int]]] = None,
        no_overlap_sets: Optional[set[frozenset[Task]]] = None,
        forbidden_intervals: Optional[dict[Task, list[tuple[int, int]]]] = None,
        objective: Objective | Iterable[tuple[Objective, int]] = Objective.MAKESPAN,
        custom_evaluate_fn: Optional[
            Callable[[GenericSchedulingImplSolution], int]
        ] = None,
        objective_resource_weights: Optional[dict[AnyResource, int]] = None,
        mode_costs: Optional[dict[Task, dict[int, int]]] = None,
        unary_resource_costs: Optional[
            dict[Task, dict[int, dict[UnaryResource, int]]]
        ] = None,
        compute_time_penalty: bool = True,
    ):
        """

        Args:
            horizon: max allowed time to finish the tasks
            durations_per_mode: task -> mode -> duration. Tasks durations, mode by mode.
                This is used to know all available tasks, all available modes for a given task, and corresponding durations.
            resource_consumptions: task -> mode -> resource -> conso.
                Cumulative or non-renewable resource consumption, task by task, mode by mode. The resource can be a skill.
                Missing key => conso = 0
            successors: maps a task to its successors in the precedence graph.
                Each successor task must start after the given task ends.
                Default to no precedence constraints. Note that a consolidated version of it will
                be constructed using the time lags constraints.
            unary_resources: available unary resources.  Default to none.
            unary_resources_skills: skill values of each unary resource. Missing key => skill value = 0
            unary_resources_availabilities: availability of unary resources on the form of list of intervals (start, end)
                Missing key => always available.
            unary_resources_task_compatibility: maps a task to its compatible unary resources. Missing key => all unary resources allowed.
            skills: available skills
            non_skill_cumulative_resources: cumulative resources (excluding skills) availabilities.
                Format: either int => always available at the given max capacity,
                or list of intervals + capacity (start, end, value)
            non_renewable_resources: non-renewable resources max capacities
            time_windows: maps task to start_lb, end_lb, start_ub, end_ub s.t.
                start_lb <= start(task) <= start_ub and end_lb <= end(task) <= end_ub
                missing or none value means 0 (lb) or self.horizon (ub)
            start_to_start_min_time_lags: min time lags constraints between task starts.
                task1, task2, offset meaning start(task1) + offset <= start(task2)
                Note that using negative offset can model start-to-start max time lags.
            start_to_end_min_time_lags: min time lags constraints first task start and second task end.
                task1, task2, offset meaning start(task1) + offset <= end(task2)
                Note that using negative offset can model end-to-start max time lags.
            end_to_start_min_time_lags: min time lags constraints between first task end and second task start.
                task1, task2, offset meaning end(task1) + offset <= start(task2)
                Note that using negative offset can model start-to-end max time lags.
            end_to_end_min_time_lags: min time lags constraints between task ends.
                task1, task2, offset meaning end(task1) + offset <= end(task2)
                Note that using negative offset can model end-to-end max time lags.
            no_overlap_sets: a set of (set of tasks that should not overlap together)
            forbidden_intervals: maps task to forbidden intervals that cannot overlap with it. Missing key => no forbidden intervals.
            objective: objective for the problem. Default to minimization of makespan.
                Either an iterable of (objective, weight) so that the problem should *maximize* the aggregated objective
                resulting from weighted sum of objectives, or a single objective in which case we use the corresponding
                default weight from
                `discrete_optimization.generic_tasks_tools.generic_scheduling_utils.OBJECTIVE_DEFAULT_WEIGHTS`
                and maximize it. For instance the default weight for makespan is -1 so that it will
                actually minimize the makespan.
            custom_evaluate_fn: function used to evaluate the "custom" objective (to be maximized).
            objective_resource_weights: Weights to be used by the objective when summing used resources
                (`Objective.NB_RESOURCES_USED`) or resources levels (`Objective.RESOURCES_LEVELS`).
                Default to 1 for resources not mentioned.
            mode_costs: cost of choosing each mode. Missing key => cost = 0.
            unary_resource_costs: cost of allocating each unary resource. Missing key => cost = 0.
            compute_time_penalty: whether to include time penalties in evaluation

        """
        self.horizon = horizon
        self.durations_per_mode = durations_per_mode
        # default values
        if resource_consumptions is None:
            self.resource_consumptions: dict[
                Task, dict[int, dict[CumulativeResource | NonRenewableResource, int]]
            ] = {}
        else:
            self.resource_consumptions = resource_consumptions
        if successors is None:
            self.successors: dict[Task, Iterable[Task]] = {}
        else:
            self.successors = successors
        if unary_resources is None:
            self.unary_resources: set[UnaryResource] = set()
        else:
            self.unary_resources = unary_resources
        if unary_resources_skills is None:
            self.unary_resources_skills: dict[UnaryResource, dict[Skill, int]] = {}
        else:
            self.unary_resources_skills = unary_resources_skills
        if unary_resources_availabilities is None:
            self.unary_resources_availabilities: dict[
                UnaryResource, UnaryAvailabilityIntervals
            ] = {}
        else:
            self.unary_resources_availabilities = unary_resources_availabilities
        if unary_resources_task_compatibility is None:
            self.unary_resources_task_compatibility: dict[Task, set[UnaryResource]] = {}
        else:
            self.unary_resources_task_compatibility = unary_resources_task_compatibility
        if skills is None:
            self.skills: set[Skill] = set()
        else:
            self.skills = skills
        if non_skill_cumulative_resources is None:
            self.non_skill_cumulative_resources: dict[
                CumulativeResource, int | AvailabilityIntervals
            ] = {}
        else:
            self.non_skill_cumulative_resources = non_skill_cumulative_resources
        if non_renewable_resources is None:
            self.non_renewable_resources: dict[NonRenewableResource, int] = {}
        else:
            self.non_renewable_resources = non_renewable_resources
        if time_windows is None:
            self.time_windows: dict[
                Task, tuple[int | None, int | None, int | None, int | None]
            ] = {}
        else:
            self.time_windows = time_windows
        if start_to_start_min_time_lags is None:
            self.start_to_start_min_time_lags: list[tuple[Task, Task, int]] = []
        else:
            self.start_to_start_min_time_lags = start_to_start_min_time_lags
        if start_to_end_min_time_lags is None:
            self.start_to_end_min_time_lags: list[tuple[Task, Task, int]] = []
        else:
            self.start_to_end_min_time_lags = start_to_end_min_time_lags
        if end_to_start_min_time_lags is None:
            self.end_to_start_min_time_lags: list[tuple[Task, Task, int]] = []
        else:
            self.end_to_start_min_time_lags = end_to_start_min_time_lags
        if end_to_end_min_time_lags is None:
            self.end_to_end_min_time_lags: list[tuple[Task, Task, int]] = []
        else:
            self.end_to_end_min_time_lags = end_to_end_min_time_lags
        if no_overlap_sets is None:
            self.no_overlap_sets = set()
        else:
            self.no_overlap_sets = no_overlap_sets
        if forbidden_intervals is None:
            self.forbidden_intervals = {}
        else:
            self.forbidden_intervals = forbidden_intervals
        if isinstance(objective, Objective):
            self.weighted_objectives: tuple[tuple[Objective, int], ...] = (
                (objective, OBJECTIVE_DEFAULT_WEIGHTS[objective]),
            )
        else:
            self.weighted_objectives = tuple(objective)
        self.custom_evaluate_fn = custom_evaluate_fn
        if objective_resource_weights is None:
            self.objective_resource_weights: dict[AnyResource, int] = {}
        else:
            self.objective_resource_weights = objective_resource_weights
        if mode_costs is None:
            self.mode_costs = {}
        else:
            self.mode_costs = mode_costs
        if unary_resource_costs is None:
            self.unary_resource_costs = {}
        else:
            self.unary_resource_costs = unary_resource_costs
        self.compute_time_penalty = compute_time_penalty
        self.update_problem()

    def update_problem(self):
        """Method to call when some attributes of the problem are modified."""
        self._tasks_list = list(self.durations_per_mode)
        self._skills_list = list(self.skills)
        self._non_skill_cumulative_resources_list = list(
            self.non_skill_cumulative_resources
        )
        self._non_renewable_resources_list = list(self.non_renewable_resources)
        self._unary_resources_list = list(self.unary_resources)

        self.check_resources_lists()
        self.update_tasks_list()
        self.update_skills()
        self.update_resource_availabilities()
        self.update_task_bounds()
        self.update_time_lags()
        self.update_precedence_constraints()

        if (
            Objective.CUSTOM in {objective for objective, _ in self.weighted_objectives}
            and self.custom_evaluate_fn is None
        ):
            raise RuntimeError(
                "self.custom_evaluate_fn is not defined but custom objective used."
            )

    def update_resource_availabilities(self) -> None:
        self.get_resource_availabilities.cache_clear()
        super().update_resource_availabilities()

    def check_resources_lists(self) -> None:
        """Check duplicates in resources."""
        resources_list = (
            self.calendar_resources_list + self.non_renewable_resources_list
        )
        assert len(resources_list) == len(set(resources_list)), (
            "There are duplicates in resources list, "
            "potentially because calendar and non-renewable resources intersect."
        )

    @property
    def skills_list(self) -> list[Skill]:
        return self._skills_list

    @property
    def non_skill_cumulative_resources_list(self) -> list[Skill]:
        return self._non_skill_cumulative_resources_list

    def get_unary_resource_skill_value(
        self, unary_resource: UnaryResource, skill: Skill
    ) -> int:
        try:
            return self.unary_resources_skills[unary_resource][skill]
        except KeyError:
            return 0

    def is_compatible_task_unary_resource(
        self, task: Task, unary_resource: UnaryResource
    ) -> bool:
        compatible_unary_resources = self.unary_resources_task_compatibility.get(
            task, None
        )
        if compatible_unary_resources is None:
            return super().is_compatible_task_unary_resource(task, unary_resource)
        else:
            return unary_resource in compatible_unary_resources

    def get_cumulative_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> int:
        try:
            return self.resource_consumptions[task][mode][resource]
        except KeyError:
            return 0

    def get_no_overlap(self) -> set[frozenset[Task]]:
        return self.no_overlap_sets

    def get_forbidden_intervals(self, task: Task) -> list[tuple[int, int]]:
        return self.forbidden_intervals.get(task, [])

    @wrapt.lru_cache(maxsize=None)
    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        if resource in self.skills:
            return self.compute_skill_availabilities(resource)
        elif resource in self.unary_resources:
            try:
                return [
                    (start, end, 1)
                    for start, end in self.unary_resources_availabilities[resource]
                ]
            except KeyError:
                return [(0, self.horizon, 1)]
        elif resource in self.non_skill_cumulative_resources:
            intervals_or_max_capacity = self.non_skill_cumulative_resources[resource]
            if isinstance(intervals_or_max_capacity, int):
                return [(0, self.horizon, intervals_or_max_capacity)]
            else:
                return intervals_or_max_capacity
        else:
            raise ValueError(
                f"{resource} is neither an actual cumulative resource, nor a skill, nor a unary resource."
            )

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        return self.durations_per_mode[task][mode]

    @property
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        return self._non_renewable_resources_list

    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        return self.non_renewable_resources[resource]

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        try:
            return self.resource_consumptions[task][mode][resource]
        except KeyError:
            return 0

    def get_start_to_start_min_time_lags(self) -> list[tuple[Task, Task, int]]:
        return self.start_to_start_min_time_lags

    def get_end_to_start_min_time_lags(self) -> list[tuple[Task, Task, int]]:
        return self.end_to_start_min_time_lags

    def get_end_to_end_min_time_lags(self) -> list[tuple[Task, Task, int]]:
        return self.end_to_end_min_time_lags

    def get_start_to_end_min_time_lags(self) -> list[tuple[Task, Task, int]]:
        return self.start_to_end_min_time_lags

    def get_task_start_or_end_lower_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        try:
            start_lb, end_lb, start_ub, end_ub = self.time_windows[task]
            if start_or_end == StartOrEnd.START:
                lb = start_lb
            else:
                lb = end_lb
            if lb is not None:
                return lb
        except KeyError:
            pass
        return super().get_task_start_or_end_lower_bound(task, start_or_end)

    def get_task_start_or_end_upper_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        try:
            start_lb, end_lb, start_ub, end_ub = self.time_windows[task]
            if start_or_end == StartOrEnd.START:
                ub = start_ub
            else:
                ub = end_ub
            if ub is not None:
                return ub
        except KeyError:
            pass
        return super().get_task_start_or_end_upper_bound(task, start_or_end)

    def get_precedence_constraints(self) -> dict[Task, Iterable[Task]]:
        return self.successors

    def get_makespan_upper_bound(self) -> int:
        return self.horizon

    def get_task_modes(self, task: Task) -> set[int]:
        return set(self.durations_per_mode[task])

    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return self._unary_resources_list

    @property
    def tasks_list(self) -> list[Task]:
        return self._tasks_list

    def get_solution_type(self) -> type[Solution]:
        return GenericSchedulingImplSolution

    def get_attribute_register(self) -> EncodingRegister:
        raise NotImplementedError()

    def set_fixed_attributes(self, attribute_name: str, solution: Solution) -> None:
        raise NotImplementedError()

    def evaluate(self, variable: Solution) -> dict[str, float]:
        dict_eval = {
            objective.value: self.compute_subobjective(
                variable=variable, objective=objective
            )
            for objective, _ in self.weighted_objectives
        }
        if self.compute_time_penalty:
            penalty = Penalty.TIME
            dict_eval[penalty.value] = self.compute_penalty(
                variable=variable, penalty=penalty
            )
        return dict_eval

    def compute_subobjective(
        self,
        variable: GenericSchedulingSolution,
        objective: Objective,
        resource_weights: Optional[dict[AnyResource, int]] = None,
    ) -> int:
        if resource_weights is None:
            resource_weights = self.objective_resource_weights
        match objective:
            case Objective.CUSTOM:
                if self.custom_evaluate_fn is None:
                    raise RuntimeError(
                        "self.custom_evaluate_fn is not defined but custom objective used."
                    )
                assert isinstance(variable, GenericSchedulingImplSolution)
                return self.custom_evaluate_fn(variable)
            case _:
                return super().compute_subobjective(
                    variable=variable,
                    objective=objective,
                    resource_weights=resource_weights,
                )

    def get_objective_register(self) -> ObjectiveRegister:
        if len(self.weighted_objectives) == 1:
            handling = ObjectiveHandling.SINGLE
        else:
            handling = ObjectiveHandling.AGGREGATE
        dict_objective = {
            objective.value: ObjectiveDoc(
                type=TypeObjective.OBJECTIVE, default_weight=weight
            )
            for objective, weight in self.weighted_objectives
        }
        if self.compute_time_penalty:
            penalty = Penalty.TIME
            dict_objective[penalty.value] = ObjectiveDoc(
                type=TypeObjective.PENALTY,
                default_weight=PENALTY_DEFAULT_WEIGHTS[penalty],
            )
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=handling,
            dict_objective_to_doc=dict_objective,
        )

    def get_dummy_solution(self) -> Solution:
        raise NotImplementedError()

    def get_mode_cost(self, task: Task, mode: int) -> int:
        try:
            return self.mode_costs[task][mode]
        except KeyError:
            return super().get_mode_cost(task, mode)

    def get_unary_resource_cost(
        self, task: Task, mode: int, unary_resource: UnaryResource
    ) -> int:
        try:
            return self.unary_resource_costs[task][mode][unary_resource]
        except KeyError:
            return super().get_unary_resource_cost(task, mode, unary_resource)

    def create_subproblem_from_partial_solution(
        self, partial_solution: RawSolution[Task, UnaryResource, Skill]
    ) -> GenericSchedulingImplProblem:
        """Create a subproblem according to a partial solution.

        - Tasks already scheduled are removed from subproblem.
        - Add time windows constraints to model timelags/precedence constraints with removed tasks.
        - Update resource calendars with scheduled allocations
        - Update non-renewable resources max capacities
        - Transform no_overlap constraints into forbidden intervals constraints

        """
        scheduled_tasks = partial_solution.task_variables

        # restrict tasks list
        new_tasks_list = [
            task for task in self.tasks_list if task not in scheduled_tasks
        ]
        new_durations_per_mode = {
            task: self.durations_per_mode[task] for task in new_tasks_list
        }
        new_successors = {
            task: [
                next_task
                for next_task in next_tasks
                if next_task not in scheduled_tasks
            ]
            for task, next_tasks in self.successors.items()
            if task not in scheduled_tasks
        }
        new_start_to_start_min_time_lags = _restrict_timelags(
            self.start_to_start_min_time_lags, tasks_to_remove=scheduled_tasks
        )
        new_end_to_start_min_time_lags = _restrict_timelags(
            self.end_to_start_min_time_lags, tasks_to_remove=scheduled_tasks
        )
        new_start_to_end_min_time_lags = _restrict_timelags(
            self.start_to_end_min_time_lags, tasks_to_remove=scheduled_tasks
        )
        new_end_to_end_min_time_lags = _restrict_timelags(
            self.end_to_end_min_time_lags, tasks_to_remove=scheduled_tasks
        )

        # translate time lags/precedence constraints involving missing tasks into time windows constraints
        new_time_windows_nested: dict[tuple[Task, StartOrEnd, MinOrMax], set[int]] = {
            (task, start_or_end, min_or_max): {
                # at least the original bound
                self.get_task_bound(
                    task=task,
                    start_or_end=start_or_end,
                    min_or_max=min_or_max,
                )
            }
            for task in new_tasks_list
            for start_or_end in StartOrEnd
            for min_or_max in MinOrMax
        }
        for task1_start_or_end in StartOrEnd:
            for task2_start_or_end in StartOrEnd:
                for min_or_max in MinOrMax:
                    for task1, task2, offset in self.get_consolidated_time_lags(
                        task1_start_or_end=task1_start_or_end,
                        task2_start_or_end=task2_start_or_end,
                        min_or_max=min_or_max,
                    ):
                        if task1 in scheduled_tasks and task2 not in scheduled_tasks:
                            scheduled_task = task1
                            task = task2
                            scheduled_task_start_or_end = task1_start_or_end
                            task_start_or_end = task2_start_or_end
                            scheduled_offset = offset
                            task_min_or_max = min_or_max
                        elif task1 not in scheduled_tasks and task2 in scheduled_tasks:
                            scheduled_task = task2
                            task = task1
                            scheduled_task_start_or_end = task2_start_or_end
                            task_start_or_end = task1_start_or_end
                            scheduled_offset = -offset
                            task_min_or_max = ~min_or_max

                        else:
                            continue

                        bound_from_schedule = (
                            scheduled_tasks[scheduled_task].get_start_or_end(
                                scheduled_task_start_or_end
                            )
                            + scheduled_offset
                        )
                        new_time_windows_nested[
                            (task, task_start_or_end, task_min_or_max)
                        ].add(bound_from_schedule)
        new_time_windows = {
            task: (
                max(new_time_windows_nested[(task, StartOrEnd.START, MinOrMax.MIN)]),
                max(new_time_windows_nested[(task, StartOrEnd.END, MinOrMax.MIN)]),
                min(new_time_windows_nested[(task, StartOrEnd.START, MinOrMax.MAX)]),
                min(new_time_windows_nested[(task, StartOrEnd.END, MinOrMax.MAX)]),
            )
            for task in new_tasks_list
        }

        # Update non-renewable resources max capacities
        new_non_renewable_resources = {
            resource: old_max_capacity
            - sum(
                self.get_non_renewable_resource_consumption(
                    resource=resource, task=task, mode=task_variable.mode
                )
                for task, task_variable in scheduled_tasks.items()
            )
            for resource, old_max_capacity in self.non_renewable_resources.items()
        }

        # Update resources availabilities
        calendar_resources = (
            self.unary_resources_list + self.non_skill_cumulative_resources_list
        )
        resources_calendars = {
            resource: np.array(
                convert_availability_intervals_to_calendar(
                    intervals=self.get_resource_availabilities(resource=resource),
                    horizon=self.horizon,
                ),
                dtype=int,
            )
            for resource in calendar_resources
        }
        for task, task_variable in scheduled_tasks.items():
            start = task_variable.start
            end = task_variable.end
            for resource in self.unary_resources_list:
                if resource in task_variable.allocated:
                    resources_calendars[resource][start:end] -= 1
            for resource in self.non_skill_cumulative_resources_list:
                conso = self.get_cumulative_resource_consumption(
                    resource=resource, task=task, mode=task_variable.mode
                )
                resources_calendars[resource][start:end] -= conso
        new_non_skill_cumulative_resources = {
            resource: convert_calendar_to_availability_intervals(
                calendar=resources_calendars[resource],
                horizon=self.horizon,
            )
            for resource in self.non_skill_cumulative_resources_list
        }
        new_unary_resources_availabilities = {
            resource: [
                (start, end)
                for start, end, value in convert_calendar_to_availability_intervals(
                    calendar=resources_calendars[resource],
                    horizon=self.horizon,
                )
                if value > 0
            ]
            for resource in self.unary_resources_list
        }

        # Transform no_overlap constraints into forbidden intervals constraints
        new_no_overlap_sets: set[frozenset[Task]] = set()
        new_forbidden_intervals: dict[Task, list[tuple[int, int]]] = {
            task: list(self.get_forbidden_intervals(task=task))
            for task in self.tasks_list
        }
        for not_overlapping_tasks in self.get_no_overlap():
            if not_overlapping_tasks.isdisjoint(scheduled_tasks):
                # no removed tasks in it => ok
                new_no_overlap_sets.add(not_overlapping_tasks)
            else:
                # remove scheduled tasks and replace them by forbidden intervals
                new_not_overlapping_tasks = not_overlapping_tasks.difference(
                    scheduled_tasks
                )
                scheduled_tasks_in_not_overlapping_tasks = (
                    not_overlapping_tasks.difference(new_not_overlapping_tasks)
                )
                new_no_overlap_sets.add(new_not_overlapping_tasks)
                for task in new_not_overlapping_tasks:
                    new_forbidden_intervals[task].extend(
                        [
                            (
                                scheduled_tasks[scheduled_task].start,
                                scheduled_tasks[scheduled_task].end,
                            )
                            for scheduled_task in scheduled_tasks_in_not_overlapping_tasks
                        ]
                    )

        return GenericSchedulingImplProblem(
            horizon=self.horizon,
            durations_per_mode=new_durations_per_mode,
            resource_consumptions=self.resource_consumptions,
            successors=new_successors,
            unary_resources=self.unary_resources,
            unary_resources_skills=self.unary_resources_skills,
            unary_resources_availabilities=new_unary_resources_availabilities,
            unary_resources_task_compatibility=self.unary_resources_task_compatibility,
            skills=self.skills,
            non_skill_cumulative_resources=new_non_skill_cumulative_resources,
            non_renewable_resources=new_non_renewable_resources,
            time_windows=new_time_windows,
            start_to_start_min_time_lags=new_start_to_start_min_time_lags,
            start_to_end_min_time_lags=new_start_to_end_min_time_lags,
            end_to_start_min_time_lags=new_end_to_start_min_time_lags,
            end_to_end_min_time_lags=new_end_to_end_min_time_lags,
            no_overlap_sets=new_no_overlap_sets,
            forbidden_intervals=new_forbidden_intervals,
            objective=self.weighted_objectives,
            custom_evaluate_fn=self.custom_evaluate_fn,
            objective_resource_weights=self.objective_resource_weights,
            mode_costs=self.mode_costs,
            unary_resource_costs=self.unary_resource_costs,
            compute_time_penalty=self.compute_time_penalty,
        )


def _restrict_timelags(
    timelags: list[tuple[Task, Task, int]], tasks_to_remove: Container[Task]
) -> list[tuple[Task, Task, int]]:
    return [
        (task1, task2, offset)
        for task1, task2, offset in timelags
        if (task1 not in tasks_to_remove) and (task2 not in tasks_to_remove)
    ]


class GenericSchedulingImplSolution(
    GenericSchedulingSolution[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]
):
    """Generic implementation of a solution to a scheduling problem.

    It implements the abstract class `GenericSchedulingSolution`.

    """

    problem: GenericSchedulingImplProblem

    def __init__(
        self,
        problem: GenericSchedulingImplProblem,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
    ):
        super().__init__(problem)
        self.raw_sol = raw_sol

    def is_skill_used(
        self, task: Task, unary_resource: UnaryResource, skill: Skill
    ) -> bool:
        try:
            return skill in self.raw_sol.task_variables[task].allocated[unary_resource]
        except KeyError:
            return False

    def get_end_time(self, task: Task) -> int:
        return self.raw_sol.task_variables[task].end

    def get_start_time(self, task: Task) -> int:
        return self.raw_sol.task_variables[task].start

    def get_mode(self, task: Task) -> int:
        return self.raw_sol.task_variables[task].mode

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        return unary_resource in self.raw_sol.task_variables[task].allocated

    def get_task_allocation(self, task: Task) -> set[UnaryResource]:
        return set(self.raw_sol.task_variables[task].allocated)

    def copy(self) -> Solution:
        return GenericSchedulingImplSolution(
            problem=self.problem, raw_sol=deepcopy(self.raw_sol)
        )

    def lazy_copy(self) -> Solution:
        return GenericSchedulingImplSolution(problem=self.problem, raw_sol=self.raw_sol)
