#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from copy import deepcopy
from typing import Optional

import wrapt

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
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
        objective: Objective | Iterable[tuple[Objective, int]] = Objective.MAKESPAN,
        custom_evaluate_fn: Optional[
            Callable[[GenericSchedulingImplSolution], int]
        ] = None,
        objective_resource_weights: Optional[dict[AnyResource, int]] = None,
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
            unary_resources_skills: skill values of each unary resource. Mssing key => skill value = 0
            unary_resources_availabilities: availability of unary resources on the form of list of intervals (start, end)
                Missing key => always available.
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

    def get_cumulative_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> int:
        try:
            return self.resource_consumptions[task][mode][resource]
        except KeyError:
            return 0

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

    def satisfy(self, variable: Solution) -> bool:
        assert isinstance(variable, GenericSchedulingImplSolution)
        return self.satisfy_partial(variable=variable)

    def satisfy_partial(
        self,
        variable: GenericSchedulingImplSolution,
        duration: bool = True,
        calendar: bool = True,
        non_renewable_capacity: bool = True,
        precedence: bool = True,
        skill: bool = True,
        allocation: bool = True,
        time_lags: bool = True,
        time_windows: bool = True,
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
        )

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


class GenericSchedulingImplSolution(
    GenericSchedulingSolution[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]
):
    """Generic implementation of a solution to a scheduling problem.

    It implements the abstract class `GenericSchedulingSolution`.

    """

    problem: GenericSchedulingImplProblem

    def __init__(self, problem: GenericSchedulingImplProblem, raw_sol: RawSolution):
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

    def copy(self) -> Solution:
        return GenericSchedulingImplSolution(
            problem=self.problem, raw_sol=deepcopy(self.raw_sol)
        )

    def lazy_copy(self) -> Solution:
        return GenericSchedulingImplSolution(problem=self.problem, raw_sol=self.raw_sol)
