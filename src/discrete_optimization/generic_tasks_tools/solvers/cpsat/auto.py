#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, Union

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    Domain,
    IntervalVar,
    IntVar,
    LinearExprT,
)

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    CumulativeResource,
    GenericSchedulingSolution,
    Resource,
)
from discrete_optimization.generic_tasks_tools.multimode import SinglemodeProblem
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    Task,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.generic_scheduling import (
    GenericSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode_scheduling import (
    SinglemodeSchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin

logger = logging.getLogger(__name__)


class Objective(Enum):
    """Objective to be used by the solver."""

    MAKESPAN = "makespan"
    """Global makespan of the schedule, to minimize."""
    NB_TASKS_DONE = "nb_tasks_done"
    """Number of tasks with at least one resource allocated, to maximize."""
    NB_UNARY_RESOURCES_USED = "nb_unary_resources_used"
    """Number of allocated unary resources, to minimize."""
    NB_RESOURCES_USED = "nb_resources_used"
    """Weighted sum of resources used, to minimize.

    Include non-renewable, cumulative, and unary resources.
    The weigths are to be defined in `solver.objective_resource_weights`.

    """
    RESOURCES_CONSUMPTION = "resources_consumption"
    """Weighted sum of resources consumptions, to minimize.

    Include non-renewable, cumulative, and unary resources.
    The weigths are to be defined in `solver.objective_resource_weights`.

    """
    CUSTOM = "custom_objective"


@dataclass
class TaskVariable(Generic[UnaryResource]):
    """Task characteristics found by a cpsat solution."""

    start: int  # start time of the task
    end: int  # end time of the task
    mode: int  # chosen mode for the task
    allocated: list[UnaryResource] = field(
        default_factory=list
    )  # resources allocated to the task
    info: dict[str, Any] = field(
        default_factory=dict
    )  # additional information if needed


@dataclass
class TemporarySolution(Generic[Task, UnaryResource]):
    """Temporary format for a cpsat solution."""

    task_variables: dict[Task, TaskVariable[UnaryResource]]
    metadata: dict[str, Any] = field(default_factory=dict)


class GenericSchedulingAutoCpSatSolver(
    GenericSchedulingCpSatSolver[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ],
    WarmstartMixin,
):
    """Generic cpsat solver for scheduling problems (with or without allocation).

    The needed variables are automatically created, with common constraints (precedence, resource capacity).
    The objective is set by default to global makespan but can be changed by modifying `solver.default_objective` value.

    This solver class needs still to be derived to create a solution from the proper class.
    You will need at least to implement the conversion from the task variables to the actual solution object.

    If custom constraints are needed, override `init_model()`.
    If a custom objective is needed, set `solver.default_objective` to `Objective.CUSTOM` so that no objective is set
    by default and override `init_model()` to define your objective.

    """

    # objective settings
    objective = Objective.MAKESPAN  # Objective set by `init_model()`
    objective_resource_weights: Optional[
        dict[Union[CumulativeResource, UnaryResource, NonRenewableResource], int]
    ] = None
    """Weights to be used by the objective when summing used resources or resources consumption.

    This is the case if `objective` is set to `Objective.NB_RESOURCES_USED` or  `Objective.RESOURCES_CONSUMPTION`.
    Default to 1 for resources not mentioned.

    Hypothesis: cumulative, unary, and non-renewable resources have different values.
    (It could happen that non-renewable resources and renewable lists intersect which whould be a problem
    for weights definition).

    """

    # allocation settings
    exactly_one_unary_resource_per_task = (
        False  # if True, enforce exactly one resource allocated to each task
    )

    # multimode settings
    duplicate_start_var_per_mode = (
        False  # if True, add a start variable for each task mode
    )

    # cpsat variables
    start_or_end_variables: dict[tuple[Task, StartOrEnd], LinearExprT]
    duration_variables: dict[Task, LinearExprT]
    task_interval_variables = dict[Task, IntervalVar]
    modes_is_present: dict[Task, dict[int, LinearExprT]]
    modes_intervals: dict[Task, dict[int, IntervalVar]]
    modes_start_variables: dict[Task, dict[int, LinearExprT]]
    allocation_is_present: dict[Task, dict[UnaryResource, LinearExprT]]
    allocation_intervals: dict[Task, dict[UnaryResource, IntervalVar]]
    all_used_variables: dict[
        Union[NonRenewableResource, UnaryResource, CumulativeResource], IntVar
    ]
    """Variables tracking whether a (unary, cumulative, or non-renewable) resource has been used at least once."""
    all_used_variables_created = False
    """Flag telling whether 'all_used_variables' have been created"""
    resource_consumption_variables: dict[
        Union[NonRenewableResource, UnaryResource, CumulativeResource], LinearExprT
    ]
    """Variables tracking total consumption of each (unary, cumulative, or non-renewable) resource."""
    resource_consumption_variables_created = False
    """Flag telling whether 'resource_consumption_variables' have been created"""

    @property
    def needs_duration_variables(self) -> bool:
        """Whether the task duration variables are needed by the model.

        Default implementation, returns True only if the problem is an allocation one (at least one unary resource).
        If additional custom constraints require them, override it.

        """
        return len(self.problem.unary_resources_list) > 0

    @property
    def needs_task_interval(self) -> bool:
        """Whether the task interval variables are needed by the model.

        By default, these variables are only constraints on durations variables and need not to be stored.
        If additional custom constraints require them, override this property.

        """
        return False

    def include_constraint_on_cumulative_resource(
        self, resource: CumulativeResource
    ) -> bool:
        """Whether the cp model should take into account the constraint on the given cumulative resource.

        Some problems define "redundant" cumulative resources that are computed from others.
        If you want to avoid adding redundant constraints in your model, please override this method.

        Args:
            resource:

        Returns:

        """
        return True

    def get_task_start_or_end_lower_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        """Get a lower bound on start or end of a given task.

        Default implementation: calls self.problem.get_task_start_or_end_lower_bound()

        Args:
            task:
            start_or_end:

        Returns:

        """
        return self.problem.get_task_start_or_end_lower_bound(
            task=task, start_or_end=start_or_end
        )

    def get_task_start_or_end_upper_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        """Get an upper bound on start or end of a given task.

        Default implementation: takes best of
        - self.problem.get_task_start_or_end_upper_bound()
        - self.get_makespan_upper_bound()

        Args:
            task:
            start_or_end:

        Returns:

        """
        return min(
            self.problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=start_or_end
            ),
            self.get_makespan_upper_bound(),
        )

    def init_model(self, **kwargs: Any) -> None:
        """Init cp model and reset stored variables if any."""
        super().init_model(**kwargs)

        self._reset_variables()
        self._create_variables()
        self._add_constraints()
        self._set_objective()

    def _reset_variables(self):
        """Forget about previous variables."""
        self.start_or_end_variables = {}
        self.duration_variables = {}
        self.task_interval_variables = {}
        self.modes_is_present = {}
        self.modes_intervals = {}
        self.modes_start_variables = {}
        self.allocation_is_present = {}
        self.allocation_intervals = {}

        self.all_used_variables_created = False
        self.all_used_variables = {}
        self.resource_consumption_variables_created = False
        self.resource_consumption_variables = {}

    def _create_variables(self):
        self._create_start_or_end_variables()
        if self.needs_duration_variables or self.needs_task_interval:
            self._create_task_duration_and_interval_variables()
        self._create_mode_variables()
        self._create_allocation_variables()

    def _create_start_or_end_variables(self):
        for task in self.problem.tasks_list:
            for start_or_end in StartOrEnd:
                self.start_or_end_variables[task, start_or_end] = (
                    self.cp_model.new_int_var(
                        lb=self.get_task_start_or_end_lower_bound(
                            task=task, start_or_end=start_or_end
                        ),
                        ub=self.get_task_start_or_end_upper_bound(
                            task=task, start_or_end=start_or_end
                        ),
                        name=f"{start_or_end.value}_{task}",
                    )
                )

    def _create_task_duration_and_interval_variables(self):
        """Create task duration variables.

        Also add the interval constraint that tells duration = end - start

        """
        for task in self.problem.tasks_list:
            possible_durations = [
                self.problem.get_task_mode_duration(task=task, mode=mode)
                for mode in self.problem.get_task_modes(task)
            ]
            if len(possible_durations) == 1:
                # single mode: fixed duration
                self.duration_variables[task] = possible_durations[0]
                if self.needs_task_interval:
                    # interval var required
                    self.task_interval_variables[task] = (
                        self.cp_model.new_fixed_size_interval_var(
                            start=self.start_or_end_variables[task, StartOrEnd.START],
                            size=self.duration_variables[task],
                            name=f"interval_{task}",
                        )
                    )
            else:
                # multi mode
                self.duration_variables[task] = self.cp_model.new_int_var_from_domain(
                    domain=Domain.from_values(possible_durations),
                    name=f"duration_{task}",
                )
                # interval constraint
                task_interval = self.cp_model.new_interval_var(
                    start=self.start_or_end_variables[task, StartOrEnd.START],
                    size=self.duration_variables[task],
                    end=self.start_or_end_variables[task, StartOrEnd.END],
                    name=f"interval_{task}",
                )
                if self.needs_task_interval:
                    # interval var required
                    self.task_interval_variables[task] = task_interval

    def _create_mode_variables(self):
        for task in self.problem.tasks_list:
            self.modes_is_present[task] = {}
            self.modes_intervals[task] = {}
            self.modes_start_variables[task] = {}
            modes = self.problem.get_task_modes(task=task)
            if len(modes) == 1:
                # single mode (at least for this very task)
                mode = next(iter(modes))
                self.modes_is_present[task][mode] = 1
                # create the interval var with start and end => constraint on end - start
                self.modes_intervals[task][mode] = self.cp_model.new_interval_var(
                    start=self.start_or_end_variables[task, StartOrEnd.START],
                    size=self.problem.get_task_mode_duration(task=task, mode=mode),
                    end=self.start_or_end_variables[task, StartOrEnd.END],
                    name=f"interval_mode_{task}_{mode}",
                )
                if self.duplicate_start_var_per_mode:
                    self.modes_start_variables[task][mode] = (
                        self.start_or_end_variables[task, StartOrEnd.START]
                    )
            else:
                for mode in modes:
                    # multi mode
                    is_present_mode = self.cp_model.new_bool_var(
                        name=f"is_present_mode_{task}_{mode}"
                    )
                    self.modes_is_present[task][mode] = is_present_mode
                    start = self.start_or_end_variables[task, StartOrEnd.START]
                    end = self.start_or_end_variables[task, StartOrEnd.END]
                    duration_mode = self.problem.get_task_mode_duration(
                        task=task, mode=mode
                    )
                    if self.duplicate_start_var_per_mode:
                        # create new start variable per mode to model the interval
                        start_mode = self.cp_model.new_int_var(
                            lb=self.get_task_start_or_end_lower_bound(
                                task=task, start_or_end=StartOrEnd.START
                            ),
                            ub=self.get_task_start_or_end_upper_bound(
                                task=task, start_or_end=StartOrEnd.START
                            ),
                            name=f"start_{task}_{mode}",
                        )
                        self.modes_start_variables[task][mode] = start_mode
                        self.modes_intervals[task][mode] = (
                            self.cp_model.new_optional_fixed_size_interval_var(
                                start=start_mode,
                                size=duration_mode,
                                is_present=is_present_mode,
                                name=f"interval_mode_{task}_{mode}",
                            )
                        )
                        self.cp_model.add(start_mode == start).only_enforce_if(
                            is_present_mode
                        )
                        self.cp_model.add(
                            start_mode + duration_mode == end
                        ).only_enforce_if(is_present_mode)
                    else:
                        self.modes_intervals[task][mode] = (
                            self.cp_model.new_optional_interval_var(
                                start=start,
                                size=duration_mode,
                                end=end,
                                is_present=is_present_mode,
                                name=f"interval_mode_{task}_{mode}",
                            )
                        )
                self.cp_model.add_exactly_one(
                    self.modes_is_present[task][mode] for mode in modes
                )

    def _create_allocation_variables(self):
        for task in self.problem.tasks_list:
            self.allocation_is_present[task] = {}
            self.allocation_intervals[task] = {}
            for unary_resource in self.problem.unary_resources_list:
                if self.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                ):
                    is_allocated = self.cp_model.new_bool_var(
                        name=f"is_allocated_{task}_{unary_resource}"
                    )
                    self.allocation_is_present[task][unary_resource] = is_allocated
                    self.allocation_intervals[task][unary_resource] = (
                        self.cp_model.new_optional_interval_var(
                            start=self.start_or_end_variables[task, StartOrEnd.START],
                            size=self.duration_variables[task],
                            end=self.start_or_end_variables[task, StartOrEnd.END],
                            is_present=is_allocated,
                            name=f"interval_allocated_{task}_{unary_resource}",
                        )
                    )

    def _create_all_used_variables(self):
        if not self.all_used_variables_created:
            self.check_resources_lists()
            self.create_used_variables()
            self.all_used_variables = {}
            for resource in self.problem.unary_resources_list:
                self.all_used_variables[resource] = self.used_variables[resource]
            for resource in self.problem.cumulative_resources_list:

                def conso_fn(task: Task, mode: int) -> int:
                    return self.problem.get_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )

                self.all_used_variables[resource] = (
                    self._create_mode_resource_used_variable(
                        resource=resource, conso_fn=conso_fn
                    )
                )
            for resource in self.problem.non_renewable_resources_list:

                def conso_fn(task: Task, mode: int) -> int:
                    return self.problem.get_non_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )

                self.all_used_variables[resource] = (
                    self._create_mode_resource_used_variable(
                        resource=resource, conso_fn=conso_fn
                    )
                )
            self.all_used_variables_created = True

    def _create_mode_resource_used_variable(
        self,
        resource: Union[Resource, NonRenewableResource],
        conso_fn: Callable[[Task, int], int],
    ) -> IntVar:
        used = self.cp_model.new_bool_var(f"used_{resource}")
        list_is_present_variables = [
            self.get_task_mode_is_present_variable(task=task, mode=mode)
            for task in self.problem.tasks_list
            for mode in self.problem.get_task_modes(task=task)
            if conso_fn(task, mode) > 0
        ]
        if len(list_is_present_variables) > 0:
            self.cp_model.add_max_equality(used, list_is_present_variables)
        else:
            self.cp_model.add(used == 0)
        return used

    def _create_resource_consumption_variables(self):
        if not self.resource_consumption_variables_created:
            self.check_resources_lists()
            self.create_used_variables()
            # allocated unary resources
            for resource in self.problem.unary_resources_list:
                self.resource_consumption_variables[resource] = self.used_variables[
                    resource
                ]
            # cumulative resources
            for resource in self.problem.cumulative_resources_list:
                max_capacity = self.problem.get_resource_max_capacity(resource)
                conso_var = self.cp_model.new_int_var(
                    lb=0, ub=max_capacity, name=f"conso_{resource}"
                )
                if max_capacity > 1:
                    # cumulative constraint on the new "conso" variable
                    mode_intervals_consumptions_is_present = [
                        (
                            self.get_task_mode_interval(task=task, mode=mode),
                            conso,
                            self.get_task_mode_is_present_variable(
                                task=task, mode=mode
                            ),
                        )
                        for task in self.problem.tasks_list
                        for mode in self.problem.get_task_modes(task=task)
                        if (
                            conso := self.problem.get_renewable_resource_consumption(
                                resource=resource, task=task, mode=mode
                            )
                        )
                        > 0
                    ]
                    intervals = [
                        interval
                        for interval, conso, is_present in mode_intervals_consumptions_is_present
                    ]
                    demands = [
                        conso
                        for interval, conso, is_present in mode_intervals_consumptions_is_present
                    ]
                    if len(intervals) > 0:
                        self.cp_model.add_cumulative(
                            intervals=intervals,
                            demands=demands,
                            capacity=conso_var,
                        )
                        for (
                            _,
                            conso,
                            is_present,
                        ) in mode_intervals_consumptions_is_present:
                            self.cp_model.add(conso_var >= conso * is_present)
                    else:
                        conso_var = 0
                else:
                    # disjunctive resource, no need to use the interval variables
                    # (no overlap constraint already handled by `create_renewable_resources_constraint()`
                    list_is_present_variables = [
                        self.get_task_mode_is_present_variable(task=task, mode=mode)
                        for task in self.problem.tasks_list
                        for mode in self.problem.get_task_modes(task=task)
                        if self.problem.get_renewable_resource_consumption(
                            resource=resource, task=task, mode=mode
                        )
                        > 0
                    ]
                    if len(list_is_present_variables) > 0:
                        self.cp_model.add_max_equality(
                            conso_var, list_is_present_variables
                        )
                    else:
                        conso_var = 0

                self.resource_consumption_variables[resource] = conso_var
            # non-renewable resources
            for resource in self.problem.non_renewable_resources_list:
                self.resource_consumption_variables[resource] = sum(
                    conso * self.get_task_mode_is_present_variable(task=task, mode=mode)
                    for task in self.problem.tasks_list
                    for mode in self.problem.get_task_modes(task=task)
                    if (
                        conso := self.problem.get_non_renewable_resource_consumption(
                            resource=resource, task=task, mode=mode
                        )
                    )
                    > 0
                )

            self.resource_consumption_variables_created = True

    def check_resources_lists(self):
        resources_list = (
            self.problem.renewable_resources_list
            + self.problem.non_renewable_resources_list
        )
        assert len(resources_list) == len(set(resources_list)), (
            "There are duplicates in resources list, "
            "potentially because renewable and non-renewable resources intersect."
        )

    def get_nb_resources_used_variable(self) -> Any:
        """Get cpsat variable tracking number of resources used at least in one task.

        If necessary, intermediate variables tracking is a specific resource is used are created.

        """
        weights = self.objective_resource_weights
        if weights is None:
            weights = {}
        self._create_all_used_variables()
        return sum(
            weights.get(resource, 1) * used
            for resource, used in self.all_used_variables.items()
        )

    def get_aggregated_resources_consumptions_variable(self) -> Any:
        """Get cpsat variable aggregating consumptions of each resource."""
        weights = self.objective_resource_weights
        if weights is None:
            weights = {}
        self._create_resource_consumption_variables()
        return sum(
            weights.get(resource, 1) * conso
            for resource, conso in self.resource_consumption_variables.items()
        )

    def _add_constraints(self) -> None:
        # non-renewable resources capacity
        for resource in self.problem.non_renewable_resources_list:
            self.create_non_renewable_resources_constraint(resource=resource)
        # cumulative + unary resources calendar
        for resource in self.problem.renewable_resources_list:
            if not self.problem.is_cumulative_resource(
                resource
            ) or self.include_constraint_on_cumulative_resource(resource=resource):
                self.create_renewable_resources_constraint(resource=resource)
        # precedence
        self.create_precedence_constraints()
        # at most or exactly one resource allocated per task?
        self._add_unary_resources_per_task_constraints()

    def _add_unary_resources_per_task_constraints(self) -> None:
        if self.exactly_one_unary_resource_per_task:
            if not self.at_most_one_unary_resource_per_task:
                logger.warning(
                    "`at_most_one_unary_resource_per_task` False `exactly_one_unary_resource_per_task` set to True. "
                    "`exactly_one_unary_resource_per_task` will take the precedence."
                )

            for is_allocated_task_vars_mapping in self.allocation_is_present.values():
                self.cp_model.add_exactly_one(is_allocated_task_vars_mapping.values())
                # avoid adding at most constraint when creating done variables
                self.at_most_one_unary_resource_per_task_constraints_added = True
        elif self.at_most_one_unary_resource_per_task:
            self.add_at_most_one_unary_resource_per_task_constraints()

    def _set_objective(self) -> None:
        objective = None
        match self.objective:
            case Objective.MAKESPAN:
                objective = self.get_global_makespan_variable()
            case Objective.NB_TASKS_DONE:
                objective = -self.get_nb_tasks_done_variable()
            case Objective.NB_UNARY_RESOURCES_USED:
                objective = self.get_nb_unary_resources_used_variable()
            case Objective.NB_RESOURCES_USED:
                objective = self.get_nb_resources_used_variable()
            case Objective.RESOURCES_CONSUMPTION:
                objective = self.get_aggregated_resources_consumptions_variable()
            case Objective.CUSTOM:
                # do not define it here. User will do it in overridden `init_model()`.
                ...
            case _:
                raise NotImplementedError()
        if objective is not None:
            self.cp_model.minimize(objective)

    def get_task_unary_resource_interval(
        self, task: Task, unary_resource: UnaryResource
    ) -> IntervalVar:
        return self.allocation_intervals[task][unary_resource]

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        """Return a 0-1 variable/expression telling if the unary_resource is used for the task.

        NB: sometimes the given resource is never to be used by a task and the variable has not been created.
        The convention is to return 0 in that case.

        """
        try:
            return self.allocation_is_present[task][unary_resource]
        except KeyError:
            return 0

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        return self.start_or_end_variables[task, start_or_end]

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        return self.modes_is_present[task][mode]

    def get_task_mode_interval(self, task: Task, mode: int) -> IntervalVar:
        """Get the interval variable corresponding to given task and mode."""
        return self.modes_intervals[task][mode]

    def retrieve_tasks_variables(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> TemporarySolution[Task, UnaryResource]:
        """Construct each task variable from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.Value(VARIABLE_NAME)`.

        This method is called in `self.retrieve_solution()` before `self.convert_task_variables_to_solution()`.
        Override it if you need additional information to be stored
        (either in res.metadata or res.task_variables[task].info).

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the task variables for the intermediate solution

        """
        task_variables = {}
        for task in self.problem.tasks_list:
            start = cpsolvercb.Value(
                self.start_or_end_variables[task, StartOrEnd.START]
            )
            end = cpsolvercb.Value(self.start_or_end_variables[task, StartOrEnd.END])
            modes = self.problem.get_task_modes(task)
            if len(modes) == 1:
                mode = next(iter(modes))
            else:
                for mode in modes:
                    if cpsolvercb.Value(self.modes_is_present[task][mode]):
                        break
            allocated = [
                unary_resource
                for unary_resource, is_allocated_var in self.allocation_is_present[
                    task
                ].items()
                if cpsolvercb.Value(is_allocated_var)
            ]
            task_variables[task] = TaskVariable(
                start=start, end=end, mode=mode, allocated=allocated
            )
        return TemporarySolution(task_variables=task_variables)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        # construct generic tasks variables
        temp_sol = self.retrieve_tasks_variables(cpsolvercb=cpsolvercb)
        # convert to specific solution type
        return self.convert_task_variables_to_solution(temp_sol=temp_sol)

    def set_warm_start(self, solution: Solution) -> None:
        solution: GenericSchedulingSolution[
            Task, UnaryResource, CumulativeResource, NonRenewableResource
        ]
        # warm start cp_model
        self.cp_model.clear_hints()
        for task in self.problem.tasks_list:
            self.cp_model.add_hint(
                self.start_or_end_variables[task, StartOrEnd.START],
                solution.get_start_time(task),
            )
            self.cp_model.add_hint(
                self.start_or_end_variables[task, StartOrEnd.END],
                solution.get_end_time(task),
            )
            modes = self.problem.get_task_modes(task)
            if len(modes) > 1:
                hinted_mode = solution.get_mode(task)
                for mode in modes:
                    self.cp_model.add_hint(
                        self.modes_is_present[task][mode], mode == hinted_mode
                    )
                if self.needs_duration_variables:
                    self.cp_model.add_hint(
                        self.duration_variables[task],
                        self.problem.get_task_mode_duration(
                            task=task, mode=hinted_mode
                        ),
                    )
            for unary_resource, is_allocated_var in self.allocation_is_present[
                task
            ].items():
                self.cp_model.add_hint(
                    is_allocated_var,
                    solution.is_allocated(task=task, unary_resource=unary_resource),
                )

    @abstractmethod
    def convert_task_variables_to_solution(
        self, temp_sol: TemporarySolution[Task, UnaryResource]
    ) -> GenericSchedulingSolution[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ]:
        """Convert solution from autosolver format into do format.

        To be used in `self.retrieve_solution()`.

        Args:
            temp_sol:

        Returns:

        """
        ...


class SinglemodeGenericSchedulingAutoCpSatSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ],
    SinglemodeSchedulingCpSatSolver[Task],
):
    """Subclass of GenericSchedulingAutoCpSatSolver for single mode problems.

    Give access to task intervals without dealing with modes.

    """

    problem: SinglemodeProblem[Task]

    def get_task_interval(self, task: Task) -> IntervalVar:
        """Task interval with fixed duration, single mode."""
        return self.get_task_mode_interval(task=task, mode=self.problem.default_mode)
