#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional

import networkx as nx
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
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.cpm import Cpm
from discrete_optimization.generic_tasks_tools.solvers.cpsat.generic_scheduling import (
    GenericSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode_scheduling import (
    SinglemodeSchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)

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
class TaskVariable(Generic[UnaryResource, Skill]):
    """Task characteristics found by a cpsat solution."""

    start: int  # start time of the task
    end: int  # end time of the task
    mode: int  # chosen mode for the task
    allocated: dict[UnaryResource, set[Skill]] = field(
        default_factory=dict
    )  # resources allocated to the task
    info: dict[str, Any] = field(
        default_factory=dict
    )  # additional information if needed


@dataclass
class TemporarySolution(Generic[Task, UnaryResource, Skill]):
    """Temporary format for a cpsat solution."""

    task_variables: dict[Task, TaskVariable[UnaryResource, Skill]]
    metadata: dict[str, Any] = field(default_factory=dict)


AnyResource = NonRenewableResource | UnaryResource | Skill | NonSkillCumulativeResource


class GenericSchedulingAutoCpSatSolver(
    GenericSchedulingCpSatSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
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

    hyperparameters = [
        CategoricalHyperparameter(
            name="avoid_interval_optional", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="add_redundant_skill_cumulative_constraints",
            choices=[True, False],
            default=False,
        ),
        CategoricalHyperparameter(
            name="use_cpm_for_task_bounds", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="duplicate_start_var_per_mode",
            choices=[True, False],
            default=False,
            depends_on=("avoid_interval_optional", [False]),
        ),
        CategoricalHyperparameter(
            name="use_energy_constraints", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="keep_only_most_nested_energy_constraints",
            choices=[True, False],
            default=True,
            depends_on=("use_energy_constraints", [True]),
        ),
    ]

    # objective settings
    objective = Objective.MAKESPAN  # Objective set by `init_model()`
    objective_resource_weights: Optional[dict[AnyResource, int]] = None
    """Weights to be used by the objective when summing used resources or resources consumption.

    This is the case if `objective` is set to `Objective.NB_RESOURCES_USED` or  `Objective.RESOURCES_CONSUMPTION`.
    Default to 1 for resources not mentioned.

    Hypothesis: cumulative, unary, and non-renewable resources have different values.
    (It could happen that non-renewable resources and renewable lists intersect which whould be a problem
    for weights definition).

    """
    # Task start/end bounds settings
    use_cpm_for_task_bounds = False
    """Flag telling whether cpm should be used to refine task bounds."""
    # Multimode settings
    duplicate_start_var_per_mode = False
    """Whether adding a start variable for each task mode."""
    avoid_interval_optional: bool = True
    """Whether using task intervals + demand vars instead of optional intervals depending on is_present[mode]."""
    # Energy constraints settings
    use_energy_constraints = False
    """Whether using energy constraints."""
    keep_only_most_nested_energy_constraints = True
    """Whether to keep only most nested subgraphs for energy constraints."""
    # Calendar constraints settings
    add_redundant_skill_cumulative_constraints = False
    """Whether adding redundant calendar cumulative constraints on skills.

    These constraints are redundant with the calendar constraints on unary_resources
    as the calendar for a skill is deduce from unary_resource calendars.

    """

    # cpsat variables
    start_or_end_variables: dict[tuple[Task, StartOrEnd], LinearExprT]
    duration_variables: dict[Task, LinearExprT]
    task_interval_variables = dict[Task, IntervalVar]
    modes_is_present: dict[Task, dict[int, LinearExprT]]
    modes_intervals: dict[Task, dict[int, IntervalVar]]
    modes_start_variables: dict[Task, dict[int, LinearExprT]]
    allocation_is_present: dict[Task, dict[UnaryResource, LinearExprT]]
    allocation_intervals: dict[Task, dict[UnaryResource, IntervalVar]]
    skill_variables: dict[Task, dict[UnaryResource, dict[Skill, LinearExprT]]]
    demand_variables: dict[Task, dict[AnyResource, LinearExprT]]
    energy_variables: dict[Task, dict[AnyResource, LinearExprT]]
    all_used_variables: dict[AnyResource, IntVar]
    """Variables tracking whether a (unary, cumulative, or non-renewable) resource has been used at least once."""
    all_used_variables_created = False
    """Flag telling whether 'all_used_variables' have been created"""
    resource_consumption_variables: dict[AnyResource, LinearExprT]
    """Variables tracking total consumption of each (unary, cumulative, or non-renewable) resource."""
    resource_consumption_variables_created = False
    """Flag telling whether 'resource_consumption_variables' have been created"""

    @property
    def needs_duration_variables(self) -> bool:
        """Whether the task duration variables are needed by the model.

        Default implementation, returns True only if the problem is an allocation one (at least one unary resource).
        If additional custom constraints require them, override it.

        """
        return (
            len(self.problem.unary_resources_list) > 0 or self.avoid_interval_optional
        )

    @property
    def needs_task_interval(self) -> bool:
        """Whether the task interval variables are needed by the model.

        By default, these variables are only constraints on durations variables and need not to be stored.
        If additional custom constraints require them, override this property.

        """
        return self.avoid_interval_optional

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
        if resource in self.problem.skills_list:
            return self.add_redundant_skill_cumulative_constraints
        else:
            return True

    def compute_task_bounds(self) -> None:
        """Compute tighter bounds for tasks.

        - if `use_cpm_for_task_bounds`, propagate bounds forward and backward in the precedence by
        using the min possible duration for the task.
        - else only use min possible duration with 0, new_horizon (set by the solver in `self.get_makespan_upper_bound()`)

        """
        if self.use_cpm_for_task_bounds:
            cpm = Cpm(problem=self.problem, horizon=self.get_makespan_upper_bound())
            cpm.compute_task_bounds()
            self.tasks_bounds = cpm.get_task_bounds()
        else:
            self.tasks_bounds = {}
            for task in self.problem.tasks_list:
                start_lower_bound = self.problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.START
                )
                end_upper_bound = min(
                    # use current bound + new "horizon"
                    self.problem.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.END
                    ),
                    self.get_makespan_upper_bound(),
                )
                min_duration = min(
                    self.problem.get_task_mode_duration(task=task, mode=mode)
                    for mode in self.problem.get_task_modes(task=task)
                )
                end_lower_bound = max(
                    # use start_lower bound + min_durations
                    self.problem.get_task_start_or_end_lower_bound(
                        task=task, start_or_end=StartOrEnd.END
                    ),
                    start_lower_bound + min_duration,
                )
                start_upper_bound = min(
                    # use end_upper_bound - min_durations
                    self.problem.get_task_start_or_end_upper_bound(
                        task=task, start_or_end=StartOrEnd.START
                    ),
                    end_upper_bound - min_duration,
                )
                self.tasks_bounds[task] = (
                    start_lower_bound,
                    end_lower_bound,
                    start_upper_bound,
                    end_upper_bound,
                )

    def get_task_start_or_end_lower_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        """Get a lower bound on start or end of a given task.

        Use either the bounds given in `__init__()` or computed via `compute_tasks_bounds()`.

        Args:
            task:
            start_or_end:

        Returns:

        """
        start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound = (
            self.tasks_bounds[task]
        )
        match start_or_end:
            case StartOrEnd.START:
                return start_lower_bound
            case _:
                return end_lower_bound

    def get_task_start_or_end_upper_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        """Get an upper bound on start or end of a given task.

        Use either the bounds given in `__init__()` or computed via `compute_tasks_bounds()`.

        Args:
            task:
            start_or_end:

        Returns:

        """
        start_lower_bound, end_lower_bound, start_upper_bound, end_upper_bound = (
            self.tasks_bounds[task]
        )
        match start_or_end:
            case StartOrEnd.START:
                return start_upper_bound
            case _:
                return end_upper_bound

    def init_model(
        self,
        tasks_bounds: Optional[dict[Task, tuple[int, int, int, int]]] = None,
        use_cpm_for_task_bounds: Optional[bool] = None,
        avoid_interval_optional: Optional[bool] = None,
        duplicate_start_var_per_mode: Optional[bool] = None,
        use_energy_constraints: Optional[bool] = None,
        keep_only_most_nested_energy_constraints: Optional[bool] = None,
        add_redundant_skill_cumulative_constraints: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Init cp model and reset stored variables if any."""
        super().init_model(**kwargs)

        # update default settings
        if add_redundant_skill_cumulative_constraints is not None:
            self.add_redundant_skill_cumulative_constraints = (
                add_redundant_skill_cumulative_constraints
            )
        if use_cpm_for_task_bounds is not None:
            self.use_cpm_for_task_bounds = use_cpm_for_task_bounds
        if use_energy_constraints is not None:
            self.use_energy_constraints = use_energy_constraints
        if keep_only_most_nested_energy_constraints is not None:
            self.keep_only_most_nested_energy_constraints = (
                keep_only_most_nested_energy_constraints
            )
        if avoid_interval_optional is not None:
            self.avoid_interval_optional = avoid_interval_optional
        if duplicate_start_var_per_mode is not None:
            self.duplicate_start_var_per_mode = duplicate_start_var_per_mode

        # pre-compute tasks start/end bounds ?
        if tasks_bounds is None:
            self.compute_task_bounds()
        else:
            self.tasks_bounds = tasks_bounds

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
        self.demand_variables = {}
        self.energy_variables = {}
        self.allocation_is_present = {}
        self.allocation_intervals = {}
        self.skill_variables = {}

        self.all_used_variables_created = False
        self.all_used_variables = {}
        self.resource_consumption_variables_created = False
        self.resource_consumption_variables = {}

    def _create_variables(self):
        self._create_start_or_end_variables()
        self._create_mode_variables()
        if self.needs_duration_variables or self.needs_task_interval:
            self._create_task_duration_and_interval_variables()
        self._create_allocation_variables()
        self._create_skill_variables()
        if self.avoid_interval_optional:
            self._create_demand_variables()
        if self.use_energy_constraints:
            self._create_energy_variables()

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
            possible_durations = {
                self.problem.get_task_mode_duration(task=task, mode=mode)
                for mode in self.problem.get_task_modes(task)
            }
            if len(possible_durations) == 1:
                # single mode: fixed duration
                self.duration_variables[task] = next(iter(possible_durations))
            else:
                # multi mode
                self.duration_variables[task] = self.cp_model.new_int_var_from_domain(
                    domain=Domain.from_values(list(possible_durations)),
                    name=f"duration_{task}",
                )
                # duration per mode constraint  (managed by interval opt else)
                if self.avoid_interval_optional or not self.needs_task_interval:
                    # not needed if intervals optional per mode + intervals constraints already defined
                    for mode in self.problem.get_task_modes(task=task):
                        self.cp_model.add(
                            self.duration_variables[task]
                            == self.problem.get_task_mode_duration(task=task, mode=mode)
                        ).only_enforce_if(self.modes_is_present[task][mode])
            if self.needs_task_interval:
                # interval constraint
                self.task_interval_variables[task] = self.cp_model.new_interval_var(
                    start=self.start_or_end_variables[task, StartOrEnd.START],
                    size=self.duration_variables[task],
                    end=self.start_or_end_variables[task, StartOrEnd.END],
                    name=f"interval_{task}",
                )

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
            else:
                for mode in modes:
                    # multi mode
                    self.modes_is_present[task][mode] = self.cp_model.new_bool_var(
                        name=f"is_present_mode_{task}_{mode}"
                    )
                self.cp_model.add_exactly_one(
                    self.modes_is_present[task][mode] for mode in modes
                )
            if not self.avoid_interval_optional:
                for mode in modes:
                    self._create_mode_interval_on_the_fly(
                        task=task, mode=mode, modes=modes
                    )

    def _create_mode_interval_on_the_fly(
        self, task: Task, mode: int, modes: Optional[set[int]] = None
    ) -> None:
        if modes is None:
            modes = self.problem.get_task_modes(task=task)
        if len(modes) == 1:  # single mode
            # create the interval var with start and end => constraint on end - start
            self.modes_intervals[task][mode] = self.cp_model.new_interval_var(
                start=self.start_or_end_variables[task, StartOrEnd.START],
                size=self.problem.get_task_mode_duration(task=task, mode=mode),
                end=self.start_or_end_variables[task, StartOrEnd.END],
                name=f"interval_mode_{task}_{mode}",
            )
            if self.duplicate_start_var_per_mode:
                self.modes_start_variables[task][mode] = self.start_or_end_variables[
                    task, StartOrEnd.START
                ]
        else:  # multi mode
            is_present_mode = self.modes_is_present[task][mode]
            start = self.start_or_end_variables[task, StartOrEnd.START]
            end = self.start_or_end_variables[task, StartOrEnd.END]
            duration_mode = self.problem.get_task_mode_duration(task=task, mode=mode)
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
                self.cp_model.add(start_mode == start).only_enforce_if(is_present_mode)
                self.cp_model.add(start_mode + duration_mode == end).only_enforce_if(
                    is_present_mode
                )
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

    def _create_skill_variables(self):
        for task in self.problem.tasks_list:
            skills_of_task = self.problem.get_skills_of_task(task)
            if len(skills_of_task) == 0:
                # no skill usage to track
                continue
            self.skill_variables[task] = {}
            for unary_resource in self.problem.unary_resources_list:
                if self.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                ):
                    self.skill_variables[task][unary_resource] = {}
                    is_allocated = self.get_task_unary_resource_is_present_variable(
                        task=task, unary_resource=unary_resource
                    )
                    skills_of_unary_resource = (
                        self.problem.get_skills_of_unary_resource(unary_resource)
                    )

                    common_skills = skills_of_task.intersection(
                        skills_of_unary_resource
                    )
                    for skill in common_skills:
                        if len(common_skills) == 1:
                            self.skill_variables[task][unary_resource][skill] = (
                                is_allocated
                            )
                        else:
                            skill_var = self.cp_model.new_bool_var(
                                name=f"skill_{task}_{skill}_{unary_resource}"
                            )
                            self.skill_variables[task][unary_resource][skill] = (
                                skill_var
                            )
                            # no skill used if not allocated
                            self.cp_model.add(skill_var <= is_allocated)

                    # constraints on skill variables due to options
                    if (
                        len(common_skills) > 1
                    ):  # next constraints are useless if skill_var == is_allocated
                        if self.use_only_skill_to_allocate:
                            if self.problem.only_one_skill_per_task:
                                # allocation => exactly one skill used
                                self.cp_model.add_exactly_one(
                                    self.skill_variables[task][unary_resource].values()
                                ).only_enforce_if(is_allocated)
                            else:
                                # allocation => at least one skill used
                                self.cp_model.add_at_least_one(
                                    self.skill_variables[task][unary_resource].values()
                                ).only_enforce_if(is_allocated)
                        elif self.problem.only_one_skill_per_task:
                            # at most one skill from each resource used for a given task
                            self.cp_model.add_at_most_one(
                                self.skill_variables[task][unary_resource].values()
                            )

    def _create_demand_variables(self):
        for task in self.problem.tasks_list:
            self.demand_variables[task] = {}
            for resource in self.problem.unary_resources_list:
                self.demand_variables[task][resource] = (
                    self.get_task_unary_resource_is_present_variable(
                        task=task, unary_resource=resource
                    )
                )
            for resource in self.problem.cumulative_resources_list:
                self.demand_variables[task][resource] = self._create_var_per_mode(
                    name=f"demand_{task}_{resource}",
                    mode2value={
                        mode: self.problem.get_cumulative_resource_consumption(
                            resource=resource, task=task, mode=mode
                        )
                        for mode in self.problem.get_task_modes(task=task)
                    },
                    task=task,
                )
            for resource in self.problem.non_renewable_resources_list:
                self.demand_variables[task][resource] = self._create_var_per_mode(
                    name=f"demand_{task}_{resource}",
                    mode2value={
                        mode: self.problem.get_non_renewable_resource_consumption(
                            resource=resource, task=task, mode=mode
                        )
                        for mode in self.problem.get_task_modes(task=task)
                    },
                    task=task,
                )

    def _create_var_per_mode(
        self, name: str, mode2value: dict[int, int], task: Task
    ) -> LinearExprT:
        possible_values = set(mode2value.values())
        if len(possible_values) == 1:
            var = next(iter(possible_values))
        else:
            var = self.cp_model.new_int_var_from_domain(
                domain=Domain.from_values(list(possible_values)), name=name
            )
            for mode, value in mode2value.items():
                self.cp_model.add(var == value).only_enforce_if(
                    self.modes_is_present[task][mode]
                )
        return var

    def _create_energy_variables(self):
        for task in self.problem.tasks_list:
            self.energy_variables[task] = {}
            for resource in self.problem.unary_resources_list:
                is_allocated = self.get_task_unary_resource_is_present_variable(
                    task=task, unary_resource=resource
                )
                if isinstance(is_allocated, int) and is_allocated == 0:
                    # never allocated
                    var = 0
                else:
                    # value depending on allocation + mode
                    name = f"energy_{task}_{resource}"
                    mode2value = {
                        mode: self.problem.get_task_mode_duration(task=task, mode=mode)
                        for mode in self.problem.get_task_modes(task=task)
                    }
                    possible_values = set(mode2value.values())
                    possible_values.add(0)  # no allocation
                    var = self.cp_model.new_int_var_from_domain(
                        domain=Domain.from_values(list(possible_values)), name=name
                    )
                    self.cp_model.add(var == 0).only_enforce_if(~is_allocated)
                    for mode, value in mode2value.items():
                        self.cp_model.add(var == value).only_enforce_if(
                            [
                                is_allocated,
                                self.modes_is_present[task][mode],
                            ]
                        )
                self.energy_variables[task][resource] = var

            for resource in self.problem.cumulative_resources_list:
                self.energy_variables[task][resource] = sum(
                    self.modes_is_present[task][mode]
                    * self.problem.get_task_mode_duration(task=task, mode=mode)
                    * self.problem.get_cumulative_resource_consumption(
                        task=task, mode=mode, resource=resource
                    )
                    for mode in self.problem.get_task_modes(task=task)
                )

    def _create_allocation_variables(self):
        for task in self.problem.tasks_list:
            self.allocation_is_present[task] = {}
            self.allocation_intervals[task] = {}
            for unary_resource in self.problem.unary_resources_list:
                if self.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                ):
                    self.allocation_is_present[task][unary_resource] = (
                        self.cp_model.new_bool_var(
                            name=f"is_allocated_{task}_{unary_resource}"
                        )
                    )
                    if not self.avoid_interval_optional:
                        self._create_allocation_interval_on_the_fly(
                            task=task, unary_resource=unary_resource
                        )

    def _create_allocation_interval_on_the_fly(
        self, task: Task, unary_resource: UnaryResource
    ) -> None:
        self.allocation_intervals[task][unary_resource] = (
            self.cp_model.new_optional_interval_var(
                start=self.start_or_end_variables[task, StartOrEnd.START],
                size=self.duration_variables[task],
                end=self.start_or_end_variables[task, StartOrEnd.END],
                is_present=self.get_task_unary_resource_is_present_variable(
                    task=task, unary_resource=unary_resource
                ),
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
                    return self.problem.get_cumulative_resource_consumption(
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
        resource: Resource | NonRenewableResource,
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
                conso_tot_var = self.cp_model.new_int_var(
                    lb=0, ub=max_capacity, name=f"conso_{resource}"
                )
                if max_capacity > 1:
                    # cumulative constraint on the new "conso" variable
                    if self.avoid_interval_optional:
                        intervals_n_consumptions = (
                            self.get_resource_consumption_intervals(resource=resource)
                        )
                        intervals = [
                            interval
                            for interval, value in intervals_n_consumptions
                            if not isinstance(value, int) or value > 0
                        ]
                        demands = [
                            value
                            for interval, value in intervals_n_consumptions
                            if not isinstance(value, int) or value > 0
                        ]
                        if len(intervals) > 0:
                            self.cp_model.add_cumulative(
                                intervals=intervals,
                                demands=demands,
                                capacity=conso_tot_var,
                            )
                            for (
                                _,
                                conso_var,
                            ) in intervals_n_consumptions:
                                self.cp_model.add(conso_tot_var >= conso_var)
                        else:
                            conso_tot_var = 0
                    else:
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
                                conso
                                := self.problem.get_cumulative_resource_consumption(
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
                                capacity=conso_tot_var,
                            )
                            for (
                                _,
                                conso,
                                is_present,
                            ) in mode_intervals_consumptions_is_present:
                                self.cp_model.add(conso_tot_var >= conso * is_present)
                        else:
                            conso_tot_var = 0
                else:
                    # disjunctive resource, no need to use the interval variables
                    # (no overlap constraint already handled by `create_calendar_resources_constraint()`
                    list_is_present_variables = [
                        self.get_task_mode_is_present_variable(task=task, mode=mode)
                        for task in self.problem.tasks_list
                        for mode in self.problem.get_task_modes(task=task)
                        if self.problem.get_cumulative_resource_consumption(
                            resource=resource, task=task, mode=mode
                        )
                        > 0
                    ]
                    if len(list_is_present_variables) > 0:
                        self.cp_model.add_max_equality(
                            conso_tot_var, list_is_present_variables
                        )
                    else:
                        conso_tot_var = 0

                self.resource_consumption_variables[resource] = conso_tot_var
            # non-renewable resources
            for resource in self.problem.non_renewable_resources_list:
                self.resource_consumption_variables[resource] = sum(
                    self.get_non_renewable_resource_demand_variable(
                        task=task, resource=resource
                    )
                    for task in self.problem.tasks_list
                )

            self.resource_consumption_variables_created = True

    def check_resources_lists(self):
        resources_list = (
            self.problem.calendar_resources_list
            + self.problem.non_renewable_resources_list
        )
        assert len(resources_list) == len(set(resources_list)), (
            "There are duplicates in resources list, "
            "potentially because calendar and non-renewable resources intersect."
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
        # time lag
        self.create_timelag_constraints()
        # non-renewable resources capacity
        for resource in self.problem.non_renewable_resources_list:
            self.create_non_renewable_resources_constraint(resource=resource)
        # cumulative + unary resources calendar
        for resource in self.problem.calendar_resources_list:
            if not self.problem.is_cumulative_resource(
                resource
            ) or self.include_constraint_on_cumulative_resource(resource=resource):
                self.create_calendar_resources_constraint(resource=resource)
        # precedence
        self.create_precedence_constraints()
        # at most or exactly one resource allocated per task?
        self.add_unary_resources_per_task_constraints()
        # skill value per task constraints
        self.create_fine_skill_constraints()
        self.create_coarse_skill_constraints()  # redundant
        # energy constraints
        if self.use_energy_constraints:
            branches = self.prepare_energy_constraints()
            self.create_energy_constraints(branches=branches)

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

    def get_task_interval(self, task: Task) -> IntervalVar:
        if self.needs_task_interval:
            return self.task_interval_variables[task]
        else:
            return super().get_task_interval(task=task)

    def get_cumulative_resource_demand_variable(
        self, task: Task, resource: CumulativeResource
    ) -> LinearExprT:
        if self.avoid_interval_optional:
            return self.demand_variables[task][resource]
        return super().get_cumulative_resource_demand_variable(
            task=task, resource=resource
        )

    def get_non_renewable_resource_demand_variable(
        self, task: Task, resource: NonRenewableResource
    ) -> LinearExprT:
        if self.avoid_interval_optional:
            return self.demand_variables[task][resource]
        return super().get_non_renewable_resource_demand_variable(
            task=task, resource=resource
        )

    def get_task_unary_resource_interval(
        self, task: Task, unary_resource: UnaryResource
    ) -> IntervalVar:
        try:
            return self.allocation_intervals[task][unary_resource]
        except KeyError as e:
            if self.avoid_interval_optional:
                logger.warning(
                    f"Creating optional allocation interval for task {task} and unary_resource {unary_resource},"
                    " even though `self.avoid_interval_optional` is True."
                )
                self._create_allocation_interval_on_the_fly(
                    task=task, unary_resource=unary_resource
                )
                return self.allocation_intervals[task][unary_resource]
            else:
                raise e

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
        try:
            return self.modes_intervals[task][mode]
        except KeyError as e:
            if self.avoid_interval_optional:
                logger.warning(
                    f"Creating optional interval for task {task} and mode {mode},"
                    " even though `self.avoid_interval_optional` is True."
                )
                self._create_mode_interval_on_the_fly(task=task, mode=mode)
                return self.modes_intervals[task][mode]
            else:
                raise e

    def get_skill_variable(
        self, task: Task, unary_resource: UnaryResource, skill: Skill
    ) -> LinearExprT:
        try:
            return self.skill_variables[task][unary_resource][skill]
        except KeyError:
            return 0

    def retrieve_tasks_variables(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> TemporarySolution[Task, UnaryResource, Skill]:
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

            def get_skill_used(task: Task, unary_resource: UnaryResource) -> set[Skill]:
                try:
                    skill_variables = self.skill_variables[task][unary_resource]
                except KeyError:
                    return set()
                else:
                    return {
                        skill
                        for skill, skill_var in skill_variables.items()
                        if cpsolvercb.Value(skill_var)
                    }

            allocated = {
                unary_resource: get_skill_used(task=task, unary_resource=unary_resource)
                for unary_resource, is_allocated_var in self.allocation_is_present[
                    task
                ].items()
                if cpsolvercb.Value(is_allocated_var)
            }
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
            Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
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
        self, temp_sol: TemporarySolution[Task, UnaryResource, Skill]
    ) -> GenericSchedulingSolution[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]:
        """Convert solution from autosolver format into do format.

        To be used in `self.retrieve_solution()`.

        Args:
            temp_sol:

        Returns:

        """
        ...

    def prepare_energy_constraints(self) -> list[tuple[Task, Task, set[Task]]]:
        """Analyses the dependency graph to improve the model.

        Args:
          problem: the protobuf of the problem to solve.

        Returns:
          a list of (task1, task2, in_between_tasks) with task2 and indirect successor
          of task1, and in_between_tasks being the list of all tasks after task1 and
          before task2.
        """

        # Search for pair of tasks, containing at least two parallel branch between
        # them in the precedence graph.
        result = []
        graph_nx = self.problem.get_precedence_graph().to_networkx()
        outs = {node: set(nx.neighbors(graph_nx, node)) for node in graph_nx.nodes()}
        ins = {node: set(graph_nx.predecessors(node)) for node in graph_nx.nodes()}
        before = {
            node: nx.algorithms.ancestors(graph_nx, node) for node in graph_nx.nodes()
        }
        after = {
            node: nx.algorithms.descendants(graph_nx, node) for node in graph_nx.nodes()
        }
        for source, start_outs in outs.items():
            if len(start_outs) <= 1:
                # Starting with the unique successor of source will be as good.
                continue
            for sink, end_ins in ins.items():
                if len(end_ins) <= 1:
                    # Ending with the unique predecessor of sink will be as good.
                    continue
                if sink == source:
                    continue
                if sink not in after[source]:
                    continue

                num_active_outgoing_branches = 0
                num_active_incoming_branches = 0
                for succ in outs[source]:
                    if sink in after[succ]:
                        num_active_outgoing_branches += 1
                for pred in ins[sink]:
                    if source in before[pred]:
                        num_active_incoming_branches += 1

                if (
                    num_active_outgoing_branches <= 1
                    or num_active_incoming_branches <= 1
                ):
                    continue

                common = after[source].intersection(before[sink])
                if len(common) <= 1:
                    continue
                result.append((source, sink, common))

        # Sort entries lexicographically by (len(common), source, sink)
        def representation_multidiscrete(
            entry: tuple[Task, Task, set[Task]],
        ) -> tuple[int, int, int]:
            source, sink, common = entry
            return (
                len(common),
                self.problem.get_index_from_task(source),
                self.problem.get_index_from_task(sink),
            )

        result.sort(key=representation_multidiscrete)
        logger.debug(
            f"Energy constraints preparation:  created {len(result)} pairs of nodes to examine"
        )

        # filter and keep only most nested subgraphs
        if self.keep_only_most_nested_energy_constraints:
            subgraphs = []
            for source, sink, common in result:
                if not any(
                    (
                        (subsource == source or subsource in common)
                        and (subsink == sink or subsink in common)
                    )
                    for subsource, subsink, _ in subgraphs
                ):
                    # no smaller subgraph already added in this subgraph => ok
                    subgraphs.append((source, sink, common))
            result = subgraphs
            logger.debug(
                f"Energy constraints preparation:  keep {len(result)} most nested subgraphs"
            )
        return result

    def create_energy_constraints(
        self, branches: list[tuple[Task, Task, set[Task]]]
    ) -> None:
        for local_start_task, local_end_task, common in branches:
            for resource in self.problem.cumulative_resources_list:
                self.cp_model.add(
                    sum(self.energy_variables[task][resource] for task in common)
                    <= self.problem.get_resource_max_capacity(resource=resource)
                    * (
                        self.start_or_end_variables[local_end_task, StartOrEnd.START]
                        - self.start_or_end_variables[local_start_task, StartOrEnd.END]
                    )
                )
            for resource in self.problem.unary_resources_list:
                self.cp_model.add(
                    sum(self.energy_variables[task][resource] for task in common)
                    <= (
                        self.start_or_end_variables[local_end_task, StartOrEnd.START]
                        - self.start_or_end_variables[local_start_task, StartOrEnd.END]
                    )
                )


class SinglemodeGenericSchedulingAutoCpSatSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
    SinglemodeSchedulingCpSatSolver[Task],
):
    """Subclass of GenericSchedulingAutoCpSatSolver for single mode problems.

    Give access to task intervals without dealing with modes.

    """

    problem: SinglemodeProblem[Task]

    def get_task_interval(self, task: Task) -> IntervalVar:
        """Task interval with fixed duration, single mode."""
        if self.needs_task_interval:
            return super().get_task_interval(task=task)
        else:
            return self.get_task_mode_interval(
                task=task, mode=self.problem.default_mode
            )
