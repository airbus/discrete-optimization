#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from enum import Enum
from typing import Any, Iterable, Optional

from ortools.sat.python.cp_model import IntVar, LinearExprT

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationCpSolver,
    AllocationSolution,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.multimode import MultimodeCpSolver
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingCpSolver,
    Task,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class SchedulingCpSatSolver(OrtoolsCpSatSolver, SchedulingCpSolver[Task]):
    """Base class for most ortools/cpsat solvers handling scheduling problems.

    Allows to have common code.

    """

    _makespan: Optional[IntVar] = None
    """Internal variable use to define the global makespan."""

    _subtasks_makespan: Optional[IntVar] = None
    """Internal variable use to define the partial makespan."""

    constraints_on_makespan: Optional[list[Any]] = None
    """Constraints on partial makespan so that it can be considered as the objective."""

    def init_model(self, **kwargs: Any) -> None:
        """Init cp model and reset stored variables if any."""
        super().init_model(**kwargs)
        self._makespan = None
        self._subtasks_makespan = None
        self.constraints_on_makespan = None

    @abstractmethod
    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        """Retrieve the variable storing the start or end time of given task.

        Args:
            task:
            start_or_end:

        Returns:

        """
        ...

    def add_constraint_on_task(
        self, task: Task, start_or_end: StartOrEnd, sign: SignEnum, time: int
    ) -> list[Any]:
        var = self.get_task_start_or_end_variable(task=task, start_or_end=start_or_end)
        return self.add_bound_constraint(var=var, sign=sign, value=time)

    def add_constraint_chaining_tasks(self, task1: Task, task2: Task) -> list[Any]:
        var1 = self.get_task_start_or_end_variable(
            task=task1, start_or_end=StartOrEnd.END
        )
        var2 = self.get_task_start_or_end_variable(
            task=task2, start_or_end=StartOrEnd.START
        )
        return [self.cp_model.add(var1 == var2)]

    def _get_makespan_var(self) -> IntVar:
        """Get the makespan variable used to track global makespan."""
        if self._makespan is None:
            self._makespan = self.cp_model.NewIntVar(
                lb=self.get_makespan_lower_bound(),
                ub=self.get_makespan_upper_bound(),
                name="makespan",
            )
        return self._makespan

    def _get_subtasks_makespan_var(self) -> IntVar:
        """Get the makespan variable used to track subtasks makespan."""
        if self._subtasks_makespan is None:
            self._subtasks_makespan = self.cp_model.NewIntVar(
                lb=0,  # lower bound for any tasks subset
                ub=self.get_makespan_upper_bound(),
                name="subtasks_makespan",
            )
        return self._subtasks_makespan

    def remove_constraints_on_objective(self) -> None:
        if self.constraints_on_makespan is not None:
            self.remove_constraints(self.constraints_on_makespan)

    def get_global_makespan_variable(self) -> Any:
        # remove previous constraints on makespan variable from cp model
        self.remove_constraints_on_objective()
        # get makespan variable
        makespan = self._get_makespan_var()
        # update those constraints
        self.constraints_on_makespan = [
            self.cp_model.AddMaxEquality(
                makespan,
                [
                    self.get_task_start_or_end_variable(task, StartOrEnd.END)
                    for task in self.problem.get_last_tasks()
                ],
            )
        ]
        return makespan

    def get_subtasks_makespan_variable(self, subtasks: Iterable[Task]) -> Any:
        # remove previous constraints on makespan variable from cp model
        self.remove_constraints_on_objective()
        # get makespan variable
        makespan = self._get_subtasks_makespan_var()
        # update those constraints
        self.constraints_on_makespan = [
            self.cp_model.AddMaxEquality(
                makespan,
                [
                    self.get_task_start_or_end_variable(task, StartOrEnd.END)
                    for task in subtasks
                ],
            )
        ]
        return makespan

    def get_subtasks_sum_end_time_variable(self, subtasks: Iterable[Task]) -> Any:
        self.remove_constraints_on_objective()
        return sum(
            self.get_task_start_or_end_variable(task, StartOrEnd.END)
            for task in subtasks
        )

    def get_subtasks_sum_start_time_variable(self, subtasks: Iterable[Task]) -> Any:
        self.remove_constraints_on_objective()
        return sum(
            self.get_task_start_or_end_variable(task, StartOrEnd.START)
            for task in subtasks
        )


class MultimodeCpSatSolver(OrtoolsCpSatSolver, MultimodeCpSolver[Task]):
    @abstractmethod
    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        """Retrieve the 0-1 variable/expression telling if the mode is used for the task.

        Args:
            task:
            mode:

        Returns:

        """
        ...

    def add_constraint_on_task_mode(self, task: Task, mode: int) -> list[Any]:
        possible_modes = self.problem.get_task_modes(task)
        if mode not in possible_modes:
            raise ValueError(f"Task {task} cannot be done with mode {mode}.")
        if len(possible_modes) == 1:
            return []
        constraints = []
        for other_mode in possible_modes:
            var = self.get_task_mode_is_present_variable(task=task, mode=other_mode)
            if other_mode == mode:
                constraints.append(self.cp_model.add(var == True))
            else:
                constraints.append(self.cp_model.add(var == False))
        return constraints


class AllocationCpSatSolver(
    OrtoolsCpSatSolver,
    AllocationCpSolver[Task, UnaryResource],
):
    """Base class for allocation cp-sat solvers using a binary modelling.

    I.e. using 0-1 variables to model allocation status of each couple (task, unary_resource)
    This is a more general modelisation thant the integer one as it allows allocation of multiple resources.

    """

    allocation_changes_variables_created = False
    """Flag telling whether 'allocation changes variables' have been created"""
    allocation_changes_variables: dict[tuple[Task, UnaryResource], IntVar]
    """Variables tracking allocation changes from a given reference."""
    used_variables_created = False
    """Flag telling whether 'used variables' have been created"""
    used_variables: dict[UnaryResource, IntVar]
    """Variables tracking whether a unary resource has been used at least once."""
    done_variables_created = False
    """Flag telling whether 'done variables' have been created"""
    done_variables: dict[Task, IntVar]
    """Variables tracking whether a task has at least one unary resource allocated."""

    at_most_one_unary_resource_per_task = False
    """Flag telling if the problem accept at most one unary_resource per task.

    Default to False, ie several resources allowed per task.

    """

    @property
    def subset_tasks_of_interest(self) -> Iterable[Task]:
        """Subset of tasks of interest used for the objective.

        By default, all tasks.

        """
        return self.problem.tasks_list

    @property
    def subset_unaryresources_allowed(self) -> Iterable[UnaryResource]:
        """Unary resources allowed to solve the problem.

        By default, all unary resources.

        """
        return self.problem.unary_resources_list

    def get_default_tasks_n_unary_resources(
        self,
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> tuple[Iterable[Task], Iterable[UnaryResource]]:
        if tasks is None:
            tasks = self.subset_tasks_of_interest
        if unary_resources is None:
            unary_resources = self.subset_unaryresources_allowed
        return tasks, unary_resources

    def init_model(self, **kwargs: Any) -> None:
        """Init cp model and reset stored variables if any."""
        super().init_model(**kwargs)
        self.allocation_changes_variables_created = False
        self.allocation_changes_variables = {}
        self.used_variables_created = False
        self.used_variables = {}
        self.done_variables_created = False
        self.done_variables_created = {}

    @abstractmethod
    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        """Return a 0-1 variable/expression telling if the unary_resource is used for the task.

        NB: sometimes the given resource is never to be used by a task and the variable has not been created.
        The convention is to return 0 in that case.

        """
        ...

    def add_constraint_on_task_unary_resource_allocation(
        self, task: Task, unary_resource: UnaryResource, used: bool
    ) -> list[Any]:
        var = self.get_task_unary_resource_is_present_variable(
            task=task, unary_resource=unary_resource
        )
        return [self.cp_model.add(var == used)]

    def add_constraint_on_nb_allocation_changes(
        self,
        ref: AllocationSolution[Task, UnaryResource],
        nb_changes: int,
        sign: SignEnum = SignEnum.LEQ,
    ) -> list[Any]:
        self.create_allocation_changes_variables()
        # constraints so that change variables reflect diff to ref
        constraints = self.add_allocation_changes_constraints(ref=ref)
        # nb of changes variable
        var = sum(self.allocation_changes_variables.values())
        constraints += self.add_bound_constraint(var=var, sign=sign, value=nb_changes)
        return constraints

    def add_allocation_changes_constraints(
        self, ref: AllocationSolution[Task, UnaryResource]
    ) -> list[Any]:
        """Add and return constraints so that change variables reflect diff to ref."""
        tasks, unary_resources = self.get_default_tasks_n_unary_resources()
        constraints = []
        for task in tasks:
            for unary_resource in unary_resources:
                is_present = self.get_task_unary_resource_is_present_variable(
                    task=task, unary_resource=unary_resource
                )
                allocation_change = self.allocation_changes_variables[
                    (task, unary_resource)
                ]
                is_allocated_ref = ref.is_allocated(
                    task=task, unary_resource=unary_resource
                )
                if is_a_trivial_zero(is_present):
                    # can never be allocated: change <=> ref has allocated
                    constraints.append(
                        self.cp_model.add(allocation_change == is_allocated_ref)
                    )
                else:
                    constraints += [
                        self.cp_model.add(
                            is_present != is_allocated_ref
                        ).only_enforce_if(allocation_change),
                        self.cp_model.add(
                            is_present == is_allocated_ref
                        ).only_enforce_if(~allocation_change),
                    ]
        return constraints

    def create_allocation_changes_variables(self):
        """Create variables necessary for constraint on nb of changes."""
        if not self.allocation_changes_variables_created:
            tasks, unary_resources = self.get_default_tasks_n_unary_resources()
            self.allocation_changes_variables = {
                (task, unary_resource): self.cp_model.new_bool_var(
                    f"change_{task}_{unary_resource}"
                )
                for task in tasks
                for unary_resource in unary_resources
            }
            self.allocation_changes_variables_created = True

    def add_constraint_nb_unary_resource_usages(
        self,
        sign: SignEnum,
        target: int,
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> list[Any]:
        tasks, unary_resources = self.get_default_tasks_n_unary_resources(
            tasks=tasks, unary_resources=unary_resources
        )
        var = sum(
            is_present
            for task in tasks
            for unary_resource in unary_resources
            # filter out trivial 0's corresponding to incompatible (task, resource)
            if not (
                is_a_trivial_zero(
                    is_present := self.get_task_unary_resource_is_present_variable(
                        task, unary_resource
                    )
                )
            )
        )
        return self.add_bound_constraint(var=var, sign=sign, value=target)

    def add_constraint_on_total_nb_usages(
        self, sign: SignEnum, target: int
    ) -> list[Any]:
        return self.add_constraint_nb_unary_resource_usages(sign=sign, target=target)

    def add_constraint_on_unary_resource_nb_usages(
        self, unary_resource: UnaryResource, sign: SignEnum, target: int
    ) -> list[Any]:
        return self.add_constraint_nb_unary_resource_usages(
            sign=sign, target=target, unary_resources=(unary_resource,)
        )

    def create_used_variables(self):
        if not self.used_variables_created:
            self.used_variables = {}
            for unary_resource in self.subset_unaryresources_allowed:
                used = self.cp_model.new_bool_var(f"used_{unary_resource}")
                self.used_variables[unary_resource] = used
                list_is_present_variables = [
                    is_present
                    for task in self.subset_tasks_of_interest
                    # filter out trivial 0's corresponding to incompatible (task, resource)
                    if not (
                        is_a_trivial_zero(
                            is_present
                            := self.get_task_unary_resource_is_present_variable(
                                task, unary_resource
                            )
                        )
                    )
                ]
                if len(list_is_present_variables) > 0:
                    self.cp_model.add_max_equality(used, list_is_present_variables)
                else:
                    self.cp_model.add(used == 0)
            self.used_variables_created = True

    def create_done_variables(self):
        if not self.done_variables_created:
            self.done_variables = {}
            for task in self.subset_tasks_of_interest:
                done = self.cp_model.new_bool_var(f"{task}_done")
                self.done_variables[task] = done
                list_is_present_variables = [
                    is_present
                    for unary_resource in self.subset_unaryresources_allowed
                    # filter out trivial 0's corresponding to incompatible (task, resource)
                    if not (
                        is_a_trivial_zero(
                            is_present
                            := self.get_task_unary_resource_is_present_variable(
                                task, unary_resource
                            )
                        )
                    )
                ]
                if len(list_is_present_variables) > 0:
                    if self.at_most_one_unary_resource_per_task:
                        self.cp_model.add_at_most_one(list_is_present_variables)
                        nb_teams_allocated_to_task = sum(list_is_present_variables)
                        self.cp_model.add(
                            nb_teams_allocated_to_task == 1
                        ).only_enforce_if(done)
                        self.cp_model.add(
                            nb_teams_allocated_to_task == 0
                        ).only_enforce_if(~done)
                    else:
                        self.cp_model.add_max_equality(done, list_is_present_variables)
                else:
                    self.cp_model.add(done == 0)
            self.done_variables_created = True

    def get_nb_tasks_done_variable(self) -> Any:
        self.create_done_variables()
        return sum(self.done_variables.values())

    def get_nb_unary_resources_used_variable(self) -> Any:
        self.create_used_variables()
        return sum(self.used_variables.values())


class AllocationIntegerModellingCpSatSolver(
    AllocationCpSatSolver[Task, UnaryResource],
):
    """Base class for allocation cp-sat solvers using an integer modelling.

    I.e. using integer variables to model allocation of a task.
    This assumes that at most one unary_resource can be allocated to a task.

    """

    is_present_variables_created = False
    is_present_variables: dict[tuple[Task, UnaryResource], IntVar]

    @abstractmethod
    def get_task_allocation_variable(
        self,
        task: Task,
    ) -> LinearExprT:
        """Return an integer variable/expression storing the index of the allocated unary_resource.

        Assumes that exactly one unary resource is allocated to a task.

        """
        ...

    def create_is_present_variables(self) -> None:
        if not self.is_present_variables_created:
            tasks, unary_resources = self.get_default_tasks_n_unary_resources()
            self.is_present_variables = {}
            for task in tasks:
                for unary_resource in unary_resources:
                    if self.problem.is_compatible_task_unary_resource(
                        task=task, unary_resource=unary_resource
                    ):
                        boolvar = self.cp_model.new_bool_var(
                            f"is_present_{task}_{unary_resource}"
                        )
                        self.is_present_variables[(task, unary_resource)] = boolvar
                        var = self.get_task_allocation_variable(task=task)
                        value = self.problem.get_index_from_unary_resource(
                            unary_resource
                        )
                        self.cp_model.add(var == value).only_enforce_if(boolvar)
                        self.cp_model.add(var != value).only_enforce_if(~boolvar)

            self.is_present_variables_created = True

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        """Return a 0-1 variable/expression telling if the unary_resource is used for the task.

        NB: sometimes the given resource is never to be used by a task and the variable has not been created.
        The convention is to return 0 in that case.

        """
        self.create_is_present_variables()
        try:
            return self.is_present_variables[(task, unary_resource)]
        except KeyError:
            return 0

    def add_constraint_on_task_unary_resource_allocation(
        self, task: Task, unary_resource: UnaryResource, used: bool
    ) -> list[Any]:
        var = self.get_task_allocation_variable(task=task)
        if used:
            return [self.cp_model.add(var == unary_resource)]
        else:
            return [self.cp_model.add(var != unary_resource)]

    def add_allocation_changes_constraints(
        self, ref: AllocationSolution[Task, UnaryResource]
    ) -> list[Any]:
        """Add and return constraints so that change variables reflect diff to ref."""
        tasks, unary_resources = self.get_default_tasks_n_unary_resources()
        constraints = []
        for task in tasks:
            for unary_resource in unary_resources:
                task_allocation = self.get_task_allocation_variable(task=task)
                i_unary_resource = self.problem.get_index_from_unary_resource(
                    unary_resource=unary_resource
                )
                allocation_change = self.allocation_changes_variables[
                    (task, unary_resource)
                ]
                if ref.is_allocated(task=task, unary_resource=unary_resource):
                    subconstraints = [
                        self.cp_model.add(
                            task_allocation != i_unary_resource
                        ).only_enforce_if(allocation_change),
                        self.cp_model.add(
                            task_allocation == i_unary_resource
                        ).only_enforce_if(~allocation_change),
                    ]
                else:
                    subconstraints = [
                        self.cp_model.add(
                            task_allocation == i_unary_resource
                        ).only_enforce_if(allocation_change),
                        self.cp_model.add(
                            task_allocation != i_unary_resource
                        ).only_enforce_if(~allocation_change),
                    ]
                constraints += subconstraints
        return constraints


class AllocationModelling(Enum):
    BINARY = "binary"
    INTEGER = "integer"


class AllocationBinaryOrIntegerModellingCpSatSolver(
    AllocationIntegerModellingCpSatSolver[Task, UnaryResource],
):
    """Base class for allocation cp-sat solvers using a binary or integer modelling."""

    allocation_modelling: AllocationModelling

    @abstractmethod
    def get_binary_allocation_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        """ "Return a 0-1 variable/expression telling if the unary_resource is used for the task.

        Only to be called when allocation_modelling == AllocationModelling.BINARY.

        NB: sometimes the given resource is never to be used by a task and the variable has not been created.
        The convention is to return 0 in that case.

        """
        ...

    @abstractmethod
    def get_integer_allocation_variable(self, task: Task) -> LinearExprT:
        """Return an integer variable/expression storing the index of the allocated unary_resource.

        Assumes that exactly one unary resource is allocated to a task.
        Only to be called when allocation_modelling == AllocationModelling.INTEGER.


        Args:
            task:

        Returns:

        """
        ...

    def get_task_allocation_variable(self, task: Task) -> LinearExprT:
        if self.allocation_modelling == AllocationModelling.INTEGER:
            return self.get_integer_allocation_variable(task=task)
        else:
            raise RuntimeError(
                "get_task_allocation_variable() cannot be called with binary allocation modelling."
            )

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        if self.allocation_modelling == AllocationModelling.INTEGER:
            return super().get_task_unary_resource_is_present_variable(
                task=task, unary_resource=unary_resource
            )
        else:
            return self.get_binary_allocation_variable(
                task=task, unary_resource=unary_resource
            )

    def add_constraint_on_task_unary_resource_allocation(
        self, task: Task, unary_resource: UnaryResource, used: bool
    ) -> list[Any]:
        if self.allocation_modelling == AllocationModelling.INTEGER:
            return super().add_constraint_on_task_unary_resource_allocation(
                task=task, unary_resource=unary_resource, used=used
            )
        else:
            return (
                AllocationCpSatSolver.add_constraint_on_task_unary_resource_allocation(
                    self, task=task, unary_resource=unary_resource, used=used
                )
            )

    def add_allocation_changes_constraints(
        self, ref: AllocationSolution[Task, UnaryResource]
    ) -> list[Any]:
        if self.allocation_modelling == AllocationModelling.INTEGER:
            return super().add_allocation_changes_constraints(ref=ref)
        else:
            return AllocationCpSatSolver.add_allocation_changes_constraints(
                self, ref=ref
            )


def is_a_trivial_zero(var: LinearExprT) -> bool:
    """Return whether the variable is actually a plain 0 integer.

    For instance, tells if is_present variables are real variables or not to avoid
    including them in sum, max, ...

    """
    return isinstance(var, int) and var == 0
