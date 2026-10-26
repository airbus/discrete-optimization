#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from abc import abstractmethod
from enum import Enum
from functools import reduce
from typing import Any, Iterable, Optional

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationCpSolver,
    AllocationSolution,
    MultimodeAllocationProblem,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.base.base import (
    TasksCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.base.multimode import (
    MultimodeCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.base.typing import LinearExpr
from discrete_optimization.generic_tasks_tools.solvers.utils import is_a_trivial_zero
from discrete_optimization.generic_tools.cp_tools import SignEnum

logger = logging.getLogger(__name__)


class AllocationCpBaseSolver(
    TasksCpBaseSolver[Task],
    AllocationCpSolver[Task, UnaryResource],
):
    """Base class for allocation cp-sat solvers using a binary modelling.

    I.e. using 0-1 variables to model allocation status of each couple (task, unary_resource)
    This is a more general modelisation thant the integer one as it allows allocation of multiple resources.

    """

    at_most_one_unary_resource_per_task = False
    """Flag telling if the problem accept at most one unary_resource per task.

    Default to False, ie several resources allowed per task.

    """
    exactly_one_unary_resource_per_task = False
    """Whether enforcing exactly one resource allocated to each task."""

    allocation_changes_variables_created = False
    """Flag telling whether 'allocation changes variables' have been created"""
    allocation_changes_variables: dict[tuple[Task, UnaryResource], LinearExpr]
    """Variables tracking allocation changes from a given reference."""
    used_variables_created = False
    """Flag telling whether 'used variables' have been created"""
    used_variables: dict[UnaryResource, LinearExpr]
    """Variables tracking whether a unary resource has been used at least once."""
    task_allocated_variables_created = False
    """Flag telling whether 'task_allocated_variables' have been created"""
    task_allocated_variables: dict[Task, LinearExpr]
    """Variables tracking whether a task has at least one unary resource allocated."""

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
        self.task_allocated_variables_created = False
        self.task_allocated_variables_created = {}

    @abstractmethod
    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExpr:
        """Return a 0-1 variable/expression telling if the unary_resource is used for the task.

        This variable should be 0 whenever an optional task is absent.

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
        return [self.add_bool_constraint(var == used)]

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
                        self.add_bool_constraint(allocation_change == is_allocated_ref)
                    )
                else:
                    constraints += self.add_implication(
                        allocation_change, is_present != is_allocated_ref
                    ) + self.add_implication(
                        ~allocation_change, is_present == is_allocated_ref
                    )
        return constraints

    def create_allocation_changes_variables(self):
        """Create variables necessary for constraint on nb of changes."""
        if not self.allocation_changes_variables_created:
            tasks, unary_resources = self.get_default_tasks_n_unary_resources()
            self.allocation_changes_variables = {
                (task, unary_resource): self.create_bool_var(
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
                used = self.create_bool_var(f"used_{unary_resource}")
                self.used_variables[unary_resource] = used
                list_is_allocated_variables = [
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
                if len(list_is_allocated_variables) > 0:
                    self.add_max_equality(used, list_is_allocated_variables)
                else:
                    self.add_bool_constraint(used == 0)
            self.used_variables_created = True

    def create_task_allocated_variables(self):
        if not self.task_allocated_variables_created:
            self.task_allocated_variables = {}
            for task in self.subset_tasks_of_interest:
                allocated = self.create_bool_var(f"{task}_allocated")
                self.task_allocated_variables[task] = allocated
                list_is_allocated_variables = [
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
                if len(list_is_allocated_variables) > 0:
                    if self.at_most_one_unary_resource_per_task:
                        nb_teams_allocated_to_task = sum(list_is_allocated_variables)
                        self.add_implication(allocated, nb_teams_allocated_to_task == 1)
                        self.add_implication(
                            ~allocated, nb_teams_allocated_to_task == 0
                        )
                    else:
                        self.add_max_equality(allocated, list_is_allocated_variables)
                else:
                    self.add_bool_constraint(allocated == 0)
            self.task_allocated_variables_created = True

    def add_unary_resources_per_task_constraints(self):
        """Add constraints on number min/max of allocated resources per task.

        According to options `at_most_one_unary_resource_per_task` and `exactly_one_unary_resource_per_task`.

        """
        if (
            self.exactly_one_unary_resource_per_task
            and not self.at_most_one_unary_resource_per_task
        ):
            # warning if not exactly => at_most
            logger.warning(
                "`at_most_one_unary_resource_per_task` is False while `exactly_one_unary_resource_per_task` is True. "
                "`exactly_one_unary_resource_per_task` will take the precedence."
            )
        if (
            self.exactly_one_unary_resource_per_task
            or self.at_most_one_unary_resource_per_task
        ):
            for task in self.subset_tasks_of_interest:
                list_is_allocated_variables = [
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
                if (
                    len(list_is_allocated_variables) > 0
                    and self.exactly_one_unary_resource_per_task
                ):
                    if self.problem.is_optional(task):
                        # enforce exactly one allocation only if task is not absent
                        # NB: only_enforce_if does not work with `add_exactly_one`constraint, so use sum()==1 instead
                        # see https://groups.google.com/g/or-tools-discuss/c/2G4Y7AArZks
                        self.add_implication(
                            self.get_task_is_present_variable(task),
                            sum(list_is_allocated_variables) == 1,
                        )
                    else:
                        self.add_exactly_one(list_is_allocated_variables)
                elif (
                    len(list_is_allocated_variables) > 1
                    and self.at_most_one_unary_resource_per_task
                ):
                    self.add_at_most_one(list_is_allocated_variables)

    def add_same_unary_allocation_constraints(self):
        for set_tasks in self.problem.get_same_unary_allocation():
            common_ur = reduce(
                lambda x, y: x.intersection(self.problem.compatible_unary_resources(y)),
                list(set_tasks),
                set(self.problem.unary_resources_list),
            )
            for c_ur in common_ur:
                self.add_allowed_assignments(
                    [
                        self.get_task_unary_resource_is_present_variable(task, c_ur)
                        for task in set_tasks
                    ],
                    [
                        tuple([1] * len(set_tasks)),
                        tuple([0] * len(set_tasks)),
                    ],
                )
            # Redundant
            list_tasks = list(set_tasks)
            for c_ur in common_ur:
                for i in range(len(list_tasks) - 1):
                    self.add_bool_constraint(
                        self.get_task_unary_resource_is_present_variable(
                            list_tasks[i], c_ur
                        )
                        == self.get_task_unary_resource_is_present_variable(
                            list_tasks[i + 1], c_ur
                        )
                    )

    def add_allocation_constraints(self):
        self.add_same_unary_allocation_constraints()
        self.add_unary_resources_per_task_constraints()

    def get_nb_tasks_allocated_variable(self) -> LinearExpr:
        self.create_task_allocated_variables()
        return sum(self.task_allocated_variables.values())

    def get_nb_unary_resources_used_variable(self) -> LinearExpr:
        self.create_used_variables()
        return sum(self.used_variables.values())


class MultimodeAllocationCpBaseSolver(
    AllocationCpBaseSolver[Task, UnaryResource],
    MultimodeCpBaseSolver[Task],
):
    problem: MultimodeAllocationProblem[Task, UnaryResource]

    def create_compatibility_task_mode_unary_resource_constraint(self):
        for task in self.problem.tasks_list:
            ur = self.problem.compatible_unary_resources(task)
            for u in ur:
                forbidden_modes = (
                    self.problem.get_forbidden_modes_for_task_and_unary_resource(
                        task, unary_resource=u
                    )
                )
                for fm in forbidden_modes:
                    lit1 = self.get_task_mode_is_present_variable(task=task, mode=fm)
                    lit2 = self.get_task_unary_resource_is_present_variable(
                        task=task, unary_resource=u
                    )
                    self.add_at_most_one([lit1, lit2])


class AllocationIntegerModellingCpBaseSolver(
    AllocationCpBaseSolver[Task, UnaryResource],
):
    """Base class for allocation cp-sat solvers using an integer modelling.

    I.e. using integer variables to model allocation of a task.
    This assumes that at most one unary_resource can be allocated to a task.

    """

    is_present_variables_created = False
    is_present_variables: dict[tuple[Task, UnaryResource], LinearExpr]

    def init_model(self, **kwargs: Any) -> None:
        """Init cp model and reset stored variables if any."""
        super().init_model(**kwargs)
        self.is_present_variables_created = False
        self.is_present_variables = {}

    @abstractmethod
    def get_task_allocation_variable(
        self,
        task: Task,
    ) -> LinearExpr:
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
                    if self.is_compatible_task_unary_resource(
                        task=task, unary_resource=unary_resource
                    ):
                        boolvar = self.create_bool_var(
                            f"is_present_{task}_{unary_resource}"
                        )
                        self.is_present_variables[(task, unary_resource)] = boolvar
                        var = self.get_task_allocation_variable(task=task)
                        value = self.problem.get_index_from_unary_resource(
                            unary_resource
                        )
                        self.add_implication(boolvar, var == value)
                        self.add_implication(~boolvar, var != value)

            self.is_present_variables_created = True

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExpr:
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
            return [self.add_bool_constraint(var == unary_resource)]
        else:
            return [self.add_bool_constraint(var != unary_resource)]

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
                    subconstraints = self.add_implication(
                        allocation_change, task_allocation != i_unary_resource
                    ) + self.add_implication(
                        ~allocation_change, task_allocation == i_unary_resource
                    )
                else:
                    subconstraints = self.add_implication(
                        allocation_change, task_allocation == i_unary_resource
                    ) + self.add_implication(
                        ~allocation_change, task_allocation != i_unary_resource
                    )
                constraints += subconstraints
        return constraints


class AllocationModelling(Enum):
    BINARY = "binary"
    INTEGER = "integer"


class AllocationBinaryOrIntegerModellingCpBaseSolver(
    AllocationIntegerModellingCpBaseSolver[Task, UnaryResource],
):
    """Base class for allocation cp-sat solvers using a binary or integer modelling."""

    allocation_modelling: AllocationModelling

    @abstractmethod
    def get_binary_allocation_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExpr:
        """ "Return a 0-1 variable/expression telling if the unary_resource is used for the task.

        Only to be called when allocation_modelling == AllocationModelling.BINARY.

        NB: sometimes the given resource is never to be used by a task and the variable has not been created.
        The convention is to return 0 in that case.

        """
        ...

    @abstractmethod
    def get_integer_allocation_variable(self, task: Task) -> LinearExpr:
        """Return an integer variable/expression storing the index of the allocated unary_resource.

        Assumes that exactly one unary resource is allocated to a task.
        Only to be called when allocation_modelling == AllocationModelling.INTEGER.


        Args:
            task:

        Returns:

        """
        ...

    def get_task_allocation_variable(self, task: Task) -> LinearExpr:
        if self.allocation_modelling == AllocationModelling.INTEGER:
            return self.get_integer_allocation_variable(task=task)
        else:
            raise RuntimeError(
                "get_task_allocation_variable() cannot be called with binary allocation modelling."
            )

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExpr:
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
                AllocationCpBaseSolver.add_constraint_on_task_unary_resource_allocation(
                    self, task=task, unary_resource=unary_resource, used=used
                )
            )

    def add_allocation_changes_constraints(
        self, ref: AllocationSolution[Task, UnaryResource]
    ) -> list[Any]:
        if self.allocation_modelling == AllocationModelling.INTEGER:
            return super().add_allocation_changes_constraints(ref=ref)
        else:
            return AllocationCpBaseSolver.add_allocation_changes_constraints(
                self, ref=ref
            )
