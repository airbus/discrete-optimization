#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, Optional

try:
    import optalcp as cp
except ImportError:
    cp = None

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
from discrete_optimization.generic_tasks_tools.solvers.cpsat import is_a_trivial_zero
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)


class SchedulingOptalSolver(OptalCpSolver, SchedulingCpSolver[Task]):
    @abstractmethod
    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        """Retrieve the interval variable of given task."""
        ...

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> "cp.IntExpr":
        """Retrieve the variable storing the start or end time of given task.

        Args:
            task:
            start_or_end:

        Returns:

        """
        itv = self.get_task_interval_variable(task)
        if start_or_end == StartOrEnd.START:
            return self.cp_model.start(itv)
        if start_or_end == StartOrEnd.END:
            return self.cp_model.end(itv)
        return None

    def add_constraint_on_task(
        self, task: Task, start_or_end: StartOrEnd, sign: SignEnum, time: int
    ) -> list["cp.BoolExpr"]:
        var = self.get_task_start_or_end_variable(task, start_or_end)
        return self.add_bound_constraint(var, sign, time)

    def add_constraint_chaining_tasks(self, task1: Task, task2: Task) -> list[Any]:
        itv1 = self.get_task_interval_variable(task1)
        itv2 = self.get_task_interval_variable(task2)
        return [self.cp_model.start_at_end(itv2, itv1)]

    def get_subtasks_makespan_variable(self, subtasks: Iterable[Task]) -> Any:
        return self.cp_model.max(
            [
                self.get_task_start_or_end_variable(
                    task=task, start_or_end=StartOrEnd.END
                )
                for task in subtasks
            ]
        )

    def get_subtasks_sum_end_time_variable(self, subtasks: Iterable[Task]) -> Any:
        return self.cp_model.sum(
            [
                self.get_task_start_or_end_variable(
                    task=task, start_or_end=StartOrEnd.END
                )
                for task in subtasks
            ]
        )

    def get_subtasks_sum_start_time_variable(self, subtasks: Iterable[Task]) -> Any:
        return self.cp_model.max(
            [
                self.get_task_start_or_end_variable(
                    task=task, start_or_end=StartOrEnd.START
                )
                for task in subtasks
            ]
        )


class MultimodeOptalSolver(OptalCpSolver, MultimodeCpSolver[Task]):
    @abstractmethod
    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> "cp.BoolExpr":
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
                constraints.append(self.cp_model.enforce(var == True))
            else:
                constraints.append(self.cp_model.enforce(var == False))
        return constraints


class AllocationOptalSolver(
    OptalCpSolver,
    AllocationCpSolver[Task, UnaryResource],
):
    """Base class for allocation cp-sat solvers using a binary modelling.
    I.e. using 0-1 variables to model allocation status of each couple (task, unary_resource)
    This is a more general modelisation thant the integer one as it allows allocation of multiple resources.
    """

    allocation_changes_variables_created = False
    """Flag telling whether 'allocation changes variables' have been created"""
    allocation_changes_variables: dict[tuple[Task, UnaryResource], "cp.IntExpr"]
    """Variables tracking allocation changes from a given reference."""
    used_variables_created = False
    """Flag telling whether 'used variables' have been created"""
    used_variables: dict[UnaryResource, cp.BoolVar]
    """Variables tracking whether a unary resource has been used at least once."""
    done_variables_created = False
    """Flag telling whether 'done variables' have been created"""
    done_variables: dict[Task, cp.BoolExpr]
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
        self.used_variables_created = False
        self.used_variables = {}
        self.done_variables_created = False
        self.done_variables_created = {}

    @abstractmethod
    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> "cp.BoolExpr":
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
        expr = var == used
        self.cp_model.enforce(expr)
        return [expr]

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
                    self.cp_model.enforce(allocation_change == is_allocated_ref)
                    constraints.append(allocation_change == is_allocated_ref)
                else:
                    self.cp_model.enforce(
                        allocation_change == (is_present != is_allocated_ref)
                    )
                    constraints.append(
                        allocation_change == (is_present != is_allocated_ref)
                    )
        return constraints

    def create_allocation_changes_variables(self):
        """Create variables necessary for constraint on nb of changes."""
        if not self.allocation_changes_variables_created:
            tasks, unary_resources = self.get_default_tasks_n_unary_resources()
            self.allocation_changes_variables = {
                (task, unary_resource): self.cp_model.bool_var(
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
                used = self.cp_model.bool_var(f"used_{unary_resource}")
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
                    self.cp_model.enforce(
                        used == self.cp_model.max(list_is_present_variables)
                    )
                else:
                    self.cp_model.enforce(used == 0)
            self.used_variables_created = True

    def create_done_variables(self):
        if not self.done_variables_created:
            self.done_variables = {}
            for task in self.subset_tasks_of_interest:
                done = self.cp_model.bool_var(f"{task}_done")
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
                    self.cp_model.enforce(
                        done == self.cp_model.max(list_is_present_variables)
                    )
                else:
                    self.cp_model.enforce(done == 0)
            self.done_variables_created = True

    def get_nb_tasks_done_variable(self) -> Any:
        self.create_done_variables()
        return sum(self.done_variables.values())

    def get_nb_unary_resources_used_variable(self) -> Any:
        self.create_used_variables()
        return sum(self.used_variables.values())
