#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Iterable, Optional

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
    UnaryResource,
    get_default_tasks_n_unary_resources,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.base.base import TasksSolver
from discrete_optimization.generic_tools.enums import SignEnum


class AllocationSolver(TasksSolver[Task], Generic[Task, UnaryResource]):
    """Base class for solver managing constraints on allocation."""

    problem: AllocationProblem[Task, UnaryResource]

    @abstractmethod
    def add_constraint_on_task_unary_resource_allocation(
        self, task: Task, unary_resource: UnaryResource, used: bool
    ) -> list[Any]:
        """Add constraint on allocation of given unary resource for the given task

        Args:
            task:
            unary_resource:
            used: if True, we enforce the allocation of `unary_resource` to `task`, else we prevent it

        Returns:
            resulting constraints

        """
        ...

    @abstractmethod
    def add_constraint_on_nb_allocation_changes(
        self, ref: AllocationSolution[Task, UnaryResource], nb_changes: int
    ) -> list[Any]:
        """Add contraint on maximal number of allocation changes from the given reference.

        Args:
            ref:
            nb_changes: maximal number of changes

        Returns:
            resulting constraints

        """
        ...

    @abstractmethod
    def add_constraint_on_total_nb_usages(
        self, sign: SignEnum, target: int
    ) -> list[Any]: ...

    @abstractmethod
    def add_constraint_on_unary_resource_nb_usages(
        self, unary_resource: UnaryResource, sign: SignEnum, target: int
    ) -> list[Any]: ...

    def get_default_tasks_n_unary_resources(
        self,
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> tuple[Iterable[Task], Iterable[UnaryResource]]:
        return get_default_tasks_n_unary_resources(self.problem, tasks, unary_resources)

    def add_constraint_same_allocation_as_ref(
        self,
        ref: AllocationSolution[Task, UnaryResource],
        tasks: Optional[Iterable[Task]] = None,
        unary_resources: Optional[Iterable[UnaryResource]] = None,
    ) -> list[Any]:
        """Add constraint to keep same allocation as the reference for the given tasks and unary resources subsets.

        Args:
            ref:
            tasks:
            unary_resources:

        Returns:
            resulting constraints

        """
        constraints = []
        tasks, unary_resources = self.get_default_tasks_n_unary_resources(
            tasks, unary_resources
        )
        for unary_resource in unary_resources:
            for task in tasks:
                constraints += self.add_constraint_on_task_unary_resource_allocation(
                    task=task,
                    unary_resource=unary_resource,
                    used=ref.is_allocated(task=task, unary_resource=unary_resource),
                )
        return constraints

    @abstractmethod
    def get_nb_tasks_allocated_variable(self) -> Any:
        """Construct and get the variable tracking number of tasks with at least a resource allocated.

        Returns:
            objective variable to minimize

        """
        ...

    @abstractmethod
    def get_nb_unary_resources_used_variable(self) -> Any:
        """Construct and get the variable tracking number of tasks with at least a resource allocated.

        Returns:
            objective variable to minimize

        """
        ...

    def is_compatible_task_unary_resource(
        self, task: Task, unary_resource: UnaryResource
    ) -> bool:
        """Should return False if the unary_resource can never be allocated to task.

        This is only a hint used to reduce the number of variables or constraints generated.
        Default to use `problem.is_compatible_task_unary_resource()`.

        But you can override it if you want to have more constraints in the solver than in the problem.

        """
        return self.problem.is_compatible_task_unary_resource(
            task=task, unary_resource=unary_resource
        )
