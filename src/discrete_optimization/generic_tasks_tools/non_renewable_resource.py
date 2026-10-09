#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Hashable, Iterable
from functools import reduce
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tasks_tools.allocation import (
    MultimodeAllocationProblem,
    MultiModeAllocationSolution,
    NoUnaryResource,
    UnaryResource,
    WithoutAllocationProblem,
    WithoutAllocationSolution,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.utils import optional_override

logger = logging.getLogger(__name__)

NonRenewableResource = TypeVar("NonRenewableResource", bound=Hashable)


class NonRenewableResourceProblem(
    MultimodeAllocationProblem[Task, UnaryResource],
    Generic[Task, NonRenewableResource, UnaryResource],
):
    """Base class for problems dealing with non-renewable resources consumed by tasks.
    Just like CumulativeResourceProblem, it supports two consumption modes:

    1. **Standard**: Task consumption is fixed by task mode.
       Example: Task A in mode 1 always consumes 5 units.

    2. **Resource-dependent**: Task consumption depends on other tasks' modes.
       Modeled via a consumption mapping.
       If the task/mode dont depend on any other task, returns empty condition
       with the static resource need.
    3. **Resource via unary resource allocation** : when given unary resource is allocated
       it consumes a given nr resource.

    """

    @property
    @abstractmethod
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        """Non-renewable resources used by the tasks."""
        ...

    @abstractmethod
    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        """Get resource max capacity

        Args:
            resource:

        Returns:

        """
        ...

    @abstractmethod
    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        """Get resource consumption of the task in the given mode

        Args:
            resource: non-renewable resource
            task:
            mode: not used for single mode problems

        Returns:.

        Raises:
            ValueError: if resource consumption is depending on other variables than mode

        """
        ...

    @optional_override
    def get_non_renewable_resource_consumption_mapping(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> dict[frozenset[tuple[Task, int]], int]:
        return {
            frozenset(): self.get_non_renewable_resource_consumption(
                resource, task, mode
            )
        }

    def get_possible_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> set[int]:
        if self.is_non_renewable_resource_task_mode_consumption_dependent(
            resource=resource, task=task, mode=mode
        ):
            return set(
                self.get_non_renewable_resource_consumption_mapping(
                    resource=resource, task=task, mode=mode
                ).values()
            )
        return {
            self.get_non_renewable_resource_consumption(
                resource=resource, task=task, mode=mode
            )
        }

    def get_possible_non_renewable_resource_consumption_all_modes(
        self, resource: NonRenewableResource, task: Task
    ) -> set[int]:
        return reduce(
            lambda prev, y: prev.union(
                self.get_possible_non_renewable_resource_consumption(
                    resource=resource, task=task, mode=y
                )
            ),
            list(self.get_task_modes(task)),
            set(),
        )

    @optional_override
    def is_non_renewable_resource_task_mode_consumption_dependent(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> bool:
        return False

    def is_non_renewable_resource_task_consumption_dependent(
        self, resource: NonRenewableResource, task: Task
    ):
        return any(
            self.is_non_renewable_resource_task_mode_consumption_dependent(
                resource=resource, task=task, mode=mode
            )
            for mode in self.get_task_modes(task)
        )

    def is_task_non_renewable_consumption_dependent(self, task: Task):
        return any(
            self.is_non_renewable_resource_task_consumption_dependent(
                resource=resource, task=task
            )
            for resource in self.non_renewable_resources_list
        )

    def has_any_non_renewable_consumption_dependent(self):
        return any(
            self.is_task_non_renewable_consumption_dependent(task)
            for task in self.tasks_list
        )

    @optional_override
    def get_nr_resource_consumption_when_unary_resource_allocated(
        self,
        task: Task,
        mode: int,
        resource: NonRenewableResource,
        unary_resource: UnaryResource,
    ):
        return 0

    def has_any_nr_resource_consumption_depend_on_unary_resource(self):
        return any(
            len(self.get_non_zero_mode_nr_res_unary(task)) > 0
            for task in self.tasks_list
        )

    def get_tasks_of_interest_for_nr_resource(self, resource: NonRenewableResource):
        return set(
            [
                t
                for t in self.tasks_list
                if any(
                    self.get_nr_resource_consumption_when_unary_resource_allocated(
                        task=t, mode=m, resource=resource, unary_resource=ur
                    )
                    > 0
                    for m in self.get_task_modes(t)
                    for ur in self.unary_resources_list
                )
            ]
        )

    def get_non_zero_mode_nr_res_unary(
        self, task: Task
    ) -> list[tuple[int, NonRenewableResource, UnaryResource]]:
        return [
            (mode, res, unary)
            for mode in self.get_task_modes(task)
            for res in self.non_renewable_resources_list
            for unary in self.unary_resources_list
            if self.get_nr_resource_consumption_when_unary_resource_allocated(
                task, mode, res, unary
            )
            > 0
        ]


class NonRenewableResourceSolution(
    MultiModeAllocationSolution[Task, UnaryResource],
    Generic[Task, NonRenewableResource, UnaryResource],
):
    problem: NonRenewableResourceProblem[Task, NonRenewableResource, UnaryResource]

    def get_non_renewable_resource_consumption_from_mapping(
        self, resource: NonRenewableResource, task: Task
    ) -> int:
        mode = self.get_mode(task)
        mapping = self.problem.get_non_renewable_resource_consumption_mapping(
            resource=resource, task=task, mode=mode
        )
        set_of_tasks = set([frozenset([t for t, m in k]) for k in mapping])
        value = next(
            (
                mapping[key_mapping]
                for set_task in set_of_tasks
                if (key_mapping := frozenset([(t, self.get_mode(t)) for t in set_task]))
                in mapping
            ),
            None,
        )
        if value is None:
            logger.info(f"No found mapping on resource {resource} and task {task}")
            return 0
        return value

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task
    ) -> int:
        """Get resource consumption by given task.

        Args:
            resource:
            task:

        Returns:

        """
        if self.is_present(task):
            if not self.problem.is_non_renewable_resource_task_mode_consumption_dependent(
                resource=resource, task=task, mode=self.get_mode(task)
            ):
                return self.problem.get_non_renewable_resource_consumption(
                    resource=resource, task=task, mode=self.get_mode(task)
                )
            else:
                return self.get_non_renewable_resource_consumption_from_mapping(
                    resource=resource, task=task
                )
        else:
            return 0

    def get_non_renewable_resource_consumption_by_unary_allocation(
        self, task: Task, resource: NonRenewableResource
    ):
        if not self.is_present(task):
            return 0
        mode = self.get_mode(task)
        allocated = self.get_task_allocation(task)
        value = 0
        for unary in allocated:
            value += (
                self.problem.get_nr_resource_consumption_when_unary_resource_allocated(
                    task=task, mode=mode, resource=resource, unary_resource=unary
                )
            )
        return value

    def check_non_renewable_resource_capacity_constraint(
        self, resource: NonRenewableResource
    ) -> bool:
        """Check capacity constraint on given renewable resource."""
        return self.check_non_renewable_resource_capacity_constraints(
            resources=(resource,)
        )

    def check_non_renewable_resource_capacity_constraints(
        self, resources: Iterable[NonRenewableResource]
    ):
        resources_consumption = {resource: 0 for resource in resources}
        for task in self.get_present_tasks():
            for resource in resources:
                resources_consumption[resource] += (
                    self.get_non_renewable_resource_consumption(
                        resource=resource, task=task
                    )
                )
                resources_consumption[resource] += (
                    self.get_non_renewable_resource_consumption_by_unary_allocation(
                        task=task, resource=resource
                    )
                )
        resources_capa_violation = {
            resource: conso
            > self.problem.get_non_renewable_resource_capacity(resource=resource)
            for resource, conso in resources_consumption.items()
        }
        if any(resources_capa_violation.values()):
            logger.debug("Violations on non-renewable resource capacities:")
            for resource, violation in resources_capa_violation.items():
                if violation:
                    logger.debug(f"resource '{resource}'")
            return False
        else:
            return True

    def check_all_non_renewable_resource_capacity_constraints(self) -> bool:
        """Check capacity constraint on all renewable resources."""
        return self.check_non_renewable_resource_capacity_constraints(
            resources=self.problem.non_renewable_resources_list
        )

    def compute_non_renewable_resources_consumptions(
        self,
    ) -> dict[NonRenewableResource, int]:
        """Compute total consumption of each non-renewable resource by the solution."""
        return {
            resource: sum(
                self.get_non_renewable_resource_consumption(
                    resource=resource, task=task
                )
                + self.get_non_renewable_resource_consumption_by_unary_allocation(
                    resource=resource, task=task
                )
                for task in self.get_present_tasks()
            )
            for resource in self.problem.non_renewable_resources_list
        }

    def compute_aggregated_non_renewable_resources_consumptions(
        self, weights: Optional[dict[NonRenewableResource, int]] = None
    ):
        """Compute aggregated consumption of each non-renewable resource by the solution.

        Args:
            weights: optional weights to apply to each resource in the sum. Default to 1.

        """
        if weights is None:
            weights = {}
        return sum(
            conso * weights.get(resource, 1)
            for resource, conso in self.compute_non_renewable_resources_consumptions().items()
        )

    def compute_nb_non_renewable_resources_used(
        self, weights: Optional[dict[NonRenewableResource, int]] = None
    ) -> int:
        """Compute number of non-renewable resources used by at least one task.

        Args:
            weights: optional weights to apply to each resource in the sum. Default to 1.


        Returns:

        """
        if weights is None:
            weights = {}
        return sum(
            (conso > 0) * weights.get(resource, 1)
            for resource, conso in self.compute_non_renewable_resources_consumptions().items()
        )


NoNonRenewableResource = None


class NonRenewableResourceWithoutAllocationProblem(
    NonRenewableResourceProblem[Task, NonRenewableResource, NoUnaryResource],
    WithoutAllocationProblem[Task],
):
    pass


class NonRenewableResourceWithoutAllocationSolution(
    NonRenewableResourceSolution[Task, NonRenewableResource, NoUnaryResource],
    WithoutAllocationSolution[Task],
):
    pass


class WithoutNonRenewableResourceProblem(
    NonRenewableResourceProblem[Task, NoNonRenewableResource, UnaryResource],
    Generic[Task, UnaryResource],
):
    """Mixin for problem without non-renewable resources.

    To be used has an additional mixin with generic `GenericSchedulingProblem`.

    """

    @property
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        return []

    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        raise RuntimeError("This problem has no non-renewable resource.")

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        raise RuntimeError("This problem has no non-renewable resource.")


class WithoutNonRenewableResourceSolution(
    NonRenewableResourceSolution[Task, NoNonRenewableResource, UnaryResource],
    Generic[Task, UnaryResource],
):
    """Mixin for solution without non-renewable resources.

    To be used has an additional mixin with generic `GenericSchedulingSolution`.

    """

    ...

    def check_non_renewable_resource_capacity_constraint(
        self, resource: NonRenewableResource
    ) -> bool:
        return True

    def check_non_renewable_resource_capacity_constraints(
        self, resources: Iterable[NonRenewableResource]
    ):
        return True

    def check_all_non_renewable_resource_capacity_constraints(self) -> bool:
        return True

    def compute_non_renewable_resources_consumptions(
        self,
    ) -> dict[NonRenewableResource, int]:
        return {}

    def compute_aggregated_non_renewable_resources_consumptions(
        self, weights: Optional[dict[NonRenewableResource, int]] = None
    ):
        return 0

    def compute_nb_non_renewable_resources_used(
        self, weights: Optional[dict[NonRenewableResource, int]] = None
    ) -> int:
        return 0
