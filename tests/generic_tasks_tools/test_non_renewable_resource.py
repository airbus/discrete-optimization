#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging

from discrete_optimization.generic_tasks_tools.allocation import (
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResourceProblem,
    NonRenewableResourceSolution,
    NonRenewableResourceWithoutAllocationProblem,
    NonRenewableResourceWithoutAllocationSolution,
)
from discrete_optimization.generic_tools.do_problem import ObjectiveRegister, Solution

NonRenewableResource = str
Task = str


class MyNonRenewableResourceProblem(
    NonRenewableResourceWithoutAllocationProblem[Task, NonRenewableResource]
):
    resource_capacities = {"R0": 2, "R1": 5}
    mode_details = {
        "task-1": {0: {"R0": 2}, 1: {"R1": 3}},
        "task-2": {
            0: {"R0": 2, "R1": 1},
        },
    }

    @property
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        return list(self.resource_capacities)

    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        return self.resource_capacities[resource]

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        return self.mode_details[task][mode].get(resource, 0)

    def get_task_modes(self, task: Task) -> set[int]:
        return set(self.mode_details[task])

    @property
    def tasks_list(self) -> list[Task]:
        return list(self.mode_details)

    def evaluate(self, variable: Solution) -> dict[str, float]:
        pass

    def satisfy(self, variable: MyNonRenewableResourceSolution) -> bool:
        return variable.check_all_non_renewable_resource_capacity_constraints()

    def get_solution_type(self) -> type[Solution]:
        return MyNonRenewableResourceSolution

    def get_objective_register(self) -> ObjectiveRegister:
        pass


class MyNonRenewableResourceSolution(
    NonRenewableResourceWithoutAllocationSolution[Task, NonRenewableResource],
):
    problem: MyNonRenewableResourceProblem

    def __init__(
        self,
        problem: MyNonRenewableResourceProblem,
        modes: dict[Task, int],
    ):
        super().__init__(problem)
        self.modes = modes

    def is_present(self, task: Task) -> bool:
        return task in self.modes

    def get_mode(self, task: Task) -> int:
        return self.modes[task]

    def copy(self) -> Solution:
        pass


class MyNonRenewableResourceWithUnaryProblem(
    NonRenewableResourceProblem[Task, NonRenewableResource, UnaryResource]
):
    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return list(self.unary)

    def get_nr_resource_consumption_when_unary_resource_allocated(
        self,
        task: Task,
        mode: int,
        resource: NonRenewableResource,
        unary_resource: UnaryResource,
    ):
        if task in self.unary_nr_consumption:
            if mode in self.unary_nr_consumption[task]:
                if resource in self.unary_nr_consumption[task][mode]:
                    if (
                        unary_resource
                        in self.unary_nr_consumption[task][mode][resource]
                    ):
                        return self.unary_nr_consumption[task][mode][resource][
                            unary_resource
                        ]
        return 0

    resource_capacities = {"R0": 2, "R1": 5}
    mode_details = {
        "task-1": {0: {"R0": 2}, 1: {"R1": 3}},
        "task-2": {
            0: {"R0": 2, "R1": 1},
        },
    }
    unary = {"worker-1", "worker-2"}
    unary_nr_consumption = {
        "task-1": {
            0: {
                "R0": {"worker-1": 0, "worker-2": 1},
                "R1": {"worker-1": 1, "worker-2": 0},
            },
            1: {
                "R0": {"worker-1": 0, "worker-2": 0},
                "R1": {"worker-1": 0, "worker-2": 2},
            },
        }
    }

    @property
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        return list(self.resource_capacities)

    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        return self.resource_capacities[resource]

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        return self.mode_details[task][mode].get(resource, 0)

    def get_task_modes(self, task: Task) -> set[int]:
        return set(self.mode_details[task])

    @property
    def tasks_list(self) -> list[Task]:
        return list(self.mode_details)

    def evaluate(self, variable: Solution) -> dict[str, float]:
        pass

    def satisfy(self, variable: MyNonRenewableResourceWithUnarySolution) -> bool:
        return variable.check_all_non_renewable_resource_capacity_constraints()

    def get_solution_type(self) -> type[Solution]:
        return MyNonRenewableResourceWithUnarySolution

    def get_objective_register(self) -> ObjectiveRegister:
        pass


class MyNonRenewableResourceWithUnarySolution(
    NonRenewableResourceSolution[Task, NonRenewableResource, UnaryResource],
):
    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        if task in self.allocated:
            return unary_resource in self.allocated[task]
        return False

    problem: MyNonRenewableResourceProblem

    def __init__(
        self,
        problem: MyNonRenewableResourceProblem,
        modes: dict[Task, int],
        allocated: dict[Task, set[UnaryResource]],
    ):
        super().__init__(problem)
        self.modes = modes
        self.allocated = allocated

    def is_present(self, task: Task) -> bool:
        return task in self.modes

    def get_mode(self, task: Task) -> int:
        return self.modes[task]

    def copy(self) -> Solution:
        pass


def test_non_renewable_resource_check(caplog):
    pb = MyNonRenewableResourceProblem()
    # ok
    solution = MyNonRenewableResourceSolution(
        problem=pb,
        modes={"task-1": 1, "task-2": 0},
    )
    assert pb.satisfy(solution)
    # nok
    solution = MyNonRenewableResourceSolution(
        problem=pb,
        modes={"task-1": 0, "task-2": 0},
    )
    with caplog.at_level(logging.DEBUG):
        assert not pb.satisfy(solution)
    assert "R0" in caplog.text
    assert "R1" not in caplog.text


def test_non_renewable_with_unary_resource_check(caplog):
    pb = MyNonRenewableResourceWithUnaryProblem()
    # ok
    solution = MyNonRenewableResourceWithUnarySolution(
        problem=pb, modes={"task-1": 1, "task-2": 0}, allocated={}
    )
    d = solution.compute_non_renewable_resources_consumptions()
    assert pb.satisfy(solution)
    # nok
    solution = MyNonRenewableResourceWithUnarySolution(
        problem=pb, modes={"task-1": 1, "task-2": 0}, allocated={"task-1": {"worker-2"}}
    )
    d = solution.compute_non_renewable_resources_consumptions()
    with caplog.at_level(logging.DEBUG):
        assert not pb.satisfy(solution)
    assert "R0" not in caplog.text
    assert "R1" in caplog.text
