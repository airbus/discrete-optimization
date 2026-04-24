#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging

from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResourceProblem,
    NonRenewableResourceSolution,
)
from discrete_optimization.generic_tools.do_problem import ObjectiveRegister, Solution

NonRenewableResource = str
Task = str


class MyNonRenewableResourceProblem(
    NonRenewableResourceProblem[Task, NonRenewableResource]
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
    NonRenewableResourceSolution[Task, NonRenewableResource]
):
    problem: MyNonRenewableResourceProblem

    def __init__(
        self,
        problem: MyNonRenewableResourceProblem,
        modes: dict[Task, int],
    ):
        super().__init__(problem)
        self.modes = modes

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
