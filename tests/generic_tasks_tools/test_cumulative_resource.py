#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResourceProblem,
    CumulativeResourceSolution,
)
from discrete_optimization.generic_tasks_tools.renewable_resource import (
    convert_calendar_to_availability_intervals,
)
from discrete_optimization.generic_tools.do_problem import ObjectiveRegister, Solution
from discrete_optimization.generic_tools.encoding_register import EncodingRegister

Resource = str
Task = str


class MyCumulativeResourceProblem(CumulativeResourceProblem[Task, Resource]):
    resource_availabilities = dict(
        R1=[
            (5, 7, 4),
            (2, 3, 1),
        ],
        R2=[(0, 9, 0)],
        R3=[(5, 7, 4), (2, 3, 1), (0, 2, 0)],
        R5=[(1, 3, 3), (3, 5, 2)],
    )
    max_capacities = dict(R1=4, R2=0, R3=4, R5=3)

    consolidated_availabilities = dict(
        R1=[(0, 2, 0), (2, 3, 1), (3, 5, 0), (5, 7, 4), (7, 8, 0)],
        R2=[(0, 8, 0)],
        R3=[(0, 2, 0), (2, 3, 1), (3, 5, 0), (5, 7, 4), (7, 8, 0)],
        R5=[(0, 1, 0), (1, 3, 3), (3, 5, 2), (5, 8, 0)],
    )

    fake_tasks = dict(
        R1=[(0, 2, 4), (2, 3, 3), (3, 5, 4), (7, 8, 4)],
        R2=[],
        R3=[(0, 2, 4), (2, 3, 3), (3, 5, 4), (7, 8, 4)],
        R5=[(0, 1, 3), (3, 5, 1), (5, 8, 3)],
    )

    mode_details = {
        "task-1": {0: {"duration": 4, "R5": 1}, 1: {"duration": 2, "R5": 2}},
        "task-2": {
            0: {"duration": 2, "R5": 1},
        },
    }

    def is_cumulative_resource(self, resource: Resource) -> bool:
        return True

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        return self.mode_details[task][mode]["duration"]

    def get_makespan_upper_bound(self) -> int:
        return 8

    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        return self.resource_availabilities[resource]

    @property
    def renewable_resources_list(self) -> list[Resource]:
        return list(self.resource_availabilities)

    @property
    def tasks_list(self) -> list[Task]:
        return list(self.mode_details)

    def get_resource_consumption(
        self, resource: Resource, task: Task, mode: int
    ) -> int:
        try:
            return self.mode_details[task][mode][resource]
        except KeyError:
            return 0

    def get_task_modes(self, task: Task) -> set[int]:
        return set(self.mode_details[task])

    def evaluate(self, variable: Solution) -> dict[str, float]:
        pass

    def set_fixed_attributes(self, attribute_name: str, solution: Solution) -> None:
        pass

    def satisfy(self, variable: Solution) -> bool:
        pass

    def get_attribute_register(self) -> EncodingRegister:
        pass

    def get_solution_type(self) -> type[Solution]:
        pass

    def get_objective_register(self) -> ObjectiveRegister:
        pass

    def get_dummy_solution(self) -> Solution:
        pass


class MyKOCumulativeResourceProblem(MyCumulativeResourceProblem):
    resource_availabilities = dict(R4=[(5, 7, 4), (2, 3, 1), (0, 3, 0)])


class MyCumulativeResourceSolution(CumulativeResourceSolution[Task, Resource]):
    problem: MyCumulativeResourceProblem

    def __init__(
        self,
        problem: MyCumulativeResourceProblem,
        modes: dict[Task, int],
        starts: dict[Task, int],
    ):
        super().__init__(problem)
        self.modes = modes
        self.starts = starts

    def get_mode(self, task: Task) -> int:
        return self.modes[task]

    def get_end_time(self, task: Task) -> int:
        return (
            self.starts[task]
            + self.problem.mode_details[task][self.modes[task]]["duration"]
        )

    def get_start_time(self, task: Task) -> int:
        return self.starts[task]

    def copy(self) -> Solution:
        pass


def test_cumulative_resource_problem():
    pb = MyCumulativeResourceProblem()
    for resource in pb.renewable_resources_list:
        assert pb.get_resource_max_capacity(resource) == pb.max_capacities[resource]
        expected_result = pb.consolidated_availabilities[resource]
        assert pb.get_resource_consolidated_availabilities(resource) == expected_result
        expected_result = pb.fake_tasks[resource]
        assert pb.get_fake_tasks(resource) == expected_result


def test_cumulative_resource_problem_ko():
    pb = MyKOCumulativeResourceProblem()
    for resource in pb.renewable_resources_list:
        with pytest.raises(ValueError):
            pb.get_resource_consolidated_availabilities(resource)
        with pytest.raises(ValueError):
            pb.get_fake_tasks(resource)


def test_cumulative_resource_solution():
    pb = MyCumulativeResourceProblem()

    # nok
    solution = MyCumulativeResourceSolution(
        problem=pb,
        modes={"task-1": 0, "task-2": 0},
        starts={"task-1": 0, "task-2": 0},
    )
    assert solution.check_resource_capacity_constraint("R1")
    assert not solution.check_resource_capacity_constraint("R5")
    assert not solution.check_all_resource_capacity_constraints()
    solution = MyCumulativeResourceSolution(
        problem=pb,
        modes={"task-1": 1, "task-2": 0},
        starts={"task-1": 2, "task-2": 3},
    )
    assert not solution.check_all_resource_capacity_constraints()
    # ok
    solution = MyCumulativeResourceSolution(
        problem=pb,
        modes={"task-1": 0, "task-2": 0},
        starts={"task-1": 1, "task-2": 2},
    )
    assert solution.check_all_resource_capacity_constraints()
    solution = MyCumulativeResourceSolution(
        problem=pb,
        modes={"task-1": 1, "task-2": 0},
        starts={"task-1": 1, "task-2": 1},
    )
    assert solution.check_all_resource_capacity_constraints()
    solution = MyCumulativeResourceSolution(
        problem=pb,
        modes={"task-1": 1, "task-2": 0},
        starts={"task-1": 1, "task-2": 3},
    )
    assert solution.check_all_resource_capacity_constraints()


def test_convert_calendar_to_availability_intervals():
    assert convert_calendar_to_availability_intervals(
        calendar=[0, 0, 1, 0, 0, 4, 4, 0], horizon=8
    ) == [(0, 2, 0), (2, 3, 1), (3, 5, 0), (5, 7, 4), (7, 8, 0)]
    assert convert_calendar_to_availability_intervals(
        calendar=[0, 0, 1, 0, 0, 4, 4, 0], horizon=9
    ) == [(0, 2, 0), (2, 3, 1), (3, 5, 0), (5, 7, 4), (7, 8, 0)]
    assert convert_calendar_to_availability_intervals(
        calendar=[0, 0, 1, 0, 0, 4, 4, 0], horizon=7
    ) == [(0, 2, 0), (2, 3, 1), (3, 5, 0), (5, 7, 4)]
    assert convert_calendar_to_availability_intervals(calendar=4, horizon=8) == [
        (0, 8, 4)
    ]
