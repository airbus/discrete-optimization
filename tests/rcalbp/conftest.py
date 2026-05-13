#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pytest_cases import fixture

from discrete_optimization.alb.base.problem import ResourceTaskData
from discrete_optimization.alb.rcalbp.problem import RCALBPProblem


@fixture
def simple_problem():
    """Create a simple RC-ALBP problem for testing."""
    # 4 tasks with precedences: 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
    tasks_data = [
        ResourceTaskData(
            task_id="T1", processing_time=5, resource_consumption={"R1": 2, "R2": 1}
        ),
        ResourceTaskData(
            task_id="T2", processing_time=3, resource_consumption={"R1": 1, "R2": 2}
        ),
        ResourceTaskData(
            task_id="T3", processing_time=4, resource_consumption={"R1": 1, "R2": 1}
        ),
        ResourceTaskData(
            task_id="T4", processing_time=2, resource_consumption={"R1": 2, "R2": 1}
        ),
    ]

    precedences = [("T1", "T2"), ("T1", "T3"), ("T2", "T4"), ("T3", "T4")]
    stations = ["WS1", "WS2"]
    resources = ["R1", "R2"]

    station_resources = {
        "WS1": {"R1": 3, "R2": 2},
        "WS2": {"R1": 2, "R2": 3},
    }

    return RCALBPProblem(
        tasks_data=tasks_data,
        precedences=precedences,
        stations=stations,
        resources=resources,
        station_resources=station_resources,
    )


@fixture
def shared_resource_problem():
    """Create RC-ALBP problem with shared resources."""
    tasks_data = [
        ResourceTaskData(
            task_id="T1", processing_time=4, resource_consumption={"R1": 1, "AGV": 1}
        ),
        ResourceTaskData(
            task_id="T2", processing_time=3, resource_consumption={"R1": 2, "AGV": 1}
        ),
        ResourceTaskData(
            task_id="T3", processing_time=2, resource_consumption={"R2": 1, "AGV": 1}
        ),
    ]

    precedences = [("T1", "T2")]
    stations = ["WS1", "WS2"]
    resources = ["R1", "R2"]  # Station-specific resources only

    station_resources = {
        "WS1": {"R1": 2, "R2": 1},
        "WS2": {"R1": 1, "R2": 2},
    }

    shared_resources = {"AGV"}  # Shared resources separate
    shared_resource_capacities = {"AGV": 1}  # Only 1 AGV available globally

    return RCALBPProblem(
        tasks_data=tasks_data,
        precedences=precedences,
        stations=stations,
        resources=resources,
        station_resources=station_resources,
        shared_resources=shared_resources,
        shared_resource_capacities=shared_resource_capacities,
    )
