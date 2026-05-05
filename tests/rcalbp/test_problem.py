#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.alb.rcalbp.problem import RCALBPSolution


def test_problem_structure(simple_problem):
    """Test basic problem structure."""
    assert simple_problem.nb_tasks == 4
    assert simple_problem.nb_stations == 2
    assert len(simple_problem.resources) == 2
    assert len(simple_problem.precedences) == 4


def test_task_data_access(simple_problem):
    """Test accessing task data."""
    task_data = simple_problem.get_task_data("T1")
    assert task_data.task_id == "T1"
    assert task_data.processing_time == 5
    assert task_data.get_resource_consumption("R1") == 2
    assert task_data.get_resource_consumption("R2") == 1


def test_resource_queries(simple_problem):
    """Test resource-related queries."""
    assert simple_problem.get_task_demand("T1", "R1") == 2
    assert simple_problem.get_task_demand("T2", "R2") == 2
    assert simple_problem.get_station_capacity("WS1", "R1") == 3
    assert simple_problem.get_station_capacity("WS2", "R2") == 3


def test_shared_resources(shared_resource_problem):
    """Test shared resource configuration."""
    assert "AGV" in shared_resource_problem.shared_resources
    assert shared_resource_problem.shared_resource_capacities["AGV"] == 1
    assert len(shared_resource_problem.get_station_specific_resources()) == 2


def test_solution_evaluation(simple_problem):
    """Test solution evaluation."""
    # Create a valid solution
    task_assignment = {"T1": "WS1", "T2": "WS1", "T3": "WS2", "T4": "WS2"}
    task_schedule = {"T1": 0, "T2": 5, "T3": 0, "T4": 4}
    cycle_time = 10

    solution = RCALBPSolution(
        problem=simple_problem,
        task_assignment=task_assignment,
        task_schedule=task_schedule,
        cycle_time=cycle_time,
    )

    eval_result = simple_problem.evaluate(solution)
    assert "cycle_time" in eval_result
    assert "penalty_precedence" in eval_result
    assert "penalty_resource_station" in eval_result


def test_precedence_graph(simple_problem):
    """Test precedence graph structure."""
    successors = simple_problem.get_successors()
    assert "T2" in successors["T1"]
    assert "T3" in successors["T1"]
    assert "T4" in successors["T2"]
    assert "T4" in successors["T3"]

    first_tasks = simple_problem.get_first_tasks()
    assert "T1" in first_tasks

    last_tasks = simple_problem.get_last_tasks()
    assert "T4" in last_tasks
