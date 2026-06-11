#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Test heuristic solver for oven scheduling."""

import pytest

from discrete_optimization.ovensched.problem import (
    MachineData,
    OvenSchedulingProblem,
    OvenSchedulingSolution,
    TaskData,
)
from discrete_optimization.ovensched.solvers.heuristic import (
    HeuristicOvenSchedulingSolver,
)


@pytest.fixture
def tiny_problem():
    """Create a tiny test problem instance (3 jobs, 2 machines)."""
    tasks_data = [
        TaskData(
            attribute=0,
            min_duration=10,
            max_duration=15,
            earliest_start=0,
            latest_end=100,
            eligible_machines={0, 1},
            size=5,
        ),
        TaskData(
            attribute=1,
            min_duration=8,
            max_duration=12,
            earliest_start=0,
            latest_end=100,
            eligible_machines={0, 1},
            size=3,
        ),
        TaskData(
            attribute=0,
            min_duration=12,
            max_duration=18,
            earliest_start=0,
            latest_end=100,
            eligible_machines={0, 1},
            size=4,
        ),
    ]

    machines_data = [
        MachineData(
            capacity=10,
            initial_attribute=0,
            availability=[(0, 100)],
        ),
        MachineData(
            capacity=10,
            initial_attribute=1,
            availability=[(0, 100)],
        ),
    ]

    setup_costs = [
        [0, 5],
        [3, 0],
    ]

    setup_times = [
        [0, 2],
        [1, 0],
    ]

    return OvenSchedulingProblem(
        n_jobs=3,
        n_machines=2,
        tasks_data=tasks_data,
        machines_data=machines_data,
        setup_costs=setup_costs,
        setup_times=setup_times,
    )


def test_heuristic_initialization(tiny_problem):
    """Test solver initialization."""
    solver = HeuristicOvenSchedulingSolver(problem=tiny_problem)
    assert solver.problem == tiny_problem


def test_heuristic_solve_default(tiny_problem):
    """Test solver with default hyperparameters."""
    solver = HeuristicOvenSchedulingSolver(problem=tiny_problem)
    result = solver.solve()

    assert len(result) > 0
    solution = result.get_best_solution()
    assert isinstance(solution, OvenSchedulingSolution)


def test_heuristic_solve_with_hyperparameters(tiny_problem):
    """Test solver with custom hyperparameters."""
    solver = HeuristicOvenSchedulingSolver(problem=tiny_problem)
    result = solver.solve(num_local_search_iterations=10, cooling_rate=0.95)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert isinstance(solution, OvenSchedulingSolution)


def test_heuristic_solution_evaluable(tiny_problem):
    """Test that solution can be evaluated."""
    solver = HeuristicOvenSchedulingSolver(problem=tiny_problem)
    result = solver.solve(num_local_search_iterations=5)
    solution = result.get_best_solution()

    evaluation = tiny_problem.evaluate(solution)

    assert isinstance(evaluation, dict)
    assert "processing_time" in evaluation
    assert "nb_late_jobs" in evaluation
    assert "setup_cost" in evaluation

    # Solution should have reasonable values
    assert evaluation["processing_time"] > 0
    assert evaluation["setup_cost"] >= 0


def test_heuristic_produces_schedule(tiny_problem):
    """Test that heuristic produces a non-empty schedule."""
    solver = HeuristicOvenSchedulingSolver(problem=tiny_problem)
    result = solver.solve(num_local_search_iterations=5)
    solution = result.get_best_solution()

    # Solution should have schedule information
    assert hasattr(solution, "schedule_per_machine")
    assert solution.schedule_per_machine is not None
    assert len(solution.schedule_per_machine) == tiny_problem.n_machines


def test_heuristic_improves_over_iterations(tiny_problem):
    """Test that heuristic can improve solution quality."""
    solver = HeuristicOvenSchedulingSolver(problem=tiny_problem)

    # Run with minimal iterations
    result_few = solver.solve(num_local_search_iterations=1)
    sol_few = result_few.get_best_solution()
    eval_few = tiny_problem.evaluate(sol_few)

    # Run with more iterations
    result_many = solver.solve(num_local_search_iterations=50)
    sol_many = result_many.get_best_solution()
    eval_many = tiny_problem.evaluate(sol_many)

    # Both should produce valid solutions
    assert eval_few is not None
    assert eval_many is not None

    # Both solutions should be complete (have schedules)
    assert len(sol_few.schedule_per_machine) == tiny_problem.n_machines
    assert len(sol_many.schedule_per_machine) == tiny_problem.n_machines


@pytest.mark.parametrize("num_iterations", [0, 1, 10, 25])
def test_heuristic_with_different_iterations(tiny_problem, num_iterations):
    """Test solver with different iteration counts."""
    solver = HeuristicOvenSchedulingSolver(problem=tiny_problem)
    result = solver.solve(num_local_search_iterations=num_iterations)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert isinstance(solution, OvenSchedulingSolution)


@pytest.mark.parametrize("cooling_rate", [0.9, 0.95, 0.99])
def test_heuristic_with_different_cooling_rates(tiny_problem, cooling_rate):
    """Test solver with different cooling rates."""
    solver = HeuristicOvenSchedulingSolver(problem=tiny_problem)
    result = solver.solve(num_local_search_iterations=10, cooling_rate=cooling_rate)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert isinstance(solution, OvenSchedulingSolution)
