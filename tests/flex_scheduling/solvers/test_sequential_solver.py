#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pytest

from discrete_optimization.flex_scheduling.generator import FlexProblemGenerator
from discrete_optimization.flex_scheduling.problem import (
    FlexProblem,
    ScheduleSolution,
)
from discrete_optimization.flex_scheduling.solvers.sequential_solver import (
    SequentialFlexSolver,
)


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


def test_sequential_solver_basic(random_seed):
    """Test SequentialFlexSolver on a small problem."""
    generator = FlexProblemGenerator(
        nb_msn=3,  # 3 products
        seed=42,
        tardiness_weight=10,
        earliness_weight=1,
        tightness_factor=1.5,
        nb_tools=3,
        nb_stations=10,  # Need at least 7 for large products
    )
    problem = generator.generate()

    assert isinstance(problem, FlexProblem)
    assert problem.nb_tasks > 0

    # Solve with sequential solver using small batches and short time limit
    solver = SequentialFlexSolver(problem=problem)
    result_storage = solver.solve(nb_batches=2, time_limit_per_batch=15.0, nb_process=4)

    # Check we got a solution
    assert len(result_storage) > 0
    solution, fit = result_storage.get_best_solution_fit()
    assert isinstance(solution, ScheduleSolution)

    # Verify solution satisfies constraints
    assert problem.satisfy(solution)

    # Verify evaluation
    evaluation = problem.evaluate(solution)
    assert "makespan" in evaluation
    assert evaluation["makespan"] > 0


def test_sequential_solver_tiny_problem(random_seed):
    """Test SequentialFlexSolver on a very small problem."""
    generator = FlexProblemGenerator(
        nb_msn=2,  # Only 2 products
        seed=123,
        tardiness_weight=5,
        earliness_weight=1,
        tightness_factor=2.0,
        nb_tools=2,
        nb_stations=8,  # Need enough for product routes
    )
    problem = generator.generate()

    solver = SequentialFlexSolver(problem=problem)
    result_storage = solver.solve(nb_batches=1, time_limit_per_batch=20.0, nb_process=4)

    assert len(result_storage) > 0
    solution = result_storage.get_best_solution()

    # Verify constraints
    assert problem.satisfy(solution)

    # Check solution structure
    assert solution.schedule.shape[0] == problem.nb_tasks
    assert solution.modes.shape[0] == problem.nb_tasks


def test_sequential_solver_multiple_batches(random_seed):
    """Test SequentialFlexSolver with multiple batches."""
    generator = FlexProblemGenerator(nb_msn=4, seed=456, nb_tools=4, nb_stations=10)
    problem = generator.generate()

    solver = SequentialFlexSolver(problem=problem)
    # Use 3 batches with short time per batch
    result_storage = solver.solve(nb_batches=3, time_limit_per_batch=10.0, nb_process=4)

    assert len(result_storage) > 0
    solution = result_storage.get_best_solution()

    # Verify solution is valid
    assert problem.satisfy(solution)

    # All tasks should be scheduled
    for i in range(problem.nb_tasks):
        assert solution.schedule[i, 0] >= 0
        assert solution.schedule[i, 1] >= solution.schedule[i, 0]


def test_sequential_solver_vs_single_batch(random_seed):
    """Compare sequential solver with single batch (equivalent to direct solve)."""
    generator = FlexProblemGenerator(nb_msn=2, seed=111, nb_tools=2, nb_stations=8)
    problem = generator.generate()

    # Single batch should work like a regular solve
    solver = SequentialFlexSolver(problem=problem)
    result_storage = solver.solve(nb_batches=1, time_limit_per_batch=15.0, nb_process=4)

    assert len(result_storage) > 0
    solution = result_storage.get_best_solution()
    assert problem.satisfy(solution)
