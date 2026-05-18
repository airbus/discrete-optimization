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
from discrete_optimization.flex_scheduling.solvers.cpsat import CpSatFlexSolver
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


def test_cpsat_small_problem(random_seed):
    """Test CP-SAT solver on a small generated problem."""
    # Generate a small problem
    generator = FlexProblemGenerator(
        nb_msn=3,  # 3 products
        seed=42,
        tardiness_weight=10,
        wip_weight=1,
        tightness_factor=1.5,
        nb_tools=3,
        nb_stations=10,  # Need at least 7 for large products
    )
    problem = generator.generate()

    assert isinstance(problem, FlexProblem)
    assert problem.nb_tasks > 0
    assert len(problem.resources) > 0

    # Solve with CP-SAT
    solver = CpSatFlexSolver(problem=problem)
    result_storage = solver.solve(
        time_limit=30, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )

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


def test_cpsat_tiny_problem_full_solve(random_seed):
    """Test CP-SAT solver on a tiny problem to find optimal solution."""
    # Generate a very small problem
    generator = FlexProblemGenerator(
        nb_msn=2,  # Only 2 products
        seed=123,
        tardiness_weight=5,
        wip_weight=1,
        tightness_factor=2.0,
        nb_tools=2,
        nb_stations=8,  # Need enough for product routes
    )
    problem = generator.generate()

    # Solve with more time to get better solution
    solver = CpSatFlexSolver(problem=problem)
    result_storage = solver.solve(time_limit=60)

    solution, fit = result_storage.get_best_solution_fit()

    # Verify constraints
    assert problem.satisfy(solution)

    # Check that multiple solutions were found
    assert len(result_storage) > 0

    # Verify all objectives are computed
    evaluation = problem.evaluate(solution)
    assert "makespan" in evaluation


def test_cpsat_with_objectives(random_seed):
    """Test that CP-SAT solver handles multiple objectives."""
    generator = FlexProblemGenerator(
        nb_msn=3,
        seed=456,
        tardiness_weight=20,
        wip_weight=2,
        tightness_factor=1.3,
        nb_tools=4,
        nb_stations=10,  # Need enough for product routes
    )
    problem = generator.generate()

    solver = CpSatFlexSolver(problem=problem)
    result_storage = solver.solve(
        time_limit=30, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )

    solution, fit = result_storage.get_best_solution_fit()
    assert problem.satisfy(solution)

    # Check evaluation includes expected objectives
    evaluation = problem.evaluate(solution)
    assert "makespan" in evaluation

    # Check if tardiness/wip objectives are computed
    if "tardiness" in evaluation:
        assert evaluation["tardiness"] >= 0
    if "wip_cost" in evaluation:
        assert isinstance(evaluation["wip_cost"], (int, float))


def test_cpsat_solution_structure(random_seed):
    """Test that the solution has the correct structure."""
    generator = FlexProblemGenerator(nb_msn=2, seed=789, nb_tools=2, nb_stations=8)
    problem = generator.generate()

    solver = CpSatFlexSolver(problem=problem)
    result_storage = solver.solve(
        time_limit=20, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )

    solution: ScheduleSolution = result_storage.get_best_solution()

    # Check solution structure
    assert solution.schedule.shape[0] == problem.nb_tasks
    assert solution.schedule.shape[1] == 2  # start, end times
    assert solution.modes.shape[0] == problem.nb_tasks

    # Check all tasks are scheduled
    for i in range(problem.nb_tasks):
        start_time = solution.schedule[i, 0]
        end_time = solution.schedule[i, 1]
        assert end_time >= start_time
        assert start_time >= 0
        assert end_time <= problem.horizon
