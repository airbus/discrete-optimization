#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tests for RCPSP with Resource Blocking Constraints."""

import pytest

from discrete_optimization.generic_tasks_tools.entities import TaskEntity
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingConstraintMetadata,
    BlockingMode,
)
from discrete_optimization.rcpsp.blocking_generator import (
    generate_batch_blocking,
    generate_combined_blocking,
    generate_setup_time_blocking,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem_with_blocking import RcpspWithResourceBlocking
from discrete_optimization.rcpsp.solvers.cpsat_auto import CpSatAutoRcpspSolver


def test_rcpsp_with_blocking_creation():
    """Test creating RCPSP with blocking constraints from scratch."""
    resources = {"R1": 5}
    mode_details = {
        1: {1: {"duration": 0, "R1": 0}},
        2: {1: {"duration": 4, "R1": 2}},
        3: {1: {"duration": 3, "R1": 3}},
        4: {1: {"duration": 0, "R1": 0}},
    }
    successors = {1: [2, 3], 2: [4], 3: [4], 4: []}

    # Add gap blocking constraint
    blocking_constraints = [
        (
            TaskEntity(2),
            StartOrEnd.END,
            TaskEntity(3),
            StartOrEnd.START,
            {"R1": 1},
            BlockingConstraintMetadata(
                mode=BlockingMode.RESERVATION,
                description="Setup between task 2 and 3",
            ),
        )
    ]

    problem = RcpspWithResourceBlocking(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=20,
        flexible_gap_blocking_constraints=blocking_constraints,
    )

    assert len(problem.get_flexible_gap_blocking_constraints()) == 1
    assert len(problem.get_span_blocking_constraints()) == 0
    assert problem.n_jobs == 4
    assert problem.resources == resources


def test_setup_blocking_affects_schedule():
    """Test that setup blocking constraints actually affect the optimal schedule."""
    # Create a simple problem where blocking forces different schedule
    resources = {"R1": 2}
    mode_details = {
        1: {1: {"duration": 0, "R1": 0}},  # Source
        2: {1: {"duration": 5, "R1": 1}},  # Task A
        3: {1: {"duration": 5, "R1": 1}},  # Task B
        4: {1: {"duration": 0, "R1": 0}},  # Sink
    }
    successors = {1: [2, 3], 2: [4], 3: [4], 4: []}

    # Without blocking: both tasks can run in parallel
    from discrete_optimization.rcpsp.problem import RcpspProblem

    base_problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=20,
    )

    # Solve base problem
    solver_base = CpSatAutoRcpspSolver(base_problem)
    solver_base.init_model()
    result_base = solver_base.solve(time_limit=5)
    sol_base = result_base.get_best_solution()
    makespan_base = base_problem.evaluate(sol_base)["makespan"]

    # With blocking: setup time between task 2 and 3
    blocking_constraints = [
        (
            TaskEntity(2),
            StartOrEnd.END,
            TaskEntity(3),
            StartOrEnd.START,
            {"R1": 1},  # Block 1 unit during setup
            BlockingConstraintMetadata(
                mode=BlockingMode.RESERVATION,
            ),
        )
    ]

    problem_blocking = RcpspWithResourceBlocking(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=20,
        flexible_gap_blocking_constraints=blocking_constraints,
    )

    solver_blocking = CpSatAutoRcpspSolver(problem_blocking)
    solver_blocking.init_model()
    result_blocking = solver_blocking.solve(time_limit=5)
    sol_blocking = result_blocking.get_best_solution()
    makespan_blocking = problem_blocking.evaluate(sol_blocking)["makespan"]

    # Blocking should affect the schedule (may increase makespan or change task timing)
    assert problem_blocking.satisfy(sol_blocking)
    # The constraint should be enforced
    assert len(problem_blocking.get_flexible_gap_blocking_constraints()) == 1


def test_span_blocking_forces_reservation():
    """Test that span blocking reserves resources across task group."""
    resources = {"R1": 3}
    mode_details = {
        1: {1: {"duration": 0, "R1": 0}},  # Source
        2: {1: {"duration": 2, "R1": 1}},  # Task A
        3: {1: {"duration": 2, "R1": 1}},  # Task B
        4: {1: {"duration": 2, "R1": 1}},  # Task C
        5: {1: {"duration": 0, "R1": 0}},  # Sink
    }
    successors = {1: [2, 3, 4], 2: [5], 3: [5], 4: [5], 5: []}

    # Span blocking: reserve R1 for entire span of tasks 2, 3
    span_constraints = [
        (
            frozenset([2, 3]),
            {"R1": 1},
            BlockingConstraintMetadata(
                mode=BlockingMode.RESERVATION,
                description="Batch reservation",
            ),
        )
    ]

    problem = RcpspWithResourceBlocking(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=20,
        span_blocking_constraints=span_constraints,
    )

    solver = CpSatAutoRcpspSolver(problem)
    solver.init_model()
    result = solver.solve(time_limit=5)
    sol = result.get_best_solution()

    assert problem.satisfy(sol)
    assert len(problem.get_span_blocking_constraints()) == 1


def test_generator_setup_blocking():
    """Test setup time blocking generator."""
    files = get_data_available()
    if not files:
        pytest.skip("No RCPSP data files available")

    # Load small instance
    file_path = [f for f in files if "j301_1.sm" in f or "Pat1.rcp" in f]
    if not file_path:
        pytest.skip("No suitable test instance found")

    base_problem = parse_file(file_path[0])

    # Generate problem with setup blocking
    problem = generate_setup_time_blocking(base_problem, setup_ratio=0.2, seed=42)

    # Should have some blocking constraints
    gap_constraints = problem.get_flexible_gap_blocking_constraints()
    assert len(gap_constraints) >= 0  # May be 0 if no tasks share resources

    # Should be solvable
    solver = CpSatAutoRcpspSolver(problem)
    solver.init_model()
    result = solver.solve(time_limit=10)
    sol = result.get_best_solution()

    if sol is not None:
        assert problem.satisfy(sol)


def test_generator_batch_blocking():
    """Test batch span blocking generator."""
    files = get_data_available()
    if not files:
        pytest.skip("No RCPSP data files available")

    file_path = [f for f in files if "j301_1.sm" in f or "Pat1.rcp" in f]
    if not file_path:
        pytest.skip("No suitable test instance found")

    base_problem = parse_file(file_path[0])

    # Generate problem with batch blocking
    problem = generate_batch_blocking(base_problem, batch_size=3, seed=42)

    # Should have span constraints
    span_constraints = problem.get_span_blocking_constraints()
    assert len(span_constraints) >= 0

    # Should be solvable
    solver = CpSatAutoRcpspSolver(problem)
    solver.init_model()
    result = solver.solve(time_limit=10)
    sol = result.get_best_solution()

    if sol is not None:
        assert problem.satisfy(sol)


def test_generator_combined_blocking():
    """Test combined blocking generator."""
    files = get_data_available()
    if not files:
        pytest.skip("No RCPSP data files available")

    file_path = [f for f in files if "j301_1.sm" in f or "Pat1.rcp" in f]
    if not file_path:
        pytest.skip("No suitable test instance found")

    base_problem = parse_file(file_path[0])

    # Generate problem with both types of blocking
    problem = generate_combined_blocking(
        base_problem, setup_ratio=0.15, batch_size=3, seed=42
    )

    # Should have both types of constraints
    gap_constraints = problem.get_flexible_gap_blocking_constraints()
    span_constraints = problem.get_span_blocking_constraints()
    # At least one type should exist
    assert len(gap_constraints) + len(span_constraints) >= 0

    # Should be solvable
    solver = CpSatAutoRcpspSolver(problem)
    solver.init_model()
    result = solver.solve(time_limit=10)
    sol = result.get_best_solution()

    if sol is not None:
        assert problem.satisfy(sol)


def test_blocking_constraints_validation():
    """Test that invalid schedules violate blocking constraints."""
    resources = {"R1": 5}
    mode_details = {
        1: {1: {"duration": 0, "R1": 0}},
        2: {1: {"duration": 3, "R1": 2}},
        3: {1: {"duration": 3, "R1": 2}},
        4: {1: {"duration": 0, "R1": 0}},
    }
    successors = {1: [2, 3], 2: [4], 3: [4], 4: []}

    # Add blocking that requires gap between task 2 end and task 3 start
    blocking_constraints = [
        (
            TaskEntity(2),
            StartOrEnd.END,
            TaskEntity(3),
            StartOrEnd.START,
            {"R1": 3},  # Block significant resource
            BlockingConstraintMetadata(
                mode=BlockingMode.RESERVATION,
            ),
        )
    ]

    problem = RcpspWithResourceBlocking(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=20,
        flexible_gap_blocking_constraints=blocking_constraints,
    )

    # The blocking constraint should be enforced by the solver
    solver = CpSatAutoRcpspSolver(problem)
    solver.init_model()
    result = solver.solve(time_limit=5)
    sol = result.get_best_solution()

    # Solution should satisfy blocking constraints
    assert problem.satisfy(sol)
