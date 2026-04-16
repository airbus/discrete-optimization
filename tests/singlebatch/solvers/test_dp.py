#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Test DP solver for single batch processing."""

import pytest

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.singlebatch.problem import BatchProcessingSolution
from discrete_optimization.singlebatch.solvers.dp import DpSingleBatchSolver


class TestDPSolver:
    """Test BpmDpSolver."""

    @pytest.fixture
    def solver_and_result(self, tiny_problem):
        """Solve once and reuse result for multiple tests."""
        solver = DpSingleBatchSolver(problem=tiny_problem)
        solver.init_model()
        result = solver.solve(time_limit=10)
        return solver, result, tiny_problem

    def test_inheritance(self):
        """Test that solver inherits from SolverDO."""
        assert issubclass(DpSingleBatchSolver, SolverDO)

    def test_initialization(self, tiny_problem):
        """Test solver initialization."""
        solver = DpSingleBatchSolver(problem=tiny_problem)

        assert solver.problem == tiny_problem

    def test_init_model(self, tiny_problem):
        """Test init_model method."""
        solver = DpSingleBatchSolver(problem=tiny_problem)
        solver.init_model()

        # Check that model was created
        assert solver.model is not None
        assert hasattr(solver, "sorted_indices")

    def test_solve_returns_result_storage(self, solver_and_result):
        """Test that solve returns ResultStorage."""
        solver, result, problem = solver_and_result

        assert isinstance(result, ResultStorage)
        assert len(result) > 0

    def test_solve_returns_solution(self, solver_and_result):
        """Test that DP produces a solution."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()

        assert isinstance(solution, BatchProcessingSolution)

    def test_solution_is_feasible(self, solver_and_result):
        """Test that DP produces feasible solutions."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()

        is_feasible = problem.satisfy(solution)
        assert is_feasible, "Solution must be feasible"

    def test_solution_quality(self, solver_and_result):
        """Test solution quality (makespan is reasonable)."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()
        evaluation = problem.evaluate(solution)

        # Makespan should be <= sum of all processing times
        max_makespan = sum(j.processing_time for j in problem.jobs)
        assert evaluation["makespan"] <= max_makespan
