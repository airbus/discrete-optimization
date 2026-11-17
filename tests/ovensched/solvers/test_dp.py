#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Test DP solver for oven scheduling."""

import pytest

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.ovensched import OvenSchedulingSolution
from discrete_optimization.ovensched.solvers.dp import DpOvenSchedulingSolver


class TestDPSolver:
    """Test DpOvenSchedulingSolver if available."""

    @pytest.fixture
    def solver_and_result(self, tiny_problem):
        """Solve once and reuse result for multiple tests."""
        solver = DpOvenSchedulingSolver(problem=tiny_problem)
        solver.init_model()
        result = solver.solve(time_limit=10)
        return solver, result, tiny_problem

    def test_initialization(self, tiny_problem):
        """Test solver initialization."""
        solver = DpOvenSchedulingSolver(problem=tiny_problem)

        assert solver.problem == tiny_problem

    def test_init_model(self, tiny_problem):
        """Test init_model method."""
        solver = DpOvenSchedulingSolver(problem=tiny_problem)
        solver.init_model()
        assert True
        # Should not raise

    def test_solve_returns_result_storage(self, solver_and_result):
        """Test that solve returns ResultStorage."""
        solver, result, problem = solver_and_result

        assert isinstance(result, ResultStorage)
        assert len(result) > 0

    def test_solve_returns_solution(self, solver_and_result):
        """Test that DP produces a solution."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()

        assert isinstance(solution, OvenSchedulingSolution)


def test_dp_vs_greedy(tiny_problem):
    """Test DP solver against greedy (needs both solves for comparison)."""
    from discrete_optimization.ovensched.solvers.greedy import (
        GreedyOvenSchedulingSolver,
    )

    # Greedy solution
    greedy_solver = GreedyOvenSchedulingSolver(problem=tiny_problem)
    greedy_result = greedy_solver.solve()
    greedy_eval = tiny_problem.evaluate(greedy_result.get_best_solution())

    # DP solution
    dp_solver = DpOvenSchedulingSolver(problem=tiny_problem)
    dp_solver.init_model()
    dp_result = dp_solver.solve(time_limit=10)
    dp_eval = tiny_problem.evaluate(dp_result.get_best_solution())

    # Both should find feasible solutions
    assert greedy_eval["processing_time"] > 0
    assert dp_eval["processing_time"] > 0
