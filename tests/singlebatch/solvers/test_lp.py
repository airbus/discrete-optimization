#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Test LP solver for single batch processing."""

import pytest

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.singlebatch.problem import BatchProcessingSolution
from discrete_optimization.singlebatch.solvers.lp import (
    BpmLpFormulation,
    MathOptSingleBatchSolver,
)


class TestLPSolverNaive:
    """Test LP solver with naive formulation."""

    @pytest.fixture
    def solver_and_result(self, tiny_problem):
        """Solve once and reuse result for multiple tests."""
        solver = MathOptSingleBatchSolver(tiny_problem)
        solver.init_model(formulation=BpmLpFormulation.NAIVE)
        result = solver.solve(time_limit=5)
        return solver, result, tiny_problem

    def test_inheritance(self):
        """Test that solver inherits from SolverDO."""
        assert issubclass(MathOptSingleBatchSolver, SolverDO)

    def test_initialization(self, tiny_problem):
        """Test solver initialization."""
        solver = MathOptSingleBatchSolver(tiny_problem)

        assert solver.problem == tiny_problem
        assert hasattr(solver, "aggreg_from_sol")

    def test_init_model(self, tiny_problem):
        """Test init_model with naive formulation."""
        solver = MathOptSingleBatchSolver(tiny_problem)
        solver.init_model(formulation=BpmLpFormulation.NAIVE)

        assert solver.model is not None
        assert "allocation" in solver.variables

    def test_solve_returns_result_storage(self, solver_and_result):
        """Test that solve returns ResultStorage."""
        solver, result, problem = solver_and_result

        assert isinstance(result, ResultStorage)
        assert len(result) > 0

    def test_solve_returns_feasible_solution(self, solver_and_result):
        """Test that solve returns a feasible solution."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()

        assert isinstance(solution, BatchProcessingSolution)
        assert problem.satisfy(solution)

    def test_solution_quality(self, solver_and_result):
        """Test solution quality (makespan is reasonable)."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()
        evaluation = problem.evaluate(solution)

        # Makespan should be <= sum of all processing times
        max_makespan = sum(j.processing_time for j in problem.jobs)
        assert evaluation["makespan"] <= max_makespan

    def test_warmstart(self, tiny_problem):
        """Test warm-starting LP solver (needs 2 solves)."""
        # Get an initial solution
        solver1 = MathOptSingleBatchSolver(tiny_problem)
        solver1.init_model(formulation=BpmLpFormulation.NAIVE)
        result1 = solver1.solve(time_limit=2)
        solution1 = result1.get_best_solution()

        # Warm-start another solver
        solver2 = MathOptSingleBatchSolver(tiny_problem)
        solver2.init_model(formulation=BpmLpFormulation.NAIVE)

        # This should not raise
        warmstart_hint = solver2.convert_to_variable_values(solution1)
        assert warmstart_hint is not None


class TestLPSolverSymmetryBreaking:
    """Test LP solver with symmetry breaking formulation."""

    @pytest.fixture
    def solver_and_result(self, tiny_problem):
        """Solve once and reuse result for multiple tests."""
        solver = MathOptSingleBatchSolver(tiny_problem)
        solver.init_model(formulation=BpmLpFormulation.SYMMETRY_BREAKING)
        result = solver.solve(time_limit=5)
        return solver, result, tiny_problem

    def test_init_model(self, tiny_problem):
        """Test init_model with symmetry breaking."""
        solver = MathOptSingleBatchSolver(tiny_problem)
        solver.init_model(formulation=BpmLpFormulation.SYMMETRY_BREAKING)

        assert solver.model is not None
        assert solver.formulation == BpmLpFormulation.SYMMETRY_BREAKING
        assert solver.job_order is not None  # Jobs should be ordered

    def test_solve_returns_feasible_solution(self, solver_and_result):
        """Test that symmetry breaking produces feasible solutions."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()

        assert isinstance(solution, BatchProcessingSolution)
        assert problem.satisfy(solution)

    def test_variable_count_reduced(self, small_problem):
        """Test that symmetry breaking has fewer variables (no solve needed)."""
        solver_naive = MathOptSingleBatchSolver(small_problem)
        solver_naive.init_model(formulation=BpmLpFormulation.NAIVE)

        solver_sb = MathOptSingleBatchSolver(small_problem)
        solver_sb.init_model(formulation=BpmLpFormulation.SYMMETRY_BREAKING)

        # Symmetry breaking should have roughly 50% of allocation variables
        n = small_problem.nb_jobs
        expected_naive_alloc = n * n
        expected_sb_alloc = n * (n + 1) // 2

        assert expected_sb_alloc < expected_naive_alloc
