#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Test CP-SAT solver for single batch processing."""

import pytest

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.singlebatch.problem import BatchProcessingSolution
from discrete_optimization.singlebatch.solvers.cpsat import (
    CpSatSingleBatchSolver,
    ModelingBpm,
)


class TestCPSATSolverNaive:
    """Test CP-SAT solver with naive formulation."""

    @pytest.fixture
    def solver_and_result(self, tiny_problem):
        """Solve once and reuse result for multiple tests."""
        solver = CpSatSingleBatchSolver(tiny_problem)
        solver.init_model(modeling=ModelingBpm.BINARY, symmetry_breaking=False)
        result = solver.solve(time_limit=5)
        return solver, result, tiny_problem

    def test_inheritance(self):
        """Test that solver inherits from SolverDO."""
        assert issubclass(CpSatSingleBatchSolver, SolverDO)

    def test_initialization(self, tiny_problem):
        """Test solver initialization."""
        solver = CpSatSingleBatchSolver(tiny_problem)

        assert solver.problem == tiny_problem

    def test_init_model_binary(self, tiny_problem):
        """Test init_model with binary modeling."""
        solver = CpSatSingleBatchSolver(tiny_problem)
        solver.init_model(modeling=ModelingBpm.BINARY, symmetry_breaking=False)

        assert solver.cp_model is not None
        assert "x" in solver.variables

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

    def test_warmstart(self, tiny_problem):
        """Test warm-starting CP-SAT solver (needs 2 solves)."""
        # Get an initial solution
        solver1 = CpSatSingleBatchSolver(tiny_problem)
        solver1.init_model(modeling=ModelingBpm.BINARY)
        result1 = solver1.solve(time_limit=2)
        solution1 = result1.get_best_solution()

        # Warm-start another solver
        solver2 = CpSatSingleBatchSolver(tiny_problem)
        solver2.init_model(modeling=ModelingBpm.BINARY)

        # This should not raise
        solver2.set_warm_start(solution1)

        result2 = solver2.solve(time_limit=2)
        assert len(result2) > 0


class TestCPSATSolverSymmetryBreaking:
    """Test CP-SAT solver with symmetry breaking."""

    @pytest.fixture
    def solver_and_result(self, tiny_problem):
        """Solve once and reuse result for multiple tests."""
        solver = CpSatSingleBatchSolver(tiny_problem)
        solver.init_model(modeling=ModelingBpm.BINARY, symmetry_breaking=True)
        result = solver.solve(time_limit=5)
        return solver, result, tiny_problem

    def test_init_model(self, tiny_problem):
        """Test init_model with symmetry breaking."""
        solver = CpSatSingleBatchSolver(tiny_problem)
        solver.init_model(modeling=ModelingBpm.BINARY, symmetry_breaking=True)

        assert solver.cp_model is not None
        assert solver.symmetry_breaking is True
        assert solver.job_order is not None

    def test_solve_returns_feasible_solution(self, solver_and_result):
        """Test that symmetry breaking produces feasible solutions."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()

        assert isinstance(solution, BatchProcessingSolution)
        assert problem.satisfy(solution)


class TestCPSATSchedulingModeling:
    """Test CP-SAT with scheduling modeling."""

    @pytest.fixture
    def solver_and_result(self, tiny_problem):
        """Solve once and reuse result for multiple tests."""
        solver = CpSatSingleBatchSolver(tiny_problem)
        solver.init_model(modeling=ModelingBpm.SCHEDULING)
        result = solver.solve(time_limit=5)
        return solver, result, tiny_problem

    def test_init_model_scheduling(self, tiny_problem):
        """Test init_model with scheduling modeling."""
        solver = CpSatSingleBatchSolver(tiny_problem)
        solver.init_model(modeling=ModelingBpm.SCHEDULING)

        assert solver.cp_model is not None
        assert "batch_idx" in solver.variables

    def test_solve_returns_feasible_solution(self, solver_and_result):
        """Test that scheduling modeling produces feasible solutions."""
        solver, result, problem = solver_and_result
        solution = result.get_best_solution()

        assert isinstance(solution, BatchProcessingSolution)
        assert problem.satisfy(solution)
