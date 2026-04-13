# #  Copyright (c) 2026 AIRBUS and its affiliates.
# #  This source code is licensed under the MIT license found in the
# #  LICENSE file in the root directory of this source tree.
# """Test greedy solver for oven scheduling."""
#
# import pytest
#
# from discrete_optimization.generic_tools.do_solver import SolverDO
# from discrete_optimization.generic_tools.result_storage.result_storage import (
#     ResultStorage,
# )
# from discrete_optimization.ovensched.problem import OvenSchedulingSolution
# from discrete_optimization.ovensched.solvers.greedy import GreedyOvenSchedulingSolver
#
#
# class TestGreedySolver:
#     """Test GreedyOvenSchedulingSolver."""
#
#     @pytest.fixture
#     def solver_and_result(self, small_problem):
#         """Solve once and reuse result for multiple tests."""
#         solver = GreedyOvenSchedulingSolver(problem=small_problem)
#         result = solver.solve()
#         return solver, result, small_problem
#
#     def test_inheritance(self):
#         """Test that solver inherits from SolverDO."""
#         assert issubclass(GreedyOvenSchedulingSolver, SolverDO)
#
#     def test_initialization(self, small_problem):
#         """Test solver initialization."""
#         solver = GreedyOvenSchedulingSolver(problem=small_problem)
#
#         assert solver.problem == small_problem
#         assert hasattr(solver, "aggreg_from_sol")
#
#     def test_solve_without_init_model(self, solver_and_result):
#         """Test that greedy can solve without explicit init_model."""
#         solver, result, problem = solver_and_result
#
#         assert isinstance(result, ResultStorage)
#         assert len(result) > 0
#
#     def test_solve_returns_solution(self, solver_and_result):
#         """Test that greedy produces a solution."""
#         solver, result, problem = solver_and_result
#         solution = result.get_best_solution()
#
#         assert isinstance(solution, OvenSchedulingSolution)
#
#         # Note: Greedy may not always produce feasible solutions on all instances
#         # This is expected behavior for a fast heuristic
#
#     def test_solution_is_evaluable(self, solver_and_result):
#         """Test that solution can be evaluated."""
#         solver, result, problem = solver_and_result
#         solution = result.get_best_solution()
#
#         evaluation = problem.evaluate(solution)
#
#         assert isinstance(evaluation, dict)
#         assert "processing_time" in evaluation
#         assert "setup_cost" in evaluation
#
#     def test_greedy_is_deterministic(self, small_problem):
#         """Test that greedy produces same result each time (needs 2 calls)."""
#         solver = GreedyOvenSchedulingSolver(problem=small_problem)
#
#         result1 = solver.solve()
#         result2 = solver.solve()
#
#         sol1 = result1.get_best_solution()
#         sol2 = result2.get_best_solution()
#
#         eval1 = small_problem.evaluate(sol1)
#         eval2 = small_problem.evaluate(sol2)
#
#         # Should be identical (greedy is deterministic)
#         assert eval1 == eval2
#
#     def test_greedy_produces_schedule(self, solver_and_result):
#         """Test that greedy produces a non-empty schedule."""
#         solver, result, problem = solver_and_result
#         solution = result.get_best_solution()
#
#         # Solution should have schedule information
#         assert hasattr(solution, "schedule_per_machine")
#         assert solution.schedule_per_machine is not None
#
#     def test_greedy_finds_reasonable_solution(self, solver_and_result):
#         """Test that greedy finds a reasonable solution."""
#         solver, result, problem = solver_and_result
#         solution = result.get_best_solution()
#         evaluation = problem.evaluate(solution)
#
#         # Solution should have non-zero processing time and finite setup cost
#         assert evaluation["processing_time"] > 0
#         assert evaluation["setup_cost"] >= 0
#         assert evaluation["setup_cost"] < float("inf")
