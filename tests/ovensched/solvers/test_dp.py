# #  Copyright (c) 2026 AIRBUS and its affiliates.
# #  This source code is licensed under the MIT license found in the
# #  LICENSE file in the root directory of this source tree.
# """Test DP solver for oven scheduling."""
#
# from discrete_optimization.ovensched.solvers.dp import DpOvenSchedulingSolver
#
#
# class TestDPSolver:
#     """Test DpOvenSchedulingSolver if available."""
#
#     pass
#     # @pytest.fixture
#     # def solver_and_result(self, small_problem):
#     #     """Solve once and reuse result for multiple tests."""
#     #     solver = DpOvenSchedulingSolver(problem=small_problem)
#     #     solver.init_model()
#     #     result = solver.solve(time_limit=10)
#     #     return solver, result, small_problem
#
#     # def test_initialization(self, small_problem):
#     #     """Test solver initialization."""
#     #     solver = DpOvenSchedulingSolver(problem=small_problem)
#     #
#     #     assert solver.problem == small_problem
#     #
#     # def test_init_model(self, small_problem):
#     #     """Test init_model method."""
#     #     solver = DpOvenSchedulingSolver(problem=small_problem)
#     #     solver.init_model()
#     #     assert True
#     #     # Should not raise
#     #
#     # def test_solve_returns_result_storage(self, solver_and_result):
#     #     """Test that solve returns ResultStorage."""
#     #     solver, result, problem = solver_and_result
#     #
#     #     assert isinstance(result, ResultStorage)
#     #     assert len(result) > 0
#     #
#     # def test_solve_returns_solution(self, solver_and_result):
#     #     """Test that DP produces a solution."""
#     #     solver, result, problem = solver_and_result
#     #     solution = result.get_best_solution()
#     #
#     #     assert isinstance(solution, OvenSchedulingSolution)
#
#
# def test_dp_vs_greedy(small_problem):
#     """Test DP solver against greedy (needs both solves for comparison)."""
#     from discrete_optimization.ovensched.solvers.greedy import (
#         GreedyOvenSchedulingSolver,
#     )
#
#     # Greedy solution
#     greedy_solver = GreedyOvenSchedulingSolver(problem=small_problem)
#     greedy_result = greedy_solver.solve()
#     greedy_eval = small_problem.evaluate(greedy_result.get_best_solution())
#
#     # DP solution
#     dp_solver = DpOvenSchedulingSolver(problem=small_problem)
#     dp_solver.init_model()
#     dp_result = dp_solver.solve(time_limit=10)
#     dp_eval = small_problem.evaluate(dp_result.get_best_solution())
#
#     # Both should find feasible solutions
#     assert greedy_eval["processing_time"] > 0
#     assert dp_eval["processing_time"] > 0
