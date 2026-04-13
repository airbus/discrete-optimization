# #  Copyright (c) 2026 AIRBUS and its affiliates.
# #  This source code is licensed under the MIT license found in the
# #  LICENSE file in the root directory of this source tree.
# """Test OptalCP solver for oven scheduling."""
#
# import pytest
#
# from discrete_optimization.generic_tools.do_solver import SolverDO
# from discrete_optimization.ovensched.solvers.optal import (
#     OvenSchedulingOptalSolver,
#     optalcp_available,
# )
#
#
# @pytest.mark.skipif(not optalcp_available, reason="OptalCP not available")
# class TestOptalSolver:
#     """Test OvenSchedulingOptalSolver."""
#
#     @pytest.fixture
#     def solver_and_result(self, tiny_problem):
#         """Solve once and reuse result for multiple tests."""
#         solver = OvenSchedulingOptalSolver(
#             problem=tiny_problem, max_nb_batch_per_machine=5
#         )
#         solver.init_model()
#         result = solver.solve(time_limit=10)
#         return solver, result, tiny_problem
#
#     def test_inheritance(self):
#         """Test that solver inherits from SolverDO."""
#         assert issubclass(OvenSchedulingOptalSolver, SolverDO)
#
#     def test_initialization(self, tiny_problem):
#         """Test solver initialization."""
#         solver = OvenSchedulingOptalSolver(problem=tiny_problem)
#
#         assert solver.problem == tiny_problem
#         assert hasattr(solver, "max_nb_batch")
#
#     def test_init_model(self, tiny_problem):
#         """Test init_model method."""
#         solver = OvenSchedulingOptalSolver(
#             problem=tiny_problem, max_nb_batch_per_machine=5
#         )
#         solver.init_model()
#
#         # Check that model was created
#         assert solver.cp_model is not None
#         assert "intervals_per_machines" in solver.variables
#         assert "interval_job" in solver.variables
