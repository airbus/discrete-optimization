# #  Copyright (c) 2026 AIRBUS and its affiliates.
# #  This source code is licensed under the MIT license found in the
# #  LICENSE file in the root directory of this source tree.
# """Test CP-SAT solver for oven scheduling."""
#
# import pytest
#
# from discrete_optimization.generic_tools.do_solver import SolverDO
# from discrete_optimization.generic_tools.result_storage.result_storage import (
#     ResultStorage,
# )
# from discrete_optimization.ovensched.problem import OvenSchedulingSolution
# from discrete_optimization.ovensched.solvers.cpsat import OvenSchedulingCpSatSolver
#
#
# class TestCpSatSolver:
#     """Test OvenSchedulingCpSatSolver."""
#
#     @pytest.fixture
#     def solver_and_result(self, tiny_problem):
#         """Solve once and reuse result for multiple tests."""
#         solver = OvenSchedulingCpSatSolver(
#             problem=tiny_problem, max_nb_batch_per_machine=5
#         )
#         solver.init_model()
#         result = solver.solve(time_limit=10)
#         return solver, result, tiny_problem
#
#     def test_inheritance(self):
#         """Test that solver inherits from SolverDO."""
#         assert issubclass(OvenSchedulingCpSatSolver, SolverDO)
#
#     def test_initialization(self, tiny_problem):
#         """Test solver initialization."""
#         solver = OvenSchedulingCpSatSolver(problem=tiny_problem)
#
#         assert solver.problem == tiny_problem
#         assert hasattr(solver, "max_nb_batch")
#
#     def test_init_model(self, tiny_problem):
#         """Test init_model method."""
#         solver = OvenSchedulingCpSatSolver(
#             problem=tiny_problem, max_nb_batch_per_machine=5
#         )
#         solver.init_model()
#
#         # Check that model was created
#         assert solver.cp_model is not None
#         assert "task_start" in solver.variables
#         assert "task_end" in solver.variables
#         assert "batch_present" in solver.variables
#
#     def test_solve_returns_result_storage(self, solver_and_result):
#         """Test that solve returns ResultStorage."""
#         solver, result, problem = solver_and_result
#
#         assert isinstance(result, ResultStorage)
#         assert len(result) > 0
#
#     def test_solve_returns_solution(self, solver_and_result):
#         """Test that CP-SAT produces a solution."""
#         solver, result, problem = solver_and_result
#         solution = result.get_best_solution()
#
#         assert isinstance(solution, OvenSchedulingSolution)
#         assert len(solution.schedule_per_machine) == problem.n_machines
#
#         # Verify all tasks are scheduled
#         all_tasks = set()
#         for machine_schedule in solution.schedule_per_machine.values():
#             for batch in machine_schedule:
#                 all_tasks.update(batch.tasks)
#
#         assert all_tasks == set(range(problem.n_jobs)), "All tasks must be scheduled"
#
#     def test_solution_is_feasible(self, solver_and_result):
#         """Test that CP-SAT produces feasible solutions."""
#         solver, result, problem = solver_and_result
#         solution = result.get_best_solution()
#
#         # Check feasibility
#         is_feasible = problem.satisfy(solution)
#         assert is_feasible, "Solution must be feasible"
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
#         assert "nb_late_jobs" in evaluation
#         assert "setup_cost" in evaluation
#
#     def test_batch_capacity_respected(self, solver_and_result):
#         """Test that batch capacity constraints are respected."""
#         solver, result, problem = solver_and_result
#         solution = result.get_best_solution()
#
#         # Check capacity for each batch
#         for machine_idx, batches in solution.schedule_per_machine.items():
#             machine_capacity = problem.machines_data[machine_idx].capacity
#             for batch in batches:
#                 total_size = sum(problem.tasks_data[t].size for t in batch.tasks)
#                 assert total_size <= machine_capacity, (
#                     f"Batch exceeds capacity: {total_size} > {machine_capacity}"
#                 )
#
#     def test_batch_attributes_consistent(self, solver_and_result):
#         """Test that tasks in same batch have same attribute."""
#         solver, result, problem = solver_and_result
#         solution = result.get_best_solution()
#
#         # Check that all tasks in a batch have the same attribute
#         for batches in solution.schedule_per_machine.values():
#             for batch in batches:
#                 attributes = [problem.tasks_data[t].attribute for t in batch.tasks]
#                 assert len(set(attributes)) == 1, (
#                     "All tasks in a batch must have the same attribute"
#                 )
#                 assert attributes[0] == batch.task_attribute, (
#                     "Batch attribute must match task attributes"
#                 )
