# #  Copyright (c) 2026 AIRBUS and its affiliates.
# #  This source code is licensed under the MIT license found in the
# #  LICENSE file in the root directory of this source tree.
# """Test local search solvers for oven scheduling."""
#
# from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
# from discrete_optimization.generic_tools.ls.hill_climber import HillClimber
# from discrete_optimization.generic_tools.ls.local_search import (
#     ModeMutation,
#     RestartHandlerLimit,
# )
# from discrete_optimization.generic_tools.ls.simulated_annealing import (
#     SimulatedAnnealing,
#     TemperatureSchedulingFactor,
# )
# from discrete_optimization.ovensched.solvers.mutations import (
#     OvenPermutationMixedMutation,
# )
#
#
# def test_sa_ovensched(small_problem):
#     """Test simulated annealing on oven scheduling."""
#     solution = small_problem.get_dummy_solution()
#     mutation = OvenPermutationMixedMutation(small_problem)
#     res = RestartHandlerLimit(1000)
#     sa = SimulatedAnnealing(
#         problem=small_problem,
#         mutator=mutation,
#         restart_handler=res,
#         temperature_handler=TemperatureSchedulingFactor(100, res, 0.95),
#         mode_mutation=ModeMutation.MUTATE,
#     )
#     result = sa.solve(
#         initial_variable=solution,
#         nb_iteration_max=10,
#         callbacks=[TimerStopper(total_seconds=5, check_nb_steps=5)],
#     )
#     sol, fit = result.get_best_solution_fit()
#     # Note: SA may not always produce feasible solutions
#     assert sol is not None
#
#
# def test_hc_ovensched(small_problem):
#     """Test hill climber on oven scheduling."""
#     solution = small_problem.get_dummy_solution()
#     mutation = OvenPermutationMixedMutation(small_problem)
#     res = RestartHandlerLimit(1000)
#     hc = HillClimber(
#         problem=small_problem,
#         mutator=mutation,
#         restart_handler=res,
#         mode_mutation=ModeMutation.MUTATE,
#     )
#     result = hc.solve(
#         initial_variable=solution,
#         nb_iteration_max=10,
#         callbacks=[TimerStopper(total_seconds=5, check_nb_steps=5)],
#     )
#     sol, fit = result.get_best_solution_fit()
#     # Note: HC may not always produce feasible solutions
#     assert sol is not None
