#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.dp import DpKnapsackSolver, dp
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver


def test_dp():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_problem = parse_file(files[0])
    solver = DpKnapsackSolver(problem=knapsack_problem)
    results = solver.solve(solver=dp.LNBS, time_limit=10)
    sol, fit = results.get_best_solution_fit()
    assert knapsack_problem.satisfy(sol)


def test_dp_ws():
    file = [f for f in get_data_available() if "ks_100_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = GreedyBestKnapsackSolver(problem=knapsack_problem)
    sol, fit = solver.solve().get_best_solution_fit()
    print(fit, " current sol")
    solver = DpKnapsackSolver(problem=knapsack_problem)
    solver.init_model(float_cost=True, dual_bound=True)
    solver.set_warm_start(solution=sol)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        retrieve_intermediate_solutions=True,
        solver=dp.LNBS,
        time_limit=100,
    )
    sol_ = res.get_best_solution()
    assert sol.list_taken == sol_.list_taken
    assert knapsack_problem.satisfy(sol_)
