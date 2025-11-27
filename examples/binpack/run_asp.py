#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
from discrete_optimization.binpack.solvers.greedy import (
    GreedyBinPackOpenEvolve,
)

logging.basicConfig(level=logging.INFO)


def run_asp():
    f = [ff for ff in get_data_available_bppc() if "BPPC_8_9_10.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)

    solver = GreedyBinPackOpenEvolve(problem)
    sol = solver.solve().get_best_solution()
    nb_bins = problem.evaluate(sol)["nb_bins"]
    print("Solution from Greedy", problem.evaluate(sol))
    solver = AspBinPackingSolver(problem=problem)
    solver.init_model(upper_bound=nb_bins, solution=sol)
    res = solver.solve(
        time_limit=20,
    )
    sol = res[-1][0]
    print("Status", solver.status_solver)
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_asp()
