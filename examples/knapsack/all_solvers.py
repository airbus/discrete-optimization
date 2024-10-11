#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers_map import (
    GurobiKnapsackSolver,
    solve,
    solvers_map,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


def main_run():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_40_0" in f][0]
    knapsack_problem = parse_file(file)
    solvers = solvers_map.keys()
    for s in solvers:
        logging.info(f"Solver {s}")
        if s == GurobiKnapsackSolver:
            # skip Except if you have a licence
            continue
        results = solve(method=s, problem=knapsack_problem, **solvers_map[s][1])
        s, f = results.get_best_solution_fit()
        logging.info(f"sol={s}")
        logging.info(f"fitness={f}")


if __name__ == "__main__":
    main_run()
