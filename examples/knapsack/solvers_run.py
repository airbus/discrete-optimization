#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.lp_tools import MilpSolverName
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.knapsack_solvers import (
    CPKnapsackMZN2,
    LPKnapsack,
    LPKnapsackGurobi,
    ParametersMilp,
    solve,
    solvers,
    solvers_map,
)
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyDummy

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


def main_run():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_40_0" in f][0]
    knapsack_model = parse_file(file)
    solvers = solvers_map.keys()
    for s in solvers:
        logging.info(f"Solver {s}")
        if s == LPKnapsackGurobi:
            # skip Except if you have a licence
            continue
        results = solve(method=s, problem=knapsack_model, **solvers_map[s][1])
        s, f = results.get_best_solution_fit()
        logging.info(f"sol={s}")
        logging.info(f"fitness={f}")


def run_lp():
    file = [f for f in get_data_available() if "ks_10000_0" in f][0]
    knapsack_model = parse_file(file)
    pymip_solver = LPKnapsack(knapsack_model, milp_solver_name=MilpSolverName.CBC)

    pymip_solver.init_model()
    parameters_milp = ParametersMilp.default()
    solutions = pymip_solver.solve(parameters_milp=parameters_milp)
    print(solutions.get_best_solution())


if __name__ == "__main__":
    main_run()
