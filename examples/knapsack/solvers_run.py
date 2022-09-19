#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import time

import matplotlib.pyplot as plt
from mip import GRB

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
)
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyDummy

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


def main_run():
    file = [f for f in get_data_available() if "ks_60_0" in f][0]
    knapsack_model = parse_file(file)
    methods = solvers.keys()
    methods = ["cp"]
    for method in methods:
        print("method : ", method)
        for submethod in solvers[method]:
            if submethod[0] == LPKnapsackGurobi and not gurobi_available:
                continue
            print(submethod[0])
            t = time.time()
            solution = solve(submethod[0], knapsack_model, **submethod[1])
            print(time.time() - t, " seconds to solve")
            print("Solution : ", solution[0])


def run_lp():
    file = [f for f in get_data_available() if "ks_10000_0" in f][0]
    knapsack_model = parse_file(file)
    pymip_solver = LPKnapsack(knapsack_model, milp_solver_name=MilpSolverName.CBC)

    pymip_solver.init_model()
    parameters_milp = ParametersMilp.default()
    solutions = pymip_solver.solve(parameters_milp=parameters_milp)
    print(solutions.get_best_solution())


if __name__ == "__main__":
    run_lp()
