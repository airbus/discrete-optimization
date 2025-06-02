#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import didppy as dp

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver

logging.basicConfig(level=logging.INFO)


def run_greedy():
    f = [ff for ff in get_data_available_bppc() if "BPPC_2_2_1.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    solver = GreedyBinPackSolver(problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_greedy()
