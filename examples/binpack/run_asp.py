#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
    SubBrick,
)


def run_asp():
    f = [ff for ff in get_data_available_bppc() if "BPPC_1_1_10.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    solver = AspBinPackingSolver(problem=problem)
    solver.init_model(upper_bound=450)
    res = solver.solve(
        time_limit=20,
    )
    sol = res[-1][0]
    print("Status", solver.status_solver)
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_asp()
