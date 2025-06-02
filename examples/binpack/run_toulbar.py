#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
from discrete_optimization.binpack.solvers.toulbar import ToulbarBinPackSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)


def run_toulbar():
    f = [ff for ff in get_data_available_bppc() if "BPPC_1_9_7.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    solver = ToulbarBinPackSolver(problem=problem)
    solver.init_model()
    print("Init done")
    res = solver.solve(time_limit=20)
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


def run_toulbar_ws():
    f = [ff for ff in get_data_available_bppc() if "BPPC_1_9_7.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    sequential_solver = SequentialMetasolver(
        problem=problem,
        list_subbricks=[
            SubBrick(GreedyBinPackSolver, {}),
            SubBrick(
                ToulbarBinPackSolver,
                kwargs=dict(time_limit=100, vns=2),
                kwargs_from_solution={"upper_bound": lambda sol: max(sol.allocation)},
            ),
        ],
    )
    res = sequential_solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_toulbar_ws()
