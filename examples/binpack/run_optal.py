#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.optal import OptalBinPackSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp


def run_optal():
    f = [ff for ff in get_data_available_bppc() if "BPPC_4_1_1.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    solver = OptalBinPackSolver(problem=problem)
    solver.init_model(upper_bound=450)
    p = ParametersCp.default_cpsat()
    p.nb_process = 6
    res = solver.solve(
        parameters_cp=p,
        time_limit=100,
    )
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    # best:414   next:[309,414]


if __name__ == "__main__":
    run_optal()
