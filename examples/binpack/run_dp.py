#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import didppy as dp

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.binpack.solvers.dp import DpBinPackSolver, ModelingDpBinPack
from discrete_optimization.generic_tools.cp_tools import ParametersCp

logging.basicConfig(level=logging.INFO)


def run_dp():
    f = [ff for ff in get_data_available_bppc() if "BPPC_2_2_1.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    problem.has_constraint = False
    solver = DpBinPackSolver(problem=problem)
    solver.init_model(
        upper_bound=min(500, problem.nb_items),
        modeling=ModelingDpBinPack.ASSIGN_ITEM_BINS,
    )
    res = solver.solve(time_limit=20, solver=dp.CABS, threads=10)
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_dp()
