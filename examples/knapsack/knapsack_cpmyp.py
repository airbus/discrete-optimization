#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.lp_tools import MilpSolverName
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.knapsack_cpmpy import (
    CPMPYKnapsackSolver,
    SolverLookup,
)
from discrete_optimization.knapsack.solvers.lp_solvers import LPKnapsack, ParametersMilp


def run():
    logging.basicConfig(level=logging.DEBUG)
    file = [f for f in get_data_available() if "ks_60_0" in f][0]
    knapsack_model = parse_file(file)
    # solver_lp = LPKnapsack(knapsack_model=knapsack_model, milp_solver_name=MilpSolverName.CBC)
    # solver_lp.init_model()
    # res = solver_lp.solve(parameters_milp=ParametersMilp.default())
    # sol_lp = res.get_best_solution()
    a = SolverLookup.base_solvers()
    print(SolverLookup.base_solvers())
    solver = CPMPYKnapsackSolver(knapsack_model=knapsack_model)
    solver.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 20
    res = solver.solve(solver="ortools", parameters_cp=parameters_cp)
    sol = res.get_best_solution()
    print(knapsack_model.satisfy(sol))
    print(knapsack_model.max_capacity)
    print(sol, "\n", sol)


if __name__ == "__main__":
    run()
