#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ortools.math_opt.python import mathopt

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.lp import MathOptBinPackSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lp_tools import ParametersMilp


def run_mathopt():
    f = [ff for ff in get_data_available_bppc() if "BPPC_1_2_5.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    solver = MathOptBinPackSolver(problem=problem)
    solver.init_model(upper_bound=150)
    p = ParametersMilp.default()
    res = solver.solve(
        parameters_milp=p,
        mathopt_solver_type=mathopt.SolverType.GSCIP,
        mathopt_additional_solve_parameters=mathopt.SolveParameters(threads=10),
        mathopt_enable_output=True,
        time_limit=20,
    )
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_mathopt()
