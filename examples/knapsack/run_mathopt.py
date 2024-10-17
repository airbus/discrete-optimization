#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Playing with the new Mathopt api from ortools"""

from ortools.math_opt.python import mathopt

from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.lp import MathOptKnapsackSolver


def run_mathopt_knapsack():
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = MathOptKnapsackSolver(problem=knapsack_problem)
    solver.init_model()
    res = solver.solve(
        time_limit=30,
        mathopt_enable_output=True,
        mathopt_solver_type=mathopt.SolverType.GSCIP,
    )
    print(solver.termination)
    sol = res.get_best_solution_fit()[0]
    print(sol)


if __name__ == "__main__":
    run_mathopt_knapsack()
