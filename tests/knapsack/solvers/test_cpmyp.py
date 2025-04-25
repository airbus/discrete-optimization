#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from cpmpy.solvers.solver_interface import ExitStatus, SolverInterface, SolverStatus

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackSolution
from discrete_optimization.knapsack.solvers.cpmpy import CpmpyKnapsackSolver


def test_knapsack_cpmyp():
    file = [f for f in get_data_available() if "ks_30_0" in f][0]
    problem = parse_file(file)
    solver = CpmpyKnapsackSolver(problem=problem)
    solver.init_model()
    res = solver.solve(solver="ortools", time_limit=20)
    sol = res.get_best_solution()
    assert isinstance(sol, KnapsackSolution)
    assert problem.satisfy(sol)

    assert solver.cpm_status.exitstatus == ExitStatus.OPTIMAL

    # add impossible constraint "improve optimal bound"
    values = [item.value for item in problem.list_items]
    solver.model += [
        sum(solver.variables["x"] * values)
        > sum(solver.variables["x"] * values).value()
    ]

    # re-solve => unsatisfiable
    solver.solve()
    assert solver.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE

    # explain it
    constraints = solver.explain_unsat()
    assert len(constraints) > 0
