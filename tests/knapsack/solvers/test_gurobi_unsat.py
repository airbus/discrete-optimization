#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.lp import GurobiKnapsackSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_knapsack_gurobi_unsat():
    file = [f for f in get_data_available() if "ks_30_0" in f][0]
    problem = parse_file(file)
    solver = GurobiKnapsackSolver(problem=problem)
    solver.solve()
    assert solver.status_solver == StatusSolver.OPTIMAL

    # add impossible constraint "improve optimal bound"
    obj = solver.model.getObjective()
    solver.add_linear_constraint(obj >= 2 * obj.getValue())
    solver.model.update()

    # re-solve => unsatisfiable
    solver.solve()
    assert solver.status_solver == StatusSolver.UNSATISFIABLE

    # explain it
    constraints = solver.explain_unsat_fine()
    assert len(constraints) > 0
