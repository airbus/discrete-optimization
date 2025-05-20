#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from cpmpy.solvers.solver_interface import ExitStatus

from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackSolution
from discrete_optimization.knapsack.solvers.cpmpy import CpmpyKnapsackSolver


def test_knapsack_cpmpy():
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
    constraints = solver.explain_unsat_fine()
    assert len(constraints) > 0

    with pytest.raises(NotImplementedError):
        solver.explain_unsat_meta()

    # correct it
    constraints = solver.correct_unsat_fine()
    assert len(constraints) > 0

    with pytest.raises(NotImplementedError):
        solver.correct_unsat_meta()

    # NB: solver.model.constraints.remove(cstr) not working as expected
    constraints_ids = {id(c) for c in constraints}
    solver.model.constraints = [
        c for c in solver.model.constraints if id(c) not in constraints_ids
    ]
    solver.solve()
    assert solver.status_solver == StatusSolver.OPTIMAL
