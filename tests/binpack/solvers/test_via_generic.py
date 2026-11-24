#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Copyright (c) 2026s AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.binpack.solvers.cpsat import CpSatBinPackSolver
from discrete_optimization.binpack.solvers.generic_cp import (
    BinPackAutoCpmpyImplSolver,
    BinPackAutoCpSatImplSolver,
    BinPackAutoOptalImplSolver,
    BinPackGenericAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.transformation import TransformationSolver


@pytest.mark.parametrize(
    "solver_cls",
    [
        BinPackAutoCpSatImplSolver,
        BinPackAutoCpmpyImplSolver,
        BinPackAutoOptalImplSolver,
    ],
)
def test_via_generic_allocation(problem, solver_cls, optal_preview_only):
    kwargs = dict()
    check_solution = True
    if solver_cls == BinPackAutoOptalImplSolver and optal_preview_only:
        kwargs["do_not_retrieve_solutions"] = True
        check_solution = False
    if solver_cls == BinPackAutoCpSatImplSolver:
        kwargs["ortools_cpsat_solver_kwargs"] = {"log_search_progress": True}
    if solver_cls == BinPackAutoCpmpyImplSolver:
        kwargs["log_search_progress"] = True
    parameters_cp = ParametersCp.default()
    specific_solver = CpSatBinPackSolver(problem=problem)
    specific_sol = specific_solver.solve().get_best_solution()
    assert specific_solver.status_solver == StatusSolver.OPTIMAL
    nb_bins_max = 8
    nb_bins_opt = problem.evaluate(specific_sol)["nb_bins"]

    solver: TransformationSolver = solver_cls(problem=problem, nb_bins=nb_bins_max)
    res = solver.solve(parameters_cp=parameters_cp, **kwargs)
    assert solver.status_solver == StatusSolver.OPTIMAL
    if check_solution:
        sol = res[-1][0]
        assert problem.satisfy(sol)
        kpis = problem.evaluate(sol)
        assert kpis["nb_bins"] == nb_bins_opt
    elif solver_cls == BinPackAutoOptalImplSolver:
        assert isinstance(solver.wrapped_solver, BinPackGenericAutoCpSatImplSolver)
        assert (
            solver.wrapped_solver.get_current_best_internal_objective_value()
            == nb_bins_opt
        )
