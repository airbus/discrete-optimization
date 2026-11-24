#  Copyright (c) 2026s AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.coloring.solvers.cpsat import CpSatColoringSolver
from discrete_optimization.coloring.solvers.generic_cp import (
    ColoringAutoCpmpyImplSolver,
    ColoringAutoCpSatImplSolver,
    ColoringAutoOptalImplSolver,
    ColoringGenericAutoCpSatImplSolver,
)
from discrete_optimization.coloring.transformations.to_generic import (
    ColoringToGenericSchedulingTransformation,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver


def test_via_generic_scheduling(
    problem, generic_solver_cls, generic_solver_kwargs, generic_solver_check_solution
):
    p = ParametersCp.default()
    solver = TransformationSolver(
        transformation=ColoringToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            generic_solver_cls,
            {
                "time_limit": 20,
                "parameters_cp": p,
                **generic_solver_kwargs,
            },
        ),
        source_problem=problem,
    )
    res = solver.solve()
    if generic_solver_check_solution:
        sol = res[-1][0]
        assert problem.satisfy(sol)


@pytest.mark.parametrize(
    "solver_cls",
    [
        ColoringAutoCpSatImplSolver,
        ColoringAutoCpmpyImplSolver,
        ColoringAutoOptalImplSolver,
    ],
)
def test_via_generic_allocation(problem, solver_cls, optal_preview_only):
    kwargs = dict()
    check_solution = True
    if solver_cls == ColoringAutoOptalImplSolver and optal_preview_only:
        kwargs["do_not_retrieve_solutions"] = True
        check_solution = False
    if solver_cls == ColoringAutoCpSatImplSolver:
        kwargs["ortools_cpsat_solver_kwargs"] = {"log_search_progress": True}
    if solver_cls == ColoringAutoCpmpyImplSolver:
        kwargs["log_search_progress"] = True
    parameters_cp = ParametersCp.default()
    specific_solver = CpSatColoringSolver(problem=problem)
    specific_sol = specific_solver.solve().get_best_solution()
    assert specific_solver.status_solver == StatusSolver.OPTIMAL
    nb_colors_max = specific_solver._nb_colors
    nb_colors_opt = problem.evaluate(specific_sol)["nb_colors"]

    solver: TransformationSolver = solver_cls(problem=problem, nb_colors=nb_colors_max)
    res = solver.solve(parameters_cp=parameters_cp, **kwargs)
    assert solver.status_solver == StatusSolver.OPTIMAL
    if check_solution:
        sol = res[-1][0]
        assert problem.satisfy(sol)
        kpis = problem.evaluate(sol)
        assert kpis["nb_colors"] == nb_colors_opt
    elif solver_cls == ColoringAutoOptalImplSolver:
        assert isinstance(solver.wrapped_solver, ColoringGenericAutoCpSatImplSolver)
        assert (
            solver.wrapped_solver.get_current_best_internal_objective_value()
            == nb_colors_opt
        )
