#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest
from pytest_cases import fixture, param_fixtures

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import (
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.cpsat import (
    CpSatColoringSolver,
    ModelingCpSat,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.generic_tools.do_solver import StatusSolver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


modeling, value_sequence_chain, used_variable, symmetry_on_used = param_fixtures(
    "modeling, value_sequence_chain, used_variable, symmetry_on_used",
    [
        (ModelingCpSat.BINARY, False, False, False),
        (ModelingCpSat.INTEGER, False, False, False),
        (ModelingCpSat.INTEGER, True, False, False),
        (ModelingCpSat.INTEGER, False, True, False),
        (ModelingCpSat.INTEGER, False, True, True),
        (ModelingCpSat.INTEGER, True, True, True),
    ],
)


@fixture
def solver(
    problem,
    with_coloring_constraint,
    modeling,
    value_sequence_chain,
    used_variable,
    symmetry_on_used,
):
    if with_coloring_constraint and value_sequence_chain:
        pytest.skip(
            "chosen coloring constraint not compatible with value_sequence_chain"
        )
    init_model_kwargs = dict(
        nb_colors=20,
        modeling=modeling,
        value_sequence_chain=value_sequence_chain,
        used_variable=used_variable,
        symmetry_on_used=symmetry_on_used,
    )
    solver = CpSatColoringSolver(problem)
    solver.init_model(**init_model_kwargs)
    return solver


def test_cpsat_solver(problem, solver, start_solution):
    p = ParametersCp.default()
    result_store = solver.solve(parameters_cp=p)
    solution, fit = result_store.get_best_solution_fit()
    assert problem.satisfy(solution)

    # test warm start
    # first solution is not start_solution
    assert result_store[0][0].colors != start_solution.colors

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_store = solver.solve(
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_store[0][0].colors == start_solution.colors


def test_constraint_nb_allocation_changes(problem, solver, start_solution):
    nb_changes = 3
    sol: ColoringSolution
    ref = start_solution

    # force to be away from start_solution
    constraints = solver.add_constraint_on_nb_allocation_changes(
        ref=ref, nb_changes=nb_changes, sign=SignEnum.UP
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert (
        sol.compute_nb_allocation_changes(
            ref,
            tasks=solver.subset_tasks_of_interest,
            unary_resources=solver.subset_unaryresources_allowed,
        )
        > nb_changes
    )

    solver.remove_constraints(constraints)

    # force to be close to start_solution
    solver.add_constraint_on_nb_allocation_changes(ref=ref, nb_changes=nb_changes)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert (
        sol.compute_nb_allocation_changes(
            ref,
            tasks=solver.subset_tasks_of_interest,
            unary_resources=solver.subset_unaryresources_allowed,
        )
        <= nb_changes
    )


def test_cpsat_solver_warmstart_prev():
    small_example = [f for f in get_data_available() if "gc_50_3" in f][0]
    color_problem = parse_file(small_example)
    solver = CpSatColoringSolver(color_problem)
    solver.init_model(nb_colors=20)
    p = ParametersCp.default()

    # must call solver.solve before set_warm_start_from_previous_run
    with pytest.raises(RuntimeError):
        solver.set_warm_start_from_previous_run()

    # first solve
    result_store0 = solver.solve(parameters_cp=p)
    assert len(result_store0) > 1

    # second solve w/o warmstart: restart from scratch
    result_store1 = solver.solve(parameters_cp=p)
    assert len(result_store1) == len(result_store0)

    # third solve with warmstart from previous run: find directly the optimal sol
    solver.set_warm_start_from_previous_run()
    result_store2 = solver.solve(parameters_cp=p)
    assert len(result_store2) == 1
    assert result_store1[-1][0].colors == result_store2[-1][0].colors


def test_cpsat_solver_internal_bound_and_objective():
    small_example = [f for f in get_data_available() if "gc_50_1" in f][0]
    color_problem = parse_file(small_example)
    solver = CpSatColoringSolver(color_problem)
    # timeout => bound and obj is None (not a single solution)
    result_store = solver.solve(time_limit=1e-5)
    assert solver.get_current_best_internal_objective_bound() is None
    assert solver.get_current_best_internal_objective_value() is None
    # not optimal => obj>bound
    result_store = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    assert solver.status_solver == StatusSolver.SATISFIED
    assert 0 < solver.get_current_best_internal_objective_bound()
    assert (
        solver.get_current_best_internal_objective_bound()
        < solver.get_current_best_internal_objective_value()
    )
    # optimal => obj == bound
    result_store = solver.solve()
    assert solver.status_solver == StatusSolver.OPTIMAL
    assert (
        solver.get_current_best_internal_objective_bound()
        == solver.get_current_best_internal_objective_value()
    )
    # infeasible => None
    solver.cp_model.add(solver.variables["nbc"] <= 1)
    res = solver.solve()
    assert solver.get_current_best_internal_objective_bound() is None
    assert solver.get_current_best_internal_objective_value() is None


def test_cpsat_solver_finetuned():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(small_example)
    solver = CpSatColoringSolver(color_problem)
    solver.init_model(nb_colors=20)
    p = ParametersCp.default()

    # must use existing attribute name for ortools CpSolver
    with pytest.raises(AttributeError):
        result_store = solver.solve(
            parameters_cp=p, ortools_cpsat_solver_kwargs=dict(toto=4)
        )
    # must use correct value
    with pytest.raises(ValueError):
        result_store = solver.solve(
            parameters_cp=p, ortools_cpsat_solver_kwargs=dict(search_branching=-4)
        )
    # works
    from ortools.sat.sat_parameters_pb2 import SatParameters

    result_store = solver.solve(
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(
            search_branching=SatParameters.PSEUDO_COST_SEARCH
        ),
    )

    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)
