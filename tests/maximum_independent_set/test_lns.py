import logging

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.lns_cp import LNS_OrtoolsCPSat
from discrete_optimization.generic_tools.lns_tools import TrivialInitialSolution
from discrete_optimization.maximum_independent_set.mis_model import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.mis_parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.solvers.mis_lns import (
    MisOrtoolsCPSatConstraintHandler,
)
from discrete_optimization.maximum_independent_set.solvers.mis_ortools import (
    MisOrtoolsSolver,
)


def test_lns():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    params_cp = ParametersCP.default()
    params_cp.time_limit = 10
    params_cp.time_limit_iter0 = 1
    solver = MisOrtoolsSolver(mis_model)
    solver.init_model()

    initial_solution_provider = TrivialInitialSolution(
        solver.create_result_storage(
            list_solution_fits=[(mis_model.get_dummy_solution(), 0.0)]
        )
    )
    constraint_handler = MisOrtoolsCPSatConstraintHandler(
        problem=mis_model, fraction_to_fix=0.1
    )

    lns_solver = LNS_OrtoolsCPSat(
        problem=mis_model,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
    )
    result_store = lns_solver.solve(
        parameters_cp=params_cp,
        nb_iteration_lns=200,
        callbacks=[TimerStopper(total_seconds=30)],
    )
    solution, fit = result_store.get_best_solution_fit()
    assert mis_model.satisfy(solution)


def test_constraint_handler():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    params_cp = ParametersCP.default()
    params_cp.time_limit = 10
    params_cp.time_limit_iter0 = 1

    # look best solution found by solver unconstrained => find a node not taken
    solver = MisOrtoolsSolver(mis_model)
    solver.init_model()
    sol: MisSolution
    res = solver.solve(parameters_cp=params_cp)
    sol, fit = res.get_best_solution_fit()
    idx = sol.chosen.index(0)

    # create a solution taking only first node not taken
    chosen = [0] * mis_model.number_nodes
    chosen[idx] = 1
    sol = MisSolution(problem=mis_model, chosen=chosen)
    assert mis_model.satisfy(sol)
    res = solver.create_result_storage(list_solution_fits=[(sol, 0.0)])

    # use constraint to force first node
    solver = MisOrtoolsSolver(mis_model)
    solver.init_model()
    constraint_handler = MisOrtoolsCPSatConstraintHandler(
        problem=mis_model, fraction_to_fix=1.0
    )
    constraints = constraint_handler.adding_constraint_from_results_store(
        solver=solver, result_storage=res
    )

    # solve => should take forced node
    sol: MisSolution
    res = solver.solve(parameters_cp=params_cp)
    sol, fit = res.get_best_solution_fit()
    assert sol.chosen[idx] == 1

    # check that w/o constraint the node is dropped
    solver = MisOrtoolsSolver(mis_model)
    solver.init_model()
    sol: MisSolution
    res = solver.solve(parameters_cp=params_cp)
    sol, fit = res.get_best_solution_fit()
    assert sol.chosen[idx] == 0
