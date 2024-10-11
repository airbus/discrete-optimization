#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.problem import TspSolution
from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver
from discrete_optimization.tsp.solvers.lns_cpsat import (
    SubpathTspConstraintHandler,
    TspConstraintHandler,
)
from discrete_optimization.tsp.utils import closest_greedy


@pytest.mark.parametrize("end_index", [0, 10])
def test_cpsat_solver(end_index):
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    model = parse_file(files[0], start_index=0, end_index=end_index)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = CpSatTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    res = solver.solve(
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=ParametersCp.default_cpsat(),
    )
    sol, fitness = res.get_best_solution_fit()
    sol: TspSolution
    assert model.satisfy(sol)
    assert sol.end_index == end_index

    assert len(res) > 2

    # test warm start
    start_solution = res[1][0]

    # first solution is not start_solution
    assert res[0][0].permutation != start_solution.permutation

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert res[0][0].permutation == start_solution.permutation


def test_lns_cpsat_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = CpSatTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    lns_solver = LnsOrtoolsCpSat(
        problem=model,
        subsolver=solver,
        constraint_handler=ConstraintHandlerMix(
            problem=model,
            list_constraints_handler=[
                SubpathTspConstraintHandler(problem=model, fraction_segment_to_fix=0.7),
                TspConstraintHandler(problem=model, fraction_segment_to_fix=0.7),
            ],
            list_proba=[0.5, 0.5],
        ),
    )
    res = lns_solver.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=20,
        time_limit_subsolver=10,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=p,
    )
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
