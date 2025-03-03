#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import pytest

from discrete_optimization.fjsp.parser import get_data_available, parse_file
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.fjsp.solvers.lns_cpsat import (
    FjspConstraintHandler,
    NeighborBuilderSubPart,
    NeighFjspConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix


@pytest.mark.skip("fjsp datasets temporary not available.")
def test_lnscpsat_fjsp():
    files = get_data_available()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    print(file)
    problem = parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=solver,
        constraint_handler=ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=[
                FjspConstraintHandler(problem=problem, fraction_segment_to_fix=0.65),
                NeighFjspConstraintHandler(
                    problem=problem,
                    neighbor_builder=NeighborBuilderSubPart(
                        problem=problem, nb_cut_part=8
                    ),
                ),
            ],
            tag_constraint_handler=["random", "cut"],
            list_proba=[0.5, 0.5],
        ),
    )
    res = lns_solver.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=1000,
        callbacks=[
            TimerStopper(total_seconds=300),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
        ],
        parameters_cp=p,
        time_limit_subsolver_iter0=1,
        time_limit_subsolver=2,
    )
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)
