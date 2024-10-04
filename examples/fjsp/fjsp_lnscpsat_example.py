#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.fjsp.flex_job_shop_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.fjsp.solvers.cpsat_fjsp_solver import CPSatFJspSolver
from discrete_optimization.fjsp.solvers.cpsat_lns_fjsp_solver import (
    ConstraintHandlerFJSP,
    ConstraintHandlerNeighFJSP,
    NeighborBuilderSubPart,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.lns_cp import LNS_OrtoolsCPSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix

logging.basicConfig(level=logging.INFO)


def run_lnscpsat_fjsp():
    files = get_data_available()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    print(file)
    problem = parse_file(file)
    solver = CPSatFJspSolver(problem=problem)
    p = ParametersCP.default_cpsat()
    p.nb_process = 10
    lns_solver = LNS_OrtoolsCPSat(
        problem=problem,
        subsolver=solver,
        constraint_handler=ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=[
                ConstraintHandlerFJSP(problem=problem, fraction_segment_to_fix=0.65),
                ConstraintHandlerNeighFJSP(
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


if __name__ == "__main__":
    run_lnscpsat_fjsp()
