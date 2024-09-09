#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.lns_cp import LNS_OrtoolsCPSat
from discrete_optimization.generic_tools.lns_tools import (
    ConstraintHandlerMix,
    TrivialInitialSolution,
)
from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.maximum_independent_set.mis_parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.solvers.mis_kamis import (
    MisKamisSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_lns import (
    MisOrtoolsCPSatConstraintHandlerLocalMoves,
)
from discrete_optimization.maximum_independent_set.solvers.mis_ortools import (
    MisOrtoolsSolver,
)

logging.basicConfig(level=logging.DEBUG)


def run_kamis_solver():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = MisKamisSolver(problem=mis_model)
    solver.init_model()
    res = solver.solve(method="redumis", time_limit=10000)
    sol, fit = res.get_best_solution_fit()
    print(mis_model.satisfy(sol), mis_model.evaluate(sol))
    solver = MisOrtoolsSolver(mis_model)
    solver.init_model()

    params_cp = ParametersCP.default_cpsat()
    params_cp.nb_process = 6
    initial_solution_provider = TrivialInitialSolution(
        solver.create_result_storage(list_solution_fits=[(sol, fit)])
    )
    list_constraints_handler = [
        # MisOrtoolsCPSatConstraintHandlerAllVars(problem=mis_model, fraction_to_fix=0.92),
        # MisOrtoolsCPSatConstraintHandler(problem=mis_model, fraction_to_fix=0.92),
        # MisOrtoolsCPSatConstraintHandlerDestroy(problem=mis_model, fraction_to_fix=0.05),
        MisOrtoolsCPSatConstraintHandlerLocalMoves(
            problem=mis_model, nb_moves=25, fraction_to_fix=0.35
        )
    ]
    constraint_handler = ConstraintHandlerMix(
        problem=mis_model,
        list_constraints_handler=list_constraints_handler,
        list_proba=[1 / len(list_constraints_handler)] * len(list_constraints_handler),
        update_proba=False,
        tag_constraint_handler=[  # "allvars",
            # "activated",
            # "destroy",
            "localmoves"
        ],
    )
    lns_solver = LNS_OrtoolsCPSat(
        problem=mis_model,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
    )
    result_store = lns_solver.solve(
        skip_initial_solution_provider=False,
        parameters_cp=params_cp,
        time_limit_subsolver=2,
        time_limit_subsolver_iter0=1,
        nb_iteration_lns=5000,
        callbacks=[
            TimerStopper(total_seconds=10000),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
        ],
    )
    solution, fit = result_store.get_best_solution_fit()
    assert mis_model.satisfy(solution)


if __name__ == "__main__":
    run_kamis_solver()
