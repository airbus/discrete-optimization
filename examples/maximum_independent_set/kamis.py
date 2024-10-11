#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import (
    ConstraintHandlerMix,
    TrivialInitialSolution,
)
from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.kamis import KamisMisSolver
from discrete_optimization.maximum_independent_set.solvers.lns import (
    LocalMovesOrtoolsCpSatMisConstraintHandler,
)

logging.basicConfig(level=logging.DEBUG)


def run_kamis_solver():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = KamisMisSolver(problem=mis_model)
    solver.init_model()
    res = solver.solve(method="redumis", time_limit=10000)
    sol, fit = res.get_best_solution_fit()
    print(mis_model.satisfy(sol), mis_model.evaluate(sol))
    solver = CpSatMisSolver(mis_model)
    solver.init_model()

    params_cp = ParametersCp.default_cpsat()
    params_cp.nb_process = 6
    initial_solution_provider = TrivialInitialSolution(
        solver.create_result_storage(list_solution_fits=[(sol, fit)])
    )
    list_constraints_handler = [
        # AllVarsOrtoolsCpSatMisConstraintHandler(problem=mis_model, fraction_to_fix=0.92),
        # OrtoolsCpSatMisConstraintHandler(problem=mis_model, fraction_to_fix=0.92),
        # DestroyOrtoolsCpSatMisConstraintHandler(problem=mis_model, fraction_to_fix=0.05),
        LocalMovesOrtoolsCpSatMisConstraintHandler(
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
    lns_solver = LnsOrtoolsCpSat(
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
