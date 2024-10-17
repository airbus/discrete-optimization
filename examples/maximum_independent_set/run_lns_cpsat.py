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
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.lns import (
    AllVarsOrtoolsCpSatMisConstraintHandler,
    DestroyOrtoolsCpSatMisConstraintHandler,
    LocalMovesOrtoolsCpSatMisConstraintHandler,
    OrtoolsCpSatMisConstraintHandler,
)

logging.basicConfig(level=logging.DEBUG)


def run_lns():
    small_example = [f for f in get_data_available() if "1dc.2048" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    params_cp = ParametersCp.default_cpsat()
    solver = CpSatMisSolver(mis_model)
    solver.init_model()
    res = solver.solve(
        parameters_cp=params_cp,
        time_limit=10,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    sol, fit = res.get_best_solution_fit()
    print("Found by cpsat :", fit)
    print("Status : ", solver.status_solver)

    initial_solution_provider = TrivialInitialSolution(
        solver.create_result_storage(
            list_solution_fits=[(mis_model.get_dummy_solution(), 0.0)]
        )
    )
    # constraint_handler = LocalMovesOrtoolsCpSatMisConstraintHandler(
    #    problem=mis_model, nb_moves=20, fraction_to_fix=0.3
    # )
    constraint_handler = AllVarsOrtoolsCpSatMisConstraintHandler(
        problem=mis_model, fraction_to_fix=0.2
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
        time_limit_subsolver=10,
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


def run_lns_mix():
    small_example = [f for f in get_data_available() if "1zc.4096" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    params_cp = ParametersCp.default_cpsat()
    params_cp.nb_process = 6
    solver = CpSatMisSolver(mis_model)
    solver.init_model()
    res = solver.solve(
        parameters_cp=params_cp,
        time_limit=2,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    sol, fit = res.get_best_solution_fit()
    print("Found by cpsat :", fit)
    print("Status : ", solver.status_solver)

    initial_solution_provider = TrivialInitialSolution(
        solver.create_result_storage(
            list_solution_fits=[(mis_model.get_dummy_solution(), 0.0)]
        )
    )
    list_constraints_handler = [
        AllVarsOrtoolsCpSatMisConstraintHandler(problem=mis_model, fraction_to_fix=0.8),
        OrtoolsCpSatMisConstraintHandler(problem=mis_model, fraction_to_fix=0.8),
        # DestroyOrtoolsCpSatMisConstraintHandler(problem=mis_model, fraction_to_fix=0.05),
        LocalMovesOrtoolsCpSatMisConstraintHandler(
            problem=mis_model, nb_moves=25, fraction_to_fix=0.35
        ),
    ]
    constraint_handler = ConstraintHandlerMix(
        problem=mis_model,
        list_constraints_handler=list_constraints_handler,
        list_proba=[1 / len(list_constraints_handler)] * len(list_constraints_handler),
        update_proba=False,
        tag_constraint_handler=[
            "allvars",
            "activated",
            # "destroy",
            "localmoves",
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


def run_constraint_handler():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    # look best solution found by solver unconstrained => find a node not taken
    solver = CpSatMisSolver(mis_model)
    solver.init_model()
    sol: MisSolution
    res = solver.solve(time_limit=10)
    sol, fit = res.get_best_solution_fit()
    idx = sol.chosen.index(0)

    # create a solution taking only first node not taken
    chosen = [0] * mis_model.number_nodes
    chosen[idx] = 1
    sol = MisSolution(problem=mis_model, chosen=chosen)
    assert mis_model.satisfy(sol)
    res = solver.create_result_storage(list_solution_fits=[(sol, 0.0)])

    # use constraint to force first node
    solver = CpSatMisSolver(mis_model)
    solver.init_model()
    constraint_handler = OrtoolsCpSatMisConstraintHandler(
        problem=mis_model, fraction_to_fix=1.0
    )
    constraints = constraint_handler.adding_constraint_from_results_store(
        solver=solver, result_storage=res
    )

    # solve => should take forced node
    sol: MisSolution
    res = solver.solve(time_limit=10)
    sol, fit = res.get_best_solution_fit()
    assert sol.chosen[idx] == 1

    # check that w/o constraint the node is dropped
    solver = CpSatMisSolver(mis_model)
    solver.init_model()
    sol: MisSolution
    res = solver.solve(time_limit=10)
    sol, fit = res.get_best_solution_fit()
    assert sol.chosen[idx] == 0


if __name__ == "__main__":
    run_lns()
