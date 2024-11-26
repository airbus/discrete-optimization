#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    TrivialInitialSolution,
    from_solutions_to_result_storage,
)
from discrete_optimization.generic_tools.toulbar_tools import to_lns_toulbar
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.toulbar import (
    RcpspConstraintHandlerToulbar,
    ToulbarMultimodeRcpspSolver,
    ToulbarRcpspSolver,
    ToulbarRcpspSolverForLns,
)

logging.basicConfig(level=logging.INFO)


def run_toulbar_rcpsp():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_8.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = ToulbarRcpspSolver(rcpsp_problem)
    solver.init_model()
    res = solver.solve(time_limit=30)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)
    print(rcpsp_problem.evaluate(sol))


def run_toulbar_mrcpsp():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_3.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = ToulbarMultimodeRcpspSolver(rcpsp_problem)
    solver.init_model()
    res = solver.solve(time_limit=30)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)
    print(rcpsp_problem.evaluate(sol))


def run_toulbar_all_mrcpsp():
    files_available = get_data_available()
    for file in files_available:
        if "mm" in file:
            rcpsp_problem = parse_file(file)
            solver = ToulbarMultimodeRcpspSolver(rcpsp_problem)
            solver.init_model()
            res = solver.solve(time_limit=30)
            sol, fit = res.get_best_solution_fit()
            assert rcpsp_problem.satisfy(sol)
            print(rcpsp_problem.evaluate(sol))


def run_toulbar_rcpsp_ws():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_8.sm" in f][0]
    rcpsp_problem = parse_file(file)
    dummy = rcpsp_problem.get_dummy_solution()
    print(rcpsp_problem.evaluate(dummy), " dummy solution ")
    rcpsp_problem.horizon = int(rcpsp_problem.evaluate(dummy)["makespan"])
    solver = ToulbarRcpspSolver(rcpsp_problem)
    solver.init_model(ub=rcpsp_problem.horizon, vns=-2)
    solver.set_warm_start(dummy)
    res = solver.solve(time_limit=100)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)
    print(rcpsp_problem.evaluate(sol))


def run_toulbar_rcpsp_lns():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_9.sm" in f][0]
    rcpsp_problem = parse_file(file)
    dummy = rcpsp_problem.get_dummy_solution()
    print(rcpsp_problem.evaluate(dummy), " dummy solution ")
    rcpsp_problem.horizon = int(dummy.get_start_time(rcpsp_problem.sink_task))
    solver = to_lns_toulbar(ToulbarRcpspSolver)(rcpsp_problem)
    solver.init_model(vns=None)
    solver_lns = BaseLns(
        problem=rcpsp_problem,
        subsolver=solver,
        initial_solution_provider=TrivialInitialSolution(
            solution=from_solutions_to_result_storage([dummy], problem=rcpsp_problem)
        ),
        constraint_handler=RcpspConstraintHandlerToulbar(
            problem=rcpsp_problem, fraction_task=0.8
        ),
    )
    res = solver_lns.solve(
        nb_iteration_lns=1000,
        time_limit_subsolver=5,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)
    print(rcpsp_problem.evaluate(sol))


def run_optuna_study():
    from discrete_optimization.generic_tools.optuna.utils import (
        generic_optuna_experiment_monoproblem,
    )

    files_available = get_data_available()
    file = [f for f in files_available if "j1201_10.sm" in f][0]
    rcpsp_problem = parse_file(file)
    dummy = rcpsp_problem.get_dummy_solution()
    rcpsp_problem.horizon = int(dummy.get_start_time(rcpsp_problem.sink_task))
    solvers_to_test = [ToulbarRcpspSolver, BaseLns]
    ToulbarRcpspSolver.hyperparameters = (
        ToulbarRcpspSolver.copy_and_update_hyperparameters(
            ["vns"], **{"vns": {"choices": [None, -4]}}
        )
    )
    generic_optuna_experiment_monoproblem(
        problem=rcpsp_problem,
        solvers_to_test=solvers_to_test,
        kwargs_fixed_by_solver={
            ToulbarRcpspSolver: {"time_limit": 50},
            BaseLns: {
                "callbacks": [TimerStopper(total_seconds=50)],
                "constraint_handler": RcpspConstraintHandlerToulbar(
                    problem=rcpsp_problem, fraction_task=0.8
                ),
                "post_process_solution": None,
                "initial_solution_provider": TrivialInitialSolution(
                    solution=from_solutions_to_result_storage(
                        [dummy], problem=rcpsp_problem
                    )
                ),
                "nb_iteration_lns": 1000,
                "time_limit_subsolver": 5,
            },
        },
        suggest_optuna_kwargs_by_name_by_solver={
            BaseLns: {"subsolver": {"choices": [to_lns_toulbar(ToulbarRcpspSolver)]}}
        },
    )


if __name__ == "__main__":
    run_toulbar_rcpsp_lns()
