#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from time import sleep

import pytest
from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import LnsCpMzn, LnsOrtoolsCpSat
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackSolution
from discrete_optimization.knapsack.solvers.cp_mzn import (
    Cp2KnapsackSolver,
    KnapsackProblem,
)
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver
from discrete_optimization.knapsack.solvers.lns_cp import (
    KnapsackMznConstraintHandler,
    OrtoolsCpSatKnapsackConstraintHandler,
)
from discrete_optimization.knapsack.solvers.lns_lp import (
    InitialKnapsackMethod,
    InitialKnapsackSolution,
)


def test_knapsack_lns():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackProblem = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCp.default()
    solver = Cp2KnapsackSolver(
        model,
        cp_solver_name=CpSolverName.CHUFFED,
        params_objective_function=params_objective_function,
    )
    solver.init_model()
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = KnapsackMznConstraintHandler(
        problem=model, fraction_to_fix=0.83
    )
    lns_solver = LnsCpMzn(
        problem=model,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store_pure_cp = solver.solve(parameters_cp=params_cp)
    solution_pure_cp = result_store_pure_cp.get_best_solution_fit()
    result_store = lns_solver.solve(
        parameters_cp=params_cp,
        time_limit_subsolver=10,
        time_limit_subsolver_iter0=1,
        nb_iteration_lns=200,
        callbacks=[TimerStopper(total_seconds=30)],
    )
    solution = result_store.get_best_solution_fit()[0]
    assert model.satisfy(solution)
    model.evaluate(solution)

    fitness = [f for s, f in result_store]


def test_knapsack_lns_ortools():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackProblem = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCp.default()
    solver = CpSatKnapsackSolver(
        model,
        params_objective_function=params_objective_function,
    )
    solver.init_model()
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = OrtoolsCpSatKnapsackConstraintHandler(
        problem=model, fraction_to_fix=0.83
    )
    lns_solver = LnsOrtoolsCpSat(
        problem=model,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve(
        parameters_cp=params_cp,
        time_limit_subsolver=10,
        time_limit_subsolver_iter0=1,
        nb_iteration_lns=200,
        callbacks=[TimerStopper(total_seconds=30)],
    )
    solution = result_store.get_best_solution_fit()[0]
    assert model.satisfy(solution)
    model.evaluate(solution)


def test_knapsack_ortools_constraint_handler():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackProblem = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCp.default()
    solver = CpSatKnapsackSolver(
        model,
        params_objective_function=params_objective_function,
    )
    solver.init_model()

    # add constraint to force dummy solution
    res = solver.create_result_storage([(model.get_dummy_solution(), 0.0)])
    constraint_handler = OrtoolsCpSatKnapsackConstraintHandler(
        problem=model, fraction_to_fix=1.0
    )
    constraints = constraint_handler.adding_constraint_from_results_store(
        solver=solver, result_storage=res
    )
    # solve => should find dummy solution
    sol: KnapsackSolution
    res = solver.solve(
        parameters_cp=params_cp,
        time_limit=10,
    )
    sol, fit = res.get_best_solution_fit()
    assert all(taken == 0.0 for taken in sol.list_taken)
    assert fit == 0.0

    # remove constraint + solve => should find a better solution
    constraint_handler.remove_constraints_from_previous_iteration(
        solver=solver, previous_constraints=constraints
    )
    res = solver.solve(
        parameters_cp=params_cp,
        time_limit=10,
    )
    sol, fit = res.get_best_solution_fit()
    assert not all(taken == 0.0 for taken in sol.list_taken)
    assert fit > 0.0


def test_knapsack_lns_timer():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackProblem = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCp.default()
    solver = Cp2KnapsackSolver(
        model,
        cp_solver_name=CpSolverName.CHUFFED,
        params_objective_function=params_objective_function,
    )
    solver.init_model()
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = KnapsackMznConstraintHandler(
        problem=model, fraction_to_fix=0.83
    )
    lns_solver = LnsCpMzn(
        problem=model,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store_pure_cp = solver.solve(
        parameters_cp=params_cp,
        time_limit=10,
        time_limit_iter0=1,
    )
    solution_pure_cp = result_store_pure_cp.get_best_solution_fit()

    class SleepCallback(Callback):
        def on_step_end(self, step: int, res, solver):
            print("zzz")
            sleep(1)

    nb_iteration_tracker = NbIterationTracker()
    callbacks = [
        SleepCallback(),
        TimerStopper(total_seconds=3, check_nb_steps=5),
        nb_iteration_tracker,
    ]
    result_store = lns_solver.solve(
        parameters_cp=params_cp,
        time_limit_subsolver=10,
        time_limit_subsolver_iter0=1,
        nb_iteration_lns=200,
        skip_initial_solution_provider=False,
        callbacks=callbacks,
    )
    assert nb_iteration_tracker.nb_iteration <= 6


class NbIterationTrackerWithAssert(Callback):
    """
    Log the number of iteration of a given solver
    """

    def __init__(
        self,
    ):
        self.nb_iteration = 0

    def on_step_end(self, step, res, solver):
        assert step == self.nb_iteration
        self.nb_iteration += 1


@pytest.mark.parametrize("skip_initial_solution_provider", [False, True])
def test_knapsack_lns_cb_nbiter(skip_initial_solution_provider):
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackProblem = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCp.default()
    solver = Cp2KnapsackSolver(
        model,
        cp_solver_name=CpSolverName.CHUFFED,
        params_objective_function=params_objective_function,
    )
    solver.init_model()
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = KnapsackMznConstraintHandler(
        problem=model, fraction_to_fix=0.83
    )
    lns_solver = LnsCpMzn(
        problem=model,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store_pure_cp = solver.solve(
        parameters_cp=params_cp,
        time_limit=10,
    )
    solution_pure_cp = result_store_pure_cp.get_best_solution_fit()

    nb_iteration_tracker = NbIterationTrackerWithAssert()
    callbacks = [
        nb_iteration_tracker,
    ]
    result_store = lns_solver.solve(
        parameters_cp=params_cp,
        time_limit_subsolver=10,
        time_limit_subsolver_iter0=1,
        nb_iteration_lns=2,
        skip_initial_solution_provider=skip_initial_solution_provider,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    test_knapsack_lns()
