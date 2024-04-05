#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from time import sleep

import pytest

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import LNS_CP
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.cp_solvers import (
    CPKnapsackMZN2,
    KnapsackModel,
)
from discrete_optimization.knapsack.solvers.knapsack_lns_cp_solver import (
    ConstraintHandlerKnapsack,
)
from discrete_optimization.knapsack.solvers.knapsack_lns_solver import (
    InitialKnapsackMethod,
    InitialKnapsackSolution,
)


def test_knapsack_lns():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackModel = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCP.default()
    params_cp.time_limit = 10
    params_cp.time_limit_iter0 = 1
    solver = CPKnapsackMZN2(
        model,
        cp_solver_name=CPSolverName.CHUFFED,
        params_objective_function=params_objective_function,
    )
    solver.init_model()
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = ConstraintHandlerKnapsack(problem=model, fraction_to_fix=0.83)
    lns_solver = LNS_CP(
        problem=model,
        cp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store_pure_cp = solver.solve(parameters_cp=params_cp)
    solution_pure_cp = result_store_pure_cp.get_best_solution_fit()
    result_store = lns_solver.solve_lns(
        parameters_cp=params_cp,
        nb_iteration_lns=200,
        callbacks=[TimerStopper(total_seconds=30)],
    )
    solution = result_store.get_best_solution_fit()[0]
    assert model.satisfy(solution)
    model.evaluate(solution)

    fitness = [f for s, f in result_store.list_solution_fits]


def test_knapsack_lns_timer():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackModel = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCP.default()
    params_cp.time_limit = 10
    params_cp.time_limit_iter0 = 1
    solver = CPKnapsackMZN2(
        model,
        cp_solver_name=CPSolverName.CHUFFED,
        params_objective_function=params_objective_function,
    )
    solver.init_model()
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = ConstraintHandlerKnapsack(problem=model, fraction_to_fix=0.83)
    lns_solver = LNS_CP(
        problem=model,
        cp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store_pure_cp = solver.solve(parameters_cp=params_cp)
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
        nb_iteration_lns=200,
        skip_first_iteration=False,
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


@pytest.mark.parametrize("skip_first_iteration", [False, True])
def test_knapsack_lns_cb_nbiter(skip_first_iteration):
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackModel = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCP.default()
    params_cp.time_limit = 10
    params_cp.time_limit_iter0 = 1
    solver = CPKnapsackMZN2(
        model,
        cp_solver_name=CPSolverName.CHUFFED,
        params_objective_function=params_objective_function,
    )
    solver.init_model()
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = ConstraintHandlerKnapsack(problem=model, fraction_to_fix=0.83)
    lns_solver = LNS_CP(
        problem=model,
        cp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store_pure_cp = solver.solve(parameters_cp=params_cp)
    solution_pure_cp = result_store_pure_cp.get_best_solution_fit()

    nb_iteration_tracker = NbIterationTrackerWithAssert()
    callbacks = [
        nb_iteration_tracker,
    ]
    result_store = lns_solver.solve(
        parameters_cp=params_cp,
        nb_iteration_lns=2,
        skip_first_iteration=skip_first_iteration,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    test_knapsack_lns()
