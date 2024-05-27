#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import platform

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_mip import LNS_MILP
from discrete_optimization.generic_tools.lp_tools import MilpSolverName, ParametersMilp
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.knapsack_lns_solver import (
    ConstraintHandlerKnapsack,
    InitialKnapsackMethod,
    InitialKnapsackSolution,
)
from discrete_optimization.knapsack.solvers.lp_solvers import KnapsackModel, LPKnapsack


@pytest.mark.skipif(
    platform.machine() == "arm64",
    reason=(
        "Python-mip has issues with cbclib on macos arm64. "
        "See https://github.com/coin-or/python-mip/issues/167"
    ),
)
def test_knapsack_lns():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][0]
    model: KnapsackModel = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_milp = ParametersMilp(
        time_limit=10,
        pool_solutions=1000,
        mip_gap=0.0001,
        mip_gap_abs=0.001,
        retrieve_all_solution=True,
        n_solutions_max=1000,
    )
    solver = LPKnapsack(
        model,
        milp_solver_name=MilpSolverName.CBC,
        params_objective_function=params_objective_function,
    )
    solver.init_model(use_matrix_indicator_heuristic=False)
    result_lp = solver.solve(parameters_milp=params_milp)
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = ConstraintHandlerKnapsack(problem=model, fraction_to_fix=0.95)
    lns_solver = LNS_MILP(
        problem=model,
        milp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )

    result_store = lns_solver.solve_lns(
        parameters_milp=params_milp,
        nb_iteration_lns=10000,
        callbacks=[TimerStopper(total_seconds=30)],
    )
    solution = result_store.get_best_solution_fit()[0]
    assert model.satisfy(solution)
    model.evaluate(solution)


if __name__ == "__main__":
    test_knapsack_lns()
