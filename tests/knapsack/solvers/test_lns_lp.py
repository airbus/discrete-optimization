#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_mip import LnsMilp
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.lns_lp import (
    InitialKnapsackMethod,
    InitialKnapsackSolution,
    MathOptKnapsackConstraintHandler,
)
from discrete_optimization.knapsack.solvers.lp import (
    KnapsackProblem,
    MathOptKnapsackSolver,
)


def test_knapsack_lns():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][0]
    model: KnapsackProblem = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_milp = ParametersMilp(
        pool_solutions=1000,
        mip_gap=0.0001,
        mip_gap_abs=0.001,
        retrieve_all_solution=True,
    )
    solver = MathOptKnapsackSolver(
        model,
        params_objective_function=params_objective_function,
    )
    solver.init_model(use_matrix_indicator_heuristic=False)
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = MathOptKnapsackConstraintHandler(
        problem=model, fraction_to_fix=0.95
    )
    lns_solver = LnsMilp(
        problem=model,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )

    result_store = lns_solver.solve(
        parameters_milp=params_milp,
        time_limit_subsolver=10,
        nb_iteration_lns=10,
        callbacks=[TimerStopper(total_seconds=30)],
    )
    solution = result_store.get_best_solution_fit()[0]
    assert model.satisfy(solution)
    model.evaluate(solution)


def test_knapsack_constraint_handler():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][0]
    model: KnapsackProblem = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_milp = ParametersMilp(
        pool_solutions=1000,
        mip_gap=0.0001,
        mip_gap_abs=0.001,
        retrieve_all_solution=True,
    )
    solver = MathOptKnapsackSolver(
        model,
        params_objective_function=params_objective_function,
    )
    solver.init_model(use_matrix_indicator_heuristic=False)
    dummy_result_storage = solver.create_result_storage(
        [(model.get_dummy_solution(), 0.0)]
    )
    constraint_handler = MathOptKnapsackConstraintHandler(
        problem=model, fraction_to_fix=0.95
    )
    constraints = constraint_handler.adding_constraint_from_results_store(
        solver=solver, result_storage=dummy_result_storage
    )
    assert len(constraints) > 0


if __name__ == "__main__":
    test_knapsack_lns()
