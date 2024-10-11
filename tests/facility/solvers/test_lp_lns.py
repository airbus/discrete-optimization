#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os

import pytest

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver
from discrete_optimization.facility.solvers.lns_lp import (
    GurobiFacilityConstraintHandler,
    InitialFacilityMethod,
    InitialFacilitySolution,
    MathOptConstraintHandlerFacility,
)
from discrete_optimization.facility.solvers.lp import (
    GurobiFacilitySolver,
    MathOptFacilitySolver,
    ParametersMilp,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_mip import LnsMilp
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.mark.parametrize(
    "solver_cls, constraint_handler_cls",
    [
        (MathOptFacilitySolver, MathOptConstraintHandlerFacility),
        (GurobiFacilitySolver, GurobiFacilityConstraintHandler),
    ],
)
def test_facility_lns(solver_cls, constraint_handler_cls):
    if issubclass(solver_cls, GurobiMilpSolver) and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_16_1"][0]
    facility_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=facility_problem)
    params_milp = ParametersMilp(
        pool_solutions=1000,
        mip_gap=0.0001,
        mip_gap_abs=0.001,
        retrieve_all_solution=True,
    )
    solver = solver_cls(
        facility_problem,
        params_objective_function=params_objective_function,
    )
    solver.init_model(use_matrix_indicator_heuristic=False)
    initial_solution_provider = InitialFacilitySolution(
        problem=facility_problem,
        initial_method=InitialFacilityMethod.GREEDY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = constraint_handler_cls(
        problem=facility_problem, fraction_to_fix=0.5
    )

    lns_solver = LnsMilp(
        problem=facility_problem,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )

    result_store = lns_solver.solve(
        parameters_milp=params_milp,
        time_limit_subsolver=20,
        nb_iteration_lns=5,
        callbacks=[TimerStopper(total_seconds=100)],
        stop_first_iteration_if_optimal=False,
    )
    solution = result_store.get_best_solution_fit()[0]
    assert facility_problem.satisfy(solution)
    facility_problem.evaluate(solution)


@pytest.mark.parametrize(
    "solver_cls, constraint_handler_cls",
    [
        (MathOptFacilitySolver, MathOptConstraintHandlerFacility),
        (GurobiFacilitySolver, GurobiFacilityConstraintHandler),
    ],
)
def test_facility_constraint_handler(solver_cls, constraint_handler_cls):
    if issubclass(solver_cls, GurobiMilpSolver) and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_16_1"][0]
    facility_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=facility_problem)
    solver = solver_cls(
        facility_problem,
        params_objective_function=params_objective_function,
    )
    solver.init_model(use_matrix_indicator_heuristic=False)
    constraint_handler = constraint_handler_cls(
        problem=facility_problem, fraction_to_fix=0.5
    )

    greedy_solver = GreedyFacilitySolver(
        facility_problem, params_objective_function=params_objective_function
    )
    solution = greedy_solver.solve().get_best_solution()
    dummy_result_storage = solver.create_result_storage([(solution, 0.0)])
    constraints = constraint_handler.adding_constraint_from_results_store(
        solver=solver, result_storage=dummy_result_storage
    )
    assert len(constraints) > 0


if __name__ == "__main__":
    test_facility_lns()
