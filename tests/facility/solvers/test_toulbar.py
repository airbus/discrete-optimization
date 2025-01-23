#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver
from discrete_optimization.facility.solvers.toulbar import (
    FacilityConstraintHandlerDestroyFacilityToulbar,
    FacilityConstraintHandlerToulbar,
    ModelingToulbarFacility,
    ToulbarFacilitySolver,
    ToulbarFacilitySolverForLns,
    toulbar_available,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    ConstraintHandlerMix,
    TrivialInitialSolution,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    from_solutions_to_result_storage,
)

logging.basicConfig(level=logging.INFO)


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
@pytest.mark.parametrize(
    "modeling", [ModelingToulbarFacility.INTEGER, ModelingToulbarFacility.BINARY]
)
def test_facility_toulbar(modeling):
    file = [f for f in get_data_available() if "fl_16_2" in f][0]
    problem: FacilityProblem = parse_file(file)
    solver = ToulbarFacilitySolver(problem=problem)
    solver.init_model(modeling=modeling)
    res = solver.solve(time_limit=10)
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_facility_toulbar_ws():
    file = [f for f in get_data_available() if "fl_16_2" in f][0]
    problem: FacilityProblem = parse_file(file)
    ws = GreedyFacilitySolver(problem=problem).solve().get_best_solution_fit()[0]
    solver = ToulbarFacilitySolver(problem=problem)
    solver.init_model(modeling=ModelingToulbarFacility.INTEGER)
    solver.set_warm_start(ws)
    res = solver.solve(time_limit=10)
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_facility_toulbar_lns():
    file = [f for f in get_data_available() if "fl_16_2" in f][0]
    problem: FacilityProblem = parse_file(file)
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    ws = GreedyFacilitySolver(problem=problem).solve().get_best_solution_fit()[0]
    print(problem.evaluate(ws))
    solver = ToulbarFacilitySolverForLns(problem=problem)
    solver.init_model(modeling=ModelingToulbarFacility.INTEGER)
    solver_lns = BaseLns(
        problem=problem,
        subsolver=solver,
        initial_solution_provider=TrivialInitialSolution(
            solution=from_solutions_to_result_storage([ws], problem=problem)
        ),
        constraint_handler=ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=[
                FacilityConstraintHandlerToulbar(
                    problem=problem, fraction_of_customers=0.5
                ),
                FacilityConstraintHandlerDestroyFacilityToulbar(problem=problem),
            ],
            list_proba=[0.5, 0.5],
        ),
    )
    res = solver_lns.solve(
        nb_iteration_lns=1000,
        time_limit_subsolver=5,
        time_limit_subsolver_iter0=10,
        skip_initial_solution_provider=False,
        callbacks=[TimerStopper(total_seconds=20)],
    )
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)
