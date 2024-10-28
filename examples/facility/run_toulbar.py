#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver
from discrete_optimization.facility.solvers.toulbar import (
    FacilityConstraintHandlerDestroyFacilityToulbar,
    FacilityConstraintHandlerToulbar,
    ModelingToulbarFacility,
    ToulbarFacilitySolver,
    ToulbarFacilitySolverForLns,
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
from discrete_optimization.generic_tools.toulbar_tools import to_lns_toulbar

logging.basicConfig(level=logging.INFO)


def run_facility_toulbar():
    file = [f for f in get_data_available() if "fl_50_5" in f][0]
    problem: FacilityProblem = parse_file(file)
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    solver = ToulbarFacilitySolver(problem=problem)
    solver.init_model(modeling=ModelingToulbarFacility.INTEGER)
    res = solver.solve(time_limit=100)
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    print(fit)


def run_facility_toulbar_ws():
    file = [f for f in get_data_available() if "fl_50_5" in f][0]
    problem: FacilityProblem = parse_file(file)
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    ws = GreedyFacilitySolver(problem=problem).solve().get_best_solution_fit()[0]
    print(problem.evaluate(ws))
    solver = ToulbarFacilitySolver(problem=problem)
    solver.init_model(modeling=ModelingToulbarFacility.INTEGER)
    solver.set_warm_start(ws)
    print("Starting solve")
    res = solver.solve(time_limit=100)
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    print(fit)


def run_facility_toulbar_lns():
    file = [f for f in get_data_available() if "fl_50_5" in f][0]
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
    print("LNS start")
    res = solver_lns.solve(
        nb_iteration_lns=1000000,
        time_limit_subsolver=5,
        time_limit_subsolver_iter0=10,
        skip_initial_solution_provider=False,
        callbacks=[TimerStopper(total_seconds=300)],
    )
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    print(fit)


if __name__ == "__main__":
    run_facility_toulbar_lns()
