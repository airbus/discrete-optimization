#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os.path

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers.dp import (
    DpFacilityModeling,
    DpFacilitySolver,
    dp,
)
from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver
from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_monoproblem,
    generic_optuna_experiment_multiproblem,
)


def dp_facility_example():
    file = [f for f in get_data_available() if "fl_25_4" in f][0]
    problem: FacilityProblem = parse_file(file)
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    solver = DpFacilitySolver(problem=problem)
    res = solver.solve(
        solver=dp.LNBS, modeling=DpFacilityModeling.FACILITY, time_limit=10
    )
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    print(fit)


def dp_facility_example_ws():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "fl_25_4" in f][0]
    problem: FacilityProblem = parse_file(file)
    g_sol: FacilitySolution = GreedyFacilitySolver(problem).solve()[0][0]
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    solver = DpFacilitySolver(problem=problem)
    solver.init_model(modeling=DpFacilityModeling.CUSTOMER)
    solver.set_warm_start(g_sol)
    res = solver.solve(
        solver=dp.LNBS, retrieve_intermediate_solutions=True, time_limit=3
    )
    sol: FacilitySolution = res[0][0]
    assert sol.facility_for_customers == g_sol.facility_for_customers
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


def dp_optuna_example():

    s = """fl_100_14
       fl_3_1
       fl_16_1
        fl_16_2
        fl_25_1
        fl_25_2
        fl_25_3
        fl_25_4
        fl_25_5
        fl_50_1
        fl_50_2
        fl_50_3
        fl_50_4
        fl_50_5
        fl_50_6"""
    #
    # fl_100_1
    # fl_100_2
    # fl_100_3
    # fl_100_4
    # fl_100_5
    # fl_100_6
    # fl_100_7
    # fl_100_8
    # fl_100_9
    # fl_100_10
    # fl_100_11
    # fl_100_12
    # fl_100_13
    files = [f for f in get_data_available() if os.path.basename(f) in s]
    problems = [parse_file(file) for file in files]
    generic_optuna_experiment_multiproblem(
        problems=problems,
        solvers_to_test=[DpFacilitySolver],
        kwargs_fixed_by_solver={
            DpFacilitySolver: {"time_limit": 30, "solver": dp.LNBS}
        },
        report_cumulated_fitness=False,
    )


if __name__ == "__main__":
    dp_facility_example()
