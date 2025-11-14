#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os.path

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers.cpsat import CpSatFacilitySolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp

logging.basicConfig(level=logging.INFO)


def cp_facility_example():
    file = [f for f in get_data_available() if "fl_200_7" in f][0]
    problem: FacilityProblem = parse_file(file)
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    solver = CpSatFacilitySolver(problem=problem)
    print("Initializing...")
    solver.init_model()
    print("finished initializing")
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=100,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    print(fit)


if __name__ == "__main__":
    cp_facility_example()
