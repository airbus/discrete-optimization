#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os

from discrete_optimization.facility.facility_model import FacilityProblem
from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ea.ga import DeapMutation, Ga


def test_ga_facility():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_50_6"][0]
    facility_problem: FacilityProblem = parse_file(file)
    params = get_default_objective_setup(facility_problem)
    ga_solver = Ga(
        facility_problem,
        encoding="facility_for_customers",
        objective_handling=params.objective_handling,
        objectives=params.objectives,
        objective_weights=params.weights,
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=1000,
    )
    facility_solution = ga_solver.solve().get_best_solution()
    facility_problem.evaluate(facility_solution)
    assert facility_problem.satisfy(facility_solution)


if __name__ == "__main__":
    test_ga_facility()
