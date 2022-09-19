#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.facility.facility_model import FacilityProblem
from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)


def test_model_satisfy():
    file = [f for f in get_data_available() if "fl_50_1" in f][0]
    facility_problem: FacilityProblem = parse_file(file)
    dummy_solution = facility_problem.get_dummy_solution()
    facility_problem.evaluate(dummy_solution)
    assert not facility_problem.satisfy(dummy_solution)


if __name__ == "__main__":
    test_model_satisfy()
