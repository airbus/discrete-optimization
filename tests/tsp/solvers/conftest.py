#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from pytest_cases import fixture, param_fixture

from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.problem import Point2DTspProblem, TspProblem

end_index = param_fixture("end_index", [0, 10])


@fixture
def problem(end_index) -> Point2DTspProblem:
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    return parse_file(files[0], start_index=0, end_index=end_index)
