#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from pytest_cases import fixture, param_fixture

from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.problem import Point2DTspProblem, TspSolution

end_index = param_fixture("end_index", [0, 4])


@fixture
def problem(end_index) -> Point2DTspProblem:
    files = get_data_available()
    files = [f for f in files if "tsp_5_1" in f]
    return parse_file(files[0], start_index=0, end_index=end_index)


@fixture
def start_solutions(end_index, problem) -> tuple[TspSolution, TspSolution]:
    if end_index == 0:
        permutations = [[1, 4, 3, 2], [4, 3, 2, 1]]
    elif end_index == 4:
        permutations = [[2, 3, 1], [3, 1, 2]]
    else:
        raise NotImplementedError()
    return tuple(
        TspSolution(problem=problem, permutation=permutation)
        for permutation in permutations
    )
