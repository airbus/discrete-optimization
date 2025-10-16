#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.problem import WeightedTardinessProblem


def test_parser():
    problems = parse_file(get_data_available()[0])
    for problem in problems:
        sol = problem.get_dummy_solution()
        res = problem.evaluate(sol)
        assert "tardiness" in res
        assert "penalty" in res
        assert problem.satisfy(sol)
