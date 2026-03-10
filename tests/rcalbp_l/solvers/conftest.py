#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pytest_cases import fixture

from discrete_optimization.rcalbp_l.parser import get_data_available, parse_rcalbpl_json
from discrete_optimization.rcalbp_l.problem import RCALBPLProblem


@fixture
def problem() -> RCALBPLProblem:
    """Fixture providing a small RC-ALBP/L problem instance for testing."""
    files = get_data_available()
    # Use the smallest instance (187 tasks) for fast tests
    file = [f for f in files if "187_2_26_2880.json" in f][0]
    problem = parse_rcalbpl_json(file)
    # Reduce number of periods for faster testing
    problem.nb_periods = min(3, problem.nb_periods)
    problem.periods = range(problem.nb_periods)
    return problem
