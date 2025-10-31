#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.problem import TeamAllocationProblem


def test_problem():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance)
    assert isinstance(allocation_problem, TeamAllocationProblem)
