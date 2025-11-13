#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os

from pytest_cases import fixture

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.problem import FacilitySolution
from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver


@fixture
def problem():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_16_1"][0]
    return parse_file(file)


@fixture
def start_solution(problem) -> FacilitySolution:
    solver = GreedyFacilitySolver(problem)
    return solver.solve().get_best_solution()
