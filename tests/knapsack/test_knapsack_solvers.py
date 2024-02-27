#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import random

import numpy as np
import pytest

from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.knapsack_solvers import solve, solvers_map
from discrete_optimization.knapsack.solvers.lp_solvers import LPKnapsackGurobi

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize("knapsack_problem_file", get_data_available())
def test_load_file(knapsack_problem_file):
    knapsack_model: KnapsackModel = parse_file(knapsack_problem_file)
    dummy_solution = knapsack_model.get_dummy_solution()
    assert knapsack_model.satisfy(dummy_solution)


def test_solvers():
    logging.basicConfig(level=logging.INFO)
    small_example = [f for f in get_data_available() if "ks_40_0" in f][0]
    knapsack_model: KnapsackModel = parse_file(small_example)
    solvers = solvers_map.keys()
    for s in solvers:
        if s == LPKnapsackGurobi:
            continue
        logging.info(f"Solver {s}")
        results = solve(method=s, problem=knapsack_model, **solvers_map[s][1])
        s, f = results.get_best_solution_fit()
        logging.info(f"fitness={f}")
