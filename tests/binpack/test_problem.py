#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution


def test_model_satisfy():
    f = [ff for ff in get_data_available_bppc() if "BPPC_4_1_1.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    dummy_solution = BinPackSolution(
        problem=problem, allocation=[i for i in range(problem.nb_items)]
    )
    problem.evaluate(dummy_solution)
    assert problem.satisfy(dummy_solution)
    dummy_solution = BinPackSolution(
        problem=problem, allocation=[0 for _ in range(problem.nb_items)]
    )
    assert not problem.satisfy(dummy_solution)
