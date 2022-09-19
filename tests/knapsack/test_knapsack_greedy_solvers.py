#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.greedy_solvers import best_of_greedy


def test_greedy():
    files = [f for f in get_data_available() if "ks_4_0" in f]
    knapsack_model = parse_file(files[0])
    solution = best_of_greedy(knapsack_model)


if __name__ == "__main__":
    test_greedy()
