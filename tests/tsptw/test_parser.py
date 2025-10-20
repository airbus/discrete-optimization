#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


def test_parser(problem):
    print(
        problem.nb_nodes,
        problem.distance_matrix,
        problem.distance_matrix,
        problem.time_windows,
        problem.depot_node,
        problem.customers,
        problem.nb_customers,
    )
