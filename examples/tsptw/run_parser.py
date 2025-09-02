#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


def run_parser():
    problem = parse_tsptw_file(get_data_available()[0])
    print(problem)
    print(
        problem.nb_nodes,
        problem.distance_matrix,
        problem.distance_matrix,
        problem.time_windows,
        problem.depot_node,
        problem.customers,
        problem.nb_customers,
    )


if __name__ == "__main__":
    run_parser()
