import os
import sys

from discrete_optimization.tsp.tsp_model import SolutionTSP, TSPModel, compute_length
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_load():
    files = get_data_available()
    file = [f for f in files if "85900" in f][0]
    one_model = parse_file(file)
    basic_solution = SolutionTSP(0, 0, list(range(one_model.node_count)), None, None)
    for k in range(100):
        compute_length(
            0,
            0,
            basic_solution.permutation,
            one_model.list_points,
            one_model.node_count,
            one_model.length_permutation,
        )


if __name__ == "__main__":
    test_load()
