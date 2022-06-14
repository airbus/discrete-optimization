import os
import sys

from discrete_optimization.tsp.tsp_model import SolutionTSP, TSPModel, compute_length
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_load():
    files = get_data_available()
    file = [f for f in files if "85900" in f][0]
    one_model = parse_file(file)
    basic_solution = SolutionTSP(
        problem=one_model,
        permutation=list(range(1, one_model.node_count))
    )
    compute_length(
        start_index=basic_solution.start_index,
        end_index=basic_solution.end_index,
        solution=basic_solution.permutation,
        list_points=one_model.list_points,
        node_count=one_model.node_count,
        length_permutation=one_model.length_permutation,
    )


if __name__ == "__main__":
    test_load()
