import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

from discrete_optimization.generic_tools.path_tools import abspath_from_file
from discrete_optimization.knapsack.knapsack_parser import parse_file
from discrete_optimization.knapsack.solvers.greedy_solvers import best_of_greedy


def testing_greedy():
    knapsack_model = parse_file(abspath_from_file(__file__, "../../data/ks_4_0"))
    solution = best_of_greedy(knapsack_model)
    print("Model : ", knapsack_model)
    print("Solution : ", solution)


def testi():
    knapsack_model = parse_file(abspath_from_file(__file__, "../../data/ks_4_0"))
    solution = best_of_greedy(knapsack_model)
    print("Model : ", knapsack_model)
    print("Solution : ", solution)


if __name__ == "__main__":
    testing_greedy()
