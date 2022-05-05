from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.knapsack.knapsack_parser import files_available, parse_file
from discrete_optimization.knapsack.knapsack_solvers import (
    CPKnapsackMZN,
    CPKnapsackMZN2,
)


def cp_knapsack_1():
    file = [f for f in files_available if "ks_100_0" in f][0]
    knapsack_model = parse_file(file)
    cp_model = CPKnapsackMZN(knapsack_model)
    cp_model.init_model()
    result_storage = cp_model.solve(parameters_cp=ParametersCP.default())
    best = result_storage.get_best_solution_fit()
    print(result_storage.list_solution_fits)
    print(best[0], best[1])


def cp_knapsack_2():
    file = [f for f in files_available if "ks_100_0" in f][0]
    knapsack_model = parse_file(file)
    cp_model = CPKnapsackMZN2(knapsack_model)
    cp_model.init_model()
    result_storage = cp_model.solve(parameters_cp=ParametersCP.default())
    best = result_storage.get_best_solution_fit()
    print(result_storage.list_solution_fits)
    print(best[0], best[1])


if __name__ == "__main__":
    cp_knapsack_1()
    cp_knapsack_2()
