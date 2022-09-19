#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.tsp.solver.solver_ortools import TSP_ORtools
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_ortools():
    files = get_data_available()
    files = [f for f in files if "tsp_200_2" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    solution = model.get_dummy_solution()
    params_objective_function = get_default_objective_setup(problem=model)
    solver = TSP_ORtools(model, params_objective_function=params_objective_function)
    solver.init_model()
    sol, fitness = solver.solve().get_best_solution_fit()
    assert model.satisfy(sol)


if __name__ == "__main__":
    test_ortools()
