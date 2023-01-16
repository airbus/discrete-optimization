#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import time

import matplotlib.pyplot as plt
from classic_ortools_example import create_matrix_data

from discrete_optimization.pickup_vrp.solver.ortools_solver import (
    ORToolsGPDP,
    ParametersCost,
    first_solution_strategy_enum,
    local_search_metaheuristic_enum,
    plot_ortools_solution,
)

logging.basicConfig(level=logging.DEBUG)


def run_time_windows():
    gpdp = create_matrix_data()
    solver = ORToolsGPDP(
        problem=gpdp, factor_multiplier_time=1, factor_multiplier_distance=1
    )
    solver.init_model(
        one_visit_per_cluster=False,
        one_visit_per_node=True,
        include_demand=False,
        include_time_dimension=True,
        include_pickup_and_delivery=False,
        include_pickup_and_delivery_per_cluster=False,
        include_cumulative=False,
        include_mandatory=False,
        include_time_windows=True,
        include_time_windows_cluster=False,
        local_search_metaheuristic=local_search_metaheuristic_enum.SIMULATED_ANNEALING,
        first_solution_strategy=first_solution_strategy_enum.PATH_CHEAPEST_ARC,
        time_limit=45,
        n_solutions=200,
    )
    results = solver.solve_intern()
    t_deb = time.time()
    res_to_plot = min([r for r in results], key=lambda x: x[-1])
    dimension_data = res_to_plot[1]
    plot_ortools_solution(res_to_plot, gpdp)
    plt.show()


def run_pickup():
    gpdp = create_matrix_data()
    solver = ORToolsGPDP(
        problem=gpdp, factor_multiplier_time=1, factor_multiplier_distance=1
    )
    solver.init_model(
        one_visit_per_cluster=False,
        one_visit_per_node=True,
        include_demand=False,
        include_time_dimension=True,
        include_pickup_and_delivery=True,
        include_pickup_and_delivery_per_cluster=False,
        include_cumulative=False,
        include_mandatory=False,
        include_time_windows=False,
        include_time_windows_cluster=False,
        local_search_metaheuristic=local_search_metaheuristic_enum.TABU_SEARCH,
        first_solution_strategy=first_solution_strategy_enum.PARALLEL_CHEAPEST_INSERTION,
        time_limit=15,
        n_solutions=200,
    )
    results = solver.solve_intern()
    t_deb = time.time()
    res_to_plot = min([r for r in results], key=lambda x: x[-1])
    plot_ortools_solution(res_to_plot, gpdp)
    plt.show()


def run_demand():
    gpdp = create_matrix_data()
    solver = ORToolsGPDP(
        problem=gpdp, factor_multiplier_time=1, factor_multiplier_distance=1
    )
    list_parameters_cost = [
        ParametersCost(
            dimension_name="Distance",
            global_span=False,
            sum_over_vehicles=True,
            coefficient_vehicles=[1, 100, 1, 5],
        )
    ]
    solver.init_model(
        one_visit_per_cluster=False,
        one_visit_per_node=True,
        include_demand=True,
        include_time_dimension=True,
        include_pickup_and_delivery=False,
        include_pickup_and_delivery_per_cluster=False,
        include_cumulative=False,
        include_mandatory=False,
        include_time_windows=False,
        include_time_windows_cluster=False,
        local_search_metaheuristic=local_search_metaheuristic_enum.SIMULATED_ANNEALING,
        first_solution_strategy=first_solution_strategy_enum.PATH_CHEAPEST_ARC,
        time_limit=15,
        parameters_cost=list_parameters_cost,
        n_solutions=200,
    )
    results = solver.solve_intern()
    res_to_plot = min([r for r in results], key=lambda x: x[-1])
    plot_ortools_solution(res_to_plot, gpdp)
    plt.show()


if __name__ == "__main__":
    run_time_windows()
    run_pickup()
    run_demand()
