from typing import Hashable, Set

import numpy as np
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    GPDP,
    create_ortools_example,
    create_pickup_and_delivery,
    create_selective_tsp,
    load_tsp_and_transform,
    load_vrp_and_transform,
)
from discrete_optimization.pickup_vrp.solver.ortools_solver import (
    ORToolsGPDP,
    ParametersCost,
    first_solution_strategy_enum,
    local_search_metaheuristic_enum,
    plot_ortools_solution,
    plt,
)


def check_solution(res, gpdp: GPDP):
    for p, d in gpdp.list_pickup_deliverable:
        index_vehicles_p = set(
            [[i for i in range(len(res)) if pp in res[i]][0] for pp in p]
        )
        index_vehicles_d = set(
            [[i for i in range(len(res)) if dd in res[i]][0] for dd in d]
        )
        assert len(index_vehicles_p) == 1
        assert len(index_vehicles_d) == 1
        vehicle_p = list(index_vehicles_p)[0]
        vehicle_d = list(index_vehicles_d)[0]
        assert vehicle_p == vehicle_d
        index_p = [res[vehicle_p].index(pp) for pp in p]
        index_d = [res[vehicle_d].index(dd) for dd in d]
        assert max(index_p) < min(index_d)


def test_pickup_and_delivery():
    model = create_pickup_and_delivery(
        number_of_vehicles=4,
        number_of_node=75,
        include_pickup=True,
        fraction_of_pickup_deliver=0.125,
        include_cluster=False,
        pickup_per_cluster=False,
    )
    list_params_cost = [
        ParametersCost(
            dimension_name="Distance",
            global_span=True,
            sum_over_vehicles=False,
            coefficient_vehicles=10,
        )
    ]
    solver = ORToolsGPDP(problem=model)
    solver.init_model(
        one_visit_per_cluster=False,
        one_visit_per_node=True,
        include_demand=False,
        include_pickup_and_delivery=True,
        parameters_cost=list_params_cost,
        use_lns=True,
        local_search_metaheuristic=local_search_metaheuristic_enum.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy_enum.SAVINGS,
        time_limit=100,
        n_solutions=10000,
    )
    results = solver.solve()
    res_to_plot = min([r for r in results], key=lambda x: x[-1])
    check_solution(res_to_plot[0], model)
    # plot_ortools_solution(res_to_plot, model)
    # plt.show()


def test_pickup_and_delivery_equilibrate():
    model = create_pickup_and_delivery(
        number_of_vehicles=4,
        number_of_node=75,
        include_pickup=True,
        fraction_of_pickup_deliver=0.125,
        include_cluster=False,
        pickup_per_cluster=False,
    )
    list_params_cost = [
        ParametersCost(
            dimension_name="Distance",
            global_span=True,
            sum_over_vehicles=False,
            coefficient_vehicles=10,
        )
    ]
    solver = ORToolsGPDP(problem=model)
    solver.init_model(
        one_visit_per_cluster=False,
        one_visit_per_node=True,
        include_demand=False,
        include_pickup_and_delivery=True,
        parameters_cost=list_params_cost,
        use_lns=True,
        include_equilibrate_charge=True,
        charge_constraint={
            v: (
                len(model.all_nodes) // (4 * model.number_vehicle),
                int(0.5 * len(model.all_nodes)),
            )
            for v in range(model.number_vehicle)
        },
        local_search_metaheuristic=local_search_metaheuristic_enum.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy_enum.AUTOMATIC,
        time_limit=20,
        n_solutions=10000,
    )
    results = solver.solve()
    res_to_plot = min([r for r in results], key=lambda x: x[-1])
    check_solution(res_to_plot[0], model)
    # plot_ortools_solution(res_to_plot, model)
    # plt.show()


def test_selective_tsp():
    gpdp = create_selective_tsp(nb_nodes=1000, nb_vehicles=1, nb_clusters=100)
    solver = ORToolsGPDP(
        problem=gpdp, factor_multiplier_distance=1, factor_multiplier_time=1
    )
    solver.init_model(
        one_visit_per_cluster=True,
        one_visit_per_node=False,
        include_time_dimension=True,
        include_demand=True,
        include_mandatory=True,
        include_pickup_and_delivery=False,
        parameters_cost=[ParametersCost(dimension_name="Distance", global_span=True)],
        local_search_metaheuristic=local_search_metaheuristic_enum.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy_enum.SAVINGS,
        time_limit=20,
        n_solutions=10000,
    )
    results = solver.solve()
    res_to_plot = min([r for r in results], key=lambda x: x[-1])
    # plot_ortools_solution(res_to_plot, gpdp)
    # plt.show()
