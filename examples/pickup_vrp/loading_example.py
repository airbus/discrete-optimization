#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
from sklearn.cluster import KMeans

import discrete_optimization.tsp.tsp_parser as tsp_parser
import discrete_optimization.vrp.vrp_parser as vrp_parser
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    create_pickup_and_delivery,
    create_selective_tsp,
)
from discrete_optimization.pickup_vrp.gpdp import GPDP, ProxyClass, build_pruned_problem
from discrete_optimization.pickup_vrp.solver.lp_solver import (
    LinearFlowSolver,
    ParametersMilp,
    plot_solution,
)
from discrete_optimization.pickup_vrp.solver.ortools_solver import (
    ORToolsGPDP,
    ParametersCost,
    first_solution_strategy_enum,
    local_search_metaheuristic_enum,
    plot_ortools_solution,
)

logging.basicConfig(level=logging.DEBUG)


def load_vrp_and_transform():
    file_path = vrp_parser.get_data_available()[1]
    vrp_model = vrp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model)


def load_tsp_and_transform():
    files_available = tsp_parser.get_data_available()
    file_path = files_available[1]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model)


def debug_lp():
    vrp = False
    tsp = True
    if tsp:
        files_available = tsp_parser.get_data_available()
        file_path = files_available[16]
        tsp_model = tsp_parser.parse_file(file_path)
        gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model, compute_graph=True)
    else:
        file_path = vrp_parser.get_data_available()[1]
        vrp_model = vrp_parser.parse_file(file_path)
        gpdp = ProxyClass.from_vrp_model_to_gpdp(
            vrp_model=vrp_model, compute_graph=True
        )
    simplify = False
    if simplify:
        gpdp = build_pruned_problem(gpdp)
    print(gpdp.graph.get_nodes())
    print(len(gpdp.graph.get_nodes()))
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    p = ParametersMilp.default()
    p.time_limit = 2000
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=False, nb_iteration_max=4, include_subtour=False
    )
    plot_solution(solutions[-1], gpdp)
    plt.show()


def selective_tsp():
    gpdp = create_selective_tsp()
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    p = ParametersMilp.default()
    p.time_limit = 30
    linear_flow_solver.init_model(
        one_visit_per_cluster=True, one_visit_per_node=False, include_subtour=False
    )
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=True, nb_iteration_max=4, include_subtour=False
    )
    print(solutions[-1].flow_solution)
    plot_solution(solutions[-1], gpdp)
    plt.show()


def vrp_capacity():
    file_path = vrp_parser.get_data_available()[4]
    print(file_path)
    vrp_model = vrp_parser.parse_file(file_path)
    print("Nb vehicle : ", vrp_model.vehicle_count)
    print("Capacities : ", vrp_model.vehicle_capacities)
    gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model, compute_graph=True)
    simplify = False
    if simplify:
        gpdp = build_pruned_problem(gpdp)
    print(gpdp.graph.get_nodes())
    print(len(gpdp.graph.get_nodes()))
    linear_flow_solver = LinearFlowSolver(problem=gpdp)
    p = ParametersMilp.default()
    p.time_limit = 30
    linear_flow_solver.init_model(
        include_capacity=True,
        include_resources=False,
        one_visit_per_node=True,
        include_time_evolution=False,
    )
    solutions = linear_flow_solver.solve_iterative(
        parameters_milp=p, do_lns=True, nb_iteration_max=4
    )
    plot_solution(solutions[-1], gpdp)
    plt.show()


def run_ortools_solver():
    file_path = vrp_parser.get_data_available()[4]
    print(file_path)
    vrp_model = vrp_parser.parse_file(file_path)
    print("Nb vehicle : ", vrp_model.vehicle_count)
    print("Capacities : ", vrp_model.vehicle_capacities)
    gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model)
    solver = ORToolsGPDP(problem=gpdp)
    solver.init_model(
        one_visit_per_cluster=False,
        one_visit_per_node=True,
        time_limit=10,
        n_solutions=100,
    )
    results = solver.solve_intern()
    plot_ortools_solution(results[0], gpdp)
    plt.show()
    print(results)


def run_ortools_solver_selective():
    gpdp = create_selective_tsp(nb_nodes=600, nb_vehicles=5, nb_clusters=100)
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
        include_equilibrate_charge=True,
        charge_constraint={
            v: (len(gpdp.clusters_to_node) // (2 * gpdp.number_vehicle), None)
            for v in range(gpdp.number_vehicle)
        },
        parameters_cost=[ParametersCost(dimension_name="Distance")],
        local_search_metaheuristic=local_search_metaheuristic_enum.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy_enum.SAVINGS,
        time_limit=200,
        n_solutions=10000,
    )
    results = solver.solve_intern()
    res_to_plot = min([r for r in results], key=lambda x: x[-1])
    plot_ortools_solution(res_to_plot, gpdp)
    plt.show()
    print(results)


def run_ortools_pickup_delivery():
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

    model = create_pickup_and_delivery(
        number_of_vehicles=4,
        number_of_node=100,
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
        ),
        ParametersCost(
            dimension_name="Distance",
            global_span=False,
            sum_over_vehicles=True,
            coefficient_vehicles=[1] * model.number_vehicle,
        ),
    ]
    solver = ORToolsGPDP(problem=model)
    solver.init_model(
        one_visit_per_cluster=False,
        one_visit_per_node=True,
        include_pickup_and_delivery=True,
        use_lns=True,
        parameters_cost=list_params_cost,
        include_equilibrate_charge=True,
        charge_constraint={
            v: (len(model.all_nodes) // (4 * model.number_vehicle), None)
            for v in range(model.number_vehicle)
        },
        local_search_metaheuristic=local_search_metaheuristic_enum.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy_enum.SAVINGS,
        time_limit=100,
        n_solutions=10000,
    )
    logging.basicConfig(level=logging.DEBUG)
    results = solver.solve_intern()
    res_to_plot = min([r for r in results], key=lambda x: x[-1])
    check_solution(res_to_plot[0], model)
    plot_ortools_solution(res_to_plot, model)
    plt.show()


def run_ortools_pickup_delivery_cluster():
    gpdp = create_pickup_and_delivery(
        number_of_node=200,
        include_cluster=True,
        include_pickup=False,
        pickup_per_cluster=True,
    )
    solver = ORToolsGPDP(problem=gpdp)
    solver.init_model(
        one_visit_per_cluster=True,
        one_visit_per_node=False,
        include_pickup_and_delivery=False,
        include_mandatory=False,
        include_pickup_and_delivery_per_cluster=True,
        local_search_metaheuristic=local_search_metaheuristic_enum.GUIDED_LOCAL_SEARCH,
        first_solution_strategy=first_solution_strategy_enum.PARALLEL_CHEAPEST_INSERTION,
        time_limit=30,
        n_solutions=10000,
    )
    results = solver.solve_intern()

    def check_solution(res, gpdp: GPDP):
        for p, d in gpdp.list_pickup_deliverable_per_cluster:
            index_vehicles_p = []
            for pp in p:
                for node in gpdp.clusters_to_node[pp]:
                    l = [i for i in range(len(res)) if node in res[i]]
                    if len(l) > 0:
                        index_vehicles_p += [l[0]]
            index_vehicles_p = set(index_vehicles_p)
            index_vehicles_d = []
            for dd in d:
                for node in gpdp.clusters_to_node[dd]:
                    l = [i for i in range(len(res)) if node in res[i]]
                    if len(l) > 0:
                        index_vehicles_d += [l[0]]
            index_vehicles_d = set(index_vehicles_d)
            assert len(index_vehicles_p) == 1
            assert len(index_vehicles_d) == 1
            vehicle_p = list(index_vehicles_p)[0]
            vehicle_d = list(index_vehicles_d)[0]
            assert vehicle_p == vehicle_d
            index_p = [
                res[vehicle_p].index(node)
                for pp in p
                for node in gpdp.clusters_to_node[pp]
                if node in res[vehicle_p]
            ]
            index_d = [
                res[vehicle_d].index(node)
                for dd in d
                for node in gpdp.clusters_to_node[dd]
                if node in res[vehicle_d]
            ]
            assert max(index_p) < min(index_d)

    t_deb = time.time()
    check_solution(results[-1][0], gpdp=gpdp)
    t_end = time.time()
    print(t_end - t_deb, " seconds check")
    t_deb = time.time()
    plot_ortools_solution(results[-1], gpdp)
    t_end = time.time()
    print(t_end - t_deb, " seconds plot")
    plt.show()


def create_examples_script(folder_to_save):
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    sizes = [10, 50, 150, 300, 1000]
    sizes = [1000]
    include_cluster = [True, False]
    include_pickup = [True, False]
    pickup_per_cluster = [True, False]
    for s in sizes:
        for cl in include_cluster:
            for picdel in include_pickup:
                for pickupcluster in pickup_per_cluster:
                    if pickupcluster and not include_cluster:
                        continue
                    if picdel and not cl and pickup_per_cluster:
                        continue
                    for n in range(1):
                        gpdp = create_pickup_and_delivery(
                            number_of_node=s,
                            include_cluster=cl,
                            include_pickup=picdel,
                            pickup_per_cluster=pickupcluster,
                        )
                        t = time.time_ns()
                        pickle.dump(
                            gpdp,
                            file=open(
                                os.path.join(
                                    folder_to_save,
                                    f"gpdp_size{s}_cluster{cl}_pickup{picdel}_pickuppercluster{pickupcluster}_time{t}.pk",
                                ),
                                "wb",
                            ),
                        )


if __name__ == "__main__":
    # debug_lp()
    # selective_tsp()
    # vrp_capacity()
    run_ortools_solver()
    # run_ortools_pickup_delivery()
    # run_ortools_solver_selective()
    # run_ortools_pickup_delivery_cluster()
