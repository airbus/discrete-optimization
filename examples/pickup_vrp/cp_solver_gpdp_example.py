#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import math
import os
import random
from datetime import timedelta
from enum import Enum
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
from minizinc import Instance, Model, Solver

from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.tsp.mutation.mutation_tsp import (
    Mutation2Opt,
    MutationSwapTSP,
)
from discrete_optimization.tsp.plots.plot_tsp import plot_tsp_solution
from discrete_optimization.tsp.tsp_model import Point2D, TSPModelDistanceMatrix

this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {
    "gpdp": os.path.join(
        this_path, "../../discrete_optimization/pickup_vrp/minizinc/gpdp.mzn"
    ),
    "gpdp-flow": os.path.join(
        this_path, "../../discrete_optimization/pickup_vrp/minizinc/gpdp_flow.mzn"
    ),
    "gpdp-resources": os.path.join(
        this_path, "../../discrete_optimization/pickup_vrp/minizinc/gpdp_resources.mzn"
    ),
}

logging.basicConfig(level=logging.DEBUG)


class GPDPOutput:
    objective: int
    __output_item: Optional[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        print("One solution ", self.objective)
        print("Output ", _output_item)

    def check(self) -> bool:
        return True


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def find_intersection(trajectory, points, test_all=False, nb_tests=10):
    perm = trajectory
    intersects = []
    its = (
        range(len(perm))
        if test_all
        else random.sample(range(len(perm)), min(nb_tests, len(perm)))
    )
    jts = (
        range(len(perm) - 1)
        if test_all
        else random.sample(range(len(perm) - 1), min(nb_tests, len(perm) - 1))
    )
    for i in its:
        for j in jts:
            ii = i
            jj = j
            if jj <= ii + 1:
                continue
            A, B = points[perm[ii] - 1, :], points[perm[ii + 1] - 1, :]
            C, D = points[perm[jj] - 1, :], points[perm[jj + 1] - 1, :]
            if intersect(A, B, C, D):
                intersects += [(ii + 1, jj)]
                if len(intersects) > 5:
                    break
        if len(intersects) > 5:
            break
    return intersects


def script_example():
    n_vehicles = 1
    number_of_nodes_transportation = 20
    coordinates = np.random.randint(
        -20, 20, size=(number_of_nodes_transportation + 2 * n_vehicles, 2)
    )

    distance_delta = dist.cdist(coordinates, coordinates)
    distance_delta = np.array(distance_delta, dtype=np.int_)
    for i in range(distance_delta.shape[0]):
        distance_delta[i, i] = 0
    time_delta = distance_delta / 2
    model = Model(files_mzn["gpdp"])
    model.output_type = GPDPOutput
    solver = Solver.lookup("chuffed")
    instance = Instance(solver, model)
    instance["number_vehicle"] = n_vehicles
    instance["number_of_nodes_transportation"] = number_of_nodes_transportation
    instance["distance_delta"] = [
        [int(distance_delta[i, j]) for j in range(distance_delta.shape[1])]
        for i in range(distance_delta.shape[0])
    ]
    instance["time_delta"] = [
        [int(time_delta[i, j]) for j in range(time_delta.shape[1])]
        for i in range(time_delta.shape[0])
    ]
    result = instance.solve(
        timeout=timedelta(seconds=20), intermediate_solutions=True, free_search=False
    )
    results = []
    for i in range(len(result)):
        results += [(result[i].dict["trajectories"], result[i].objective)]
    print(result.status)
    print("HEY")

    fig, ax = plt.subplots(1)
    for i in range(len(results)):
        traj = results[i][0]
        nb_colors = n_vehicles
        colors = plt.cm.get_cmap("hsv", nb_colors)
        for vehicle in range(len(traj)):
            color = colors(vehicle)
            for j in range(len(traj[vehicle]) - 1):
                ax.plot(
                    [
                        coordinates[traj[vehicle][j] - 1, 0],
                        coordinates[traj[vehicle][j + 1] - 1, 0],
                    ],
                    [
                        coordinates[traj[vehicle][j] - 1, 1],
                        coordinates[traj[vehicle][j + 1] - 1, 1],
                    ],
                    color=color,
                )
        ax.set_title("iter " + str(i) + " obj=" + str(int(results[i][1])))
        plt.draw()
        plt.pause(0.5)
        ax.clear()
    plt.show()


def script_example_lns():
    n_vehicles = 4
    number_of_nodes_transportation = 45
    coordinates = np.random.randint(
        -20, 20, size=(number_of_nodes_transportation + 2 * n_vehicles, 2)
    )
    distance_delta = dist.cdist(coordinates, coordinates)
    distance_delta = np.array(distance_delta, dtype=np.int_)
    for i in range(distance_delta.shape[0]):
        distance_delta[i, i] = 0
    time_delta = distance_delta / 2
    model = Model(files_mzn["gpdp"])
    model.output_type = GPDPOutput
    solver = Solver.lookup("chuffed")
    instance = Instance(solver, model)
    instance["number_vehicle"] = n_vehicles
    instance["number_of_nodes_transportation"] = number_of_nodes_transportation
    instance["distance_delta"] = [
        [int(distance_delta[i, j]) for j in range(distance_delta.shape[1])]
        for i in range(distance_delta.shape[0])
    ]
    instance["time_delta"] = [
        [int(time_delta[i, j]) for j in range(time_delta.shape[1])]
        for i in range(time_delta.shape[0])
    ]
    results = []
    indexes = [
        (v, index)
        for v in range(n_vehicles)
        for index in range(number_of_nodes_transportation + 2)
    ]
    dummy_solution = [
        [number_of_nodes_transportation + n_vehicles + v + 1]
        * (number_of_nodes_transportation + 2)
        for v in range(n_vehicles)
    ]
    for v in range(n_vehicles):
        dummy_solution[v][0] = number_of_nodes_transportation + v + 1
    cut_part = list(range(number_of_nodes_transportation))

    def chunks(l, n):
        n = max(1, n)
        return [l[i : min(i + n, len(l))] for i in range(0, len(l), n)]

    cut_parts = chunks(cut_part, int(math.ceil(len(cut_part) / n_vehicles)))
    for v in range(len(cut_parts)):
        for j in range(len(cut_parts[v])):
            dummy_solution[v][j + 1] = cut_parts[v][j] + 1

    for i in range(8):
        with instance.branch() as child:
            if i > 0:
                last_result = results[-1]
                trajo = last_result[0]
            else:
                trajo = dummy_solution
            if random.random() <= 0.5:
                indexes_fix = random.sample(indexes, int(0.8 * len(indexes)))
                for v, index in indexes_fix:
                    if (
                        trajo[v][index]
                        != number_of_nodes_transportation + n_vehicles + v + 1
                    ):
                        child.add_string(
                            "constraint trajectories["
                            + str(v + 1)
                            + ","
                            + str(index + 1)
                            + "]=="
                            + str(trajo[v][index])
                            + ";\n"
                        )
            else:
                print("METHOD 2")
                vehicles = random.choice(range(n_vehicles))
                for vehicle in [vehicles]:
                    traj = trajo[vehicle]
                    nodes = [
                        n
                        for n in traj
                        if n != number_of_nodes_transportation + vehicle + 1
                        and n
                        != number_of_nodes_transportation + n_vehicles + vehicle + 1
                    ]
                    for n in nodes:
                        child.add_string(
                            "constraint "
                            + str(n)
                            + " in {trajectories["
                            + str(vehicle + 1)
                            + ", i]|i in TRAJECTORY_INDEX};\n"
                        )
                    child.add_string(
                        "array[TRAJECTORY_INDEX] of ALL_NODES: ref=" + str(traj) + ";\n"
                    )
                    child.add_string(
                        "array[1..2] of var TRAJECTORY_INDEX: opt_index;\n"
                    )
                    child.add_string("constraint opt_index[2]>opt_index[1]+1;\n")
                    child.add_string("var bool: do_2_opt;")
                    child.add_string(
                        "constraint forall(i in TRAJECTORY_INDEX)"
                        "(do_2_opt /\ opt_index[1]<=i /\ i<=opt_index[2]->trajectories["
                        + str(vehicle + 1)
                        + ",i]==ref[opt_index[2]-i+opt_index[1]]);\n"
                    )
                    child.add_string(
                        "constraint forall(i in TRAJECTORY_INDEX)"
                        "(do_2_opt=false ->trajectories["
                        + str(vehicle + 1)
                        + ",i]==ref[i]);\n"
                    )
                for v in range(n_vehicles):
                    if v in [vehicles]:
                        continue
                    for index in range(number_of_nodes_transportation + 2):
                        child.add_string(
                            "constraint trajectories["
                            + str(v + 1)
                            + ","
                            + str(index + 1)
                            + "]=="
                            + str(trajo[v][index])
                            + ";\n"
                        )
            if i > 0:
                child.add_string(
                    "constraint distance<=" + str(int(last_result[1])) + ";\n"
                )
                child.add_string(
                    "constraint final_duration<=" + str(int(last_result[2])) + ";\n"
                )

            result = child.solve(
                timeout=timedelta(seconds=10),
                intermediate_solutions=True,
                free_search=False,
            )
            for ir in range(len(result)):
                results += [
                    (
                        result[ir].dict["trajectories"],
                        result[ir].dict["distance"],
                        result[ir].dict["final_duration"],
                    )
                ]
            print("Iter n°", i, " objective=", results[-1][1], results[-1][2])
    print(result.status)
    print("HEY")

    fig, ax = plt.subplots(1)
    for i in range(len(results)):
        traj = results[i][0]
        nb_colors = n_vehicles
        colors = plt.cm.get_cmap("hsv", 3 * nb_colors)
        for vehicle in range(len(traj)):
            color = colors(vehicle)
            for j in range(len(traj[vehicle]) - 1):
                ax.plot(
                    [
                        coordinates[traj[vehicle][j] - 1, 0],
                        coordinates[traj[vehicle][j + 1] - 1, 0],
                    ],
                    [
                        coordinates[traj[vehicle][j] - 1, 1],
                        coordinates[traj[vehicle][j + 1] - 1, 1],
                    ],
                    color=color,
                )
        ax.set_title("iter " + str(i) + " obj=" + str(int(results[i][1])))
        plt.draw()
        plt.pause(0.05)
        ax.clear()
    plt.show()


def script_example_flow():
    n_vehicles = 3
    number_of_nodes_transportation = 30
    coordinates = np.random.randint(
        -20, 20, size=(number_of_nodes_transportation + 2 * n_vehicles, 2)
    )
    distance_delta = dist.cdist(coordinates, coordinates)
    distance_delta = np.array(distance_delta, dtype=np.int_)
    for i in range(distance_delta.shape[0]):
        distance_delta[i, i] = 0
    time_delta = distance_delta / 2
    model = Model(files_mzn["gpdp-flow"])
    model.output_type = GPDPOutput
    solver = Solver.lookup("cbc")
    instance = Instance(solver, model)
    instance["number_vehicle"] = n_vehicles
    instance["number_of_nodes_transportation"] = number_of_nodes_transportation
    instance["distance_delta"] = [
        [int(distance_delta[i, j]) for j in range(distance_delta.shape[1])]
        for i in range(distance_delta.shape[0])
    ]
    instance["time_delta"] = [
        [int(time_delta[i, j]) for j in range(time_delta.shape[1])]
        for i in range(time_delta.shape[0])
    ]
    result = instance.solve(
        timeout=timedelta(seconds=100), intermediate_solutions=True, free_search=False
    )
    results = []
    paths = []
    for subresult in result:
        results += [(subresult.dict["flow"], subresult.objective)]
        flow = results[-1][0]
        path_dict = {}
        for vehicle in range(len(flow)):
            print("vehicle")
            path = []
            for i in range(len(flow[vehicle])):
                for j in range(len(flow[vehicle][i])):
                    if flow[vehicle][i][j]:
                        print(i, "->", j)
                        path += [(i, j)]
            path_dict[vehicle] = path
        paths += [path_dict]

    fig, ax = plt.subplots(1)

    for i in range(len(results)):
        path = paths[i]
        nb_colors = n_vehicles
        colors = plt.cm.get_cmap("hsv", 3 * nb_colors)
        for vehicle in path:
            color = colors(vehicle)
            path_v = path[vehicle]
            for j in range(len(path_v)):
                origin = path_v[j][0]
                destination = path_v[j][1]
                ax.plot(
                    [coordinates[origin, 0], coordinates[destination, 0]],
                    [coordinates[origin, 1], coordinates[destination, 1]],
                    color=color,
                )
        ax.set_title("iter " + str(i) + " obj=" + str(int(results[i][1])))
        plt.draw()
        plt.pause(0.05)
        ax.clear()
    plt.show()
    print(result.status)
    print("HEY")


def init_model_resources():
    n_vehicles = 5
    number_of_nodes_transportation = 20
    total_node = number_of_nodes_transportation + 2 * n_vehicles
    coordinates = np.random.randint(-20, 20, size=(total_node, 2))
    index_start = {v: number_of_nodes_transportation + v for v in range(n_vehicles)}
    distance_delta = dist.cdist(coordinates, coordinates)
    distance_delta = np.array(distance_delta, dtype=np.int_)
    for i in range(distance_delta.shape[0]):
        distance_delta[i, i] = 0
    time_delta = distance_delta / 2
    instance = {}
    nodes_transportation = list(range(1, number_of_nodes_transportation + 1))
    instance["number_vehicle"] = n_vehicles
    instance["number_of_nodes_transportation"] = number_of_nodes_transportation
    instance["distance_delta"] = [
        [int(distance_delta[i, j]) for j in range(distance_delta.shape[1])]
        for i in range(distance_delta.shape[0])
    ]
    instance["time_delta"] = [
        [int(time_delta[i, j]) for j in range(time_delta.shape[1])]
        for i in range(time_delta.shape[0])
    ]
    instance["number_resource"] = 2
    instance["resource_flow_node"] = np.zeros((total_node, instance["number_resource"]))
    instance["resource_flow_edges"] = np.zeros(
        (total_node, total_node, instance["number_resource"])
    )
    instance["max_capacity_resource"] = 200 * np.ones(
        (n_vehicles, instance["number_resource"]), dtype=np.int_
    )
    instance["min_capacity_resource"] = 10 * np.zeros(
        (n_vehicles, instance["number_resource"]), dtype=np.int_
    )
    instance["max_capacity_resource"] = [
        [
            int(instance["max_capacity_resource"][i, j])
            for j in range(instance["max_capacity_resource"].shape[1])
        ]
        for i in range(instance["max_capacity_resource"].shape[0])
    ]
    instance["min_capacity_resource"] = [
        [
            int(instance["min_capacity_resource"][i, j])
            for j in range(instance["min_capacity_resource"].shape[1])
        ]
        for i in range(instance["min_capacity_resource"].shape[0])
    ]
    for v in range(n_vehicles):
        instance["resource_flow_node"][
            index_start[v], 0
        ] = 80  # all vehicles loaded with 10
        instance["resource_flow_node"][index_start[v], 1] = 30
    instance["resource_flow_edges"][:, :, 0] = np.maximum(
        -distance_delta / 2, -5
    )  # negative flow for resource 0
    carburant_node = random.sample(range(number_of_nodes_transportation), n_vehicles)
    instance["cut_to_max"] = [True, False]
    instance["resource_flow_node"][
        0:number_of_nodes_transportation, 1
    ] = np.random.randint(
        -5, 5, size=number_of_nodes_transportation
    )  # random depot/delivery.
    for i in carburant_node:
        instance["resource_flow_node"][i, 0] = 20
    instance["resource_flow_node"] = [
        [
            int(instance["resource_flow_node"][i, j])
            for j in range(instance["resource_flow_node"].shape[1])
        ]
        for i in range(instance["resource_flow_node"].shape[0])
    ]
    instance["resource_flow_edges"] = [
        [
            [
                int(instance["resource_flow_edges"][i, j, k])
                for k in range(instance["resource_flow_edges"].shape[2])
            ]
            for j in range(instance["resource_flow_edges"].shape[1])
        ]
        for i in range(instance["resource_flow_edges"].shape[0])
    ]
    instance["precedence_pickup"] = []
    nb_request_transportation = 5
    instance["nb_request_transportation"] = nb_request_transportation
    instance["include_request_transportation"] = True
    instance["consider_resource"] = True
    chosen = set()
    for k in range(nb_request_transportation):
        prec = set(
            random.sample([n for n in nodes_transportation if n not in chosen], 1)
        )
        chosen.update(prec)
        succ = set(
            random.sample([n for n in nodes_transportation if n not in chosen], 1)
        )
        chosen.update(succ)
        instance["precedence_pickup"] += [[prec, succ]]
    instance["weight_objective"] = [10, 1]  # final duration, distance
    return instance, coordinates


def init_model_ortools():
    data = {
        "distance_matrix": [
            [
                0,
                548,
                776,
                696,
                582,
                274,
                502,
                194,
                308,
                194,
                536,
                502,
                388,
                354,
                468,
                776,
                662,
            ],
            [
                548,
                0,
                684,
                308,
                194,
                502,
                730,
                354,
                696,
                742,
                1084,
                594,
                480,
                674,
                1016,
                868,
                1210,
            ],
            [
                776,
                684,
                0,
                992,
                878,
                502,
                274,
                810,
                468,
                742,
                400,
                1278,
                1164,
                1130,
                788,
                1552,
                754,
            ],
            [
                696,
                308,
                992,
                0,
                114,
                650,
                878,
                502,
                844,
                890,
                1232,
                514,
                628,
                822,
                1164,
                560,
                1358,
            ],
            [
                582,
                194,
                878,
                114,
                0,
                536,
                764,
                388,
                730,
                776,
                1118,
                400,
                514,
                708,
                1050,
                674,
                1244,
            ],
            [
                274,
                502,
                502,
                650,
                536,
                0,
                228,
                308,
                194,
                240,
                582,
                776,
                662,
                628,
                514,
                1050,
                708,
            ],
            [
                502,
                730,
                274,
                878,
                764,
                228,
                0,
                536,
                194,
                468,
                354,
                1004,
                890,
                856,
                514,
                1278,
                480,
            ],
            [
                194,
                354,
                810,
                502,
                388,
                308,
                536,
                0,
                342,
                388,
                730,
                468,
                354,
                320,
                662,
                742,
                856,
            ],
            [
                308,
                696,
                468,
                844,
                730,
                194,
                194,
                342,
                0,
                274,
                388,
                810,
                696,
                662,
                320,
                1084,
                514,
            ],
            [
                194,
                742,
                742,
                890,
                776,
                240,
                468,
                388,
                274,
                0,
                342,
                536,
                422,
                388,
                274,
                810,
                468,
            ],
            [
                536,
                1084,
                400,
                1232,
                1118,
                582,
                354,
                730,
                388,
                342,
                0,
                878,
                764,
                730,
                388,
                1152,
                354,
            ],
            [
                502,
                594,
                1278,
                514,
                400,
                776,
                1004,
                468,
                810,
                536,
                878,
                0,
                114,
                308,
                650,
                274,
                844,
            ],
            [
                388,
                480,
                1164,
                628,
                514,
                662,
                890,
                354,
                696,
                422,
                764,
                114,
                0,
                194,
                536,
                388,
                730,
            ],
            [
                354,
                674,
                1130,
                822,
                708,
                628,
                856,
                320,
                662,
                388,
                730,
                308,
                194,
                0,
                342,
                422,
                536,
            ],
            [
                468,
                1016,
                788,
                1164,
                1050,
                514,
                514,
                662,
                320,
                274,
                388,
                650,
                536,
                342,
                0,
                764,
                194,
            ],
            [
                776,
                868,
                1552,
                560,
                674,
                1050,
                1278,
                742,
                1084,
                810,
                1152,
                274,
                388,
                422,
                764,
                0,
                798,
            ],
            [
                662,
                1210,
                754,
                1358,
                1244,
                708,
                480,
                856,
                514,
                468,
                354,
                844,
                730,
                536,
                194,
                798,
                0,
            ],
        ],
        "pickups_deliveries": [
            [1, 6],
            [2, 10],
            [4, 3],
            [5, 9],
            [7, 8],
            [15, 11],
            [13, 12],
            [16, 14],
        ],
    }
    dmatrix = list(data["distance_matrix"])
    data["num_vehicles"] = 4
    for j in range(len(dmatrix)):
        original_length = len(dmatrix[j])
        dmatrix[j] += [0] * 2 * data["num_vehicles"]
        for i in range(original_length, len(dmatrix[j])):
            dmatrix[j][i] = dmatrix[j][0]
        dmatrix[j] = dmatrix[j][1:]
    dmatrix = dmatrix[1:]
    number_of_nodes_transportation = len(dmatrix)
    for k in range(2 * data["num_vehicles"]):
        l = []
        for i in range(number_of_nodes_transportation):
            l += [dmatrix[i][number_of_nodes_transportation + k]]
        l += [0] * (2 * data["num_vehicles"])
        dmatrix += [l]
    data["depot"] = 0
    instance = {
        "number_vehicle": data["num_vehicles"],
        "number_of_nodes_transportation": number_of_nodes_transportation,
        "distance_delta": dmatrix,
    }
    distance_delta = np.array(dmatrix)
    instance["time_delta"] = [
        [int(dmatrix[i][j]) for j in range(len(dmatrix[i]))]
        for i in range(len(dmatrix))
    ]
    total_node = number_of_nodes_transportation + 2 * data["num_vehicles"]
    n_vehicles = data["num_vehicles"]
    instance["number_resource"] = 2
    instance["resource_flow_node"] = np.zeros((total_node, instance["number_resource"]))
    instance["resource_flow_edges"] = np.zeros(
        (total_node, total_node, instance["number_resource"])
    )
    instance["max_capacity_resource"] = 200 * np.ones(
        (n_vehicles, instance["number_resource"]), dtype=np.int_
    )
    instance["min_capacity_resource"] = 10 * np.zeros(
        (n_vehicles, instance["number_resource"]), dtype=np.int_
    )
    instance["max_capacity_resource"] = [
        [
            int(instance["max_capacity_resource"][i, j])
            for j in range(instance["max_capacity_resource"].shape[1])
        ]
        for i in range(instance["max_capacity_resource"].shape[0])
    ]
    instance["min_capacity_resource"] = [
        [
            int(instance["min_capacity_resource"][i, j])
            for j in range(instance["min_capacity_resource"].shape[1])
        ]
        for i in range(instance["min_capacity_resource"].shape[0])
    ]
    instance["weight_objective"] = [10, 1]
    index_start = {v: number_of_nodes_transportation + v for v in range(n_vehicles)}
    for v in range(n_vehicles):
        instance["resource_flow_node"][
            index_start[v], 0
        ] = 80  # all vehicles loaded with 10
        instance["resource_flow_node"][index_start[v], 1] = 30
    instance["resource_flow_edges"][:, :, 0] = np.maximum(
        -distance_delta / 2, -5
    )  # negative flow for resource 0
    carburant_node = random.sample(range(number_of_nodes_transportation), n_vehicles)
    instance["cut_to_max"] = [True, False]
    instance["resource_flow_node"][
        0:number_of_nodes_transportation, 1
    ] = np.random.randint(
        -5, 5, size=number_of_nodes_transportation
    )  # random depot/delivery.
    for i in carburant_node:
        instance["resource_flow_node"][i, 0] = 2
    instance["resource_flow_node"] = [
        [
            int(instance["resource_flow_node"][i, j])
            for j in range(instance["resource_flow_node"].shape[1])
        ]
        for i in range(instance["resource_flow_node"].shape[0])
    ]
    instance["resource_flow_edges"] = [
        [
            [
                int(instance["resource_flow_edges"][i, j, k])
                for k in range(instance["resource_flow_edges"].shape[2])
            ]
            for j in range(instance["resource_flow_edges"].shape[1])
        ]
        for i in range(instance["resource_flow_edges"].shape[0])
    ]
    instance["precedence_pickup"] = [
        [{data["pickups_deliveries"][i][0]}, {data["pickups_deliveries"][i][1]}]
        for i in range(len(data["pickups_deliveries"]))
    ]
    nb_request_transportation = len(data["pickups_deliveries"])
    instance["nb_request_transportation"] = nb_request_transportation
    instance["include_request_transportation"] = True
    instance["consider_resource"] = False
    instance["force_all_use_of_vehicle"] = True
    coordinates = np.array(
        [
            [-2, 4],
            [4, 4],
            [-4, 3],
            [-3, 3],
            [1, 2],
            [3, 2],
            [-1, 1],
            [2, 1],
            [1, -1],
            [4, -1],
            [-3, -2],
            [-2, -2],
            [-1, -3],
            [2, -3],
            [-4, -4],
            [3, -4],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    )
    return instance, coordinates


def transform_ortools(distance_metric, nvehicle):
    dmatrix = list(distance_metric)
    for j in range(len(dmatrix)):
        original_length = len(dmatrix[j])
        dmatrix[j] += [0] * 2 * nvehicle
        for i in range(original_length, len(dmatrix[j])):
            dmatrix[j][i] = dmatrix[j][0]
        dmatrix[j] = dmatrix[j][1:]
    dmatrix = dmatrix[1:]
    number_of_nodes_transportation = len(dmatrix)
    for k in range(2 * nvehicle):
        l = []
        for i in range(number_of_nodes_transportation):
            l += [dmatrix[i][number_of_nodes_transportation + k]]
        l += [0] * (2 * nvehicle)
        dmatrix += [l]
    return dmatrix


def init_model_ortools_tsp():
    data = {
        "locations": [
            (288, 149),
            (288, 129),
            (270, 133),
            (256, 141),
            (256, 157),
            (246, 157),
            (236, 169),
            (228, 169),
            (228, 161),
            (220, 169),
            (212, 169),
            (204, 169),
            (196, 169),
            (188, 169),
            (196, 161),
            (188, 145),
            (172, 145),
            (164, 145),
            (156, 145),
            (148, 145),
            (140, 145),
            (148, 169),
            (164, 169),
            (172, 169),
            (156, 169),
            (140, 169),
            (132, 169),
            (124, 169),
            (116, 161),
            (104, 153),
            (104, 161),
            (104, 169),
            (90, 165),
            (80, 157),
            (64, 157),
            (64, 165),
            (56, 169),
            (56, 161),
            (56, 153),
            (56, 145),
            (56, 137),
            (56, 129),
            (56, 121),
            (40, 121),
            (40, 129),
            (40, 137),
            (40, 145),
            (40, 153),
            (40, 161),
            (40, 169),
            (32, 169),
            (32, 161),
            (32, 153),
            (32, 145),
            (32, 137),
            (32, 129),
            (32, 121),
            (32, 113),
            (40, 113),
            (56, 113),
            (56, 105),
            (48, 99),
            (40, 99),
            (32, 97),
            (32, 89),
            (24, 89),
            (16, 97),
            (16, 109),
            (8, 109),
            (8, 97),
            (8, 89),
            (8, 81),
            (8, 73),
            (8, 65),
            (8, 57),
            (16, 57),
            (8, 49),
            (8, 41),
            (24, 45),
            (32, 41),
            (32, 49),
            (32, 57),
            (32, 65),
            (32, 73),
            (32, 81),
            (40, 83),
            (40, 73),
            (40, 63),
            (40, 51),
            (44, 43),
            (44, 35),
            (44, 27),
            (32, 25),
            (24, 25),
            (16, 25),
            (16, 17),
            (24, 17),
            (32, 17),
            (44, 11),
            (56, 9),
            (56, 17),
            (56, 25),
            (56, 33),
            (56, 41),
            (64, 41),
            (72, 41),
            (72, 49),
            (56, 49),
            (48, 51),
            (56, 57),
            (56, 65),
            (48, 63),
            (48, 73),
            (56, 73),
            (56, 81),
            (48, 83),
            (56, 89),
            (56, 97),
            (104, 97),
            (104, 105),
            (104, 113),
            (104, 121),
            (104, 129),
            (104, 137),
            (104, 145),
            (116, 145),
            (124, 145),
            (132, 145),
            (132, 137),
            (140, 137),
            (148, 137),
            (156, 137),
            (164, 137),
            (172, 125),
            (172, 117),
            (172, 109),
            (172, 101),
            (172, 93),
            (172, 85),
            (180, 85),
            (180, 77),
            (180, 69),
            (180, 61),
            (180, 53),
            (172, 53),
            (172, 61),
            (172, 69),
            (172, 77),
            (164, 81),
            (148, 85),
            (124, 85),
            (124, 93),
            (124, 109),
            (124, 125),
            (124, 117),
            (124, 101),
            (104, 89),
            (104, 81),
            (104, 73),
            (104, 65),
            (104, 49),
            (104, 41),
            (104, 33),
            (104, 25),
            (104, 17),
            (92, 9),
            (80, 9),
            (72, 9),
            (64, 21),
            (72, 25),
            (80, 25),
            (80, 25),
            (80, 41),
            (88, 49),
            (104, 57),
            (124, 69),
            (124, 77),
            (132, 81),
            (140, 65),
            (132, 61),
            (124, 61),
            (124, 53),
            (124, 45),
            (124, 37),
            (124, 29),
            (132, 21),
            (124, 21),
            (120, 9),
            (128, 9),
            (136, 9),
            (148, 9),
            (162, 9),
            (156, 25),
            (172, 21),
            (180, 21),
            (180, 29),
            (172, 29),
            (172, 37),
            (172, 45),
            (180, 45),
            (180, 37),
            (188, 41),
            (196, 49),
            (204, 57),
            (212, 65),
            (220, 73),
            (228, 69),
            (228, 77),
            (236, 77),
            (236, 69),
            (236, 61),
            (228, 61),
            (228, 53),
            (236, 53),
            (236, 45),
            (228, 45),
            (228, 37),
            (236, 37),
            (236, 29),
            (228, 29),
            (228, 21),
            (236, 21),
            (252, 21),
            (260, 29),
            (260, 37),
            (260, 45),
            (260, 53),
            (260, 61),
            (260, 69),
            (260, 77),
            (276, 77),
            (276, 69),
            (276, 61),
            (276, 53),
            (284, 53),
            (284, 61),
            (284, 69),
            (284, 77),
            (284, 85),
            (284, 93),
            (284, 101),
            (288, 109),
            (280, 109),
            (276, 101),
            (276, 93),
            (276, 85),
            (268, 97),
            (260, 109),
            (252, 101),
            (260, 93),
            (260, 85),
            (236, 85),
            (228, 85),
            (228, 93),
            (236, 93),
            (236, 101),
            (228, 101),
            (228, 109),
            (228, 117),
            (228, 125),
            (220, 125),
            (212, 117),
            (204, 109),
            (196, 101),
            (188, 93),
            (180, 93),
            (180, 101),
            (180, 109),
            (180, 117),
            (180, 125),
            (196, 145),
            (204, 145),
            (212, 145),
            (220, 145),
            (228, 145),
            (236, 145),
            (246, 141),
            (252, 125),
            (260, 129),
            (280, 133),
        ],
        "num_vehicles": 1,
    }
    # Locations in block units
    n_vehicles = 1
    instance = {}

    def compute_euclidean_distance_matrix(locations):
        """Creates callback to return distance between points."""
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    # Euclidean distance
                    distances[from_counter][to_counter] = int(
                        math.hypot(
                            (from_node[0] - to_node[0]), (from_node[1] - to_node[1])
                        )
                    )
        return distances

    distance = compute_euclidean_distance_matrix(locations=data["locations"])
    distance = [[distance[i][j] for j in sorted(distance[i])] for i in sorted(distance)]
    dmatrix = transform_ortools(distance, 1)
    instance["number_vehicle"] = data["num_vehicles"]
    number_of_nodes_transportation = len(data["locations"]) - 1
    instance["number_of_nodes_transportation"] = number_of_nodes_transportation
    instance["distance_delta"] = dmatrix
    distance_delta = np.array(dmatrix)
    instance["time_delta"] = [
        [int(dmatrix[i][j]) for j in range(len(dmatrix[i]))]
        for i in range(len(dmatrix))
    ]
    total_node = number_of_nodes_transportation + 2 * data["num_vehicles"]
    instance["number_resource"] = 2
    instance["resource_flow_node"] = np.zeros((total_node, instance["number_resource"]))
    instance["resource_flow_edges"] = np.zeros(
        (total_node, total_node, instance["number_resource"])
    )
    instance["max_capacity_resource"] = 200 * np.ones(
        (n_vehicles, instance["number_resource"]), dtype=np.int_
    )
    instance["min_capacity_resource"] = 10 * np.zeros(
        (n_vehicles, instance["number_resource"]), dtype=np.int_
    )
    instance["max_capacity_resource"] = [
        [
            int(instance["max_capacity_resource"][i, j])
            for j in range(instance["max_capacity_resource"].shape[1])
        ]
        for i in range(instance["max_capacity_resource"].shape[0])
    ]
    instance["min_capacity_resource"] = [
        [
            int(instance["min_capacity_resource"][i, j])
            for j in range(instance["min_capacity_resource"].shape[1])
        ]
        for i in range(instance["min_capacity_resource"].shape[0])
    ]
    instance["weight_objective"] = [10, 1]
    index_start = {v: number_of_nodes_transportation + v for v in range(n_vehicles)}
    for v in range(n_vehicles):
        instance["resource_flow_node"][
            index_start[v], 0
        ] = 80  # all vehicles loaded with 10
        instance["resource_flow_node"][index_start[v], 1] = 30
    instance["resource_flow_edges"][:, :, 0] = np.maximum(
        -distance_delta / 2, -5
    )  # negative flow for resource 0
    carburant_node = random.sample(range(number_of_nodes_transportation), n_vehicles)
    instance["cut_to_max"] = [True, False]
    instance["resource_flow_node"][
        0:number_of_nodes_transportation, 1
    ] = np.random.randint(
        -5, 5, size=number_of_nodes_transportation
    )  # random depot/delivery.
    for i in carburant_node:
        instance["resource_flow_node"][i, 0] = 2
    instance["resource_flow_node"] = [
        [
            int(instance["resource_flow_node"][i, j])
            for j in range(instance["resource_flow_node"].shape[1])
        ]
        for i in range(instance["resource_flow_node"].shape[0])
    ]
    instance["resource_flow_edges"] = [
        [
            [
                int(instance["resource_flow_edges"][i, j, k])
                for k in range(instance["resource_flow_edges"].shape[2])
            ]
            for j in range(instance["resource_flow_edges"].shape[1])
        ]
        for i in range(instance["resource_flow_edges"].shape[0])
    ]
    instance["precedence_pickup"] = []
    nb_request_transportation = 0
    instance["nb_request_transportation"] = nb_request_transportation
    instance["include_request_transportation"] = False
    instance["consider_resource"] = False
    instance["force_all_use_of_vehicle"] = True
    coordinates = data["locations"]
    coordinates += [data["locations"][0], data["locations"][0]]
    coordinates = coordinates[1:]
    coordinates = np.array(coordinates)
    return instance, coordinates


def retrieve_solution_resource(result_minizinc):
    results = []
    for i in range(len(result_minizinc)):
        results += [{k: result_minizinc[i].dict[k] for k in result_minizinc[i].dict}]
    return results


def do_lns(
    child, last_result, n_vehicles, number_of_nodes_transportation, proportion_fix=0.5
):
    indexes = [
        (v, index)
        for v in range(n_vehicles)
        for index in range(number_of_nodes_transportation + 2)
    ]
    trajo = last_result["trajectories"]
    if random.random() <= 0.9:
        indexes_fix = random.sample(indexes, int(proportion_fix * len(indexes)))
        for v, index in indexes_fix:
            if trajo[v][index] != number_of_nodes_transportation + n_vehicles + v + 1:
                child.add_string(
                    "constraint trajectories["
                    + str(v + 1)
                    + ","
                    + str(index + 1)
                    + "]=="
                    + str(trajo[v][index])
                    + ";\n"
                )
    else:
        print("METHOD 2")
        vehicles = random.choice(range(n_vehicles))
        for vehicle in [vehicles]:
            traj = trajo[vehicle]
            nodes = [
                n
                for n in traj
                if n != number_of_nodes_transportation + vehicle + 1
                and n != number_of_nodes_transportation + n_vehicles + vehicle + 1
            ]
            for n in nodes:
                child.add_string(
                    "constraint "
                    + str(n)
                    + " in {trajectories["
                    + str(vehicle + 1)
                    + ", i]|i in TRAJECTORY_INDEX};\n"
                )
            child.add_string(
                "array[TRAJECTORY_INDEX] of ALL_NODES: ref=" + str(traj) + ";\n"
            )
            child.add_string("array[1..2] of var TRAJECTORY_INDEX: opt_index;\n")
            child.add_string("constraint opt_index[2]>opt_index[1]+1;\n")
            child.add_string("var bool: do_2_opt;")
            child.add_string(
                "constraint forall(i in TRAJECTORY_INDEX)"
                "(do_2_opt /\ opt_index[1]<=i /\ i<=opt_index[2]->trajectories["
                + str(vehicle + 1)
                + ",i]==ref[opt_index[2]-i+opt_index[1]]);\n"
            )
            child.add_string(
                "constraint forall(i in TRAJECTORY_INDEX)"
                "(do_2_opt=false ->trajectories[" + str(vehicle + 1) + ",i]==ref[i]);\n"
            )
        for v in range(n_vehicles):
            if v in [vehicles]:
                continue
            for index in range(number_of_nodes_transportation + 2):
                child.add_string(
                    "constraint trajectories["
                    + str(v + 1)
                    + ","
                    + str(index + 1)
                    + "]=="
                    + str(trajo[v][index])
                    + ";\n"
                )


class Example(Enum):
    RANDOM = 0
    PICKUP = 1
    TSP = 2


def run_resource(version: Example = Example.TSP):
    if version == Example.RANDOM:
        dict_instance, coordinates = init_model_resources()
    if version == Example.PICKUP:
        dict_instance, coordinates = init_model_ortools()
    if version == Example.TSP:
        dict_instance, coordinates = init_model_ortools_tsp()
    model = Model(files_mzn["gpdp-resources"])
    model.output_type = GPDPOutput
    solver = Solver.lookup("chuffed")
    instance = Instance(solver, model)
    for k in dict_instance:
        instance[k] = dict_instance[k]

    def get_dummy():
        number_of_nodes_transportation = dict_instance["number_of_nodes_transportation"]
        n_vehicles = dict_instance["number_vehicle"]
        dummy_solution = [
            [
                dict_instance["number_of_nodes_transportation"]
                + dict_instance["number_vehicle"]
                + v
                + 1
            ]
            * (dict_instance["number_of_nodes_transportation"] + 2)
            for v in range(dict_instance["number_vehicle"])
        ]
        for v in range(dict_instance["number_vehicle"]):
            dummy_solution[v][0] = number_of_nodes_transportation + v + 1
        cut_part = list(range(number_of_nodes_transportation))

        def chunks(l, n):
            n = max(1, n)
            return [l[i : min(i + n, len(l))] for i in range(0, len(l), n)]

        cut_parts = chunks(cut_part, int(math.ceil(len(cut_part) / n_vehicles)))
        for v in range(len(cut_parts)):
            indexes = list(range(len(cut_parts[v])))
            random.shuffle(indexes)
            for j in range(len(indexes)):
                dummy_solution[v][j + 1] = cut_parts[v][indexes[j]] + 1
        dist = sum(
            [
                dict_instance["distance_delta"][t1 - 1][t2 - 1]
                for t1, t2 in zip(dummy_solution[v][:-1], dummy_solution[v][1:])
                for v in range(len(dummy_solution))
            ]
        )
        time = max(
            [
                sum(
                    [
                        dict_instance["time_delta"][t1 - 1][t2 - 1]
                        for t1, t2 in zip(dummy_solution[v][:-1], dummy_solution[v][1:])
                    ]
                )
                for v in range(len(dummy_solution))
            ]
        )
        return [
            {"trajectories": dummy_solution, "final_duration": time, "distance": dist}
        ]

    if version in {Example.RANDOM, Example.PICKUP}:
        result = instance.solve(
            timeout=timedelta(seconds=300),
            intermediate_solutions=True,
            free_search=False,
        )
        print(result.status)
        results = retrieve_solution_resource(result_minizinc=result)
    else:
        results = get_dummy()

    for i in range(100):
        with instance.branch() as child:
            do_lns(
                child=child,
                last_result=results[-1],
                n_vehicles=dict_instance["number_vehicle"],
                number_of_nodes_transportation=dict_instance[
                    "number_of_nodes_transportation"
                ],
                proportion_fix=0.85,
            )
            res = child.solve(
                timeout=timedelta(seconds=100),
                intermediate_solutions=True,
                free_search=False,
                optimisation_level=2,
            )
            print("Iter n°", i)
            print(res.status)
            results += retrieve_solution_resource(res)
            try:
                print(res.objective)
                trajo = results[-1]["trajectories"]
                total_dist = 0
                for v in range(len(trajo)):
                    print("Vehicle ", v)
                    print(
                        "->".join(
                            [
                                str(trajo[v][j])
                                for j in range(len(trajo[v]))
                                if trajo[v][j]
                                != v
                                + 1
                                + dict_instance["number_of_nodes_transportation"]
                                + dict_instance["number_vehicle"]
                            ]
                        )
                    )
                    dist = sum(
                        [
                            dict_instance["distance_delta"][t1 - 1][t2 - 1]
                            for t1, t2 in zip(trajo[v][:-1], trajo[v][1:])
                        ]
                    )
                    total_dist += dist
                    print("Dist : ", dist)
                print("Total dist ", total_dist)
            except:
                pass

    fig, ax = plt.subplots(1, 2)
    for i in range(len(results)):
        traj = results[i]["trajectories"]
        nb_colors = dict_instance["number_vehicle"]
        colors = plt.cm.get_cmap("hsv", 2 * nb_colors)
        for vehicle in range(len(traj)):
            color = colors(vehicle)
            for j in range(len(traj[vehicle]) - 1):
                ax[0].plot(
                    [
                        coordinates[traj[vehicle][j] - 1, 0],
                        coordinates[traj[vehicle][j + 1] - 1, 0],
                    ],
                    [
                        coordinates[traj[vehicle][j] - 1, 1],
                        coordinates[traj[vehicle][j + 1] - 1, 1],
                    ],
                    color=color,
                )
            for resource in range(len(dict_instance["max_capacity_resource"][0])):
                ax[1].plot(
                    results[i]["resource_leaving_node"][vehicle][resource][:],
                    color=color,
                    label="load resource:" + str(i) + " vehicle: " + str(vehicle),
                )
                ax[1].axhline(
                    dict_instance["max_capacity_resource"][vehicle][resource],
                    label="max limit resource " + str(i),
                )
                ax[1].axhline(
                    dict_instance["min_capacity_resource"][vehicle][resource],
                    linestyle="--",
                    label="min limit resource " + str(i),
                )
        ax[0].set_title("iter " + str(i) + " obj=" + str(int(results[i]["distance"])))
        plt.draw()
        plt.pause(0.5)
        ax[0].lines = []
        ax[1].lines = []
    plt.show()


def run_tsp():
    dict_instance, coordinates = init_model_ortools_tsp()
    tsp_model = TSPModelDistanceMatrix(
        list_points=[Point2D(x=x[0], y=x[1]) for x in coordinates],
        distance_matrix=np.array(dict_instance["distance_delta"]),
        node_count=len(dict_instance["distance_delta"]),
        start_index=len(dict_instance["distance_delta"]) - 2,
        end_index=len(dict_instance["distance_delta"]) - 1,
    )
    solution = tsp_model.get_random_dummy_solution()
    _, list_mutation = get_available_mutations(tsp_model, solution)
    res = RestartHandlerLimit(3000)
    print(list_mutation)
    list_mutation = [
        mutate[0].build(tsp_model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [Mutation2Opt, MutationSwapTSP]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        problem=tsp_model,
        mutator=mutate_portfolio,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=10, restart_handler=res, coefficient=0.99999
        ),
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
    )
    results = sa.solve(solution, nb_iteration_max=10000)
    best_solution, fit = results.get_best_solution_fit()
    print("Fit ", fit)
    plot_tsp_solution(tsp_model=tsp_model, solution=best_solution)
    plt.show()


if __name__ == "__main__":
    run_tsp()
    script_example()
    script_example_lns()
    script_example_flow()
