#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple, cast

import numpy as np
import scipy.spatial.distance as dist

import discrete_optimization.tsp.tsp_parser as tsp_parser
import discrete_optimization.vrp.vrp_parser as vrp_parser
from discrete_optimization.pickup_vrp.gpdp import GPDP, Edge, Node, ProxyClass

logger = logging.getLogger(__name__)


try:
    from sklearn.cluster import KMeans
except ImportError:
    logger.warning(
        "You need to install scikit-learn "
        "to call create_selective_tsp() and create_pickup_and_delivery()"
    )


def create_selective_tsp(
    nb_nodes: int = 300, nb_vehicles: int = 1, nb_clusters: int = 30
) -> GPDP:
    """
    Create a random orienteering/selective TSP problem
    :param nb_nodes: number of nodes to consider in the network (excluding the origin/target of vehicles)
    :param nb_vehicles: number of vehicles to consider
    :param nb_clusters: number of cluster of nodes.
    :return: a gpdp model
    """
    number_vehicle = nb_vehicles
    nb_nodes_transportation = nb_nodes
    # non dummy nodes
    nodes_transportation: Set[Node] = set(range(nb_nodes_transportation))
    # we stack the origin nodes after
    nodes_origin: Set[Node] = {
        nb_nodes_transportation + i for i in range(number_vehicle)
    }
    # and destination nodes too
    nodes_target: Set[Node] = {
        nb_nodes_transportation + number_vehicle + i for i in range(number_vehicle)
    }
    list_pickup_deliverable: List[Tuple[List[Node], List[Node]]] = []
    origin_vehicle: Dict[int, Node] = {
        i: nb_nodes_transportation + i for i in range(number_vehicle)
    }
    target_vehicle: Dict[int, Node] = {
        i: nb_nodes_transportation + number_vehicle + i for i in range(number_vehicle)
    }
    # we don't include resources neither capacities
    resources_set: Set[str] = set()
    capacities: Dict[int, Dict[str, Tuple[float, float]]] = {
        i: {} for i in range(number_vehicle)
    }
    resources_flow_node: Dict[Node, Dict[str, float]] = {
        i: {} for i in range(nb_nodes_transportation)
    }
    resources_flow_edges: Dict[Edge, Dict[str, float]] = {
        (i, j): {}
        for i in range(nb_nodes_transportation)
        for j in range(nb_nodes_transportation)
        if j != i
    }

    # real number of nodes in the problem definition
    nb_nodes_real = nb_nodes_transportation + 2 * number_vehicle
    # we sample random 2D coordinates
    coordinates = np.random.randint(-20, 20, size=(nb_nodes_real, 2))
    coordinates[:, 0] += 40
    distance_delta = dist.cdist(coordinates, coordinates)
    distance_delta = np.array(distance_delta, dtype=np.int32)
    distance_delta_dict: Dict[Node, Dict[Node, float]] = {
        i: {j: int(distance_delta[i, j]) for j in range(nb_nodes_real) if j != i}
        for i in range(nb_nodes_real)
    }
    time_delta_dict: Dict[Node, Dict[Node, float]] = {
        i: {j: int(distance_delta_dict[i][j] / 2) for j in distance_delta_dict[i]}
        for i in distance_delta_dict
    }

    nb_clusters = nb_clusters
    # compute clusters based on geographical positions.
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(coordinates)
    labels = kmeans.labels_
    coordinates_2d: Dict[Node, Tuple[float, float]] = {
        i: cast(Tuple[float, float], tuple(coordinates[i, :]))
        for i in range(coordinates.shape[0])
    }
    clusters_dict: Dict[Node, Hashable] = {i: i + 1 for i in range(nb_nodes_real)}
    for i in clusters_dict:
        clusters_dict[i] = labels[i]
    for j in range(number_vehicle):
        clusters_dict[target_vehicle[j]] = max(labels)
        clusters_dict[origin_vehicle[j]] = min(labels)
    return GPDP(
        number_vehicle=number_vehicle,
        nodes_transportation=nodes_transportation,
        nodes_origin=nodes_origin,
        nodes_target=nodes_target,
        list_pickup_deliverable=list_pickup_deliverable,
        origin_vehicle=origin_vehicle,
        target_vehicle=target_vehicle,
        resources_set=resources_set,
        capacities=capacities,
        resources_flow_edges=resources_flow_edges,
        resources_flow_node=resources_flow_node,
        distance_delta=distance_delta_dict,
        time_delta=time_delta_dict,
        coordinates_2d=coordinates_2d,
        clusters_dict=clusters_dict,
    )


def create_pickup_and_delivery(
    number_of_vehicles: int = 1,
    number_of_node: int = 100,
    include_cluster: bool = False,
    nb_clusters: int = 10,
    include_pickup: bool = True,
    fraction_of_pickup_deliver: float = 0.125,
    pickup_per_cluster: bool = False,
) -> GPDP:
    number_vehicle = number_of_vehicles
    nb_nodes_transportation = number_of_node
    nodes_transportation: Set[Node] = set(range(nb_nodes_transportation))
    nodes_origin: Set[Node] = {
        nb_nodes_transportation + i for i in range(number_vehicle)
    }
    # and destination nodes too
    nodes_target: Set[Node] = {
        nb_nodes_transportation + number_vehicle + i for i in range(number_vehicle)
    }
    list_pickup_deliverable: List[Tuple[List[Node], List[Node]]] = []
    origin_vehicle: Dict[int, Node] = {
        i: nb_nodes_transportation + i for i in range(number_vehicle)
    }
    target_vehicle: Dict[int, Node] = {
        i: nb_nodes_transportation + number_vehicle + i for i in range(number_vehicle)
    }

    all_nodes = set(range(nb_nodes_transportation + 2 * number_vehicle))
    resources_set: Set[str] = set()
    capacities: Dict[int, Dict[str, Tuple[float, float]]] = {0: {}}
    resources_flow_node: Dict[Node, Dict[str, float]] = {i: {} for i in all_nodes}
    resources_flow_edges: Dict[Edge, Dict[str, float]] = {
        (i, j): {} for i in all_nodes for j in all_nodes if j != i
    }

    coordinates = np.random.randint(-20, 20, size=(len(all_nodes), 2))
    coordinates_2d: Optional[Dict[Node, Tuple[float, float]]] = {
        i: cast(Tuple[float, float], tuple(coordinates[i, :]))
        for i in range(coordinates.shape[0])
    }

    distance_delta = dist.cdist(coordinates, coordinates)
    distance_delta = np.array(distance_delta, dtype=np.int32)
    distance_delta_dict: Dict[Node, Dict[Node, float]] = {
        i: {j: int(distance_delta[i, j]) for j in all_nodes if j != i}
        for i in all_nodes
    }
    time_delta_dict: Dict[Node, Dict[Node, float]] = {
        i: {j: int(distance_delta_dict[i][j] / 2) for j in distance_delta_dict[i]}
        for i in distance_delta_dict
    }

    clusters_dict: Dict[Node, Hashable] = {i: i + 1 for i in all_nodes}
    if include_cluster:
        nb_clusters = nb_clusters
        kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(coordinates)
        labels = kmeans.labels_
        for i in clusters_dict:
            clusters_dict[i] = labels[i]
        for j in range(number_vehicle):
            clusters_dict[target_vehicle[j]] = max(labels)
            clusters_dict[origin_vehicle[j]] = min(labels)
    list_pickup_deliverable_per_cluster: List[Tuple[List[Node], List[Node]]] = []
    if include_pickup:
        if not pickup_per_cluster:
            nodes_possible = set(clusters_dict.keys())
            for node_origin in nodes_origin:
                nodes_possible.remove(node_origin)
            for node_target in nodes_target:
                nodes_possible.remove(node_target)
            n = int(len(nodes_possible) * fraction_of_pickup_deliver)
            for j in range(n):
                k1 = random.choice(list(nodes_possible))
                nodes_possible.remove(k1)
                k2 = random.choice(list(nodes_possible))
                nodes_possible.remove(k2)
                list_pickup_deliverable += [([k1], [k2])]
        else:
            cluster_possible = set(clusters_dict.values())
            cluster_possible.remove(clusters_dict[target_vehicle[0]])
            for j in range(number_vehicle):
                if origin_vehicle[j] in cluster_possible:
                    cluster_possible.remove(clusters_dict[origin_vehicle[j]])
                if target_vehicle[j] in cluster_possible:
                    cluster_possible.remove(clusters_dict[target_vehicle[j]])
            n = len(cluster_possible) // 4
            for j in range(n):
                cluster_1 = random.choice(list(cluster_possible))
                cluster_possible.remove(cluster_1)
                cluster_2 = random.choice(list(cluster_possible))
                cluster_possible.remove(cluster_2)
                list_pickup_deliverable_per_cluster += [([cluster_1], [cluster_2])]
    return GPDP(
        number_vehicle=number_vehicle,
        nodes_transportation=nodes_transportation,
        nodes_origin=nodes_origin,
        nodes_target=nodes_target,
        list_pickup_deliverable=list_pickup_deliverable,
        origin_vehicle=origin_vehicle,
        target_vehicle=target_vehicle,
        resources_set=resources_set,
        capacities=capacities,
        resources_flow_edges=resources_flow_edges,
        resources_flow_node=resources_flow_node,
        distance_delta=distance_delta_dict,
        time_delta=time_delta_dict,
        coordinates_2d=coordinates_2d,
        clusters_dict=clusters_dict,
        list_pickup_deliverable_per_cluster=list_pickup_deliverable_per_cluster,
    )


def load_vrp_and_transform(index_in_files_available: int = 1) -> GPDP:
    file_path = vrp_parser.get_data_available()[index_in_files_available]
    vrp_model = vrp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_vrp_model_to_gpdp(vrp_model=vrp_model)
    return gpdp


def load_tsp_and_transform(index_in_files_available: int = 1) -> GPDP:
    files_available = tsp_parser.get_data_available()
    file_path = files_available[index_in_files_available]
    tsp_model = tsp_parser.parse_file(file_path)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=tsp_model)
    return gpdp


def create_ortools_example() -> GPDP:
    """Build instances from ortools reference guide."""
    data: Dict[str, Any] = {}
    coordinates = [
        (456, 320),  # location 0 - the depot
        (228, 0),  # location 1
        (912, 0),  # location 2
        (0, 80),  # location 3
        (114, 80),  # location 4
        (570, 160),  # location 5
        (798, 160),  # location 6
        (342, 240),  # location 7
        (684, 240),  # location 8
        (570, 400),  # location 9
        (912, 400),  # location 10
        (114, 480),  # location 11
        (228, 480),  # location 12
        (342, 560),  # location 13
        (684, 560),  # location 14
        (0, 640),  # location 15
        (798, 640),
    ]
    coordinates = [
        (0, 0),
        (-2, 4),
        (4, 4),
        (-4, 3),
        (-3, 3),
        (1, 2),
        (3, 3),
        (-1, 1),
        (2, 1),
        (1, -1),
        (4, -1),
        (-3, -2),
        (-2, -2),
        (-1, -3),
        (2, -3),
        (-4, -4),
        (3, -4),
    ]
    data["coordinates"] = coordinates
    data["distance_matrix"] = [
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
    ]
    data["num_vehicles"] = 4
    data["depot"] = 0
    data["pickups_deliveries"] = [
        [1, 6],
        [2, 10],
        [4, 3],
        [5, 9],
        [7, 8],
        [15, 11],
        [13, 12],
        [16, 14],
    ]
    data["time_matrix"] = [
        [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7],
        [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14],
        [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9],
        [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16],
        [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14],
        [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8],
        [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5],
        [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10],
        [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6],
        [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5],
        [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4],
        [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10],
        [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8],
        [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6],
        [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2],
        [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9],
        [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0],
    ]
    data["time_windows"] = [
        (0, 5),  # depot
        (7, 12),  # 1
        (10, 15),  # 2
        (16, 18),  # 3
        (10, 13),  # 4
        (0, 5),  # 5
        (5, 10),  # 6
        (0, 4),  # 7
        (5, 10),  # 8
        (0, 3),  # 9
        (10, 16),  # 10
        (10, 15),  # 11
        (0, 5),  # 12
        (5, 10),  # 13
        (7, 8),  # 14
        (10, 15),  # 15
        (11, 15),  # 16
    ]
    data["demands"] = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    data["vehicle_capacities"] = [15, 15, 15, 15]
    coordinates_reindex = coordinates[1:] + [
        coordinates[data["depot"]] for i in range(2 * data["num_vehicles"])
    ]
    original_index = [i for i in range(1, len(coordinates))] + [
        data["depot"] for i in range(2 * data["num_vehicles"])
    ]
    original_index_to_new = {}
    for j in range(len(original_index)):
        original_index_to_new[original_index[j]] = j
    number_nodes_non_depot = len(coordinates) - 1
    nodes_origin: List[Node] = list(
        range(number_nodes_non_depot, number_nodes_non_depot + data["num_vehicles"])
    )
    nodes_target: List[Node] = list(
        range(
            number_nodes_non_depot + data["num_vehicles"],
            number_nodes_non_depot + 2 * data["num_vehicles"],
        )
    )
    nodes_origin_dict: Dict[int, Node] = {
        i: nodes_origin[i] for i in range(data["num_vehicles"])
    }
    nodes_target_dict: Dict[int, Node] = {
        i: nodes_target[i] for i in range(data["num_vehicles"])
    }
    distance_delta_dict: Dict[Node, Dict[Node, float]] = {}
    time_delta_dict: Dict[Node, Dict[Node, float]] = {}
    for index in range(len(original_index)):
        distance_delta_dict[index] = {}
        time_delta_dict[index] = {}
        for index_2 in range(len(original_index)):
            distance_delta_dict[index][index_2] = data["distance_matrix"][
                original_index[index]
            ][original_index[index_2]]
            time_delta_dict[index][index_2] = data["time_matrix"][
                original_index[index]
            ][original_index[index_2]]
    pickup_and_deliveries: List[Tuple[List[Node], List[Node]]] = []
    for p_d in data["pickups_deliveries"]:
        pickup_and_deliveries += [
            ([original_index_to_new[p_d[0]]], [original_index_to_new[p_d[1]]])
        ]
    capacities: Dict[int, Dict[str, Tuple[float, float]]] = {}
    resources_set: Set[str] = {"demand"}
    for i in range(data["num_vehicles"]):
        capacities[i] = {"demand": (0, data["vehicle_capacities"][i])}
    resources_flow_edges: Dict[Edge, Dict[str, float]] = {
        (x, y): {"demand": 0}
        for x in distance_delta_dict
        for y in distance_delta_dict[x]
    }
    resources_flow_nodes: Dict[Node, Dict[str, float]] = {}
    for index in range(len(original_index)):
        resources_flow_nodes[index] = {
            "demand": -data["demands"][original_index[index]]
        }
    for k in nodes_origin_dict:
        resources_flow_nodes[nodes_origin_dict[k]] = {
            "demand": capacities[k]["demand"][1]
        }
    for k in nodes_target_dict:
        resources_flow_nodes[nodes_target_dict[k]] = {"demand": 0}
    coordinates_dict: Dict[Node, Tuple[float, float]] = {}
    for i in range(len(coordinates_reindex)):
        coordinates_dict[i] = coordinates_reindex[i]
    time_windows_nodes: Dict[Node, Tuple[Optional[int], Optional[int]]] = {}
    for i in range(len(original_index)):
        if i in nodes_target:
            continue
        time_windows_nodes[i] = data["time_windows"][original_index[i]]
    return GPDP(
        number_vehicle=data["num_vehicles"],
        nodes_transportation=set(range(number_nodes_non_depot)),
        nodes_origin=set(nodes_origin),
        nodes_target=set(nodes_target),
        origin_vehicle=nodes_origin_dict,
        target_vehicle=nodes_target_dict,
        list_pickup_deliverable=pickup_and_deliveries,
        resources_set=resources_set,
        capacities=capacities,
        resources_flow_node=resources_flow_nodes,
        resources_flow_edges=resources_flow_edges,
        distance_delta=distance_delta_dict,
        time_delta=time_delta_dict,
        coordinates_2d=coordinates_dict,
        time_windows_nodes=time_windows_nodes,
    )
