#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os

import numpy as np

from discrete_optimization.datasets import get_data_home
from discrete_optimization.vrptw.problem import VRPTWProblem


def get_data_available(
    data_folder: str | None = None, data_home: str | None = None
) -> list[str]:
    """Get datasets available for vrp.

    Params:
        data_folder: folder where datasets for vrp whould be find.
            If None, we look in "vrp" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/homberger_200_customer_instances"

    try:
        datasets = [
            os.path.abspath(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
        ]
    except FileNotFoundError:
        datasets = []
    return datasets


def parse_vrptw_file(file_path: str) -> VRPTWProblem:
    """
    Parses a Solomon-style VRPTW instance file.

    Args:
        file_path (str): The path to the instance file (e.g., "RC1_2_10.TXT").

    Returns:
        VRPTWProblem: An instance of the VRPTW problem.
    """

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Read vehicle info
    vehicle_line = lines[4]
    nb_vehicles, vehicle_capacity = map(int, vehicle_line.split())

    # Read customer info
    customer_data = []
    # Lines are structured, starting from line 9
    # CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
    for line in lines[9:]:
        if line.strip():
            parts = line.split()
            if len(parts) == 7:
                customer_data.append(
                    (
                        int(parts[0]),  # Customer No.
                        float(parts[1]),  # X
                        float(parts[2]),  # Y
                        float(parts[3]),  # Demand
                        int(parts[4]),  # Ready Time
                        int(parts[5]),  # Due Date
                        float(parts[6]),  # Service Time
                    )
                )

    nb_nodes = len(customer_data)

    # Sort by customer number to ensure depot is at index 0
    customer_data.sort(key=lambda x: x[0])

    coords = np.array([[c[1], c[2]] for c in customer_data])
    demands = [c[3] for c in customer_data]
    time_windows = [(c[4], c[5]) for c in customer_data]
    service_times = [c[6] for c in customer_data]

    # Calculate Euclidean distance matrix
    distance_matrix = np.zeros((nb_nodes, nb_nodes))
    for i in range(nb_nodes):
        for j in range(i + 1, nb_nodes):
            dist = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return VRPTWProblem(
        nb_vehicles=nb_vehicles,
        vehicle_capacity=vehicle_capacity,
        nb_nodes=nb_nodes,
        distance_matrix=distance_matrix,
        time_windows=time_windows,
        service_times=service_times,
        demands=demands,
        depot_node=0,  # By convention from sorting
    )
