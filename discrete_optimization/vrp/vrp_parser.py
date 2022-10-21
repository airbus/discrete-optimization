#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from typing import Optional

from discrete_optimization.datasets import get_data_home
from discrete_optimization.vrp.vrp_model import Customer2D, VrpProblem, VrpProblem2D


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
):
    """Get datasets available for vrp.

    Params:
        data_folder: folder where datasets for vrp whould be find.
            If None, we look in "vrp" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/vrp"

    try:
        datasets = [
            os.path.abspath(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
        ]
    except FileNotFoundError:
        datasets = []
    return datasets


def parse_input(input_data, start_index=0, end_index=0, vehicle_count=None):
    # parse the input
    lines = input_data.split("\n")

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1]) if vehicle_count is None else vehicle_count
    vehicle_capacity = int(parts[2])
    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(
            Customer2D(i - 1, int(parts[0]), float(parts[1]), float(parts[2]))
        )
    vehicle_capacities = [vehicle_capacity] * vehicle_count
    start_indexes = [start_index] * vehicle_count
    end_indexes = [end_index] * vehicle_count
    return VrpProblem2D(
        vehicle_count=vehicle_count,
        vehicle_capacities=vehicle_capacities,
        customer_count=customer_count,
        customers=customers,
        start_indexes=start_indexes,
        end_indexes=end_indexes,
    )


def parse_file(file_path, start_index=0, end_index=0, vehicle_count=None) -> VrpProblem:
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        vrp_model = parse_input(
            input_data,
            start_index=start_index,
            end_index=end_index,
            vehicle_count=vehicle_count,
        )
        return vrp_model
