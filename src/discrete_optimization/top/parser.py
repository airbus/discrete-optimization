#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.top.problem import CustomerTop2D, TeamOrienteeringProblem2D


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for tsp.

    Params:
        data_folder: folder where datasets for weighted tardiness problem should be found.
            If None, we look in "wt" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/top"

    try:
        subfolders = [
            f
            for f in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, f))
        ]
        files_per_subfolder = {}
        files = []
        for subfolder in subfolders:
            sf = os.path.join(data_folder, subfolder)
            files_per_subfolder[subfolder] = [
                os.path.join(sf, f) for f in os.listdir(sf) if "txt" in f
            ]
            files.extend(files_per_subfolder[subfolder])
        return files, files_per_subfolder
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)


def parse_file(file_path: str) -> TeamOrienteeringProblem2D:
    with open(file_path, "r") as f:
        lines = f.readlines()
    # 1. Parse Metadata
    # n: total locations (depots + customers)
    # m: number of vehicles
    # tmax: max length per tour
    n = int(lines[0].split(";")[1])
    m = int(lines[1].split(";")[1])
    tmax = float(lines[2].split(";")[1])
    # 2. Parse Coordinates and Rewards
    customers = []
    for i, line in enumerate(lines[3:]):
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        x, y, reward = float(parts[0]), float(parts[1]), float(parts[2])
        customers.append(CustomerTop2D(name=i, reward=reward, x=x, y=y))
    start_index = 0
    end_index = len(customers) - 1
    problem = TeamOrienteeringProblem2D(
        customers=customers,
        customer_count=len(customers),
        vehicle_count=m,
        max_length_tours=tmax,
        start_indexes=[start_index] * m,
        end_indexes=[end_index] * m,
    )

    return problem
