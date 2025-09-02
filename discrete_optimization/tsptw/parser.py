#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
from typing import Optional

import numpy as np

from discrete_optimization.datasets import get_data_home
from discrete_optimization.tsptw.problem import TSPTWProblem


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for tsptw.

    Params:
        data_folder: folder where datasets for tsptw whould be find.
            If None, we look in "tsptw" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/tsptw/SolomonPotvinBengio"

    try:
        datasets = [
            os.path.abspath(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
        ]
    except FileNotFoundError:
        datasets = []
    return datasets


def parse_tsptw_file(filepath: str) -> TSPTWProblem:
    """
    Parses a TSP-TW problem file in the specified format.

    Args:
        filepath: The path to the TSP-TW instance file.

    Returns:
        An instance of the TSPTWProblem class.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Clean the content by removing comments, source tags, and extra whitespace
    content = content
    numbers = content.split()

    if not numbers:
        raise ValueError("File is empty or contains no parsable data.")

    # Parse number of nodes
    num_nodes = int(numbers.pop(0))

    # Parse distance matrix
    if len(numbers) < num_nodes * num_nodes:
        raise ValueError("File does not contain enough data for the distance matrix.")

    distance_matrix_list = []
    for _ in range(num_nodes):
        row = [float(numbers.pop(0)) for _ in range(num_nodes)]
        distance_matrix_list.append(row)
    distance_matrix = np.array(distance_matrix_list)

    # Parse time windows
    if len(numbers) < num_nodes * 2:
        raise ValueError("File does not contain enough data for the time windows.")

    time_windows = []
    for _ in range(num_nodes):
        # The file format might have node indices, but we assume order corresponds to node ID
        # It's safer to just parse the time windows themselves.
        # We find the next two integers in the list.
        earliest = int(numbers.pop(0))
        latest = int(numbers.pop(0))
        time_windows.append((earliest, latest))

    return TSPTWProblem(
        nb_nodes=num_nodes,
        distance_matrix=distance_matrix,
        time_windows=time_windows,
    )
