#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, Hashable, List, Optional, Tuple

from discrete_optimization.coloring.coloring_model import ColoringProblem
from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.graph_api import Graph


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
):
    """Get datasets available for coloring.

    Params:
        data_folder: folder where datasets for coloring whould be find.
            If None, we look in "coloring" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/coloring"

    try:
        datasets = [
            os.path.abspath(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
        ]
    except FileNotFoundError:
        datasets = []
    return datasets


def parse(input_data) -> ColoringProblem:
    """From a text input, initialise a coloring problem instance.

    Args:
        input_data: text input in the format of {data_home}/coloring files

    Returns: a ColoringProblem instance
    """
    # parse the input
    lines = input_data.split("\n")
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    edges: List[Tuple[Hashable, Hashable, Dict[str, Any]]] = []
    nodes: List[Tuple[Hashable, Dict[str, Any]]] = [(i, {}) for i in range(node_count)]
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1]), {}))
    return ColoringProblem(
        Graph(nodes, edges, undirected=True, compute_predecessors=False)
    )


def parse_file(file_path) -> ColoringProblem:
    """From an absolute path to a coloring text file, return the corresponding coloring instance
    Args:
        file_path (str): absolute path to the file

    Returns: a ColoringProblem instance

    """
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        coloring_model = parse(input_data)
        return coloring_model
