import os
from typing import Optional

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

    return [
        os.path.abspath(os.path.join(data_folder, f)) for f in os.listdir(data_folder)
    ]


def parse(input_data) -> ColoringProblem:
    # parse the input
    lines = input_data.split("\n")
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    edges = []
    nodes = [(i, {}) for i in range(node_count)]
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1]), {}))
    return ColoringProblem(
        Graph(nodes, edges, undirected=True, compute_predecessors=False)
    )


def parse_file(file_path) -> ColoringProblem:
    with open(file_path, "r") as input_data_file:
        input_data = input_data_file.read()
        coloring_model = parse(input_data)
        return coloring_model
