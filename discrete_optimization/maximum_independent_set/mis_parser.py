import os
from typing import Any, Dict, Hashable, List, Optional, Tuple

import networkx as nx

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.maximum_independent_set.mis_model import MisProblem

os.environ["DO_SKIP_MZN_CHECK"] = "1"


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> List[str]:
    """Get datasets available for mis.

    Params:
        data_folder: folder where datasets for coloring whould be find.
            If None, we look in "mis" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/mis"

    try:
        datasets = [
            os.path.abspath(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
        ]
    except FileNotFoundError:
        datasets = []
    return datasets


def dimacs_parser(filename: str):
    """From a file in dimacs format, initialise a MisProblem instance.

    Args:
        filename: file in the dimacs format

    Returns: a MisProblem instance
    """
    # parse the input
    input_data = open(filename, "r")
    lines = input_data.readlines()
    first_line = lines[0].split()
    node_count = int(first_line[2])
    edge_count = int(first_line[3])
    edges: List[Tuple[Hashable, Hashable, Dict[str, Any]]] = []
    nodes: List[Tuple[Hashable, Dict[str, Any]]] = [(i, {}) for i in range(node_count)]
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[1]), int(parts[2]), {}))
    return MisProblem(Graph(nodes, edges, undirected=True, compute_predecessors=False))


def basic_parser(filename: str):
    """From a file in basic format input, initialise a MisProblem instance.

    Args:
        filename: file in the basic format (same format as for coloring problem)

    Returns: a MisProblem instance
    """
    # parse the input
    input_data = open(filename, "r")
    lines = input_data.readlines()
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    edges: List[Tuple[Hashable, Hashable, Dict[str, Any]]] = []
    nodes: List[Tuple[Hashable, Dict[str, Any]]] = [(i, {}) for i in range(node_count)]
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1]), {}))
    return MisProblem(Graph(nodes, edges, undirected=True, compute_predecessors=False))


def dimacs_parser_nx(filename: str):
    """From a file in dimacs format, initialise a MisProblem instance.

    Args:
        filename: file in the dimacs format

    Returns: a MisProblem instance
    """
    # parse the input
    input_data = open(filename, "r")
    lines = input_data.readlines()
    first_line = lines[0].split()
    edge_count = int(first_line[3])
    graph = nx.Graph()
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        graph.add_edge(int(parts[1]), int(parts[2]))
    return MisProblem(graph)


def basic_parser_nx(filename: str):
    """From a file in basic format input, initialise a MisProblem instance.

    Args:
        filename: file in the basic format (same format as for coloring problem)

    Returns: a MisProblem instance
    """
    # parse the input
    input_data = open(filename, "r")
    lines = input_data.readlines()
    first_line = lines[0].split()
    edge_count = int(first_line[1])
    graph = nx.Graph()
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        graph.add_edge(int(parts[0]), int(parts[1]))
    return MisProblem(graph)
