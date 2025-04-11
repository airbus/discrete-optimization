import os
from collections.abc import Hashable
from enum import Enum
from typing import Any, Optional

import networkx as nx

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.maximum_independent_set.problem import MisProblem


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
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


class DimacsPrefixes(Enum):
    PROBLEM = "p"
    COMMENT = "c"
    EDGE = "e"


def dimacs_parser(filename: str):
    """From a file in dimacs format, initialise a MisProblem instance.

    Args:
        filename: path to the input file using dimacs format

    See http://prolland.free.fr/works/research/dsat/dimacs.html for reference about dimacs format

    Returns: a MisProblem instance
    """
    # parse the input
    input_data = open(filename, "r")
    lines = input_data.readlines()
    n_nodes = 0
    n_edges = 0
    graph = nx.Graph()
    problem_line_read = False
    for line in lines:
        tokens = line.split()
        if len(tokens) == 0:
            # ignore empty lines
            continue
        prefix = tokens[0]
        if prefix == DimacsPrefixes.COMMENT.value:
            # comment line: ignored
            continue
        elif prefix == DimacsPrefixes.PROBLEM.value:
            assert (
                not problem_line_read
            ), "The dimacs file can have only one problem line."
            problem_line_read = True
            assert tokens[1] == "edge", "The dimacs problem format must be 'edge'."
            n_nodes = int(tokens[2])
            n_edges = int(tokens[3])
            graph.add_nodes_from(range(1, 1 + n_nodes))
        elif prefix == DimacsPrefixes.EDGE.value:
            assert (
                problem_line_read
            ), "The problem line must appear before any edge descriptor."
            graph.add_edge(int(tokens[1]), int(tokens[2]))
        else:
            raise NotImplementedError(
                f"The prefix {prefix} is not allowed by dimacs format."
            )
    assert (
        len(graph.edges) == n_edges
    ), "The problem line defines a number of edges different from the number of edge lines."
    assert (
        len(graph.nodes) == n_nodes
    ), "The problem line defines a number of nodes different from the max node id used by an edge."
    return MisProblem(graph)


# alias for backward compatibility
dimacs_parser_nx = dimacs_parser
