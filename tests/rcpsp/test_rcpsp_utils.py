#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import networkx as nx

from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    Graph,
    compute_graph_rcpsp,
    plot_task_gantt,
)


def test_compute_graph_run():
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    graph: Graph = compute_graph_rcpsp(rcpsp_problem)
    graph_nx = graph.to_networkx()
    path = nx.astar_path(
        G=graph_nx,
        source=1,
        target=rcpsp_problem.n_jobs,
        heuristic=lambda x, y: -100,
        weight="minus_min_duration",
    )
    edges = [(n1, n2) for n1, n2 in zip(path[:-1], path[1:])]
    duration = sum(graph_nx[n[0]][n[1]]["min_duration"] for n in edges)
    dfs = nx.dfs_tree(G=graph_nx, source=1, depth_limit=10)
    shortest_path_length = nx.shortest_path_length(dfs, 1)
    length_to_nodes = {}
    position = {}
    for node in sorted(shortest_path_length, key=lambda x: shortest_path_length[x]):
        length = shortest_path_length[node]
        while not (length not in length_to_nodes or len(length_to_nodes[length]) <= 5):
            length += 1
        if length not in length_to_nodes:
            length_to_nodes[length] = []
        length_to_nodes[length] += [node]
        position[node] = (length, len(length_to_nodes[length]))
    nx.draw_networkx(graph_nx, pos=position)


def test_plot_gantt():
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solution = rcpsp_problem.get_dummy_solution()
    plot_task_gantt(rcpsp_problem, solution)


if __name__ == "__main__":
    test_compute_graph_run()
