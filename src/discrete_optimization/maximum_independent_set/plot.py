import networkx as nx
from matplotlib import pyplot as plt

from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)

#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


def plot_mis_solution(solution: MisSolution, name_figure: str = ""):
    problem: MisProblem = solution.problem
    graph_nx = problem.graph_nx
    color_map = []
    labels = {}
    ind = 0
    for node in solution.chosen:
        if node == 1:
            color_map.append("red")
            labels[problem.index_to_nodes[ind]] = str(problem.index_to_nodes[ind])
        else:
            color_map.append("blue")
            labels[problem.index_to_nodes[ind]] = str(problem.index_to_nodes[ind])
        ind += 1
    pos = nx.kamada_kawai_layout(graph_nx)
    fig, ax = plt.subplots(1)
    nx.draw_networkx_nodes(graph_nx, pos=pos, ax=ax, node_color=color_map)
    nx.draw_networkx_labels(graph_nx, pos=pos, ax=ax, labels=labels)
    nx.draw_networkx_edges(graph_nx, pos=pos, ax=ax)
    ax.set_title(name_figure)
    plt.show()


def plot_mis_graph(problem: MisProblem, name_figure: str = ""):
    graph_nx = problem.graph_nx
    pos = nx.kamada_kawai_layout(graph_nx)
    labels = {n: str(n) for n in problem.graph_nx.nodes()}
    fig, ax = plt.subplots(1)
    nx.draw_networkx_nodes(graph_nx, pos=pos, nodelist=problem.graph_nx.nodes(), ax=ax)
    nx.draw_networkx_labels(graph_nx, pos=pos, ax=ax, labels=labels)
    nx.draw_networkx_edges(graph_nx, pos=pos, ax=ax)
    ax.set_title(name_figure)
    plt.show()
