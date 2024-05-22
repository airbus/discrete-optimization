import networkx as nx
from matplotlib import pyplot as plt

from discrete_optimization.maximum_independent_set.mis_model import (
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
    for node in solution.chosen:
        if node == 1:
            color_map.append("red")
        else:
            color_map.append("blue")
    pos = nx.kamada_kawai_layout(graph_nx)
    fig, ax = plt.subplots(1)
    nx.draw_networkx_nodes(graph_nx, pos=pos, ax=ax, node_color=color_map)
    nx.draw_networkx_edges(graph_nx, pos=pos, ax=ax)
    ax.set_title(name_figure)
    plt.show()


def plot_mis_graph(problem: MisProblem, name_figure: str = ""):
    graph_nx = problem.graph_nx
    pos = nx.kamada_kawai_layout(graph_nx)
    fig, ax = plt.subplots(1)
    nx.draw_networkx_nodes(
        graph_nx,
        pos=pos,
        nodelist=problem.graph_nx.nodes(),
        ax=ax,
    )
    nx.draw_networkx_edges(graph_nx, pos=pos, ax=ax)
    ax.set_title(name_figure)
    plt.show()
