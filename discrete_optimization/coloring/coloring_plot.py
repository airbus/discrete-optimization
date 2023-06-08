#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import networkx as nx

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)


def plot_coloring_solution(solution: ColoringSolution, name_figure: str = ""):
    problem: ColoringProblem = solution.problem
    graph_nx = problem.graph.graph_nx
    pos = nx.kamada_kawai_layout(graph_nx)
    fig, ax = plt.subplots(1)
    nx.draw_networkx_nodes(
        graph_nx,
        pos=pos,
        nodelist=problem.graph.nodes_name,
        label=[str(solution.colors[i]) for i in range(len(problem.graph.nodes_name))],
        node_color=[solution.colors[i] for i in range(len(problem.graph.nodes_name))],
        ax=ax,
    )
    nx.draw_networkx_edges(graph_nx, pos=pos, ax=ax)
    ax.set_title(name_figure)
