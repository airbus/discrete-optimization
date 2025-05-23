#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections.abc import Hashable
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy.typing as npt
from matplotlib.axes import Axes

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution


def plot_coloring_problem(
    problem: ColoringProblem,
    name_figure: str = "",
    highlighted_nodes: Optional[list[Hashable]] = None,
    highlighted_edges: Optional[list[tuple[Hashable, Hashable]]] = None,
    highlighted_node_style: Optional[dict[str, Any]] = None,
    highlighted_edge_style: Optional[dict[str, Any]] = None,
    normal_node_style: Optional[dict[str, Any]] = None,
    normal_edge_style: Optional[dict[str, Any]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot the underlying graph of a coloring problem

    Some nodes and edges can be highlighted by using a different style.
    The other nodes and edges will be all sharing the same style (as not yet colored).

    Args:
        problem: coloring problem to plot
        name_figure: name of the matplotlib figure generated
        highlighted_nodes: nodes to highlight
        highlighted_edges: edges to highlight
        highlighted_node_style: kwargs passed to networkx.draw_networkx_nodes for nodes to highlight
        highlighted_edge_style: kwargs passed to networkx.draw_networkx_edges for edges to highlight
        normal_node_style: kwargs passed to networkx.draw_networkx_nodes for all nodes
        normal_edge_style: kwargs passed to networkx.draw_networkx_edges for all edges
        ax: if specified matplotlib Axes in which plotting the graph

    Returns:
        The matplotlib Axes containing the plot

    """
    # default args
    (
        highlighted_nodes,
        highlighted_edges,
        highlighted_node_style,
        highlighted_edge_style,
        normal_node_style,
        normal_edge_style,
    ) = _default_plot_args(
        highlighted_nodes=highlighted_nodes,
        highlighted_edges=highlighted_edges,
        highlighted_node_style=highlighted_node_style,
        highlighted_edge_style=highlighted_edge_style,
        normal_node_style=normal_node_style,
        normal_edge_style=normal_edge_style,
    )

    graph_nx = problem.graph.graph_nx

    # init the figure and nodes position
    ax, pos = _prepare_plot_graph(graph_nx=graph_nx, name_figure=name_figure, ax=ax)

    # plot "normal" edges and nodes
    normal_edges = [e for e in graph_nx.edges if e not in highlighted_edges]
    normal_nodes = [n for n in graph_nx.nodes if n not in highlighted_nodes]
    nx.draw_networkx_nodes(
        graph_nx, pos=pos, nodelist=normal_nodes, ax=ax, **normal_node_style
    )
    nx.draw_networkx_edges(
        graph_nx, pos=pos, ax=ax, edgelist=normal_edges, **normal_edge_style
    )

    # plot highlighted edges and nodes
    _plot_highlitghted_nodes_n_edges(
        ax=ax,
        pos=pos,
        graph_nx=graph_nx,
        highlighted_edges=highlighted_edges,
        highlighted_nodes=highlighted_nodes,
        highlighted_edge_style=highlighted_edge_style,
        highlighted_node_style=highlighted_node_style,
    )

    return ax


def plot_coloring_solution(
    solution: ColoringSolution,
    name_figure: str = "",
    highlighted_nodes: Optional[list[Hashable]] = None,
    highlighted_edges: Optional[list[tuple[Hashable, Hashable]]] = None,
    normal_edge_style: Optional[dict[str, Any]] = None,
    highlighted_node_style: Optional[dict[str, Any]] = None,
    highlighted_edge_style: Optional[dict[str, Any]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot the colored graph of a coloring solution

    Nodes are colored according to the coloring solution.
    Some nodes and edges can be highlighted by using a different style.
    The other nodes and edges will be all sharing the same style (as not yet colored).

    Args:
        solution: coloring solution to plot
        name_figure: name of the matplotlib figure generated
        highlighted_nodes: nodes to highlight
        highlighted_edges: edges to highlight
        highlighted_node_style: kwargs passed to networkx.draw_networkx_nodes for nodes to highlight
        highlighted_edge_style: kwargs passed to networkx.draw_networkx_edges for edges to highlight
        normal_edge_style: kwargs passed to networkx.draw_networkx_edges for all edges
        ax: if specified matplotlib Axes in which plotting the graph

    Returns:
        The matplotlib Axes containing the plot

    """
    problem: ColoringProblem = solution.problem
    graph_nx = problem.graph.graph_nx

    # default args
    (
        highlighted_nodes,
        highlighted_edges,
        highlighted_node_style,
        highlighted_edge_style,
        normal_node_style,
        normal_edge_style,
    ) = _default_plot_args(
        highlighted_nodes=highlighted_nodes,
        highlighted_edges=highlighted_edges,
        highlighted_node_style=highlighted_node_style,
        highlighted_edge_style=highlighted_edge_style,
        normal_edge_style=normal_edge_style,
        plotting_solution=True,
    )

    # init the figure and nodes position
    ax, pos = _prepare_plot_graph(graph_nx=graph_nx, name_figure=name_figure, ax=ax)

    # plot nodes with solution colors
    nx.draw_networkx_nodes(
        graph_nx,
        pos=pos,
        nodelist=problem.graph.nodes_name,
        label=[str(solution.colors[i]) for i in range(len(problem.graph.nodes_name))],
        node_color=[solution.colors[i] for i in range(len(problem.graph.nodes_name))],
        ax=ax,
        **normal_node_style
    )
    # plot "normal" edges
    normal_edges = [e for e in graph_nx.edges if e not in highlighted_edges]
    nx.draw_networkx_edges(
        graph_nx, pos=pos, ax=ax, edgelist=normal_edges, **normal_edge_style
    )

    # plot highlighted edges and nodes
    _plot_highlitghted_nodes_n_edges(
        ax=ax,
        pos=pos,
        graph_nx=graph_nx,
        highlighted_edges=highlighted_edges,
        highlighted_nodes=highlighted_nodes,
        highlighted_edge_style=highlighted_edge_style,
        highlighted_node_style=highlighted_node_style,
    )

    ax.set_title(name_figure)
    return ax


def _prepare_plot_graph(
    graph_nx: nx.Graph, name_figure: str = "", ax: Optional[Axes] = None
) -> tuple[Axes, dict[Hashable, npt.NDArray[float]]]:
    pos = nx.kamada_kawai_layout(graph_nx)
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_title(name_figure)
    return ax, pos


def _default_plot_args(
    highlighted_nodes: Optional[list[Hashable]] = None,
    highlighted_edges: Optional[list[tuple[Hashable, Hashable]]] = None,
    highlighted_node_style: Optional[dict[str, Any]] = None,
    highlighted_edge_style: Optional[dict[str, Any]] = None,
    normal_node_style: Optional[dict[str, Any]] = None,
    normal_edge_style: Optional[dict[str, Any]] = None,
    plotting_solution: bool = False,
) -> tuple[
    list[Hashable],
    list[tuple[Hashable, Hashable]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if highlighted_edges is None:
        highlighted_edges = []
    if highlighted_nodes is None:
        highlighted_nodes = []
    if highlighted_node_style is None:
        if plotting_solution:
            highlighted_node_style = dict(edgecolors="r", linewidths=2)
        else:
            highlighted_node_style = dict(node_color="r")
    if highlighted_edge_style is None:
        if plotting_solution:
            highlighted_edge_style = dict(edge_color="r", width=2, style="dashed")
        else:
            highlighted_edge_style = dict(edge_color="r", width=2)
    if normal_edge_style is None:
        normal_edge_style = dict(edge_color="grey")
    if normal_node_style is None:
        if plotting_solution:
            normal_node_style = dict()
        else:
            normal_node_style = dict(node_color="#1f78b4")
    return (
        highlighted_nodes,
        highlighted_edges,
        highlighted_node_style,
        highlighted_edge_style,
        normal_node_style,
        normal_edge_style,
    )


def _plot_highlitghted_nodes_n_edges(
    ax: Axes,
    pos: dict[Hashable, npt.NDArray[float]],
    graph_nx: nx.Graph,
    highlighted_nodes: list[Hashable],
    highlighted_edges: list[tuple[Hashable, Hashable]],
    highlighted_node_style: dict[str, Any],
    highlighted_edge_style: dict[str, Any],
):
    highlighted_edges = [e for e in graph_nx.edges if e in highlighted_edges]
    highlighted_nodes = [n for n in graph_nx.nodes if n in highlighted_nodes]
    nx.draw_networkx_nodes(
        graph_nx, pos=pos, nodelist=highlighted_nodes, ax=ax, **highlighted_node_style
    )
    nx.draw_networkx_edges(
        graph_nx, pos=pos, ax=ax, edgelist=highlighted_edges, **highlighted_edge_style
    )
