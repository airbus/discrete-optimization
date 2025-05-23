import string

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.plot import (
    plot_coloring_problem,
    plot_coloring_solution,
)
from discrete_optimization.coloring.problem import ColoringProblem
from discrete_optimization.generic_tools.graph_api import Graph


def test_plot():
    # small ex
    instance = "gc_20_1"
    filepath = [f for f in get_data_available() if instance in f][0]
    color_problem = parse_file(filepath)
    # Transform node labels for testing with generic hashable nodes
    newnodes = [
        (string.ascii_lowercase[i], info) for i, info in color_problem.graph.nodes
    ]
    newedges = [
        (string.ascii_lowercase[i], string.ascii_lowercase[j], info)
        for i, j, info in color_problem.graph.edges
    ]
    graph = Graph(newnodes, newedges)
    color_problem = ColoringProblem(graph)
    # plots
    plot_coloring_problem(color_problem)
    plot_coloring_problem(
        color_problem,
        highlighted_nodes=[newnodes[0], newnodes[-1]],
        highlighted_edges=[(newnodes[0], newnodes[-1])],
    )
    color_solution = color_problem.get_dummy_solution()
    plot_coloring_solution(color_solution)
    plot_coloring_solution(
        color_solution,
        highlighted_nodes=[newnodes[0], newnodes[-1]],
        highlighted_edges=[(newnodes[0], newnodes[-1])],
    )
