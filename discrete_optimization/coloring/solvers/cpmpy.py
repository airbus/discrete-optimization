from typing import Any

from cpmpy import Model, intvar

from discrete_optimization.coloring.problem import ColoringSolution
from discrete_optimization.coloring.solvers import ColoringSolver
from discrete_optimization.coloring.solvers.starting_solution import (
    WithStartingSolutionColoringSolver,
)
from discrete_optimization.generic_tools.cpmpy_tools import CpmpySolver


class CpmpyColoringSolver(
    CpmpySolver, WithStartingSolutionColoringSolver, ColoringSolver
):
    def init_model(self, **kwargs: Any) -> None:

        if "nb_colors" not in kwargs:
            solution = self.get_starting_solution(**kwargs)
            nb_colors = self.problem.count_colors_all_index(solution.colors)
        else:
            nb_colors = int(kwargs["nb_colors"])

        n = self.problem.number_of_nodes
        self.variables = {}
        x = intvar(0, nb_colors - 1, shape=n, name="x")
        nbc = intvar(0, nb_colors - 1, shape=1, name="nb_colors")
        constraints = []
        for node_i, node_j, _ in self.problem.graph.edges:
            i = self.problem.index_nodes_name[node_i]
            j = self.problem.index_nodes_name[node_j]
            constraints.append(x[i] != x[j])
        constraints.append(x <= nbc)

        self.variables["x"] = x
        self.variables["nb_colors"] = nbc
        self.model = Model(constraints, minimize=nbc)

    def retrieve_current_solution(self) -> ColoringSolution:
        colors = self.variables["x"].value().tolist()
        nb_colors = max(colors) + 1
        return ColoringSolution(problem=self.problem, colors=colors, nb_color=nb_colors)
