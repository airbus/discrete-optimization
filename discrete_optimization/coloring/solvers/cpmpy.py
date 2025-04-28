from typing import Any

from cpmpy import Model, intvar
from cpmpy.expressions.core import Expression

from discrete_optimization.coloring.problem import ColoringSolution
from discrete_optimization.coloring.solvers import ColoringSolver
from discrete_optimization.coloring.solvers.starting_solution import (
    WithStartingSolutionColoringSolver,
)
from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpySolver,
    MetaCpmpyConstraint,
)


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

        soft_constraints = []
        soft_meta_constraints = [
            MetaCpmpyConstraint(
                name=f"neighbours colors of node {self.problem.index_to_nodes_name[i]}"
            )
            for i in range(n)
        ]

        for node_i, node_j, _ in self.problem.graph.edges:
            i = self.problem.index_nodes_name[node_i]
            j = self.problem.index_nodes_name[node_j]
            constraint: Expression = x[i] != x[j]
            soft_constraints.append(constraint)
            soft_meta_constraints[i].append(constraint)
            soft_meta_constraints[j].append(constraint)

        self._hard_constraint: Expression = x <= nbc
        constraints = soft_constraints + [self._hard_constraint]

        self.model = Model(constraints, minimize=nbc)

        self.variables["x"] = x
        self.variables["nb_colors"] = nbc
        self._soft_meta_constraints = soft_meta_constraints
        self._soft_constraints = soft_constraints
        self._hard_meta_constraint = MetaCpmpyConstraint(
            name="nb colors", constraints=[self._hard_constraint]
        )

    def retrieve_current_solution(self) -> ColoringSolution:
        colors = self.variables["x"].value().tolist()
        nb_colors = max(colors) + 1
        return ColoringSolution(problem=self.problem, colors=colors, nb_color=nb_colors)

    def get_soft_meta_constraints(self) -> list[MetaCpmpyConstraint]:
        return self._soft_meta_constraints

    def get_hard_meta_constraints(self) -> list[MetaCpmpyConstraint]:
        return [self._hard_meta_constraint]

    def get_soft_constraints(self) -> list[Expression]:
        return self._soft_constraints

    def get_hard_constraints(self) -> list[Expression]:
        return [self._hard_constraint]
