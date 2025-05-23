import re
from collections.abc import Hashable, Iterable
from typing import Any, Optional

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

SOFT_METACONSTRAINT_PREFIX = "neighbours colors of node "
METADATA_NODE = "node"


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
                name=f"{SOFT_METACONSTRAINT_PREFIX}{self.problem.index_to_nodes_name[i]}",
                metadata={METADATA_NODE: self.problem.index_to_nodes_name[i]},
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

    def convert_constraint2edge(
        self, constraint: Expression
    ) -> Optional[tuple[Hashable, Hashable]]:
        if not constraint.has_subexpr() and len(constraint.args) == 2:
            try:
                start_idx = self._convert_variable2idx(constraint.args[0])
                end_idx = self._convert_variable2idx(constraint.args[1])
            except ValueError:
                pass
            else:
                return (
                    self.problem.graph.nodes_name[start_idx],
                    self.problem.graph.nodes_name[end_idx],
                )

    def convert_constraints2edges(
        self, constraints: Iterable[Expression]
    ) -> list[tuple[Hashable, Hashable]]:
        edges = []
        for c in constraints:
            edge = self.convert_constraint2edge(c)
            if edge is not None:
                edges.append(edge)
        return edges

    @staticmethod
    def _convert_variable2idx(variable: Expression) -> int:
        match = re.match(r"x\[([0-9]*)\]", variable.name)
        if match is None:
            raise ValueError(
                f"variable {variable} name has not the proper format 'x[i]'"
            )
        else:
            return int(match[1])

    def convert_metaconstraint2node(
        self, meta: MetaCpmpyConstraint
    ) -> Optional[Hashable]:
        if METADATA_NODE in meta.metadata:
            return meta.metadata[METADATA_NODE]

    def convert_metaconstraints2nodes(
        self, metas: list[MetaCpmpyConstraint]
    ) -> list[Hashable]:
        nodes = []
        for meta in metas:
            node = self.convert_metaconstraint2node(meta)
            if node is not None:
                nodes.append(node)
        return nodes

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
