from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.graph_api import get_node_attributes
from discrete_optimization.maximum_independent_set.solvers.mis_lp import BaseLPMisSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, LinExpr, Model, Var

from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    OrtoolsMathOptMilpSolver,
)
from discrete_optimization.maximum_independent_set.mis_model import MisSolution


class MisMathOptMilpSolver(OrtoolsMathOptMilpSolver, BaseLPMisSolver):
    def init_model(self, **kwargs: Any) -> None:

        # Create a new model
        self.model = mathopt.Model()

        # Create variables
        self.vars_node = {
            i: self.model.add_binary_variable(name=f"N{i}")
            for i in range(self.problem.number_nodes)
        }

        # Set objective
        value = get_node_attributes(self.problem.graph_nx, "value", default=1)
        obj_exp = 0.0
        obj_exp += mathopt.LinearSum(
            value[self.problem.index_to_nodes[i]] * self.vars_node[i]
            for i in range(0, self.problem.number_nodes)
        )
        self.model.maximize(obj_exp)

        # for each edge it's impossible to choose the two nodes of this edges in our solution

        for edge in self.problem.graph_nx.edges():
            self.model.add_linear_constraint(
                self.vars_node[self.problem.nodes_to_index[edge[0]]]
                <= 1 - self.vars_node[self.problem.nodes_to_index[edge[1]]]
            )

    def convert_to_variable_values(
        self, solution: MisSolution
    ) -> dict[mathopt.Variable, float]:
        return BaseLPMisSolver.convert_to_variable_values(self, solution)


class MisMathOptQuadraticSolver(OrtoolsMathOptMilpSolver, BaseLPMisSolver):
    """
    The quadratic gurobi solver work only for graph without weight on nodes,
    if there is weight, it's going to ignore them
    """

    @property
    def vars_node(self) -> dict[int, Var]:
        return dict(enumerate(self.vars_node_matrix.tolist()))

    def init_model(self, **kwargs: Any) -> None:

        # Create a new model
        self.model = mathopt.Model()

        # Create variables
        self.vars_node_matrix = np.array(
            [
                self.model.add_binary_variable(name=f"N{i}")
                for i in range(self.problem.number_nodes)
            ]
        )

        # Set objective
        adj = nx.to_numpy_array(self.problem.graph_nx)
        J = np.identity(self.problem.number_nodes)
        A = J - adj
        self.model.maximize(self.vars_node_matrix @ A @ self.vars_node_matrix)

    def convert_to_variable_values(
        self, solution: MisSolution
    ) -> dict[mathopt.Variable, float]:
        return BaseLPMisSolver.convert_to_variable_values(self, solution)
