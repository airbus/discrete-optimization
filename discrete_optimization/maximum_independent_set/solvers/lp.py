from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.graph_api import get_node_attributes
from discrete_optimization.generic_tools.lp_tools import MilpSolver
from discrete_optimization.maximum_independent_set.problem import MisSolution
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class BaseLpMisSolver(MisSolver, MilpSolver):
    vars_node: list[Any]

    def init_model(self, **kwargs: Any) -> None:

        # Create a new model
        self.model = self.create_empty_model()

        # Create variables
        self.vars_node = [
            self.add_binary_variable(name=f"N{i}")
            for i in range(self.problem.number_nodes)
        ]

        # Set objective
        value = get_node_attributes(self.problem.graph_nx, "value", default=1)
        obj_exp = 0.0
        obj_exp += self.construct_linear_sum(
            value[self.problem.index_to_nodes[i]] * self.vars_node[i]
            for i in range(0, self.problem.number_nodes)
        )
        self.set_model_objective(obj_exp, minimize=False)

        # for each edge it's impossible to choose the two nodes of this edges in our solution

        for edge in self.problem.graph_nx.edges():
            self.add_linear_constraint(
                self.vars_node[self.problem.nodes_to_index[edge[0]]]
                <= 1 - self.vars_node[self.problem.nodes_to_index[edge[1]]]
            )

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> MisSolution:

        chosen = [0] * self.problem.number_nodes

        for i in range(0, self.problem.number_nodes):
            if get_var_value_for_current_solution(self.vars_node[i]) > 0.5:
                chosen[i] = 1

        return MisSolution(self.problem, chosen)

    def convert_to_variable_values(self, solution: MisSolution) -> dict[Any, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return {
            self.vars_node[i]: solution.chosen[i]
            for i in range(0, self.problem.number_nodes)
        }


class BaseQuadMisSolver(BaseLpMisSolver):
    """Base class for quadratic solvers with gurobi or mathopt.

    Work only for graph without weight on nodes.
    If there are weights, it's going to ignore them.

    """

    vars_node_matrix: np.array

    @property
    def vars_node(self) -> list[Any]:
        return self.vars_node_matrix.tolist()

    def create_vars_node_matrix(self) -> np.array:
        return np.array(
            [
                self.add_binary_variable(name=f"N{i}")
                for i in range(self.problem.number_nodes)
            ]
        )

    def init_model(self, **kwargs: Any) -> None:
        # Create a new model
        self.model = self.create_empty_model()

        # Create variables
        self.vars_node_matrix = self.create_vars_node_matrix()

        # Set objective
        adj = nx.to_numpy_array(self.problem.graph_nx)
        J = np.identity(self.problem.number_nodes)
        A = J - adj
        self.set_model_objective(
            self.vars_node_matrix @ A @ self.vars_node_matrix, minimize=False
        )
