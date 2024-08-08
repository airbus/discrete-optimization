from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.graph_api import get_node_attributes
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, LinExpr, Model, Var

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from discrete_optimization.maximum_independent_set.mis_model import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class BaseGurobiMisSolver(MisSolver, GurobiMilpSolver, WarmstartMixin):
    vars_node: Sequence[Var]

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

    def set_warm_start(self, solution: MisSolution) -> None:
        """Make the solver warm start from the given solution."""
        for i in range(0, self.problem.number_nodes):
            self.vars_node[i].Start = solution.chosen[i]


class MisMilpSolver(BaseGurobiMisSolver):
    def init_model(self, **kwargs: Any) -> None:

        # Create a new model
        self.model = Model()

        # Create variables

        self.vars_node = self.model.addVars(
            self.problem.number_nodes, vtype=GRB.BINARY, name="N"
        )
        value = get_node_attributes(self.problem.graph_nx, "value", default=1)

        # Set objective
        obj_exp = LinExpr()
        obj_exp.addTerms(value.values(), self.vars_node.select())
        self.model.setObjective(obj_exp, GRB.MAXIMIZE)

        # for each edge it's impossible to choose the two nodes of this edges in our solution

        for edge in self.problem.graph_nx.edges():
            self.model.addConstr(
                self.vars_node[self.problem.nodes_to_index[edge[0]]]
                <= 1 - self.vars_node[self.problem.nodes_to_index[edge[1]]]
            )


class MisQuadraticSolver(BaseGurobiMisSolver):
    """
    The quadratic gurobi solver work only for graph without weight on nodes,
    if there is weight, it's going to ignore them
    """

    def init_model(self, **kwargs: Any) -> None:

        # Create a new model
        self.model = Model()

        # Create variables
        self.vars_node = self.model.addMVar(
            self.problem.number_nodes, vtype=GRB.BINARY, name="N"
        )

        # Set objective
        adj = nx.to_numpy_array(self.problem.graph_nx)
        J = np.identity(self.problem.number_nodes)
        A = J - adj
        self.model.setObjective(self.vars_node @ A @ self.vars_node, GRB.MAXIMIZE)
