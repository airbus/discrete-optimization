from typing import Any, Callable, Optional

from discrete_optimization.generic_tools.graph_api import get_node_attributes

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, LinExpr, Model

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from discrete_optimization.maximum_independent_set.mis_model import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class MisMilpSolver(MisSolver, GurobiMilpSolver):
    def __init__(
        self,
        problem: MisProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.vars_node = None

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

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> MisSolution:

        chosen = [0] * self.problem.number_nodes

        for i in range(0, self.problem.number_nodes):
            chosen[i] = int(get_var_value_for_current_solution(self.vars_node[i]))

        return MisSolution(self.problem, chosen)
