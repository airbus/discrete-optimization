from __future__ import annotations

from typing import Any

from discrete_optimization.maximum_independent_set.solvers.lp import (
    BaseLpMisSolver,
    BaseQuadMisSolver,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, LinExpr, Model, Var, quicksum

from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from discrete_optimization.maximum_independent_set.problem import MisSolution


class GurobiMisSolver(GurobiMilpSolver, BaseLpMisSolver):
    def init_model(self, **kwargs: Any) -> None:
        BaseLpMisSolver.init_model(self, **kwargs)
        self.model.update()

    def convert_to_variable_values(self, solution: MisSolution) -> dict[Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return BaseLpMisSolver.convert_to_variable_values(self, solution)


class GurobiQuadraticMisSolver(GurobiMilpSolver, BaseQuadMisSolver):
    """Quadratic solver with gurobi.

    Work only for graph without weight on nodes.
    If there are weights, it's going to ignore them.

    """

    vars_node_matrix: gurobipy.MVar

    def init_model(self, **kwargs: Any) -> None:
        BaseQuadMisSolver.init_model(self, **kwargs)
        self.model.update()

    def create_vars_node_matrix(self) -> gurobipy.MVar:
        return self.model.addMVar(self.problem.number_nodes, vtype=GRB.BINARY, name="N")

    def convert_to_variable_values(self, solution: MisSolution) -> dict[Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return BaseQuadMisSolver.convert_to_variable_values(self, solution)
