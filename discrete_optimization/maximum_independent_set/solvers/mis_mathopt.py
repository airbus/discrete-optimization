from __future__ import annotations

from discrete_optimization.maximum_independent_set.solvers.mis_lp import (
    BaseLPMisSolver,
    BaseQuadMisSolver,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, LinExpr, Model, Var

from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.lp_tools import OrtoolsMathOptMilpSolver
from discrete_optimization.maximum_independent_set.mis_model import MisSolution


class MisMathOptMilpSolver(OrtoolsMathOptMilpSolver, BaseLPMisSolver):
    def convert_to_variable_values(
        self, solution: MisSolution
    ) -> dict[mathopt.Variable, float]:
        return BaseLPMisSolver.convert_to_variable_values(self, solution)


class MisMathOptQuadraticSolver(OrtoolsMathOptMilpSolver, BaseQuadMisSolver):
    """Quadratic solver mathopt.

    Work only for graph without weight on nodes.
    If there are weights, it's going to ignore them.

    """

    def convert_to_variable_values(
        self, solution: MisSolution
    ) -> dict[mathopt.Variable, float]:
        return BaseQuadMisSolver.convert_to_variable_values(self, solution)
