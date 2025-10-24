from __future__ import annotations

from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.lp_tools import OrtoolsMathOptMilpSolver
from discrete_optimization.maximum_independent_set.problem import MisSolution
from discrete_optimization.maximum_independent_set.solvers.lp import (
    BaseLpMisSolver,
    BaseQuadMisSolver,
)


class MathOptMisSolver(OrtoolsMathOptMilpSolver, BaseLpMisSolver):
    def convert_to_variable_values(
        self, solution: MisSolution
    ) -> dict[mathopt.Variable, float]:
        return BaseLpMisSolver.convert_to_variable_values(self, solution)


class MathOptQuadraticMisSolver(OrtoolsMathOptMilpSolver, BaseQuadMisSolver):
    """Quadratic solver mathopt.

    Work only for graph without weight on nodes.
    If there are weights, it's going to ignore them.

    """

    has_quadratic_objective = True

    def convert_to_variable_values(
        self, solution: MisSolution
    ) -> dict[mathopt.Variable, float]:
        return BaseQuadMisSolver.convert_to_variable_values(self, solution)
