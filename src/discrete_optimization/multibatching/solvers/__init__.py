from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
)


class MultibatchingSolver(SolverDO):
    problem: MultibatchingProblem
