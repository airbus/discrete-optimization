from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.maximum_independent_set.mis_model import MisProblem


class MisSolver(SolverDO):
    problem: MisProblem
