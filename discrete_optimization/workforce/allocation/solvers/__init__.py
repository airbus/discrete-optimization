import os

from allocation import TeamAllocationProblem

from discrete_optimization.generic_tools.do_solver import SolverDO

folder_solver = os.path.dirname(os.path.abspath(__file__))


class TeamAllocationSolver(SolverDO):
    problem: TeamAllocationProblem
