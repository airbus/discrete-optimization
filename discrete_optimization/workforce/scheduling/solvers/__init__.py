from enum import Enum

from scheduling.problem import AllocSchedulingProblem

from discrete_optimization.generic_tools.do_solver import SolverDO


class ObjectivesEnum(Enum):
    NB_TEAMS = 1
    DISPERSION = 2
    MAKESPAN = 3
    MIN_WORKLOAD = 4
    NB_DONE_AC = 5  # number of done activities
    DELTA_TO_EXISTING_SOLUTION = 6
    DISPERSION_DISTANCE = 7
    MIN_DISTANCE = 8
    MAX_DISTANCE = 9


class SolverAllocScheduling(SolverDO):
    problem: AllocSchedulingProblem
