from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPSolutionPreemptive
from typing import Union


def get_max_time_solution(solution: Union[RCPSPSolutionPreemptive, RCPSPSolution]):
    if isinstance(solution, RCPSPSolutionPreemptive):
        max_time = max([solution.rcpsp_schedule[x]["ends"][-1]
                        for x in solution.rcpsp_schedule])
        return max_time
    if isinstance(solution, RCPSPSolution):
        max_time = max([solution.rcpsp_schedule[x]["end_time"]
                        for x in solution.rcpsp_schedule])
        return max_time


def get_tasks_ending_between_two_times(solution: Union[RCPSPSolutionPreemptive, RCPSPSolution],
                                       time_1, time_2):
    if isinstance(solution, RCPSPSolutionPreemptive):
        return [x for x in solution.rcpsp_schedule
                if time_1 <= solution.rcpsp_schedule[x]["ends"][-1] <= time_2]
    if isinstance(solution, RCPSPSolution):
        return [x for x in solution.rcpsp_schedule
                if time_1 <= solution.rcpsp_schedule[x]["end_time"] <= time_2]


