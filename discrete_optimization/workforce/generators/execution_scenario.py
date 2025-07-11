from allocation.allocation_problem import TeamAllocationProblem, TeamAllocationSolution
from allocation_and_scheduling.alloc_scheduling_problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
)
from allocation_and_scheduling.solvers.cpsat_alloc_scheduling_solver import (
    AdditionalCPConstraints,
)


class Scenario:
    def __init__(
        self,
        original_allocation_problem: TeamAllocationProblem,
        original_allocation_solution: TeamAllocationSolution,
        new_allocation_problem: TeamAllocationProblem,
        scheduling_problem: AllocSchedulingProblem,
        current_scheduling_solution: AllocSchedulingSolution = None,
        additional_constraint_scheduling: AdditionalCPConstraints = None,
    ):
        self.original_allocation_problem: TeamAllocationProblem = (
            original_allocation_problem
        )
        self.original_allocation_solution: TeamAllocationSolution = (
            original_allocation_solution
        )
        self.new_allocation_problem: TeamAllocationProblem = new_allocation_problem
        self.scheduling_problem: AllocSchedulingProblem = scheduling_problem
        self.current_scheduling_solution = current_scheduling_solution
        self.additional_constraint_scheduling = additional_constraint_scheduling
        self.description_scenario = ""

    def get_current_allocation_solution(
        self, put_none_to_invalid_allocation: bool = True
    ):
        sol = TeamAllocationSolution(
            problem=self.new_allocation_problem,
            allocation=[None] * self.new_allocation_problem.number_of_activity,
        )
        allowed_team = self.new_allocation_problem.compute_allowed_team_index_all_task()
        for i in range(len(self.original_allocation_solution.allocation)):
            ind_new = self.new_allocation_problem.index_activities_name[
                self.original_allocation_problem.activities_name[i]
            ]
            sol.allocation[ind_new] = self.new_allocation_problem.index_teams_name[
                self.original_allocation_problem.teams_name[
                    self.original_allocation_solution.allocation[i]
                ]
            ]
            if put_none_to_invalid_allocation:
                if sol.allocation[ind_new] not in allowed_team[ind_new]:
                    sol.allocation[ind_new] = None
        return sol

    def index_new_alloc_to_original(self):
        index = {}
        for i in self.new_allocation_problem.index_to_teams_name:
            team = self.new_allocation_problem.index_to_teams_name[i]
            old_index = self.original_allocation_problem.index_teams_name[team]
            index[i] = old_index
        return index
