import random
from copy import deepcopy
from typing import Hashable, Optional, Union

import numpy as np

from discrete_optimization.workforce.allocation.problem import (
    GraphBipartite,
    TeamAllocationProblem,
    TeamAllocationSolution,
)
from discrete_optimization.workforce.allocation.utils import cut_number_of_team
from discrete_optimization.workforce.generators.random_tools import RandomState
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    AdditionalCPConstraints,
)


class ParamsRandomness:
    def __init__(
        self,
        seed: int = 42,
        lower_nb_disruption: int = 1,
        upper_nb_disruption: int = 10,
        lower_nb_teams: int = 1,
        upper_nb_teams: int = 4,
        lower_time: int = 0,
        upper_time: int = 480,
        duration_discrete_distribution: tuple[list, list] = None,
    ):
        self.duration_discrete_distribution = duration_discrete_distribution
        if duration_discrete_distribution is None:
            self.duration_discrete_distribution = (
                [15, 30, 60, 120],
                [0.25, 0.25, 0.25, 0.25],
            )

        self.lower_nb_disruption = lower_nb_disruption
        self.upper_nb_disruption = upper_nb_disruption
        self.lower_nb_teams = lower_nb_teams
        self.upper_nb_teams = upper_nb_teams
        self.lower_time = lower_time
        self.upper_time = upper_time

        self.seed = seed
        self.random_state = RandomState(self.seed)

    def generate_list(
        self, solution: Union[TeamAllocationSolution, AllocSchedulingSolution]
    ):
        self.nb_disruption = self.random_state.get_discrete_truncated_uniform_sample(
            lower=self.lower_nb_disruption, upper=self.upper_nb_disruption
        )
        self.nb_teams_per_disruption = [
            self.random_state.get_discrete_truncated_uniform_sample(
                lower=self.lower_nb_teams, upper=self.upper_nb_teams
            )
            for _ in range(self.nb_disruption)
        ]
        self.starts = [
            self.random_state.get_discrete_truncated_uniform_sample(
                lower=self.lower_time, upper=self.upper_time
            )
            for _ in range(self.nb_disruption)
        ]
        self.lengths = [
            self.random_state.get_random_element_prop(
                list=self.duration_discrete_distribution[0],
                probs=self.duration_discrete_distribution[1],
            )
            for _ in range(self.nb_disruption)
        ]
        self.nb_disruption = self.random_state.get_discrete_truncated_uniform_sample(
            lower=self.lower_nb_disruption, upper=self.upper_nb_disruption
        )
        self.nb_teams_per_disruption = [
            self.random_state.get_discrete_truncated_uniform_sample(
                lower=self.lower_nb_teams, upper=self.upper_nb_teams
            )
            for _ in range(self.nb_disruption)
        ]
        self.starts = [
            self.random_state.get_discrete_truncated_uniform_sample(
                lower=self.lower_time, upper=self.upper_time
            )
            for _ in range(self.nb_disruption)
        ]
        self.lengths = [
            self.random_state.get_random_element_prop(
                list=self.duration_discrete_distribution[0],
                probs=self.duration_discrete_distribution[1],
            )
            for _ in range(self.nb_disruption)
        ]

        problem: Union[TeamAllocationProblem, AllocSchedulingProblem] = solution.problem
        index_to_team = None
        if isinstance(problem, TeamAllocationProblem):
            index_to_team = problem.index_to_teams_name
        if isinstance(problem, AllocSchedulingProblem):
            index_to_team = problem.index_to_team

        used_teams = list(
            {i for i in set(solution.allocation) if int(i) in index_to_team}
        )
        disruptions = []
        for j in range(self.nb_disruption):
            sub_teams = random.sample(
                used_teams, min(self.nb_teams_per_disruption[j], len(used_teams))
            )
            for k in range(len(sub_teams)):
                disruptions.append(
                    (self.starts[j], self.starts[j] + self.lengths[j], sub_teams[k])
                )
        return disruptions


def compute_starts_end_activities(problem: AllocSchedulingProblem):
    orig_starts_arr = np.array([problem.original_start[n] for n in problem.tasks_list])
    orig_ends_arr = np.array([problem.original_end[n] for n in problem.tasks_list])
    return orig_starts_arr, orig_ends_arr


def compute_available_teams_per_activities_alloc_problem(
    problem: TeamAllocationProblem,
    starts: np.ndarray,
    ends: np.ndarray,
    calendars_team: dict[Hashable, np.ndarray],
):
    available_team_per_activity = {}
    for i in range(len(starts)):
        available_team_per_activity[problem.activities_name[i]] = set()
        st, end = starts[i], ends[i]
        for team in calendars_team:
            if st == end:
                if calendars_team[team][int(st)] == 1:
                    available_team_per_activity[problem.activities_name[i]].add(team)
            if np.min(calendars_team[team][int(st) : int(end)]) == 1:
                available_team_per_activity[problem.activities_name[i]].add(team)
    return available_team_per_activity


def update_allocation_constraint(
    starts: np.ndarray,
    ends: np.ndarray,
    problem: TeamAllocationProblem,
    calendars_team: dict[Hashable, np.ndarray],
):
    available_team_per_activity = compute_available_teams_per_activities_alloc_problem(
        problem=problem, starts=starts, ends=ends, calendars_team=calendars_team
    )
    graph = problem.graph_allocation
    edges = []
    teams = problem.teams_name
    for ac in available_team_per_activity:
        for team in teams:
            if team not in available_team_per_activity[ac]:
                edges.append((ac, team, {}))
    problem.graph_allocation = GraphBipartite(
        nodes=graph.nodes,
        edges=edges,
        nodes_activity=graph.nodes_activity,
        nodes_team=graph.nodes_team,
        undirected=graph.undirected,
        compute_predecessors=False,
    )


def create_scheduling_problem_several_resource_dropping(
    allocation_problem: TeamAllocationProblem,
    scheduling_problem: AllocSchedulingProblem,
    list_drop_resource: list[tuple[int, int, int]],
    base_solution: TeamAllocationSolution,
):
    evaluation_allocation = allocation_problem.evaluate(base_solution)
    nb_team_used = evaluation_allocation["nb_teams"]
    used_team = set(base_solution.allocation)
    sched = np.zeros((scheduling_problem.number_tasks, 2))
    allocation = np.zeros(scheduling_problem.number_tasks)
    for i in range(scheduling_problem.number_tasks):
        sched[i, 0] = scheduling_problem.original_start[
            scheduling_problem.index_to_task[i]
        ]
        sched[i, 1] = scheduling_problem.original_end[
            scheduling_problem.index_to_task[i]
        ]
        team_index = base_solution.allocation[
            allocation_problem.index_activities_name[
                scheduling_problem.index_to_task[i]
            ]
        ]
        team = allocation_problem.teams_name[team_index]
        allocation[i] = scheduling_problem.teams_to_index[team]
    sol_sched = AllocSchedulingSolution(
        problem=scheduling_problem, schedule=sched, allocation=allocation
    )
    for from_time, to_time, index_team_name in list_drop_resource:
        nc = []
        team_name = scheduling_problem.team_names[index_team_name]
        for st, end in scheduling_problem.calendar_team[team_name]:
            if st <= from_time and end >= to_time:
                nc.append((st, from_time))
                nc.append((to_time, end))
            elif st <= from_time and end <= from_time:
                nc.append((st, end))
            elif from_time <= st <= to_time:
                # nc.append((st, min(end, to_time)))
                if end > to_time:
                    nc.append((to_time, end))
            else:
                nc.append((st, end))
        scheduling_problem.calendar_team[team_name] = nc
    used_teams_dict = {x: None for x in scheduling_problem.index_to_team}
    for team in scheduling_problem.index_to_team:
        if team not in used_team:
            used_teams_dict[team] = False
    # for x in scheduling_problem.index_to_team:
    #     if x not in used_teams:
    #         used_teams[x] = False
    additional_cp_constraints = AdditionalCPConstraints(
        nb_teams_bounds=(None, nb_team_used),  # (nb_team_used - 1, nb_team_used),
        team_used_constraint=used_teams_dict,
        set_tasks_ignore_reallocation=set(),
        forced_allocation={},
    )
    return {
        "additional_constraints": additional_cp_constraints,
        "base_solution": sol_sched,
    }


def generate_resource_disruption_scenario_from(
    original_allocation_problem: TeamAllocationProblem,
    original_scheduling_problem: AllocSchedulingProblem,
    original_solution: TeamAllocationSolution,
    list_drop_resource: Optional[list[tuple[int, int, int]]] = None,
    params_randomness: Optional[ParamsRandomness] = None,
):
    """
    Basically same code than the generator above, but starting from a known problem and original solution
    """
    if list_drop_resource is None:
        some_resource = random.choice(original_solution.allocation)
        list_drop_resource = [(3 * 60, 9 * 60, some_resource)]
    if params_randomness is not None:
        list_drop_resource = params_randomness.generate_list(solution=original_solution)

    params = create_scheduling_problem_several_resource_dropping(
        allocation_problem=original_allocation_problem,
        list_drop_resource=list_drop_resource,
        scheduling_problem=original_scheduling_problem,
        base_solution=original_solution,
    )
    starts, ends = compute_starts_end_activities(problem=original_scheduling_problem)
    calendars_array = {}
    for team in original_scheduling_problem.team_names:
        calendars_array[team] = np.zeros((original_scheduling_problem.horizon + 2))
        for slot in original_scheduling_problem.calendar_team[team]:
            calendars_array[team][
                slot[0] : min(slot[1], calendars_array[team].shape[0])
            ] = 1
    new_alloc_problem = deepcopy(original_allocation_problem)
    update_allocation_constraint(
        starts, ends, problem=new_alloc_problem, calendars_team=calendars_array
    )
    available_teams_in_this_scenario = set(
        [
            original_allocation_problem.teams_name[x]
            for x in original_solution.allocation
        ]
    )
    new_alloc_problem = cut_number_of_team(
        new_alloc_problem, subset_teams_keep=available_teams_in_this_scenario
    )
    return {
        "original_allocation_problem": original_scheduling_problem,
        "original_allocation_solution": original_solution,
        "new_allocation_problem": new_alloc_problem,
        "scheduling_problem": original_scheduling_problem,
        "current_scheduling_solution": params["base_solution"],
        "additional_constraint_scheduling": params["additional_constraints"],
        "list_drop_resource": list_drop_resource,
    }


def generate_allocation_disruption(
    original_allocation_problem: TeamAllocationProblem,
    original_solution: TeamAllocationSolution,
    list_drop_resource: Optional[list[tuple[int, int, int]]] = None,
    params_randomness: Optional[ParamsRandomness] = None,
):
    if list_drop_resource is None:
        some_resource = random.choice(original_solution.allocation)
        list_drop_resource = [(3 * 60, 9 * 60, some_resource)]
    if params_randomness is not None:
        list_drop_resource = params_randomness.generate_list(solution=original_solution)

    new_calendar = deepcopy(original_allocation_problem.calendar_team)
    for from_time, to_time, index_team_name in list_drop_resource:
        nc = []
        team_name = original_allocation_problem.teams_name[index_team_name]
        for st, end in new_calendar[team_name]:
            if st <= from_time and end >= to_time:
                nc.append((st, from_time))
                nc.append((to_time, end))
            elif st <= from_time and end <= from_time:
                nc.append((st, end))
            elif from_time <= st <= to_time:
                if end > to_time:
                    nc.append((to_time, end))
            else:
                nc.append((st, end))
        new_calendar[team_name] = nc
    new_alloc_problem = TeamAllocationProblem(
        allocation_additional_constraint=original_allocation_problem.allocation_additional_constraint,
        schedule_activity=original_allocation_problem.schedule,
        calendar_team=new_calendar,
        activities_name=original_allocation_problem.activities_name,
        graph_allocation=None,
        graph_activity=None,
    )
    available_teams_in_this_scenario = set(
        [
            original_allocation_problem.teams_name[x]
            for x in original_solution.allocation
        ]
    )
    new_alloc_problem = cut_number_of_team(
        new_alloc_problem, subset_teams_keep=available_teams_in_this_scenario
    )
    new_solution = TeamAllocationSolution(
        problem=new_alloc_problem,
        allocation=[
            new_alloc_problem.index_teams_name[
                original_allocation_problem.teams_name[original_solution.allocation[i]]
            ]
            for i in range(new_alloc_problem.number_of_activity)
        ],
    )
    return {
        "original_allocation_problem": original_allocation_problem,
        "original_allocation_solution": original_solution,
        "new_allocation_problem": new_alloc_problem,
        "new_solution": new_solution,
        "list_drop_resource": list_drop_resource,
    }


def generate_scheduling_disruption(
    original_scheduling_problem: AllocSchedulingProblem,
    original_solution: AllocSchedulingSolution,
    list_drop_resource: Optional[list[tuple[int, int, int]]] = None,
    params_randomness: Optional[ParamsRandomness] = None,
):
    if list_drop_resource is None:
        some_resource = random.choice(original_solution.allocation)
        list_drop_resource = [(3 * 60, 9 * 60, some_resource)]
    if params_randomness is not None:
        list_drop_resource = params_randomness.generate_list(solution=original_solution)

    new_calendar = deepcopy(original_scheduling_problem.calendar_team)
    for from_time, to_time, index_team_name in list_drop_resource:
        nc = []
        team_name = original_scheduling_problem.team_names[index_team_name]
        for st, end in new_calendar[team_name]:
            if st <= from_time and end >= to_time:
                nc.append((st, from_time))
                nc.append((to_time, end))
            elif st <= from_time and end <= from_time:
                nc.append((st, end))
            elif from_time <= st <= to_time:
                if end > to_time:
                    nc.append((to_time, end))
            else:
                nc.append((st, end))
        new_calendar[team_name] = nc
    new_scheduling_problem = deepcopy(original_scheduling_problem)
    new_scheduling_problem.calendar_team = new_calendar
    nb_team_used = original_scheduling_problem.evaluate(original_solution)["nb_teams"]
    used_teams_dict = {}
    used_team_set = set(original_solution.allocation)
    for team in original_scheduling_problem.index_to_team:
        if team not in used_team_set:
            used_teams_dict[team] = False
    additional_cp_constraints = AdditionalCPConstraints(
        nb_teams_bounds=(None, nb_team_used),  # (nb_team_used - 1, nb_team_used),
        team_used_constraint=used_teams_dict,
        set_tasks_ignore_reallocation=set(),
        forced_allocation={},
    )
    return {
        "scheduling_problem": new_scheduling_problem,
        "additional_constraint_scheduling": additional_cp_constraints,
        "list_drop_resource": list_drop_resource,
    }
