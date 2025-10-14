#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import itertools
import logging
from collections.abc import Hashable
from functools import reduce
from typing import Optional

import networkx as nx
import numpy as np

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
)
from discrete_optimization.generic_tasks_tools.precedence import PrecedenceProblem
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)
from discrete_optimization.rcpsp.problem import (
    PairModeConstraint,
    RcpspProblem,
    RcpspSolution,
    SpecialConstraintsDescription,
)

logger = logging.getLogger(__name__)


Task = Hashable
UnaryResource = Hashable


class AllocSchedulingSolution(
    SchedulingSolution[Task], AllocationSolution[Task, UnaryResource]
):
    problem: AllocSchedulingProblem

    def __init__(
        self,
        problem: AllocSchedulingProblem,
        schedule: np.ndarray,
        allocation: np.ndarray,
    ):
        self.problem = problem
        self.schedule = schedule
        self.allocation = allocation

    def copy(self) -> "Solution":
        return AllocSchedulingSolution(
            problem=self.problem,
            schedule=np.copy(self.schedule),
            allocation=np.copy(self.allocation),
        )

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem

    def get_end_time(self, task: Task) -> int:
        i_task = self.problem.tasks_to_index[task]
        return int(self.schedule[i_task, 1])

    def get_start_time(self, task: Task) -> int:
        i_task = self.problem.tasks_to_index[task]
        return int(self.schedule[i_task, 0])

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        i_task = self.problem.tasks_to_index[task]
        i_team = self.problem.teams_to_index[unary_resource]
        return int(self.allocation[i_task]) == i_team


class TasksDescription:
    def __init__(self, duration_task: int, resource_consumption: dict[str, int] = None):
        self.duration_task = duration_task
        self.resource_consumption = resource_consumption
        if self.resource_consumption is None:
            self.resource_consumption = {}


class AllocSchedulingProblem(
    SchedulingProblem[Task],
    AllocationProblem[Task, UnaryResource],
    PrecedenceProblem[Task],
):
    def __init__(
        self,
        team_names: list[UnaryResource],
        calendar_team: dict[
            UnaryResource, list[tuple[int, int]]
        ],  # List of available slots per team.
        horizon: int,
        tasks_list: list[Task],
        tasks_data: dict[Task, TasksDescription],
        same_allocation: list[set[Task]],
        precedence_constraints: dict[Task, set[Task]],
        available_team_for_activity: dict[Task, set[UnaryResource]],
        start_window: dict[Task, tuple[Optional[int], Optional[int]]],
        end_window: dict[Task, tuple[Optional[int], Optional[int]]],
        original_start: dict[Task, int],
        original_end: dict[Task, int],
        resources_list: list[str] = None,
        resources_capacity: dict[str, int] = None,
        horizon_start_shift: Optional[int] = 0,
        objective_handling: ObjectiveHandling = ObjectiveHandling.AGGREGATE,
    ):
        self.team_names = team_names
        self.calendar_team = realign_calendars(calendar_team)
        self.horizon = horizon
        self.tasks_list = tasks_list
        self.tasks_data = tasks_data
        self.same_allocation = same_allocation
        self.precedence_constraints = precedence_constraints

        self.available_team_for_activity = available_team_for_activity
        self.start_window = start_window
        self.end_window = end_window
        self.original_start = original_start
        self.original_end = original_end
        self.number_teams = len(self.team_names)
        self.number_tasks = len(self.tasks_list)
        self.teams_to_index = {self.team_names[i]: i for i in range(self.number_teams)}
        self.index_to_team = {i: self.team_names[i] for i in range(self.number_teams)}
        self.tasks_to_index = {self.tasks_list[i]: i for i in range(self.number_tasks)}
        self.index_to_task = {
            self.tasks_to_index[key]: key for key in self.tasks_to_index
        }
        self.rcpsp, self.ac_mode_to_team = transform_to_multimode_rcpsp(
            self,
            build_calendar=True,
            add_window_time_constraint=True,
            add_additional_constraint=True,
        )
        self.objective_handling = objective_handling
        self.resources_list = resources_list
        self.resources_capacity = resources_capacity
        self.horizon_start_shift = horizon_start_shift
        self.compatible_teams_per_activity = self.compatible_teams_all_activity()

    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return self.team_names

    def get_precedence_constraints(self) -> dict[Task, set[Task]]:
        return self.precedence_constraints

    def is_compatible_task_unary_resource(
        self, task: Task, unary_resource: UnaryResource
    ) -> bool:
        return unary_resource in self.compatible_teams_per_activity[task]

    def get_makespan_lower_bound(self) -> int:
        return max(int(self.get_lb_end_window(t)) for t in self.tasks_list)

    def get_makespan_upper_bound(self) -> int:
        return max(int(self.get_ub_end_window(t)) for t in self.tasks_list)

    def set_objective_handling(self, objective_handling: ObjectiveHandling):
        self.objective_handling = objective_handling

    def update_available_team_for_activity(
        self, available_team_for_activity: dict[Hashable, set[Hashable]]
    ):
        """Override the available team attribute and update the rcpsp accordingly"""
        self.available_team_for_activity = available_team_for_activity
        self.rcpsp, self.ac_mode_to_team = transform_to_multimode_rcpsp(
            self,
            build_calendar=True,
            add_window_time_constraint=True,
            add_additional_constraint=True,
        )

    def compute_predecessors(self):
        self.predecessors = {}
        for t in self.precedence_constraints:
            for succ in self.precedence_constraints[t]:
                if succ not in self.predecessors:
                    self.predecessors[succ] = set()
                self.predecessors[succ].add(t)

    def evaluate(self, variable: Solution) -> dict[str, float]:
        return evaluate_solution(solution=variable, problem=self)

    def satisfy(self, variable: Solution) -> bool:
        if isinstance(variable, RcpspSolution):
            return self.rcpsp.satisfy(variable)
        if isinstance(variable, AllocSchedulingSolution):
            return full_satisfy(problem=self, solution=variable, partial_solution=False)
            # Deprecated version was transforming the solution into an RCPSPSolution.
            # return self.rcpsp.satisfy(transform_alloc_solution_to_rcpsp_solution(alloc_solution=variable,
            #                                                                      rcpsp_problem=self.rcpsp,
            #                                                                      ac_mode_to_team=self.ac_mode_to_team,
            #                                                                      alloc_scheduling_problem=self))

    def get_attribute_register(self) -> EncodingRegister:
        pass

    def get_solution_type(self) -> type[Solution]:
        return AllocSchedulingSolution

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "nb_not_done": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100000
            ),
            "nb_teams": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-10000.0
            ),
            "makespan": ObjectiveDoc(type=TypeObjective.PENALTY, default_weight=-1.0),
            "workload_dispersion": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-1.0
            ),
            "nb_violations": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100.0
            ),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=self.objective_handling,
            dict_objective_to_doc=dict_objective,
        )

    def get_lb_start_window(self, task: Hashable) -> int:
        if task not in self.start_window or self.start_window[task][0] is None:
            return 0
        else:
            return self.start_window[task][0]

    def get_ub_start_window(self, task: Hashable) -> int:
        if task not in self.start_window or self.start_window[task][1] is None:
            return self.horizon - self.tasks_data[task].duration_task
        else:
            return self.start_window[task][1]

    def get_lb_end_window(self, task: Hashable) -> int:
        if task not in self.end_window or self.end_window[task][0] is None:
            return self.get_lb_start_window(task) + self.tasks_data[task].duration_task
        else:
            return self.end_window[task][0]

    def get_ub_end_window(self, task: Hashable) -> int:
        if task not in self.end_window or self.end_window[task][1] is None:
            return self.get_ub_start_window(task) + self.tasks_data[task].duration_task
        else:
            return self.end_window[task][1]

    def get_all_lb_ub(self) -> list[tuple[int, int, int, int]]:
        """
        Return a list of
        (lb_start, ub_start, lb_end, ub_end) for each task
        (lower/upper bound on start, lower/upper bound on end)
        """
        return [
            (
                int(self.get_lb_start_window(t)),
                int(self.get_ub_start_window(t)),
                int(self.get_lb_end_window(t)),
                int(self.get_ub_end_window(t)),
            )
            for t in self.tasks_list
        ]

    def get_unavailable_teams_per_activity(self) -> dict[Hashable, set[Hashable]]:
        all_teams = set(self.team_names)
        return {
            t: all_teams.difference(self.available_team_for_activity[t])
            for t in self.available_team_for_activity
        }

    def compatible_teams_all_activity(self) -> dict[Hashable, set[Hashable]]:
        all_teams = set(self.team_names)
        d = {t: all_teams for t in self.tasks_list}
        for t in self.available_team_for_activity:
            d[t] = self.available_team_for_activity[t]
        return d

    def compatible_teams_index_all_activity(self) -> dict[int, set[int]]:
        all_teams = set(self.index_to_team.keys())
        d = {index_task: all_teams for index_task in self.index_to_task}
        for t in self.available_team_for_activity:
            d[self.tasks_to_index[t]] = {
                self.teams_to_index[team]
                for team in self.available_team_for_activity[t]
            }
        return d

    def compute_unavailability_calendar(self, team: Hashable) -> list[tuple[int, int]]:
        # Compute the "complement" calendar of the availability calendar.
        cur_time = 0
        list_unavailable = []
        for i in range(len(self.calendar_team[team])):
            if self.calendar_team[team][i][0] > cur_time:
                list_unavailable.append((cur_time, self.calendar_team[team][i][0]))
            cur_time = self.calendar_team[team][i][1]
        if cur_time < self.horizon:
            list_unavailable.append((cur_time, self.horizon))
        return list_unavailable


def evaluate_solution(
    solution: AllocSchedulingSolution, problem: AllocSchedulingProblem
) -> dict[str, float]:
    dur_per_team = {}
    teams_used = set()
    nb_not_done = 0
    for i in range(len(solution.allocation)):
        team = solution.allocation[i]
        if team is None or team == -1:  # convention to be respected.
            nb_not_done += 1
        else:
            teams_used.add(team)
        if team not in dur_per_team:
            dur_per_team[team] = 0
        dur_per_team[team] += problem.tasks_data[problem.index_to_task[i]].duration_task
    makespan = max(solution.schedule[:, 1])
    sat = satisfy_detailed(problem=problem, solution=solution)
    return {
        "nb_teams": len(teams_used),
        "nb_not_done": nb_not_done,
        "makespan": makespan,
        "workload_dispersion": max(dur_per_team.values()) - min(dur_per_team.values()),
        "nb_violations": len(sat),
    }


def satisfy_all_done(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution,
    partial_solution: bool = False,
):
    nb_not_done = 0
    for i in range(len(solution.allocation)):
        team = solution.allocation[i]
        if team is None or team == -1 or np.isnan(team):  # convention to be respected.
            nb_not_done += 1
    return nb_not_done == 0


def satisfy_precedence(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution,
    partial_solution: bool = False,
) -> bool:
    """
    Partial solution = True means we ignore variable set to None when we check the constraint.
    """
    for task in problem.precedence_constraints:
        index_task = problem.tasks_to_index[task]
        end_task = solution.schedule[index_task, 1]
        if partial_solution and np.isnan(end_task):
            continue
        elif np.isnan(end_task):
            return False
        for successor_task in problem.precedence_constraints[task]:
            if partial_solution and np.isnan(
                solution.schedule[problem.tasks_to_index[successor_task], 0]
            ):
                continue
            if solution.schedule[problem.tasks_to_index[successor_task], 0] < end_task:
                logging.info("Precedence not respected")
                return False
    return True


def satisfy_same_allocation(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution,
    partial_solution: bool = False,
) -> bool:
    """
    Partial solution = True means we ignore variable set to None when we check the constraint.
    This one is a bit tricky to write when we allow partial solution.
    """
    for set_same_alloc in problem.same_allocation:
        if not partial_solution:
            one_ac = next(iter(set_same_alloc))
            val = solution.allocation[problem.tasks_to_index[one_ac]]
            if any(
                solution.allocation[problem.tasks_to_index[x]] != val
                for x in set_same_alloc
            ):
                logging.info("Same alloc not respected")
                return False
        else:
            one_ac = next(
                (
                    s
                    for s in set_same_alloc
                    if not np.isnan(solution.allocation[problem.tasks_to_index[s]])
                ),
                None,
            )
            if one_ac is None:
                continue
            else:
                val = solution.allocation[problem.tasks_to_index[one_ac]]
                if any(
                    solution.allocation[problem.tasks_to_index[x]] != val
                    for x in set_same_alloc
                    if not np.isnan(solution.allocation[problem.tasks_to_index[x]])
                ):
                    return False
    return True


def satisfy_available_team(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution,
    partial_solution: bool = False,
) -> bool:
    """
    Partial solution = True means we ignore variable set to None when we check the constraint.
    """
    for activity in problem.available_team_for_activity:
        alloc = solution.allocation[problem.tasks_to_index[activity]]
        if partial_solution and (alloc is None or np.isnan(alloc)):
            continue
        if np.isnan(alloc):
            logging.info("Team available not respected")
            return False
        if alloc == -1:
            return False
        team_alloc = problem.index_to_team[alloc]
        if team_alloc not in problem.available_team_for_activity[activity]:
            logging.info("Team available not respected")
            return False
    return True


def realign_calendars(calendars_dict: dict[Hashable, list[tuple[int, int]]]):
    new_cals = {}
    # logger.debug(calendars_dict)
    for t in calendars_dict:
        new_cal = []
        if len(calendars_dict[t]) <= 1:
            new_cals[t] = calendars_dict[t]
            continue
        else:
            current_ = (calendars_dict[t][0][0], calendars_dict[t][0][1])
            current_end = current_[1]
            current_index = 1
            while current_index < len(calendars_dict[t]):
                if calendars_dict[t][current_index][0] == current_end:
                    current_ = (current_[0], calendars_dict[t][current_index][1])
                    current_end = calendars_dict[t][current_index][1]
                else:
                    new_cal.append(current_)
                    current_ = calendars_dict[t][current_index]
                    current_end = current_[1]
                current_index += 1
            if len(new_cal) == 0 or new_cal[-1] != current_:
                new_cal.append(current_)
            new_cals[t] = new_cal
    # logger.debug(new_cals)
    return new_cals


def intervals_do_not_overlap(
    interval1: tuple[float, float], interval2: tuple[float, float]
):
    if interval1[1] <= interval2[0] or interval2[1] <= interval1[0]:
        return True
    else:
        return False


def interval_inside(
    interval1: tuple[float, float], interval_container: tuple[float, float]
):
    if interval1[0] >= interval_container[0] and interval1[1] <= interval_container[1]:
        return True
    return False


def satisfy_calendars(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution,
    partial_solution: bool = False,
):
    for i in range(len(solution.allocation)):
        val = solution.allocation[i]
        if np.isnan(val) or val not in problem.index_to_team:
            # This case is tackled by some other function
            continue
        st, end = solution.schedule[i, :]
        calendar = problem.calendar_team[problem.index_to_team[val]]
        if not any(interval_inside((st, end), x) for x in calendar):
            # logger.info(f"Calendar constraint not respected for task {i}")
            # logger.info(f"here's (availability) calendar and the task to schedule {calendar, (st, end)}")
            return False
    return True


def satisfy_time_window(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution,
    partial_solution: bool = False,
):
    for t in range(solution.schedule.shape[0]):
        tsk = problem.tasks_list[t]
        lb_s, ub_s = problem.start_window.get(tsk, (None, None))
        lb_e, ub_e = problem.end_window.get(tsk, (None, None))
        st, end = solution.schedule[t, :]
        if lb_s is not None:
            if st < lb_s:
                ##logger.info(f"Task {t} starting before {lb_s}")
                return False
        if ub_s is not None:
            if st > ub_s:
                ##logger.info(f"Task {t} starting after {ub_s}")
                return False
        if lb_e is not None:
            if end < lb_e:
                ##logger.info(f"Task {t} ending before {lb_e}")
                return False
        if ub_e is not None:
            if end > ub_e:
                ##logger.info(f"Task {t} ending after {ub_e}")
                return False
    return True


def full_satisfy(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution,
    partial_solution: bool = False,
) -> bool:
    is_satisfied = True
    for func in [
        satisfy_all_done,
        satisfy_precedence,
        satisfy_available_team,
        satisfy_same_allocation,
        satisfy_time_window,
        satisfy_calendars,
        satisfy_overlap_teams,
    ]:
        if not func(
            problem=problem, solution=solution, partial_solution=partial_solution
        ):
            logger.warning(func, " not satisfied !!")
            is_satisfied = False
    return is_satisfied


def satisfy_detailed(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution
):
    # TODO : detail computation of calendar constraint violated.
    return reduce(
        lambda x, y: x + y(problem, solution),
        [
            satisfy_detailed_all_done,
            satisfy_detailed_precedence,
            satisfy_detailed_same_allocation,
            satisfy_detailed_available_team,
            satisfy_time_window_detailed,
            satisfy_overlap_teams_detailed,
        ],
        [],
    )


def satisfy_detailed_all_done(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution
):
    violations = []
    for i in range(len(solution.allocation)):
        team = solution.allocation[i]
        if team is None or team == -1 or np.isnan(team):  # convention to be respected.
            violations.append(({"task_index": i, "tag": "is_not_done"}))
    return violations


def satisfy_time_window_detailed(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution
):
    violations = []
    for t in range(solution.schedule.shape[0]):
        tsk = problem.tasks_list[t]
        lb_s, ub_s = problem.start_window.get(tsk, (None, None))
        lb_e, ub_e = problem.end_window.get(tsk, (None, None))
        st, end = solution.schedule[t, :]
        if solution.allocation[t] not in problem.index_to_team:
            continue
        if lb_s is not None:
            if st < lb_s:
                ###logger.info(f"Task {t} starting before {lb_s}")
                violations.append(
                    {
                        "task_index": t,
                        "start": st,
                        "expected": lb_s,
                        "tag": "early",
                        "violation": lb_s - st,
                    }
                )
        if ub_s is not None:
            if st > ub_s:
                ###logger.info(f"Task {t} starting after {ub_s}")
                violations.append(
                    {
                        "task_index": t,
                        "start": st,
                        "expected": ub_s,
                        "tag": "late",
                        "violation": st - ub_s,
                    }
                )
        if lb_e is not None:
            if end < lb_e:
                ###logger.info(f"Task {t} ending before {lb_e}")
                violations.append(
                    {
                        "task_index": t,
                        "end": end,
                        "expected": lb_e,
                        "tag": "early",
                        "violation": lb_e - end,
                    }
                )
        if ub_e is not None:
            if end > ub_e:
                ###logger.info(f"Task {t} ending after {ub_e}")
                violations.append(
                    {
                        "task_index": t,
                        "end": end,
                        "expected": ub_e,
                        "tag": "late",
                        "violation": end - ub_e,
                    }
                )
    return violations


def satisfy_detailed_precedence(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution
) -> list[tuple[str, Hashable, Hashable, int]]:
    list_violated_precedence_constraint = []
    for task in problem.precedence_constraints:
        index_task = problem.tasks_to_index[task]
        end_task = solution.schedule[index_task, 1]
        if solution.allocation[index_task] == -1 or np.isnan(
            solution.allocation[index_task]
        ):
            continue
        for successor_task in problem.precedence_constraints[task]:
            index_succ = problem.tasks_to_index[successor_task]
            if solution.allocation[index_succ] == -1 or np.isnan(
                solution.allocation[index_succ]
            ):
                continue
            if solution.schedule[problem.tasks_to_index[successor_task], 0] < end_task:
                list_violated_precedence_constraint += [
                    (
                        "precedence",
                        task,
                        successor_task,
                        index_task,
                        problem.tasks_to_index[successor_task],
                        end_task
                        - solution.schedule[problem.tasks_to_index[successor_task], 0],
                        end_task,
                        solution.schedule[problem.tasks_to_index[successor_task], 0],
                    )
                ]
    return list_violated_precedence_constraint


def satisfy_detailed_same_allocation(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution
) -> list[tuple[str, set[Hashable]], set[int]]:
    list_violated_same_allocation_constraint: list[
        tuple[str, set[Hashable]], set[int]
    ] = []
    for set_same_alloc in problem.same_allocation:
        one_ac = next(iter(set_same_alloc))
        val = solution.allocation[problem.tasks_to_index[one_ac]]
        if np.isnan(val) or val == -1:
            continue
        if any(
            (
                solution.allocation[problem.tasks_to_index[x]] != val
                and solution.allocation[problem.tasks_to_index[x]]
                in problem.index_to_team
            )
            for x in set_same_alloc
        ):
            list_violated_same_allocation_constraint += [
                (
                    "same_allocation",
                    set_same_alloc,
                    {problem.tasks_to_index[i] for i in set_same_alloc},
                )
            ]
    return list_violated_same_allocation_constraint


def satisfy_detailed_available_team(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution
) -> list[tuple[str, Hashable, Hashable, int, int]]:
    list_violated_available_team = []
    for activity in problem.available_team_for_activity:
        team: int = int(solution.allocation[problem.tasks_to_index[activity]])
        if team not in problem.index_to_team:
            continue
        team_alloc = problem.index_to_team[team]
        if team_alloc not in problem.available_team_for_activity[activity]:
            list_violated_available_team += [
                (
                    "available-team",
                    activity,
                    team_alloc,
                    problem.tasks_to_index[activity],
                    problem.teams_to_index[team_alloc],
                )
            ]
    return list_violated_available_team


def satisfy_overlap_teams(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution, **kwargs
) -> bool:
    teams = set(solution.allocation)
    for team in teams:
        if team not in problem.index_to_team:
            continue
        ac_team = np.nonzero(solution.allocation == team)
        schedule_ = solution.schedule[ac_team, :]
        slots = sorted([tuple(x) for x in schedule_])
        for j in range(1, len(slots)):
            if slots[j][0] < slots[j - 1][1]:
                return False
    return True


def satisfy_overlap_teams_detailed(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution
) -> list[tuple[str, int, int, int]]:
    teams = set(solution.allocation)
    list_violated_overlap = []
    for team in teams:
        if team not in problem.index_to_team:
            continue
        ac_team = np.nonzero(solution.allocation == team)
        schedule_ = solution.schedule[ac_team[0], :]
        slots = sorted([tuple(x) for x in schedule_])
        for j in range(1, len(slots)):
            if slots[j][0] < slots[j - 1][1]:
                list_violated_overlap.append(
                    (
                        "no-overlap",
                        int(ac_team[0][j - 1]),
                        int(ac_team[0][j]),
                        int(team),
                    )
                )
    return list_violated_overlap


def compute_stats_per_team(
    problem: AllocSchedulingProblem, solution: AllocSchedulingSolution
) -> dict[int, float]:
    teams = set(solution.allocation)
    used_time_by_team = {}
    for team in teams:
        if team not in problem.index_to_team:
            continue
        slots = problem.compute_unavailability_calendar(problem.index_to_team[team])
        logger.debug("slots", slots)
        ac_team = np.nonzero(solution.allocation == team)
        schedule_ = solution.schedule[ac_team[0], :]
        used_time = np.sum(schedule_[:, 1] - schedule_[:, 0])
        used_time += sum([x[1] - x[0] for x in slots])
        used_time_by_team[team] = used_time
    return used_time_by_team


def transform_rcpsp_solution_to_alloc_solution(
    rcpsp_solution: RcpspSolution,
    rcpsp_problem: RcpspProblem,
    ac_mode_to_team: dict[tuple[Hashable, int], Hashable],
    alloc_scheduling_problem: AllocSchedulingProblem,
) -> AllocSchedulingSolution:
    schedule = np.zeros((alloc_scheduling_problem.number_tasks, 2), dtype=int)
    allocation = np.zeros(alloc_scheduling_problem.number_tasks, dtype=int)
    modes_dict = rcpsp_problem.build_mode_dict(rcpsp_solution.rcpsp_modes)
    for activity in alloc_scheduling_problem.tasks_list:
        index_ = alloc_scheduling_problem.tasks_to_index[activity]
        start = rcpsp_solution.get_start_time(activity)
        end = rcpsp_solution.get_end_time(activity)
        schedule[index_, 0] = start
        schedule[index_, 1] = end
        mode = modes_dict[activity]
        team = ac_mode_to_team[(activity, mode)]
        allocation[index_] = alloc_scheduling_problem.teams_to_index[team]
    return AllocSchedulingSolution(
        problem=alloc_scheduling_problem, schedule=schedule, allocation=allocation
    )


def transform_alloc_solution_to_rcpsp_solution(
    alloc_solution: AllocSchedulingSolution,
    rcpsp_problem: RcpspProblem,
    ac_mode_to_team: dict[tuple[Hashable, int], Hashable],
    alloc_scheduling_problem: AllocSchedulingProblem,
) -> RcpspSolution:
    schedule = {}
    modes_dict = {}
    max_ = 0
    for activity in alloc_scheduling_problem.tasks_list:
        index_ = alloc_scheduling_problem.tasks_to_index[activity]
        start = int(alloc_solution.schedule[index_, 0])
        end = int(alloc_solution.schedule[index_, 1])
        max_ = max(end, max_)
        team = alloc_scheduling_problem.index_to_team[alloc_solution.allocation[index_]]
        schedule[activity] = {"start_time": start, "end_time": end}
        modes_dict[activity] = next(
            (
                x[1]
                for x in ac_mode_to_team
                if x[0] == activity and ac_mode_to_team[x] == team
            ),
            None,
        )
        if modes_dict[activity] is None:
            pass
            ###logger.info(f"{activity}, not allocated to any mode ")
    schedule[rcpsp_problem.source_task] = {"start_time": 0, "end_time": 0}
    schedule[rcpsp_problem.sink_task] = {"start_time": max_, "end_time": max_}
    return RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_schedule=schedule,
        rcpsp_modes=[modes_dict[t] for t in rcpsp_problem.tasks_list_non_dummy],
    )


def build_calendar_array_from_availability_slot(
    availability_slots: list[tuple[int, int]], horizon: int, value: int = 1
):
    array = np.zeros(horizon, dtype=int)
    for slot in availability_slots:
        array[slot[0] : min(slot[1], horizon)] = value
    return array


def transform_to_multimode_rcpsp(
    problem: AllocSchedulingProblem,
    build_calendar: bool = True,
    add_window_time_constraint: bool = True,
    add_additional_constraint: bool = True,
) -> tuple[RcpspProblem, dict[tuple[Hashable, int], Hashable]]:
    if build_calendar:
        resources = {
            team: build_calendar_array_from_availability_slot(
                availability_slots=problem.calendar_team[team],
                horizon=problem.horizon,
                value=1,
            )
            for team in problem.calendar_team
        }
    else:
        resources = {team: 1 for team in problem.calendar_team}
    non_renewable_resources: list[str] = []
    mode_details: dict[Hashable, dict[int, dict[str, int]]] = {}
    ac_mode_to_team = {}
    for activity in problem.tasks_list:
        if activity in problem.available_team_for_activity:
            subset_teams = list(problem.available_team_for_activity[activity])
        else:
            subset_teams = problem.team_names
        mode_details[activity] = {}
        modes = [i for i in range(1, len(subset_teams) + 1)]
        for j in range(len(modes)):
            mode_details[activity][modes[j]] = {
                "duration": problem.tasks_data[activity].duration_task,
                subset_teams[j]: 1,
            }  # "consume" this team
            ac_mode_to_team[(activity, modes[j])] = subset_teams[j]
    mode_details["source"] = {1: {"duration": 0}}
    mode_details["sink"] = {1: {"duration": 0}}
    successors: dict[Hashable, list[Hashable]] = {
        t: list(problem.precedence_constraints[t])
        for t in problem.precedence_constraints
    }
    graph = nx.DiGraph()
    for activity in problem.tasks_list:
        graph.add_node(activity)
    for activity in successors:
        for succ_ac in successors[activity]:
            graph.add_edge(activity, succ_ac)
    graph.add_node("source")
    graph.add_node("sink")
    for activity in problem.tasks_list:
        graph.add_edge("source", activity)
        graph.add_edge(activity, "sink")
    reduced_graph: nx.DiGraph = nx.transitive_reduction(graph)
    recomputed_successors = {
        t: list(nx.neighbors(reduced_graph, t)) for t in reduced_graph.nodes()
    }
    horizon: int = problem.horizon
    horizon_multiplier: int = 1
    tasks_list: Optional[list[Hashable]] = ["source"] + problem.tasks_list + ["sink"]
    source_task: Optional[Hashable] = "source"
    sink_task: Optional[Hashable] = "sink"
    special_constraints: Optional[SpecialConstraintsDescription] = (
        SpecialConstraintsDescription(
            start_times_window=problem.start_window
            if add_window_time_constraint
            else None,
            end_times_window=problem.end_window if add_window_time_constraint else None,
        )
    )
    rcpsp = RcpspProblem(
        resources=resources,
        non_renewable_resources=non_renewable_resources,
        mode_details=mode_details,
        successors=recomputed_successors,
        horizon=horizon,
        tasks_list=tasks_list,
        source_task="source",
        sink_task="sink",
        special_constraints=special_constraints if add_additional_constraint else None,
    )
    if add_additional_constraint:
        pair_mode_constraint = build_pair_mode_constraint(
            problem=problem, rcpsp=rcpsp, use_score=True
        )
        rcpsp.special_constraints.pair_mode_constraint = pair_mode_constraint
    return rcpsp, ac_mode_to_team


def transform_to_monomode_rcpsp(
    problem: AllocSchedulingProblem,
    build_calendar: bool = True,
    add_additional_constraint: bool = True,
) -> tuple[RcpspProblem, dict[tuple[Hashable, int], Hashable]]:
    rcpsp, ac_mode_to_team = transform_to_multimode_rcpsp(
        problem=problem,
        build_calendar=build_calendar,
        add_additional_constraint=add_additional_constraint,
    )
    all_sets = set()

    task_to_set = {}
    for task in rcpsp.mode_details:
        teams = [
            ac_mode_to_team[task, mode]
            for mode in sorted(rcpsp.mode_details[task])
            if (task, mode) in ac_mode_to_team
        ]
        if len(teams) > 0:
            teams = tuple(sorted(teams))
            all_sets.add(teams)
            task_to_set[task] = teams
    list_all_sets: list[set] = list(all_sets)
    res = [f"res_{i}" for i in range(len(list_all_sets))]
    teams_to_res = {list_all_sets[i]: f"res_{i}" for i in range(len(list_all_sets))}
    inclusion = {}
    for i in range(len(list_all_sets)):
        for j in range(i + 1, len(list_all_sets)):
            if set(list_all_sets[i]).issubset(set(list_all_sets[j])):
                if list_all_sets[i] not in inclusion:
                    inclusion[list_all_sets[i]] = []
                inclusion[list_all_sets[i]] += [list_all_sets[j]]
            if set(list_all_sets[i]).issuperset(set(list_all_sets[j])):
                if list_all_sets[j] not in inclusion:
                    inclusion[list_all_sets[j]] = []
                inclusion[list_all_sets[j]] += [list_all_sets[i]]
    capacity = [
        sum([rcpsp.get_max_resource_capacity(x) for x in list_all_sets[i]])
        for i in range(len(list_all_sets))
    ]
    calendars = {}
    for i in range(len(list_all_sets)):
        res = f"res_{i}"
        availability = np.zeros(rcpsp.horizon)
        for res_ in list_all_sets[i]:
            availability += np.array(rcpsp.get_resource_availability_array(res_))
        calendars[res] = [int(x) for x in availability]
    new_mode_details = {}
    for task in rcpsp.mode_details:
        new_mode_details[task] = {}
        if task not in task_to_set:
            new_mode_details[task] = rcpsp.mode_details[task]
        else:
            res_tag = teams_to_res[task_to_set[task]]
            new_mode_details[task][1] = {
                "duration": rcpsp.mode_details[task][1]["duration"],
                res_tag: 1,
            }
            if task_to_set[task] in inclusion:
                for other_set in inclusion[task_to_set[task]]:
                    new_mode_details[task][1][teams_to_res[other_set]] = 1

    rcpsp_model = RcpspProblem(
        resources=calendars,
        non_renewable_resources=[],
        mode_details=new_mode_details,
        successors=rcpsp.successors,
        horizon=rcpsp.horizon,
        source_task=rcpsp.source_task,
        sink_task=rcpsp.sink_task,
        special_constraints=rcpsp.special_constraints,
    )
    return rcpsp_model, ac_mode_to_team


def build_pair_mode_constraint(
    problem: AllocSchedulingProblem, rcpsp: RcpspProblem, use_score: bool = False
):
    modes_allowed_assignment: dict[
        tuple[Hashable, Hashable], list[tuple[Hashable, Hashable]]
    ] = {}
    if problem.same_allocation is not None:
        for set_task in problem.same_allocation:
            for p in itertools.combinations(set_task, 2):
                list_modes = []
                team_to_mode_p0 = {}
                team_to_mode_p1 = {}
                for mode in rcpsp.mode_details[p[0]]:
                    team = [
                        x
                        for x in rcpsp.mode_details[p[0]][mode]
                        if x != "duration" and rcpsp.mode_details[p[0]][mode][x] > 0
                    ][0]
                    team_to_mode_p0[team] = mode
                for mode in rcpsp.mode_details[p[1]]:
                    team = [
                        x
                        for x in rcpsp.mode_details[p[1]][mode]
                        if x != "duration" and rcpsp.mode_details[p[1]][mode][x] > 0
                    ][0]
                    team_to_mode_p1[team] = mode
                teams_0 = set(team_to_mode_p0.keys())
                teams_1 = set(team_to_mode_p1.keys())
                intersect = teams_0.intersection(teams_1)
                for team in intersect:
                    list_modes += [(team_to_mode_p0[team], team_to_mode_p1[team])]
                modes_allowed_assignment[(p[0], p[1])] = list_modes
        task_mode_integer = {}
        index_team = {
            rcpsp.resources_list[i]: i + 1 for i in range(len(rcpsp.resources_list))
        }
        for task in rcpsp.mode_details:
            for mode in rcpsp.mode_details[task]:
                team = [
                    x
                    for x in rcpsp.mode_details[task][mode]
                    if x != "duration"
                    if rcpsp.mode_details[task][mode][x] > 0
                ]
                if len(team) > 0:
                    team = team[0]
                    task_mode_integer[(task, mode)] = index_team[team]
                else:
                    continue
        return PairModeConstraint(
            allowed_mode_assignment=modes_allowed_assignment if not use_score else None,
            same_score_mode=set(modes_allowed_assignment.keys()) if use_score else None,
            score_mode=task_mode_integer if use_score else None,
        )
    else:
        return None


def correct_schedule_avoid_overlap(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution,
    init_min_starting_date_lb: bool = False,
):
    routings = []
    for i in range(problem.number_teams):
        tasks = np.nonzero(solution.allocation == i)
        if len(tasks[0]) > 0:
            sorted_ = tasks[0][np.argsort(solution.schedule[tasks, 0])][0]
            routings.append(sorted_)
        else:
            routings.append([])
    used_teams = {i for i in range(len(routings)) if len(routings[i]) > 0}
    calendar_teams = {
        i: build_calendar_array_from_availability_slot(
            availability_slots=problem.calendar_team[problem.index_to_team[i]],
            horizon=problem.horizon,
            value=1,
        )
        for i in used_teams
    }
    predecessors = {j: set() for j in problem.index_to_task}
    for t in problem.precedence_constraints:
        for succ_t in problem.precedence_constraints[t]:
            predecessors[problem.tasks_to_index[succ_t]].add(problem.tasks_to_index[t])
    scheduled = set()
    min_starting_time = {
        i: max(
            solution.schedule[i, 0],
            int(problem.get_lb_start_window(problem.index_to_task[i])),
        )
        for i in problem.index_to_task
    }
    if init_min_starting_date_lb:
        min_starting_time = {
            i: int(problem.get_lb_start_window(problem.index_to_task[i]))
            for i in problem.index_to_task
        }
    nb_tasks_to_sched = len(problem.index_to_task)
    sorted_tasks = [int(x) for x in np.argsort(solution.schedule[:, 0])]
    schedule_per_team = {i: [] for i in used_teams}
    new_schedule = np.zeros(solution.schedule.shape)
    while len(scheduled) < nb_tasks_to_sched:
        next_t = next(
            t
            for t in sorted_tasks
            if t not in scheduled and all(p in scheduled for p in predecessors[t])
        )
        task_name = problem.index_to_task[next_t]
        sched = solution.schedule[next_t, :]
        team = solution.allocation[next_t]
        if team == -1:
            new_schedule[next_t, 0] = solution.schedule[next_t, 0]
            new_schedule[next_t, 1] = solution.schedule[next_t, 1]
            scheduled.add(next_t)
            continue
        dur = problem.tasks_data[task_name].duration_task
        if len(schedule_per_team[team]) == 0:
            schedule_per_team[team].append((next_t, sched[0], sched[1]))
            calendar_teams[team][sched[0] : sched[1]] = 0
        else:
            t = 0
            last_time = schedule_per_team[team][-1][-1]
            min_time = max(
                last_time + t,
                max(
                    min_starting_time[next_t],
                    int(problem.get_lb_start_window(problem.index_to_task[next_t])),
                ),
            )
            if np.min(calendar_teams[team][min_time : min_time + dur]) >= 1:
                schedule_per_team[team].append((next_t, min_time, min_time + dur))
                calendar_teams[team][min_time : min_time + dur] = 0
            else:
                time_ = next(
                    t
                    for t in range(min_time, problem.horizon)
                    if np.min(calendar_teams[team][t : t + dur]) == 1
                )
                schedule_per_team[team].append((next_t, time_, time_ + dur))
                calendar_teams[team][time_ : time_ + dur] = 0
        if problem.index_to_task[next_t] in problem.precedence_constraints:
            for t in problem.precedence_constraints[problem.index_to_task[next_t]]:
                min_starting_time[problem.tasks_to_index[t]] = max(
                    min_starting_time[problem.tasks_to_index[t]],
                    schedule_per_team[team][-1][-1],
                )
        new_schedule[next_t, 0] = schedule_per_team[team][-1][1]
        new_schedule[next_t, 1] = schedule_per_team[team][-1][2]
        scheduled.add(next_t)
    return AllocSchedulingSolution(
        problem=problem, schedule=new_schedule, allocation=solution.allocation
    )


def export_scheduling_problem_json(problem: AllocSchedulingProblem) -> dict:
    d = dict()
    d["teams"] = problem.team_names
    d["tasks"] = [str(x) for x in problem.tasks_list]
    d["calendar"] = problem.calendar_team
    for t in d["calendar"]:
        d["calendar"][t] = [(int(x[0]), int(x[1])) for x in d["calendar"][t]]
    d["teams_to_index"] = problem.teams_to_index
    d["tasks_data"] = {
        int(t): {"duration": problem.tasks_data[t].duration_task}
        for t in problem.tasks_data
    }
    d["same_allocation"] = [[str(y) for y in x] for x in problem.same_allocation]
    d["compatible_teams"] = {
        str(t): list(problem.available_team_for_activity[t])
        for t in problem.available_team_for_activity
    }
    d["start_window"] = {str(t): problem.start_window[t] for t in problem.start_window}
    d["end_window"] = {str(t): problem.end_window[t] for t in problem.end_window}
    d["successors"] = {
        str(t): [str(succ) for succ in problem.precedence_constraints[t]]
        for t in problem.precedence_constraints
    }
    d["horizon_shift"] = int(problem.horizon_start_shift)
    d["original_start"] = {
        int(x): problem.original_start[x] for x in problem.original_start
    }
    d["original_end"] = {int(x): problem.original_end[x] for x in problem.original_end}
    return d
