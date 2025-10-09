#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Hashable
from copy import deepcopy
from enum import Enum
from itertools import product
from typing import Any, Optional

import networkx as nx
import numpy as np
from networkx import bipartite

from discrete_optimization.coloring.problem import ColoringConstraints, ColoringProblem
from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeAttribute,
    TypeObjective,
)
from discrete_optimization.generic_tools.graph_api import Graph

logger = logging.getLogger(__file__)


Task = Hashable
UnaryResource = Hashable


def compute_available_teams_per_activities(
    starts: np.ndarray,
    ends: np.ndarray,
    activities_name: list[Hashable],
    calendars_team: dict[Hashable, np.ndarray],
):
    available_team_per_activity = {}
    for i in range(len(starts)):
        available_team_per_activity[activities_name[i]] = set()
        st, end = starts[i], ends[i]
        for team in calendars_team:
            if st == end:
                if calendars_team[team][int(st)] == 1:
                    available_team_per_activity[activities_name[i]].add(team)
            elif np.min(calendars_team[team][int(st) : int(end)]) == 1:
                available_team_per_activity[activities_name[i]].add(team)
    return available_team_per_activity


class AllocationAdditionalConstraint:
    """
    Object to store potentially dynamically defined constraints.
    Some can be redundant
    """

    # each set of activities should be allocated to same team.
    same_allocation: list[set[Hashable]]
    # allocated team inside the set should be different : "cliques constraint".
    # It could be modeled in the coloring graph
    all_diff_allocation: list[set[Hashable]]
    # force allocation
    # could be modeled in the "allocation" graph
    forced_allocation: dict[Hashable, Hashable]
    # forbidden allocation
    # (could be modeled in the "allocation" graph)
    forbidden_allocation: dict[Hashable, set[Hashable]]
    # allowed allocation
    # (could be modeled in the allocation graph)
    allowed_allocation: dict[Hashable, set[Hashable]]
    # Disjunction
    # each list in disjunction : [(act, team), (act2, team2)...]
    # will mean that at least 1 act_i will need to be allocated to team_i
    disjunction: list[list[tuple[Hashable, Hashable]]]
    # Number of teams to be used : useful for scenario where we limit number of resource
    nb_max_teams: int
    # Precedence constraints : not necessarily useful for allocation problem
    # Except for unsat problems where task can be dropped, then the successors should be dropped too.
    # dict = {parent: set of children tasks}
    precedences: dict[Hashable, set[Hashable]]

    def __init__(
        self,
        same_allocation: Optional[list[set[Hashable]]] = None,
        all_diff_allocation: Optional[list[set[Hashable]]] = None,
        forced_allocation: Optional[dict[Hashable, Hashable]] = None,
        forbidden_allocation: Optional[dict[Hashable, set[Hashable]]] = None,
        allowed_allocation: Optional[dict[Hashable, set[Hashable]]] = None,
        disjunction: Optional[list[list[tuple[Hashable, Hashable]]]] = None,
        nb_max_teams: Optional[int] = None,
        precedences: Optional[dict[Hashable, set[Hashable]]] = None,
    ):
        self.same_allocation = same_allocation
        self.all_diff_allocation = all_diff_allocation
        self.forced_allocation = forced_allocation
        self.forbidden_allocation = forbidden_allocation
        self.allowed_allocation = allowed_allocation
        self.disjunction = disjunction
        self.nb_max_teams = nb_max_teams
        self.precedences = precedences

    def is_empty(self):
        return all(
            getattr(self, x) is None
            for x in [
                "same_allocation",
                "all_diff_allocation",
                "forced_allocation",
                "forbidden_allocation",
                "allowed_allocation",
                "disjunction",
                "nb_max_teams",
                "precedences",
            ]
        )

    def __str__(self):
        s = "Additional constraints \n"
        for attr in [
            "same_allocation",
            "all_diff_allocation",
            "forced_allocation",
            "forbidden_allocation",
            "allowed_allocation",
            "disjunction",
            "nb_max_teams",
            "precedences",
        ]:
            s += f"{attr} = {getattr(self, attr)}\n"
        return s


class GraphBipartite(Graph):
    def __init__(
        self,
        nodes: list[tuple[Hashable, dict[str, Any]]],
        edges: list[tuple[Hashable, Hashable, dict[str, Any]]],
        nodes_activity: set[Hashable],
        nodes_team: set[Hashable],
        undirected: bool = True,
        compute_predecessors: bool = True,
    ):
        self.nodes_activity = nodes_activity
        self.nodes_team = nodes_team
        # Sanity check
        nodes_post_treated = []
        for node in nodes:
            dd = node[1]
            if node[0] in self.nodes_team:
                dd["bipartite"] = 0
            if node[0] in self.nodes_activity:
                dd["bipartite"] = 1
            nodes_post_treated += [(node[0], dd)]
        super().__init__(
            nodes=nodes_post_treated,
            edges=edges,
            undirected=undirected,
            compute_predecessors=compute_predecessors,
        )
        assert all(n in self.nodes_infos_dict for n in self.nodes_activity)
        assert all(n in self.nodes_infos_dict for n in self.nodes_team)
        self.graph_nx = self.to_networkx()
        self.nodes_team_list = list(self.nodes_team)

    def to_networkx(self) -> nx.DiGraph:
        graph_nx = nx.DiGraph() if not self.undirected else nx.Graph()
        graph_nx.add_nodes_from(self.nodes)
        graph_nx.add_edges_from(self.edges)
        return graph_nx

    def is_bipartite(self) -> bool:
        return bipartite.is_bipartite(self.graph_nx)

    def get_nodes_team(self):
        return self.nodes_team

    def get_nodes_team_list(self):
        return self.nodes_team_list

    def get_nodes_activity(self):
        return self.nodes_activity


class TeamAllocationSolution(AllocationSolution[Task, UnaryResource]):
    problem: TeamAllocationProblem

    def __init__(
        self,
        problem: TeamAllocationProblem,
        allocation: list[Optional[int]],
        **kwargs,
    ):
        self.problem = problem
        self.allocation = allocation
        self.kpis = kwargs

    def copy(self) -> "Solution":
        return TeamAllocationSolution(
            problem=self.problem, allocation=deepcopy(self.allocation), **self.kpis
        )

    def lazy_copy(self) -> "Solution":
        return TeamAllocationSolution(
            problem=self.problem, allocation=self.allocation, **self.kpis
        )

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        i_task = self.problem.index_activities_name[task]
        i_team = self.problem.index_teams_name[unary_resource]
        return self.allocation[i_task] == i_team


def build_graph_allocation_from_calendar_and_schedule(
    starts: np.ndarray,
    ends: np.ndarray,
    calendar_team: dict[Hashable, list[tuple[int, int]]],
    horizon: int,
    tasks_name: list[Hashable],
    teams_name: list[Hashable],
):
    calendars_array = {}
    for team in calendar_team:
        calendars_array[team] = np.zeros(horizon)
        for slot in calendar_team[team]:
            calendars_array[team][
                slot[0] : min(slot[1], calendars_array[team].shape[0])
            ] = 1
    available_team_per_activity = compute_available_teams_per_activities(
        starts=starts,
        ends=ends,
        calendars_team=calendars_array,
        activities_name=tasks_name,
    )
    nodes = [(ac, {"type": "activity"}) for ac in tasks_name]
    nodes_activity = set({ac for ac in tasks_name})
    nodes.extend([(team, {"type": "team"}) for team in teams_name])
    nodes_team = set({team for team in teams_name})
    edges = []
    teams = teams_name
    for ac in available_team_per_activity:
        for team in teams:
            if team not in available_team_per_activity[ac]:
                edges.append((ac, team, {}))
    graph_allocation = GraphBipartite(
        nodes=nodes,
        edges=edges,
        nodes_activity=nodes_activity,
        nodes_team=nodes_team,
        undirected=True,
        compute_predecessors=False,
    )
    return graph_allocation


def compute_graph_coloring(starts: np.ndarray, ends: np.ndarray, task_names: list):
    from discrete_optimization.generic_tools.graph_api import Graph, from_networkx

    track_presence = defaultdict(lambda: set())
    edges = []
    graph_nx = nx.DiGraph()
    for i in range(len(task_names)):
        graph_nx.add_node(
            task_names[i], start=starts[i], end=ends[i], duration=ends[i] - starts[i]
        )
    for i in range(starts.shape[0]):
        for time in range(int(starts[i]), int(ends[i])):
            edges.extend([(task_names[x], task_names[i]) for x in track_presence[time]])
            track_presence[time].add(i)
    edges = set(edges)
    graph_nx.remove_edges_from(graph_nx.edges())
    graph_nx.add_edges_from(list(edges))
    return from_networkx(graph_nx=graph_nx)


class TeamAllocationProblem(AllocationProblem[Task, UnaryResource]):
    def __init__(
        self,
        graph_activity: Optional[Graph] = None,
        graph_allocation: Optional[GraphBipartite] = None,
        allocation_additional_constraint: Optional[
            AllocationAdditionalConstraint
        ] = None,
        schedule_activity: Optional[dict[Task, tuple[int, int]]] = None,
        calendar_team: Optional[dict[UnaryResource, list[tuple[int, int]]]] = None,
        activities_name: Optional[list[Task]] = None,
    ):
        """
        :param graph_activity: graph representing coloring constraint among the activities
        :param graph_allocation: graph representing forbidden task assignment to team (or the opposite)
        :param allocation_additional_constraint: optional additional constraint objects.
        :param schedule_activity: (optional), if the underlying problem is a workforce allocation problem to some
        scheduled-activities, it is possible to specify the schedule this way. Notably, this can lead to efficient
        clique-overlap kind of constraint for some solver.
        :param calendar_team: calendar of the teams.
        """
        if activities_name is None:
            if graph_activity is None:
                raise ValueError(
                    "activities_name and graph_activity cannot be both None."
                )
            else:
                self.activities_name = graph_activity.nodes_name
        else:
            self.activities_name = activities_name
            if graph_activity is not None:
                assert set(activities_name) == set(graph_activity.nodes_name), (
                    "activities_name must have same names as in graph_activity.nodes_name"
                )
        self.calendar_team = calendar_team
        if graph_allocation is None:
            if calendar_team is None:
                raise ValueError(
                    "graph_allocation and calendar_team cannot be both None."
                )
            else:
                orig_starts_arr = np.array(
                    [schedule_activity[n][0] for n in self.activities_name]
                )
                orig_ends_arr = np.array(
                    [schedule_activity[n][1] for n in self.activities_name]
                )
                self.graph_allocation = (
                    build_graph_allocation_from_calendar_and_schedule(
                        starts=orig_starts_arr,
                        ends=orig_ends_arr,
                        calendar_team=self.calendar_team,
                        horizon=int(np.max(orig_ends_arr) + 10),
                        tasks_name=self.activities_name,
                        teams_name=list(self.calendar_team.keys()),
                    )
                )
        else:
            self.graph_allocation = graph_allocation
        if graph_activity is None:
            orig_starts_arr = np.array(
                [schedule_activity[n][0] for n in self.activities_name]
            )
            orig_ends_arr = np.array(
                [schedule_activity[n][1] for n in self.activities_name]
            )
            self.graph_activity = compute_graph_coloring(
                starts=orig_starts_arr,
                ends=orig_ends_arr,
                task_names=self.activities_name,
            )
        else:
            self.graph_activity = graph_activity
        self.number_of_activity = len(self.graph_activity.nodes_name)
        self.teams_name: list[Hashable] = self.graph_allocation.get_nodes_team_list()
        self.number_of_teams = len(self.teams_name)
        self.allocation_additional_constraint = allocation_additional_constraint
        if schedule_activity is None:
            self.schedule = {
                self.activities_name[i]: (
                    self.graph_activity.nodes_infos_dict[self.activities_name[i]][
                        "start"
                    ],
                    self.graph_activity.nodes_infos_dict[self.activities_name[i]][
                        "end"
                    ],
                )
                for i in range(self.number_of_activity)
            }
        else:
            self.schedule = schedule_activity
        self.index_activities_name = {
            self.activities_name[i]: i for i in range(self.number_of_activity)
        }
        self.index_to_activities_name = {
            i: self.activities_name[i] for i in range(self.number_of_activity)
        }
        self.index_teams_name = {
            self.teams_name[i]: i for i in range(self.number_of_teams)
        }
        self.index_to_teams_name = {
            i: self.teams_name[i] for i in range(self.number_of_teams)
        }
        self.compatibility_task_team = self.compute_compatibility_for_all_tasks()

    @property
    def tasks_list(self) -> list[Task]:
        return self.activities_name

    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return self.teams_name

    def is_compatible_task_unary_resource(
        self, task: Task, unary_resource: UnaryResource
    ) -> bool:
        return unary_resource in self.compatibility_task_team[task]

    def reorder_teams_name(self, new_order_teams_name: list[Hashable]):
        self.teams_name = new_order_teams_name
        self.index_teams_name = {
            self.teams_name[i]: i for i in range(self.number_of_teams)
        }
        self.index_to_teams_name = {
            i: self.teams_name[i] for i in range(self.number_of_teams)
        }

    @property
    def do_add_cons(self):
        return (
            self.allocation_additional_constraint is not None
            and not self.allocation_additional_constraint.is_empty()
        )

    def computed_forbidden_team_for_task(self, task: Hashable) -> list[Hashable]:
        return [
            team
            for team in self.index_teams_name
            if team not in self.compute_allowed_team_for_task(task)
        ]

    def compute_compatibility_for_all_tasks(self) -> dict[Task, set[UnaryResource]]:
        return {
            task: set(self.compute_allowed_team_for_task(task))
            for task in self.tasks_list
        }

    def compute_allowed_team_for_task(self, task: Task) -> list[UnaryResource]:
        allowed_team = [
            team
            for team in self.index_teams_name
            if team not in self.graph_allocation.get_neighbors(task)
        ]
        if self.allocation_additional_constraint is not None:
            if self.allocation_additional_constraint.forced_allocation is not None:
                if task in self.allocation_additional_constraint.forced_allocation:
                    allowed_team = [
                        self.allocation_additional_constraint.forced_allocation[task]
                    ]
            if self.allocation_additional_constraint.forbidden_allocation is not None:
                if task in self.allocation_additional_constraint.forbidden_allocation:
                    allowed_team = [
                        t
                        for t in allowed_team
                        if t
                        not in self.allocation_additional_constraint.forbidden_allocation[
                            task
                        ]
                    ]
            if self.allocation_additional_constraint.allowed_allocation:
                if task in self.allocation_additional_constraint.allowed_allocation:
                    allowed_team = [
                        t
                        for t in allowed_team
                        if t
                        in self.allocation_additional_constraint.allowed_allocation[
                            task
                        ]
                    ]
        return allowed_team

    def compute_forbidden_team_index_for_task(self, task: Hashable) -> list[int]:
        return [
            self.index_teams_name[team]
            for team in self.computed_forbidden_team_for_task(task)
        ]

    def compute_allowed_team_index_for_task(self, task: Hashable) -> list[int]:
        return [
            self.index_teams_name[team]
            for team in self.compute_allowed_team_for_task(task)
        ]

    def compute_forbidden_team_index_all_task(self) -> list[list[int]]:
        return [
            self.compute_forbidden_team_index_for_task(self.index_to_activities_name[i])
            for i in range(self.number_of_activity)
        ]

    def compute_allowed_team_index_all_task(self) -> list[list[int]]:
        return [
            self.compute_allowed_team_index_for_task(self.index_to_activities_name[i])
            for i in range(self.number_of_activity)
        ]

    def compute_pair_overlap_index_task(self) -> list[tuple[int, int]]:
        return [
            (self.index_activities_name[e[0]], self.index_activities_name[e[1]])
            for e in self.graph_activity.edges
        ]

    def get_max_teams(self) -> int:
        max_teams = self.number_of_teams
        if self.allocation_additional_constraint is not None:
            if self.allocation_additional_constraint.nb_max_teams is not None:
                max_teams = min(
                    max_teams, self.allocation_additional_constraint.nb_max_teams
                )
        return max_teams

    def evaluate(self, variable: TeamAllocationSolution) -> dict[str, float]:
        """
        Evaluation implementation for TeamAllocationProblem.

        Compute number of allocated teams and violation of the current solution.
        """
        keys = ["nb_teams", "nb_violations"]
        if variable.kpis.get("nb_teams", None) is None:
            if variable.allocation is None:
                raise ValueError("variable.allocation must not be None when evaluating")
            else:
                variable.kpis["nb_teams"] = len(set(variable.allocation))
        if variable.kpis.get("nb_violations", None) is None:
            if variable.allocation is None:
                raise ValueError("variable.allocation must not be None when evaluating")
            else:
                variable.kpis["nb_violations"] = (
                    self.count_allowed_assignment_violations(variable)
                    + self.count_color_constraints_violations(variable)
                )
        if self.do_add_cons:
            keys.append("nb_violations_add_cons")
            if variable.kpis.get("nb_violations_add_cons", None) is None:
                if variable.allocation is None:
                    raise ValueError(
                        "variable.allocation must not be None when evaluating"
                    )
                # TODO : count the number of violated constraint instead of this.
                if not satisfy_additional_constraint(
                    problem=self,
                    solution=variable,
                    additional_constraint=self.allocation_additional_constraint,
                    partial_solution=False,
                ):
                    variable.kpis["nb_violations_add_cons"] = 10
                else:
                    variable.kpis["nb_violations_add_cons"] = 0
        return {k: variable.kpis[k] for k in keys}

    def satisfy(self, variable: TeamAllocationSolution) -> bool:
        """Check the constraint of the solution.

        Check for each edges in the graph if the allocated team of the vertices are different.
        When one counterexample is found, the function directly returns False.
        Args:
            variable (TeamAllocationSolution): the solution object we want to check the feasibility

        Returns: boolean indicating if the solution fulfills the constraint.
        """
        if any(x is None for x in variable.allocation):
            return False
        b = self.satisfy_color_constraints(variable)
        if not b:
            return b
        b = self.satisfy_allowed_assignment(variable)
        if not b:
            return b
        if self.do_add_cons:
            b = satisfy_additional_constraint(
                problem=self,
                solution=variable,
                additional_constraint=self.allocation_additional_constraint,
                partial_solution=False,
            )
            if not b:
                return b
        return True

    def satisfy_color_constraints(self, variable: TeamAllocationSolution) -> bool:
        if len(self.graph_activity.edges) > 0:
            if variable.allocation is None:
                raise ValueError("variable.allocation must not be None")
            for e in self.graph_activity.edges:
                if (
                    variable.allocation[self.index_activities_name[e[0]]]
                    == variable.allocation[self.index_activities_name[e[1]]]
                ):
                    return False
        return True

    def satisfy_allowed_assignment(self, variable: TeamAllocationSolution) -> bool:
        if len(self.graph_allocation.edges) > 0:
            if variable.allocation is None:
                raise ValueError("variable.allocation must not be None")
            for e in self.graph_allocation.edges:
                if e[0] in self.index_activities_name:
                    if (
                        variable.allocation[self.index_activities_name[e[0]]]
                        == self.index_teams_name[e[1]]
                    ):
                        return False
                if e[1] in self.index_activities_name:
                    if (
                        variable.allocation[self.index_activities_name[e[1]]]
                        == self.index_teams_name[e[0]]
                    ):
                        return False
        return True

    def get_attribute_register(self) -> EncodingRegister:
        """Attribute documentation for TeamAllocation object.

        Returns: an EncodingRegister specifying the colors attribute.

        """
        dict_register = {
            "allocation": {
                "name": "allocation",
                "type": [TypeAttribute.LIST_INTEGER],
                "n": self.number_of_activity,
                "arity": self.number_of_teams,
                "low": 0,  # integer
                "up": self.number_of_teams - 1,  # integer
            }
        }
        return EncodingRegister(dict_register)

    def get_solution_type(self) -> type[Solution]:
        """Returns the class of a solution instance for ColoringProblem."""
        return TeamAllocationSolution

    def get_objective_register(self) -> ObjectiveRegister:
        """Specifies the default objective settings to be used with the evaluate function output."""
        dict_objective = {
            "nb_teams": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0),
            "nb_violations": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100.0
            ),
        }
        if self.do_add_cons:
            dict_objective["nb_violations_add_cons"] = ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100.0
            )
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def get_dummy_solution(self) -> TeamAllocationSolution:
        """Returns a dummy solution.

        A dummy feasible solution consists in giving one different color per vertices.
        Returns: A feasible and dummiest ColoringSolution

        """
        allocation = list(range(self.number_of_activity))
        allocation = [min(x, self.number_of_teams - 1) for x in allocation]
        solution = TeamAllocationSolution(self, allocation=allocation)
        return solution

    def count_color_constraints_violations(
        self, variable: TeamAllocationSolution
    ) -> int:
        nb_violation = 0
        if len(self.graph_activity.edges) > 0:
            if variable.allocation is None:
                raise ValueError("variable.allocation must not be None")
            for e in self.graph_activity.edges:
                if (
                    variable.allocation[self.index_activities_name[e[0]]]
                    == variable.allocation[self.index_activities_name[e[1]]]
                ):
                    nb_violation += 1
        return nb_violation

    def count_allowed_assignment_violations(
        self, variable: TeamAllocationSolution
    ) -> int:
        nb_violation = 0
        if len(self.graph_allocation.edges) > 0:
            if variable.allocation is None:
                raise ValueError("variable.allocation must not be None")
            for e in self.graph_allocation.edges:
                if e[0] in self.index_activities_name:
                    if (
                        variable.allocation[self.index_activities_name[e[0]]]
                        == self.index_teams_name[e[1]]
                    ):
                        nb_violation += 1
                if e[1] in self.index_activities_name:
                    if (
                        variable.allocation[self.index_activities_name[e[1]]]
                        == self.index_teams_name[e[0]]
                    ):
                        nb_violation += 1
        return nb_violation

    def evaluate_from_encoding(
        self, int_vector: list[int], encoding_name: str
    ) -> dict[str, float]:
        """Can be used in GA algorithm to build an object solution and evaluate from a int_vector representation.

        Args:
            int_vector: representing the colors vector of our problem
            encoding_name: name of the attribute in TeamAllocationSolution corresponding to the int_vector given.
             In our case, will only work for encoding_name="allocation"
        Returns: the evaluation of the (int_vector, encoding) object on the team allocation problem.

        """
        coloring_sol: TeamAllocationSolution
        if encoding_name == "allocation":
            coloring_sol = TeamAllocationSolution(problem=self, allocation=int_vector)
        else:
            raise ValueError("encoding_name can only be 'allocation'")
        objectives = self.evaluate(coloring_sol)
        return objectives

    def get_natural_explanation_unsat_constraints(
        self, variable: TeamAllocationSolution
    ) -> list[str]:
        """
        Return a list of strings describing which constraints are not fulfilled by the given solution.
        Args:
            variable (TeamAllocationSolution): solution object we want to "analyze"
        Returns: list[str]
        """
        return self.get_natural_explanation_unsat_colors(
            variable
        ) + self.get_natural_explanation_unsat_allowed_assignment(variable)

    def get_natural_explanation_unsat_colors(
        self, variable: TeamAllocationSolution
    ) -> list[str]:
        """
        Return a list of strings describing which coloring constraints are not fulfilled by the given solution.
        Args:
            variable (TeamAllocationSolution): solution object we want to "analyze"
        Returns: list[str]
        """
        list_str_description = []
        if len(self.graph_activity.edges) > 0:
            if variable.allocation is None:
                raise ValueError("variable.allocation must not be None")
            for e in self.graph_activity.edges:
                if (
                    variable.allocation[self.index_activities_name[e[0]]]
                    == variable.allocation[self.index_activities_name[e[1]]]
                ):
                    if variable.allocation[self.index_activities_name[e[0]]] is None:
                        continue
                    team = self.index_to_teams_name[
                        variable.allocation[self.index_activities_name[e[0]]]
                    ]
                    description = (
                        f"Same team ({team}, index={variable.allocation[self.index_activities_name[e[0]]]}) "
                        f"allocated to the incompatible activities ({e[0], e[1]})"
                    )
                    list_str_description.append(description)
            return list_str_description
        return []

    def get_natural_explanation_unsat_allowed_assignment(
        self, variable: TeamAllocationSolution
    ) -> list[str]:
        list_str_description = []
        if len(self.graph_allocation.edges) > 0:
            if variable.allocation is None:
                raise ValueError("variable.allocation must not be None")
            for e in self.graph_allocation.edges:
                if e[0] in self.index_activities_name:
                    if (
                        variable.allocation[self.index_activities_name[e[0]]]
                        == self.index_teams_name[e[1]]
                    ):
                        description = (
                            f"You allocated team {e[1]} (or index {self.index_teams_name[e[1]]}) "
                            f"to the forbidden activity {(e[0], self.index_activities_name[e[0]])}"
                        )
                        list_str_description.append(description)
                if e[1] in self.index_activities_name:
                    if (
                        variable.allocation[self.index_activities_name[e[1]]]
                        == self.index_teams_name[e[0]]
                    ):
                        description = (
                            f"You allocated team {e[0]} (or index {self.index_teams_name[e[0]]}) "
                            f"to the forbidden activity {(e[1], self.index_activities_name[e[1]])}"
                        )
                        list_str_description.append(description)
            return list_str_description
        return list_str_description

    def add_additional_constraint(
        self, allocation_additional_constraint: AllocationAdditionalConstraint
    ):
        self.allocation_additional_constraint = allocation_additional_constraint

    def remove_additional_constraint(self):
        self.allocation_additional_constraint = None


def transform_to_coloring_problem(
    team_allocation_problem: TeamAllocationProblem,
    add_clique_team_nodes: bool = True,
    add_constraint_color: bool = False,
) -> ColoringProblem:
    """
    Transform the list-coloring/team_allocation_problem into a classical coloring problem.
    1) We create a node for each team, linking to other original nodes this team can't be allocated to.
    2) We create a clique of nodes that are the "teams" node.
    :param team_allocation_problem: original problem to be transformed.
    :param add_constraint_color: use special structure in ColoringProblem to force the value of color of given nodes
    :param add_clique_team_nodes: use the transformation of list-coloring to classical graph coloring by adding
    artificial nodes for team and creating a clique from them.
    :return: ColoringProblem representing the same problem.
    """
    graph_coloring_nx = team_allocation_problem.graph_activity.to_networkx()
    graph_allocation_nx = team_allocation_problem.graph_allocation.to_networkx()
    teams_names = team_allocation_problem.teams_name
    activities_names = team_allocation_problem.activities_name
    merged_graph = nx.Graph()
    for activity in activities_names:
        merged_graph.add_node(
            activity,
            **team_allocation_problem.graph_activity.nodes_infos_dict[activity],
        )
    for team in teams_names:
        merged_graph.add_node(team, bipartite=0, team=True)
    for edge in graph_coloring_nx.edges():
        merged_graph.add_edge(edge[0], edge[1])
    for edge in graph_allocation_nx.edges():
        merged_graph.add_edge(edge[0], edge[1])
    team_in_graph = team_allocation_problem.graph_allocation.get_nodes_team()
    logger.debug(f"nb teams : {len(team_in_graph)}")
    i = 0
    constraint_coloring = None
    if add_constraint_color:
        color_constraint = {}
        for t in teams_names:
            color_constraint[t] = team_allocation_problem.index_teams_name[t]
        constraint_coloring = ColoringConstraints(color_constraint=color_constraint)
    if add_clique_team_nodes:
        for team1, team2 in product(team_in_graph, repeat=2):
            i += 1
            if team1 != team2:
                merged_graph.add_edge(team1, team2)
    logger.debug(f"end loop {i}")
    g = Graph(
        nodes=[(n, merged_graph.nodes[n]) for n in merged_graph.nodes()],
        edges=[(e[0], e[1], merged_graph.edges[e]) for e in merged_graph.edges],
        undirected=True,
        compute_predecessors=False,
    )
    return ColoringProblem(
        graph=g,
        subset_nodes=set(activities_names),
        constraints_coloring=constraint_coloring,
    )


class AggregateOperator(Enum):
    MAX_MINUS_MIN = 0
    MAX = 1
    MIN = 2
    MEAN = 3
    GINI = 4


class TeamAllocationProblemMultiobj(TeamAllocationProblem):
    def __init__(
        self,
        graph_activity: Graph = None,
        graph_allocation: GraphBipartite = None,
        allocation_additional_constraint: Optional[
            AllocationAdditionalConstraint
        ] = None,
        schedule_activity: Optional[dict[Hashable, tuple[int, int]]] = None,
        calendar_team: dict[Hashable, list[tuple[int, int]]] = None,
        activities_name: list[Hashable] = None,
        attributes_cumul_activities: Optional[list[str]] = None,
        objective_doc_cumul_activities: Optional[
            dict[str, tuple[ObjectiveDoc, AggregateOperator]]
        ] = None,
    ):
        super().__init__(
            graph_activity=graph_activity,
            graph_allocation=graph_allocation,
            allocation_additional_constraint=allocation_additional_constraint,
            schedule_activity=schedule_activity,
            calendar_team=calendar_team,
            activities_name=activities_name,
        )
        self.attributes_cumul_activities = attributes_cumul_activities
        self.attributes_of_activities: dict[str, dict[Hashable, float]] = {}
        for attr in self.attributes_cumul_activities:
            self.attributes_of_activities[attr] = {}
            for t in self.activities_name:
                self.attributes_of_activities[attr][t] = (
                    self.graph_activity.get_attr_node(t, attr)
                )
        self.attributes_cumul_activities = attributes_cumul_activities
        self.objective_doc_cumul_activities = objective_doc_cumul_activities
        if (
            self.attributes_cumul_activities is not None
            and self.objective_doc_cumul_activities is None
        ):
            self.objective_doc_cumul_activities = {
                attr: (
                    ObjectiveDoc(type=TypeObjective.PENALTY, default_weight=-1),
                    AggregateOperator.MAX_MINUS_MIN,
                )
                for attr in self.attributes_cumul_activities
            }

    def update_attributes_of_activities(self):
        for attr in self.attributes_cumul_activities:
            self.attributes_of_activities[attr] = {}
            for t in self.activities_name:
                self.attributes_of_activities[attr][t] = (
                    self.graph_activity.get_attr_node(t, attr)
                )

    def get_objective_register(self) -> ObjectiveRegister:
        """Specifies the default objective settings to be used with the evaluate function output."""
        o_register = super().get_objective_register()
        dict_objective = o_register.dict_objective_to_doc
        # dict_objective["nb_teams"].default_weight = -10000
        dict_objective["nb_teams"] = ObjectiveDoc(
            type=TypeObjective.OBJECTIVE, default_weight=-10000
        )
        if self.objective_doc_cumul_activities is not None:
            dict_objective.update(
                {
                    x: self.objective_doc_cumul_activities[x][0]
                    for x in self.objective_doc_cumul_activities
                }
            )
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def evaluate_cumul_nodes(
        self, variable: TeamAllocationSolution, attribute_on_node: str
    ):
        cumuls = np.zeros(shape=len(self.teams_name), dtype=int)
        for i in range(self.number_of_activity):
            act = self.index_to_activities_name[i]
            cumuls[variable.allocation[i]] += int(
                self.attributes_of_activities[attribute_on_node][act]
            )
        return cumuls

    def aggregate_cumuls(
        self, cumuls_array, aggregate_operator: AggregateOperator
    ) -> float:
        non_zeros = np.nonzero(cumuls_array)
        fit_ = 0
        if aggregate_operator == AggregateOperator.MEAN:
            fit_ = np.mean(cumuls_array[non_zeros])
        if aggregate_operator == AggregateOperator.MIN:
            fit_ = np.min(cumuls_array[non_zeros])
        if aggregate_operator == AggregateOperator.MAX:
            fit_ = np.max(cumuls_array[non_zeros])
        if aggregate_operator == AggregateOperator.MAX_MINUS_MIN:
            fit_ = np.max(cumuls_array[non_zeros]) - np.min(cumuls_array[non_zeros])
        return fit_

    def evaluate(self, variable: TeamAllocationSolution) -> dict[str, float]:
        fits = TeamAllocationProblem.evaluate(self, variable)
        cumuls = {
            attr: self.evaluate_cumul_nodes(variable, attr)
            for attr in self.attributes_cumul_activities
        }
        for attr in cumuls:
            fits[attr] = self.aggregate_cumuls(
                cumuls[attr], self.objective_doc_cumul_activities[attr][1]
            )
        return fits


def satisfy_additional_constraint(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    additional_constraint: AllocationAdditionalConstraint,
    partial_solution: bool = False,
):
    if additional_constraint.same_allocation is not None:
        b = satisfy_same_allocation(
            problem=problem,
            solution=solution,
            same_allocation=additional_constraint.same_allocation,
            partial_solution=partial_solution,
        )
        if not b:
            logger.info("Same allocation constraint violated")
            return False
    if additional_constraint.all_diff_allocation is not None:
        b = satisfy_all_diff(
            problem=problem,
            solution=solution,
            all_diffs=additional_constraint.all_diff_allocation,
            partial_solution=partial_solution,
        )
        if not b:
            logger.info("All diff allocation constraint violated")
            return False
    if additional_constraint.forced_allocation is not None:
        b = satisfy_forced_allocation(
            problem=problem,
            solution=solution,
            forced_allocation=additional_constraint.forced_allocation,
            partial_solution=partial_solution,
        )
        if not b:
            logger.info("Forced allocation constraint violated")
            return False
    if additional_constraint.forbidden_allocation is not None:
        b = satisfy_forbidden_allocation(
            problem=problem,
            solution=solution,
            forbidden_allocation=additional_constraint.forbidden_allocation,
            partial_solution=partial_solution,
        )
        if not b:
            logger.info("Forbidden allocation constraint violated")
            return False
    if additional_constraint.allowed_allocation is not None:
        b = satisfy_allowed_allocation(
            problem=problem,
            solution=solution,
            allowed_allocation=additional_constraint.allowed_allocation,
            partial_solution=partial_solution,
        )
        if not b:
            logger.info("Allowed allocation constraint violated")
            return False
    if additional_constraint.disjunction is not None:
        b = satisfy_disjunctions(
            problem=problem,
            solution=solution,
            disjunction=additional_constraint.disjunction,
            partial_solution=partial_solution,
        )
        if not b:
            logger.info("Disjunction allocation constraint violated")
            return False
    if additional_constraint.nb_max_teams is not None:
        b = satisfy_nb_teams(
            problem=problem,
            solution=solution,
            nb_teams_max=additional_constraint.nb_max_teams,
            partial_solution=partial_solution,
        )
        if not b:
            logger.info("Nb max teams constraint violated")
            return False
    return True


def satisfy_same_allocation(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    same_allocation: list[set[Hashable]],
    partial_solution: bool = False,
):
    for set_same_alloc in same_allocation:
        if not partial_solution:
            one_ac = next(iter(set_same_alloc))
            val = solution.allocation[problem.index_activities_name[one_ac]]
            # logger.debug([solution.allocation[problem.index_activities_name[x]] for x in set_same_alloc])
            if any(
                solution.allocation[problem.index_activities_name[x]] != val
                for x in set_same_alloc
            ):
                return False
        else:
            one_ac = next(
                (
                    s
                    for s in set_same_alloc
                    if solution.allocation[problem.index_activities_name[s]] is not None
                ),
                None,
            )
            if one_ac is None:
                continue
            else:
                val = solution.allocation[problem.index_activities_name[one_ac]]
                if any(
                    solution.allocation[problem.index_activities_name[x]] != val
                    for x in set_same_alloc
                    if solution.allocation[problem.index_activities_name[x]] is not None
                ):
                    return False
    return True


def satisfy_all_diff(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    all_diffs: list[set[Hashable]],
    partial_solution: bool = False,
):
    for all_diff in all_diffs:
        set_value = set()
        for t in all_diff:
            team = solution.allocation[problem.index_activities_name[t]]
            if team is None and partial_solution:
                continue
            if team in set_value:
                return False
            else:
                set_value.add(team)
    return True


def satisfy_forced_allocation(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    forced_allocation: dict[Hashable, Hashable],
    partial_solution: bool = False,
):
    for task in forced_allocation:
        team_index = solution.allocation[problem.index_activities_name[task]]
        if team_index is None and partial_solution:
            continue
        if team_index is None:
            return False
        team_name = problem.index_to_teams_name[team_index]
        if team_name != forced_allocation[task]:
            return False
    return True


def satisfy_forbidden_allocation(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    forbidden_allocation: dict[Hashable, set[Hashable]],
    partial_solution: bool = False,
):
    for task in forbidden_allocation:
        team_index = solution.allocation[problem.index_activities_name[task]]
        if team_index is None and partial_solution:
            continue
        if team_index is None:
            return False
        team_name = problem.index_to_teams_name[team_index]
        if team_name in forbidden_allocation[task]:
            return False
    return True


def satisfy_allowed_allocation(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    allowed_allocation: dict[Hashable, set[Hashable]],
    partial_solution: bool = False,
):
    for task in allowed_allocation:
        team_index = solution.allocation[problem.index_activities_name[task]]
        if team_index is None and partial_solution:
            continue
        if team_index is None:
            return False
        team_name = problem.index_to_teams_name[team_index]
        if team_name not in allowed_allocation[task]:
            return False
    return True


def satisfy_disjunctions(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    disjunction: list[list[tuple[Hashable, Hashable]]],
    partial_solution: bool = False,
):
    b = True
    for one_disjunction in disjunction:
        b = b and satisfy_disjunction(
            problem=problem,
            solution=solution,
            one_disjunction=one_disjunction,
            partial_solution=partial_solution,
        )
        if not b:
            return False
    return True


def satisfy_disjunction(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    one_disjunction: list[tuple[Hashable, Hashable]],
    partial_solution: bool = False,
):
    b = False
    for task, team in one_disjunction:
        team_index = solution.allocation[problem.index_activities_name[task]]
        if team_index is None:
            continue
        team_name = problem.index_to_teams_name[team_index]
        b |= team_name == team
        if b:
            return True
    return False


def satisfy_nb_teams(
    problem: TeamAllocationProblem,
    solution: TeamAllocationSolution,
    nb_teams_max: int,
    partial_solution: bool = False,
):
    set_teams = set()
    for i in range(problem.number_of_activity):
        v = solution.allocation[i]
        if v is not None:
            set_teams.add(v)
            if len(set_teams) > nb_teams_max:
                return False
    return True
