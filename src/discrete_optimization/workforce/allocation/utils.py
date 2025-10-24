#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Hashable, Optional, Union

import networkx as nx
import pandas as pd

from discrete_optimization.generic_tools.graph_api import Graph, from_networkx
from discrete_optimization.workforce.allocation.problem import (
    AllocationAdditionalConstraint,
    GraphBipartite,
    TeamAllocationProblem,
    TeamAllocationProblemMultiobj,
    TeamAllocationSolution,
)

try:
    import plotly.express as px
except ImportError as e:
    pass


def subgraph_teams(
    graph_allocation: GraphBipartite,
    nb_teams_keep: int = 3,
    subset_teams_keep: Optional[set[Hashable]] = None,
) -> GraphBipartite:
    if subset_teams_keep is not None:
        sub = subset_teams_keep
    else:
        nb_team = min(nb_teams_keep, len(graph_allocation.nodes_team))
        sub = set(graph_allocation.nodes_team_list.teams_name[:nb_team])
    nodes_activity = graph_allocation.nodes_activity
    new_list_nodes = [
        n for n in graph_allocation.nodes if n[0] in nodes_activity or n[0] in sub
    ]
    new_nodes_names = set([n[0] for n in new_list_nodes])
    return GraphBipartite(
        nodes=new_list_nodes,
        edges=[
            x
            for x in graph_allocation.edges
            if x[0] in new_nodes_names and x[1] in new_nodes_names
        ],
        nodes_activity=nodes_activity,
        nodes_team=sub,
        undirected=graph_allocation.undirected,
        compute_predecessors=False,
    )


def subgraph_activities(graph: Graph, subset_tasks: set[Hashable]) -> Graph:
    g = graph.graph_nx
    new_g = nx.subgraph(g, subset_tasks)
    return from_networkx(new_g, undirected=graph.undirected, compute_predecessors=False)


def subgraph_bipartite_activities(
    graph_allocation: GraphBipartite, subset_tasks: set[Hashable]
) -> Graph:
    # g = graph.graph_nx
    # ss = subset_tasks
    # new_g = nx.subgraph(g, subset_tasks.union(graph.nodes_team))
    # return from_networkx(new_g, undirected=graph.undirected,
    #                      compute_predecessors=False)
    nodes_activity = [n for n in graph_allocation.nodes_activity if n in subset_tasks]
    nodes_team = graph_allocation.nodes_team
    new_list_nodes = [
        n
        for n in graph_allocation.nodes
        if n[0] in nodes_activity or n[0] in nodes_team
    ]
    new_nodes_names = set([n[0] for n in new_list_nodes])
    return GraphBipartite(
        nodes=new_list_nodes,
        edges=[
            x
            for x in graph_allocation.edges
            if x[0] in new_nodes_names and x[1] in new_nodes_names
        ],
        nodes_activity=nodes_activity,
        nodes_team=nodes_team,
        undirected=graph_allocation.undirected,
        compute_predecessors=False,
    )


def allocation_additional_constraint_subset_tasks(
    allocation_additional_constraint: AllocationAdditionalConstraint,
    subset_tasks: set[Hashable],
):
    args = {
        "same_allocation": allocation_additional_constraint.same_allocation,
        "all_diff_allocation": allocation_additional_constraint.all_diff_allocation,
        "forced_allocation": allocation_additional_constraint.forced_allocation,
        "forbidden_allocation": allocation_additional_constraint.forbidden_allocation,
        "disjunction": allocation_additional_constraint.disjunction,
        "nb_max_teams": allocation_additional_constraint.nb_max_teams,
        "allowed_allocation": allocation_additional_constraint.allowed_allocation,
    }
    if allocation_additional_constraint.forced_allocation is not None:
        new_forced_allocation = {
            act: allocation_additional_constraint.forced_allocation[act]
            for act in allocation_additional_constraint.forced_allocation
            if act in subset_tasks
        }
        args["forced_allocation"] = new_forced_allocation
    if allocation_additional_constraint.forbidden_allocation is not None:
        new_forbidden_allocation = {
            act: allocation_additional_constraint.forbidden_allocation[act]
            for act in allocation_additional_constraint.forbidden_allocation
            if act in subset_tasks
        }
        args["forbidden_allocation"] = new_forbidden_allocation
    if allocation_additional_constraint.disjunction is not None:
        new_disjunction = [
            [(ac, team) for ac, team in d if ac in subset_tasks]
            for d in allocation_additional_constraint.disjunction
        ]
        args["disjunction"] = new_disjunction
    if allocation_additional_constraint.all_diff_allocation is not None:
        new_all_diff_allocation = [
            {ac for ac in ad if ad in subset_tasks}
            for ad in allocation_additional_constraint.all_diff_allocation
        ]
        new_all_diff_allocation = [x for x in new_all_diff_allocation if len(x) > 0]
        args["all_diff_allocation"] = new_all_diff_allocation
    if allocation_additional_constraint.same_allocation is not None:
        new_same_allocation = [
            {ac for ac in ad if ad in subset_tasks}
            for ad in allocation_additional_constraint.same_allocation
        ]
        new_same_allocation = [x for x in new_same_allocation if len(x) > 0]
        args["same_allocation"] = new_same_allocation
    if allocation_additional_constraint.allowed_allocation is not None:
        new_allowed_allocation = {
            act: allocation_additional_constraint.allowed_allocation[act]
            for act in allocation_additional_constraint.allowed_allocation
            if act in subset_tasks
        }
        args["allowed_allocation"] = new_allowed_allocation

    return AllocationAdditionalConstraint(**args)


def create_subproblem_allocation(
    problem: TeamAllocationProblem, subset_tasks: set[Hashable]
):
    g = subgraph_activities(graph=problem.graph_activity, subset_tasks=subset_tasks)
    g_alloc = subgraph_bipartite_activities(
        graph_allocation=problem.graph_allocation, subset_tasks=subset_tasks
    )
    allocation_constraint = allocation_additional_constraint_subset_tasks(
        allocation_additional_constraint=problem.allocation_additional_constraint,
        subset_tasks=subset_tasks,
    )
    return TeamAllocationProblem(
        graph_activity=g,
        graph_allocation=g_alloc,
        allocation_additional_constraint=allocation_constraint,
    )


def additional_constraint_subset_teams(
    allocation_additional_constraint: AllocationAdditionalConstraint,
    subset_teams_keep: set[Hashable],
):
    if allocation_additional_constraint is None:
        allocation_additional_constraint = AllocationAdditionalConstraint()
    args = {
        "same_allocation": allocation_additional_constraint.same_allocation,
        "all_diff_allocation": allocation_additional_constraint.all_diff_allocation,
        "forced_allocation": allocation_additional_constraint.forced_allocation,
        "forbidden_allocation": allocation_additional_constraint.forbidden_allocation,
        "disjunction": allocation_additional_constraint.disjunction,
        "nb_max_teams": allocation_additional_constraint.nb_max_teams,
        "allowed_allocation": allocation_additional_constraint.allowed_allocation,
    }
    if allocation_additional_constraint.forced_allocation is not None:
        new_forced_allocation = {
            act: allocation_additional_constraint.forced_allocation[act]
            for act in allocation_additional_constraint.forced_allocation
            if allocation_additional_constraint.forced_allocation[act]
            in subset_teams_keep
        }
        args["forced_allocation"] = new_forced_allocation
    if allocation_additional_constraint.forbidden_allocation is not None:
        new_forbidden_allocation = {
            act: set(
                [
                    s
                    for s in allocation_additional_constraint.forbidden_allocation[act]
                    if s in subset_teams_keep
                ]
            )
            for act in allocation_additional_constraint.forbidden_allocation
        }
        args["forbidden_allocation"] = new_forbidden_allocation
    if allocation_additional_constraint.disjunction is not None:
        new_disjunction = [
            [(ac, team) for ac, team in d if team in subset_teams_keep]
            for d in allocation_additional_constraint.disjunction
        ]
        args["disjunction"] = new_disjunction
    if allocation_additional_constraint.nb_max_teams is not None:
        args["nb_max_teams"] = min(args["nb_max_teams"], len(subset_teams_keep))
    if allocation_additional_constraint.allowed_allocation is not None:
        new_allowed_allocation = {
            act: {
                t
                for t in allocation_additional_constraint.allowed_allocation[act]
                if t in subset_teams_keep
            }
            for act in allocation_additional_constraint.allowed_allocation
        }
        args["allowed_allocation"] = new_allowed_allocation
    return AllocationAdditionalConstraint(**args)


def cut_number_of_team(
    team_allocation: TeamAllocationProblem,
    nb_teams_keep: int = 3,
    subset_teams_keep: Optional[set[Hashable]] = None,
):
    if subset_teams_keep is not None:
        sub = subset_teams_keep
    else:
        nb_team = min(nb_teams_keep, team_allocation.number_of_teams)
        sub = set(team_allocation.teams_name[:nb_team])
    new_graph_allocation = subgraph_teams(
        graph_allocation=team_allocation.graph_allocation, subset_teams_keep=sub
    )
    if isinstance(team_allocation, TeamAllocationProblemMultiobj):
        return TeamAllocationProblemMultiobj(
            graph_activity=team_allocation.graph_activity,
            graph_allocation=new_graph_allocation,
            allocation_additional_constraint=additional_constraint_subset_teams(
                team_allocation.allocation_additional_constraint, subset_teams_keep=sub
            ),
            attributes_cumul_activities=team_allocation.attributes_cumul_activities,
            activities_name=team_allocation.activities_name,
            schedule_activity=team_allocation.schedule,
            calendar_team={t: team_allocation.calendar_team[t] for t in sub},
            objective_doc_cumul_activities=team_allocation.objective_doc_cumul_activities,
        )
    return TeamAllocationProblem(
        graph_activity=team_allocation.graph_activity,
        graph_allocation=new_graph_allocation,
        allocation_additional_constraint=additional_constraint_subset_teams(
            team_allocation.allocation_additional_constraint, subset_teams_keep=sub
        ),
        schedule_activity=team_allocation.schedule,
        calendar_team={t: team_allocation.calendar_team[t] for t in sub},
    )


def compute_equivalent_teams(
    team_allocation_problem: TeamAllocationProblem,
) -> list[list[int]]:
    """
    Return a list of disjoint set of teams index, that can be considered as indistinguishable
    from a solution point of view. Example : in the pure coloring problem all the colors/team are equivalent
    In the team allocation problem, due to restricted compatible teams to task, the equivalent class are different
    Adaptation from the notebook/test_models.ipynb
    """
    allowed_teams = [
        set(indexes)
        for indexes in team_allocation_problem.compute_allowed_team_index_all_task()
    ]
    equiv_teams = {}
    all_teams = set(team_allocation_problem.index_to_teams_name)
    # for each team t, compute the intersection of teams that can do the same task as t.
    for t in team_allocation_problem.index_to_teams_name:
        equiv_teams[t] = all_teams.intersection(*[s for s in allowed_teams if t in s])
    for t in equiv_teams:
        for et in list(equiv_teams[t]):
            if t not in equiv_teams[et]:
                equiv_teams[t].remove(et)
    all_symm_teams = set([frozenset(equiv_teams[t]) for t in equiv_teams])
    symm_groups = [list(sorted(group)) for group in all_symm_teams]
    return symm_groups


def compute_overlapping_activities_on_start_time(
    team_allocation_problem: TeamAllocationProblem, activity: Hashable
) -> list[Hashable]:
    """
    Look at overlapping task at starting time of task
    """
    start = team_allocation_problem.graph_activity.nodes_infos_dict[activity]["start"]
    return [
        n
        for n in team_allocation_problem.graph_activity.nodes_infos_dict
        if team_allocation_problem.graph_activity.nodes_infos_dict[n]["start"] <= start
        and team_allocation_problem.graph_activity.nodes_infos_dict[n]["end"] > start
    ]


def compute_overlapping_activities_on_end_time(
    team_allocation_problem: TeamAllocationProblem, activity: Hashable
) -> list[Hashable]:
    """
    Look at overlapping task at starting time of task
    """
    end = team_allocation_problem.graph_activity.nodes_infos_dict[activity]["end"]
    return [
        n
        for n in team_allocation_problem.graph_activity.nodes_infos_dict
        if team_allocation_problem.graph_activity.nodes_infos_dict[n]["start"]
        < end
        <= team_allocation_problem.graph_activity.nodes_infos_dict[n]["end"]
    ]


def compute_active_activities_on_time(
    team_allocation_problem: TeamAllocationProblem,
    time: Union[int, pd.Timestamp],
    side="left",
) -> list[Hashable]:
    if side == "left":
        return [
            n
            for n in team_allocation_problem.graph_activity.nodes_infos_dict
            if team_allocation_problem.graph_activity.nodes_infos_dict[n]["start"]
            < time
            <= team_allocation_problem.graph_activity.nodes_infos_dict[n]["end"]
        ]
    if side == "right":
        return [
            n
            for n in team_allocation_problem.graph_activity.nodes_infos_dict
            if team_allocation_problem.graph_activity.nodes_infos_dict[n]["start"]
            <= time
            < team_allocation_problem.graph_activity.nodes_infos_dict[n]["end"]
        ]


def compute_all_overlapping(
    team_allocation_problem: TeamAllocationProblem,
) -> set[frozenset]:
    set_overlaps = set()
    for task in team_allocation_problem.activities_name:
        set_overlaps.add(
            frozenset(
                compute_overlapping_activities_on_start_time(
                    team_allocation_problem, task
                )
            )
        )
        set_overlaps.add(
            frozenset(
                compute_overlapping_activities_on_end_time(
                    team_allocation_problem, task
                )
            )
        )
    return set_overlaps


def compute_changes_between_solution_alloc(
    solution_a: TeamAllocationSolution,
    solution_b: TeamAllocationSolution,
    problem_a: TeamAllocationProblem = None,
    problem_b: TeamAllocationProblem = None,
):
    if problem_a is None:
        problem_a = solution_a.problem
    if problem_b is None:
        problem_b = solution_b.problem

    common_activities = set(problem_a.activities_name).intersection(
        problem_b.activities_name
    )
    alloc_a = {
        a: solution_a.allocation[problem_a.index_activities_name[a]]
        for a in common_activities
    }
    alloc_b = {
        a: solution_b.allocation[problem_b.index_activities_name[a]]
        for a in common_activities
    }
    changes = []
    for a in common_activities:
        if problem_b.index_to_teams_name.get(
            alloc_b[a], None
        ) != problem_a.index_to_teams_name.get(alloc_a[a], None):
            changes.append(
                (
                    a,
                    alloc_a[a],
                    alloc_b[a],
                    problem_a.teams_name[alloc_a[a]]
                    if alloc_a[a] is not None
                    else None,
                    problem_b.teams_name[alloc_b[a]]
                    if alloc_b[a] is not None
                    else None,
                    problem_a.index_activities_name[a],
                    problem_b.index_activities_name[a],
                )
            )
    # nb_changes = len(changes)
    return changes


def plot_allocation_solution(
    problem: TeamAllocationProblem,
    sol: TeamAllocationSolution,
    index_team_to_other_index: dict[int, int] = None,
    use_color_map: bool = False,
    category: bool = True,
    display: bool = True,
    plot_breaks: bool = False,
    color_break: str = None,
    ignore_unallocated: bool = False,
    name_unallocated: str = "Unset",
    text: Optional[str] = None,
    title: Optional[str] = None,
    ref_date: pd.Timestamp = None,
    showlegend=True,
):
    df = []
    if ref_date is None:
        ref_date = pd.Timestamp(year=2025, month=7, day=1)
    if index_team_to_other_index is None:
        index_team_to_other_index = {i: i for i in problem.index_to_teams_name}
    schedule_per_team: dict[Hashable, list[tuple]] = {}
    for i_task in range(len(sol.allocation)):
        if sol.allocation[i_task] is not None:
            team = problem.index_to_teams_name[sol.allocation[i_task]]
            activity = problem.index_to_activities_name[i_task]
            if team not in schedule_per_team:
                schedule_per_team[team] = []
            df += [
                {
                    "team": f"Team {index_team_to_other_index[sol.allocation[i_task]]}",
                    "team_name": problem.teams_name[sol.allocation[i_task]],
                    "activity": activity,
                    "activity_index": i_task,
                    "index_team": index_team_to_other_index[sol.allocation[i_task]],
                    "height": 20,
                    "start": ref_date
                    + pd.to_timedelta(
                        problem.graph_activity.nodes_infos_dict[activity]["start"],
                        unit="m",
                    ),
                    "end": ref_date
                    + pd.to_timedelta(
                        problem.graph_activity.nodes_infos_dict[activity]["end"],
                        unit="m",
                    ),
                }
            ]
        else:
            if not ignore_unallocated:
                activity = problem.index_to_activities_name[i_task]
                df += [
                    {
                        "team": f"Team {name_unallocated}",
                        "team_name": f"Team {name_unallocated}",
                        "activity": activity,
                        "activity_index": i_task,
                        "index_team": name_unallocated,
                        "height": 20,
                        "start": ref_date
                        + pd.to_timedelta(
                            problem.graph_activity.nodes_infos_dict[activity]["start"],
                            unit="m",
                        ),
                        "end": ref_date
                        + pd.to_timedelta(
                            problem.graph_activity.nodes_infos_dict[activity]["end"],
                            unit="m",
                        ),
                    }
                ]
    if plot_breaks:
        for team in problem.calendar_team:
            index = index_team_to_other_index[problem.index_teams_name[team]]
            for i_slot in range(len(problem.calendar_team[team]) - 1):
                df += [
                    {
                        "team": f"Break",
                        "team_name": team,
                        "activity": f"break_{index}_{i_slot}",
                        "activity_index": "nan",
                        "index_team": index,
                        "height": 10,
                        "start": ref_date
                        + pd.to_timedelta(
                            problem.calendar_team[team][i_slot][1], unit="m"
                        ),
                        "end": ref_date
                        + pd.to_timedelta(
                            problem.calendar_team[team][i_slot + 1][0], unit="m"
                        ),
                    }
                ]
    hover_data = {
        "activity": True,
        "team": True,
        "activity_index": True,
        "index_team": True,
    }
    df = pd.DataFrame(df)
    if len(df) == 0:
        return
    if not use_color_map:
        fig = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y="index_team",
            color="team",
            text=text,
            hover_data=hover_data,
            # category_orders={"index_team": df["index_team"]},
            title=title,
        )
    else:
        color_mapping = {x: "rgba(0, 128, 0, 0.6)" for x in df["team"]}
        color_mapping[f"Team {name_unallocated}"] = "rgba(255, 0, 0, 0.1)"
        color_mapping["Break"] = "rgba(0, 0, 0, 1)"
        if color_break is not None:
            color_mapping["Break"] = color_break
        fig = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y="index_team",
            color="team",
            color_discrete_map=color_mapping,
            text=text,
            hover_data=hover_data,
            # category_orders={"index_team": df["index_team"]},
            title=title,
        )
    modify_graph(fig)
    # Update the layout to reorder the legend entries alphabetically
    if category:
        # df = pd.DataFrame(df)
        cat = sorted([x for x in df["index_team"] if isinstance(x, int)]) + [
            name_unallocated
        ]
        fig.update_yaxes(  # type="category",
            type="category",
            categoryorder="array",
            categoryarray=cat,
            title_text="Team",
            showgrid=True,
        )
        fig.update_legends()
    if not showlegend:
        fig.update_layout(showlegend=False)
    if display:
        fig.show()
    else:
        return fig


def modify_graph(fig):
    # vertical line in graph that follows cursor
    fig.update_xaxes(
        showgrid=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="gray",
        spikedash="solid",
        spikethickness=0.5,
        showline=True,
    )
    # make borders around boxes # rgb(0, 48, 107)
    fig.update_traces(marker_line_color="gray", marker_line_width=0.5, opacity=1)
    # rename y axes title
    fig.update_yaxes(title_text="Team")
