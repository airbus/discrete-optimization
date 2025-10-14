#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Hashable, Union

import matplotlib.patches as patches
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.do_problem import TypeObjective
from discrete_optimization.workforce.allocation.problem import (
    AggregateOperator,
    AllocationAdditionalConstraint,
    ObjectiveDoc,
    TeamAllocationProblem,
    TeamAllocationProblemMultiobj,
    TeamAllocationSolution,
)
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    TasksDescription,
    satisfy_detailed,
)

try:
    import plotly.graph_objects as go
except:
    pass

logger = logging.getLogger(__name__)

this_folder = os.path.dirname(os.path.abspath(__file__))


def overlap_interval(interval_1: tuple[int, int], interval_2: tuple[int, int]):
    return interval_1[0] < interval_2[1] and interval_2[0] < interval_1[1]


def compute_equivalent_teams_scheduling_problem(
    scheduling_problem: AllocSchedulingProblem,
) -> list[list[int]]:
    """
    Return a list of disjoint set of teams index, that can be considered as indistinguishable
    from a solution point of view. Example : in the pure coloring problem all the colors/team are equivalent
    In the team allocation problem, due to restricted compatible teams to task, the equivalent class are different
    Adaptation from the notebook/test_models.ipynb
    """
    allowed_teams = [
        {
            scheduling_problem.teams_to_index[team]
            for team in scheduling_problem.available_team_for_activity.get(
                t, scheduling_problem.team_names
            )
        }
        for t in scheduling_problem.tasks_list
    ]
    equiv_teams = {}
    all_teams = set(scheduling_problem.index_to_team)
    # for each team t, compute the intersection of teams that can do the same task as t.
    for t in scheduling_problem.index_to_team:
        equiv_teams[t] = all_teams.intersection(*[s for s in allowed_teams if t in s])
        equiv_teams[t] = equiv_teams[t].intersection(
            {
                s
                for s in equiv_teams[t]
                if scheduling_problem.calendar_team[scheduling_problem.index_to_team[s]]
                == scheduling_problem.calendar_team[scheduling_problem.index_to_team[t]]
            }
        )
    for t in equiv_teams:
        for et in list(equiv_teams[t]):
            if t not in equiv_teams[et]:
                equiv_teams[t].remove(et)
    all_symm_teams = set([frozenset(equiv_teams[t]) for t in equiv_teams])
    symm_groups = [list(sorted(group)) for group in all_symm_teams]
    return symm_groups


def compute_changes_between_solution(
    solution_a: AllocSchedulingSolution,
    solution_b: AllocSchedulingSolution,
    problem_a: AllocSchedulingProblem = None,
    problem_b: AllocSchedulingProblem = None,
):
    if problem_a is None:
        problem_a = solution_a.problem
    if problem_b is None:
        problem_b = solution_b.problem
    if problem_a == problem_b:
        return compute_changes_between_solution_same_pb(
            solution_a, solution_b, problem=problem_a
        )

    common_activities = set(problem_a.tasks_list).intersection(problem_b.tasks_list)
    activities = [a for a in common_activities]
    alloc_a = np.array(
        [solution_a.allocation[problem_a.tasks_to_index[a]] for a in activities]
    )
    alloc_b = np.array(
        [solution_b.allocation[problem_b.tasks_to_index[a]] for a in activities]
    )
    schedule_a = np.array(
        [solution_a.schedule[problem_a.tasks_to_index[a], :] for a in activities]
    )
    schedule_b = np.array(
        [solution_b.schedule[problem_b.tasks_to_index[a], :] for a in activities]
    )

    reallocated = (alloc_a != alloc_b).nonzero()
    shifted = (schedule_a[:, 0] != schedule_b[:, 0]).nonzero()
    nb_shifted = shifted[0].shape[0]
    delta = schedule_a - schedule_b
    abs_delta = np.abs(delta)
    if shifted[0].shape[0] > 0:
        mean_shift = np.mean(abs_delta[shifted, 0])
        sum_shift = np.sum(abs_delta[shifted, 0])
        max_shift = np.max(abs_delta[shifted, 0])
    else:
        mean_shift = 0
        max_shift = 0
        sum_shift = 0

    # abs_delta = np.abs(delta)
    details = {
        "allocs": (alloc_a, alloc_b),
        "schedules": (schedule_a, schedule_b),
        "reallocated_index": reallocated[0],
        "reallocated_tasks": [activities[i] for i in reallocated[0]],
        "nb_reallocated": reallocated[0].shape[0],
        "shifted_index": shifted[0],
        "shifted_tasks": [activities[i] for i in shifted[0]],
        "shifts": delta[shifted, 0],
        "nb_shift": nb_shifted,
        "mean_shift": mean_shift,
        "sum_shift": sum_shift,
        "max_shift": max_shift,
    }
    return details


def compute_changes_between_solution_same_pb(
    solution_a: AllocSchedulingSolution,
    solution_b: AllocSchedulingSolution,
    problem: AllocSchedulingProblem = None,
):
    if problem is None:
        problem = solution_a.problem
    reallocated = (solution_a.allocation != solution_b.allocation).nonzero()
    shifted = (solution_a.schedule[:, 0] != solution_b.schedule[:, 0]).nonzero()
    nb_reallocated = reallocated[0].shape[0]
    nb_shifted = shifted[0].shape[0]
    delta = solution_a.schedule - solution_b.schedule
    abs_delta = np.abs(delta)
    if shifted[0].shape[0] > 0:
        mean_shift = np.mean(abs_delta[shifted, 0])
        max_shift = np.max(abs_delta[shifted, 0])
        sum_shift = np.sum(abs_delta[shifted, 0])
    else:
        mean_shift = 0
        max_shift = 0
        sum_shift = 0
    details = {
        "allocs": (solution_a.allocation, solution_b.allocation),
        "schedules": (solution_a.schedule, solution_b.schedule),
        "reallocated_index": reallocated[0],
        "reallocated_tasks": [problem.tasks_list[i] for i in reallocated[0]],
        "nb_reallocated": reallocated[0].shape[0],
        "shifted_index": shifted[0],
        "shifted_tasks": [problem.tasks_list[i] for i in shifted[0]],
        "shifts": delta[shifted, 0],
        "nb_shift": nb_shifted,
        "sum_shift": sum_shift,
        "mean_shift": mean_shift,
        "max_shift": max_shift,
    }
    return details


def plot_schedule_comparison(
    base_solution: AllocSchedulingSolution,
    updated_solution: AllocSchedulingSolution,
    problem: AllocSchedulingProblem,
):
    """
    Nice visu to compare 2 schedules.
    """
    base_allocation = base_solution.allocation
    base_schedule = base_solution.schedule

    updated_allocation = updated_solution.allocation
    updated_schedule = updated_solution.schedule

    # Create a color map for teams
    cmap = plt.get_cmap(
        "tab20", problem.number_teams
    )  # Using tab20 to assign unique colors for up to 20 teams
    colors = [cmap(i) for i in range(problem.number_teams)]
    fig, ax = plt.subplots(figsize=(12, 8))
    all_teams = set(base_allocation).union(updated_allocation)
    sorted_all_teams = sorted(list(all_teams))
    sorted_all_teams = [int(x) for x in sorted_all_teams]
    if -1 in sorted_all_teams:
        sorted_all_teams = sorted_all_teams[1:] + ["Unset"]
    team_to_index = {sorted_all_teams[i]: i for i in range(len(sorted_all_teams))}
    # Iterate over each task
    for i in range(len(base_allocation)):
        team_base = int(base_allocation[i])
        team_updated = int(updated_allocation[i])
        if team_base == -1:
            team_base = "Unset"
        if team_updated == -1:
            team_updated = "Unset"

        start_base, end_base = base_schedule[i]
        start_updated, end_updated = updated_schedule[i]
        # Plot updated solution as a rectangle
        rect_updated = patches.Rectangle(
            (start_updated, team_to_index[team_updated] - 0.4),
            end_updated - start_updated,
            0.8,
            facecolor=colors[team_to_index[team_updated]],
            edgecolor="black",
            lw=2,
        )
        # linestyle='--')
        ax.add_patch(rect_updated)

        # Draw an arrow from the end of the base task to the start of the updated task
        if (int(start_updated), team_updated) != (int(start_base), team_base):
            # Plot base solution as a rectangle
            rect_base = patches.Rectangle(
                (start_base, team_to_index[team_base] - 0.4),
                end_base - start_base,
                0.8,
                facecolor=colors[team_to_index[team_base]],
                edgecolor="black",
                lw=2,
                alpha=0.1,
                label=f"Team {team_base}" if i == 0 else "",
            )
            ax.add_patch(rect_base)
            ax.annotate(
                "",
                xy=((start_updated + end_updated) / 2, team_to_index[team_updated]),
                xytext=((start_base + end_base) / 2, team_to_index[team_base]),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
            )

    # Customizing the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Teams")
    # min_alloc = min(base_allocation.min(), updated_allocation.min())
    # max_alloc = max(base_allocation.max(), updated_allocation.max())
    ax.set_yticks(range(len(sorted_all_teams)))  # np.arange(min_alloc, max_alloc+1, 1))
    ax.set_yticklabels(
        [f"Team {sorted_all_teams[i]}" for i in range(len(sorted_all_teams))]
    )
    ax.set_title("Schedule Comparison (Base vs Updated)")

    # Set limits to ensure the rectangles fit well
    ax.set_xlim(
        min(base_schedule[:, 0].min(), updated_schedule[:, 0].min()) - 1,
        max(base_schedule[:, 1].max(), updated_schedule[:, 1].max()) + 1,
    )
    ax.set_ylim(-0.5, len(sorted_all_teams) - 0.5)

    # Show the grid for clarity
    ax.grid(True, which="both", axis="x", linestyle="--", lw=0.5)
    return fig


def plotly_schedule_comparison(
    base_solution: AllocSchedulingSolution,
    updated_solution: AllocSchedulingSolution,
    problem: AllocSchedulingProblem,
    index_team_to_other_index: dict[int, int] = None,
    display: bool = False,
    additional_info: dict[Hashable, dict[str, Any]] = None,
    use_color_scale: bool = True,
    use_color_map_per_task: bool = False,
    color_map_per_task: dict[int, Any] = None,
    opacity_map_per_task: dict[int, float] = None,
    show_all_changes: bool = True,
    show_change: dict[int, bool] = None,
    plot_team_breaks: bool = False,
    plot_xticks: bool = True,
    plot_text: bool = True,
    title="Scheduling Comparison (Base vs Updated)",
):
    """
    Nice visu to compare 2 schedules.
    """
    if opacity_map_per_task is None:
        opacity_map_per_task = {}
    if index_team_to_other_index is None:
        index_team_to_other_index = {i: i for i in problem.index_to_team}
    base_allocation = base_solution.allocation
    max_time = np.max(base_solution.schedule[:, 1])
    base_schedule = base_solution.schedule + problem.horizon_start_shift // 60
    updated_schedule = updated_solution.schedule + problem.horizon_start_shift // 60
    min_ = np.min(base_schedule)
    base_schedule = base_schedule  # -min_
    updated_allocation = list(updated_solution.allocation)
    for i in range(len(base_allocation)):
        if base_allocation[i] is None:
            base_allocation[i] = -1
        if updated_allocation[i] is None:
            updated_allocation[i] = -1

    import plotly.colors as pc

    colormap = "Viridis"
    value = 0.5  # Example: Normalized float value between 0 and 1
    # Extract the color for the given value using the colormap

    fig = go.Figure()
    all_teams = set(base_allocation).union(updated_allocation)
    sorted_all_teams = sorted(list(all_teams))
    sorted_all_teams = [int(x) for x in sorted_all_teams]
    indexing = deepcopy(index_team_to_other_index)
    indexing[-1] = -1
    sorted_all_teams = sorted([indexing[x] for x in sorted_all_teams])
    if -1 in sorted_all_teams:
        sorted_all_teams = sorted_all_teams[1:] + ["Unset"]
    team_to_index = {sorted_all_teams[i]: i for i in range(len(sorted_all_teams))}
    # Iterate over each task
    if use_color_scale:
        colors_ = pc.sample_colorscale(
            colormap, np.linspace(0, 1, len(sorted_all_teams))
        )
        # `value` is between 0 and 1
    else:
        colors_ = ["green" for i in range(len(sorted_all_teams))]
    for i in range(len(base_allocation)):
        team_base = indexing[int(base_allocation[i])]
        team_updated = indexing[int(updated_allocation[i])]
        if team_base == -1:
            team_base = "Unset"
        if team_updated == -1:
            team_updated = "Unset"
        start_base, end_base = base_schedule[i]
        start_updated, end_updated = updated_schedule[i]
        start_base_ts = datetime.datetime.fromtimestamp(start_base * 60)
        end_base_ts = datetime.datetime.fromtimestamp(end_base * 60)
        start_updated_ts = datetime.datetime.fromtimestamp(start_updated * 60)
        end_updated_ts = datetime.datetime.fromtimestamp(end_updated * 60)
        start_updated_ts = np.datetime64(int(start_updated * 60), "s")
        end_updated_ts = np.datetime64(int(end_updated * 60), "s")

        # Plot updated solution as a rectangle
        hover = {
            "activity": problem.index_to_task[i],
            "activity_index": i,
            "index_team": team_updated,
            "start": start_updated_ts,
            "end": end_updated_ts,
        }
        if additional_info is not None:
            if problem.index_to_task[i] in additional_info:
                for key in additional_info[problem.index_to_task[i]]:
                    hover[key] = additional_info[problem.index_to_task[i]][key]
        hover_template = ""
        for key in hover:
            hover_template += f"<b>{key}:</b> {hover[key]}<br>"
        if "ACTIVITY TYPE" in hover:
            hover_template = ""
            for key in [
                "ACTIVITY TYPE",
                "start",
                "end",
                "START LOCATION NAME",
                "DESTINATION LOCATION NAME",
                "Ressourcenname",
            ]:
                hover_template += f"<b>{key}:</b> {hover[key]}<br>"

        fig.add_trace(
            go.Bar(
                x=[(end_updated - start_updated)],
                y=[team_to_index[team_updated]],
                base=start_updated,
                orientation="h",
                name=f"{i}",
                text=i if plot_text else "",
                textangle=0,
                # textfont=dict(size=15, color="black"),
                # textposition="outside",
                marker=dict(
                    color=colors_[team_to_index[team_updated]]
                    if not use_color_map_per_task
                    else color_map_per_task.get(
                        i, "green"
                    ),  # Normalize team index for color mapping
                    # colorscale="rainbow",  # Use the custom color scale
                    showscale=False,  # No color scale bar
                    line=dict(color="black", width=1.5),
                    opacity=opacity_map_per_task.get(i, 0.8),
                ),
                # marker_color=[float(team_to_index[team_updated] / len(team_to_index))],
                # marker_colorscale="rainbow",
                hovertemplate=hover_template,
            )
        )
        # Draw an arrow from the end of the base task to the start of the updated task
        if (int(start_updated), team_updated) != (int(start_base), team_base):
            if not show_all_changes:
                if not show_change.get(i, True):
                    continue
            # Plot base solution as a rectangle
            hover = {
                "activity": problem.index_to_task[i],
                "activity_index": i,
                "index_team": team_base,
            }
            hover_template = "<b>Base schedule</b> <br>"
            for key in hover:
                hover_template += f"<b>{key}:</b> {hover[key]}<br>"
            fig.add_trace(
                go.Bar(
                    x=[(end_base - start_base)],
                    y=[team_to_index[team_base]],
                    base=start_base,
                    orientation="h",
                    name=f"{i}",
                    # marker_color=[float(team_to_index[team_base]/len(team_to_index))],
                    # marker_colorscale="rainbow",
                    marker=dict(
                        color=colors_[team_to_index[team_base]]
                        if not use_color_map_per_task
                        else color_map_per_task.get(i, "green"),
                        # colorscale="rainbow",
                        opacity=0.3,
                    ),
                    hovertemplate=hover_template,
                )
            )
            # fig.add_annotation(
            #     ax=(start_base+end_base)//2,
            #     ay=team_to_index[team_base],
            #     x=(start_updated+end_updated)//2,
            #     y=team_to_index[team_updated],
            #     xref="x",
            #     yref="y",
            #     axref="x",
            #     ayref="y",
            #     showarrow=True,
            #     arrowhead=1,
            #     # arrowhead=6,  # Larger arrowhead
            #     arrowsize=1,  # Increase size of the arrow
            #     # arrowwidth=3,  # Thicker arrow line
            #     arrowcolor="darkred",  # High-contrast arrow color
            #     opacity=0.8  # Fully opaque arrow
            # )
            fig.add_trace(
                go.Scatter(
                    x=[
                        (start_base + end_base) // 2,
                        (start_updated + end_updated) // 2,
                    ],
                    y=[team_to_index[team_base], team_to_index[team_updated]],
                    mode="lines+markers",
                    line=dict(color="darkred", width=1, dash="dot"),  # Dotted line
                    marker=dict(
                        size=[0, 7], color="darkred"
                    ),  # , symbol="arrow-bar-up")
                )
            )

    if plot_team_breaks:
        for team in team_to_index:
            if team != "Unset":
                team_ = problem.team_names[team]
                slots = problem.compute_unavailability_calendar(team_)
                for slot in slots:
                    if slot[0] < max_time:
                        right_side = min(slot[1], max_time + 30)
                        logger.debug("Slot ", slot)
                        fig.add_trace(
                            go.Bar(
                                x=[(right_side - slot[0])],
                                y=[team_to_index[team]],
                                dy=0.2,
                                base=slot[0] + problem.horizon_start_shift // 60,
                                orientation="h",
                                # name=f"Break",
                                # marker_color=[float(team_to_index[team_base]/len(team_to_index))],
                                # marker_colorscale="rainbow",
                                marker=dict(color="black", opacity=0.7),
                                hovertemplate=f"<b>Break {team}</b> <br>",
                                width=0.2,
                            )
                        )

    min_date = int(np.min(base_schedule))
    max_date = int(np.max(base_schedule))
    tickvals = range(min_date, max_date, (max_date - min_date) // 10)
    # ticktext = [datetime.datetime.fromtimestamp(ts*60).strftime('%Y-%m-%d %H:%M:%S') for ts in tickvals]
    ticktext = [str(np.datetime64(ts * 60, "s")) for ts in tickvals]

    fig.update_layout(
        title=title,
        xaxis_title="Timeline",
        xaxis=dict(
            title="Time",
            type="linear",  # Keep the scale as numeric
            tickvals=list(tickvals),  # Numeric positions for ticks
            ticktext=ticktext if plot_xticks else None,  # Custom labels for the ticks
            showticklabels=plot_xticks,
            # tickformat="%Y-%m-%d %H:%M:%S",  # Format tick labels as datetime
        ),
        yaxis=dict(
            title="Team",
            tickmode="array",
            tickvals=list(range(len(sorted_all_teams))),
            ticktext=[f"{sorted_all_teams[i]}" for i in range(len(sorted_all_teams))],
        ),
        barmode="overlay",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    # Show line on each teams
    fig.update_yaxes(showgrid=True)
    # Vertical line stuff
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
    if display:
        fig.show()
    return fig


def estimate_nb_resource_needed(problem: AllocSchedulingProblem):
    horizon = problem.horizon
    estimated_workload = np.zeros(horizon)
    lb_ub = problem.get_all_lb_ub()
    for i in problem.index_to_task:
        dur = problem.tasks_data[problem.index_to_task[i]].duration_task
        rng = lb_ub[i][3] - lb_ub[i][0]
        estimated_workload[lb_ub[i][0] : lb_ub[i][3]] += dur / rng
    return estimated_workload


def template_violated_constraint(
    satisfy_detailed_output: Union[tuple, dict], problem: AllocSchedulingProblem
):
    if isinstance(satisfy_detailed_output, dict):
        if "tag" in satisfy_detailed_output:
            tag = satisfy_detailed_output["tag"]
            if tag == "is_not_done":
                task = satisfy_detailed_output["task_index"]
                return "Task" + f" {task} " + "is not done"
            if tag == "early":
                task = satisfy_detailed_output["task_index"]
                if "start" in satisfy_detailed_output:
                    return (
                        f"Task {task} is early, starts at {satisfy_detailed_output['start']} "
                        f"but expected at least {satisfy_detailed_output['expected']}"
                    )
                if "end" in satisfy_detailed_output:
                    return (
                        f"Task {task} is early, ends at {satisfy_detailed_output['end']} "
                        f"but expected at least {satisfy_detailed_output['expected']}"
                    )
            if tag == "late":
                task = satisfy_detailed_output["task_index"]
                if "start" in satisfy_detailed_output:
                    return (
                        f"{task} is late, starts at {satisfy_detailed_output['start']} "
                        f"but expected at most {satisfy_detailed_output['expected']}"
                    )
                if "end" in satisfy_detailed_output:
                    return (
                        f"{task} is late, ends at {satisfy_detailed_output['end']} "
                        f"but expected at most {satisfy_detailed_output['expected']}"
                    )
    if isinstance(satisfy_detailed_output, tuple):
        if satisfy_detailed_output[0] == "precedence":
            succ = satisfy_detailed_output[4]
            pred = satisfy_detailed_output[3]
            return f"{pred} should be done before {succ}, it's not the case : end of {pred} is {satisfy_detailed_output[-2]}, and start of {succ} is {satisfy_detailed_output[-1]}"
        if satisfy_detailed_output[0] == "same_allocation":
            return f"Same allocation constraint is not satisfied for tasks {satisfy_detailed_output[-1]}"
        if satisfy_detailed_output[0] == "available-team":
            activity = satisfy_detailed_output[1]
            teams = {
                problem.teams_to_index[x]
                for x in problem.available_team_for_activity[activity]
            }
            return (
                f"Task {satisfy_detailed_output[3]} is allocated to {satisfy_detailed_output[4]} "
                f"while only teams {teams} can do this task"
            )
        if satisfy_detailed_output[0] == "no-overlap":
            return (
                f"Task {satisfy_detailed_output[1]} and {satisfy_detailed_output[2]} "
                f"overlaps on team {satisfy_detailed_output[3]}"
            )


def natural_explanation_unsat(
    detailed_output: list[Union[tuple, dict]], problem: AllocSchedulingProblem
) -> list[str]:
    return [template_violated_constraint(x, problem=problem) for x in detailed_output]


def natural_explanation_unsat_from_sol(solution: AllocSchedulingSolution) -> list[str]:
    return natural_explanation_unsat(
        detailed_output=satisfy_detailed(problem=solution.problem, solution=solution),
        problem=solution.problem,
    )


def compute_precedence_graph(problem: AllocSchedulingProblem) -> nx.DiGraph:
    graph = nx.DiGraph()
    compatible_teams = problem.compatible_teams_index_all_activity()
    for index_t in problem.index_to_task:
        graph.add_node(
            index_t,
            activity_name=problem.index_to_task[index_t],
            compatible_teams=compatible_teams[index_t],
        )
    for task in problem.precedence_constraints:
        index = problem.tasks_to_index[task]
        for succ in problem.precedence_constraints[task]:
            index_succ = problem.tasks_to_index[succ]
            graph.add_edge(index, index_succ)
    return graph


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
    return d


def get_working_time_teams(problem: AllocSchedulingProblem) -> dict:
    work_time = {}
    lb_ub = problem.get_all_lb_ub()
    true_ub = max([x[-1] for x in lb_ub])
    for team in problem.calendar_team:
        cumul = 0
        for slots in problem.calendar_team[team]:
            if slots[0] <= true_ub:
                cumul += max(0, min(slots[1], true_ub) - slots[0])
        work_time[team] = cumul
    return work_time


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


def build_allocation_problem_from_scheduling(
    problem: AllocSchedulingProblem,
    solution: AllocSchedulingSolution = None,
    problem_alloc: TeamAllocationProblem = None,
    multiobjective: bool = True,
) -> TeamAllocationProblem:
    calendars_team = problem.calendar_team
    # min_d = min([calendars_team[team][0][0] for team in calendars_team if len(calendars_team[team]) >= 1])
    # max_d = max([calendars_team[team][-1][1] for team in calendars_team if len(calendars_team[team]) >= 1])
    calendars_array = {
        team: np.zeros((problem.horizon + 20)) for team in calendars_team
    }
    for team in calendars_team:
        for min_, max_ in calendars_team[team]:
            calendars_array[team][min_ : min(max_, calendars_array[team].shape[0])] = 1
    if solution is not None:
        starts = solution.schedule[:, 0]
        ends = solution.schedule[:, 1]
    else:
        starts = np.array([problem.original_start[t] for t in problem.tasks_list])
        ends = np.array([problem.original_end[t] for t in problem.tasks_list])

    shift = problem.horizon_start_shift

    # UPDATE GRAPH ACTIVITY
    if problem_alloc is None:
        if not multiobjective:
            problem_alloc = TeamAllocationProblem(
                allocation_additional_constraint=AllocationAdditionalConstraint(
                    same_allocation=problem.same_allocation,
                    allowed_allocation=problem.available_team_for_activity,
                ),
                calendar_team=calendars_team,
                schedule_activity={
                    t: (
                        starts[problem.tasks_to_index[t]],
                        ends[problem.tasks_to_index[t]],
                    )
                    for t in problem.original_start
                },
                activities_name=problem.tasks_list,
            )
        else:
            problem_alloc = TeamAllocationProblemMultiobj(
                allocation_additional_constraint=AllocationAdditionalConstraint(
                    same_allocation=problem.same_allocation,
                    allowed_allocation=problem.available_team_for_activity,
                ),
                attributes_cumul_activities=["duration"],
                objective_doc_cumul_activities={
                    "duration": (
                        ObjectiveDoc(type=TypeObjective.PENALTY, default_weight=-1),
                        AggregateOperator.MAX_MINUS_MIN,
                    )
                },
                calendar_team=calendars_team,
                schedule_activity={
                    t: (
                        starts[problem.tasks_to_index[t]],
                        ends[problem.tasks_to_index[t]],
                    )
                    for t in problem.original_start
                },
                activities_name=problem.tasks_list,
            )
    return problem_alloc


def build_scheduling_problem_from_allocation(
    problem: TeamAllocationProblem, horizon_start_shift: int = 0
) -> AllocSchedulingProblem:
    d = {}
    if problem.allocation_additional_constraint is not None:
        d["same_allocation"] = problem.allocation_additional_constraint.same_allocation

    return AllocSchedulingProblem(
        team_names=problem.teams_name,
        calendar_team=problem.calendar_team,
        horizon=10000,
        horizon_start_shift=horizon_start_shift,
        tasks_list=problem.activities_name,
        tasks_data={
            t: TasksDescription(
                duration_task=problem.graph_activity.nodes_infos_dict[t]["duration"]
            )
            for t in problem.activities_name
        },
        same_allocation=problem.allocation_additional_constraint.same_allocation,
        precedence_constraints={},
        available_team_for_activity={},
        start_window={},
        end_window={},
        original_start={
            t: problem.graph_activity.nodes_infos_dict[t]["start"]
            for t in problem.activities_name
        },
        original_end={
            t: problem.graph_activity.nodes_infos_dict[t]["end"]
            for t in problem.activities_name
        },
    )


def alloc_solution_to_alloc_sched_solution(
    problem: AllocSchedulingProblem, solution: TeamAllocationSolution
):
    alloc_problem: TeamAllocationProblem = solution.problem
    new_alloc = -np.ones(len(solution.allocation), dtype=int)
    schedule = np.zeros((len(solution.allocation), 2), dtype=int)
    for task in problem.tasks_list:
        ind_alloc = alloc_problem.index_activities_name[task]
        alloc = solution.allocation[ind_alloc]
        if alloc is not None and alloc >= 0:
            new_alloc[problem.tasks_to_index[task]] = problem.teams_to_index[
                alloc_problem.teams_name[alloc]
            ]
        schedule[problem.tasks_to_index[task], 0] = problem.original_start[task]
        schedule[problem.tasks_to_index[task], 1] = problem.original_end[task]
    return AllocSchedulingSolution(
        problem=problem, schedule=schedule, allocation=new_alloc
    )


def binary_calendar(list_available: list[tuple[int, int]], horizon: int):
    calendars_array = np.zeros((horizon + 20))
    for min_, max_ in list_available:
        calendars_array[min_ : min(max_, calendars_array.shape[0])] = 1
    return calendars_array


def get_availability_slots(calendar_matrix: np.ndarray):
    availability_slots = []
    start_slot = None
    n = len(calendar_matrix)

    for i in range(n):
        if (
            calendar_matrix[i] == 1 and start_slot is None
        ):  # Start of an available block
            start_slot = i
        elif (
            calendar_matrix[i] == 0 and start_slot is not None
        ):  # End of an available block
            availability_slots.append((start_slot, i - 1))
            start_slot = None
    if start_slot is not None:  # If the array ends with an available block
        availability_slots.append((start_slot, n - 1))

    return availability_slots
