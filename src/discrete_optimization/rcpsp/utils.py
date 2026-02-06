#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import (
    annotations,  # make annotations be considered as string by default
)

import logging
from collections.abc import Hashable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pp
from shapely.geometry import Polygon

from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.plot_utils import (
    get_cmap,
    get_cmap_with_nb_colors,
)

# from discrete_optimization.rcpsp.problem_preemptive import PreemptiveRcpspSolution

if TYPE_CHECKING:  # avoid circular imports due to annotations
    from discrete_optimization.rcpsp.problem import RcpspProblem
    from discrete_optimization.rcpsp.solution import RcpspSolution

logger = logging.getLogger(__name__)


def compute_resource_consumption(
    rcpsp_problem: RcpspProblem,
    rcpsp_sol: RcpspSolution,
    list_resources: Optional[list[str]] = None,
    future_view: bool = True,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
    modes_extended.insert(0, 1)
    modes_extended.append(1)
    modes_dict = rcpsp_problem.build_mode_dict(rcpsp_sol.rcpsp_modes)
    last_activity = rcpsp_problem.sink_task
    makespan = rcpsp_sol.rcpsp_schedule[last_activity]["end_time"]
    if list_resources is None:
        list_resources = rcpsp_problem.resources_list
    consumptions = np.zeros((len(list_resources), makespan + 1), dtype=np.int_)
    for act_id in rcpsp_sol.rcpsp_schedule:
        for ir in range(len(list_resources)):
            use_ir = rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                list_resources[ir], 0
            )
            if future_view:
                consumptions[
                    ir,
                    rcpsp_sol.rcpsp_schedule[act_id]["start_time"]
                    + 1 : rcpsp_sol.rcpsp_schedule[act_id]["end_time"] + 1,
                ] += use_ir
            else:
                consumptions[
                    ir,
                    rcpsp_sol.rcpsp_schedule[act_id][
                        "start_time"
                    ] : rcpsp_sol.rcpsp_schedule[act_id]["end_time"],
                ] += use_ir

    return consumptions, np.arange(0, makespan + 1, 1, dtype=np.int_)


def compute_nice_resource_consumption(
    rcpsp_problem: RcpspProblem,
    rcpsp_sol: RcpspSolution,
    list_resources: Optional[list[str]] = None,
) -> tuple[dict[int, npt.NDArray[np.int_]], dict[int, npt.NDArray[np.int_]]]:
    if list_resources is None:
        list_resources = rcpsp_problem.resources_list
    c_future, times = compute_resource_consumption(
        rcpsp_problem, rcpsp_sol, list_resources=list_resources, future_view=True
    )
    c_past, times = compute_resource_consumption(
        rcpsp_problem, rcpsp_sol, list_resources=list_resources, future_view=False
    )
    merged_times: dict[int, list[int]] = {i: [] for i in range(len(list_resources))}
    merged_cons: dict[int, list[int]] = {i: [] for i in range(len(list_resources))}
    for r in range(len(list_resources)):
        for index_t in range(len(times)):
            merged_times[r] += [times[index_t], times[index_t]]
            merged_cons[r] += [c_future[r, index_t], c_past[r, index_t]]
    return (
        {k: np.array(v) for k, v in merged_times.items()},
        {k: np.array(v) for k, v in merged_cons.items()},
    )


def plot_ressource_view(
    rcpsp_problem: RcpspProblem,
    rcpsp_sol: RcpspSolution,
    list_resource: Optional[list[str]] = None,
    title_figure: str = "",
    x_lim: Optional[list[int]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[npt.NDArray[np.object_]] = None,
) -> plt.Figure:
    modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
    modes_extended.insert(0, 1)
    modes_extended.append(1)
    with_calendar = rcpsp_problem.is_varying_resource()
    modes_dict = rcpsp_problem.build_mode_dict(rcpsp_sol.rcpsp_modes)
    if list_resource is None:
        list_resource = rcpsp_problem.resources_list
    if ax is None:
        fig, ax = plt.subplots(nrows=len(list_resource), figsize=(10, 5), sharex=True)
        if len(list_resource) == 1:
            ax = [ax]
        fig.suptitle(title_figure)
    polygons_ax: dict[int, list[Polygon]] = {i: [] for i in range(len(list_resource))}
    labels_ax: dict[int, list[Hashable]] = {i: [] for i in range(len(list_resource))}
    sorted_activities = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: rcpsp_sol.rcpsp_schedule[x]["start_time"],
    )
    for j in sorted_activities:
        time_start = rcpsp_sol.rcpsp_schedule[j]["start_time"]
        time_end = rcpsp_sol.rcpsp_schedule[j]["end_time"]
        for i in range(len(list_resource)):
            cons = rcpsp_problem.mode_details[j][modes_dict[j]].get(list_resource[i], 0)
            if cons == 0:
                continue
            bound: int = int(rcpsp_problem.get_max_resource_capacity(list_resource[i]))
            for k in range(0, bound):
                polygon = Polygon(
                    [
                        (time_start, k),
                        (time_end, k),
                        (time_end, k + cons),
                        (time_start, k + cons),
                        (time_start, k),
                    ]
                )
                areas = [p.intersection(polygon).area for p in polygons_ax[i]]
                if len(areas) == 0 or max(areas) == 0:
                    polygons_ax[i].append(polygon)
                    labels_ax[i].append(j)
                    break
    for i in range(len(list_resource)):
        patches = []
        for polygon in polygons_ax[i]:
            x, y = polygon.exterior.xy
            ax[i].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=polygon.exterior.coords))
        p = PatchCollection(patches, cmap=get_cmap("Blues"), alpha=0.4)
        ax[i].add_collection(p)
    merged_times, merged_cons = compute_nice_resource_consumption(
        rcpsp_problem, rcpsp_sol, list_resources=list_resource
    )
    for i in range(len(list_resource)):
        ax[i].plot(
            merged_times[i],
            merged_cons[i],
            color="r",
            linewidth=2,
            label="Consumption " + str(list_resource[i]),
            zorder=1,
        )
        if not with_calendar:
            ax[i].axhline(
                y=rcpsp_problem.resources[list_resource[i]],
                linestyle="--",
                label="Limit : " + str(list_resource[i]),
                zorder=0,
            )
        else:
            ax[i].plot(
                merged_times[i],
                [rcpsp_problem.resources[list_resource[i]][m] for m in merged_times[i]],  # type: ignore
                linestyle="--",
                label="Limit : " + str(list_resource[i]),
                zorder=0,
            )
        ax[i].legend(fontsize=5)
        lims = ax[i].get_xlim()
        if x_lim is None:
            ax[i].set_xlim([lims[0], 1.0 * lims[1]])
        else:
            ax[i].set_xlim(x_lim)
    ax[-1].set_xlabel("Timestep")
    return fig


def plot_task_gantt(
    rcpsp_problem: RcpspProblem,
    rcpsp_sol: RcpspSolution,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    x_lim: Optional[list[int]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.set_title("Gantt Task")
    if title is None:
        ax.set_title("Gantt Task")
    else:
        ax.set_title(title)
    tasks = rcpsp_problem.tasks_list
    nb_task = len(tasks)
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x)
        + rcpsp_problem.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_problem.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_start[0])
    patches = []
    for j in range(nb_task):
        nb_colors = len(tasks) // 2
        colors = get_cmap_with_nb_colors("hsv", nb_colors)
        box = [
            (j - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
            (j - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]),
            (j + 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]),
            (j + 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
            (j - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
        ]
        polygon = Polygon([(b[1], b[0]) for b in box])
        x, y = polygon.exterior.xy
        ax.plot(x, y, zorder=-1, color="b")
        patches.append(
            pp(xy=polygon.exterior.coords, facecolor=colors((j - 1) % nb_colors))
        )

    p = PatchCollection(
        patches,
        match_original=True,
        alpha=0.4,
    )
    ax.add_collection(p)
    if x_lim is None:
        ax.set_xlim((min_time, max_time))
    else:
        ax.set_xlim(x_lim)
    ax.set_ylim((-0.5, nb_task))
    ax.set_yticks(range(nb_task))
    ax.set_yticklabels(
        tuple([str(tasks[j]) for j in range(nb_task)]), fontdict={"size": 5}
    )
    ax.set_ylabel("Task number")
    ax.set_xlabel("Timestep")
    return fig


def compute_schedule_per_resource_individual(
    rcpsp_problem: RcpspProblem,
    rcpsp_sol: RcpspSolution,
    resource_types_to_consider: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    modes = rcpsp_problem.build_mode_dict(rcpsp_sol.rcpsp_modes)
    if resource_types_to_consider is None:
        resources = rcpsp_problem.resources_list
    else:
        resources = resource_types_to_consider
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x)
        + rcpsp_problem.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_problem.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    with_calendar = rcpsp_problem.is_varying_resource()

    array_ressource_usage: dict[str, dict[str, Any]] = {
        resources[i]: {
            "activity": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_problem.get_max_resource_capacity(resources[i]),
                )
            ),
            "binary_activity": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_problem.get_max_resource_capacity(resources[i]),
                )
            ),
            "total_activity": np.zeros(
                rcpsp_problem.get_max_resource_capacity(resources[i])
            ),
            "activity_last_n_hours": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_problem.get_max_resource_capacity(resources[i]),
                )
            ),
            "boxes_time": [],
        }
        for i in range(len(resources))
    }
    total_time = max_time - min_time + 1
    nhour = int(min(8, total_time / 2 - 1))
    index_to_time = {i: min_time + i for i in range(max_time - min_time + 1)}
    time_to_index = {index_to_time[i]: i for i in index_to_time}

    for activity in sorted_task_by_start:
        mode = modes[activity]
        start_time = rcpsp_sol.rcpsp_schedule[activity]["start_time"]
        end_time = rcpsp_sol.rcpsp_schedule[activity]["end_time"]
        if end_time == start_time:
            continue
        resources_needed = {
            r: rcpsp_problem.mode_details[activity][mode].get(r, 0) for r in resources
        }
        for r in resources_needed:
            if r not in array_ressource_usage:
                continue
            rneeded = resources_needed[r]
            if not with_calendar:
                range_interest = range(array_ressource_usage[r]["activity"].shape[1])
            else:
                range_interest = range(
                    rcpsp_problem.resources[r][time_to_index[start_time]]  # type: ignore
                )
            while rneeded > 0:
                availables_people_r = [
                    i
                    for i in range_interest
                    if array_ressource_usage[r]["activity"][
                        time_to_index[start_time], i
                    ]
                    == 0
                ]
                logger.debug(f"{len(availables_people_r)} people available : ")
                if len(availables_people_r) > 0:
                    resource = min(
                        availables_people_r,
                        key=lambda x: array_ressource_usage[r]["total_activity"][x],
                    )
                    # greedy choice,
                    # the one who worked the less until now.
                    array_ressource_usage[r]["activity"][
                        time_to_index[start_time] : time_to_index[end_time], resource
                    ] = activity
                    array_ressource_usage[r]["binary_activity"][
                        time_to_index[start_time] : time_to_index[end_time], resource
                    ] = 1
                    array_ressource_usage[r]["total_activity"][resource] += (
                        end_time - start_time
                    )
                    array_ressource_usage[r]["activity_last_n_hours"][:, resource] = (
                        np.convolve(
                            array_ressource_usage[r]["binary_activity"][:, resource],
                            np.array([1] * nhour + [0] + [0] * nhour),
                            mode="same",
                        )
                    )
                    array_ressource_usage[r]["boxes_time"] += [
                        [
                            (resource - 0.25, start_time + 0.01, activity),
                            (resource - 0.25, end_time - 0.01, activity),
                            (resource + 0.25, end_time - 0.01, activity),
                            (resource + 0.25, start_time + 0.01, activity),
                            (resource - 0.25, start_time + 0.01, activity),
                        ]
                    ]
                    # for plot purposes.
                    rneeded -= 1
                else:
                    logger.debug(f"r_needed {rneeded}")
                    logger.debug(f"Ressource needed : {resources_needed}")
                    logger.debug(f"ressource : {r}")
                    logger.debug(f"activity : {activity}")
                    logger.warning("Problem, can't build schedule")
                    logger.debug(array_ressource_usage[r]["activity"])
                    rneeded = 0

    return array_ressource_usage


def plot_resource_individual_gantt(
    rcpsp_problem: RcpspProblem,
    rcpsp_sol: RcpspSolution,
    resource_types_to_consider: Optional[list[str]] = None,
    title_figure: str = "",
    x_lim: Optional[list[int]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[npt.NDArray[np.object_]] = None,
    current_t: Optional[int] = None,
) -> plt.Figure:
    array_ressource_usage = compute_schedule_per_resource_individual(
        rcpsp_problem, rcpsp_sol, resource_types_to_consider=resource_types_to_consider
    )
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x)
        + rcpsp_problem.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_problem.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    resources_list = list(array_ressource_usage.keys())
    if fig is None or ax is None:
        fig, ax_ = plt.subplots(len(array_ressource_usage), figsize=(10, 5))
        fig.suptitle(title_figure)
        if len(array_ressource_usage) == 1:
            ax = np.array([ax_])
        else:
            ax = ax_
    if ax is None:  # for mypy
        raise RuntimeError("ax cannot be None at this point")

    for i in range(len(resources_list)):
        patches = []
        nb_colors = len(sorted_task_by_start) // 2
        colors = get_cmap_with_nb_colors("hsv", nb_colors)
        for boxe in array_ressource_usage[resources_list[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            x, y = polygon.exterior.xy
            ax[i].plot(x, y, zorder=-1, color="b")
            patches.append(
                pp(
                    xy=polygon.exterior.coords,
                    facecolor=colors((activity - 1) % nb_colors),
                )
            )
        p = PatchCollection(
            patches,
            match_original=True,
            alpha=0.4,
        )
        ax[i].add_collection(p)
        ax[i].set_title(resources_list[i])
        if x_lim is None:
            ax[i].set_xlim((min_time, max_time))
        else:
            ax[i].set_xlim(x_lim)
        try:
            ax[i].set_ylim((-0.5, rcpsp_problem.resources[resources_list[i]]))
            ax[i].set_yticks(range(rcpsp_problem.resources[resources_list[i]]))  # type: ignore
            ax[i].set_yticklabels(
                tuple([j for j in range(rcpsp_problem.resources[resources_list[i]])]),  # type: ignore
                fontdict={"size": 7},
            )
        except:
            m = rcpsp_problem.get_max_resource_capacity(resources_list[i])
            ax[i].set_ylim((-0.5, m))
            ax[i].set_yticks(range(m))
            ax[i].set_yticklabels(tuple([j for j in range(m)]), fontdict={"size": 7})

        ax[i].grid(True)
        if current_t is not None:
            ax[i].axvline(x=current_t, label="pyplot vertical line", color="r", ls="--")
    ax[-1].set_xlabel("Timestep")
    return fig


def kendall_tau_similarity(rcpsp_sols: tuple[RcpspSolution, RcpspSolution]) -> float:
    sol1 = rcpsp_sols[0]
    sol2 = rcpsp_sols[1]

    perm1 = sol1.generate_permutation_from_schedule()
    perm2 = sol2.generate_permutation_from_schedule()

    ktd, p_value = scipy.stats.kendalltau(perm1, perm2)
    return ktd


def intersect(i1: Sequence[int], i2: Sequence[int]) -> Optional[list[int]]:
    if i2[0] >= i1[1] or i1[0] >= i2[1]:
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]


def all_diff_start_time(
    rcpsp_sols: tuple[RcpspSolution, RcpspSolution],
) -> dict[Hashable, int]:
    sol1 = rcpsp_sols[0]
    sol2 = rcpsp_sols[1]
    return {
        act_id: (
            sol1.rcpsp_schedule[act_id]["start_time"]
            - sol2.rcpsp_schedule[act_id]["start_time"]
        )
        for act_id in sol1.rcpsp_schedule
    }


def compute_graph_rcpsp(rcpsp_problem: RcpspProblem) -> Graph:
    nodes = [
        (
            n,
            {
                str(mode): rcpsp_problem.mode_details[n][mode]["duration"]
                for mode in rcpsp_problem.mode_details[n]
            },
        )
        for n in rcpsp_problem.tasks_list
    ]
    edges = []
    for n in rcpsp_problem.successors:
        for succ in rcpsp_problem.successors[n]:
            dict_transition: dict[str, int] = {
                str(mode): rcpsp_problem.mode_details[n][mode]["duration"]
                for mode in rcpsp_problem.mode_details[n]
            }
            min_duration = min(dict_transition.values())
            max_duration = max(dict_transition.values())
            dict_transition["min_duration"] = min_duration
            dict_transition["max_duration"] = max_duration
            dict_transition["minus_min_duration"] = -min_duration
            dict_transition["minus_max_duration"] = -max_duration
            dict_transition["link"] = 1
            edges += [(n, succ, dict_transition)]
    return Graph(nodes, edges, False)


def create_fake_tasks(rcpsp_problem: RcpspProblem) -> list[dict[str, int]]:
    if not rcpsp_problem.is_varying_resource():
        return []
    else:
        ressources_arrays = {
            r: np.array(rcpsp_problem.resources[r])
            for r in rcpsp_problem.resources_list
        }
        max_capacity = {r: np.max(ressources_arrays[r]) for r in ressources_arrays}
        fake_tasks: list[dict[str, int]] = []
        for r in ressources_arrays:
            delta = ressources_arrays[r][:-1] - ressources_arrays[r][1:]
            index_non_zero = np.nonzero(delta)[0]
            if ressources_arrays[r][0] < max_capacity[r]:
                consume = {
                    r: int(max_capacity[r] - ressources_arrays[r][0]),
                    "duration": int(index_non_zero[0] + 1),
                    "start": 0,
                }
                fake_tasks += [consume]
            for j in range(len(index_non_zero) - 1):
                ind = index_non_zero[j]
                value = ressources_arrays[r][ind + 1]
                if value != max_capacity[r]:
                    consume = {
                        r: int(max_capacity[r] - value),
                        "duration": int(index_non_zero[j + 1] - ind),
                        "start": int(ind + 1),
                    }
                    fake_tasks += [consume]
        return fake_tasks


def get_max_time_solution(
    solution: Union[PreemptiveRcpspSolution, RcpspSolution],
) -> int:
    if isinstance(solution, PreemptiveRcpspSolution):
        max_time = max(
            [solution.rcpsp_schedule[x]["ends"][-1] for x in solution.rcpsp_schedule]
        )
        return max_time
    else:
        max_time = max(
            [solution.rcpsp_schedule[x]["end_time"] for x in solution.rcpsp_schedule]
        )
        return max_time


def get_tasks_ending_between_two_times(
    solution: Union[PreemptiveRcpspSolution, RcpspSolution], time_1: int, time_2: int
) -> list[Hashable]:
    if isinstance(solution, PreemptiveRcpspSolution):
        return [
            x
            for x in solution.rcpsp_schedule
            if time_1 <= solution.rcpsp_schedule[x]["ends"][-1] <= time_2
        ]
    else:
        return [
            x
            for x in solution.rcpsp_schedule
            if time_1 <= solution.rcpsp_schedule[x]["end_time"] <= time_2
        ]


def get_start_bounds_from_additional_constraint(
    rcpsp_problem: RcpspProblem, activity: Hashable
) -> tuple[int, int]:
    assert activity in rcpsp_problem.index_task
    lb = 0
    ub = rcpsp_problem.horizon
    if rcpsp_problem.includes_special_constraint():
        if (
            rcpsp_problem.special_constraints.start_times is not None
            and activity in rcpsp_problem.special_constraints.start_times
            and rcpsp_problem.special_constraints.start_times[activity] is not None
        ):
            lb = ub = rcpsp_problem.special_constraints.start_times[activity]
        else:
            if activity in rcpsp_problem.special_constraints.start_times_window:
                lbs, ubs = rcpsp_problem.special_constraints.start_times_window[
                    activity
                ]
                if lbs is not None:
                    lb = lbs
                if ubs is not None:
                    ub = ubs
            if activity in rcpsp_problem.special_constraints.end_times_window:
                lbs, ubs = rcpsp_problem.special_constraints.end_times_window[activity]
                if lbs is not None:
                    max_duration = max(
                        [
                            rcpsp_problem.mode_details[activity][m]["duration"]
                            for m in rcpsp_problem.mode_details[activity]
                        ]
                    )
                    lb = max(lb, lbs - max_duration)
                if ubs is not None:
                    min_duration = min(
                        [
                            rcpsp_problem.mode_details[activity][m]["duration"]
                            for m in rcpsp_problem.mode_details[activity]
                        ]
                    )
                    ub = min(ub, ubs - min_duration)
    if ub < 0:
        logger.debug(f"ub<0, {ub}")
    return int(lb), int(ub)


def get_end_bounds_from_additional_constraint(
    rcpsp_problem: RcpspProblem, activity: Hashable
) -> tuple[int, int]:
    assert activity in rcpsp_problem.index_task
    lb = 0
    ub = rcpsp_problem.horizon
    if rcpsp_problem.includes_special_constraint():
        if (
            rcpsp_problem.special_constraints.end_times is not None
            and activity in rcpsp_problem.special_constraints.end_times
            and rcpsp_problem.special_constraints.end_times[activity] is not None
        ):
            lb = ub = rcpsp_problem.special_constraints.end_times[activity]
        else:
            if activity in rcpsp_problem.special_constraints.end_times_window:
                lbs, ubs = rcpsp_problem.special_constraints.end_times_window[activity]
                if lbs is not None:
                    lb = lbs
                if ubs is not None:
                    ub = ubs
            if activity in rcpsp_problem.special_constraints.start_times_window:
                lbs, ubs = rcpsp_problem.special_constraints.start_times_window[
                    activity
                ]
                if lbs is not None:
                    min_duration = min(
                        [
                            rcpsp_problem.mode_details[activity][m]["duration"]
                            for m in rcpsp_problem.mode_details[activity]
                        ]
                    )
                    lb = max(lb, lbs + min_duration)
                if ubs is not None:
                    max_duration = max(
                        [
                            rcpsp_problem.mode_details[activity][m]["duration"]
                            for m in rcpsp_problem.mode_details[activity]
                        ]
                    )
                    ub = min(ub, ubs + max_duration)
    return int(lb), int(ub)
