#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon as pp
from shapely.geometry import Polygon

from discrete_optimization.generic_tools.plot_utils import (
    get_cmap,
    get_cmap_with_nb_colors,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    RCPSPModelPreemptive,
    RCPSPSolutionPreemptive,
)

logger = logging.getLogger(__name__)


def compute_resource_consumption(
    rcpsp_model: RCPSPModelPreemptive,
    rcpsp_sol: RCPSPSolutionPreemptive,
    list_resources: List[Union[int, str]] = None,
    future_view=True,
):
    modes_dict = rcpsp_model.get_modes_dict(rcpsp_sol)
    modes_dict[rcpsp_model.source_task] = 1
    modes_dict[rcpsp_model.sink_task] = 1
    last_activity = rcpsp_model.sink_task
    makespan = rcpsp_sol.get_end_time(last_activity)
    if list_resources is None:
        list_resources = rcpsp_model.resources_list
    consumptions = np.zeros((len(list_resources), makespan + 1))
    for act_id in rcpsp_model.get_tasks_list():
        for ir in range(len(list_resources)):
            use_ir = rcpsp_model.mode_details[act_id][modes_dict[act_id]].get(
                list_resources[ir], 0
            )
            for s, e in zip(
                rcpsp_sol.get_start_times_list(act_id),
                rcpsp_sol.get_end_times_list(act_id),
            ):
                if future_view:
                    consumptions[ir, s + 1 : e + 1] += use_ir
                else:
                    consumptions[ir, s:e] += use_ir

    return consumptions, np.arange(0, makespan + 1, 1)


def compute_nice_resource_consumption(
    rcpsp_model: RCPSPModelPreemptive,
    rcpsp_sol: RCPSPSolutionPreemptive,
    list_resources: List[Union[int, str]] = None,
):
    if list_resources is None:
        list_resources = rcpsp_model.resources_list
    c_future, times = compute_resource_consumption(
        rcpsp_model, rcpsp_sol, list_resources=list_resources, future_view=True
    )
    c_past, times = compute_resource_consumption(
        rcpsp_model, rcpsp_sol, list_resources=list_resources, future_view=False
    )
    merged_times = {i: [] for i in range(len(list_resources))}
    merged_cons = {i: [] for i in range(len(list_resources))}
    for r in range(len(list_resources)):
        for index_t in range(len(times)):
            merged_times[r] += [times[index_t], times[index_t]]
            merged_cons[r] += [c_future[r, index_t], c_past[r, index_t]]
    for r in merged_times:
        merged_times[r] = np.array(merged_times[r])
        merged_cons[r] = np.array(merged_cons[r])
    return merged_times, merged_cons


def plot_ressource_view(
    rcpsp_model: RCPSPModelPreemptive,
    rcpsp_sol: RCPSPSolutionPreemptive,
    list_resource: List[Union[int, str]] = None,
    title_figure="",
    x_lim=None,
    fig=None,
    ax=None,
):
    modes_dict = rcpsp_model.get_modes_dict(rcpsp_sol)
    with_calendar = rcpsp_model.is_varying_resource()
    if list_resource is None:
        list_resource = rcpsp_model.resources_list
    if ax is None:
        fig, ax = plt.subplots(nrows=len(list_resource), figsize=(10, 5), sharex=True)
        if len(list_resource) == 1:
            ax = [ax]
        fig.suptitle(title_figure)
    polygons_ax = {i: [] for i in range(len(list_resource))}
    labels_ax = {i: [] for i in range(len(list_resource))}
    sorted_activities = sorted(
        rcpsp_model.get_tasks_list(), key=lambda x: rcpsp_sol.get_start_time(x)
    )
    for j in sorted_activities:
        time_starts = rcpsp_sol.get_start_times_list(j)  # rcpsp_schedule[j]["starts"]
        time_ends = rcpsp_sol.get_end_times_list(j)  # rcpsp_schedule[j]["ends"]
        index = 0
        for time_start, time_end in zip(time_starts, time_ends):
            for i in range(len(list_resource)):
                cons = rcpsp_model.mode_details[j][modes_dict[j]].get(
                    list_resource[i], 0
                )
                if cons == 0:
                    continue
                bound = (
                    rcpsp_model.get_resource_available(list_resource[i], 0)
                    if not with_calendar
                    else max(
                        rcpsp_model.get_resource_availability_array(list_resource[i])
                    )
                )
                for k in range(0, bound + 5):
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
                        ax[i].annotate(
                            j,
                            xy=((time_start + time_end) / 2, (k + cons / 2)),
                            font_properties=FontProperties(size=6, weight="bold"),
                            verticalalignment="center",
                            horizontalalignment="left",
                            color="k",
                            clip_on=True,
                        )
                        break
            index += 1
    for i in range(len(list_resource)):
        patches = []
        for polygon in polygons_ax[i]:
            x, y = polygon.exterior.xy
            ax[i].plot(x, y, zorder=-1, color="b")
            patches.append(pp(xy=polygon.exterior.coords))
        p = PatchCollection(patches, cmap=get_cmap("Blues"), alpha=0.4)

        ax[i].add_collection(p)
    merged_times, merged_cons = compute_nice_resource_consumption(
        rcpsp_model, rcpsp_sol, list_resources=list_resource
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
                y=rcpsp_model.get_resource_available(list_resource[i], 0),
                linestyle="--",
                label="Limit : " + str(list_resource[i]),
                zorder=0,
            )
        else:
            ax[i].plot(
                merged_times[i],
                [
                    rcpsp_model.get_resource_available(list_resource[i], m)
                    for m in merged_times[i]
                ],
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
    rcpsp_model: RCPSPModelPreemptive,
    rcpsp_sol: RCPSPSolutionPreemptive,
    fig=None,
    ax=None,
    x_lim=None,
    title=None,
    current_t=None,
):
    if fig is None or ax is None:

        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax.set_title("Gantt Task")
    if title is None:
        ax.set_title("Gantt Task")
    else:
        ax.set_title(title)
    tasks = rcpsp_model.tasks_list
    nb_task = len(tasks)
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["starts"][0]
        + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["ends"][-1]
        + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[-1]]["ends"][-1]
    min_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_start[0]]["starts"][0]
    patches = []
    for j in range(nb_task):
        nb_colors = len(tasks) // 2
        colors = get_cmap_with_nb_colors("hsv", nb_colors)
        for start, end in zip(
            rcpsp_sol.rcpsp_schedule[tasks[j]]["starts"],
            rcpsp_sol.rcpsp_schedule[tasks[j]]["ends"],
        ):
            box = [
                (j - 0.25, start),
                (j - 0.25, end),
                (j + 0.25, end),
                (j + 0.25, start),
                (j - 0.25, start),
            ]
            polygon = Polygon([(b[1], b[0]) for b in box])
            x, y = polygon.exterior.xy
            ax.plot(x, y, zorder=-1, color="b")
            patches.append(
                pp(xy=polygon.exterior.coords, facecolor=colors((j - 1) % nb_colors))
            )
        ax.annotate(
            tasks[j],
            xy=(
                (
                    rcpsp_sol.rcpsp_schedule[tasks[j]]["starts"][0]
                    + rcpsp_sol.rcpsp_schedule[tasks[j]]["ends"][0]
                )
                / 2,
                j,
            ),
            font_properties=FontProperties(size=7, weight="bold"),
            verticalalignment="center",
            horizontalalignment="left",
            color="k",
            clip_on=True,
        )
    p = PatchCollection(patches, match_original=True, alpha=0.4)
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
    rcpsp_model: RCPSPModelPreemptive,
    rcpsp_sol: RCPSPSolutionPreemptive,
    resource_types_to_consider: List[str] = None,
):
    modes_dict = rcpsp_model.build_mode_dict(
        rcpsp_modes_from_solution=rcpsp_sol.rcpsp_modes
    )
    if resource_types_to_consider is None:
        resources = rcpsp_model.resources_list
    else:
        resources = resource_types_to_consider
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["starts"][0]
        + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["ends"][-1]
        + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[-1]]["ends"][-1]
    min_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_start[0]]["starts"][0]
    with_calendar = rcpsp_model.is_varying_resource()

    array_ressource_usage = {
        resources[i]: {
            "activity": np.zeros(
                (
                    max_time - min_time + 1,
                    max(rcpsp_model.resources[resources[i]])
                    if with_calendar
                    else rcpsp_model.resources[resources[i]],
                )
            ),
            "binary_activity": np.zeros(
                (
                    max_time - min_time + 1,
                    max(rcpsp_model.resources[resources[i]])
                    if with_calendar
                    else rcpsp_model.resources[resources[i]],
                )
            ),
            "total_activity": np.zeros(
                max(rcpsp_model.resources[resources[i]])
                if with_calendar
                else rcpsp_model.resources[resources[i]]
            ),
            "activity_last_n_hours": np.zeros(
                (
                    max_time - min_time + 1,
                    max(rcpsp_model.resources[resources[i]])
                    if with_calendar
                    else rcpsp_model.resources[resources[i]],
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
        mode = modes_dict[activity]
        start_times = rcpsp_sol.rcpsp_schedule[activity]["starts"]
        end_times = rcpsp_sol.rcpsp_schedule[activity]["ends"]
        for start_time, end_time in zip(start_times, end_times):
            if end_time == start_time:
                continue
            resources_needed = {
                r: rcpsp_model.mode_details[activity][mode].get(r, 0) for r in resources
            }
            for r in resources_needed:
                if r not in array_ressource_usage:
                    continue
                rneeded = resources_needed[r]
                if not with_calendar:
                    range_interest = range(
                        array_ressource_usage[r]["activity"].shape[1]
                    )
                else:
                    range_interest = range(
                        rcpsp_model.resources[r][time_to_index[start_time]]
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
                            time_to_index[start_time] : time_to_index[end_time],
                            resource,
                        ] = (
                            rcpsp_model.index_task[activity]
                            if isinstance(activity, str)
                            else activity
                        )
                        array_ressource_usage[r]["binary_activity"][
                            time_to_index[start_time] : time_to_index[end_time],
                            resource,
                        ] = 1
                        array_ressource_usage[r]["total_activity"][resource] += (
                            end_time - start_time
                        )
                        array_ressource_usage[r]["activity_last_n_hours"][
                            :, resource
                        ] = np.convolve(
                            array_ressource_usage[r]["binary_activity"][:, resource],
                            np.array([1] * nhour + [0] + [0] * nhour),
                            mode="same",
                        )
                        array_ressource_usage[r]["boxes_time"] += [
                            [
                                (
                                    resource - 0.25,
                                    start_time + 0.01,
                                    rcpsp_model.index_task[activity],
                                ),
                                (
                                    resource - 0.25,
                                    end_time - 0.01,
                                    rcpsp_model.index_task[activity],
                                ),
                                (
                                    resource + 0.25,
                                    end_time - 0.01,
                                    rcpsp_model.index_task[activity],
                                ),
                                (
                                    resource + 0.25,
                                    start_time + 0.01,
                                    rcpsp_model.index_task[activity],
                                ),
                                (
                                    resource - 0.25,
                                    start_time + 0.01,
                                    rcpsp_model.index_task[activity],
                                ),
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
    rcpsp_model: RCPSPModelPreemptive,
    rcpsp_sol: RCPSPSolutionPreemptive,
    resource_types_to_consider: List[str] = None,
    title_figure="",
    x_lim=None,
    fig=None,
    ax=None,
    current_t=None,
):
    array_ressource_usage = compute_schedule_per_resource_individual(
        rcpsp_model, rcpsp_sol, resource_types_to_consider=resource_types_to_consider
    )
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["starts"][0]
        + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.rcpsp_schedule[x]["ends"][-1]
        + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[-1]]["ends"][-1]
    min_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_start[0]]["starts"][0]
    resources_list = list(array_ressource_usage.keys())
    if fig is None or ax is None:
        fig, ax = plt.subplots(len(array_ressource_usage), figsize=(10, 5))
        fig.suptitle(title_figure)
        if len(array_ressource_usage) == 1:
            ax = [ax]

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
            ax[i].set_ylim((-0.5, rcpsp_model.resources[resources_list[i]]))
            ax[i].set_yticks(range(rcpsp_model.resources[resources_list[i]]))
            ax[i].set_yticklabels(
                tuple([j for j in range(rcpsp_model.resources[resources_list[i]])]),
                fontdict={"size": 7},
            )
        except:
            m = max(rcpsp_model.resources[resources_list[i]])
            ax[i].set_ylim((-0.5, m))
            ax[i].set_yticks(range(m))
            ax[i].set_yticklabels(tuple([j for j in range(m)]), fontdict={"size": 7})

        ax[i].grid(True)
        if current_t is not None:
            ax[i].axvline(x=current_t, label="pyplot vertical line", color="r", ls="--")
    ax[-1].set_xlabel("Timestep")
    return fig
