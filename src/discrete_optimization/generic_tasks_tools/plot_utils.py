#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import (
    annotations,  # make annotations be considered as string by default
)

from typing import Hashable, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pp
from matplotlib.pyplot import get_cmap
from shapely.geometry import Polygon

from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tools.plot_utils import get_cmap_with_nb_colors


def compute_resource_consumption(
    scheduling_problem: GenericSchedulingProblem,
    scheduling_sol: GenericSchedulingSolution,
    list_resources: Optional[list[str]] = None,
    future_view: bool = True,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    makespan = scheduling_sol.get_max_end_time()
    if list_resources is None:
        list_resources = scheduling_problem.cumulative_resources_list
    consumptions = np.zeros((len(list_resources), makespan + 1), dtype=np.int_)
    for act_id in scheduling_problem.tasks_list:
        for ir in range(len(list_resources)):
            use_ir = scheduling_sol.get_calendar_resource_consumption(
                resource=list_resources[ir], task=act_id
            )
            st = scheduling_sol.get_start_time(act_id)
            end = scheduling_sol.get_end_time(act_id)
            if future_view:
                consumptions[
                    ir,
                    st + 1 : end + 1,
                ] += use_ir
            else:
                consumptions[ir, st:end] += use_ir

    return consumptions, np.arange(0, makespan + 1, 1, dtype=np.int_)


def compute_nice_resource_consumption(
    scheduling_problem: GenericSchedulingProblem,
    scheduling_sol: GenericSchedulingSolution,
    list_resources: Optional[list[str]] = None,
) -> tuple[dict[int, npt.NDArray[np.int_]], dict[int, npt.NDArray[np.int_]]]:
    if list_resources is None:
        list_resources = scheduling_problem.cumulative_resources_list
    c_future, times = compute_resource_consumption(
        scheduling_problem,
        scheduling_sol,
        list_resources=list_resources,
        future_view=True,
    )
    c_past, times = compute_resource_consumption(
        scheduling_problem,
        scheduling_sol,
        list_resources=list_resources,
        future_view=False,
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
    scheduling_problem: GenericSchedulingProblem,
    scheduling_sol: GenericSchedulingSolution,
    list_resource: Optional[list[str]] = None,
    title_figure: str = "",
    x_lim: Optional[list[int]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[npt.NDArray[np.object_]] = None,
) -> plt.Figure:
    if list_resource is None:
        list_resource = scheduling_problem.cumulative_resources_list
    if ax is None:
        fig, ax = plt.subplots(nrows=len(list_resource), figsize=(10, 5), sharex=True)
        if len(list_resource) == 1:
            ax = [ax]
        fig.suptitle(title_figure)
    polygons_ax: dict[int, list[Polygon]] = {i: [] for i in range(len(list_resource))}
    labels_ax: dict[int, list[Hashable]] = {i: [] for i in range(len(list_resource))}
    sorted_activities = sorted(
        scheduling_problem.tasks_list, key=lambda x: scheduling_sol.get_start_time(x)
    )
    for j in sorted_activities:
        time_start = scheduling_sol.get_start_time(j)
        time_end = scheduling_sol.get_end_time(j)
        for i in range(len(list_resource)):
            cons = scheduling_sol.get_calendar_resource_consumption(list_resource[i], j)
            if cons == 0:
                continue
            bound: int = int(
                scheduling_problem.get_resource_max_capacity(list_resource[i])
            )
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
        scheduling_problem, scheduling_sol, list_resources=list_resource
    )
    for i in range(len(list_resource)):
        calendar = scheduling_problem.get_resource_calendar(list_resource[i])
        ax[i].plot(
            merged_times[i],
            merged_cons[i],
            color="r",
            linewidth=2,
            label="Consumption " + str(list_resource[i]),
            zorder=1,
        )
        ax[i].plot(
            merged_times[i],
            [calendar[x] for x in merged_times[i]],  # type: ignore
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
    scheduling_problem: GenericSchedulingProblem,
    scheduling_sol: GenericSchedulingSolution,
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
    tasks = scheduling_problem.tasks_list
    nb_task = len(tasks)
    sorted_task_by_start = sorted(
        tasks,
        key=lambda x: 100000 * scheduling_sol.get_start_time(x)
        + scheduling_problem.get_index_from_task(x),
    )
    sorted_task_by_end = sorted(
        tasks,
        key=lambda x: 100000 * scheduling_sol.get_end_time(x)
        + scheduling_problem.get_index_from_task(x),
    )
    max_time = scheduling_sol.get_end_time(sorted_task_by_end[-1])
    min_time = scheduling_sol.get_start_time(sorted_task_by_start[0])
    patches = []
    for j in range(nb_task):
        nb_colors = len(tasks) // 2
        colors = get_cmap_with_nb_colors("hsv", nb_colors)
        box = [
            (j - 0.25, scheduling_sol.get_start_time(tasks[j])),
            (j - 0.25, scheduling_sol.get_end_time(tasks[j])),
            (j + 0.25, scheduling_sol.get_end_time(tasks[j])),
            (j + 0.25, scheduling_sol.get_start_time(tasks[j])),
            (j - 0.25, scheduling_sol.get_start_time(tasks[j])),
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
