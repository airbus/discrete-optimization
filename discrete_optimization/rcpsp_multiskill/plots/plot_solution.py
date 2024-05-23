#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon as pp
from shapely.geometry import Polygon

from discrete_optimization.generic_tools.plot_utils import get_cmap_with_nb_colors
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Preemptive,
)


def compute_schedule_per_resource_individual_preemptive(
    rcpsp_model: MS_RCPSPModel, rcpsp_sol: MS_RCPSPSolution_Preemptive
):
    sorted_task_by_start = sorted(
        rcpsp_sol.schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    employee_usage = {
        employee: {
            "activity": np.zeros((max_time - min_time + 1)),
            "binary_activity": np.zeros((max_time - min_time + 1)),
            "total_activity": 0,
            "boxes_time": [],
        }
        for employee in rcpsp_model.employees
    }
    index_to_time = {i: min_time + i for i in range(max_time - min_time + 1)}
    time_to_index = {index_to_time[i]: i for i in index_to_time}
    sorted_employees = list(sorted(rcpsp_model.employees))
    for activity in sorted_task_by_start:
        for i in range(len(rcpsp_sol.employee_usage.get(activity, []))):
            if isinstance(rcpsp_sol.employee_usage[activity], dict):
                employees_i = rcpsp_sol.employee_usage[activity].keys()
            else:
                employees_i = rcpsp_sol.employee_usage[activity][i]
            for employee in employees_i:
                start_time = rcpsp_sol.schedule[activity]["starts"][i]
                end_time = rcpsp_sol.schedule[activity]["ends"][i]
                employee_usage[employee]["activity"][
                    time_to_index[start_time] : time_to_index[end_time]
                ] = (
                    rcpsp_model.index_task[activity]
                    if isinstance(activity, str)
                    else activity
                )
                employee_usage[employee]["binary_activity"][
                    time_to_index[start_time] : time_to_index[end_time]
                ] = 1
                employee_usage[employee]["total_activity"] += end_time - start_time
                index_employee = sorted_employees.index(employee)
                employee_usage[employee]["boxes_time"] += [
                    [
                        (index_employee - 0.25, start_time + 0.01, activity),
                        (index_employee - 0.25, end_time - 0.01, activity),
                        (index_employee + 0.25, end_time - 0.01, activity),
                        (index_employee + 0.25, start_time + 0.01, activity),
                        (index_employee - 0.25, start_time + 0.01, activity),
                    ]
                ]
    return employee_usage


def compute_schedule_per_resource_individual(
    rcpsp_model: MS_RCPSPModel, rcpsp_sol: MS_RCPSPSolution
):
    modes = rcpsp_sol.modes
    sorted_task_by_start = sorted(
        rcpsp_sol.schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    employee_usage = {
        employee: {
            "activity": np.zeros((max_time - min_time + 1)),
            "binary_activity": np.zeros((max_time - min_time + 1)),
            "total_activity": 0,
            "boxes_time": [],
        }
        for employee in rcpsp_model.employees
    }
    index_to_time = {i: min_time + i for i in range(max_time - min_time + 1)}
    time_to_index = {index_to_time[i]: i for i in index_to_time}
    sorted_employees = list(sorted(rcpsp_model.employees))
    for activity in sorted_task_by_start:
        start_time = rcpsp_sol.get_start_time(activity)
        end_time = rcpsp_sol.get_end_time(activity)
        for employee in rcpsp_sol.employee_usage.get(activity, {}):
            employee_usage[employee]["activity"][
                time_to_index[start_time] : time_to_index[end_time]
            ] = (
                rcpsp_model.index_task[activity]
                if isinstance(activity, str)
                else activity
            )
            employee_usage[employee]["binary_activity"][
                time_to_index[start_time] : time_to_index[end_time]
            ] = 1
            employee_usage[employee]["total_activity"] += end_time - start_time
            index_employee = sorted_employees.index(employee)
            employee_usage[employee]["boxes_time"] += [
                [
                    (index_employee - 0.25, start_time + 0.01, activity),
                    (index_employee - 0.25, end_time - 0.01, activity),
                    (index_employee + 0.25, end_time - 0.01, activity),
                    (index_employee + 0.25, start_time + 0.01, activity),
                    (index_employee - 0.25, start_time + 0.01, activity),
                ]
            ]
    return employee_usage


def plot_resource_individual_gantt(
    rcpsp_model: MS_RCPSPModel,
    rcpsp_sol: MS_RCPSPSolution,
    title_figure="",
    name_task=None,
    fig=None,
    ax=None,
    current_t=None,
):
    array_ressource_usage = compute_schedule_per_resource_individual(
        rcpsp_model, rcpsp_sol
    )
    sorted_task_by_start = sorted(
        rcpsp_sol.schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    sorted_employees = list(sorted(rcpsp_model.employees))

    if name_task is None:
        name_task = {}
        for t in rcpsp_model.mode_details:
            name_task[t] = str(t)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, figsize=(12, 6))
        fig.suptitle(title_figure)
    position_label = {}
    for i in range(len(sorted_employees)):
        patches = []
        nb_colors = len(sorted_task_by_start) // 2
        colors = get_cmap_with_nb_colors("hsv", nb_colors)
        for boxe in array_ressource_usage[sorted_employees[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            x, y = polygon.exterior.xy
            ax.plot(x, y, zorder=-1, color="b")
            patches.append(
                pp(
                    xy=polygon.exterior.coords,
                    facecolor=colors((rcpsp_model.index_task[activity]) % nb_colors),
                )
            )
            activity = boxe[0][2]
            if abs(boxe[0][1] - boxe[1][1]) >= 0.4:
                # (resource - 0.25, start_time + 0.01, activity),
                # (resource - 0.25, end_time - 0.01, activity),
                # (resource + 0.25, end_time - 0.01, activity),
                # (resource + 0.25, start_time + 0.01, activity),
                # (resource - 0.25, start_time + 0.01, activity)
                center = (
                    sum([b[1] for b in boxe[:4]]) / 4 - 0.4,
                    sum(b[0] for b in boxe[:4]) / 4,
                )
                if activity not in position_label:
                    position_label[activity] = center
                position_label[activity] = max(center, position_label[activity])
        p = PatchCollection(
            patches,
            match_original=True,
            alpha=0.4,
        )
        ax.add_collection(p)
        ax.set_xlim((min_time, max_time))
        ax.set_ylim((-0.5, len(sorted_employees)))
        ax.set_yticks(range(len(sorted_employees)))
        ax.set_yticklabels(tuple(sorted_employees), fontdict={"size": 7})
        for activity in position_label:
            ax.annotate(
                name_task[activity],
                xy=position_label[activity],
                font_properties=FontProperties(size=7, weight="bold"),
                verticalalignment="center",
                horizontalalignment="left",
                color="k",
                clip_on=True,
            )
        ax.grid(True)
        if current_t is not None:
            ax.axvline(x=current_t, label="pyplot vertical line", color="r", ls="--")
    return fig


def plot_resource_individual_gantt_preemptive(
    rcpsp_model: MS_RCPSPModel,
    rcpsp_sol: MS_RCPSPSolution_Preemptive,
    title_figure="",
    name_task=None,
    subtasks=None,
    fig=None,
    ax=None,
    current_t=None,
):
    array_ressource_usage = compute_schedule_per_resource_individual_preemptive(
        rcpsp_model, rcpsp_sol
    )
    sorted_task_by_start = sorted(
        rcpsp_sol.schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    sorted_employees = list(sorted(rcpsp_model.employees))

    if name_task is None:
        name_task = {}
        for t in rcpsp_model.mode_details:
            name_task[t] = str(t)
    if subtasks is None:
        subtasks = set(rcpsp_model.tasks_list)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, figsize=(12, 6))
        fig.suptitle(title_figure)
    position_label = {}
    for i in range(len(sorted_employees)):
        patches = []
        nb_colors = len(sorted_task_by_start) // 2
        colors = get_cmap_with_nb_colors("hsv", nb_colors)
        for boxe in array_ressource_usage[sorted_employees[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            if activity not in subtasks:
                continue
            x, y = polygon.exterior.xy
            ax.plot(x, y, zorder=-1, color="b")
            patches.append(
                pp(
                    xy=polygon.exterior.coords,
                    facecolor=colors((rcpsp_model.index_task[activity]) % nb_colors),
                )
            )
            activity = boxe[0][2]
            if abs(boxe[0][1] - boxe[1][1]) >= 0.4:
                center = (
                    sum([b[1] for b in boxe[:4]]) / 4 - 0.4,
                    sum(b[0] for b in boxe[:4]) / 4,
                )
                if activity not in position_label:
                    position_label[activity] = center
                position_label[activity] = max(center, position_label[activity])
        p = PatchCollection(
            patches,
            match_original=True,
            alpha=0.4,
        )
        ax.add_collection(p)
        ax.set_xlim((min_time, max_time))
        ax.set_ylim((-0.5, len(sorted_employees)))
        ax.set_yticks(range(len(sorted_employees)))
        ax.set_yticklabels(tuple(sorted_employees), fontdict={"size": 7})
        for activity in position_label:
            ax.annotate(
                name_task[activity],
                xy=position_label[activity],
                font_properties=FontProperties(size=7, weight="bold"),
                verticalalignment="center",
                horizontalalignment="left",
                color="k",
                clip_on=True,
            )
        ax.grid(True)
        if current_t is not None:
            ax.axvline(x=current_t, label="pyplot vertical line", color="r", ls="--")
    return fig


def plot_task_gantt(
    rcpsp_model: MS_RCPSPModel,
    rcpsp_sol: MS_RCPSPSolution,
    subtasks=None,
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
    if subtasks is None:
        subtasks = set(rcpsp_model.tasks_list)
    tasks = [t for t in rcpsp_model.tasks_list if t in subtasks]
    nb_task = len(tasks)
    sorted_task_by_start = sorted(
        tasks,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        tasks,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_start[0])
    patches = []
    for j in range(nb_task):
        nb_colors = len(tasks) // 2
        colors = get_cmap_with_nb_colors("hsv", nb_colors)
        for start, end in zip(
            rcpsp_sol.get_start_times_list(tasks[j]),
            rcpsp_sol.get_end_times_list(tasks[j]),
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
                    rcpsp_sol.get_start_times_list(tasks[j])[0]
                    + +rcpsp_sol.get_end_times_list(tasks[j])[0]
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
