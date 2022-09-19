#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Optional, Tuple

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from discrete_optimization.pickup_vrp.gpdp import GPDP


class VehicleStatus:
    time: float
    position: Tuple[float, float]
    distance: float
    capacity: Dict[str, int]
    status: str

    def __init__(self, time, position, distance, capacity, status, name_node):
        self.time = time
        self.position = position
        self.distance = distance
        self.capacity = capacity
        self.status = status
        self.name_node = name_node

    def copy(self):
        return VehicleStatus(
            self.time,
            tuple(self.position),
            distance=self.distance,
            capacity=deepcopy(self.capacity),
            status=self.status,
            name_node=self.name_node,
        )

    def __str__(self):
        return str(self.time) + " " + str(self.position) + " " + str(self.name_node)


def post_process_solution(result, problem: GPDP, delta_time: int = 10):
    path_aircraft = {}
    current_vehicle_status: Dict[int, VehicleStatus] = {
        v: None for v in range(problem.number_vehicle)
    }
    vehicle_status_history = {v: [] for v in range(problem.number_vehicle)}
    time_of_actual_events = []
    for v in result[0]:
        node_index = result[0][v]
        real_name = [problem.list_nodes[k] for k in node_index]
        print("----Vehicle----")
        print(real_name)
        path_aircraft[v] = {
            "path": real_name,
            "details": [
                result[1][(v, k)] if (v, k) in result[1] else result[1][(v, k, "end")]
                for k in node_index
            ],
        }
        time_of_actual_events += [k["Time"][0] for k in path_aircraft[v]["details"]]
        time_of_actual_events += [k["Time"][1] for k in path_aircraft[v]["details"]]
    for v in path_aircraft:
        current_vehicle_status[v] = VehicleStatus(
            time=path_aircraft[v]["details"][0]["Time"][0],
            position=problem.coordinates_2d[path_aircraft[v]["path"][0]],
            distance=path_aircraft[v]["details"][0]["Distance"][0],
            capacity={
                r: path_aircraft[v]["details"][0][r]
                for r in path_aircraft[v]["details"][0]
                if "Capacity" in r
            },
            status="at the depot",
            name_node=path_aircraft[v]["path"][0],
        )
        vehicle_status_history[v] += [current_vehicle_status[v].copy()]
        for j in range(len(path_aircraft[v]["path"])):
            current_vehicle_status[v] = current_vehicle_status[v].copy()
            current_vehicle_status[v].time = path_aircraft[v]["details"][j]["Time"][0]
            current_vehicle_status[v].position = problem.coordinates_2d[
                path_aircraft[v]["path"][j]
            ]
            current_vehicle_status[v].capacity = {
                r: path_aircraft[v]["details"][j][r]
                for r in path_aircraft[v]["details"][j]
                if "Capacity" in r
            }
            current_vehicle_status[v].name_node = path_aircraft[v]["path"][j]
            current_vehicle_status[v].status = "arrived at destination"
            if path_aircraft[v]["details"][j]["Time"][2] > 0:
                current_vehicle_status[v].status = (
                    "waiting time starting...will last"
                    + str(path_aircraft[v]["details"][j]["Time"][2])
                    + " minutes"
                )
                vehicle_status_history[v] += [current_vehicle_status[v].copy()]

                current_vehicle_status[v] = current_vehicle_status[v].copy()
                current_vehicle_status[v].time = (
                    current_vehicle_status[v].time
                    + path_aircraft[v]["details"][j]["Time"][2]
                )
                current_vehicle_status[v].capacity = {
                    r: path_aircraft[v]["details"][j + 1][r]
                    for r in path_aircraft[v]["details"][j + 1]
                    if "Capacity" in r
                }
                current_vehicle_status[v].status = "waiting time finished !"
                vehicle_status_history[v] += [current_vehicle_status[v].copy()]
            else:
                vehicle_status_history[v] += [current_vehicle_status[v].copy()]
                # that's how ortools store the solution, a bit cumbersome

    return path_aircraft, vehicle_status_history


def compute_bounds(history: Dict[int, List[VehicleStatus]]):
    x_0_min = min(
        [history[v][k].position[0] for v in history for k in range(len(history[v]))]
    )
    x_0_max = max(
        [history[v][k].position[0] for v in history for k in range(len(history[v]))]
    )
    x_1_min = min(
        [history[v][k].position[1] for v in history for k in range(len(history[v]))]
    )
    x_1_max = max(
        [history[v][k].position[1] for v in history for k in range(len(history[v]))]
    )
    return (x_0_min, x_0_max), (x_1_min, x_1_max)


def plot_flights(
    history: Dict[int, List[VehicleStatus]],
    problem: GPDP,
    plot_all_flight: bool = False,
    plot_network: bool = False,
    time_delta: int = 30,
    map_view=True,
    save_video: bool = False,
    name_file: str = "video_gpdp",
    folder_to_save_video: Optional[str] = None,
):
    previous_time = 0
    current_time = 0
    nb_colors = problem.number_vehicle
    colors = plt.cm.get_cmap("hsv", nb_colors + 1)
    colors_vehicle = {v: colors(v) for v in range(problem.number_vehicle)}
    x_0, x_1 = compute_bounds(history)
    if save_video:
        if folder_to_save_video is None:
            raise ValueError(
                "folder_to_save_video must not be None if save_video is True."
            )
        if not os.path.exists(folder_to_save_video):
            os.makedirs(folder_to_save_video)
        else:
            for k in os.listdir(folder_to_save_video):
                os.remove(os.path.join(folder_to_save_video, k))

    with plt.xkcd():
        fig = plt.figure("Fleet scheduling render", figsize=(15, 15))
        plt.tight_layout()
        gs = GridSpec(10, 10, figure=fig)
        ressources = list(problem.resources_set)
        real_name = list(ressources)
        ressources = ["Capacity_" + str(r) for r in ressources]

        if map_view:
            if len(ressources) > 0:
                ax = fig.add_subplot(gs[:5, :], projection=ccrs.PlateCarree())
            else:
                ax = fig.add_subplot(gs[:, :], projection=ccrs.PlateCarree())
        else:
            if len(ressources) > 0:
                ax = fig.add_subplot(gs[:5, :])
            else:
                ax = fig.add_subplot(gs[:, :])
        if plot_network:
            nb_colors_clusters = len(problem.clusters_set)
            colors_nodes = plt.cm.get_cmap("rainbow", nb_colors_clusters)
            ax.scatter(
                [problem.coordinates_2d[node][1] for node in problem.clusters_dict],
                [problem.coordinates_2d[node][0] for node in problem.clusters_dict],
                s=2,
                zorder=2,
                color=[
                    colors_nodes(problem.clusters_dict[node])
                    for node in problem.clusters_dict
                ],
            )
        max_time = max([history[v][-1].time for v in history])
        ressources = list(problem.resources_set)
        real_name = list(ressources)
        ressources = ["Capacity_" + str(r) for r in ressources]
        dict_ax_ressources = {}
        if len(ressources) > 0:
            k = int(10 / len(ressources))
            for j in range(len(ressources)):

                if j == len(ressources) - 1:
                    dict_ax_ressources[ressources[j]] = fig.add_subplot(gs[5:, j * k :])
                else:
                    dict_ax_ressources[ressources[j]] = fig.add_subplot(
                        gs[5:, j * k : (j + 1) * k]
                    )
                dict_ax_ressources[ressources[j]].set_title(
                    "Transported " + str(real_name[j])
                )
        if map_view:
            LAND_10m = cartopy.feature.NaturalEarthFeature(
                "physical",
                "land",
                "50m",
                edgecolor="face",
                facecolor=cartopy.feature.COLORS["land"],
            )
            OCEAN_50m = cartopy.feature.NaturalEarthFeature(
                "physical",
                "ocean",
                "50m",
                edgecolor="face",
                facecolor=cartopy.feature.COLORS["water"],
                zorder=1,
            )
            ax.coastlines(resolution="50m", zorder=2)
            ax.add_feature(LAND_10m)
            ax.add_feature(OCEAN_50m)
            if map_view:
                ax.stock_img()
        ax.set_xlim(
            [x_1[0] - (x_1[1] - x_1[0]) * 0.1, x_1[1] + (x_1[1] - x_1[0]) * 0.1]
        )
        ax.set_ylim(
            [x_0[0] - (x_0[1] - x_0[0]) * 0.1, x_0[1] + (x_0[1] - x_0[0]) * 0.1]
        )
        plt.ion()
        ax.set_title("Title")
        index_per_flights = {v: 0 for v in history}
        position_v = {v: None for v in history}
        current_flight = {v: None for v in history}
        finished = {v: False for v in history}
        if plot_all_flight:
            all_flights = {v: {"plot": None, "history": None} for v in history}
        for v in index_per_flights:
            if map_view:
                (current_flight[v],) = ax.plot(
                    [history[v][0].position[1], history[v][1].position[1]],
                    [history[v][0].position[0], history[v][1].position[0]],
                    color=colors_vehicle[v],
                    alpha=1,
                    transform=ccrs.Geodetic(),
                )
            else:
                (current_flight[v],) = ax.plot(
                    [history[v][0].position[1], history[v][1].position[1]],
                    [history[v][0].position[0], history[v][1].position[0]],
                    alpha=1,
                    linewidth=3,
                    color=colors_vehicle[v],
                )
            (position_v[v],) = ax.plot(
                [history[v][0].position[1]],
                [history[v][0].position[0]],
                markersize=4 + 1.5 * v,
                color=colors_vehicle[v],
                marker="o",
            )
            if plot_all_flight:
                if map_view:
                    (all_flights[v]["plot"],) = ax.plot(
                        [history[v][0].position[1], history[v][1].position[1]],
                        [history[v][0].position[0], history[v][1].position[0]],
                        marker="*",
                        markersize=3,
                        color=colors_vehicle[v],
                        alpha=0.5,
                        transform=ccrs.Geodetic(),
                    )
                else:
                    (all_flights[v]["plot"],) = ax.plot(
                        [history[v][0].position[1], history[v][1].position[1]],
                        [history[v][0].position[0], history[v][1].position[0]],
                        color=colors_vehicle[v],
                        marker="*",
                        markersize=3,
                        alpha=0.5,
                    )
                all_flights[v]["history"] = (
                    [history[v][0].position[1], history[v][1].position[1]],
                    [history[v][0].position[0], history[v][1].position[0]],
                )
        plot_vehicle_resource = {
            r: {v: {"plot": None, "cur_history": None} for v in history}
            for r in ressources
        }
        current_vehicle_resource = {r: {v: None for v in history} for r in ressources}
        for ressource in dict_ax_ressources:
            for v in history:
                print(history[v][0].capacity)
                (plot_vehicle_resource[ressource][v]["plot"],) = dict_ax_ressources[
                    ressource
                ].plot(
                    [history[v][0].time][history[v][0].capacity[ressource][0]],
                    color=colors_vehicle[v],
                    label="vehicle n°" + str(v),
                )
                plot_vehicle_resource[ressource][v]["cur_history"] = (
                    [history[v][0].time],
                    [history[v][0].capacity[ressource][0]],
                )
                (current_vehicle_resource[ressource][v],) = dict_ax_ressources[
                    ressource
                ].plot(
                    [history[v][0].time],
                    [history[v][0].capacity[ressource][0]],
                    markersize=1.5 * v + 4,
                    alpha=0.8,
                    color=colors_vehicle[v],
                    marker="o",
                )
            dict_ax_ressources[ressource].legend(prop={"size": 10})
        index = 0
        while not (all(finished[v] for v in finished)):
            previous_time = current_time
            current_time += time_delta
            status_to_plot = {}
            for v in history:
                status_to_plot[v] = [
                    (j, history[v][j].time, history[v][j], v)
                    for j in range(index_per_flights[v] + 1, len(history[v]))
                    if previous_time <= history[v][j].time <= current_time
                ]
                if len(status_to_plot[v]) > 0:
                    index_per_flights[v] = status_to_plot[v][-1][0]
                    finished[v] = index_per_flights[v] == (len(history[v]) - 1)
            allstat_to_plot = sorted(
                reduce(
                    lambda x, y: x + y, [status_to_plot[v] for v in status_to_plot], []
                ),
                key=lambda x: (x[1], x[0]),
            )
            for j in range(len(ressources)):
                ressource = ressources[j]
                dict_ax_ressources[ressource].set_xlim(
                    [-1, current_time + 5 * time_delta]
                )
                dict_ax_ressources[ressource].set_ylim(
                    [
                        -0.5,
                        max(
                            [
                                problem.capacities[v][real_name[j]][1] + 1
                                for v in problem.capacities
                            ]
                        ),
                    ]
                )
            for status in allstat_to_plot:
                position_v[status[-1]].set_xdata([status[2].position[1]])
                position_v[status[-1]].set_ydata([status[2].position[0]])
                for ressource in dict_ax_ressources:
                    plot_vehicle_resource[ressource][status[-1]]["cur_history"] = (
                        plot_vehicle_resource[ressource][status[-1]]["cur_history"][0]
                        + [status[2].time],
                        plot_vehicle_resource[ressource][status[-1]]["cur_history"][1]
                        + [status[2].capacity[ressource][0]],
                    )
                    plot_vehicle_resource[ressource][status[-1]]["plot"].set_xdata(
                        plot_vehicle_resource[ressource][status[-1]]["cur_history"][0]
                    )
                    plot_vehicle_resource[ressource][status[-1]]["plot"].set_ydata(
                        plot_vehicle_resource[ressource][status[-1]]["cur_history"][1]
                    )
                    current_vehicle_resource[ressource][status[-1]].set_xdata(
                        [
                            plot_vehicle_resource[ressource][status[-1]]["cur_history"][
                                0
                            ][-1]
                        ]
                    )
                    current_vehicle_resource[ressource][status[-1]].set_ydata(
                        [
                            plot_vehicle_resource[ressource][status[-1]]["cur_history"][
                                1
                            ][-1]
                        ]
                    )
                    current_vehicle_resource[ressource][status[-1]].set_xdata(
                        [status[2].time]
                    )
                    current_vehicle_resource[ressource][status[-1]].set_ydata(
                        [status[2].capacity[ressource][0]]
                    )

                if status[0] < len(history[status[-1]]) - 1:
                    current_flight[status[-1]].set_xdata(
                        [
                            history[status[-1]][status[0]].position[1],
                            history[status[-1]][status[0] + 1].position[1],
                        ]
                    )
                    current_flight[status[-1]].set_ydata(
                        [
                            history[status[-1]][status[0]].position[0],
                            history[status[-1]][status[0] + 1].position[0],
                        ]
                    )
                    if plot_all_flight:
                        all_flights[status[-1]]["history"] = (
                            all_flights[status[-1]]["history"][0]
                            + [history[status[-1]][status[0] + 1].position[1]],
                            all_flights[status[-1]]["history"][1]
                            + [history[status[-1]][status[0] + 1].position[0]],
                        )
                        all_flights[status[-1]]["plot"].set_xdata(
                            all_flights[status[-1]]["history"][0]
                        )
                        all_flights[status[-1]]["plot"].set_ydata(
                            all_flights[status[-1]]["history"][1]
                        )
                for v in history:
                    if v == status[-1]:
                        continue
                    if finished[v]:
                        if finished[v]:
                            for ressource in dict_ax_ressources:
                                plot_vehicle_resource[ressource][v]["cur_history"] = (
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        0
                                    ]
                                    + [current_time],
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        1
                                    ]
                                    + [history[v][-1].capacity[ressource][0]],
                                )
                                plot_vehicle_resource[ressource][v]["plot"].set_xdata(
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        0
                                    ]
                                )
                                plot_vehicle_resource[ressource][v]["plot"].set_ydata(
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        1
                                    ]
                                )
                                current_vehicle_resource[ressource][v].set_xdata(
                                    [
                                        plot_vehicle_resource[ressource][v][
                                            "cur_history"
                                        ][0][-1]
                                    ]
                                )
                                current_vehicle_resource[ressource][v].set_ydata(
                                    [
                                        plot_vehicle_resource[ressource][v][
                                            "cur_history"
                                        ][1][-1]
                                    ]
                                )
                                current_vehicle_resource[ressource][v].set_xdata(
                                    [current_time]
                                )
                                current_vehicle_resource[ressource][v].set_ydata(
                                    [history[v][-1].capacity[ressource][0]]
                                )
                            continue
                    t = status[1]
                    jj = next(
                        (
                            i
                            for i in range(len(history[v]) - 1)
                            if history[v][i].time <= t < history[v][i + 1].time
                        ),
                        None,
                    )
                    if jj is not None:
                        tt_1 = history[v][jj].time
                        tt_2 = history[v][jj + 1].time
                        coef = (
                            history[v][jj + 1].position[1] - history[v][jj].position[1]
                        ) / (tt_2 - tt_1)
                        b = history[v][jj].position[1]
                        pos_0 = coef * (t - tt_1) + b
                        coef = (
                            history[v][jj + 1].position[0] - history[v][jj].position[0]
                        ) / (tt_2 - tt_1)
                        b = history[v][jj].position[0]
                        pos_1 = coef * (t - tt_1) + b
                        position_v[v].set_xdata([pos_0])
                        position_v[v].set_ydata([pos_1])
                        for ressource in dict_ax_ressources:
                            plot_vehicle_resource[ressource][v]["cur_history"] = (
                                plot_vehicle_resource[ressource][v]["cur_history"][0]
                                + [t],
                                plot_vehicle_resource[ressource][v]["cur_history"][1]
                                + [history[v][jj].capacity[ressource][0]],
                            )
                            plot_vehicle_resource[ressource][v]["plot"].set_xdata(
                                plot_vehicle_resource[ressource][v]["cur_history"][0]
                            )
                            plot_vehicle_resource[ressource][v]["plot"].set_ydata(
                                plot_vehicle_resource[ressource][v]["cur_history"][1]
                            )
                            current_vehicle_resource[ressource][v].set_xdata(
                                [
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        0
                                    ][-1]
                                ]
                            )
                            current_vehicle_resource[ressource][v].set_ydata(
                                [
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        1
                                    ][-1]
                                ]
                            )
                            current_vehicle_resource[ressource][v].set_xdata([t])
                            current_vehicle_resource[ressource][v].set_ydata(
                                [history[v][jj].capacity[ressource][0]]
                            )

                ax.set_title("Time = " + str(status[1]))
                if not save_video:
                    plt.draw()
                    plt.pause(0.1)
                if save_video:
                    fig.savefig(
                        os.path.join(folder_to_save_video, f"{name_file}{index:04d}")
                    )
                    index += 1
            if len(allstat_to_plot) == 0:
                for v in history:
                    if finished[v]:
                        for ressource in dict_ax_ressources:
                            plot_vehicle_resource[ressource][v]["cur_history"] = (
                                plot_vehicle_resource[ressource][v]["cur_history"][0]
                                + [current_time],
                                plot_vehicle_resource[ressource][v]["cur_history"][1]
                                + [history[v][-1].capacity[ressource][0]],
                            )
                            plot_vehicle_resource[ressource][v]["plot"].set_xdata(
                                plot_vehicle_resource[ressource][v]["cur_history"][0]
                            )
                            plot_vehicle_resource[ressource][v]["plot"].set_ydata(
                                plot_vehicle_resource[ressource][v]["cur_history"][1]
                            )
                            current_vehicle_resource[ressource][v].set_xdata(
                                [
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        0
                                    ][-1]
                                ]
                            )
                            current_vehicle_resource[ressource][v].set_ydata(
                                [
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        1
                                    ][-1]
                                ]
                            )
                            current_vehicle_resource[ressource][v].set_xdata(
                                [current_time]
                            )
                            current_vehicle_resource[ressource][v].set_ydata(
                                [history[v][-1].capacity[ressource][0]]
                            )
                        continue
                    t = current_time
                    jj = next(
                        (
                            i
                            for i in range(len(history[v]) - 1)
                            if history[v][i].time <= t < history[v][i + 1].time
                        ),
                        None,
                    )
                    if jj is not None:
                        tt_1 = history[v][jj].time
                        tt_2 = history[v][jj + 1].time
                        coef = (
                            history[v][jj + 1].position[1] - history[v][jj].position[1]
                        ) / (tt_2 - tt_1)
                        b = history[v][jj].position[1]
                        pos_0 = coef * (t - tt_1) + b
                        coef = (
                            history[v][jj + 1].position[0] - history[v][jj].position[0]
                        ) / (tt_2 - tt_1)
                        b = history[v][jj].position[0]
                        pos_1 = coef * (t - tt_1) + b
                        position_v[v].set_xdata([pos_0])
                        position_v[v].set_ydata([pos_1])
                        for ressource in dict_ax_ressources:
                            plot_vehicle_resource[ressource][v]["cur_history"] = (
                                plot_vehicle_resource[ressource][v]["cur_history"][0]
                                + [t],
                                plot_vehicle_resource[ressource][v]["cur_history"][1]
                                + [history[v][jj].capacity[ressource][0]],
                            )
                            plot_vehicle_resource[ressource][v]["plot"].set_xdata(
                                plot_vehicle_resource[ressource][v]["cur_history"][0]
                            )
                            plot_vehicle_resource[ressource][v]["plot"].set_ydata(
                                plot_vehicle_resource[ressource][v]["cur_history"][1]
                            )
                            current_vehicle_resource[ressource][v].set_xdata(
                                [
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        0
                                    ][-1]
                                ]
                            )
                            current_vehicle_resource[ressource][v].set_ydata(
                                [
                                    plot_vehicle_resource[ressource][v]["cur_history"][
                                        1
                                    ][-1]
                                ]
                            )
                            current_vehicle_resource[ressource][v].set_xdata([t])
                            current_vehicle_resource[ressource][v].set_ydata(
                                [history[v][jj].capacity[ressource][0]]
                            )

                ax.set_title("Time = " + str(current_time))
                if not save_video:
                    plt.draw()
                    plt.pause(0.1)
                if save_video:
                    fig.savefig(
                        os.path.join(folder_to_save_video, f"{name_file}{index:04d}")
                    )
                    index += 1
        if save_video:
            a = (
                "ffmpeg -r 2 -pattern_type glob -i '"
                + os.path.join(folder_to_save_video, name_file)
                + "*.png' -c:v libx264 -vf "
                + '"fps=5,format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2" '
                + os.path.join(folder_to_save_video, name_file + ".mp4")
            )
            os.system(a)

        return os.path.join(folder_to_save_video, name_file + ".mp4")
