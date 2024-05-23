#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import print_function

from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from discrete_optimization.generic_tools.plot_utils import get_cmap_with_nb_colors
from discrete_optimization.pickup_vrp.gpdp import GPDP, GPDPSolution


def plot_gpdp_solution(
    sol: GPDPSolution,
    problem: GPDP,
) -> Tuple[Figure, Axes]:
    if problem.coordinates_2d is None:
        raise ValueError(
            "problem.coordinates_2d cannot be None when calling plot_ortools_solution."
        )
    vehicle_tours = sol.trajectories
    fig, ax = plt.subplots(1)
    nb_colors_clusters = len(problem.clusters_set)
    colors_nodes = get_cmap_with_nb_colors("hsv", nb_colors_clusters)
    ax.scatter(
        [problem.coordinates_2d[node][0] for node in problem.clusters_dict],
        [problem.coordinates_2d[node][1] for node in problem.clusters_dict],
        s=1,
        color=[
            colors_nodes(problem.clusters_dict[node]) for node in problem.clusters_dict
        ],
    )
    for v, traj in vehicle_tours.items():
        ax.plot(
            [problem.coordinates_2d[node][0] for node in traj],
            [problem.coordinates_2d[node][1] for node in traj],
            label="vehicle n°" + str(v),
        )
        ax.scatter(
            [problem.coordinates_2d[node][0] for node in traj],
            [problem.coordinates_2d[node][1] for node in traj],
            s=10,
            color=[colors_nodes(problem.clusters_dict[node]) for node in traj],
        )
    ax.legend()
    return fig, ax
