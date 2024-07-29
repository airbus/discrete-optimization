#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

import matplotlib.pyplot as plt

from discrete_optimization.vrp.vrp_model import VrpProblem2D, VrpSolution


def plot_vrp_solution(
    vrp_model: VrpProblem2D, solution: VrpSolution, ax: Any = None
) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 5))
    for vehicle in range(vrp_model.vehicle_count):
        ax.plot(
            [
                vrp_model.customers[n].x
                for n in [vrp_model.start_indexes[vehicle]]
                + solution.list_paths[vehicle]
                + [vrp_model.end_indexes[vehicle]]
            ],
            [
                vrp_model.customers[n].y
                for n in [vrp_model.start_indexes[vehicle]]
                + solution.list_paths[vehicle]
                + [vrp_model.end_indexes[vehicle]]
            ],
        )
