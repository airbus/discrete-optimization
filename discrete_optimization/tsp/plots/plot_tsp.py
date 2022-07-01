import os

import matplotlib.pyplot as plt

from discrete_optimization.tsp.tsp_model import (
    SolutionTSP,
    TSPModel,
    TSPModel2D,
    TSPModelDistanceMatrix,
)


def plot_tsp_solution(tsp_model: TSPModel, solution: SolutionTSP, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.plot(
        [
            tsp_model.list_points[n].x
            for n in [tsp_model.start_index]
            + solution.permutation
            + [tsp_model.end_index]
        ],
        [
            tsp_model.list_points[n].y
            for n in [tsp_model.start_index]
            + solution.permutation
            + [tsp_model.end_index]
        ],
        color="orange",
    )
