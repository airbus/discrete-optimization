#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Visualization utilities for lot sizing solutions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from discrete_optimization.lotsizing.generic_lotsizing import (
        GenericLotSizingProblem,
        GenericLotSizingSolution,
    )


def plot_production_schedule(
    problem: GenericLotSizingProblem,
    solution: GenericLotSizingSolution,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Plot production schedule with capacity constraints.

    Shows stacked bar chart of:
    - Production quantities per item per period
    - Setup times (if applicable)
    - Available capacity as reference line

    Args:
        problem: The lot sizing problem
        solution: The solution to visualize
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    horizon = problem.horizon
    items = problem.items_list
    n_items = len(items)

    # Prepare data
    production_data = np.zeros((n_items, horizon))
    setup_data = np.zeros((n_items, horizon))

    for idx, item in enumerate(items):
        for t in range(horizon):
            qty = solution.get_production_quantity(item, t)
            production_data[idx, t] = qty

            # Setup time (if production occurs)
            if solution.has_setup(item, t):
                setup_data[idx, t] = problem.get_setup_time(item, t)

    # Plot stacked bars
    periods = np.arange(horizon)
    width = 0.8

    # Color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, n_items))

    # Plot production quantities
    bottom_prod = np.zeros(horizon)
    for idx, item in enumerate(items):
        ax.bar(
            periods,
            production_data[idx],
            width,
            bottom=bottom_prod,
            label=f"Item {item} (production)",
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
        )
        bottom_prod += production_data[idx]

    # Plot setup times on top
    bottom_setup = bottom_prod.copy()
    for idx, item in enumerate(items):
        if np.any(setup_data[idx] > 0):
            ax.bar(
                periods,
                setup_data[idx],
                width,
                bottom=bottom_setup,
                label=f"Item {item} (setup time)",
                color=colors[idx],
                alpha=0.4,
                edgecolor="red",
                linewidth=1.5,
                hatch="///",
            )
            bottom_setup += setup_data[idx]

    # Plot capacity limit
    capacity = [problem.get_available_production_time(t) for t in range(horizon)]
    ax.plot(
        periods,
        capacity,
        "r--",
        linewidth=2,
        label="Capacity limit",
        marker="o",
        markersize=4,
    )

    ax.set_xlabel("Period", fontsize=12)
    ax.set_ylabel("Production time / Capacity", fontsize=12)
    ax.set_title("Production Schedule with Setup Times", fontsize=14, fontweight="bold")
    ax.set_xticks(periods)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_inventory_and_costs(
    problem: GenericLotSizingProblem,
    solution: GenericLotSizingSolution,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Plot inventory levels and cumulated costs over time.

    Creates a 2x2 subplot grid showing:
    1. Inventory levels per item over time
    2. Cumulated inventory cost over time
    3. Backlog quantities per item over time
    4. Cumulated costs (inventory + backlog + changeover) over time

    Args:
        problem: The lot sizing problem
        solution: The solution to visualize
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    horizon = problem.horizon
    items = problem.items_list
    periods = np.arange(horizon)
    colors = plt.cm.Set3(np.linspace(0, 1, len(items)))

    # 1. Inventory levels
    for idx, item in enumerate(items):
        inventory = [solution.get_inventory_level(item, t) for t in range(horizon)]
        ax1.plot(
            periods,
            inventory,
            marker="o",
            linewidth=2,
            color=colors[idx],
            label=f"Item {item}",
        )
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Inventory level")
    ax1.set_title("Inventory Levels per Item", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Cumulated inventory cost (use solution method)
    cost_evolution = solution.get_cost_evolution()
    cumulated_inv_cost = cost_evolution["inventory"]

    ax2.plot(periods, cumulated_inv_cost, marker="o", linewidth=2, color="green")
    ax2.fill_between(periods, cumulated_inv_cost, alpha=0.3, color="green")
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Cumulated cost")
    ax2.set_title("Cumulated Inventory Cost", fontweight="bold")
    ax2.grid(alpha=0.3)

    # 3. Backlog quantities
    for idx, item in enumerate(items):
        backlog = [solution.get_backlog_quantity(item, t) for t in range(horizon)]
        if any(b > 0 for b in backlog):
            ax3.plot(
                periods,
                backlog,
                marker="x",
                linewidth=2,
                linestyle="--",
                color=colors[idx],
                label=f"Item {item}",
            )
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Backlog quantity")
    ax3.set_title("Backlog per Item", fontweight="bold")
    if ax3.get_legend_handles_labels()[0]:  # Only show legend if there are items
        ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Cumulated total costs (use solution method)
    cumulated_total_cost = cost_evolution["total"]

    ax4.plot(periods, cumulated_total_cost, marker="o", linewidth=2, color="red")
    ax4.fill_between(periods, cumulated_total_cost, alpha=0.3, color="red")
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Cumulated total cost")
    ax4.set_title("Cumulated Total Cost (All Components)", fontweight="bold")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_solution_summary(
    problem: GenericLotSizingProblem,
    solution: GenericLotSizingSolution,
    figsize: tuple[float, float] = (16, 12),
) -> plt.Figure:
    """Create comprehensive visualization of lot sizing solution.

    Combines production schedule and cost tracking in one figure.

    Args:
        problem: The lot sizing problem
        solution: The solution to visualize
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with 3 subplots
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    horizon = problem.horizon
    items = problem.items_list
    n_items = len(items)
    periods = np.arange(horizon)
    colors = plt.cm.Set3(np.linspace(0, 1, n_items))

    # Top: Production schedule (spans 2 columns)
    ax_prod = fig.add_subplot(gs[0, :])

    # Prepare production and setup data
    production_data = np.zeros((n_items, horizon))
    setup_data = np.zeros((n_items, horizon))

    for idx, item in enumerate(items):
        for t in range(horizon):
            qty = solution.get_production_quantity(item, t)
            production_data[idx, t] = qty
            if solution.has_setup(item, t):
                setup_data[idx, t] = problem.get_setup_time(item, t)

    # Stacked bars
    width = 0.8
    bottom_prod = np.zeros(horizon)
    for idx, item in enumerate(items):
        ax_prod.bar(
            periods,
            production_data[idx],
            width,
            bottom=bottom_prod,
            label=f"Item {item}",
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
        )
        bottom_prod += production_data[idx]

    # Setup times
    bottom_setup = bottom_prod.copy()
    for idx, item in enumerate(items):
        if np.any(setup_data[idx] > 0):
            ax_prod.bar(
                periods,
                setup_data[idx],
                width,
                bottom=bottom_setup,
                color=colors[idx],
                alpha=0.3,
                edgecolor="red",
                linewidth=1,
                hatch="///",
            )
            bottom_setup += setup_data[idx]

    # Capacity limit
    capacity = [problem.get_available_production_time(t) for t in range(horizon)]
    ax_prod.plot(
        periods,
        capacity,
        "r--",
        linewidth=2,
        label="Capacity",
        marker="o",
        markersize=4,
    )
    ax_prod.set_xlabel("Period")
    ax_prod.set_ylabel("Production + Setup Time")
    ax_prod.set_title("Production Schedule", fontweight="bold", fontsize=14)
    ax_prod.set_xticks(periods)
    ax_prod.legend(loc="upper left", fontsize=9)
    ax_prod.grid(axis="y", alpha=0.3)

    # Middle left: Inventory levels
    ax_inv = fig.add_subplot(gs[1, 0])
    for idx, item in enumerate(items):
        inventory = [solution.get_inventory_level(item, t) for t in range(horizon)]
        ax_inv.plot(
            periods,
            inventory,
            marker="o",
            linewidth=2,
            color=colors[idx],
            label=f"Item {item}",
        )
    ax_inv.set_xlabel("Period")
    ax_inv.set_ylabel("Inventory level")
    ax_inv.set_title("Inventory Levels", fontweight="bold")
    ax_inv.legend(fontsize=9)
    ax_inv.grid(alpha=0.3)

    # Middle right: Backlog
    ax_backlog = fig.add_subplot(gs[1, 1])
    has_backlog = False
    for idx, item in enumerate(items):
        backlog = [solution.get_backlog_quantity(item, t) for t in range(horizon)]
        if any(b > 0 for b in backlog):
            has_backlog = True
            ax_backlog.plot(
                periods,
                backlog,
                marker="x",
                linewidth=2,
                linestyle="--",
                color=colors[idx],
                label=f"Item {item}",
            )
    ax_backlog.set_xlabel("Period")
    ax_backlog.set_ylabel("Backlog quantity")
    ax_backlog.set_title("Backlog/Delays", fontweight="bold")
    if has_backlog:
        ax_backlog.legend(fontsize=9)
    else:
        ax_backlog.text(
            0.5,
            0.5,
            "No backlog",
            ha="center",
            va="center",
            transform=ax_backlog.transAxes,
            fontsize=12,
            alpha=0.5,
        )
    ax_backlog.grid(alpha=0.3)

    # Bottom: Cumulated costs (spans 2 columns)
    ax_cost = fig.add_subplot(gs[2, :])

    # Get cumulated costs from solution method
    cost_evolution = solution.get_cost_evolution()
    costs_inv = cost_evolution["inventory"]
    costs_backlog = cost_evolution["backlog"]
    costs_changeover = cost_evolution["changeover"]
    costs_setup = cost_evolution["setup"]
    costs_production = cost_evolution["production"]

    # Stacked area plot
    ax_cost.fill_between(
        periods, 0, costs_production, alpha=0.6, label="Production cost"
    )
    ax_cost.fill_between(
        periods,
        costs_production,
        np.array(costs_production) + np.array(costs_setup),
        alpha=0.6,
        label="Setup cost",
    )
    ax_cost.fill_between(
        periods,
        np.array(costs_production) + np.array(costs_setup),
        np.array(costs_production) + np.array(costs_setup) + np.array(costs_inv),
        alpha=0.6,
        label="Inventory cost",
    )
    ax_cost.fill_between(
        periods,
        np.array(costs_production) + np.array(costs_setup) + np.array(costs_inv),
        np.array(costs_production)
        + np.array(costs_setup)
        + np.array(costs_inv)
        + np.array(costs_changeover),
        alpha=0.6,
        label="Changeover cost",
    )
    ax_cost.fill_between(
        periods,
        np.array(costs_production)
        + np.array(costs_setup)
        + np.array(costs_inv)
        + np.array(costs_changeover),
        np.array(costs_production)
        + np.array(costs_setup)
        + np.array(costs_inv)
        + np.array(costs_changeover)
        + np.array(costs_backlog),
        alpha=0.6,
        label="Backlog cost",
        color="red",
    )

    ax_cost.set_xlabel("Period")
    ax_cost.set_ylabel("Cumulated cost")
    ax_cost.set_title("Cumulated Costs Over Time", fontweight="bold", fontsize=14)
    ax_cost.legend(loc="upper left", fontsize=9)
    ax_cost.grid(alpha=0.3)

    # Overall title
    cost_dict = problem.evaluate(solution)
    total_cost = sum(cost_dict.values())
    fig.suptitle(
        f"Lot Sizing Solution Summary - Total Cost: {total_cost:.2f}",
        fontsize=16,
        fontweight="bold",
    )

    return fig
