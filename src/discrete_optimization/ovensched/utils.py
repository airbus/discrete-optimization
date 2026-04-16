#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Utility functions for visualization and analysis of oven scheduling solutions."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from discrete_optimization.ovensched.problem import (
    OvenSchedulingSolution,
)


def plot_solution(
    solution: "OvenSchedulingSolution",
    figsize: tuple[int, int] = (16, 10),
    show_task_ids: bool = True,
    show_setup_times: bool = True,
    title: str | None = None,
) -> None:
    """Plot a Gantt chart visualization of the oven scheduling solution.

    Args:
        solution: The solution to visualize
        figsize: Figure size (width, height) in inches
        show_task_ids: Whether to show task IDs on the chart
        show_setup_times: Whether to show setup times as hatched areas
        title: Custom title for the plot (default: auto-generated)

    Raises:
        ImportError: If matplotlib is not available
    """
    problem = solution.problem
    fig, (ax_gantt, ax_kpi) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[3, 1], gridspec_kw={"hspace": 0.3}
    )

    # Get all unique attributes and create a color map
    all_attributes = problem.get_set_task_attributes()
    n_attributes = len(all_attributes)
    colors = plt.cm.Set3(np.linspace(0, 1, max(n_attributes, 3)))
    attr_to_color = {attr: colors[i] for i, attr in enumerate(sorted(all_attributes))}

    # Track y-positions for machines (reversed so machine 0 is at top)
    machine_positions = {
        m: problem.n_machines - m - 1 for m in range(problem.n_machines)
    }

    # Plot batches
    for machine in range(problem.n_machines):
        y_pos = machine_positions[machine]
        batches = solution.schedule_per_machine[machine]

        for batch_idx, batch in enumerate(batches):
            # Plot setup time if requested
            if show_setup_times and batch.start_time > 0:
                if batch_idx == 0:
                    prev_attr = problem.machines_data[machine].initial_attribute
                    prev_end = 0
                else:
                    prev_attr = batches[batch_idx - 1].task_attribute
                    prev_end = batches[batch_idx - 1].end_time

                setup_time = problem.setup_times[prev_attr][batch.task_attribute]
                if setup_time > 0:
                    # Draw setup time as hatched rectangle
                    ax_gantt.barh(
                        y_pos,
                        setup_time,
                        left=prev_end,
                        height=0.8,
                        color="white",
                        edgecolor="black",
                        hatch="///",
                        alpha=0.7,
                        label="Setup" if machine == 0 and batch_idx == 0 else "",
                    )

            # Plot batch
            color = attr_to_color[batch.task_attribute]
            duration = batch.end_time - batch.start_time

            ax_gantt.barh(
                y_pos,
                duration,
                left=batch.start_time,
                height=0.8,
                color=color,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add task IDs if requested
            if show_task_ids:
                task_text = ",".join(str(t) for t in sorted(batch.tasks))
                if len(task_text) > 20:
                    task_text = f"{len(batch.tasks)} tasks"

                ax_gantt.text(
                    batch.start_time + duration / 2,
                    y_pos,
                    task_text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

    # Configure Gantt chart axis
    ax_gantt.set_yticks(list(machine_positions.values()))
    ax_gantt.set_yticklabels([f"Machine {m}" for m in range(problem.n_machines)])
    ax_gantt.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax_gantt.set_ylabel("Machine", fontsize=12, fontweight="bold")
    ax_gantt.grid(True, axis="x", alpha=0.3, linestyle="--")

    # Create legend for attributes
    legend_patches = [
        mpatches.Patch(color=attr_to_color[attr], label=f"Attribute {attr}")
        for attr in sorted(all_attributes)
    ]
    if show_setup_times:
        legend_patches.append(
            mpatches.Patch(
                facecolor="white", edgecolor="black", hatch="///", label="Setup Time"
            )
        )
    ax_gantt.legend(handles=legend_patches, loc="upper right", fontsize=9)

    # Set title
    if title is None:
        title = f"Oven Scheduling Solution - {problem.n_jobs} jobs on {problem.n_machines} machines"
    ax_gantt.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Plot KPIs in the bottom subplot
    evaluation = problem.evaluate(solution)
    kpi_names = ["Processing\nTime", "Late\nJobs", "Setup\nCost"]
    kpi_values = [
        evaluation["processing_time"],
        evaluation["nb_late_jobs"],
        evaluation["setup_cost"],
    ]

    # Create bar chart for KPIs
    bars = ax_kpi.bar(
        kpi_names, kpi_values, color=["#3498db", "#e74c3c", "#f39c12"], alpha=0.7
    )

    # Add value labels on bars
    for bar, value in zip(bars, kpi_values):
        height = bar.get_height()
        ax_kpi.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax_kpi.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax_kpi.set_title("Key Performance Indicators", fontsize=12, fontweight="bold")
    ax_kpi.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Add feasibility check
    is_feasible = problem.satisfy(solution)
    feasibility_text = "✓ Feasible" if is_feasible else "✗ Infeasible"
    feasibility_color = "green" if is_feasible else "red"

    fig.text(
        0.98,
        0.98,
        feasibility_text,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=feasibility_color,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor=feasibility_color,
            linewidth=2,
        ),
    )

    plt.tight_layout()
    plt.show()


def plot_machine_utilization(
    solution: "OvenSchedulingSolution", figsize: tuple[int, int] = (12, 6)
) -> None:
    """Plot machine utilization statistics.

    Args:
        solution: The solution to analyze
        figsize: Figure size (width, height) in inches

    Raises:
        ImportError: If matplotlib is not available
    """
    problem = solution.problem
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Calculate metrics per machine
    n_batches = []
    processing_times = []
    setup_times = []
    setup_costs = []

    for machine in range(problem.n_machines):
        batches = solution.schedule_per_machine[machine]
        n_batches.append(len(batches))

        # Calculate processing time (actual work)
        proc_time = sum(batch.end_time - batch.start_time for batch in batches)
        processing_times.append(proc_time)

        # Calculate total setup time and cost
        total_setup_time = 0
        total_setup_cost = 0

        for i, batch in enumerate(batches):
            if i == 0:
                prev_attr = problem.machines_data[machine].initial_attribute
            else:
                prev_attr = batches[i - 1].task_attribute

            total_setup_time += problem.setup_times[prev_attr][batch.task_attribute]
            total_setup_cost += problem.setup_costs[prev_attr][batch.task_attribute]

        setup_times.append(total_setup_time)
        setup_costs.append(total_setup_cost)

    machines = [f"M{m}" for m in range(problem.n_machines)]

    # Plot 1: Number of batches per machine
    axes[0].bar(machines, n_batches, color="#3498db", alpha=0.7)
    axes[0].set_title("Batches per Machine", fontweight="bold")
    axes[0].set_ylabel("Number of Batches")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Add value labels
    for i, v in enumerate(n_batches):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    # Plot 2: Processing vs Setup Time
    x = np.arange(len(machines))
    width = 0.35

    axes[1].bar(
        x - width / 2,
        processing_times,
        width,
        label="Processing",
        color="#2ecc71",
        alpha=0.7,
    )
    axes[1].bar(
        x + width / 2, setup_times, width, label="Setup", color="#e74c3c", alpha=0.7
    )
    axes[1].set_title("Time Distribution", fontweight="bold")
    axes[1].set_ylabel("Time")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(machines)
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    # Plot 3: Setup costs
    axes[2].bar(machines, setup_costs, color="#f39c12", alpha=0.7)
    axes[2].set_title("Setup Cost per Machine", fontweight="bold")
    axes[2].set_ylabel("Setup Cost")
    axes[2].grid(True, axis="y", alpha=0.3)

    # Add value labels
    for i, v in enumerate(setup_costs):
        if v > 0:
            axes[2].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    plt.suptitle(
        "Machine Utilization Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_attribute_distribution(
    solution: "OvenSchedulingSolution", figsize: tuple[int, int] = (10, 6)
) -> None:
    """Plot distribution of task attributes across batches.

    Args:
        solution: The solution to analyze
        figsize: Figure size (width, height) in inches

    Raises:
        ImportError: If matplotlib is not available
    """
    problem = solution.problem

    # Count batches and tasks per attribute
    from collections import defaultdict

    batches_per_attr = defaultdict(int)
    tasks_per_attr = defaultdict(int)

    for machine in range(problem.n_machines):
        for batch in solution.schedule_per_machine[machine]:
            batches_per_attr[batch.task_attribute] += 1
            tasks_per_attr[batch.task_attribute] += len(batch.tasks)

    attributes = sorted(batches_per_attr.keys())
    batch_counts = [batches_per_attr[attr] for attr in attributes]
    task_counts = [tasks_per_attr[attr] for attr in attributes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot batches per attribute
    ax1.bar([f"Attr {a}" for a in attributes], batch_counts, color="#9b59b6", alpha=0.7)
    ax1.set_title("Batches per Attribute", fontweight="bold")
    ax1.set_ylabel("Number of Batches")
    ax1.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(batch_counts):
        ax1.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    # Plot tasks per attribute
    ax2.bar([f"Attr {a}" for a in attributes], task_counts, color="#1abc9c", alpha=0.7)
    ax2.set_title("Tasks per Attribute", fontweight="bold")
    ax2.set_ylabel("Number of Tasks")
    ax2.grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(task_counts):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Attribute Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
