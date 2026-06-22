#  Copyright (c) 2024-26 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem


def transform_shop_to_rcpsp(problem: CommonShopProblem) -> RcpspProblem:
    # TODO after the transformation PR
    return None


def plot_shop_solution(solution: AnyShopSolution, title: str = "Job Shop Schedule"):
    """
    Creates a Gantt chart visualization for a JobShopSolution.

    Args:
        solution: A JobShopSolution object containing the problem and schedule.
        title: The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate a set of colors for the jobs
    colors = list(mcolors.TABLEAU_COLORS.values())

    for job_index, job_schedule in enumerate(solution.schedule):
        job_color = colors[job_index % len(colors)]
        for subjob_index, (start_time, end_time) in enumerate(job_schedule):
            mode = solution.get_mode((job_index, subjob_index))
            machine = (
                solution.problem.list_jobs[job_index]
                .subjobs[subjob_index]
                .recipes[mode]
                .machine_index
            )
            duration = end_time - start_time

            # Draw the bar for the subjob
            ax.barh(
                y=machine,
                width=duration,
                left=start_time,
                height=0.6,
                color=job_color,
                edgecolor="black",
                label=f"Job {job_index + 1}" if subjob_index == 0 else "",
            )

            # Add text label for the job and subjob index, with a smaller font size
            if duration > 0:  # Only add text if the bar is wide enough
                ax.text(
                    start_time + duration / 2,
                    machine,
                    f"J{job_index + 1}.{subjob_index + 1}",
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                    fontsize=8,
                )

    # --- Formatting the plot ---
    ax.set_yticks(range(solution.problem.n_machines))
    ax.set_yticklabels([f"Machine {i + 1}" for i in range(solution.problem.n_machines)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Create a legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    plt.tight_layout()
    plt.show()
