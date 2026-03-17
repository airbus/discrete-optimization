#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from discrete_optimization.rcalbp_l.problem import RCALBPLProblem, RCALBPLSolution


def plot_rcalbpl_dashboard(problem: RCALBPLProblem, solution: RCALBPLSolution):
    """
    Creates an interactive matplotlib dashboard to visualize RC-ALBP/L solutions.
    - Top plot: Gantt chart of the assembly line for a selected period.
    - Bottom plot: Evolution of the Cycle Times (Target, Chosen, Real) across all periods.
    """
    # 1. Prepare global data for the Cycle Time evolution chart
    periods = problem.periods
    chosen_cycs = [solution.cyc[p] for p in periods]
    real_cycs = []

    for p in periods:
        max_end = 0
        for t in problem.tasks:
            w = solution.wks[t]
            dur = problem.get_duration(t, p, w)
            if dur > 0:
                end_t = solution.start.get((t, p), 0) + dur
                if end_t > max_end:
                    max_end = end_t
        real_cycs.append(max_end)

    target_cycs = [problem.c_target] * len(periods)

    # 2. Setup Figure and Grid
    fig, (ax_gantt, ax_line) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2.5, 1]}
    )
    plt.subplots_adjust(bottom=0.15, hspace=0.35)

    # 3. Create a stable color map for tasks
    cmap = plt.get_cmap("tab20")
    task_colors = {t: cmap(i % 20) for i, t in enumerate(problem.tasks)}

    # --- BOTTOM PLOT: Cycle Time Evolution ---
    ax_line.plot(
        periods,
        chosen_cycs,
        color="red",
        linestyle="-",
        label="Chosen Cycle Time",
        linewidth=2,
    )
    ax_line.plot(
        periods,
        real_cycs,
        color="blue",
        linestyle=":",
        label="Real Cycle Time",
        linewidth=2,
    )
    ax_line.plot(
        periods,
        target_cycs,
        color="green",
        linestyle="--",
        label="Target Cycle Time",
        linewidth=2,
    )

    # Highlight the boundary between unstable (fill-up) and stable periods [cite: 170, 171]
    boundary = problem.nb_stations - 0.5
    ax_line.axvline(
        boundary, color="grey", linestyle="-.", label="Stable Period Boundary"
    )
    ax_line.text(
        boundary,
        max(chosen_cycs) * 0.9,
        " Ramp-up",
        color="grey",
        verticalalignment="top",
    )

    # Marker for the currently selected period
    vline_current_period = ax_line.axvline(
        0, color="orange", linewidth=4, alpha=0.5, label="Current Period"
    )

    ax_line.set_xlabel("Periods")
    ax_line.set_ylabel("Time")
    ax_line.set_title("Cycle Time Evolution (Learning Curve Effect)")
    ax_line.legend(loc="upper right")
    ax_line.grid(True, linestyle="--", alpha=0.6)
    ax_line.set_xticks(periods)
    if len(periods) > 20:
        ax_line.set_xticks(
            periods[::5]
        )  # Clean up x-ticks if there are too many periods

    # --- TOP PLOT: Interactive Gantt Chart ---
    def draw_gantt(p: int):
        ax_gantt.clear()

        stations = problem.stations
        ax_gantt.set_yticks(stations)
        ax_gantt.set_yticklabels([f"WS {w}" for w in stations])
        ax_gantt.set_ylim(min(stations) - 0.5, max(stations) + 0.5)

        # Compute maximum time limit to anchor the X-axis across all frames
        ax_gantt.set_xlim(0, max(max(chosen_cycs), max(real_cycs)) * 1.05)
        ax_gantt.set_xlabel("Time")
        ax_gantt.set_ylabel("Workstations")

        period_type = "Unstable (Fill-up)" if p < problem.nb_stations else "Stable"
        ax_gantt.set_title(f"Assembly Line Schedule | Period: {p} ({period_type})")

        # Plot tasks as horizontal bars
        for t in problem.tasks:
            w = solution.wks[t]
            start_t = solution.start.get((t, p), 0)
            dur_t = problem.get_duration(t, p, w)

            if dur_t > 0:  # Task is active
                ax_gantt.barh(
                    w,
                    dur_t,
                    left=start_t,
                    color=task_colors[t],
                    edgecolor="black",
                    height=0.5,
                    alpha=0.8,
                )

                # Center text inside the bar
                text_color = (
                    "white"
                    if np.mean(mcolors.to_rgb(task_colors[t])[:3]) < 0.5
                    else "black"
                )
                ax_gantt.text(
                    start_t + dur_t / 2,
                    w,
                    f"T{t}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                )

        # Plot vertical limit lines
        ax_gantt.axvline(
            problem.c_target,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Target ({problem.c_target})",
        )
        ax_gantt.axvline(
            solution.cyc[p],
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Chosen ({solution.cyc[p]})",
        )
        ax_gantt.axvline(
            real_cycs[p],
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"Real ({real_cycs[p]})",
        )

        ax_gantt.legend(loc="upper right")
        ax_gantt.grid(axis="x", linestyle="--", alpha=0.5)

    # Initial draw for period 0
    draw_gantt(periods[0])

    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.03], facecolor="lightgray")
    slider = Slider(
        ax=ax_slider,
        label="Select Period",
        valmin=min(periods),
        valmax=max(periods),
        valinit=min(periods),
        valstep=1,
    )

    # Update function called when slider moves
    def update(val):
        p = int(slider.val)
        draw_gantt(p)
        vline_current_period.set_xdata([p])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    # Return the slider object to prevent it from being garbage collected by Python
    plt.show()
    return fig, slider
