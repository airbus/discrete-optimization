#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
Utility functions for RC-ALBP-Problem
Provides helpers for:
- Creating problem instances (from rcpsp)
- Visualizing solutions
- Computing lower bounds
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from discrete_optimization.alb.base.problem import ResourceTaskData
from discrete_optimization.alb.rcalbp.problem import (
    RCALBPProblem,
    RCALBPSolution,
)
from discrete_optimization.datasets import fetch_data_from_psplib
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem


def _compute_task_lanes(tasks_with_times):
    """
    Compute lanes (vertical stacking) for tasks to avoid visual overlap.

    Args:
        tasks_with_times: List of (task_id, start, end) tuples

    Returns:
        Tuple of (task_lanes dict, num_lanes int)
    """
    if not tasks_with_times:
        return {}, 0

    # Sort tasks by start time
    sorted_tasks = sorted(tasks_with_times, key=lambda x: x[1])

    # Track end time of last task in each lane
    lanes = []  # Each element is the end time of the last task in that lane
    task_lanes = {}  # Maps task_id to lane number

    for task_id, start, end in sorted_tasks:
        # Find first lane where this task fits (doesn't overlap)
        assigned = False
        for lane_idx, lane_end_time in enumerate(lanes):
            if start >= lane_end_time:  # No overlap
                lanes[lane_idx] = end
                task_lanes[task_id] = lane_idx
                assigned = True
                break

        if not assigned:
            # Create new lane
            lanes.append(end)
            task_lanes[task_id] = len(lanes) - 1

    return task_lanes, len(lanes)


def visualize_solution(
    problem: RCALBPProblem,
    solution: RCALBPSolution,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize the assembly line solution as a Gantt chart with stacked tasks.

    Args:
        problem: Problem instance
        solution: Solution to visualize
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    # Color map for tasks
    colors = plt.cm.Set3(range(problem.nb_tasks))
    task_colors = {t: colors[i % len(colors)] for i, t in enumerate(problem.tasks)}

    # Compute lanes for each station to handle overlapping tasks
    station_lanes = {}
    station_num_lanes = {}

    for station in problem.stations:
        station_tasks = [t for t, s in solution.task_assignment.items() if s == station]

        # Get task timing info
        tasks_with_times = []
        for task in station_tasks:
            if task in solution.task_schedule:
                start = solution.task_schedule[task]
                end = start + problem.task_times[task]
                tasks_with_times.append((task, start, end))
        task_lanes, num_lanes = _compute_task_lanes(tasks_with_times)
        station_lanes[station] = task_lanes
        station_num_lanes[station] = num_lanes

    # Calculate figure height based on total lanes
    total_lanes = sum(station_num_lanes.values())
    fig_height = max(6, total_lanes * 0.8 + 2)

    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Track y positions for stations
    y_pos = 0
    station_y_positions = {}
    station_y_centers = {}

    for station in reversed(problem.stations):
        num_lanes = station_num_lanes.get(station, 1)
        lane_height = 0.8

        station_y_positions[station] = y_pos
        station_y_centers[station] = y_pos + (num_lanes * lane_height) / 2

        station_tasks = [t for t, s in solution.task_assignment.items() if s == station]

        for task in station_tasks:
            if task not in solution.task_schedule:
                continue

            start = solution.task_schedule[task]
            duration = problem.task_times[task]

            # Get lane for this task
            lane = station_lanes[station].get(task, 0)
            y_position = y_pos + lane * lane_height

            # Draw task bar
            ax.barh(
                y_position,
                duration,
                left=start,
                height=lane_height * 0.85,
                color=task_colors[task],
                edgecolor="black",
                linewidth=1.2,
                alpha=0.8,
            )

            # Add task label
            ax.text(
                start + duration / 2,
                y_position,
                task,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        y_pos += num_lanes * lane_height + 0.3  # Add spacing between stations

    # Format plot
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Station", fontsize=12)
    ax.set_title(
        f"Assembly Line Schedule (Cycle Time: {solution.cycle_time})", fontsize=14
    )

    # Set y-axis labels at station centers
    ax.set_yticks(list(station_y_centers.values()))
    ax.set_yticklabels([f"Station {s}" for s in reversed(problem.stations)])

    ax.set_ylim(-0.5, y_pos - 0.3)
    ax.grid(axis="x", alpha=0.3)

    # Add cycle time line
    if solution.cycle_time:
        ax.axvline(
            solution.cycle_time,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Cycle Time: {solution.cycle_time}",
        )
        ax.legend()

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def visualize_interactive_flow(
    problem: RCALBPProblem,
    solution: RCALBPSolution,
):
    """
    Create an interactive visualization of assembly line flow with time slider.

    Shows:
    - Aircraft/product flow through stations over time
    - Active tasks at each station
    - Resource usage tracking per station
    - Constraint violation warnings

    Args:
        problem: Problem instance
        solution: Solution to visualize
    """
    # Validate solution first
    eval_dict = problem.evaluate(solution)
    has_violations = (
        eval_dict["penalty_precedence"] > 0
        or eval_dict["penalty_resource_station"] > 0
        or eval_dict["penalty_resource_shared"] > 0
        or eval_dict["penalty_unscheduled"] > 0
    )

    if has_violations:
        print("\n[!] WARNING: Solution has constraint violations!")
        if eval_dict["penalty_precedence"] > 0:
            print(f"   - Precedence violations: {eval_dict['penalty_precedence']}")
        if eval_dict["penalty_resource_station"] > 0:
            print(
                f"   - Station resource violations: {eval_dict['penalty_resource_station']}"
            )
        if eval_dict["penalty_resource_shared"] > 0:
            print(
                f"   - Shared resource violations: {eval_dict['penalty_resource_shared']}"
            )
        if eval_dict["penalty_unscheduled"] > 0:
            print(f"   - Unscheduled tasks: {eval_dict['penalty_unscheduled']}")
        print()

    cycle_time = solution.cycle_time
    nb_stations = len(problem.stations)
    nb_periods = 5  # Show 5 cycles

    # Create dynamic layout that adapts to number of stations
    # Calculate resource plot grid (max 3 columns)
    n_res_cols = min(3, nb_stations)
    n_res_rows = (nb_stations + n_res_cols - 1) // n_res_cols  # Ceiling division

    # Create figure with dynamic grid
    # Rows: [Gantt (tall), Resource rows (shorter), Slider (thin)]
    total_rows = 1 + n_res_rows + 1
    fig = plt.figure(figsize=(18, 10))

    # Height ratios: Gantt gets most space, resources get less, slider minimal
    height_ratios = [4] + [1] * n_res_rows + [0.25]

    gs = fig.add_gridspec(
        total_rows, n_res_cols, height_ratios=height_ratios, hspace=0.4, wspace=0.3
    )

    # Gantt chart: full width at top (spans all columns)
    ax_gantt = fig.add_subplot(gs[0, :])

    # Resource usage plots: dynamic grid for ALL stations
    ax_resources = []
    for i in range(nb_stations):
        row = 1 + i // n_res_cols
        col = i % n_res_cols
        ax = fig.add_subplot(gs[row, col])
        ax_resources.append(ax)

    # Time slider: spans all columns at bottom
    ax_slider = fig.add_subplot(gs[-1, :])

    # Prepare data structures
    # Map tasks to stations
    station_to_tasks = {s: [] for s in problem.stations}
    for task, station in solution.task_assignment.items():
        station_to_tasks[station].append(
            {
                "task": task,
                "start": solution.get_start_time_in_cycle(task),
                "end": solution.get_end_time_in_cycle(task),
                "duration": problem.task_times[task],
            }
        )
    print(station_to_tasks)
    # Sort tasks by start time
    for station in station_to_tasks:
        station_to_tasks[station].sort(key=lambda x: x["start"])

    # Color map for tasks
    colors = plt.cm.Set3(range(problem.nb_tasks))
    task_colors = {t: colors[i % len(colors)] for i, t in enumerate(problem.tasks)}

    def get_active_tasks_at_time(station, time):
        """Get tasks active at given time on station."""
        active = []
        for task_info in station_to_tasks[station]:
            if task_info["start"] <= time < task_info["end"]:
                active.append(task_info)
        return active

    def get_resource_usage_at_time(station, time):
        """Get resource usage at given time on station."""
        active_tasks = get_active_tasks_at_time(station, time)
        usage = {r: 0 for r in problem.resources}

        for task_info in active_tasks:
            task = task_info["task"]
            for resource in problem.resources:
                usage[resource] += problem.get_task_demand(task, resource)

        return usage

    def update_plot(current_time):
        """Update visualization for current time."""
        # Clear axes
        ax_gantt.clear()
        for ax in ax_resources:
            ax.clear()

        # Calculate current period and time within cycle
        period_idx = int(current_time // cycle_time)
        time_in_cycle = current_time % cycle_time

        # =====================================================================
        # Update Resource Usage Plots (ALL STATIONS)
        # =====================================================================
        for idx, ax in enumerate(ax_resources):
            station = problem.stations[idx]
            usage = get_resource_usage_at_time(station, time_in_cycle)

            # Plot bar chart
            resource_names = list(problem.resources)
            usages = [usage[r] for r in resource_names]
            capacities = [
                problem.get_station_capacity(station, r) for r in resource_names
            ]

            x = np.arange(len(resource_names))
            bars1 = ax.bar(
                x - 0.2, usages, 0.4, label="Usage", color="steelblue", alpha=0.7
            )
            bars2 = ax.bar(
                x + 0.2,
                capacities,
                0.4,
                label="Capacity",
                color="lightcoral",
                alpha=0.7,
            )

            # Highlight violations
            for i, (u, c) in enumerate(zip(usages, capacities)):
                if u > c:
                    ax.bar(i - 0.2, u, 0.4, color="red", alpha=0.9)
                    ax.text(
                        i - 0.2,
                        u + 0.1,
                        "[!]",
                        ha="center",
                        fontsize=10,
                        color="red",
                        fontweight="bold",
                    )

            ax.set_ylabel("Units", fontsize=9)
            ax.set_title(
                f"{station} - Resource Usage (t={time_in_cycle:.1f})",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(resource_names, fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, max(max(capacities) * 1.2, 1) if capacities else 1)

        # =====================================================================
        # Draw Gantt Chart with Current Time Indicator and Lane-based Stacking
        # =====================================================================

        # Compute lanes for each station
        station_task_lanes = {}
        station_num_lanes = {}

        for station in problem.stations:
            tasks_with_times = [
                (info["task"], info["start"], info["end"])
                for info in station_to_tasks[station]
            ]
            task_lanes, num_lanes = _compute_task_lanes(tasks_with_times)
            station_task_lanes[station] = task_lanes
            station_num_lanes[station] = max(num_lanes, 1)

        # Draw tasks with lane-based stacking
        y_pos = 0
        station_y_positions = {}
        station_y_centers = {}
        lane_height = 0.7

        for station in reversed(problem.stations):
            num_lanes = station_num_lanes[station]
            station_y_positions[station] = y_pos
            station_y_centers[station] = y_pos + (num_lanes * lane_height) / 2

            # Get active tasks at current time
            active_tasks_now = get_active_tasks_at_time(station, time_in_cycle)
            active_task_ids = {t["task"] for t in active_tasks_now}

            # Get tasks for this station
            tasks_in_station = station_to_tasks[station]

            for task_info in tasks_in_station:
                task = task_info["task"]
                start = task_info["start"]
                duration = task_info["duration"]

                # Get lane for this task
                lane = station_task_lanes[station].get(task, 0)
                y_position = y_pos + lane * lane_height

                # Check if task is active
                is_active = task in active_task_ids

                # Choose color and style based on activity
                if is_active:
                    color = "gold"  # Highlight active tasks
                    edge_color = "red"
                    edge_width = 2.5
                    alpha = 1.0
                else:
                    color = task_colors[task]
                    edge_color = "black"
                    edge_width = 1
                    alpha = 0.7

                # Draw task bar
                ax_gantt.barh(
                    y_position,
                    duration,
                    left=start,
                    height=lane_height * 0.85,
                    color=color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    alpha=alpha,
                )

                # Add task label
                ax_gantt.text(
                    start + duration / 2,
                    y_position,
                    task,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="black" if is_active else "black",
                )

            y_pos += num_lanes * lane_height + 0.3

        # Draw vertical line at current time in cycle (resets at each cycle)
        ax_gantt.axvline(
            time_in_cycle,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"t={time_in_cycle:.1f}",
            alpha=0.8,
            zorder=10,
        )

        # Draw cycle time boundary
        ax_gantt.axvline(
            cycle_time,
            color="darkgreen",
            linestyle="-",
            linewidth=1.5,
            alpha=0.5,
            label=f"Cycle={cycle_time}",
            zorder=10,
        )

        # Format Gantt chart
        ax_gantt.set_yticks(list(station_y_centers.values()))
        ax_gantt.set_yticklabels([s for s in reversed(problem.stations)], fontsize=10)
        ax_gantt.set_xlabel("Time within Cycle", fontsize=11)
        ax_gantt.set_title(
            f"Assembly Line Schedule - Time: {current_time:.1f} (Cycle {period_idx + 1}, t={time_in_cycle:.1f}/{cycle_time})",
            fontsize=12,
            fontweight="bold",
        )
        ax_gantt.set_xlim(0, cycle_time * 1.05)
        ax_gantt.set_ylim(-0.5, y_pos - 0.3)
        ax_gantt.grid(axis="x", alpha=0.3)
        ax_gantt.legend(fontsize=8, loc="upper right")

    # Create slider
    slider = Slider(
        ax_slider, "Time", 0, cycle_time * nb_periods, valinit=0, valstep=0.1
    )

    def update(val):
        update_plot(slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Initial plot
    update_plot(0)

    plt.tight_layout()

    # Show the interactive plot
    print(
        "  [Interactive plot ready - use the slider to explore different time periods]"
    )
    print(
        "  [Gantt chart shows active tasks (highlighted in gold) and time indicator (red line)]"
    )
    print(f"  [Resource plots show usage vs capacity for all {nb_stations} stations]")
    plt.show()


def create_from_rcpsp(
    rcpsp_problem: RcpspProblem,
    nb_stations: int = 3,
    seed: int = 42,
) -> RCALBPProblem:
    """
    Create a realistic RC-ALBP instance from an RCPSP problem.

    This converts a project scheduling problem into an assembly line balancing
    problem by:
    - Using RCPSP tasks, durations, and precedences
    - Splitting RCPSP global resources across stations
    - Using RCPSP resource consumption per task

    Args:
        rcpsp_problem: RCPSP problem instance
        nb_stations: Number of assembly line stations
        seed: Random seed for resource allocation

    Returns:
        RCALBPProblem instance
    """
    random.seed(seed)

    # Create ResourceTaskData for each RCPSP task
    tasks_data = []
    for task in rcpsp_problem.tasks_list:
        task_id = f"T{task}"
        processing_time = rcpsp_problem.mode_details[task][1]["duration"]

        # Build resource consumption dict
        resource_consumption = {}
        for r_name in rcpsp_problem.resources_list:
            consumption = rcpsp_problem.mode_details[task][1].get(r_name, 0)
            if consumption > 0:  # Only store non-zero consumption
                resource_consumption[f"R{r_name}"] = consumption

        tasks_data.append(
            ResourceTaskData(
                task_id=task_id,
                processing_time=processing_time,
                resource_consumption=resource_consumption,
            )
        )

    # Precedences: Convert RCPSP precedences
    precedences = []
    for task in rcpsp_problem.tasks_list:
        for succ in rcpsp_problem.successors.get(task, []):
            precedences.append((f"T{task}", f"T{succ}"))

    # Stations: Create station names
    stations = [f"WS{i + 1}" for i in range(nb_stations)]

    # Resources: Use RCPSP resources
    resources = [f"R{r}" for r in rcpsp_problem.resources_list]

    # Resource allocation: Create tighter constraints
    # Strategy: Find max task requirement and set station capacity slightly above it
    # This makes the problem more challenging and realistic
    station_resources = {}

    for station in stations:
        station_resources[station] = {}
        for r_idx, r_name in enumerate(rcpsp_problem.resources_list):
            # Find maximum task requirement for this resource
            max_task_requirement = max(
                rcpsp_problem.mode_details[task][1].get(r_name, 0)
                for task in rcpsp_problem.tasks_list
            )

            # Set station capacity to max_requirement + small buffer (20-40%)
            # This ensures at least one task can run, but limits parallelism
            if max_task_requirement > 0:
                buffer_factor = 1.2 + random.random() * 0.2  # 1.2 to 1.4
                station_capacity = int(np.ceil(max_task_requirement * buffer_factor))
            else:
                station_capacity = 1  # Minimal capacity if no tasks use this resource

            # Add some variation between stations (±10%)
            variation = random.uniform(0.9, 1.1)
            capacity = max(1, int(station_capacity * variation))
            station_resources[station][f"R{r_name}"] = capacity

    return RCALBPProblem(
        tasks_data=tasks_data,
        precedences=precedences,
        stations=stations,
        resources=resources,
        station_resources=station_resources,
    )


def load_rcpsp_as_albp(
    instance_name: str = "j301_1",
    nb_stations: int = 3,
    seed: int = 42,
) -> RCALBPProblem:
    """
    Load an RCPSP instance from PSPLib and convert to RC-ALBP.

    Args:
        instance_name: Name of RCPSP instance (e.g., "j301_1")
        nb_stations: Number of assembly line stations
        seed: Random seed for resource allocation

    Returns:
        RCALBPProblem instance
    """
    # Load RCPSP instance
    try:
        files = get_data_available()
    except:
        fetch_data_from_psplib()
        files = get_data_available()

    matching_files = [f for f in files if instance_name in f]
    if not matching_files:
        raise ValueError(f"No RCPSP instance found matching '{instance_name}'")

    filepath = matching_files[0]
    rcpsp_problem = parse_file(filepath)

    # Convert to RC-ALBP
    return create_from_rcpsp(rcpsp_problem, nb_stations=nb_stations, seed=seed)


def print_solution_info(problem: RCALBPProblem, solution: RCALBPSolution):
    """
    Print detailed information about a solution.

    Args:
        problem: Problem instance
        solution: Solution to analyze
    """
    print("=" * 60)
    print("SOLUTION INFORMATION")
    print("=" * 60)

    # Evaluation
    eval_dict = problem.evaluate(solution)
    print(f"\nCycle Time: {solution.cycle_time}")
    print(f"Valid: {problem.satisfy(solution)}")

    if eval_dict["penalty_precedence"] > 0:
        print(f"⚠ Precedence violations: {eval_dict['penalty_precedence']}")
    if eval_dict["penalty_resource_station"] > 0:
        print(f"⚠ Station resource violations: {eval_dict['penalty_resource_station']}")
    if eval_dict["penalty_resource_shared"] > 0:
        print(f"⚠ Shared resource violations: {eval_dict['penalty_resource_shared']}")
    if eval_dict["penalty_unscheduled"] > 0:
        print(f"⚠ Unscheduled tasks: {eval_dict['penalty_unscheduled']}")

    # Station workload
    print("\nStation Workloads:")
    for station in problem.stations:
        station_tasks = [t for t, s in solution.task_assignment.items() if s == station]
        if station_tasks:
            makespan = max(
                solution.task_schedule.get(t, 0) + problem.task_times[t]
                for t in station_tasks
            )
            total_work = sum(problem.task_times[t] for t in station_tasks)
            print(
                f"  Station {station}: {len(station_tasks)} tasks, "
                f"total work={total_work}, makespan={makespan}"
            )

            # Resource usage
            for resource in problem.resources:
                max_usage = 0
                for t in station_tasks:
                    usage = problem.get_task_demand(t, resource)
                    max_usage = max(max_usage, usage)
                capacity = problem.get_station_capacity(station, resource)
                print(
                    f"    Resource {resource}: max_usage={max_usage}, capacity={capacity}"
                )

    print("=" * 60)


def create_shared_resource_example() -> RCALBPProblem:
    """
    Create a small example with both station-specific and shared resources.

    Resources:
    - R1, R2: Station-specific (different capacity per station)
    - R_AGV: Shared mobile robot (capacity = 2 across all stations)

    Returns:
        RCALBPProblem instance with shared resources
    """
    # 3 stations
    stations = ["S1", "S2", "S3"]

    # Create tasks with resource requirements
    tasks_data = [
        ResourceTaskData(
            task_id="T1", processing_time=5, resource_consumption={"R1": 1}
        ),
        ResourceTaskData(
            task_id="T2", processing_time=3, resource_consumption={"R2": 1}
        ),
        ResourceTaskData(
            task_id="T3", processing_time=4, resource_consumption={"R1": 1, "R_AGV": 1}
        ),
        ResourceTaskData(
            task_id="T4", processing_time=6, resource_consumption={"R2": 1}
        ),
        ResourceTaskData(
            task_id="T5", processing_time=4, resource_consumption={"R1": 1, "R_AGV": 1}
        ),
        ResourceTaskData(
            task_id="T6", processing_time=3, resource_consumption={"R2": 1}
        ),
        ResourceTaskData(
            task_id="T7", processing_time=5, resource_consumption={"R1": 1, "R_AGV": 1}
        ),
        ResourceTaskData(
            task_id="T8", processing_time=2, resource_consumption={"R2": 1}
        ),
    ]

    # Precedences
    precedences = [
        ("T1", "T3"),
        ("T2", "T3"),
        ("T3", "T5"),
        ("T4", "T6"),
        ("T5", "T7"),
        ("T6", "T7"),
        ("T7", "T8"),
    ]

    # Station-specific resources (R1, R2) - R_AGV is shared, not here!
    resources = ["R1", "R2"]

    # Station capacities for station-specific resources {station: {resource: capacity}}
    station_resources = {
        "S1": {"R1": 2, "R2": 1},
        "S2": {"R1": 1, "R2": 2},
        "S3": {"R1": 2, "R2": 1},
    }

    # Shared resources and their global capacities
    shared_resources = {"R_AGV"}
    shared_resource_capacities = {
        "R_AGV": 2  # Only 2 AGVs available globally
    }

    return RCALBPProblem(
        tasks_data=tasks_data,
        precedences=precedences,
        stations=stations,
        resources=resources,
        station_resources=station_resources,
        shared_resources=shared_resources,
        shared_resource_capacities=shared_resource_capacities,
    )


def create_large_shared_resource_instance(
    base_instance_name: str, nb_stations: int, shared_ratio: float = 0.25
) -> RCALBPProblem:
    """
    Create a large instance with shared resources from an RCPSP instance.

    Args:
        base_instance_name: Name of RCPSP instance to load
        nb_stations: Number of stations
        shared_ratio: Fraction of resources that should be shared (default 0.25 = 25%)

    Returns:
        RCALBPProblem instance with both station-specific and shared resources
    """
    # Load base problem (without shared resources)
    base_problem = load_rcpsp_as_albp(base_instance_name, nb_stations)

    # Determine which resources should be shared
    all_resources = list(base_problem.resources)
    num_shared = max(1, int(len(all_resources) * shared_ratio))

    # Make the first num_shared resources shared
    shared_resource_names = set(all_resources[:num_shared])
    station_specific_names = [
        r for r in all_resources if r not in shared_resource_names
    ]

    print(
        f"Creating instance with {len(shared_resource_names)} shared resources: {shared_resource_names}"
    )
    print(f"Station-specific resources: {station_specific_names}")

    # Compute global capacities for shared resources
    # Use the sum of capacities across all stations as the global pool
    shared_resource_capacities = {}
    for resource in shared_resource_names:
        total_capacity = 0
        for station in base_problem.stations:
            capacity = base_problem.get_station_capacity(station, resource)
            total_capacity += capacity
        shared_resource_capacities[resource] = total_capacity

    # Remove shared resources from station_resources
    new_station_resources = {}
    for station in base_problem.stations:
        new_station_resources[station] = {}
        for resource in station_specific_names:
            capacity = base_problem.get_station_capacity(station, resource)
            if capacity > 0:
                new_station_resources[station][resource] = capacity

    return RCALBPProblem(
        tasks_data=base_problem.tasks_data,  # Reuse existing ResourceTaskData objects
        precedences=base_problem.precedences,
        stations=base_problem.stations,
        resources=station_specific_names,  # Only station-specific!
        station_resources=new_station_resources,
        shared_resources=shared_resource_names,
        shared_resource_capacities=shared_resource_capacities,
    )
