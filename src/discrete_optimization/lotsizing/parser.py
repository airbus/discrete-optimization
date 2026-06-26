#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.lotsizing.problem import LotSizingProblem


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for knapsack.

    Params:
        data_folder: folder where datasets for knapsack whould be find.
            If None, we look in "knapsack" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/lotsizing"

    try:
        instances_files = None
        folders = os.listdir(data_folder)
        for f in folders:
            folder_path = os.path.join(data_folder, f)
            if os.path.isdir(folder_path):
                instances_files = [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if "psp" in f
                ]
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)
    return instances_files


def parse_input_data(input_data: str) -> LotSizingProblem:
    """Parse lot sizing problem from string data.

    Format:
        nbPeriods
        nbItems
        demands (nbItems lines of nbPeriods boolean integers)
        stocking cost h
        [empty line]
        transition costs (nbItems lines of nbItems integers)
        [empty line]
        optimal cost (or bounds)
    """
    lines = input_data.strip().split("\n")

    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]

    line_idx = 0

    # Parse nbPeriods
    nb_periods = int(lines[line_idx])
    line_idx += 1

    # Parse nbItems
    nb_items = int(lines[line_idx])
    line_idx += 1

    # Parse demands (nbItems lines)
    demands = []
    for i in range(nb_items):
        demand_line = lines[line_idx].split()
        demands.append([int(x) for x in demand_line])
        line_idx += 1

    # Parse stocking cost
    stocking_cost = int(lines[line_idx])
    line_idx += 1

    # Parse transition costs (changeover costs matrix)
    changeover_costs = []
    for i in range(nb_items):
        cost_line = lines[line_idx].split()
        changeover_costs.append([int(x) for x in cost_line])
        line_idx += 1

    # Optional: parse optimal cost (last line if present)
    optimal_cost = None
    if line_idx < len(lines):
        try:
            optimal_cost = int(lines[line_idx])
        except ValueError:
            pass

    # Create the problem
    # Since demands are binary (0 or 1), capacity is 1
    capacity_machine = 1

    # Stock capacity: sum of all demands should be enough
    total_demand = sum(sum(d) for d in demands)
    stock_capacity = total_demand

    # All items have the same stocking cost
    stock_cost_per_type = [stocking_cost] * nb_items

    # No delay costs in this formulation (or set to high penalty)
    delay_cost_per_type = [100000] * nb_items  # High penalty for delays

    problem = LotSizingProblem(
        nb_items_type=nb_items,
        capacity_machine=capacity_machine,
        changeover_costs=changeover_costs,
        demands=demands,
        stock_capacity=stock_capacity,
        stock_cost_per_type_per_time_per_unit=stock_cost_per_type,
        delay_cost_per_type_per_time_per_unit=delay_cost_per_type,
        allow_delays=False,
        known_bound=optimal_cost,
    )

    return problem


def parse_file(file_path: str) -> LotSizingProblem:
    with open(file_path, "r", encoding="utf-8") as f:
        input_data = f.read()
        problem = parse_input_data(input_data)
        return problem
