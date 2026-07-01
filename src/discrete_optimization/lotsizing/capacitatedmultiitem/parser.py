#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Parser for capacitated multi-item lot sizing problem instances."""

import os
from typing import Optional

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
)

try:
    import pymzn
except ImportError:
    pymzn = None


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for lot sizing.

    Args:
        data_folder: folder where datasets for lot sizing should be found.
            If None, we look in "lotsizing" subdirectory of `data_home`.
        data_home: root directory for all datasets. If None, set by
            default to "~/discrete_optimization_data"

    Returns:
        List of available instance file paths
    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/lotsizing"

    try:
        instances_files = []
        folders = os.listdir(data_folder)
        for f in folders:
            folder_path = os.path.join(data_folder, f)
            if os.path.isdir(folder_path):
                instances_files += [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if "psp" in f or "dzn" in f
                ]
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)
    return instances_files


def parse_input_data(input_data: str) -> CapacitatedMultiItemLSP:
    """Parse lot sizing problem from string data (txt format).

    Format:
        nbPeriods
        nbItems
        demands (nbItems lines of nbPeriods boolean integers)
        stocking cost h
        [empty line]
        transition costs (nbItems lines of nbItems integers)
        [empty line]
        optimal cost (or bounds)

    Args:
        input_data: String containing the problem data

    Returns:
        CapacitatedMultiItemLSP instance
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

    # Since demands are typically binary (0 or 1), capacity is 1
    capacity_machine = 1

    # All items have the same stocking cost
    stock_cost_per_type = [float(stocking_cost)] * nb_items

    # High penalty for delays (same as old implementation)
    delay_cost_per_type = [100000.0] * nb_items

    # Create the problem
    problem = CapacitatedMultiItemLSP(
        nb_items=nb_items,
        horizon=nb_periods,
        demands=demands,
        capacity_machine=capacity_machine,
        changeover_costs=changeover_costs,
        stock_cost_per_type=stock_cost_per_type,
        stock_capacity=None,  # Will be set to sum of demands
        allow_delays=False,  # Hard constraint: no backlog allowed
        delay_cost_per_type=delay_cost_per_type,  # But high penalty in objective
        known_bound=optimal_cost,
    )

    return problem


def parse_dzn_file(file_path: str) -> CapacitatedMultiItemLSP:
    """Parse lot sizing problem from .dzn MiniZinc data file.

    Expected format:
        Periods = <int>;
        Items = <int>;
        Demands = [|...|];  % Items x Periods matrix
        StockingCosts = [<int>, ...];  % per item
        SetupCosts = [|...|];  % Items x Items matrix (changeover costs)

    Args:
        file_path: Path to .dzn file

    Returns:
        CapacitatedMultiItemLSP instance
    """
    if pymzn is None:
        raise ImportError(
            "pymzn is required to parse .dzn files. Install it with: pip install pymzn"
        )

    # Parse the .dzn file
    data = pymzn.dzn2dict(file_path)

    # Extract data
    nb_periods = data["Periods"]
    nb_items = data["Items"]

    # pymzn.dzn2dict flattens 2D arrays, so we need to reshape them
    # Demands is stored as a flat list (Items x Periods) in row-major order
    demands_flat = data["Demands"]
    demands = [
        demands_flat[i * nb_periods : (i + 1) * nb_periods] for i in range(nb_items)
    ]

    # StockingCosts is a list per item
    stocking_costs = [float(x) for x in data["StockingCosts"]]

    # SetupCosts is stored as a flat list (Items x Items) in row-major order
    setup_flat = data["SetupCosts"]
    changeover_costs = [
        setup_flat[i * nb_items : (i + 1) * nb_items] for i in range(nb_items)
    ]

    # Create the problem
    capacity_machine = 1

    # High penalty for delays
    delay_cost_per_type = [100000.0] * nb_items

    problem = CapacitatedMultiItemLSP(
        nb_items=nb_items,
        horizon=nb_periods,
        demands=demands,
        capacity_machine=capacity_machine,
        changeover_costs=changeover_costs,
        stock_cost_per_type=stocking_costs,
        stock_capacity=None,  # Will be set to sum of demands
        allow_delays=False,  # Hard constraint: no backlog allowed
        delay_cost_per_type=delay_cost_per_type,  # But high penalty in objective
    )

    return problem


def parse_file(file_path: str) -> CapacitatedMultiItemLSP:
    """Parse lot sizing problem from file.

    Automatically detects file format based on extension:
    - .txt: plain text format
    - .dzn: MiniZinc data format

    Args:
        file_path: Path to problem file

    Returns:
        CapacitatedMultiItemLSP instance
    """
    _, ext = os.path.splitext(file_path)

    if ext.lower() == ".dzn":
        return parse_dzn_file(file_path)
    else:
        # Default to txt format
        with open(file_path, "r", encoding="utf-8") as f:
            input_data = f.read()
            return parse_input_data(input_data)
