#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from discrete_optimization.binpack.problem import BinPackProblem, ItemBinPack
from discrete_optimization.datasets import get_data_home


def get_data_available_bppc(
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
        data_folder = f"{data_home}/bppc"

    try:
        datasets = [
            os.path.abspath(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
            if "txt" in f
        ]
    except FileNotFoundError:
        datasets = []
    return datasets


def parse_bin_packing_constraint_file(file_path):
    """
    Parses a bin packing instance file with incompatibility constraints.
    (Bin Packing Problem with Conflicts benchmarks from here :
    https://site.unibo.it/operations-research/en/research/library-of-codes-and-instances-1)
    """
    N = 0
    C = 0
    items = []
    incompatible_items = set()
    with open(file_path, "r") as f:
        # Read the first line for N and C
        first_line = f.readline().strip().split()
        if len(first_line) == 2:
            N = int(first_line[0])
            C = int(first_line[1])
        else:
            raise ValueError("Invalid format: First line must contain N and C.")

        # Read the remaining lines for item details
        for line in f:
            parts = line.strip().split()
            if not parts:  # Skip empty lines
                continue

            if len(parts) >= 2:
                item_id = int(parts[0]) - 1
                weight = int(parts[1])
                incompatible = [int(x) - 1 for x in parts[2:]]
                for inc in incompatible:
                    incompatible_items.add((item_id, inc))
                items.append(ItemBinPack(index=item_id, weight=weight))
            else:
                raise ValueError(
                    f"Invalid format: Item line '{line.strip()}' must have at least item ID and weight."
                )

    # Basic validation: Check if the number of parsed items matches N
    if len(items) != N:
        print(
            f"Warning: Number of items parsed ({len(items)}) does not match N ({N}) from the header."
        )

    return BinPackProblem(
        list_items=items, capacity_bin=C, incompatible_items=incompatible_items
    )
