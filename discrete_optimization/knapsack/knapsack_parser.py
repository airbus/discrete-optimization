#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional

from discrete_optimization.datasets import get_data_home
from discrete_optimization.knapsack.knapsack_model import Item, KnapsackModel


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> List[str]:
    """Get datasets available for knapsack.

    Params:
        data_folder: folder where datasets for knapsack whould be find.
            If None, we look in "knapsack" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/knapsack"

    try:
        datasets = [
            os.path.abspath(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
        ]
    except FileNotFoundError:
        datasets = []
    return datasets


def parse_input_data(
    input_data: str, force_recompute_values: bool = False
) -> KnapsackModel:
    """
    Parse a string of the following form :
    item_count max_capacity
    item1_value item1_weight
    ...
    itemN_value itemN_weight
    """
    lines = input_data.split("\n")
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    items = []
    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))
    return KnapsackModel(
        list_items=items,
        max_capacity=capacity,
        force_recompute_values=force_recompute_values,
    )


def parse_file(file_path: str, force_recompute_values: bool = False) -> KnapsackModel:
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        knapsack_model = parse_input_data(
            input_data, force_recompute_values=force_recompute_values
        )
        return knapsack_model
