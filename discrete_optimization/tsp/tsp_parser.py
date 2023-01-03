#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional

from discrete_optimization.datasets import get_data_home
from discrete_optimization.tsp.tsp_model import Point2D, TSPModel2D


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> List[str]:
    """Get datasets available for tsp.

    Params:
        data_folder: folder where datasets for tsp whould be find.
            If None, we look in "tsp" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/tsp"

    try:
        files = [
            f
            for f in os.listdir(data_folder)
            if not f.endswith(".pk") and not f.endswith(".json")
        ]
    except FileNotFoundError:
        files = []
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def parse_input_data(
    input_data: str, start_index: int = 0, end_index: int = 0
) -> TSPModel2D:
    lines = input_data.split("\n")
    node_count = int(lines[0])
    points = []
    for i in range(1, node_count + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point2D(float(parts[0]), float(parts[1])))
    return TSPModel2D(
        list_points=points,
        node_count=node_count,
        start_index=start_index,
        end_index=end_index,
        use_numba=False,
    )


def parse_file(file_path: str, start_index: int = 0, end_index: int = 0) -> TSPModel2D:
    # parse the input
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        return parse_input_data(
            input_data, start_index=start_index, end_index=end_index
        )
