#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from discrete_optimization.datasets import get_data_home
from discrete_optimization.singlemachine.problem import WeightedTardinessProblem


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for tsp.

    Params:
        data_folder: folder where datasets for weighted tardiness problem should be found.
            If None, we look in "wt" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/wt"

    try:
        files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    except FileNotFoundError:
        files = []
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def parse_file(path: str, num_jobs: int = None):
    if num_jobs is None:
        basename = os.path.basename(path)
        number_string = "".join(filter(str.isdigit, basename))
        if number_string:
            num_jobs = int(number_string)
    with open(path, "r") as f:
        return parse_wt_content(f.read(), num_jobs=num_jobs)


def parse_wt_content(
    file_content: str, num_jobs: int
) -> list[WeightedTardinessProblem]:
    """
    Parses a weighted tardiness file with a known number of jobs per instance.

    Args:
        file_content (str): The full content of the text file.
        num_jobs (int): The number of jobs per instance (e.g., 40 for wt40.txt).

    Returns:
        List[WeightedTardinessProblem]: A list of parsed problem instances.
    """
    temp_content = file_content
    numbers = [int(num) for num in temp_content.split() if num.isdigit()]

    data_points_per_instance = num_jobs * 3
    if len(numbers) % data_points_per_instance != 0:
        raise ValueError(
            f"The total number of data points is not a multiple of "
            f"{data_points_per_instance} (3 data points per job). "
            f"Please check the file format or the specified number of jobs."
        )

    num_instances = len(numbers) // data_points_per_instance
    problem_instances = []

    for i in range(num_instances):
        start_index = i * data_points_per_instance
        end_index = start_index + data_points_per_instance
        instance_data = numbers[start_index:end_index]

        processing_times = instance_data[0:num_jobs]
        weights = instance_data[num_jobs : 2 * num_jobs]
        due_dates = instance_data[2 * num_jobs : 3 * num_jobs]
        problem_instances.append(
            WeightedTardinessProblem(num_jobs, processing_times, weights, due_dates)
        )
    return problem_instances
