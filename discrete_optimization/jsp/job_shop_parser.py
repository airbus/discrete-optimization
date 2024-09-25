#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from discrete_optimization.datasets import get_data_home
from discrete_optimization.jsp.job_shop_problem import JobShopProblem, Subjob


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for jobshop.

    Params:
        data_folder: folder where datasets for jobshop whould be find.
            If None, we look in "jobshop" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/jobshop"

    try:
        files = [f for f in os.listdir(data_folder)]
    except FileNotFoundError:
        files = []
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def parse_file(file_path: str):
    with open(file_path, "r") as file:
        lines = file.readlines()
        processed_line = 0
        problem = []
        for line in lines:
            if not (line.startswith("#")):
                split_line = line.split()
                job = []
                if processed_line == 0:
                    nb_jobs = int(split_line[0])
                    nb_machines = int(split_line[1])
                else:
                    for num, n in enumerate(split_line):
                        if num % 2 == 0:
                            machine = int(n)
                        else:
                            job.append(
                                {"machine_id": machine, "processing_time": int(n)}
                            )
                    problem.append(job)
                processed_line += 1
    return JobShopProblem(list_jobs=[[Subjob(**x) for x in y] for y in problem])
