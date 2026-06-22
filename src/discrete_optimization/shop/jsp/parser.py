#  Copyright (c) 2024-2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.shop.base import Job, Subjob, SubjobRecipe
from discrete_optimization.shop.jsp.problem import JobShopProblem


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
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def parse_file(file_path: str):
    with open(file_path, "r") as file:
        lines = file.readlines()
        processed_line = 0
        jobs = []
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
                                {"machine_index": machine, "processing_time": int(n)}
                            )
                    jobs.append(job)
                processed_line += 1
    list_jobs = []
    for job_index, job in enumerate(jobs):
        subjobs = [
            Subjob(
                subjob_index=subjob_index,
                job_index=job_index,
                recipes=[
                    SubjobRecipe(
                        machine_index=subjob["machine_index"],
                        processing_time=subjob["processing_time"],
                    )
                ],
            )
            for subjob_index, subjob in enumerate(job)
        ]
        list_jobs.append(Job(job_index=job_index, subjobs=subjobs))
    return JobShopProblem(list_jobs=list_jobs, n_jobs=len(list_jobs))
