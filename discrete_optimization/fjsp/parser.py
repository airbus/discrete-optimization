#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
import re
from typing import Optional

from discrete_optimization.datasets import get_data_home
from discrete_optimization.fjsp.problem import (
    FJobShopProblem,
    Job,
    Subjob,
    SubjobOptions,
)


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for jobshop."""
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/jfsp_openhsu/FJSSPinstances/"

    try:
        subfolders = [
            f
            for f in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, f))
        ]
        files = []
        for fold in subfolders:
            files += [
                os.path.join(data_folder, fold, f)
                for f in os.listdir(os.path.join(data_folder, fold))
            ]
    except FileNotFoundError:
        files = []
    return files


def parse_file(file_path: str):
    with open(file_path, "r") as file:
        lines = file.readlines()
        list_jobs = []
        for k in range(len(lines)):
            numbers = re.findall(r"\d+", lines[k])
            numbers = list(map(int, numbers))
            if k == 0:
                nb_jobs = numbers[0]
                nb_machines = numbers[1]
            else:
                nb_subjob = numbers[0]
                i_subjob = 0
                cur_index = 1
                subjobs = []
                while i_subjob < nb_subjob:
                    options = []
                    nb_alternative = numbers[cur_index]
                    cur_index += 1
                    for i in range(nb_alternative):
                        sub = Subjob(
                            machine_id=numbers[cur_index] - 1,
                            processing_time=numbers[cur_index + 1],
                        )
                        options.append(sub)
                        cur_index += 2
                    subjobs.append(options)
                    i_subjob += 1
                list_jobs.append(Job(job_id=k - 1, sub_jobs=subjobs))
        return FJobShopProblem(list_jobs=list_jobs)
