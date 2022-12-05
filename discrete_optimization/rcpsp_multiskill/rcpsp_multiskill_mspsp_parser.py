#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional

import numpy as np
import pymzn

from discrete_optimization.datasets import fetch_data_from_mspsplib_repo, get_data_home
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    SkillDetail,
)

logger = logging.getLogger(__name__)
this_folder = os.path.dirname(os.path.abspath(__file__))


def get_data_available_mspsp(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
):
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/MSPSP_Instances"
        if not os.path.exists(data_folder):
            logger.info(f"Fetching data from MSPSP_Lib repo")
            fetch_data_from_mspsplib_repo(data_home)
    try:
        file_paths = {}
        for sub_folder in os.listdir(data_folder):
            folder_set = os.path.join(data_folder, sub_folder)
            if os.path.isfile(folder_set):
                continue
            file_paths[sub_folder] = {}
            for subset in os.listdir(folder_set):
                folder_subset = os.path.join(folder_set, subset)
                if os.path.isfile(folder_subset):
                    continue
                files = os.listdir(folder_subset)
                file_paths[sub_folder][subset] = [
                    os.path.join(folder_subset, f) for f in files if f.endswith(".dzn")
                ]
    except FileNotFoundError:
        logger.debug("folder with MSPSP instances not found")
        file_paths = {}
    return file_paths


def parse_dzn_file(file_path) -> MS_RCPSPModel:
    data = pymzn.dzn2dict(file_path, rebase_arrays=True)
    number_of_acts = data["nActs"]
    durations = data["dur"]
    nb_skills = data["nSkills"]
    skill_req = data["sreq"]
    number_employee = data["nResources"]
    master = np.array(data["mastery"])
    master = np.reshape(master, (number_employee, nb_skills))
    skill_req = np.array(skill_req).reshape((number_of_acts, nb_skills))

    pred = data["pred"]
    succ = data["succ"]
    skills_set = set(["S_" + str(j) for j in range(nb_skills)])
    resources_set = set()
    non_renewable_resources = set()
    resources_availability = {}
    employees = {
        "employee_"
        + str(j): Employee(
            {
                "S_" + str(k): SkillDetail(1, 0, 0)
                for k in range(len(master[j]))
                if master[j][k]
            },
            calendar_employee=[True] * 2000,
        )
        for j in range(number_employee)
    }
    mode_details = {t + 1: {1: {}} for t in range(number_of_acts)}
    for k in range(number_of_acts):
        mode_details[k + 1][1]["duration"] = durations[k]
        for s in range(nb_skills):
            if skill_req[k][s] > 0:
                mode_details[k + 1][1]["S_" + str(s)] = skill_req[k][s]
    successors = {k + 1: [] for k in range(number_of_acts)}
    for p, s in zip(pred, succ):
        successors[p] += [s]
    tasks_list = [i + 1 for i in range(number_of_acts)]
    employees_list = ["employee_" + str(j) for j in range(number_employee)]
    source_task = 1
    sink_task = number_of_acts
    return MS_RCPSPModel(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employees,
        employees_availability=None,
        mode_details=mode_details,
        successors=successors,
        horizon=1000,
        tasks_list=tasks_list,
        employees_list=employees_list,
        horizon_multiplier=1,
        sink_task=sink_task,
        source_task=source_task,
        one_unit_per_task_max=False,
        preemptive=False,
        preemptive_indicator=None,
        special_constraints=None,
    )
