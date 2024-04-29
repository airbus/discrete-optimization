#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Hashable, List, Optional, Union

from discrete_optimization.datasets import get_data_home
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> List[str]:
    """Get datasets available for rcpsp.

    Params:
        data_folder: folder where datasets for rcpsp whould be find.
            If None, we look in "rcpsp" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/rcpsp"

    try:
        files = [
            f
            for f in os.listdir(data_folder)
            if f.endswith(".sm") or f.endswith(".mm") or f.endswith(".rcp")
        ]
    except FileNotFoundError:
        files = []
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def parse_psplib(input_data: str) -> RCPSPModel:
    # parse the input
    lines = input_data.split("\n")

    # Retrieving section bounds
    horizon_ref_line_index = lines.index("RESOURCES") - 1

    prec_ref_line_index = lines.index("PRECEDENCE RELATIONS:")
    prec_start_line_index = prec_ref_line_index + 2
    duration_ref_line_index = lines.index("REQUESTS/DURATIONS:")
    prec_end_line_index = duration_ref_line_index - 2
    duration_start_line_index = duration_ref_line_index + 3
    res_ref_line_index = lines.index("RESOURCEAVAILABILITIES:")
    duration_end_line_index = res_ref_line_index - 2
    res_start_line_index = res_ref_line_index + 1

    # Parsing horizon
    tmp = lines[horizon_ref_line_index].split()
    horizon = int(tmp[2])

    # Parsing resource information
    tmp1 = lines[res_start_line_index].split()
    tmp2 = lines[res_start_line_index + 1].split()
    resources: Dict[str, Union[int, List[int]]] = {
        str(tmp1[(i * 2)]) + str(tmp1[(i * 2) + 1]): int(tmp2[i])
        for i in range(len(tmp2))
    }
    non_renewable_resources = [
        name for name in list(resources.keys()) if name.startswith("N")
    ]
    n_resources = len(resources.keys())

    # Parsing precedence relationship
    successors: Dict[Hashable, List[Hashable]] = {}
    for i in range(prec_start_line_index, prec_end_line_index + 1):
        tmp = lines[i].split()
        task_id = int(tmp[0])
        n_successors = int(tmp[2])
        successors[task_id] = [int(x) for x in tmp[3 : (3 + n_successors)]]

    # Parsing mode and duration information
    mode_details: Dict[Hashable, Dict[int, Dict[str, int]]] = {}
    for i_line in range(duration_start_line_index, duration_end_line_index + 1):
        tmp = lines[i_line].split()
        if len(tmp) == 3 + n_resources:
            task_id = int(tmp[0])
            mode_id = int(tmp[1])
            duration = int(tmp[2])
            resources_usage = [int(x) for x in tmp[3 : (3 + n_resources)]]
        else:
            mode_id = int(tmp[0])
            duration = int(tmp[1])
            resources_usage = [int(x) for x in tmp[2 : (3 + n_resources)]]

        if int(task_id) not in list(mode_details.keys()):
            mode_details[int(task_id)] = {}
        mode_details[int(task_id)][mode_id] = {}  # Dict[int, Dict[str, int]]
        mode_details[int(task_id)][mode_id]["duration"] = duration
        for i in range(n_resources):
            mode_details[int(task_id)][mode_id][
                list(resources.keys())[i]
            ] = resources_usage[i]

    return RCPSPModel(
        resources=resources,
        non_renewable_resources=non_renewable_resources,
        mode_details=mode_details,
        successors=successors,
        horizon=horizon,
        horizon_multiplier=30,
    )


def parse_patterson(input_data: str) -> RCPSPModel:
    lines = input_data.split()
    parsed_values = []

    for line in lines:
        parsed_values.extend([int(_) for _ in line.split()])

    # Number of all activities, including dummy activities
    n_all_activities = parsed_values[0]

    # Number of renewable resources
    n_renewable_resources = parsed_values[1]

    # Creating resource dict with only renewable resources
    resources: Dict[str, Union[int, List[int]]] = {
        "R" + str(i + 1): parsed_values[2 + i] for i in range(n_renewable_resources)
    }

    # no non-renewable resources in patterson files
    non_renewable_resources = [
        name for name in list(resources.keys()) if name.startswith("N")
    ]

    # setting up dict data structure for successor and mode_detail information
    successors: Dict[Hashable, List[Hashable]] = {}
    mode_details: Dict[Hashable, Dict[int, Dict[str, int]]] = {}

    # pruning irrelevant content from parsed values
    start_index_activity_information = 2 + n_renewable_resources
    parsed_values = parsed_values[start_index_activity_information:]

    task_id = 0
    horizon = 0

    # Patterson instances are not multi-mode, every task has only mode 1
    mode_id = 1

    # iterationg over remaining parsed values to populate previously created dicts (successors and mode_details)
    while True:
        task_id += 1
        mode_details[int(task_id)] = {}
        duration = parsed_values.pop(0)

        mode_details[int(task_id)][mode_id] = {}  # Dict[int, Dict[str, int]]
        mode_details[int(task_id)][mode_id]["duration"] = duration

        horizon += duration

        for res in range(n_renewable_resources):
            mode_details[int(task_id)][mode_id][
                list(resources.keys())[res]
            ] = parsed_values.pop(0)

        n_successors = parsed_values.pop(0)
        task_successors = []
        for suc in range(n_successors):
            task_successors.append(parsed_values.pop(0))

        successors[task_id] = task_successors

        if task_id == n_all_activities:
            break

    return RCPSPModel(
        resources=resources,
        non_renewable_resources=non_renewable_resources,
        mode_details=mode_details,
        successors=successors,
        horizon=horizon,
        horizon_multiplier=30,
    )


def parse_file(file_path: str) -> RCPSPModel:
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        if file_path.endswith(".rcp"):
            rcpsp_model = parse_patterson(input_data)
        else:
            rcpsp_model = parse_psplib(input_data)
        return rcpsp_model
