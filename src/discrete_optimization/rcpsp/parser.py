#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from collections.abc import Hashable
from typing import Optional, Union

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.rcpsp.problem import RcpspProblem


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
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
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)
    return sorted([os.path.abspath(os.path.join(data_folder, f)) for f in files])


def parse_psplib(input_data: str) -> RcpspProblem:
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
    resources: dict[str, Union[int, list[int]]] = {
        str(tmp1[(i * 2)]) + str(tmp1[(i * 2) + 1]): int(tmp2[i])
        for i in range(len(tmp2))
    }
    non_renewable_resources = [
        name for name in list(resources.keys()) if name.startswith("N")
    ]
    n_resources = len(resources.keys())

    # Parsing precedence relationship
    successors: dict[Hashable, list[Hashable]] = {}
    for i in range(prec_start_line_index, prec_end_line_index + 1):
        tmp = lines[i].split()
        task_id = int(tmp[0])
        n_successors = int(tmp[2])
        successors[task_id] = [int(x) for x in tmp[3 : (3 + n_successors)]]

    # Parsing mode and duration information
    mode_details: dict[Hashable, dict[int, dict[str, int]]] = {}
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
        mode_details[int(task_id)][mode_id] = {}  # dict[int, dict[str, int]]
        mode_details[int(task_id)][mode_id]["duration"] = duration
        for i in range(n_resources):
            mode_details[int(task_id)][mode_id][list(resources.keys())[i]] = (
                resources_usage[i]
            )

    return RcpspProblem(
        resources=resources,
        non_renewable_resources=non_renewable_resources,
        mode_details=mode_details,
        successors=successors,
        horizon=horizon,
    )


def parse_patterson(input_data: str) -> RcpspProblem:
    lines = input_data.split()
    parsed_values = []

    for line in lines:
        parsed_values.extend([int(_) for _ in line.split()])

    # Number of all activities, including dummy activities
    n_all_activities = parsed_values[0]

    # Number of renewable resources
    n_renewable_resources = parsed_values[1]

    # Creating resource dict with only renewable resources
    resources: dict[str, Union[int, list[int]]] = {
        "R" + str(i + 1): parsed_values[2 + i] for i in range(n_renewable_resources)
    }

    # no non-renewable resources in patterson files
    non_renewable_resources = [
        name for name in list(resources.keys()) if name.startswith("N")
    ]

    # setting up dict data structure for successor and mode_detail information
    successors: dict[Hashable, list[Hashable]] = {}
    mode_details: dict[Hashable, dict[int, dict[str, int]]] = {}

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

        mode_details[int(task_id)][mode_id] = {}  # dict[int, dict[str, int]]
        mode_details[int(task_id)][mode_id]["duration"] = duration

        horizon += duration

        for res in range(n_renewable_resources):
            mode_details[int(task_id)][mode_id][list(resources.keys())[res]] = (
                parsed_values.pop(0)
            )

        n_successors = parsed_values.pop(0)
        task_successors = []
        for suc in range(n_successors):
            task_successors.append(parsed_values.pop(0))

        successors[task_id] = task_successors

        if task_id == n_all_activities:
            break

    return RcpspProblem(
        resources=resources,
        non_renewable_resources=non_renewable_resources,
        mode_details=mode_details,
        successors=successors,
        horizon=horizon,
    )


def parse_mmlib(input_data: str) -> RcpspProblem:
    """Parse MMLIB (Multi-Mode Library) format files.

    MMLIB format is used for multi-mode RCPSP instances where each task
    can be executed in different modes with different durations and resource
    requirements.
    """
    lines = input_data.split("\n")

    # Find section markers (more robust matching)
    resources_line_idx = None
    prec_line_idx = None
    duration_line_idx = None
    avail_line_idx = None

    for i, line in enumerate(lines):
        if "RESOURCES" in line and resources_line_idx is None:
            resources_line_idx = i
        elif "PRECEDENCE RELATIONS" in line:
            prec_line_idx = i
        elif "REQUESTS/DURATIONS" in line or "REQUESTS / DURATIONS" in line:
            duration_line_idx = i
        elif "RESOURCEAVAILABILITIES" in line or "RESOURCE AVAILABILITIES" in line:
            avail_line_idx = i

    if any(
        x is None
        for x in [resources_line_idx, prec_line_idx, duration_line_idx, avail_line_idx]
    ):
        raise ValueError("Could not find all required sections in MMLIB file")

    # Parse number of jobs from first line (format: "jobs  (incl. supersource/sink ):    102")
    first_line = lines[0].split()
    n_jobs = int(first_line[-1])  # Last token is the number

    # Parse resource types (renewable R, nonrenewable N, doubly-constrained D)
    n_renewable = 0
    n_nonrenewable = 0
    for i in range(resources_line_idx + 1, prec_line_idx):
        line = lines[i].strip()
        if not line or line.startswith("*"):
            continue
        if "renewable" in line and ":" in line:
            parts = line.split(":")
            count_part = parts[1].strip().split()
            if "nonrenewable" not in line and "doubly" not in line:
                n_renewable = int(count_part[0])
            elif "nonrenewable" in line:
                n_nonrenewable = int(count_part[0])

    # Parse precedence relations and number of modes
    successors: dict[Hashable, list[Hashable]] = {}
    n_modes_per_job: dict[int, int] = {}

    prec_start = prec_line_idx + 2
    for i in range(prec_start, duration_line_idx):
        line = lines[i].strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue

        job_id = int(parts[0])
        n_modes = int(parts[1])
        n_succ = int(parts[2])
        succ_list = [int(parts[3 + j]) for j in range(n_succ)]

        successors[job_id] = succ_list
        n_modes_per_job[job_id] = n_modes

    # Parse mode details (durations and resource requirements)
    mode_details: dict[Hashable, dict[int, dict[str, int]]] = {}
    duration_start = duration_line_idx + 3

    current_job = None
    for i in range(duration_start, avail_line_idx):
        line = lines[i].strip()
        if not line or line.startswith("*"):
            continue

        parts = line.split()
        if len(parts) == 0:
            continue

        # Check if this line starts a new job or continues current job
        # New job: job_id, mode, duration, resources...
        # Continuation: mode, duration, resources...
        n_resources = n_renewable + n_nonrenewable

        if len(parts) == 3 + n_resources:
            # New job line
            current_job = int(parts[0])
            mode_id = int(parts[1])
            duration = int(parts[2])
            resource_values = [int(parts[3 + j]) for j in range(n_resources)]

            if current_job not in mode_details:
                mode_details[current_job] = {}

        elif len(parts) == 2 + n_resources and current_job is not None:
            # Continuation line (additional mode for current job)
            mode_id = int(parts[0])
            duration = int(parts[1])
            resource_values = [int(parts[2 + j]) for j in range(n_resources)]
        else:
            continue

        # Store mode details
        mode_details[current_job][mode_id] = {"duration": duration}

    # Parse resource availabilities
    avail_start = avail_line_idx + 1
    resource_names = []
    resource_amounts = []

    for i in range(avail_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("*"):
            continue

        parts = line.split()
        if len(parts) == 0:
            continue

        # First non-empty line has resource names (R 1, R 2, N 1, N 2)
        if not resource_names:
            # Reconstruct resource names from pairs
            j = 0
            while j < len(parts):
                if j + 1 < len(parts):
                    resource_names.append(parts[j] + parts[j + 1])
                    j += 2
                else:
                    j += 1
        else:
            # Second line has amounts
            resource_amounts = [int(x) for x in parts]
            break

    # Build resources dict
    resources: dict[str, Union[int, list[int]]] = {}
    for i, name in enumerate(resource_names):
        if i < len(resource_amounts):
            resources[name] = resource_amounts[i]

    # Identify non-renewable resources (those starting with 'N')
    non_renewable_resources = [
        name for name in resources.keys() if name.startswith("N")
    ]

    # Re-parse mode details with actual resource names
    mode_details = {}
    current_job = None

    for i in range(duration_start, avail_line_idx):
        line = lines[i].strip()
        if not line or line.startswith("*"):
            continue

        parts = line.split()
        if len(parts) == 0:
            continue

        n_resources = len(resource_names)

        if len(parts) == 3 + n_resources:
            # New job line
            current_job = int(parts[0])
            mode_id = int(parts[1])
            duration = int(parts[2])
            resource_values = [int(parts[3 + j]) for j in range(n_resources)]

            if current_job not in mode_details:
                mode_details[current_job] = {}

        elif len(parts) == 2 + n_resources and current_job is not None:
            # Continuation line
            mode_id = int(parts[0])
            duration = int(parts[1])
            resource_values = [int(parts[2 + j]) for j in range(n_resources)]
        else:
            continue

        # Store mode details with resource names
        mode_details[current_job][mode_id] = {"duration": duration}
        for j, res_name in enumerate(resource_names):
            mode_details[current_job][mode_id][res_name] = resource_values[j]

    # Compute horizon as sum of max durations
    horizon = 0
    for job_id, modes in mode_details.items():
        max_duration = max(mode_data["duration"] for mode_data in modes.values())
        horizon += max_duration

    return RcpspProblem(
        resources=resources,
        non_renewable_resources=non_renewable_resources,
        mode_details=mode_details,
        successors=successors,
        horizon=horizon,
    )


def parse_file(file_path: str) -> RcpspProblem:
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        if file_path.endswith(".rcp"):
            rcpsp_problem = parse_patterson(input_data)
        else:
            rcpsp_problem = parse_psplib(input_data)
        return rcpsp_problem
