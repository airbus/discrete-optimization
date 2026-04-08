#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
import re
from typing import Optional

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.ovensched.problem import (
    MachineData,
    OvenSchedulingProblem,
    TaskData,
)


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for oven scheduling problem.

    Params:
        data_folder: folder where datasets for ovensched should be found.
            If None, we look in "ovensched" subdirectory of `data_home`.
        data_home: root directory for all datasets. If None, set by
            default to "~/discrete_optimization_data"

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/ovensched"

    try:
        # Walk through all subdirectories to find .dat files
        files = []
        for root, dirs, filenames in os.walk(data_folder):
            for filename in filenames:
                if filename.endswith(".dat"):
                    files.append(os.path.abspath(os.path.join(root, filename)))
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)
    return files


def parse_dat_file(file_path: str) -> OvenSchedulingProblem:
    """
    Parses a .dat file and returns an OvenSchedulingProblem instance.
    This version includes robust error handling.
    """
    with open(file_path, "r") as f:
        content = f.read()

    def _robust_search_and_eval(name: str, is_int: bool = False):
        """Generic helper to find a value with a regex and raise a clear error on failure."""
        # The pattern uses \s* for optional whitespace and a non-greedy capture (.+?)
        # to correctly handle nested lists and multiline values.
        full_pattern = rf"{name}\s*=\s*(.+?);"
        match = re.search(full_pattern, content, re.DOTALL)

        if match is None:
            raise ValueError(
                f"Parser error: Could not find field '{name}' in '{file_path}'."
            )

        # The captured string is evaluated as Python code.
        value = eval(match.group(1))
        return int(value) if is_int else value

    try:
        n_jobs = _robust_search_and_eval("nJobs", is_int=True)
        n_machines = _robust_search_and_eval("nMachines", is_int=True)
        n_attributes = _robust_search_and_eval("nAttributes", is_int=True)

        upper_bound_integer_objective = _robust_search_and_eval(
            "upper_bound_integer_objective", is_int=True
        )
        mult_factor_total_runtime = _robust_search_and_eval(
            "mult_factor_total_runtime", is_int=True
        )
        mult_factor_finished_toolate = _robust_search_and_eval(
            "mult_factor_finished_toolate", is_int=True
        )
        mult_factor_total_setuptimes = _robust_search_and_eval(
            "mult_factor_total_setuptimes", is_int=True
        )
        mult_factor_total_setupcosts = _robust_search_and_eval(
            "mult_factor_total_setupcosts", is_int=True
        )
        running_time_bound = _robust_search_and_eval(
            "upper_bound_integer_objective", is_int=True
        )
        setup_costs_raw = _robust_search_and_eval("SetupCosts")
        setup_times_raw = _robust_search_and_eval("SetupTimes")
        setup_costs = setup_costs_raw[1:]
        setup_times = setup_times_raw[1:]
        shift_starts = _robust_search_and_eval("ShiftStartTimes")
        shift_ends = _robust_search_and_eval("ShiftEndTimes")
        job_sizes = _robust_search_and_eval("JobSize")
        job_attributes = [x - 1 for x in _robust_search_and_eval("Attribute")]
        min_times = _robust_search_and_eval("MinTime")
        max_times = _robust_search_and_eval("MaxTime")
        earliest_starts = _robust_search_and_eval("EarliestStart")
        latest_ends = _robust_search_and_eval("LatestEnd")
        eligible_machines_raw = _robust_search_and_eval("EligibleMachines")
        machine_capacities = _robust_search_and_eval("MaxCap")
        initial_states = [x - 1 for x in _robust_search_and_eval("initState")]

        # Build tasks_data
        tasks_data = []
        for j in range(n_jobs):
            tasks_data.append(
                TaskData(
                    attribute=job_attributes[j],
                    min_duration=min_times[j],
                    max_duration=max_times[j],
                    earliest_start=earliest_starts[j],
                    latest_end=latest_ends[j],
                    eligible_machines=set([x - 1 for x in eligible_machines_raw[j]]),
                    size=job_sizes[j],
                )
            )

        # Build machines_data
        machines_data = []
        for m in range(n_machines):
            avail_intervals = [
                (s, e) for s, e in zip(shift_starts[m], shift_ends[m]) if e > s
            ]
            machines_data.append(
                MachineData(
                    capacity=machine_capacities[m],
                    initial_attribute=initial_states[m],
                    availability=avail_intervals,
                )
            )

        problem = OvenSchedulingProblem(
            n_jobs=n_jobs,
            n_machines=n_machines,
            tasks_data=tasks_data,
            machines_data=machines_data,
            setup_costs=setup_costs,
            setup_times=setup_times,
        )

        additional_data = {
            "ub": upper_bound_integer_objective,
            "weight_tardiness": mult_factor_finished_toolate,
            "weight_processing": mult_factor_total_runtime,
            "weight_setup_cost": mult_factor_total_setupcosts,
            "running_time_ub": running_time_bound,
        }
        problem.additional_data = additional_data
        return problem

    except (ValueError, SyntaxError, NameError) as e:
        print(f"--- PARSER FAILED ---")
        print(
            f"An error occurred while parsing '{file_path}'. Please check the file format."
        )
        print(f"Details: {e}")
        raise
