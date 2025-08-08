#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# You should compile the executable tempo_scheduler,
# from the tempo library developed here : https://gepgitlab.laas.fr/roc/emmanuel-hebrard/tempo
import datetime
import json
import logging
import os.path
import subprocess
from typing import Any, Optional

import numpy as np
import pandas as pd

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

# Tempo wrapper for scheduling
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    export_scheduling_problem_json,
)
from discrete_optimization.workforce.scheduling.solvers import SolverAllocScheduling

logger = logging.getLogger(__name__)


def parse_solver_log_robust(file_content: str) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Parses a solver log to analyze multiple, sequential optimization phases with robust transition logic.

    Args:
        file_content: A string containing the full content of the log file.

    Returns:
        A tuple containing:
        - A list of pandas DataFrames, one for each optimization phase.
        - A list of strings describing the end status of each phase.
    """
    # Initialization for multiple phases
    all_phases_data = [[]]
    phase_statuses = []
    phase_start_times = [0.0]

    last_fit_value = float("inf")
    seen_cpu_times = set()

    # NEW: Flag to prevent a value-increase trigger immediately after a keyword trigger.
    just_had_keyword_transition = False

    lines = file_content.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("--") or "objective" in line or "unit" in line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        current_cpu = None
        current_fit = None
        is_transition_keyword = False
        try:
            if parts[0] in ["limit", "optimal"]:
                is_transition_keyword = True
                current_cpu = float(parts[-2])
            else:
                current_fit = float(parts[0])
                current_cpu = float(parts[-2])
        except (ValueError, IndexError):
            continue

        if current_cpu is None or current_cpu in seen_cpu_times:
            continue
        seen_cpu_times.add(current_cpu)

        # Determine if a transition should occur, suppressing value-increase check if needed.
        is_increasing_value = (
            current_fit is not None
            and current_fit > last_fit_value
            and not just_had_keyword_transition  # The important new condition
        )
        is_transition = is_transition_keyword or is_increasing_value

        if is_transition:
            # A transition terminates the CURRENT phase.
            status = "unknown_transition"
            if parts[0] == "optimal":
                status = "optimal"
            elif parts[0] == "limit":
                status = "limit"
            elif is_increasing_value:
                status = "value_increase"
            phase_statuses.append(status)

            # Prepare for the NEXT phase.
            phase_start_times.append(current_cpu)
            all_phases_data.append([])

        else:
            # It's a regular data point for the current phase.
            if current_fit is not None:
                all_phases_data[-1].append(
                    {"time": current_cpu, "obj": current_fit, "fit": current_fit}
                )

        # Update state for the NEXT iteration.
        if current_fit is not None:
            last_fit_value = current_fit

        # Set the flag for the next line based on whether THIS line was a keyword transition.
        if is_transition_keyword:
            just_had_keyword_transition = True
        else:
            just_had_keyword_transition = False

    # Finalize the last phase after the loop
    if not all_phases_data[-1]:
        all_phases_data.pop()
        phase_start_times.pop()
    else:
        phase_statuses.append("finished")

    # Create the final list of DataFrames
    list_of_dfs = []
    for i, phase_data in enumerate(all_phases_data):
        if phase_data:
            df = pd.DataFrame(phase_data)
            df["time"] = df["time"] - phase_start_times[i]
            df = df.set_index("time").sort_index()
        else:
            df = pd.DataFrame(columns=["fit", "obj"])
            df.index.name = "time"
        list_of_dfs.append(df)

    return list_of_dfs, phase_statuses


class TempoLogsCallback(Callback):
    def __init__(self):
        self.logs: list[dict] = None

    def on_solve_end(self, res: ResultStorage, solver: "TempoScheduler"):
        self.logs = solver.extracted_log

    def get_df_metrics(self, phase: int = 0) -> pd.DataFrame:
        """Construct a dataframe indexed by time of the recorded metrics (fitness, bounds...)."""
        return self.logs[phase]


class TempoScheduler(SolverAllocScheduling):
    hyperparameters = [
        CategoricalHyperparameter(name="use_lns", choices=[True, False], default=True),
        IntegerHyperparameter(name="greedy_runs", low=0, high=100, step=1, default=1),
    ]

    def __init__(
        self,
        problem: AllocSchedulingProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        path_to_tempo_scheduler: str = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.path_to_tempo_scheduler = path_to_tempo_scheduler
        self.file_json = None
        self.tmp_folder = None
        self.extracted_log = None

    def init_model(self, **kwargs: Any) -> None:

        this_folder = os.path.abspath(os.path.dirname(__file__))
        tmp_folder = os.path.join(this_folder, "data_temp/")
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        d = export_scheduling_problem_json(problem=self.problem)
        json.dump(d, open(os.path.join(tmp_folder, "tmp.json"), "w"))
        self.file_json = os.path.join(tmp_folder, "tmp.json")
        self.tmp_folder = tmp_folder

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: int = 10,
        **kwargs: Any,
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        use_lns = kwargs["use_lns"]
        greedy_runs = kwargs["greedy_runs"]
        search_limit = kwargs.get("search_limit", 1000)
        callback = CallbackList(callbacks)
        now = datetime.datetime.now().timestamp()
        output_json_file = os.path.join(self.tmp_folder, f"results_{now}.json")

        # Command to run the external C++ solver
        command = [
            self.path_to_tempo_scheduler,
            self.file_json,
            "--input-format",
            "airbus",
            "--greedy-runs",
            str(greedy_runs),
        ]
        if use_lns:
            command.extend(["--lns", "RandomTasks"])

        command.extend(
            [
                "--search-limit",
                str(search_limit),
                "--time-limit",
                str(time_limit),
                "--save-solution",
                output_json_file,
                "--seed",
                "42",
            ]
        )
        try:
            callback.on_solve_start(self)
            # Execute the command and capture stdout and stderr
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,  # Decode stdout/stderr as text
                # check=True,  # Raise a CalledProcessError if the command returns a non-zero exit code
                timeout=time_limit
                + 5,  # Add some buffer to timeout for the C++ program to finish cleanly
            )
            # Process the captured stdout line by line
            dfs_s, status = parse_solver_log_robust(file_content=process.stdout)
            logger.info(f"{status} status of the optim phases")
            logger.info(f"Last objective {[df.iloc[-1]['obj'] for df in dfs_s]}")
            assert len(dfs_s) == len(status)
            self.extracted_log = dfs_s
            result_store = self.create_result_storage([])
            if os.path.exists(output_json_file):
                res = json.load(open(output_json_file, "r"))
                schedule = np.zeros((self.problem.number_tasks, 2))
                allocation = np.zeros(self.problem.number_tasks)
                used_resource = np.zeros(
                    (self.problem.number_teams, self.problem.horizon)
                )

                for i in range(len(res["tasks"])):
                    min_time = res["tasks"][i]["start"][0]
                    duration = res["tasks"][i]["duration"]
                    resource = int(res["tasks"][i]["resource"])

                    schedule[i, 0] = min_time
                    schedule[i, 1] = min_time + duration
                    allocation[i] = int(res["tasks"][i]["resource"])
                    used_resource[resource, min_time : min_time + duration] = 1

                solution = AllocSchedulingSolution(
                    problem=self.problem, schedule=schedule, allocation=allocation
                )
                fit = self.aggreg_from_sol(solution)
                result_store.append((solution, fit))
                os.remove(output_json_file)
            callback.on_solve_end(result_store, solver=self)
            # Clean up temporary files after successful execution
            os.remove(self.file_json)
            # Return the result storage, potentially including the extracted logs in the 'infos' dictionary
            # The 'infos' dictionary can be used to pass additional data alongside the solution.
            return result_store
        except subprocess.CalledProcessError as e:
            # Handle errors where the subprocess returns a non-zero exit code
            logger.error(f"Subprocess failed with error: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            # Ensure temporary files are cleaned up even on error
            if os.path.exists(self.file_json):
                os.remove(self.file_json)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise  # Re-raise the exception after logging details
        except FileNotFoundError:
            if os.path.exists(self.file_json):
                os.remove(self.file_json)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise
        except json.JSONDecodeError as e:
            # Handle errors during JSON decoding of the output file
            logger.error(f"Error decoding JSON output file: {e}")
            if os.path.exists(self.file_json):
                os.remove(self.file_json)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred: {e}")
            if os.path.exists(self.file_json):
                os.remove(self.file_json)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise
