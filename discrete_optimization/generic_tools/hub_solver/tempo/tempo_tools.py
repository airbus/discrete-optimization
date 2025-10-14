#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
import logging
import os
import subprocess
from abc import abstractmethod
from enum import Enum
from typing import Any, Optional

import pandas as pd

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import CpSolver
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class FormatEnum(Enum):
    WORKFORCE = 0
    PSPLIB = 1
    JSP = 2
    FJSP = 3


def from_format_to_str_arg(format_enum: FormatEnum):
    map_ = {
        FormatEnum.WORKFORCE: "airbus",
        FormatEnum.PSPLIB: "psplib",
        FormatEnum.JSP: "jsp",
        FormatEnum.FJSP: "fjsp",
    }
    return map_[format_enum]


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

    def on_solve_end(self, res: ResultStorage, solver: "TempoSchedulingSolver"):
        self.logs = solver._processed_logs

    def get_df_metrics(self, phase: int = 0) -> pd.DataFrame:
        """Construct a dataframe indexed by time of the recorded metrics (fitness, bounds...)."""
        return self.logs[phase]


class TempoSchedulingSolver(SolverDO):
    hyperparameters = [
        CategoricalHyperparameter(name="use_lns", choices=[True, False], default=False),
        IntegerHyperparameter(name="greedy_runs", low=0, high=100, step=1, default=1),
    ]
    _input_format: Optional[FormatEnum]
    _path_to_tempo_executable: str
    # Path to the file that is passed to tempo, this will be updated in the init_model() of the solver.
    # (specific to each problem)
    _file_input: str

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        input_format: Optional[FormatEnum] = None,
        path_to_tempo_executable: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._input_format = input_format
        self._path_to_tempo_executable = path_to_tempo_executable
        if path_to_tempo_executable is None and "TEMPO_PATH" in os.environ:
            self._path_to_tempo_executable = os.environ["TEMPO_PATH"]
        self._file_input = None
        self._raw_logs = None
        self._processed_logs = None
        self._processed_status = None

    def get_input_format(self) -> str:
        return self._input_format

    def get_processed_logs(self):
        return self._processed_logs

    def get_processed_status(self):
        return self._processed_status

    def get_tmp_folder_path(self) -> str:
        this_folder = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(this_folder, f"{self._input_format.name}/")

    @abstractmethod
    def retrieve_solution(
        self, path_to_output: str, process_stdout: str
    ) -> Solution: ...

    @abstractmethod
    def init_model(self, **kwargs: Any) -> None:
        """For tempo solver, this should transform the python object into some format that tempo can understand.
        For now it's via the creation of a temporary _file_input."""
        ...

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: float = 10,
        **kwargs: Any,
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        use_lns = kwargs["use_lns"]
        greedy_runs = kwargs["greedy_runs"]
        search_limit = kwargs.get("search_limit", 1000)
        if self._file_input is None:
            self.init_model(**kwargs)
        callback = CallbackList(callbacks)
        now = datetime.datetime.now().timestamp()
        output_json_file = os.path.join(
            self.get_tmp_folder_path(), f"results_{now}.json"
        )
        if not os.path.exists(self.get_tmp_folder_path()):
            os.makedirs(self.get_tmp_folder_path())
        # Command to run the external C++ solver
        path_to_tempo = self._path_to_tempo_executable
        command = [
            path_to_tempo,
            self._file_input,
            "--input-format",
            from_format_to_str_arg(self._input_format),
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
            self._raw_logs = process.stdout
            self._processed_logs = dfs_s
            self._processed_status = status
            result_store = self.create_result_storage([])
            sol = self.retrieve_solution(
                path_to_output=output_json_file, process_stdout=process.stdout
            )
            if sol is not None:
                fit = self.aggreg_from_sol(sol)
                result_store.append((sol, fit))
            callback.on_solve_end(result_store, solver=self)
            # Clean up temporary files after successful execution
            os.remove(self._file_input)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            # Return the result storage, potentially including the extracted logs in the 'infos' dictionary
            # The 'infos' dictionary can be used to pass additional data alongside the solution.
            return result_store
        except subprocess.CalledProcessError as e:
            # Handle errors where the subprocess returns a non-zero exit code
            logger.error(f"Subprocess failed with error: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            # Ensure temporary files are cleaned up even on error
            if os.path.exists(self._file_input):
                os.remove(self._file_input)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise  # Re-raise the exception after logging details
        except FileNotFoundError:
            if os.path.exists(self._file_input):
                os.remove(self._file_input)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred: {e}")
            if os.path.exists(self._file_input):
                os.remove(self._file_input)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise
