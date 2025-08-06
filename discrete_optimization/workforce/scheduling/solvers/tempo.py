#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
import logging
import os.path
import subprocess
from typing import Any, Dict, List, Optional, Union

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


class TempoLogsCallback(Callback):
    def __init__(self):
        self.logs: list[dict] = None

    def on_solve_end(self, res: ResultStorage, solver: "TempoScheduler"):
        self.logs = solver.extracted_log

    def get_df_metrics(self, phase: int = 0) -> pd.DataFrame:
        """Construct a dataframe indexed by time of the recorded metrics (fitness, bounds...)."""
        column_names = ["cpu_time", "fit", "obj"]
        pp_logs = []
        init_time = 0
        if phase != 0:
            phase_minus_1 = phase - 1
            init_time = [log for log in self.logs if log["phase"] == phase_minus_1][-1][
                "cpu_time"
            ]
        for l_ in self.logs:
            if isinstance(l_["objective"], str) or not isinstance(l_["objective"], int):
                # End of optimisation.
                continue
            pp_logs.append(l_.copy())
            pp_logs[-1]["fit"] = pp_logs[-1]["objective"]
            pp_logs[-1]["obj"] = pp_logs[-1]["objective"]
            pp_logs[-1]["cpu_time"] = pp_logs[-1]["cpu_time"] - init_time
            pp_logs[-1].pop("objective")
        df = pd.DataFrame(
            [
                {k: v for k, v in st.items() if k in column_names}
                for st in pp_logs
                if st["phase"] == phase
            ]
        ).set_index("cpu_time")
        df.columns.name = "metric"
        return df


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
        extracted_logs: List[Dict[str, Union[int, float, str]]] = []
        current_phase = 0
        # Initialize with a very high value for the first objective to ensure the first numeric objective
        # is always considered a decrease or the start of a new phase.
        last_objective = float("inf")
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
            stdout_lines = process.stdout.splitlines()
            log_section_started = False

            for line in stdout_lines:
                stripped_line = line.strip()

                # Identify the start of the statistics table by checking for the header
                if (
                    stripped_line.startswith("objective")
                    and "failures" in stripped_line
                ):
                    log_section_started = True
                    continue  # Skip the header line itself
                if log_section_started:
                    # Split the line by spaces, filtering out any empty strings that result from multiple spaces
                    parts = [part for part in stripped_line.split(" ") if part]

                    # Ensure enough columns are present to extract objective, cpu, and wall-time
                    if len(parts) >= 11:
                        try:
                            objective = None
                            # Check if the first part is a digit (numeric objective)
                            if parts[0].isdigit():
                                objective = int(parts[0])
                            # Handle 'limit' rows, storing 'limit' as a string for the objective
                            elif parts[0] == "limit" and len(parts) > 1:
                                objective = "limit"

                                # Parse CPU time and wall-time, converting them to float
                            cpu_time = float(parts[9])
                            wall_time = float(parts[10])

                            log_entry = {
                                "objective": objective,
                                "cpu_time": cpu_time,
                                "wall_time": wall_time,
                                "phase": current_phase,  # Assign the current optimization phase to the log entry
                            }
                            extracted_logs.append(log_entry)

                            # Logic to identify lexicographic optimization phase changes:
                            # If the objective value increases, it indicates a new phase.
                            # This check only applies to numeric objective values.
                            if isinstance(objective, int):
                                if objective > last_objective:
                                    current_phase += 1
                                    log_entry["phase"] = current_phase
                                # Update last_objective with the current numeric objective
                                last_objective = objective
                            elif objective == "limit" or objective == "":
                                # If a 'limit' row is encountered, reset last_objective to infinity.
                                # This ensures that the next numeric objective (which would typically be lower
                                # after a limit is hit and search continues) is correctly identified as part
                                # of a new phase if it's indeed higher than the previous last_objective
                                # before the 'limit' was hit, or correctly continues the current phase otherwise.
                                last_objective = float("inf")
                                current_phase += 1

                        except (ValueError, IndexError) as e:
                            # If parsing fails for a line (e.g., non-numeric data where numbers are expected),
                            # print an error and skip that line.
                            print(
                                f"Skipping line due to parsing error: {stripped_line} - {e}"
                            )
                            pass

                            # Load the final solution from the JSON file generated by the C++ solver

            # Print the extracted logs for immediate review
            logger.info("Extracted Optimization Logs:")
            for log_entry in extracted_logs:
                logger.info(str(log_entry))
            self.extracted_log: list[dict] = extracted_logs
            res = json.load(open(output_json_file, "r"))
            schedule = np.zeros((self.problem.number_tasks, 2))
            allocation = np.zeros(self.problem.number_tasks)
            used_resource = np.zeros((self.problem.number_teams, self.problem.horizon))

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
            res = self.create_result_storage([(solution, fit)])
            callback.on_solve_end(res, solver=self)
            # Clean up temporary files after successful execution
            os.remove(self.file_json)
            os.remove(output_json_file)
            # Return the result storage, potentially including the extracted logs in the 'infos' dictionary
            # The 'infos' dictionary can be used to pass additional data alongside the solution.
            return res

        except subprocess.CalledProcessError as e:
            # Handle errors where the subprocess returns a non-zero exit code
            print(f"Subprocess failed with error: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            # Ensure temporary files are cleaned up even on error
            if os.path.exists(self.file_json):
                os.remove(self.file_json)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise  # Re-raise the exception after logging details
        except FileNotFoundError:
            # Handle cases where the C++ executable is not found
            print(
                f"Error: The executable '{self.path_to_tempo_scheduler}' was not found."
            )
            if os.path.exists(self.file_json):
                os.remove(self.file_json)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise
        except json.JSONDecodeError as e:
            # Handle errors during JSON decoding of the output file
            print(f"Error decoding JSON output file: {e}")
            if os.path.exists(self.file_json):
                os.remove(self.file_json)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {e}")
            if os.path.exists(self.file_json):
                os.remove(self.file_json)
            if os.path.exists(output_json_file):
                os.remove(output_json_file)
            raise
