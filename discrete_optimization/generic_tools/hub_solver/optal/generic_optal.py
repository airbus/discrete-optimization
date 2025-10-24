#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import json
import logging
import os
import subprocess
import sys
from abc import abstractmethod
from typing import Any, Optional

import pandas as pd

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
)
from discrete_optimization.generic_tools.cp_tools import (
    CpSolver,
    ParametersCp,
    SignEnum,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)

this_folder = os.path.abspath(os.path.dirname(__file__))

NO_SOLUTION_STR = "No solution found."
INFEASIBLE_STR = "The problem is infeasible."


class OptalBasicCallback(BasicStatsCallback):
    def __init__(self, stats: dict):
        super().__init__()
        self.brut_stats = stats

    def get_df_metrics(self) -> pd.DataFrame:
        """Construct a dataframe indexed by time of the recorded metrics (fitness, bounds...)."""
        obj_hist = [x["objective"] for x in self.brut_stats["objectiveHistory"]]
        time_obj = [x["solveTime"] for x in self.brut_stats["objectiveHistory"]]
        bound_hist = [x["value"] for x in self.brut_stats["lowerBoundHistory"]]
        time_bound = [x["solveTime"] for x in self.brut_stats["lowerBoundHistory"]]
        records = []
        for i in range(len(time_obj)):
            t = time_obj[i]
            index_on_bound = next(
                (
                    j
                    for j in range(len(time_bound) - 1)
                    if time_bound[j] <= t <= time_bound[j + 1]
                ),
                None,
            )
            if index_on_bound is None:
                index_on_bound = len(time_bound) - 1
            d = {
                "time": t,
                "obj": obj_hist[i],
                "fit": obj_hist[i],
                "bound": bound_hist[index_on_bound],
            }
            records.append(d)
        df = pd.DataFrame.from_records(records)
        df.set_index("time", inplace=True)
        return df


class OptalSolver(CpSolver):
    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.temp_directory = os.path.join(this_folder, "temp/")
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)
        self._file_input: str = None
        self._logs_path: str = None
        self._result_path: str = None
        self._stats: dict = None
        self._script_model: str = None
        self._is_init: bool = False

    def minimize_variable(self, var: Any) -> None:
        pass

    def add_bound_constraint(self, var: Any, sign: SignEnum, value: int) -> list[Any]:
        pass

    def init_model(self, **args: Any) -> None:
        self._is_init = True

    def build_command(
        self,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: int = 10,
        **args: Any,
    ):
        """
        Build the command line call for optal cp. You can pass parameters from the Parameters class of optal cp
        for example : searchType=fds, worker0-1.noOverlapPropagationLevel=4 if you want worker 0 and 1 to use this
        parameters etc. TODO : list such parameters in hyperparameter of this wrapped solver.
        """
        if parameters_cp is None:
            parameters_cp = ParametersCp.default()
        arguments = [
            "--timeLimit",
            str(time_limit),
            "--nbWorkers",
            str(parameters_cp.nb_process),
        ]
        for k in args:
            arguments += [f"--{k}", str(args[k])]
        command = ["node", self._script_model, self._file_input, *arguments]
        return command

    @abstractmethod
    def retrieve_current_solution(self, dict_results: dict) -> Solution: ...

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: int = 10,
        do_not_retrieve_solutions: bool = False,
        verbose: bool = True,
        debug: bool = False,
        **args: Any,
    ) -> ResultStorage:
        if not self._is_init:
            self.init_model(**args)
            self._is_init = True
        cb_list = CallbackList(callbacks=callbacks)
        cb_list.on_solve_start(self)
        command = self.build_command(
            parameters_cp=parameters_cp, time_limit=time_limit, **args
        )
        res = self.create_result_storage()
        try:
            str_command = " ".join(command)
            if debug:
                print(f"\nLaunching:\n{str_command}\n")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, encoding="utf8")
            for line in iter(process.stdout.readline, ""):
                if NO_SOLUTION_STR in line:
                    self.status_solver = StatusSolver.UNKNOWN  # timeout
                if INFEASIBLE_STR in line:
                    self.status_solver = StatusSolver.UNSATISFIABLE
                if verbose:
                    sys.stdout.write(line)

            dict_results = json.load(open(self._result_path, "r"))
            if not do_not_retrieve_solutions:
                sol = self.retrieve_current_solution(dict_results=dict_results)
                fit = self.aggreg_from_sol(sol)
                res.append((sol, fit))
            self._stats = dict_results
            if (
                len(self._stats["lowerBoundHistory"]) > 0
                and len(self._stats["objectiveHistory"]) > 0
            ):
                if (
                    self._stats["lowerBoundHistory"][-1]["value"]
                    == self._stats["objectiveHistory"][-1]["objective"]
                ):
                    self.status_solver = StatusSolver.OPTIMAL
                else:
                    self.status_solver = StatusSolver.SATISFIED
            cb_list.on_solve_end(res=res, solver=self)

        except FileNotFoundError as e:
            logger.error(
                f"Error: The command 'node' or script '{self._script_model}' was not found."
            )
            logger.error(f"Error: The command {str_command} failed")
            logger.error(
                "Please ensure Node.js is installed and the path to your script is correct."
            )
            self.status_solver = StatusSolver.ERROR
        except subprocess.CalledProcessError as e:
            logger.error(f"Error: Command failed with exit code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            self.status_solver = StatusSolver.ERROR
        finally:
            if not debug:
                # Clean temporary files
                if os.path.exists(self._logs_path):
                    os.remove(self._logs_path)
                if os.path.exists(self._result_path):
                    os.remove(self._result_path)
                if os.path.exists(self._file_input):
                    os.remove(self._file_input)
        return res

    def get_output_stats(self):
        return self._stats
