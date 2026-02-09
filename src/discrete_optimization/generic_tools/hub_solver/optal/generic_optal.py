#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import json
import logging
import os
import subprocess
import sys
from abc import abstractmethod
from typing import Any, Optional, TYPE_CHECKING

import pandas as pd

# Use native Python API for OptalCP, if available. If not, we'll fall back to the subprocess-based OptalSolver.
try:
    import optalcp as cp
    if TYPE_CHECKING:
        from optalcp import Model as OptalModel, Solution as OptalSolution  # type: ignore
except ImportError:
    cp = None
    
    
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
IMPORT_WARNING_STR = ("'optalcp' is not installed. OptalPythonSolver will not work. "
                "Please install it with pip install 'discrete-optimization[optalcp]'")


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
    """
    Solver wrapper for OptalCP. 
    This class uses subprocess to call the OptalCP CLI and parse results from JSON output files.
    This solver requires a Node.js environment and the OptalCP CLI to be installed. 
    It is a *fallback* option if the native Python API (OptalPythonSolver) is not available.
    """
    
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


class OptalPythonSolver(CpSolver):
    """
    Solver wrapper for OptalCP.
    This class directly uses the Python API (from OptalCP 2026.1.0 Release) to build and solve the model without relying on subprocess calls.
    Falls back to OptalSolver (using Node.js) if needed.
    """

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._stats: dict[str, Any] = {}
        if cp is None:
            logger.warning(IMPORT_WARNING_STR)

    def minimize_variable(self, var: Any) -> None:
        """Add the variable to the list of variables to minimize. Optional for native solver but kept for API parity."""
        pass

    def add_bound_constraint(self, var: Any, sign: SignEnum, value: int) -> list[Any]:
        """Add a bound constraint to the model. Optional for native solver but kept for API parity."""
        return []

    def init_model(self, **args: Any) -> None:
        """Initialize the model. Optional for native solver but kept for API parity."""
        pass

    @abstractmethod
    def build_model(self, **kwargs: Any) -> "OptalModel":
        """Build the OptalCP model natively in Python."""
        pass

    @abstractmethod
    def retrieve_current_solution(self, solution: "OptalSolution") -> Solution:
        """Extract solution from OptalCP native solution."""
        pass

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: int = 10,
        do_not_retrieve_solutions: bool = False,
        **args: Any,
    ) -> ResultStorage:
        """
        Solve the problem using the native OptalCP Python API. If OptalCP is not available, this method will raise an ImportError.
        
        :param callbacks: List of callbacks to call during the solving process
        :type callbacks: Optional[list[Callback]]
        :param parameters_cp: Parameters for the CP solver, such as number of workers, search type, etc. If None, default parameters will be used.
        :type parameters_cp: Optional[ParametersCp]
        :param time_limit: Time limit for the solver in seconds
        :type time_limit: int
        :param do_not_retrieve_solutions: If True, the method will not attempt to retrieve and store solutions, only stats. 
        Useful for very large problems where solution retrieval is costly.
        :type do_not_retrieve_solutions: bool
        :param args: Additional arguments that can be passed to the solver (check the documentation of OptalCP for supported parameters).
        :type args: Any
        :return: Storage of results containing solutions and stats from the solving process.
        :rtype: ResultStorage
        """
        
        # Check if the native OptalCP API is available
        if cp is None:
            raise ImportError(IMPORT_WARNING_STR)

        # Set default parameters if not provided
        if parameters_cp is None:
            parameters_cp = ParametersCp.default()
        
        # Call callbacks for solve start
        cb_list = CallbackList(callbacks=callbacks)
        cb_list.on_solve_start(self)

        # Build the model and set up parameters
        model = self.build_model(**args)
        params = cp.Parameters(
            timeLimit=time_limit,
            nbWorkers=parameters_cp.nb_process,
            logLevel=1 if args.get("verbose", True) else 0,
        )
        # Handle additional parameters from args
        for k, v in args.items():
            if hasattr(params, k):
                setattr(params, k, v)

        # Solve the model and retrieve results
        result = model.solve(params)
        res = self.create_result_storage()

        # Store stats for get_output_stats()
        # Note: Mapping native properties to the dict structure expected by DO
        self._stats = {
            # History of objective values over time
            "objectiveHistory": [
                {"solveTime": entry.solve_time, "objective": entry.objective} 
                for entry in result.objective_history
            ]
            if hasattr(result, "objective_history")
            else [],
            # History of lower bound values over time
            "lowerBoundHistory": [
                {"solveTime": entry.solve_time, "value": entry.value} 
                for entry in result.objective_bound_history
            ]
            if hasattr(result, "objective_bound_history")
            else [],
            # Total solving time (in seconds)
            "duration": result.duration if hasattr(result, "duration") else 0
        }

        # Determinie solving status
        if result.solution:
            # Retrieve sol if required
            if not do_not_retrieve_solutions:
                sol = self.retrieve_current_solution(solution=result.solution)
                fit = self.aggreg_from_sol(sol)
                res.append((sol, fit))
            # Update status based on whether the solution is optimal or just feasible
            self.status_solver = StatusSolver.SATISFIED
            if result.proof:
                self.status_solver = StatusSolver.OPTIMAL
        elif result.proof:
            self.status_solver = StatusSolver.UNSATISFIABLE
        else:
            self.status_solver = StatusSolver.UNKNOWN
        
        # Call callbacks for solve end
        cb_list.on_solve_end(res=res, solver=self)
        return res

    def get_output_stats(self) -> dict[str, Any]:
        """Return the stats collected during the solving process"""
        return self._stats
