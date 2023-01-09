"""
Constraint programming common utilities and class that should be used by any solver using CP

"""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, Optional, Union

from minizinc import Instance, Result, Status

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class CPSolverName(Enum):
    """
    Enum choice of underlying CP/LP solver used by Minizinc typically
    """

    CHUFFED = 0
    GECODE = 1
    CPLEX = 2
    CPOPT = 3
    GUROBI = 4
    ORTOOLS = 5


map_cp_solver_name = {
    CPSolverName.CHUFFED: "chuffed",
    CPSolverName.GECODE: "gecode",
    CPSolverName.CPLEX: "cplex",
    CPSolverName.CPOPT: "cpo",
    # need to install https://github.com/IBMDecisionOptimization/cpofzn
    CPSolverName.GUROBI: "gurobi",
    CPSolverName.ORTOOLS: "ortools",
}


class ParametersCP:
    """
    Parameters that can be used by any cp - solver
    """

    time_limit: int
    time_limit_iter0: int
    intermediate_solution: bool
    all_solutions: bool
    nr_solutions: int
    free_search: bool
    multiprocess: bool
    nb_process: int
    optimisation_level: int

    def __init__(
        self,
        time_limit: int,
        intermediate_solution: bool,
        all_solutions: bool,
        nr_solutions: int,
        time_limit_iter0: Optional[int] = None,
        free_search: bool = False,
        multiprocess: bool = False,
        nb_process: int = 1,
        optimisation_level: int = 1,
    ):
        """

        :param time_limit: in seconds, the time limit of solving the cp model
        :param intermediate_solution: retrieve intermediate solutions
        :param all_solutions: returns all solutions found by the cp solver
        :param nr_solutions: the requested number of solutions
        """
        self.time_limit = time_limit
        if time_limit_iter0 is None:
            self.time_limit_iter0 = time_limit
        else:
            self.time_limit_iter0 = time_limit_iter0
        self.intermediate_solution = intermediate_solution
        self.all_solutions = all_solutions
        self.nr_solutions = nr_solutions
        self.free_search = free_search
        self.multiprocess = multiprocess
        self.nb_process = nb_process
        self.optimisation_level = optimisation_level

    @staticmethod
    def default() -> "ParametersCP":
        return ParametersCP(
            time_limit=100,
            intermediate_solution=True,
            all_solutions=False,
            nr_solutions=1000,
            free_search=False,
            optimisation_level=1,
        )

    @staticmethod
    def default_fast_lns() -> "ParametersCP":
        return ParametersCP(
            time_limit=10,
            intermediate_solution=True,
            all_solutions=False,
            nr_solutions=1000,
            free_search=False,
        )

    @staticmethod
    def default_free() -> "ParametersCP":
        return ParametersCP(
            time_limit=100,
            intermediate_solution=True,
            all_solutions=False,
            nr_solutions=1000,
            free_search=True,
        )

    def copy(self) -> "ParametersCP":
        return ParametersCP(
            time_limit=self.time_limit,
            time_limit_iter0=self.time_limit_iter0,
            intermediate_solution=self.intermediate_solution,
            all_solutions=self.all_solutions,
            nr_solutions=self.nr_solutions,
            free_search=self.free_search,
            multiprocess=self.multiprocess,
            nb_process=self.nb_process,
            optimisation_level=self.optimisation_level,
        )


class SignEnum(Enum):
    EQUAL = "=="
    LEQ = "<="
    UEQ = ">="
    LESS = "<"
    UP = ">"


class StatusSolver(Enum):
    SATISFIED = "SATISFIED"
    UNSATISFIABLE = "UNSATISFIABLE"
    OPTIMAL = "OPTIMAL"
    UNKNOWN = "UNKNOWN"


map_mzn_status_to_do_status: Dict[Status, StatusSolver] = {
    Status.SATISFIED: StatusSolver.SATISFIED,
    Status.UNSATISFIABLE: StatusSolver.UNSATISFIABLE,
    Status.OPTIMAL_SOLUTION: StatusSolver.OPTIMAL,
    Status.UNKNOWN: StatusSolver.UNKNOWN,
}


class CPSolver(SolverDO):
    """
    Additional function to be implemented by a CP Solver.
    """

    instance: Optional[Any]
    status_solver: Optional[StatusSolver] = None

    @abstractmethod
    def init_model(self, **args: Any) -> None:
        """
        Instantiate a CP model instance

        Afterwards, self.instance should not be None anymore.

        """
        ...

    @abstractmethod
    def retrieve_solutions(
        self, result: Result, parameters_cp: ParametersCP
    ) -> ResultStorage:
        """
        Returns a storage solution coherent with the given parameters.
        :param result: Result storage returned by the cp solver
        :param parameters_cp: parameters of the CP solver.
        :return:
        """
        ...

    @abstractmethod
    def solve(
        self, parameters_cp: Optional[ParametersCP] = None, **args: Any
    ) -> ResultStorage:
        ...

    def get_status_solver(self) -> Union[StatusSolver, None]:
        return self.status_solver


class MinizincCPSolver(CPSolver):
    """CP solver wrapping a minizinc solver."""

    instance: Optional[Instance] = None
    silent_solve_error: bool = False
    """If True and `solve` should raise an error, a warning is raised instead and an empty ResultStorage returned."""

    def solve(
        self, parameters_cp: Optional[ParametersCP] = None, **kwargs: Any
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        if self.instance is None:
            self.init_model(**kwargs)
            if self.instance is None:
                raise RuntimeError(
                    "self.instance must not be None after self.init_model()."
                )
        limit_time_s = parameters_cp.time_limit
        intermediate_solutions = parameters_cp.intermediate_solution
        if self.silent_solve_error:
            try:
                result = self.instance.solve(
                    timeout=timedelta(seconds=limit_time_s),
                    intermediate_solutions=intermediate_solutions,
                    processes=parameters_cp.nb_process
                    if parameters_cp.multiprocess
                    else None,
                    free_search=parameters_cp.free_search,
                    optimisation_level=parameters_cp.optimisation_level,
                )
            except Exception as e:
                logger.warning(e)
                return ResultStorage(
                    list_solution_fits=[],
                )
        else:
            result = self.instance.solve(
                timeout=timedelta(seconds=limit_time_s),
                intermediate_solutions=intermediate_solutions,
                processes=parameters_cp.nb_process
                if parameters_cp.multiprocess
                else None,
                free_search=parameters_cp.free_search,
                optimisation_level=parameters_cp.optimisation_level,
            )
        logger.info("Solving finished")
        logger.debug(result.status)
        logger.debug(result.statistics)
        self.status_solver = map_mzn_status_to_do_status[result.status]
        return self.retrieve_solutions(result=result, parameters_cp=parameters_cp)
