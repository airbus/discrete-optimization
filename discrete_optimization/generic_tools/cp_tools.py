"""
Constraint programming common utilities and class that should be used by any solver using CP

"""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import timedelta
from enum import Enum
from typing import Any, Optional, Union

import minizinc
from minizinc import Instance, Status

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class CpSolverName(Enum):
    """
    Enum choice of underlying CP/LP solver used by Minizinc typically
    """

    CHUFFED = 0
    GECODE = 1
    CPLEX = 2
    CPOPT = 3
    GUROBI = 4
    ORTOOLS = 5
    HIGHS = 6


map_cp_solver_name = {
    CpSolverName.CHUFFED: "chuffed",
    CpSolverName.GECODE: "gecode",
    CpSolverName.CPLEX: "cplex",
    CpSolverName.CPOPT: "cpo",
    # need to install https://github.com/IBMDecisionOptimization/cpofzn
    CpSolverName.GUROBI: "gurobi",
    CpSolverName.ORTOOLS: "ortools",
    CpSolverName.HIGHS: "highs",
}


_minizinc_minimal_parsed_version = (2, 8)
_minizinc_minimal_str_version = ".".join(
    str(i) for i in _minizinc_minimal_parsed_version
)


def find_right_minizinc_solver_name(cp_solver_name: CpSolverName):
    """
    This small utility function is adapting the ortools tag if needed.
    :param cp_solver_name: desired cp solver backend
    :return: the tag for minizinc corresponding to the given cpsolver.
    """
    driver = minizinc.default_driver

    # Check minzinc binary is found and has proper version
    if minizinc.default_driver is None:
        raise RuntimeError(
            "Minizinc binary has not been found.\n"
            "You need to install it and/or configure the PATH environment variable.\n"
            "See minizinc documentation for more details: https://www.minizinc.org/doc-latest/en/installation.html."
        )
    if minizinc.default_driver.parsed_version < _minizinc_minimal_parsed_version:
        raise RuntimeError(
            f"Minizinc binary version must be at least {_minizinc_minimal_str_version}.\n"
            "Install an appropriate version of minizinc and/or configure the PATH environment variable.\n"
            "See minizinc documentation for more details: https://www.minizinc.org/doc-latest/en/installation.html."
        )

    tag_map = driver.available_solvers(False)
    if map_cp_solver_name[cp_solver_name] not in tag_map:
        if cp_solver_name == CpSolverName.ORTOOLS:
            if "com.google.ortools.sat" in tag_map:
                return "com.google.ortools.sat"
        else:
            # You will get a minizinc exception when you will request for this solver.
            return map_cp_solver_name[cp_solver_name]
    else:
        return map_cp_solver_name[cp_solver_name]


class ParametersCp:
    """
    Parameters that can be used by any cp - solver
    """

    intermediate_solution: bool
    free_search: bool
    multiprocess: bool
    nb_process: int
    optimisation_level: int

    def __init__(
        self,
        intermediate_solution: bool,
        free_search: bool = False,
        multiprocess: bool = False,
        nb_process: int = 1,
        optimisation_level: int = 1,
    ):
        """

        :param intermediate_solution: retrieve intermediate solutions
        """
        self.intermediate_solution = intermediate_solution
        self.free_search = free_search
        self.multiprocess = multiprocess
        self.nb_process = nb_process
        self.optimisation_level = optimisation_level

    @staticmethod
    def default() -> "ParametersCp":
        return ParametersCp(
            intermediate_solution=True,
            free_search=False,
            optimisation_level=1,
        )

    @staticmethod
    def default_cpsat() -> "ParametersCp":
        return ParametersCp(
            intermediate_solution=True,
            free_search=False,
            multiprocess=True,
            nb_process=6,
            optimisation_level=1,
        )

    @staticmethod
    def default_fast_lns() -> "ParametersCp":
        return ParametersCp(
            intermediate_solution=True,
            free_search=False,
        )

    @staticmethod
    def default_free() -> "ParametersCp":
        return ParametersCp(
            intermediate_solution=True,
            free_search=True,
        )

    def copy(self) -> "ParametersCp":
        return ParametersCp(
            intermediate_solution=self.intermediate_solution,
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


map_mzn_status_to_do_status: dict[Status, StatusSolver] = {
    Status.SATISFIED: StatusSolver.SATISFIED,
    Status.UNSATISFIABLE: StatusSolver.UNSATISFIABLE,
    Status.OPTIMAL_SOLUTION: StatusSolver.OPTIMAL,
    Status.UNKNOWN: StatusSolver.UNKNOWN,
}


class CpSolver(SolverDO):
    """
    Additional function to be implemented by a CP Solver.
    """

    @abstractmethod
    def init_model(self, **args: Any) -> None:
        """
        Instantiate a CP model instance

        Afterwards, self.instance should not be None anymore.

        """
        ...

    @abstractmethod
    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        **args: Any,
    ) -> ResultStorage:
        ...


class MinizincCpSolver(CpSolver):
    """CP solver wrapping a minizinc solver."""

    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CpSolverName, default=CpSolverName.CHUFFED
        )
    ]
    instance: Optional[Instance] = None
    silent_solve_error: bool = False
    """If True and `solve` should raise an error, a warning is raised instead and an empty ResultStorage returned."""

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        instance: Optional[Instance] = None,
        time_limit: Optional[float] = 100.0,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve the CP problem with minizinc

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            parameters_cp: parameters specific to CP solvers
            instance: if specified, use this minizinc instance (and underlying model) rather than `self.instance`
               Useful in iterative solvers like LnsCpMzn.
            time_limit: the solve process stops after this time limit (in seconds).
                If None, no time limit is applied.
            **kwargs: any argument specific to the solver

        Returns:

        """
        # wrap callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)

        # callback: solve start
        callbacks_list.on_solve_start(solver=self)

        if parameters_cp is None:
            parameters_cp = ParametersCp.default()

        if instance is None:
            if self.instance is None:
                self.init_model(**kwargs)
                if self.instance is None:
                    raise RuntimeError(
                        "self.instance must not be None after self.init_model()."
                    )
            instance = self.instance

        intermediate_solutions = parameters_cp.intermediate_solution

        # set model output type to use
        output_type = MinizincCpSolution.generate_subclass_for_solve(
            solver=self, callback=callbacks_list
        )
        instance.output_type = output_type
        if time_limit is None:
            timeout = None
        else:
            timeout = timedelta(seconds=time_limit)
        try:
            result = instance.solve(
                timeout=timeout,
                intermediate_solutions=intermediate_solutions,
                processes=parameters_cp.nb_process
                if parameters_cp.multiprocess
                else None,
                free_search=parameters_cp.free_search,
                optimisation_level=parameters_cp.optimisation_level,
            )
        except Exception as e:
            if len(output_type.res) > 0:
                self.status_solver = StatusSolver.SATISFIED
            else:
                self.status_solver = StatusSolver.UNKNOWN
            if isinstance(e, SolveEarlyStop):
                logger.info(e)
            elif self.silent_solve_error:
                logger.warning(e)
            else:
                raise e
        else:
            logger.info("Solving finished")
            logger.info(result.status)
            logger.info(result.statistics)
            self.status_solver = map_mzn_status_to_do_status[result.status]

        # callback: solve end
        callbacks_list.on_solve_end(res=output_type.res, solver=self)

        return output_type.res

    @abstractmethod
    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> Solution:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        ...


class MinizincCpSolution:
    """Base class used by minizinc when building a new solution.

    This is used as an entry point for callbacks.
    It is actually a child class dynamically created during solve that will be used by minizinc,
    with appropriate callbacks, resultstorage and reference to the solver.

    """

    callback: Callback
    """User-definied callback to be called at each step."""

    solution: Solution
    """Solution wrapped."""

    step: int
    """Step number, updated as a class attribute."""

    res: ResultStorage
    """ResultStorage in which the solution will be added, class attribute."""

    solver: MinizincCpSolver
    """Instance of the solver using this class as an output_type."""

    def __init__(self, _output_item: Optional[str] = None, **kwargs: Any):
        # Convert minizinc variables into a d-o solution
        self.solution = self.solver.retrieve_solution(
            _output_item=_output_item, **kwargs
        )
        # Actual fitness
        fit = self.solver.aggreg_from_sol(self.solution)

        # update class attributes to remember step number and global resultstorage
        self.__class__.res.append((self.solution, fit))
        self.__class__.step += 1

        # callback: step end
        stopping = self.callback.on_step_end(
            step=self.step, res=self.res, solver=self.solver
        )
        # Should we be stopping the solve process?
        if stopping:
            raise SolveEarlyStop(
                f"{self.solver.__class__.__name__}.solve() stopped by user callback."
            )

    @staticmethod
    def generate_subclass_for_solve(
        solver: MinizincCpSolver, callback: Callback
    ) -> type[MinizincCpSolution]:
        """Generate dynamically a subclass with initialized class attributes.

        Args:
            solver:
            callback:

        Returns:

        """
        return type(
            f"MinizincCpSolution{id(solver)}",
            (MinizincCpSolution,),
            dict(
                solver=solver,
                callback=callback,
                step=0,
                res=solver.create_result_storage(),
            ),
        )
