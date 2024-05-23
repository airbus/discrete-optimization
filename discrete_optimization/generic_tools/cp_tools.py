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
from typing import Any, Dict, List, Optional, Type, Union

import minizinc
from minizinc import Instance, Result, Status

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
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
    HIGHS = 6


map_cp_solver_name = {
    CPSolverName.CHUFFED: "chuffed",
    CPSolverName.GECODE: "gecode",
    CPSolverName.CPLEX: "cplex",
    CPSolverName.CPOPT: "cpo",
    # need to install https://github.com/IBMDecisionOptimization/cpofzn
    CPSolverName.GUROBI: "gurobi",
    CPSolverName.ORTOOLS: "ortools",
    CPSolverName.HIGHS: "highs",
}


def find_right_minizinc_solver_name(cp_solver_name: CPSolverName):
    """
    This small utility function is adapting the ortools tag if needed.
    :param cp_solver_name: desired cp solver backend
    :return: the tag for minizinc corresponding to the given cpsolver.
    """
    driver = minizinc.default_driver
    assert driver is not None
    tag_map = driver.available_solvers(False)
    if map_cp_solver_name[cp_solver_name] not in tag_map:
        if cp_solver_name == CPSolverName.ORTOOLS:
            if "com.google.ortools.sat" in tag_map:
                return "com.google.ortools.sat"
        else:
            # You will get a minizinc exception when you will request for this solver.
            return map_cp_solver_name[cp_solver_name]
    else:
        return map_cp_solver_name[cp_solver_name]


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
    def default_cpsat() -> "ParametersCP":
        return ParametersCP(
            time_limit=100,
            intermediate_solution=True,
            all_solutions=False,
            nr_solutions=1000,
            free_search=False,
            multiprocess=True,
            nb_process=6,
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

    status_solver: Optional[StatusSolver] = None

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
        callbacks: Optional[List[Callback]] = None,
        parameters_cp: Optional[ParametersCP] = None,
        **args: Any,
    ) -> ResultStorage:
        ...

    def get_status_solver(self) -> Union[StatusSolver, None]:
        return self.status_solver


class MinizincCPSolver(CPSolver):
    """CP solver wrapping a minizinc solver."""

    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CPSolverName, default=CPSolverName.CHUFFED
        )
    ]
    instance: Optional[Instance] = None
    silent_solve_error: bool = False
    """If True and `solve` should raise an error, a warning is raised instead and an empty ResultStorage returned."""

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        parameters_cp: Optional[ParametersCP] = None,
        instance: Optional[Instance] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve the CP problem with minizinc

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            parameters_cp: parameters specific to CP solvers
            instance: if specified, use this minizinc instance (and underlying model) rather than `self.instance`
               Useful in iterative solvers like LNS_CP.
            **kwargs: any argument specific to the solver

        Returns:

        """
        # wrap callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)

        # callback: solve start
        callbacks_list.on_solve_start(solver=self)

        if parameters_cp is None:
            parameters_cp = ParametersCP.default()

        if instance is None:
            if self.instance is None:
                self.init_model(**kwargs)
                if self.instance is None:
                    raise RuntimeError(
                        "self.instance must not be None after self.init_model()."
                    )
            instance = self.instance

        limit_time_s = parameters_cp.time_limit
        intermediate_solutions = parameters_cp.intermediate_solution

        # set model output type to use
        output_type = MinizincCPSolution.generate_subclass_for_solve(
            solver=self, callback=callbacks_list
        )
        instance.output_type = output_type

        try:
            result = instance.solve(
                timeout=timedelta(seconds=limit_time_s),
                intermediate_solutions=intermediate_solutions,
                processes=parameters_cp.nb_process
                if parameters_cp.multiprocess
                else None,
                free_search=parameters_cp.free_search,
                optimisation_level=parameters_cp.optimisation_level,
            )
        except Exception as e:
            if len(output_type.res.list_solution_fits) > 0:
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


class MinizincCPSolution:
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

    solver: MinizincCPSolver
    """Instance of the solver using this class as an output_type."""

    def __init__(self, _output_item: Optional[str] = None, **kwargs: Any):
        # Convert minizinc variables into a d-o solution
        self.solution = self.solver.retrieve_solution(
            _output_item=_output_item, **kwargs
        )
        # Actual fitness
        fit = self.solver.aggreg_from_sol(self.solution)

        # update class attributes to remember step number and global resultstorage
        self.__class__.res.add_solution(solution=self.solution, fitness=fit)
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
        solver: MinizincCPSolver, callback: Callback
    ) -> Type[MinizincCPSolution]:
        """Generate dynamically a subclass with initialized class attributes.

        Args:
            solver:
            callback:

        Returns:

        """
        return type(
            f"MinizincCPSolution{id(solver)}",
            (MinizincCPSolution,),
            dict(
                solver=solver,
                callback=callback,
                step=0,
                res=ResultStorage(
                    [],
                    mode_optim=solver.params_objective_function.sense_function,
                    limit_store=False,
                ),
            ),
        )
