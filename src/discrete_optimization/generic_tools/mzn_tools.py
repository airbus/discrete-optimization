#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from datetime import timedelta
from typing import Any, Optional

import minizinc
from minizinc import Instance, Status

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import (
    CpSolver,
    CpSolverName,
    ParametersCp,
    SignEnum,
    logger,
    map_cp_solver_name,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

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


map_mzn_status_to_do_status: dict[Status, StatusSolver] = {
    Status.SATISFIED: StatusSolver.SATISFIED,
    Status.UNSATISFIABLE: StatusSolver.UNSATISFIABLE,
    Status.OPTIMAL_SOLUTION: StatusSolver.OPTIMAL,
    Status.UNKNOWN: StatusSolver.UNKNOWN,
}


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

    def minimize_variable(self, var: Any) -> None:
        """Set the cp solver objective as minimizing `var`."""
        raise NotImplementedError()

    def add_bound_constraint(self, var: Any, sign: SignEnum, value: int) -> list[Any]:
        raise NotImplementedError()

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
