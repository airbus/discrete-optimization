"""
Constraint programming common utilities and class that should be used by any solver using CP

"""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from abc import abstractmethod
from enum import Enum
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
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
    ) -> ResultStorage: ...

    @abstractmethod
    def minimize_variable(self, var: Any) -> None:
        """Set the cp solver objective as minimizing `var`."""
        pass

    @abstractmethod
    def add_bound_constraint(self, var: Any, sign: SignEnum, value: int) -> list[Any]:
        """Add constraint of bound type on an integer variable (or expression) of the underlying cp model.

        `var` must compare to `value` according to `value`.

        Args:
            var:
            sign:
            value:

        Returns:

        """
        ...
