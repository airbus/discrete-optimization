"""
Constraint programming common utilities and class that should be used by any solver using CP

"""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, Iterable, Optional

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
    def maximize_variable(self, var: Any) -> None:
        """Set the cp solver objective as minimizing `var`."""
        self.minimize_variable(-var)

    def add_bound_constraint(self, var: Any, sign: SignEnum, value: int) -> list[Any]:
        """Add constraint of bound type on an integer variable (or expression) of the underlying cp model.

        `var` must compare to `value` according to `value`.

        Args:
            var:
            sign:
            value:

        Returns:

        """
        if sign == SignEnum.LEQ:
            cstr = var <= value
        elif sign == SignEnum.UEQ:
            cstr = var >= value
        elif sign == SignEnum.EQUAL:
            cstr = var == value
        elif sign == SignEnum.LESS:
            cstr = var < value
        elif sign == SignEnum.UP:
            cstr = var > value
        else:
            raise NotImplementedError(sign)
        return [self.add_bool_constraint(cstr)]

    @abstractmethod
    def add_bool_constraint(self, cstr: Any) -> Any:
        """Add boolean constraint on the underlying cp model.

        Args:
            cstr: boolean constraint

        Returns:
            the created constraint, linked to the model

        """
        ...

    @abstractmethod
    def create_int_var(self, lb: int, ub: int, name: str) -> Any:
        """Create an integer variable in the underlying cp model.

        Args:
            lb: lower bound
            ub: upper bound
            name:

        Returns:
            created variable

        """
        ...

    @abstractmethod
    def create_int_var_from_possible_values(
        self, possible_values: Iterable[int], name: str
    ) -> Any:
        """Create an integer variable in the underlying cp model.

        Args:
            possible_values:
            name:

        Returns:
            created variable

        """
        ...

    @abstractmethod
    def create_int_var_from_flat_intervals(
        self, flat_intervals: list[int], name: str
    ) -> Any:
        """Create an integer variable in the underlying cp model.

        Args:
            flat_intervals: [lb0, ub0, lb1, ub1, ...]
            name:

        The created variable takes values in the union of intervals
        [lb0, ub0] U [lb1, ub1] U ...

        Returns:
            created variable

        """
        ...

    @abstractmethod
    def create_bool_var(self, name: str) -> Any:
        """Create a boolean variable in the underlying cp model.

        Args:
            min:
            max:
            name:

        Returns:
            created variable

        """
        ...

    @abstractmethod
    def add_max_equality(self, target: Any, variables: Iterable[Any]) -> list[Any]:
        """Add constraint making target variable the maximum of given variables.

        Args:
            target:
            variables:

        Returns:
            the constraints

        """
        ...

    @abstractmethod
    def add_min_equality(self, target: Any, variables: Iterable[Any]) -> list[Any]:
        """Add constraint making target variable the minimum of given variables.

        Args:
            target:
            variables:

        Returns:
            the constraints

        """
        ...

    @abstractmethod
    def add_abs_equality(self, target: Any, expr: Any) -> list[Any]:
        """Add `target == Abs(expr)`."""
        ...

    @abstractmethod
    def add_element(
        self,
        index: Any,
        expressions: Sequence[Any],
        target: Any,
    ) -> list[Any]:
        """Adds the element constraint: `expressions[index] == target`.

        Args:
          index: The index of the selected expression in the array.
          expressions: A list of affine expressions.
          target: The expression constrained to be equal to the selected expression.

        Returns:

        """
        ...

    @abstractmethod
    def add_modulo_equality(self, target: Any, expr: Any, mod: Any) -> list[Any]:
        """Adds `target = expr % mod`.

        It uses the C convention, that is the result is the remainder of the
        integral division rounded towards 0.

            For example:
            * 10 % 3 = 1
            * -10 % 3 = -1
            * 10 % -3 = 1
            * -10 % -3 = -1

        Args:
          target: the target expression.
          expr: the expression to compute the modulo of.
          mod: the modulus expression.

        Returns:

        """
        ...

    def in_intervals(self, expr: Any, intervals: Iterable[tuple[int, int]]) -> Any:
        """Construct boolean expression true iff the given expr to be in given intervals."""
        return self.at_least_one(
            tuple(self.bool_and(tuple(lb <= expr, expr <= ub)) for lb, ub in intervals)
        )

    @abstractmethod
    def add_implication(self, left_bool_expr: Any, right_bool_expr: Any) -> list[Any]:
        """Add implication constraint on the underlying cp model.

            left_bool_expr => right_bool_expr
        or if left_bool_expr is an iterable
            all(left_bool_expr) => right_bool_expr

        Args:
            left_bool_expr: literal or iterable of literals
            right_bool_expr: either a boolean expression or a constraint of the model or a list of such constraints

        Returns:
            Implication constraints

        """
        ...

    @abstractmethod
    def add_implication_bool_and(
        self, left_bool_expr: Any, right_bool_exprs: Sequence[Any]
    ) -> list[Any]:
        """Add implication constraint left_bool_expr => exactly_one(right_bool_exprs)"""
        return self.add_implication(left_bool_expr, self.bool_and(right_bool_exprs))

    @abstractmethod
    def add_implication_exactly_one(
        self, left_bool_expr: Any, right_bool_exprs: Iterable[Any]
    ) -> list[Any]:
        """Add implication constraint left_bool_expr => exactly_one(right_bool_exprs)"""
        return self.add_implication(left_bool_expr, self.exactly_one(right_bool_exprs))

    @abstractmethod
    def add_implication_at_least_one(
        self, left_bool_expr: Any, right_bool_exprs: Iterable[Any]
    ) -> list[Any]:
        """Add implication constraint left_bool_expr => any(right_bool_exprs)"""
        return self.add_implication(left_bool_expr, self.at_least_one(right_bool_exprs))

    @abstractmethod
    def add_implication_in_intervals(
        self, left_bool_expr: Any, right_expr: Any, intervals: Iterable[tuple[int, int]]
    ) -> list[Any]:
        """Add implication constraint left_bool_expr => in_intervals(target, intervals)"""
        return self.add_implication(
            left_bool_expr, self.in_intervals(right_expr, intervals)
        )

    @abstractmethod
    def add_allowed_assignments(
        self, variables: Sequence[Any], values: Iterable[Sequence[int]]
    ) -> list[Any]:
        """Enforce that variables evaluate to a sequence of values

        Args:
            variables:
            values: possible sequences for variables values

        Returns:
            created constraints

        """
        return [
            self.add_bool_constraint(
                self.at_least_one(
                    tuple(
                        self.bool_and(
                            tuple(var == value for var, value in zip(variables, row))
                        )
                        for row in values
                    )
                )
            )
        ]

    @abstractmethod
    def add_exactly_one(self, exprs: Iterable[Any]) -> list[Any]:
        """Enforce that exactly one expression is true.

        Args:
            exprs:

        Returns:

        """
        return [self.add_bool_constraint(self.exactly_one(exprs))]

    @abstractmethod
    def add_at_most_one(self, exprs: Iterable[Any]) -> list[Any]:
        """Enforce that at most one expression is true.

        Args:
            exprs:

        Returns:

        """
        return [self.add_bool_constraint(self.at_most_one(exprs))]

    @abstractmethod
    def add_at_least_one(self, exprs: Iterable[Any]) -> list[Any]:
        """Enforce that at least one expression is true.

        Args:
            exprs:

        Returns:

        """
        return [self.add_bool_constraint(self.at_least_one(exprs))]

    def bool_and(self, exprs: Sequence[Any]) -> Any:
        """Construct conjunction of boolean expressions."""
        return sum(exprs) == len(exprs)

    def exactly_one(self, exprs: Iterable[Any]) -> Any:
        """Construct boolean expression true iff exactly one expression is true."""
        return sum(exprs) == 1

    def at_most_one(self, exprs: Iterable[Any]) -> Any:
        """Construct boolean expression true iff at most expression is true."""
        return sum(exprs) <= 1

    def at_least_one(self, exprs: Iterable[Any]) -> Any:
        """Construct boolean expression true iff at least expression is true."""
        return sum(exprs) >= 1

    def hint_solution(self, hints: dict[Any, int | bool]) -> None:
        """Warm-start the underlying cp-solver with hints on some variables

        Args:
            hints: mapping between variables and their hinted values

        Returns:

        """
        raise NotImplementedError("Warmstart not implemented for this solver.")
