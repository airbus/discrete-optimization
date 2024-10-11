#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections.abc import Iterable
from typing import Any, Optional

from ortools.sat.python.cp_model import (
    Constraint,
    CpModel,
    CpSolverSolutionCallback,
    IntVar,
    LinearExpr,
    ObjLinearExprT,
)

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers import KnapsackSolver

logger = logging.getLogger(__name__)


class CpSatKnapsackSolver(OrtoolsCpSatSolver, KnapsackSolver, WarmstartMixin):
    def __init__(
        self,
        problem: KnapsackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables: dict[str, list[IntVar]] = {}

    def init_model(self, **args: Any) -> None:
        """Init CP model."""
        model = CpModel()
        variables = [
            model.NewBoolVar(name=f"x_{i}") for i in range(self.problem.nb_items)
        ]
        self.variables["taken"] = variables
        self.cp_model = model

        model.Add(-self._internal_weight() <= self.problem.max_capacity)

        self.set_lexico_objective("value")

    def set_warm_start(self, solution: KnapsackSolution) -> None:
        """Make the solver warm start from the given solution."""
        self.cp_model.clear_hints()
        for i in range(len(solution.list_taken)):
            self.cp_model.AddHint(self.variables["taken"][i], solution.list_taken[i])

    def _internal_value(self) -> LinearExpr:
        return sum(
            [
                self.variables["taken"][i] * self.problem.list_items[i].value
                for i in range(self.problem.nb_items)
            ]
        )

    def _internal_weight(self) -> LinearExpr:
        return sum(
            [
                -self.variables["taken"][i] * self.problem.list_items[i].weight
                for i in range(self.problem.nb_items)
            ]
        )

    def _internal_heaviest_item(self) -> IntVar:
        if "heaviest_item" not in self.variables:
            heaviest_item_var = self.cp_model.new_int_var(
                name="heaviest_item",
                lb=0,
                ub=int(
                    max(
                        [
                            self.problem.list_items[i].weight
                            for i in range(self.problem.nb_items)
                        ]
                    )
                ),
            )
            self.variables["heaviest_item"] = [heaviest_item_var]
            self.cp_model.add_max_equality(
                target=heaviest_item_var,
                exprs=[
                    self.variables["taken"][i] * self.problem.list_items[i].weight
                    for i in range(self.problem.nb_items)
                ],
            )
        else:
            heaviest_item_var = self.variables["heaviest_item"][0]
        return heaviest_item_var

    def _internal_objective(self, obj: str) -> ObjLinearExprT:
        internal_objective_mapping = {
            "value": self._internal_value,
            "weight": self._internal_weight,
            "heaviest_item": self._internal_heaviest_item,
        }
        if obj in internal_objective_mapping:
            return internal_objective_mapping[obj]()
        else:
            if obj == "weight_violation":
                raise ValueError(
                    "weight_violation cannot be used as objective. "
                    "Indeed, no violation is allowed with this solver."
                )
            else:
                raise ValueError(f"Unknown objective '{obj}'.")

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.Maximize(self._internal_objective(obj))

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Constraint]:
        """

        Args:
            obj: a string representing the desired objective.
                Should be one of `self.problem.get_objective_names()`.
            value: the limiting value.
                If the optimization direction is maximizing, this is a lower bound,
                else this is an upper bound.

        Returns:
            the created constraints.

        """
        return [self.cp_model.Add(self._internal_objective(obj) >= int(value))]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return ["value", "weight", "heaviest_item"]

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> KnapsackSolution:
        """Construct a do solution from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.Value(VARIABLE_NAME)`.

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the intermediate solution, at do format.

        """
        taken = [int(cpsolvercb.Value(var)) for var in self.variables["taken"]]
        return KnapsackSolution(problem=self.problem, list_taken=taken)
