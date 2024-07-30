#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Dict, Iterable, List, Optional

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
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

logger = logging.getLogger(__name__)


class CPSatKnapsackSolver(OrtoolsCPSatSolver, SolverKnapsack, WarmstartMixin):
    def __init__(
        self,
        problem: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables: Dict[str, List[IntVar]] = {}

    def init_model(self, **args: Any) -> None:
        """Init CP model."""
        model = CpModel()
        variables = [
            model.NewBoolVar(name=f"x_{i}") for i in range(self.problem.nb_items)
        ]
        self.variables["taken"] = variables
        self.cp_model = model

        model.Add(-self._intern_weight() <= self.problem.max_capacity)

        self.set_model_objective("value")

    def set_warm_start(self, solution: KnapsackSolution) -> None:
        """Make the solver warm start from the given solution."""
        self.cp_model.clear_hints()
        for i in range(len(solution.list_taken)):
            self.cp_model.AddHint(self.variables["taken"][i], solution.list_taken[i])

    def _intern_value(self) -> LinearExpr:
        return sum(
            [
                self.variables["taken"][i] * self.problem.list_items[i].value
                for i in range(self.problem.nb_items)
            ]
        )

    def _intern_weight(self) -> LinearExpr:
        return sum(
            [
                -self.variables["taken"][i] * self.problem.list_items[i].weight
                for i in range(self.problem.nb_items)
            ]
        )

    def _intern_heaviest_item(self) -> IntVar:
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

    def _intern_objective(self, obj: str) -> ObjLinearExprT:
        intern_objective_mapping = {
            "value": self._intern_value,
            "weight": self._intern_weight,
            "heaviest_item": self._intern_heaviest_item,
        }
        if obj in intern_objective_mapping:
            return intern_objective_mapping[obj]()
        else:
            if obj == "weight_violation":
                raise ValueError(
                    "weight_violation cannot be used as objective. "
                    "Indeed, no violation is allowed with this solver."
                )
            else:
                raise ValueError(f"Unknown objective '{obj}'.")

    def set_model_objective(self, obj: str) -> None:
        self.cp_model.Maximize(self._intern_objective(obj))

    def add_model_constraint(self, obj: str, value: float) -> Iterable[Constraint]:
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
        return [self.cp_model.Add(self._intern_objective(obj) >= int(value))]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def get_model_objectives_available(self) -> List[str]:
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
