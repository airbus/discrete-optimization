#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional, Union

from ortools.algorithms.python import knapsack_solver
from ortools.linear_solver import pywraplp
from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
    VariableType,
)
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers import KnapsackSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, Constr, GenConstr, MConstr, Model, QConstr, Var, quicksum


logger = logging.getLogger(__name__)


class _BaseLpKnapsackSolver(MilpSolver, KnapsackSolver):
    """Base class for Knapsack LP solvers."""

    def __init__(
        self,
        problem: KnapsackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variable_decision: dict[str, dict[int, VariableType]] = {}
        self.constraints_dict: dict[
            str,
            Union[
                "Constr",
                "QConstr",
                "MConstr",
                "GenConstr",
                "mip.Constr",
                "mathopt.LinearConstraint",
            ],
        ] = {}
        self.description_variable_description: dict[str, dict[str, Any]] = {}
        self.description_constraint: dict[str, dict[str, str]] = {}

    def init_model(self, **kwargs: Any) -> None:
        self.model = self.create_empty_model("Knapsack")
        self.variable_decision = {"x": {}}
        self.description_variable_description = {
            "x": {
                "shape": self.problem.nb_items,
                "type": bool,
                "descr": "dictionary with key the item index \
                                                                 and value the boolean value corresponding \
                                                                 to taking the item or not",
            }
        }
        self.description_constraint["weight"] = {
            "descr": "sum of weight of used items doesn't exceed max capacity"
        }
        weight = {}
        list_item = self.problem.list_items
        max_capacity = self.problem.max_capacity
        x = {}
        for item in list_item:
            i = item.index
            x[i] = self.add_binary_variable(name="x_" + str(i))
            weight[i] = item.weight
        self.set_model_objective(
            self.construct_linear_sum(item.value * x[item.index] for item in list_item),
            minimize=False,
        )
        self.variable_decision["x"] = x
        self.constraints_dict["weight"] = self.add_linear_constraint(
            self.construct_linear_sum([weight[i] * x[i] for i in x]) <= max_capacity
        )

    def convert_to_variable_values(
        self, solution: KnapsackSolution
    ) -> dict[Any, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return {
            self.variable_decision["x"][variable_decision_key]: solution.list_taken[i]
            for i, variable_decision_key in enumerate(
                sorted(self.variable_decision["x"])
            )
        }

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> KnapsackSolution:
        weight = 0.0
        value_kp = 0.0
        xs = {}
        for (
            variable_decision_key,
            variable_decision_value,
        ) in self.variable_decision["x"].items():
            value = get_var_value_for_current_solution(variable_decision_value)
            if value <= 0.1:
                xs[variable_decision_key] = 0
                continue
            xs[variable_decision_key] = 1
            weight += self.problem.index_to_item[variable_decision_key].weight
            value_kp += self.problem.index_to_item[variable_decision_key].value

        return KnapsackSolution(
            problem=self.problem,
            value=value_kp,
            weight=weight,
            list_taken=[xs[e] for e in sorted(xs)],
        )


class GurobiKnapsackSolver(GurobiMilpSolver, _BaseLpKnapsackSolver):
    def init_model(self, **kwargs: Any) -> None:
        _BaseLpKnapsackSolver.init_model(self, **kwargs)
        self.model.update()

    def convert_to_variable_values(
        self, solution: KnapsackSolution
    ) -> dict[Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return _BaseLpKnapsackSolver.convert_to_variable_values(self, solution)


class MathOptKnapsackSolver(OrtoolsMathOptMilpSolver, _BaseLpKnapsackSolver):
    def convert_to_variable_values(
        self, solution: KnapsackSolution
    ) -> dict[mathopt.Variable, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start() to provide a suitable SolutionHint.variable_values.
        See https://or-tools.github.io/docs/pdoc/ortools/math_opt/python/model_parameters.html#SolutionHint
        for more information.

        """
        return _BaseLpKnapsackSolver.convert_to_variable_values(self, solution)


class CbcKnapsackSolver(KnapsackSolver):
    def __init__(
        self,
        problem: KnapsackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.model: Optional[pywraplp.Solver] = None
        self.variable_decision: dict[str, dict[int, Any]] = {}
        self.constraints_dict: dict[str, Any] = {}
        self.description_variable_description: dict[str, dict[str, Any]] = {}
        self.description_constraint: dict[str, dict[str, str]] = {}

    def init_model(
        self, warm_start: Optional[dict[int, int]] = None, **kwargs: Any
    ) -> None:
        if warm_start is None:
            warm_start = {}
        self.description_variable_description = {
            "x": {
                "shape": self.problem.nb_items,
                "type": bool,
                "descr": "dictionary with key the item index \
                                                                 and value the boolean value corresponding \
                                                                 to taking the item or not",
            }
        }
        self.description_constraint["weight"] = {
            "descr": "sum of weight of used items doesn't exceed max capacity"
        }
        S = pywraplp.Solver("knapsack", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        x = {}
        weight = {}
        value = {}
        list_item = self.problem.list_items
        max_capacity = self.problem.max_capacity
        for item in list_item:
            i = item.index
            x[i] = S.BoolVar("x_" + str(i))
            if i in warm_start:
                S.SetHint([x[i]], [warm_start[i]])
            weight[i] = item.weight
            value[i] = item.value
        self.constraints_dict["weight"] = S.Add(
            S.Sum([x[i] * weight[i] for i in x]) <= max_capacity
        )
        value_knap = S.Sum([x[i] * value[i] for i in x])
        S.Maximize(value_knap)
        self.model = S
        self.variable_decision["x"] = x

    def solve(self, **kwargs: Any) -> ResultStorage:
        if self.model is None:
            self.init_model()
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.model.SetTimeLimit(60000)
        res = self.model.Solve()
        resdict = {
            0: "OPTIMAL",
            1: "FEASIBLE",
            2: "INFEASIBLE",
            3: "UNBOUNDED",
            4: "ABNORMAL",
            5: "MODEL_INVALID",
            6: "NOT_SOLVED",
        }
        logger.debug(f"Result: {resdict[res]}")
        objective = self.model.Objective().Value()
        xs = {}
        x = self.variable_decision["x"]
        weight = 0.0
        for i in x:
            sv = x[i].solution_value()
            if sv >= 0.5:
                xs[i] = 1
                weight += self.problem.index_to_item[i].weight
            else:
                xs[i] = 0
        sol = KnapsackSolution(
            problem=self.problem,
            value=objective,
            weight=weight,
            list_taken=[xs[e] for e in sorted(xs)],
        )
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )


class OrtoolsKnapsackSolver(KnapsackSolver):
    model: Optional[knapsack_solver.KnapsackSolver] = None

    def init_model(self, **kwargs: Any) -> None:
        solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample",
        )
        list_item = self.problem.list_items
        max_capacity = self.problem.max_capacity
        values = [item.value for item in list_item]
        weights = [[item.weight for item in list_item]]
        capacities = [max_capacity]
        solver.init(values, weights, capacities)
        self.model = solver

    def solve(self, **kwargs: Any) -> ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        computed_value = self.model.solve()
        logger.debug(f"Total value = {computed_value}")
        xs = {}
        weight = 0
        value = 0
        for i in range(self.problem.nb_items):
            if self.model.best_solution_contains(i):
                weight += self.problem.list_items[i].weight
                value += self.problem.list_items[i].value
                xs[self.problem.list_items[i].index] = 1
            else:
                xs[self.problem.list_items[i].index] = 0
        sol = KnapsackSolution(
            problem=self.problem,
            value=value,
            weight=weight,
            list_taken=[xs[e] for e in sorted(xs)],
        )
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )
