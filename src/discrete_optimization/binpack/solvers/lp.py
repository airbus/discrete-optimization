#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from ortools.math_opt.python import mathopt

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.lp_tools import (
    ConstraintType,
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
    VariableType,
)
from discrete_optimization.generic_tools.unsat_tools import MetaConstraint


class _BaseLpBinPackSolver(MilpSolver):
    problem: BinPackProblem

    def __init__(
        self,
        problem: BinPackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.model = self.create_empty_model("bin_pack")
        upper_bound = kwargs.get("upper_bound", self.problem)
        used_bin = {}
        variables_allocation = {}
        for bin in range(upper_bound):
            used_bin[bin] = self.add_binary_variable(f"used_{bin}")
            if bin >= 1:
                self.add_linear_constraint(
                    used_bin[bin] <= used_bin[bin - 1], name=f"symm_used_{bin}"
                )
        for i in range(self.problem.nb_items):
            for bin in range(upper_bound):
                variables_allocation[(i, bin)] = self.add_binary_variable(
                    f"alloc_{i}_{bin}"
                )
                self.add_linear_constraint(
                    used_bin[bin] >= variables_allocation[(i, bin)]
                )
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [variables_allocation[(i, bin)] for bin in range(upper_bound)]
                )
                == 1,
                name=f"one_bin_for_{i}",
            )

        if self.problem.has_constraint:
            for i, j in self.problem.incompatible_items:
                for bin in range(upper_bound):
                    self.add_linear_constraint(
                        variables_allocation[(i, bin)] + variables_allocation[(j, bin)]
                        <= 1,
                        name=f"incomp_{i}_{j}_{bin}",
                    )
        for bin in range(upper_bound):
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [
                        self.problem.list_items[i].weight
                        * variables_allocation[(i, bin)]
                        for i in range(self.problem.nb_items)
                    ]
                )
                <= self.problem.capacity_bin,
                name=f"capacity_bin_{bin}",
            )
        self.variables["allocation"] = variables_allocation
        self.variables["used"] = used_bin
        self.set_model_objective(
            self.construct_linear_sum([used_bin[bin] for bin in used_bin]),
            minimize=True,
        )

    def convert_to_variable_values(self, solution: BinPackSolution) -> dict[Any, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        # Init all variables to 0
        hinted_variables = {var: 0 for var in self.variables["allocation"].values()}
        hinted_variables.update({var: 0 for var in self.variables["used"].values()})
        # Set var(node, color) to 1 according to the solution
        for i, bin in enumerate(solution.allocation):
            variable_decision_key = (i, bin)
            hinted_variables[self.variables["allocation"][variable_decision_key]] = 1
            hinted_variables[self.variables["used"][bin]] = 1
        return hinted_variables

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> BinPackSolution:
        allocation = [None] * self.problem.nb_items
        for (
            variable_decision_key,
            variable_decision_value,
        ) in self.variables["allocation"].items():
            value = get_var_value_for_current_solution(variable_decision_value)
            if value >= 0.5:
                item = variable_decision_key[0]
                bin = variable_decision_key[1]
                allocation[item] = bin
        return BinPackSolution(self.problem, allocation)


class MathOptBinPackSolver(_BaseLpBinPackSolver, OrtoolsMathOptMilpSolver):
    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[mathopt.Variable, float]:
        return _BaseLpBinPackSolver.convert_to_variable_values(self, solution)
