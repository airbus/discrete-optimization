#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Callable

import gurobipy

from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    InequalitySense,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem.problem import (
    UncapacitatedSingleItemLSP,
    UncapacitatedSingleItemSolution,
)


class _BaseLpUncapacitatedSingleItemSolver(MilpSolver):
    problem: UncapacitatedSingleItemLSP
    variables: dict

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.model = self.create_empty_model("uncapacitated_single_item")
        self.create_vars()
        self._set_objective()

    def create_vars(self):
        self.variables = {}
        bool_production_per_time = {
            t: self.add_binary_variable(name=f"is_producing_per_time_{t}")
            for t in range(self.problem.horizon)
        }
        production_per_time = {
            t: self.add_integer_variable(
                lb=0,
                ub=self.problem.get_total_demand(self.problem.items_list[0]),
                name=f"production_per_time_{t}",
            )
            for t in range(self.problem.horizon)
        }
        inventory_per_time = {
            t: self.add_integer_variable(
                lb=0,
                ub=self.problem.get_total_demand(self.problem.items_list[0]),
                name=f"stock_{t}",
            )
            for t in range(self.problem.horizon)
        }
        for t in range(self.problem.horizon):
            self.add_linear_constraint_with_indicator(
                binvar=bool_production_per_time[t],
                binval=1,
                lhs=production_per_time[t],
                sense=InequalitySense.GREATER_OR_EQUAL,
                rhs=1,
                penalty_coeff=self.problem.get_total_demand(self.problem.items_list[0]),
            )
            self.add_linear_constraint_with_indicator(
                binvar=bool_production_per_time[t],
                binval=0,
                lhs=production_per_time[t],
                sense=InequalitySense.EQUAL,
                rhs=0,
                penalty_coeff=self.problem.get_total_demand(self.problem.items_list[0]),
            )
        for t in range(self.problem.horizon):
            if t == 0:
                self.add_linear_constraint(
                    inventory_per_time[t]
                    == production_per_time[t]
                    - self.problem.get_demand(self.problem.items_list[0], t)
                )
            else:
                self.add_linear_constraint(
                    inventory_per_time[t]
                    == inventory_per_time[t - 1]
                    + production_per_time[t]
                    - self.problem.get_demand(self.problem.items_list[0], t)
                )
        self.variables["inventory"] = inventory_per_time
        self.variables["production_per_time"] = production_per_time
        self.variables["bool_production_per_time"] = bool_production_per_time

    def _set_objective(self):
        """Set the objective function."""
        self.variables["objectives"] = {}
        horizon = self.problem.horizon
        item = 0
        objectives = []
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "setup_cost":
                # Setup cost: sum_t (s_t * Y_t)
                setup_cost_terms = []
                for t in range(horizon):
                    cost = int(self.problem.get_setup_cost(item, t))
                    if cost > 0:
                        setup_cost_terms.append(
                            cost * self.variables["bool_production_per_time"][t]
                        )
                self.variables["objectives"]["setup_cost"] = self.construct_linear_sum(
                    setup_cost_terms
                )
                objectives.append(weight * self.variables["objectives"]["setup_cost"])

            if obj == "production_cost":
                # Production cost: sum_t (v_t * X_t)
                production_cost_terms = []
                for t in range(horizon):
                    cost_per_unit = int(
                        self.problem.get_production_cost_per_unit(item, t)
                    )
                    if cost_per_unit > 0:
                        production_cost_terms.append(
                            cost_per_unit * self.variables["production_per_time"][t]
                        )
                self.variables["objectives"]["production_cost"] = (
                    self.construct_linear_sum(production_cost_terms)
                )
                objectives.append(
                    weight * self.variables["objectives"]["production_cost"]
                )

            if obj == "inventory_cost":
                # Inventory cost: sum_t (c_t * I_t)
                inventory_cost_terms = []
                for t in range(horizon):
                    cost_per_unit = int(
                        self.problem.get_inventory_cost_per_unit(item, t)
                    )
                    if cost_per_unit > 0:
                        inventory_cost_terms.append(
                            cost_per_unit * self.variables["inventory"][t]
                        )
                self.variables["objectives"]["inventory_cost"] = (
                    self.construct_linear_sum(inventory_cost_terms)
                )
                objectives.append(
                    weight * self.variables["objectives"]["inventory_cost"]
                )
        # Minimize total cost
        if objectives:
            self.set_model_objective(
                self.construct_linear_sum(objectives), minimize=True
            )

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> UncapacitatedSingleItemSolution:
        production_periods = []
        production_quantity = []
        for t in self.variables["production_per_time"]:
            val = get_var_value_for_current_solution(
                self.variables["production_per_time"][t]
            )
            if val > 0:
                production_periods.append(t)
                production_quantity.append(val)
        for t in self.variables["inventory"]:
            print(
                "Inventory",
                t,
                get_var_value_for_current_solution(self.variables["inventory"][t]),
            )
        return UncapacitatedSingleItemSolution(
            problem=self.problem,
            production_periods=production_periods,
            production_quantities=production_quantity,
        )


class MathoptUncapacitatedSingleItemSolver(
    _BaseLpUncapacitatedSingleItemSolver, OrtoolsMathOptMilpSolver
): ...


class GurobiUncapacitatedSingleItemSolver(
    _BaseLpUncapacitatedSingleItemSolver, GurobiMilpSolver
):
    def convert_to_variable_values(
        self, solution: UncapacitatedSingleItemSolution
    ) -> dict[gurobipy.Var, float]:
        dict_variable = {}
        for t in range(self.problem.horizon):
            prod_t = solution.get_production_quantity(
                item=self.problem.items_list[0], period=t
            )
            dict_variable[self.variables["production_per_time"][t]] = prod_t
            dict_variable[self.variables["bool_production_per_time"][t]] = (
                1 if prod_t > 0 else 0
            )
            inventory_t = solution.get_inventory_level(
                item=self.problem.items_list[0], period=t
            )
            dict_variable[self.variables["inventory"][t]] = int(inventory_t)
        return dict_variable
