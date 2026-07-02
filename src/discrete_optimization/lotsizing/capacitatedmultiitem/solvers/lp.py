#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""MILP solvers for capacitated multi-item lot sizing problem."""

from typing import Any, Callable

from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
)
from discrete_optimization.lotsizing import ProductionDecision
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
    CapacitatedMultiItemSolution,
)


class _BaseLpCapacitatedLotSizingSolver(MilpSolver):
    """Base MILP solver for capacitated multi-item lot sizing."""

    problem: CapacitatedMultiItemLSP
    variables: dict

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the MILP model."""
        self.model = self.create_empty_model("CapacitatedMultiItemLotSizing")
        self.variables = {}
        self._create_main_vars()
        self._set_objective()

    def _create_main_vars(self):
        """Create main decision variables and constraints."""
        total_demands_per_item = {
            item: self.problem.get_total_demand(item)
            for item in self.problem.items_list
        }
        horizon = self.problem.horizon

        # Boolean variables: is item produced at time t?
        bool_produce_type_time = {
            (item, t): self.add_binary_variable(name=f"bool_prod_item_{item}_time_{t}")
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # At most one item type produced per time period
        for t in range(horizon):
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [
                        bool_produce_type_time[(item, t)]
                        for item in self.problem.items_list
                    ]
                )
                <= 1
            )

        # Quantity produced variables
        if self.problem.is_binary:
            quantity_produce = bool_produce_type_time
        else:
            quantity_produce = {
                (item, t): self.add_integer_variable(
                    lb=0,
                    ub=min(
                        int(self.problem.get_available_production_time(t)),
                        total_demands_per_item[item],
                    ),
                    name=f"prod_item_{item}_time_{t}",
                )
                for item in self.problem.items_list
                for t in range(horizon)
            }
            # Link quantity to boolean: if producing, quantity >= 1; if not, quantity = 0
            for item, t in quantity_produce:
                # quantity >= 1 * bool_produce  (if bool=1, then quantity>=1)
                self.add_linear_constraint(
                    quantity_produce[(item, t)] >= bool_produce_type_time[(item, t)]
                )
                # quantity <= M * bool_produce  (if bool=0, then quantity=0)
                M = min(
                    int(self.problem.get_available_production_time(t)),
                    total_demands_per_item[item],
                )
                self.add_linear_constraint(
                    quantity_produce[(item, t)] <= M * bool_produce_type_time[(item, t)]
                )

        # Delivery variables
        delivery = {
            (item, t): self.add_integer_variable(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"delivery_item_{item}_time_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Stock variables
        stocks = {
            (item, t): self.add_integer_variable(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"stock_item_{item}_time_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Delay variables
        delays = {
            (item, t): self.add_integer_variable(
                lb=0,
                ub=0
                if not self.problem.is_backlog_allowed()
                else total_demands_per_item[item],
                name=f"delay_item_{item}_time_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Stock balance constraints
        for item in self.problem.items_list:
            for t in range(horizon):
                prev_stock = 0 if t == 0 else stocks[(item, t - 1)]
                demand = self.problem.get_demand(item, t)

                # stock[t] = stock[t-1] + production[t] - delivery[t]
                self.add_linear_constraint(
                    stocks[(item, t)]
                    == prev_stock + quantity_produce[(item, t)] - delivery[(item, t)]
                )

                # Delay tracking: delay[t] = delay[t-1] + demand[t] - delivery[t]
                prev_delay = 0 if t == 0 else delays[(item, t - 1)]
                self.add_linear_constraint(
                    delays[(item, t)] == prev_delay + demand - delivery[(item, t)]
                )

        self.variables["deliveries"] = delivery
        self.variables["bool_productions"] = bool_produce_type_time
        self.variables["productions"] = quantity_produce
        self.variables["delays"] = delays
        self.variables["stocks"] = stocks

    def _create_changeover_variables(self):
        """Create variables and constraints for changeover costs using flow-based model.

        Models production sequence as a shortest path through production events:
        - Dummy start node connects to first production
        - Each production connects to next production (with lookahead window)
        - Last production connects to dummy end node
        - Flow conservation ensures valid sequence
        """
        produce = self.variables["bool_productions"]
        horizon = self.problem.horizon

        # Lookahead window: how far ahead can we transition?
        # If horizon >> total_demand, we have many idle periods so limit lookahead
        total_demand = sum(
            self.problem.get_total_demand(item) for item in self.problem.items_list
        )
        lookahead = min(10, max(1, horizon - total_demand + 1))

        # Binary variables: transition from (item0, t) to (item1, t')
        transition = {}

        # Transitions from dummy start to first productions
        for item in self.problem.items_list:
            for t in range(min(lookahead + 1, horizon)):
                transition[("dummy", -1), (item, t)] = self.add_binary_variable(
                    name=f"trans_start_to_{item}_{t}"
                )

        # Transitions between productions
        for item0 in self.problem.items_list:
            for t in range(horizon):
                for item1 in self.problem.items_list:
                    for tprime in range(t + 1, min(t + lookahead + 1, horizon)):
                        transition[(item0, t), (item1, tprime)] = (
                            self.add_binary_variable(
                                name=f"trans_{item0}_{t}_to_{item1}_{tprime}"
                            )
                        )

        # Transitions to dummy end from last productions
        for item in self.problem.items_list:
            for t in range(max(0, horizon - lookahead - 1), horizon):
                transition[(item, t), ("dummy", horizon)] = self.add_binary_variable(
                    name=f"trans_{item}_{t}_to_end"
                )

        # Flow conservation constraints
        nodes = set([k[0] for k in transition] + [k[1] for k in transition])

        for node in nodes:
            incoming = [k for k in transition if k[1] == node]
            outgoing = [k for k in transition if k[0] == node]

            if node == ("dummy", -1):
                # Start: exactly one outgoing
                self.add_linear_constraint(
                    self.construct_linear_sum([transition[k] for k in outgoing]) == 1
                )
            elif node == ("dummy", horizon):
                # End: exactly one incoming
                self.add_linear_constraint(
                    self.construct_linear_sum([transition[k] for k in incoming]) == 1
                )
            else:
                # Production node: incoming = outgoing = bool_produce
                self.add_linear_constraint(
                    self.construct_linear_sum([transition[k] for k in incoming])
                    == produce[node]
                )
                self.add_linear_constraint(
                    self.construct_linear_sum([transition[k] for k in outgoing])
                    == produce[node]
                )

        self.variables["transitions"] = transition

    def _set_objective(self):
        """Set the objective function."""
        horizon = self.problem.horizon
        delays = self.variables["delays"]
        stocks = self.variables["stocks"]
        objectives = []

        # Delay cost
        delay_obj = self.construct_linear_sum(
            [
                delays[(item, t)] * self.problem.get_backlog_cost_per_unit(item, t)
                for t in range(horizon)
                for item in self.problem.items_list
            ]
        )
        objectives.append(delay_obj)

        # Stock cost
        stock_obj = self.construct_linear_sum(
            [
                stocks[(item, t)] * self.problem.get_inventory_cost_per_unit(item, t)
                for t in range(horizon)
                for item in self.problem.items_list
            ]
        )
        objectives.append(stock_obj)

        # Changeover cost
        self._create_changeover_variables()
        transitions = self.variables["transitions"]
        changeover_obj = self.construct_linear_sum(
            [
                self.problem.get_changeover_cost(key_0[0], key_1[0])
                * transitions[key_0, key_1]
                for key_0, key_1 in transitions
                if key_0[0] != "dummy" and key_1[0] != "dummy"
            ]
        )
        objectives.append(changeover_obj)

        self.set_model_objective(self.construct_linear_sum(objectives), minimize=True)

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> CapacitatedMultiItemSolution:
        """Extract solution from the solved model."""
        productions = []

        for item, t in self.variables["productions"]:
            value = get_var_value_for_current_solution(
                self.variables["productions"][(item, t)]
            )
            if value > 0.5:  # Handle numerical precision
                productions.append(
                    ProductionDecision(item=item, period=t, quantity=int(round(value)))
                )

        return CapacitatedMultiItemSolution(
            problem=self.problem, productions=productions
        )

    def convert_to_variable_values(
        self, solution: CapacitatedMultiItemSolution
    ) -> dict[Any, float]:
        """Convert a solution to variable values for warm-start."""
        var_values = {}

        # Initialize all variables to 0
        for item, t in self.variables["productions"]:
            var_values[self.variables["productions"][(item, t)]] = 0
            var_values[self.variables["bool_productions"][(item, t)]] = 0

        for item, t in self.variables["deliveries"]:
            var_values[self.variables["deliveries"][(item, t)]] = 0

        # Set production values
        for prod in solution.productions:
            key = (prod.item, prod.period)
            var_values[self.variables["productions"][key]] = prod.quantity
            var_values[self.variables["bool_productions"][key]] = 1

        # Compute deliveries from solution
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                delivery = solution.get_delivery_quantity(item, t)
                if delivery > 0:
                    var_values[self.variables["deliveries"][(item, t)]] = delivery

        return var_values


class MathOptCapacitatedLotSizingSolver(
    _BaseLpCapacitatedLotSizingSolver, OrtoolsMathOptMilpSolver
):
    """MathOpt-based MILP solver for capacitated multi-item lot sizing."""

    ...


class GurobiCapacitatedLotSizingSolver(
    _BaseLpCapacitatedLotSizingSolver, GurobiMilpSolver
):
    """Gurobi-based MILP solver for capacitated multi-item lot sizing."""

    ...
