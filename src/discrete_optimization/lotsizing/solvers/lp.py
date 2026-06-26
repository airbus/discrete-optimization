#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Callable

from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
)
from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
    ProductionItem,
)


class _BaseLpLotSizingSolver(MilpSolver):
    problem: LotSizingProblem
    variables: dict

    def init_model(self, **kwargs: Any) -> None:
        self.model = self.create_empty_model("LotSizing")
        self.variables = {}
        self._create_main_vars()
        self._set_objective()

    def _create_main_vars(self):
        """Create main decision variables and constraints."""
        total_demands_per_item = {
            item: sum(self.problem.demands[item]) for item in self.problem.items_range
        }
        horizon = self.problem.horizon

        # Boolean variables: is item produced at time t?
        bool_produce_type_time = {
            (item, t): self.add_binary_variable(name=f"bool_prod_item_{item}_time_{t}")
            for item in self.problem.items_range
            for t in range(horizon)
        }

        # At most one item type produced per time period
        for t in range(horizon):
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [
                        bool_produce_type_time[(item, t)]
                        for item in self.problem.items_range
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
                    ub=min(self.problem.capacity_machine, total_demands_per_item[item]),
                    name=f"prod_item_{item}_time_{t}",
                )
                for item in self.problem.items_range
                for t in range(horizon)
            }
            # Link quantity to boolean: if producing, quantity >= 1; if not, quantity = 0
            for item, t in quantity_produce:
                # quantity >= 1 * bool_produce  (if bool=1, then quantity>=1)
                self.add_linear_constraint(
                    quantity_produce[(item, t)] >= bool_produce_type_time[(item, t)]
                )
                # quantity <= M * bool_produce  (if bool=0, then quantity=0)
                self.add_linear_constraint(
                    quantity_produce[(item, t)]
                    <= min(self.problem.capacity_machine, total_demands_per_item[item])
                    * bool_produce_type_time[(item, t)]
                )

        # Delivery variables
        delivery = {
            (item, t): self.add_integer_variable(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"delivery_item_{item}_time_{t}",
            )
            for item in self.problem.items_range
            for t in range(horizon)
        }

        # Stock variables
        stocks = {
            (item, t): self.add_integer_variable(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"stock_item_{item}_time_{t}",
            )
            for item in self.problem.items_range
            for t in range(horizon)
        }

        # Delay variables
        delays = {
            (item, t): self.add_integer_variable(
                lb=0,
                ub=0 if not self.problem.allow_delays else total_demands_per_item[item],
                name=f"delay_item_{item}_time_{t}",
            )
            for item in self.problem.items_range
            for t in range(horizon)
        }

        # Stock balance constraints
        for item in self.problem.items_range:
            for t in range(horizon):
                prev_stock = 0 if t == 0 else stocks[(item, t - 1)]
                # stock[t] = stock[t-1] + production[t] - delivery[t]
                self.add_linear_constraint(
                    stocks[(item, t)]
                    == prev_stock + quantity_produce[(item, t)] - delivery[(item, t)]
                )

                # Delay tracking: delay[t] = delay[t-1] + demand[t] - delivery[t]
                prev_delay = 0 if t == 0 else delays[(item, t - 1)]
                self.add_linear_constraint(
                    delays[(item, t)]
                    == prev_delay + self.problem.demands[item][t] - delivery[(item, t)]
                )

        self.variables["deliveries"] = delivery
        self.variables["bool_productions"] = bool_produce_type_time
        self.variables["productions"] = quantity_produce
        self.variables["delays"] = delays
        self.variables["stocks"] = stocks

    def _create_changeover_variables(self, lookahead: int):
        """Create variables and constraints for changeover costs."""
        produce = self.variables["bool_productions"]
        horizon = self.problem.horizon

        # Binary variables: transition from (item0, t) to (item1, t')
        transition = {
            ((item0, t), (item1, tprime)): self.add_binary_variable(
                name=f"trans_{item0}_{t}_to_{item1}_{tprime}"
            )
            for item0 in self.problem.items_range
            for item1 in self.problem.items_range
            for t in range(horizon)
            for tprime in range(t + 1, min(t + lookahead + 1, horizon))
        }

        # Helper: has any production at time t?
        has_a_production = {
            t: self.add_binary_variable(name=f"has_prod_{t}") for t in range(horizon)
        }

        for t in range(horizon):
            # has_a_production[t] = 1 if sum(produce[item, t]) >= 1
            # Using indicator: sum >= 1 iff has_a_production = 1
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [produce[item, t] for item in self.problem.items_range]
                )
                >= has_a_production[t]
            )
            # Also: if has_a_production=0, then sum = 0
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [produce[item, t] for item in self.problem.items_range]
                )
                <= len(self.problem.items_range) * has_a_production[t]
            )

        # Transition constraints
        for orig, dest in transition:
            item0, t = orig
            item1, tprime = dest

            if tprime == t + 1:
                # Direct consecutive transition: transition = 1 iff both produce
                # transition <= produce[orig]
                self.add_linear_constraint(transition[orig, dest] <= produce[orig])
                # transition <= produce[dest]
                self.add_linear_constraint(transition[orig, dest] <= produce[dest])
                # transition >= produce[orig] + produce[dest] - 1
                self.add_linear_constraint(
                    transition[orig, dest] >= produce[orig] + produce[dest] - 1
                )
            else:
                # Transition with idle times in between
                # transition = 1 iff produce[orig]=1 AND produce[dest]=1 AND all intermediate times are idle
                idle_times = [1 - has_a_production[tau] for tau in range(t + 1, tprime)]
                # transition <= produce[orig]
                self.add_linear_constraint(transition[orig, dest] <= produce[orig])
                # transition <= produce[dest]
                self.add_linear_constraint(transition[orig, dest] <= produce[dest])
                # transition <= each idle indicator
                for idle in idle_times:
                    self.add_linear_constraint(transition[orig, dest] <= idle)

                # transition >= produce[orig] + produce[dest] + sum(idle) - (1 + len(idle))
                self.add_linear_constraint(
                    transition[orig, dest]
                    >= produce[orig]
                    + produce[dest]
                    + self.construct_linear_sum(idle_times)
                    - (1 + len(idle_times))
                )

        # Flow constraint: number of transitions = number of productions - 1
        # (creates a connected sequence)
        self.add_linear_constraint(
            self.construct_linear_sum([transition[x] for x in transition])
            == self.construct_linear_sum([produce[x] for x in produce]) - 1
        )

        self.variables["transitions"] = transition

    def _set_objective(self):
        """Set the objective function based on params_objective_function."""
        horizon = self.problem.horizon
        delays = self.variables["delays"]
        stocks = self.variables["stocks"]
        objectives = []
        self.variables["objectives"] = {}

        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "delays":
                delay_obj = self.construct_linear_sum(
                    [
                        delays[(item, t)]
                        * self.problem.delay_cost_per_type_per_time_per_unit[item]
                        for t in range(horizon)
                        for item in self.problem.items_range
                    ]
                )
                self.variables["objectives"][obj] = delay_obj
                objectives.append(weight * delay_obj)

            if obj == "stock":
                stock_obj = self.construct_linear_sum(
                    [
                        stocks[(item, t)]
                        * self.problem.stock_cost_per_type_per_time_per_unit[item]
                        for t in range(horizon)
                        for item in self.problem.items_range
                    ]
                )
                self.variables["objectives"][obj] = stock_obj
                objectives.append(weight * stock_obj)

            if obj == "changeover":
                self._create_changeover_variables(
                    lookahead=min(5, self.problem.horizon)
                )
                transitions = self.variables["transitions"]
                changeover_obj = self.construct_linear_sum(
                    [
                        self.problem.changeover_costs[key_0[0]][key_1[0]]
                        * transitions[key_0, key_1]
                        for key_0, key_1 in transitions
                    ]
                )
                self.variables["objectives"][obj] = changeover_obj
                objectives.append(weight * changeover_obj)

        self.set_model_objective(self.construct_linear_sum(objectives), minimize=True)

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> LotSizingSolution:
        """Extract solution from the solved model."""
        productions = []
        deliveries = []

        for item, t in self.variables["deliveries"]:
            value = get_var_value_for_current_solution(
                self.variables["deliveries"][(item, t)]
            )
            if value > 0.5:  # Handle numerical precision
                deliveries.append(
                    ProductionItem(item_type=item, quantity=int(round(value)), time=t)
                )

        for item, t in self.variables["productions"]:
            value = get_var_value_for_current_solution(
                self.variables["productions"][(item, t)]
            )
            if value > 0.5:
                productions.append(
                    ProductionItem(item_type=item, quantity=int(round(value)), time=t)
                )

        return LotSizingSolution(
            problem=self.problem, productions=productions, deliveries=deliveries
        )

    def convert_to_variable_values(
        self, solution: LotSizingSolution
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
            key = (prod.item_type, prod.time)
            var_values[self.variables["productions"][key]] = prod.quantity
            var_values[self.variables["bool_productions"][key]] = 1

        # Set delivery values
        for delivery in solution.deliveries:
            key = (delivery.item_type, delivery.time)
            var_values[self.variables["deliveries"][key]] = delivery.quantity

        return var_values


class MathOptLotSizingSolver(_BaseLpLotSizingSolver, OrtoolsMathOptMilpSolver):
    """MathOpt-based MILP solver for lot sizing problems."""

    ...


class GurobiLotSizingSolver(_BaseLpLotSizingSolver, GurobiMilpSolver):
    """Gurobi-based MILP solver for lot sizing problems."""

    ...
