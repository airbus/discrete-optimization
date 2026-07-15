#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Toulbar2 solver for capacitated lot sizing with setup times."""

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.toulbar_tools import ToulbarSolver
from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesLSP,
    CapacitatedSetupTimesSolution,
)
from discrete_optimization.lotsizing.production_solution import (
    ProductionBasedSolution,
    ProductionDecision,
)

try:
    import pytoulbar2

    toulbar_available = True
except ImportError:
    toulbar_available = False


class ToulbarCapacitatedSetupTimesSolver(ToulbarSolver, WarmstartMixin):
    """Toulbar2 solver for capacitated lot sizing with setup times.

    This solver models the problem using Cost Function Networks (CFN) in Toulbar2.
    It handles:
    - Multiple items (with parallel production allowed)
    - Production capacity constraints (including setup times)
    - Inventory holding costs
    - Optional backlog

    Variables:
    - production[item][t]: Production quantity for item in period t
    - setup[item][t]: Binary, 1 if item is produced in period t
    - inventory[item][t]: Inventory level for item at end of period t
    - backlog[item][t]: Backlog quantity for item at end of period t (if allowed)

    Note: This problem does not have changeover costs (inherits from WithoutChangeoverCostsProblem).
    """

    problem: CapacitatedSetupTimesLSP

    def init_model(self, **kwargs) -> None:
        """Initialize the Toulbar2 CFN model.

        Creates variables and constraints for the lot sizing problem.
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        horizon = self.problem.horizon
        nb_items = len(self.problem.items_list)

        # Compute upper bounds
        total_demands = {}
        for item in self.problem.items_list:
            total_demands[item] = int(self.problem.get_total_demand(item))

        # Create CFN model
        model = pytoulbar2.CFN()

        # Variables: production[item][t]
        production_vars = {}
        for item in self.problem.items_list:
            production_vars[item] = [
                model.AddVariable(
                    name=f"prod_{item}_{t}",
                    values=range(total_demands[item] + 1),
                )
                for t in range(horizon)
            ]

        # Variables: setup[item][t] - binary
        setup_vars = {}
        for item in self.problem.items_list:
            setup_vars[item] = [
                model.AddVariable(name=f"setup_{item}_{t}", values=[0, 1])
                for t in range(horizon)
            ]

        # Variables: inventory[item][t]
        inventory_vars = {}
        for item in self.problem.items_list:
            inventory_vars[item] = [
                model.AddVariable(
                    name=f"inv_{item}_{t}",
                    values=range(total_demands[item] + 1),
                )
                for t in range(horizon)
            ]

        # Variables: backlog[item][t] (if backlog allowed)
        backlog_vars = {}
        if self.problem.is_backlog_allowed():
            for item in self.problem.items_list:
                backlog_vars[item] = [
                    model.AddVariable(
                        name=f"backlog_{item}_{t}",
                        values=range(total_demands[item] + 1),
                    )
                    for t in range(horizon)
                ]

        # Constraint: setup[item][t] = 1 iff production[item][t] > 0
        for item in self.problem.items_list:
            for t in range(horizon):
                # If setup = 0, then production must be 0
                # If setup = 1, production can be anything
                top = model.Top

                # Cost function for (setup, production)
                # setup has domain [0, 1], production has domain [0, ..., total_demands[item]]
                # Costs are flattened: [cost(setup=0,prod=0), cost(setup=0,prod=1), ..., cost(setup=1,prod=0), ...]
                setup_prod_costs = []
                for setup_val in [0, 1]:
                    for prod_val in range(total_demands[item] + 1):
                        if setup_val == 0:
                            # If no setup, only prod=0 is allowed
                            cost = 0 if prod_val == 0 else top
                        else:
                            # If setup, any production is ok
                            cost = 0
                        setup_prod_costs.append(cost)

                model.AddFunction(
                    [setup_vars[item][t], production_vars[item][t]],
                    costs=setup_prod_costs,
                )

                # Cost function for (production, setup)
                # production has domain [0, ..., total_demands[item]], setup has domain [0, 1]
                # If prod > 0, setup must be 1
                prod_setup_costs = []
                for prod_val in range(total_demands[item] + 1):
                    for setup_val in [0, 1]:
                        if prod_val == 0:
                            # If no production, any setup is ok
                            cost = 0
                        else:
                            # If production > 0, setup must be 1
                            cost = top if setup_val == 0 else 0
                        prod_setup_costs.append(cost)

                model.AddFunction(
                    [production_vars[item][t], setup_vars[item][t]],
                    costs=prod_setup_costs,
                )

        # Constraint: inventory balance
        for item in self.problem.items_list:
            demand_details = [
                int(self.problem.get_demand(item, t)) for t in range(horizon)
            ]

            for t in range(horizon):
                if self.problem.is_backlog_allowed():
                    # With backlog: inv[t] = inv[t-1] + prod[t] - delivery[t]
                    #               backlog[t] = backlog[t-1] + demand[t] - delivery[t]
                    # Combining: inv[t] - inv[t-1] - prod[t] + backlog[t-1] - backlog[t] + demand[t] = 0
                    if t == 0:
                        # inv[0] - prod[0] - backlog[0] + demand[0] = 0
                        model.AddLinearConstraint(
                            [1, -1, -1],
                            [
                                inventory_vars[item][0],
                                production_vars[item][0],
                                backlog_vars[item][0],
                            ],
                            operand="==",
                            rightcoef=-demand_details[0],
                        )
                    else:
                        # inv[t] - inv[t-1] - prod[t] + backlog[t-1] - backlog[t] + demand[t] = 0
                        model.AddLinearConstraint(
                            [1, -1, -1, 1, -1],
                            [
                                inventory_vars[item][t],
                                inventory_vars[item][t - 1],
                                production_vars[item][t],
                                backlog_vars[item][t - 1],
                                backlog_vars[item][t],
                            ],
                            operand="==",
                            rightcoef=-demand_details[t],
                        )
                else:
                    # Without backlog: standard inventory balance
                    if t == 0:
                        # inv[0] = prod[0] - demand[0]
                        model.AddLinearConstraint(
                            [1, -1],
                            [inventory_vars[item][0], production_vars[item][0]],
                            operand="==",
                            rightcoef=-demand_details[0],
                        )
                    else:
                        # inv[t] = inv[t-1] + prod[t] - demand[t]
                        model.AddLinearConstraint(
                            [1, -1, -1],
                            [
                                inventory_vars[item][t],
                                inventory_vars[item][t - 1],
                                production_vars[item][t],
                            ],
                            operand="==",
                            rightcoef=-demand_details[t],
                        )

        # Constraint: capacity (production time + setup time <= capacity)
        for t in range(horizon):
            capacity_vars = []
            capacity_coeffs = []

            for item in self.problem.items_list:
                # Production time contribution
                prod_time_per_unit = int(
                    self.problem.get_production_time_per_unit(item, t)
                )
                if prod_time_per_unit > 0:
                    capacity_vars.append(production_vars[item][t])
                    capacity_coeffs.append(prod_time_per_unit)

                # Setup time contribution
                setup_time = int(self.problem.get_setup_time(item, t))
                if setup_time > 0:
                    capacity_vars.append(setup_vars[item][t])
                    capacity_coeffs.append(setup_time)

            if capacity_vars:
                available_capacity = int(self.problem.get_available_production_time(t))
                model.AddLinearConstraint(
                    capacity_coeffs,
                    capacity_vars,
                    operand="<=",
                    rightcoef=available_capacity,
                )

        # Constraint: at most one item produced per period (if not allowing parallel)
        if not self.problem.allows_parallel_production():
            for t in range(horizon):
                # Sum of setups <= 1
                model.AddLinearConstraint(
                    [1] * nb_items,
                    [setup_vars[item][t] for item in self.problem.items_list],
                    operand="<=",
                    rightcoef=1,
                )

        # Get objective weights from params_objective_function
        obj_weights = {}
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            obj_weights[obj] = weight

        # Objective: inventory costs
        inv_weight = obj_weights.get("inventory_cost", 1.0)
        for item in self.problem.items_list:
            for t in range(horizon):
                inv_cost = int(
                    self.problem.get_inventory_cost_per_unit(item, t) * inv_weight
                )
                if inv_cost > 0:
                    model.AddFunction(
                        [inventory_vars[item][t]],
                        [i * inv_cost for i in range(total_demands[item] + 1)],
                    )

        # Objective: backlog costs (if allowed)
        if self.problem.is_backlog_allowed():
            backlog_weight = obj_weights.get("backlog_cost", 1.0)
            for item in self.problem.items_list:
                for t in range(horizon):
                    backlog_cost = int(
                        self.problem.get_backlog_cost_per_unit(item, t) * backlog_weight
                    )
                    if backlog_cost > 0:
                        model.AddFunction(
                            [backlog_vars[item][t]],
                            [i * backlog_cost for i in range(total_demands[item] + 1)],
                        )

        # Store variables for warm-start
        self.production_vars = production_vars
        self.setup_vars = setup_vars
        self.inventory_vars = inventory_vars
        self.backlog_vars = backlog_vars

        self.model = model

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> CapacitatedSetupTimesSolution:
        """Convert Toulbar2 solution to problem solution.

        Args:
            solution_from_toulbar2: Tuple of (values, cost, status) from Toulbar2

        Returns:
            CapacitatedSetupTimesSolution
        """
        values = solution_from_toulbar2[0]

        # Extract variable values
        # Order: production vars for all items/periods, then setup, inventory, backlog (if enabled)
        horizon = self.problem.horizon

        idx = 0
        production_values = {}
        for item in self.problem.items_list:
            production_values[item] = values[idx : idx + horizon]
            idx += horizon

        # Skip setup variables
        idx += len(self.problem.items_list) * horizon

        # Extract inventory values
        inventory_values = {}
        for item in self.problem.items_list:
            inventory_values[item] = values[idx : idx + horizon]
            idx += horizon

        # Extract backlog values (if applicable)
        if self.problem.is_backlog_allowed():
            for item in self.problem.items_list:
                # Skip backlog variables (not needed for solution construction)
                idx += horizon

        # Build production decisions
        productions = []
        for item in self.problem.items_list:
            for t in range(horizon):
                qty = production_values[item][t]
                if qty > 0:
                    productions.append(
                        ProductionDecision(item=item, period=t, quantity=qty)
                    )

        # Create solution WITHOUT explicit deliveries (let ProductionBasedSolution compute them)
        # This is simpler and matches the expected behavior
        sol = ProductionBasedSolution(problem=self.problem, productions=productions)

        return sol

    def set_warm_start(self, solution: CapacitatedSetupTimesSolution) -> None:
        """Set warm start from a solution.

        Args:
            solution: Initial solution to warm-start from
        """
        horizon = self.problem.horizon

        var_idx = 0

        # Set production variables
        for item in self.problem.items_list:
            for t in range(horizon):
                prod = solution.get_production_quantity(item=item, period=t)
                self.model.CFN.wcsp.setBestValue(var_idx, prod)
                var_idx += 1

        # Set setup variables
        for item in self.problem.items_list:
            for t in range(horizon):
                prod = solution.get_production_quantity(item=item, period=t)
                setup = 1 if prod > 0 else 0
                self.model.CFN.wcsp.setBestValue(var_idx, setup)
                var_idx += 1

        # Set inventory variables
        for item in self.problem.items_list:
            for t in range(horizon):
                inv = solution.get_inventory_level(item=item, period=t)
                self.model.CFN.wcsp.setBestValue(var_idx, inv)
                var_idx += 1

        # Set backlog variables (if applicable)
        if self.problem.is_backlog_allowed():
            for item in self.problem.items_list:
                for t in range(horizon):
                    # Compute backlog from solution
                    cumulative_demand = sum(
                        self.problem.get_demand(item, period) for period in range(t + 1)
                    )
                    cumulative_delivery = sum(
                        solution.get_delivery_quantity(item, period)
                        for period in range(t + 1)
                    )
                    backlog = max(0, cumulative_demand - cumulative_delivery)
                    self.model.CFN.wcsp.setBestValue(var_idx, backlog)
                    var_idx += 1
