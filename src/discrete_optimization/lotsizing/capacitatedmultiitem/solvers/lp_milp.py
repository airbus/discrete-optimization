#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""MILP solver for capacitated multi-item lot sizing problem.

Implementation of the milp3 model from:
Ceschia, Di Gaspero, Schaerf (2017) - "Solving discrete lot-sizing and
scheduling by simulated annealing and mixed integer programming"
"""

import logging
from typing import Any

import gurobipy

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from discrete_optimization.lotsizing import ProductionDecision
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
    CapacitatedMultiItemSolution,
)

try:
    import gurobipy as grb
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True

logger = logging.getLogger(__name__)


class MilpCapacitatedLotSizingSolver(GurobiMilpSolver):
    """MILP solver for capacitated multi-item lot sizing using milp3 formulation.

    Variables:
    - x[i,t]: binary, 1 if item i is produced in period t
    - y[i,t]: binary, 1 if machine is ready for item i at start of period t
    - s[i,t]: integer, stock of item i at end of period t
    - z[i,j,t]: binary, 1 if changeover from i to j happens in period t
    """

    problem: CapacitatedMultiItemLSP

    hyperparameters = [
        CategoricalHyperparameter(
            name="use_valid_inequalities",
            choices=[True, False],
            default=True,
        ),
    ]

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the MILP model."""
        if not gurobi_available:
            raise RuntimeError("Gurobi is required for MilpLotSizingSolver")

        use_valid_inequalities = kwargs.get("use_valid_inequalities", True)

        # Create model
        self.model = grb.Model("CapacitatedMultiItemLotSizing_MILP3")

        # Shortcuts
        horizon = self.problem.horizon
        items = self.problem.items_list
        m = len(items)

        # Get demands
        demands = {}
        for item in items:
            for t in range(horizon):
                demands[item, t] = self.problem.get_demand(item, t)

        # Variables
        # x[i,t]: 1 if item i is produced in period t
        x = {}
        for i in items:
            for t in range(horizon):
                x[i, t] = self.model.addVar(vtype=grb.GRB.BINARY, name=f"x_{i}_{t}")

        # y[i,t]: 1 if machine is ready for item i in period t
        y = {}
        for i in items:
            for t in range(horizon):
                y[i, t] = self.model.addVar(vtype=grb.GRB.BINARY, name=f"y_{i}_{t}")

        # s[i,t]: stock of item i at end of period t
        max_stock = sum(self.problem.get_total_demand(i) for i in items)
        s = {}
        for i in items:
            for t in range(horizon):
                s[i, t] = self.model.addVar(
                    vtype=grb.GRB.INTEGER, lb=0, ub=max_stock, name=f"s_{i}_{t}"
                )

        # z[i,j,t]: 1 if changeover from item i to j in period t
        z = {}
        for i in items:
            for j in items:
                if i != j:
                    for t in range(1, horizon):  # Start from period 1
                        z[i, j, t] = self.model.addVar(
                            vtype=grb.GRB.BINARY, name=f"z_{i}_{j}_{t}"
                        )

        self.model.update()

        # Constraints

        # (1) Inventory balance: s[i,t-1] + x[i,t] = demand[i,t] + s[i,t]
        for i in items:
            for t in range(horizon):
                if t == 0:
                    # Initial inventory is 0
                    self.model.addConstr(
                        x[i, t] == demands[i, t] + s[i, t], name=f"inv_balance_{i}_{t}"
                    )
                else:
                    self.model.addConstr(
                        s[i, t - 1] + x[i, t] == demands[i, t] + s[i, t],
                        name=f"inv_balance_{i}_{t}",
                    )

        # (2) Production requires setup: x[i,t] <= y[i,t]
        for i in items:
            for t in range(horizon):
                self.model.addConstr(x[i, t] <= y[i, t], name=f"setup_req_{i}_{t}")

        # (3) Exactly one item ready per period: sum_i y[i,t] = 1
        for t in range(horizon):
            self.model.addConstr(
                grb.quicksum(y[i, t] for i in items) == 1, name=f"one_ready_{t}"
            )

        # (4) Changeover detection: z[i,j,t] >= y[i,t-1] + y[j,t] - 1
        for i in items:
            for j in items:
                if i != j:
                    for t in range(1, horizon):
                        self.model.addConstr(
                            z[i, j, t] >= y[i, t - 1] + y[j, t] - 1,
                            name=f"changeover_{i}_{j}_{t}",
                        )

        # Valid inequalities (milp3 specific)
        if use_valid_inequalities:
            # Aggregate demand constraints
            for i in items:
                for t1 in range(horizon):
                    for t2 in range(t1, min(t1 + 10, horizon)):  # Limit window size
                        total_demand = sum(demands[i, t] for t in range(t1, t2 + 1))
                        if total_demand > 0:
                            # Production + initial stock must cover demand
                            self.model.addConstr(
                                grb.quicksum(x[i, t] for t in range(t1, t2 + 1))
                                + (s[i, t1 - 1] if t1 > 0 else 0)
                                >= total_demand,
                                name=f"demand_cover_{i}_{t1}_{t2}",
                            )

        # Objective function
        obj_terms = []

        # Inventory cost
        for i in items:
            stock_cost_per_unit = self.problem.get_inventory_cost_per_unit(
                item=i, period=0
            )
            for t in range(horizon):
                obj_terms.append(stock_cost_per_unit * s[i, t])

        # Changeover cost (no initial setup cost, matches circuit with dummy node)
        for i in items:
            for j in items:
                if i != j:
                    changeover_cost = self.problem.get_changeover_cost(i, j)
                    for t in range(1, horizon):
                        obj_terms.append(changeover_cost * z[i, j, t])

        self.model.setObjective(grb.quicksum(obj_terms), grb.GRB.MINIMIZE)

        # Store variables
        self.variables = {
            "x": x,
            "y": y,
            "s": s,
            "z": z,
        }

        logger.info(f"MILP model created: {len(items)} items, {horizon} periods")
        logger.info(
            f"Variables: {self.model.NumVars}, Constraints: {self.model.NumConstrs}"
        )

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[gurobipy.Var, float]:
        """Convert solution to Gurobi variable values for warmstart."""
        if not isinstance(solution, CapacitatedMultiItemSolution):
            raise TypeError(
                f"Expected CapacitatedMultiItemSolution, got {type(solution)}"
            )

        sol = solution
        var_values = {}

        # Get variables
        x = self.variables["x"]
        y = self.variables["y"]
        s = self.variables["s"]
        z = self.variables["z"]

        items = self.problem.items_list
        horizon = self.problem.horizon

        # Build production schedule from solution
        prod_at_time = {}  # {time: item}
        for prod in sol.productions:
            if prod.period in prod_at_time:
                logger.warning(
                    f"Multiple productions at time {prod.period}, using last one"
                )
            prod_at_time[prod.period] = prod.item

        # Determine machine ready state
        ready_item = [None] * horizon
        last_produced = items[0]  # Default: first item

        for t in range(horizon):
            if t in prod_at_time:
                ready_item[t] = prod_at_time[t]
                last_produced = prod_at_time[t]
            else:
                # Idle: keep machine ready for last produced
                ready_item[t] = last_produced

        # Set x and y variables
        for i in items:
            for t in range(horizon):
                # x[i,t]: production
                if t in prod_at_time and prod_at_time[t] == i:
                    var_values[x[i, t]] = 1.0
                else:
                    var_values[x[i, t]] = 0.0

                # y[i,t]: machine ready for item i
                if ready_item[t] == i:
                    var_values[y[i, t]] = 1.0
                else:
                    var_values[y[i, t]] = 0.0

        # Compute stock levels
        stock = {i: 0.0 for i in items}  # Current stock

        for t in range(horizon):
            # Production this period
            for i in items:
                if var_values[x[i, t]] > 0.5:
                    stock[i] += 1.0  # Binary lot sizing: produce 1 unit

            # Satisfy demands
            for i in items:
                demand = self.problem.get_demand(i, t)
                if stock[i] >= demand:
                    stock[i] -= demand
                else:
                    stock[i] = 0.0  # Shortage

            # Set stock variables
            for i in items:
                var_values[s[i, t]] = stock[i]

        # Set changeover variables z[i,j,t]
        for i in items:
            for j in items:
                if i != j:
                    for t in range(1, horizon):
                        # Changeover if machine was ready for i at t-1 and ready for j at t
                        if ready_item[t - 1] == i and ready_item[t] == j:
                            var_values[z[i, j, t]] = 1.0
                        else:
                            var_values[z[i, j, t]] = 0.0

        return var_values

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution,
        get_obj_value_for_current_solution,
    ) -> CapacitatedMultiItemSolution:
        """Extract solution from MILP model."""
        x = self.variables["x"]

        productions = []

        # Extract production schedule
        for i in self.problem.items_list:
            for t in range(self.problem.horizon):
                if get_var_value_for_current_solution(x[i, t]) > 0.5:
                    # Binary lot sizing: x[i,t]=1 means produce 1 unit
                    productions.append(ProductionDecision(item=i, period=t, quantity=1))

        sol = CapacitatedMultiItemSolution(
            problem=self.problem,
            productions=productions,
        )

        if self.problem.infos.get("known_bound", None) is not None:
            logger.info(
                f"{self.aggreg_from_sol(sol) / self.problem.infos.get('known_bound', None)} "
                f"relative perf"
            )

        return sol
