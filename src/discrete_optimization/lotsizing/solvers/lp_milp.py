#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""MILP solver for lot sizing problem.

Implementation of the milp3 model from:
Ceschia, Di Gaspero, Schaerf (2017) - "Solving discrete lot-sizing and
scheduling by simulated annealing and mixed integer programming"
"""

import logging
from typing import Any

import gurobipy

from discrete_optimization.generic_tools.do_problem import (
    Solution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
)
from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
    ProductionItem,
)

try:
    import gurobipy as grb
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True

logger = logging.getLogger(__name__)


class MilpLotSizingSolver(GurobiMilpSolver):
    """MILP solver for lot sizing using the milp3 formulation.

    This implements the strongest MILP formulation (milp3) from the paper,
    which includes valid inequalities for better performance.

    Variables:
    - x[i,t]: binary, 1 if item i is produced in period t
    - y[i,t]: binary, 1 if machine is ready for item i at start of period t
    - s[i,t]: integer, stock of item i at end of period t
    - z[i,j,t]: binary, 1 if changeover from i to j happens in period t
    """

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[gurobipy.Var, float]:
        pass

    problem: LotSizingProblem

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
        self.model = grb.Model("LotSizing_MILP3")

        # Shortcuts
        horizon = self.problem.horizon
        items = list(self.problem.items_range)
        m = len(items)

        # Get demands as dict for easier access
        demands = {}
        for item in items:
            for t in range(horizon):
                demands[item, t] = self.problem.demands[item][t]

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
        max_stock = sum(sum(self.problem.demands[i]) for i in items)
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
                if i != j:  # No self-changeover
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
            # For each interval [t1, t2] and item i, if there are demands,
            # ensure sufficient production or stock
            # This helps tighten the LP relaxation

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

        # Stocking cost
        for i in items:
            stock_cost_per_unit = self.problem.stock_cost_per_type_per_time_per_unit[i]
            for t in range(horizon):
                obj_terms.append(stock_cost_per_unit * s[i, t])

        # Setup/changeover cost
        # Only charge for transitions between actual production periods (t >= 1)
        # No cost for initial setup at period 0 (like CP-SAT scheduler with dummy node)
        for i in items:
            for j in items:
                if i != j:
                    changeover_cost = self.problem.changeover_costs[i][j]
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

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution,
        get_obj_value_for_current_solution,
    ) -> LotSizingSolution:
        """Extract solution from MILP model."""
        x = self.variables["x"]

        productions = []

        # Extract production schedule
        # In the binary MILP model, x[i,t]=1 means produce 1 unit of item i at time t
        # This includes production for stock buildup, not just immediate demand
        for i in self.problem.items_range:
            for t in range(self.problem.horizon):
                if get_var_value_for_current_solution(x[i, t]) > 0.5:
                    # In binary lot-sizing, x[i,t]=1 means produce 1 unit
                    productions.append(ProductionItem(item_type=i, quantity=1, time=t))

        # Pass deliveries=None to trigger recompute_deliveries()
        # which properly handles inventory flow and demand satisfaction
        return LotSizingSolution(
            problem=self.problem,
            productions=productions,
            deliveries=None,
        )
