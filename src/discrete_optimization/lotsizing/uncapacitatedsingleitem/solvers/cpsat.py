#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""CP-SAT solver for uncapacitated single-item lot sizing problem."""

from __future__ import annotations

import logging
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.lotsizing.production_solution import ProductionDecision
from discrete_optimization.lotsizing.uncapacitatedsingleitem.problem import (
    UncapacitatedSingleItemLSP,
    UncapacitatedSingleItemSolution,
)

logger = logging.getLogger(__name__)


class CpSatSingleItemSolver(OrtoolsCpSatSolver):
    """CP-SAT solver for uncapacitated single-item lot sizing.

    This solver models the problem using:
    - Binary setup variables Y_t ∈ {0,1} for each period
    - Integer production variables X_t >= 0 for each period
    - Integer inventory variables I_t >= 0 for each period

    Constraints:
    - Inventory balance: I_t = I_{t-1} + X_t - d_t
    - Setup forcing: X_t > 0 implies Y_t = 1
    - Initial inventory: I_0 = 0

    Objective:
    - Minimize: sum_t (s_t * Y_t + v_t * X_t + c_t * I_t)

    The problem is uncapacitated, so there are no capacity constraints.
    """

    problem: UncapacitatedSingleItemLSP
    variables: dict

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the CP-SAT model.

        Args:
            **kwargs: Additional parameters passed to parent class
        """
        super().init_model(**kwargs)
        self._create_main_vars()
        self._set_objective()

    def _create_main_vars(self):
        """Create decision variables for the CP-SAT model."""
        self.variables = {}
        horizon = self.problem.horizon
        item = 0  # Single item

        # Compute total demand as upper bound for production
        total_demand = sum(self.problem.get_demand(item, t) for t in range(horizon))

        # Binary setup variables: Y_t ∈ {0,1}
        setup_vars = {
            t: self.cp_model.NewBoolVar(name=f"setup_t{t}") for t in range(horizon)
        }

        # Production quantity variables: X_t >= 0
        # Upper bound: at most total remaining demand
        production_vars = {
            t: self.cp_model.NewIntVar(
                lb=0,
                ub=total_demand,
                name=f"production_t{t}",
            )
            for t in range(horizon)
        }

        # Inventory variables: I_t >= 0
        # Upper bound: at most total demand (worst case: produce all at start)
        inventory_vars = {
            t: self.cp_model.NewIntVar(
                lb=0,
                ub=total_demand,
                name=f"inventory_t{t}",
            )
            for t in range(horizon)
        }

        # Constraints: Setup forcing (X_t > 0 implies Y_t = 1)
        # This is modeled as: X_t <= M * Y_t where M = total_demand
        for t in range(horizon):
            self.cp_model.Add(production_vars[t] <= total_demand * setup_vars[t])

        # Constraints: Inventory balance
        # I_t = I_{t-1} + X_t - d_t
        for t in range(horizon):
            demand = self.problem.get_demand(item, t)

            if t == 0:
                # I_0 = 0 + X_0 - d_0
                self.cp_model.Add(inventory_vars[t] == production_vars[t] - demand)
            else:
                # I_t = I_{t-1} + X_t - d_t
                self.cp_model.Add(
                    inventory_vars[t]
                    == inventory_vars[t - 1] + production_vars[t] - demand
                )

        self.variables["setup"] = setup_vars
        self.variables["production"] = production_vars
        self.variables["inventory"] = inventory_vars

    def _set_objective(self):
        """Set the objective function."""
        horizon = self.problem.horizon
        item = 0
        objectives = []

        # Setup cost: sum_t (s_t * Y_t)
        setup_cost_terms = []
        for t in range(horizon):
            cost = int(self.problem.get_setup_cost(item, t))
            if cost > 0:
                setup_cost_terms.append(cost * self.variables["setup"][t])

        if setup_cost_terms:
            setup_cost = self.cp_model.NewIntVar(
                lb=0,
                ub=sum(
                    int(self.problem.get_setup_cost(item, t)) for t in range(horizon)
                ),
                name="setup_cost",
            )
            self.cp_model.Add(setup_cost == sum(setup_cost_terms))
            objectives.append(setup_cost)
            self.variables["setup_cost"] = setup_cost

        # Production cost: sum_t (v_t * X_t)
        production_cost_terms = []
        for t in range(horizon):
            cost_per_unit = int(self.problem.get_production_cost_per_unit(item, t))
            if cost_per_unit > 0:
                production_cost_terms.append(
                    cost_per_unit * self.variables["production"][t]
                )

        if production_cost_terms:
            total_demand = sum(self.problem.get_demand(item, t) for t in range(horizon))
            max_prod_cost = (
                max(
                    int(self.problem.get_production_cost_per_unit(item, t))
                    for t in range(horizon)
                )
                * total_demand
            )

            production_cost = self.cp_model.NewIntVar(
                lb=0,
                ub=max_prod_cost,
                name="production_cost",
            )
            self.cp_model.Add(production_cost == sum(production_cost_terms))
            objectives.append(production_cost)
            self.variables["production_cost"] = production_cost

        # Inventory cost: sum_t (c_t * I_t)
        inventory_cost_terms = []
        for t in range(horizon):
            cost_per_unit = int(self.problem.get_inventory_cost_per_unit(item, t))
            if cost_per_unit > 0:
                inventory_cost_terms.append(
                    cost_per_unit * self.variables["inventory"][t]
                )

        if inventory_cost_terms:
            total_demand = sum(self.problem.get_demand(item, t) for t in range(horizon))
            max_inv_cost = (
                max(
                    int(self.problem.get_inventory_cost_per_unit(item, t))
                    for t in range(horizon)
                )
                * total_demand
                * horizon
            )  # Worst case: hold all demand for all periods

            inventory_cost = self.cp_model.NewIntVar(
                lb=0,
                ub=max_inv_cost,
                name="inventory_cost",
            )
            self.cp_model.Add(inventory_cost == sum(inventory_cost_terms))
            objectives.append(inventory_cost)
            self.variables["inventory_cost"] = inventory_cost

        # Minimize total cost
        if objectives:
            self.cp_model.Minimize(sum(objectives))

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> UncapacitatedSingleItemSolution:
        """Extract solution from CP-SAT solver.

        Args:
            cpsolvercb: CP-SAT solver callback

        Returns:
            UncapacitatedSingleItemSolution
        """
        # Log objective components
        if "setup_cost" in self.variables:
            logger.info(f"Setup cost: {cpsolvercb.Value(self.variables['setup_cost'])}")
        if "production_cost" in self.variables:
            logger.info(
                f"Production cost: {cpsolvercb.Value(self.variables['production_cost'])}"
            )
        if "inventory_cost" in self.variables:
            logger.info(
                f"Inventory cost: {cpsolvercb.Value(self.variables['inventory_cost'])}"
            )

        # Extract production decisions
        productions = []
        for t in range(self.problem.horizon):
            qty = cpsolvercb.Value(self.variables["production"][t])
            if qty > 0:
                productions.append(ProductionDecision(item=0, period=t, quantity=qty))

        # Create solution
        solution = UncapacitatedSingleItemSolution(
            problem=self.problem,
            production_periods=[p.period for p in productions],
            production_quantities=[p.quantity for p in productions],
        )

        return solution
