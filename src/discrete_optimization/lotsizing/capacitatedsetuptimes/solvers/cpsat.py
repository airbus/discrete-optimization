#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""CP-SAT solver for capacitated lot sizing with setup times."""

import logging
from typing import Any

from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.lotsizing import ProductionDecision
from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesLSP,
    CapacitatedSetupTimesSolution,
)

logger = logging.getLogger(__name__)


class CpSatSetupTimesSolver(OrtoolsCpSatSolver):
    """CP-SAT solver for capacitated lot sizing with setup times.

    Key difference from basic CLSP: capacity constraint includes setup times
    Σ_i (p_it * X_it + τ_it * Y_it) ≤ h_t
    """

    problem: CapacitatedSetupTimesLSP
    variables: dict

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the CP-SAT model."""
        super().init_model(**kwargs)
        self._create_main_vars()
        self._set_objective()

    def _create_main_vars(self):
        """Create main decision variables and constraints."""
        self.variables = {}
        total_demands_per_item = {
            item: self.problem.get_total_demand(item)
            for item in self.problem.items_list
        }
        horizon = self.problem.horizon

        # Boolean variables: is item produced at time t?
        bool_produce = {
            (item, t): self.cp_model.NewBoolVar(name=f"prod_{item}_{t}")
            for item in self.problem.items_list
            for t in range(horizon)
        }
        # Quantity produced variables
        # Upper bound is min(capacity, total_demand)
        # Note: We don't subtract setup_time here - the capacity constraint handles it
        quantity_produce = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=min(
                    int(
                        self.problem.get_available_production_time(t)
                        / self.problem.get_production_time_per_unit(item, t)
                    ),
                    total_demands_per_item[item],
                ),
                name=f"qty_{item}_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Link quantity to boolean
        for item, t in quantity_produce:
            self.cp_model.Add(quantity_produce[(item, t)] >= 1).OnlyEnforceIf(
                bool_produce[(item, t)]
            )
            self.cp_model.Add(quantity_produce[(item, t)] == 0).OnlyEnforceIf(
                bool_produce[(item, t)].Not()
            )

        # Stock variables
        stocks = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"stock_{item}_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Delay/backlog variables
        delays = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=0
                if not self.problem.is_backlog_allowed()
                else total_demands_per_item[item],
                name=f"delay_{item}_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Stock balance constraints
        # Cumulative production - cumulative demand = stock - backlog
        for item in self.problem.items_list:
            for t in range(horizon):
                # Cumulative sums
                cumulative_production = sum(
                    quantity_produce[(item, tt)] for tt in range(t + 1)
                )
                cumulative_demand = sum(
                    self.problem.get_demand(item, tt) for tt in range(t + 1)
                )

                # cumulative_production - cumulative_demand = stock - delay
                self.cp_model.Add(
                    cumulative_production - cumulative_demand
                    == stocks[(item, t)] - delays[(item, t)]
                )

        # Capacity constraint with setup times: Σ_i (p_it * X_it + τ_it * Y_it) ≤ h_t
        for t in range(horizon):
            capacity_terms = []
            for item in self.problem.items_list:
                # Production time
                capacity_terms.append(
                    quantity_produce[(item, t)]
                    * self.problem.get_production_time_per_unit(item, t)
                )
                # Setup time (only when production occurs)
                setup_time = int(self.problem.get_setup_time(item, t))
                if setup_time > 0:
                    capacity_terms.append(bool_produce[(item, t)] * setup_time)
            self.cp_model.Add(
                sum(capacity_terms)
                <= int(self.problem.get_available_production_time(t))
            )

        # Store variables
        self.variables["bool_produce"] = bool_produce
        self.variables["quantity_produce"] = quantity_produce
        self.variables["stocks"] = stocks
        self.variables["delays"] = delays

    def _set_objective(self):
        """Set the objective function."""
        horizon = self.problem.horizon
        bool_produce = self.variables["bool_produce"]
        quantity_produce = self.variables["quantity_produce"]
        stocks = self.variables["stocks"]
        delays = self.variables["delays"]

        # Inventory costs
        inventory_costs = []
        for item in self.problem.items_list:
            for t in range(horizon):
                cost = int(self.problem.get_inventory_cost_per_unit(item, t))
                inventory_costs.append(cost * stocks[(item, t)])

        # Backlog costs
        backlog_costs = []
        for item in self.problem.items_list:
            for t in range(horizon):
                cost = int(self.problem.get_backlog_cost_per_unit(item, t))
                backlog_costs.append(cost * delays[(item, t)])

        # Total objective
        self.cp_model.Minimize(sum(inventory_costs) + sum(backlog_costs))

    def retrieve_solution(self, cpsolvercb=None) -> CapacitatedSetupTimesSolution:
        """Extract solution from CP-SAT solver."""
        quantity_produce = self.variables["quantity_produce"]

        # Extract productions
        productions = []
        for (item, t), var in quantity_produce.items():
            qty = cpsolvercb.Value(var)
            if qty > 0:
                productions.append(
                    ProductionDecision(item=item, period=t, quantity=qty)
                )

        return CapacitatedSetupTimesSolution(
            problem=self.problem, productions=productions
        )
