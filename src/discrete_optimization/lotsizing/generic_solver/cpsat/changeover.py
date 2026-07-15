#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum

from discrete_optimization.lotsizing.generic_solver.cpsat.lotsizing_solver_cpsat import (
    Item,
    LotSizingCpSatSolver,
)


class ChangeoverModel(Enum):
    """Modeling approach for changeover costs in CP-SAT solver."""

    STATE_BASED = "state_based"
    SHORTEST_PATH_BASED = "shortest_path_based"


class ChangeOverConstraintCpsat(LotSizingCpSatSolver[Item]):
    changeover_vars: dict

    def create_changeover_constraint_and_cost(self, modeling: ChangeoverModel):
        self.changeover_vars = {}
        if modeling == ChangeoverModel.STATE_BASED:
            self._create_changeover_vars_state_based()
        if modeling == ChangeoverModel.SHORTEST_PATH_BASED:
            self._create_changeover_vars_shortest_path()
        return self.changeover_vars["changeover_cost"]

    def _create_changeover_vars_state_based(self):
        """Create changeover variables using state-based model (element constraints).

        Most efficient: O(horizon) variables.
        """
        produce = {
            item: [
                self.get_production_binary_var(item=item, period=t)
                for t in range(self.problem.horizon)
            ]
            for item in self.problem.items_list
        }
        horizon = self.problem.horizon
        # Which item was produced in each period (nb_items = idle)
        item_produced = {
            t: self.cp_model.NewIntVar(
                lb=0, ub=self.problem.nb_items, name=f"item_at_{t}"
            )
            for t in range(horizon)
        }
        self.changeover_vars["item_produced"] = item_produced
        # Boolean: is there any production at time t?
        has_production = {
            t: self.cp_model.NewBoolVar(name=f"has_prod_t{t}") for t in range(horizon)
        }
        self.changeover_vars["has_production"] = has_production
        for t in range(horizon):
            # has_production[t] = OR(produce[i,t] for all i)
            self.cp_model.AddMaxEquality(
                has_production[t],
                [produce[(item, t)] for item in self.problem.items_list],
            )

        # Link item_produced to bool_productions
        for t in range(horizon):
            if t == 0:
                self.cp_model.Add(
                    item_produced[t] == self.problem.nb_items
                ).OnlyEnforceIf(has_production[t].Not())
            # If item i is produced at t, then item_produced[t] = i
            for item in self.problem.items_list:
                self.cp_model.Add(item_produced[t] == item).OnlyEnforceIf(
                    produce[(item, t)]
                )
            if t >= 1:
                self.cp_model.add(
                    item_produced[t] == item_produced[t - 1]
                ).only_enforce_if(has_production[t].Not())

        # Changeover cost variables
        changeover_costs_vars = []
        for t in range(1, horizon):
            # Cost from item at t-1 to item at t
            cost_var = self.cp_model.NewIntVar(
                lb=0,
                ub=int(self.problem.get_max_changeover_cost()),
                name=f"changeover_cost_{t}",
            )

            # Use AddElement to lookup changeover cost
            # cost = changeover_costs[item_produced[t-1]][item_produced[t]]
            # Flat costs matrix: (nb_items+1) x (nb_items+1) to include idle state
            flat_costs = []
            for i in range(self.problem.nb_items + 1):  # Include idle (nb_items)
                for j in range(self.problem.nb_items + 1):  # Include idle
                    if i == self.problem.nb_items or j == self.problem.nb_items:
                        # Changeover from/to idle has zero cost
                        flat_costs.append(0)
                    else:
                        # Regular item-to-item changeover
                        flat_costs.append(int(self.problem.get_changeover_cost(i, j)))

            # Index = item_produced[t-1] * (nb_items+1) + item_produced[t]
            index = self.cp_model.NewIntVar(
                lb=0,
                ub=(self.problem.nb_items + 1) ** 2 - 1,
                name=f"changeover_index_{t}",
            )
            self.cp_model.Add(
                index
                == item_produced[t - 1] * (self.problem.nb_items + 1) + item_produced[t]
            )

            self.cp_model.AddElement(index, flat_costs, cost_var)
            changeover_costs_vars.append(cost_var)

        # Total changeover cost
        total_changeover_cost = self.cp_model.NewIntVar(
            lb=0,
            ub=int(self.problem.get_max_changeover_cost()) * horizon,
            name="total_changeover_cost",
        )
        if changeover_costs_vars:
            self.cp_model.Add(total_changeover_cost == sum(changeover_costs_vars))
        else:
            self.cp_model.Add(total_changeover_cost == 0)
        self.changeover_vars["changeover_cost"] = sum(changeover_costs_vars)

    def _create_changeover_vars_shortest_path(self):
        """Create changeover variables using shortest path model.

        Model production sequence as shortest path through production events.
        Similar to TSP formulation. O(n_productions^2) variables.
        """
        produce = {
            item: [
                self.get_production_binary_var(item=item, period=t)
                for t in range(self.problem.horizon)
            ]
            for item in self.problem.items_list
        }
        horizon = self.problem.horizon

        # Create dummy start and end nodes
        # Transition variables: from (item, t) or "start" to (item', t') or "end"
        lookahead = min(
            10,
            horizon
            - sum(self.problem.get_total_demand(i) for i in self.problem.items_list)
            + 1,
        )
        lookahead = max(1, lookahead)
        transitions = {}
        # Transitions from start to first productions
        nodes = [("dummy", -1)] + [
            (item, t)
            for item in self.problem.items_list
            for t in range(self.problem.horizon)
        ]
        source = ("dummy", -1)
        target = ("dummy", -1)
        for item in self.problem.items_list:
            for t in range(min(lookahead + 1, horizon)):
                transitions[(source, (item, t))] = self.cp_model.NewBoolVar(
                    name=f"trans_start_to_{item}_{t}"
                )
        # Transitions between productions
        for item0 in self.problem.items_list:
            for t in range(horizon):
                for item1 in self.problem.items_list:
                    for tprime in range(t + 1, min(t + lookahead + 1, horizon)):
                        transitions[((item0, t), (item1, tprime))] = (
                            self.cp_model.NewBoolVar(
                                name=f"trans_{item0}_{t}_to_{item1}_{tprime}"
                            )
                        )
        # Transitions to end
        for item in self.problem.items_list:
            for t in range(max(0, horizon - lookahead - 1), horizon):
                transitions[((item, t), target)] = self.cp_model.NewBoolVar(
                    name=f"trans_{item}_{t}_to_end"
                )
        # Flow conservation constraints
        # Start: exactly one outgoing
        outgoing_start = [v for k, v in transitions.items() if k[0] == source]
        self.cp_model.Add(sum(outgoing_start) == 1)
        # End: exactly one incoming
        incoming_end = [v for k, v in transitions.items() if k[-1] == target]
        self.cp_model.Add(sum(incoming_end) == 1)
        self.changeover_vars["transitions"] = transitions
        # Production nodes: incoming = outgoing = bool_produce
        for item in self.problem.items_list:
            for t in range(horizon):
                # Incoming transitions
                incoming = [v for k, v in transitions.items() if k[1] == (item, t)]
                # Outgoing transitions
                outgoing = [v for k, v in transitions.items() if k[0] == (item, t)]
                if incoming:
                    self.cp_model.Add(sum(incoming) == produce[item][t])
                if outgoing:
                    self.cp_model.Add(sum(outgoing) == produce[item][t])

        # Changeover cost
        changeover_cost_terms = []
        for key, var in transitions.items():
            if key[0] == source or key[-1] == target:
                continue  # No cost for start/end
            key0, key1 = key
            item0, t0 = key0
            item1, t1 = key1
            cost = int(self.problem.get_changeover_cost(item0, item1))
            if cost > 0:
                changeover_cost_terms.append(cost * var)

        changeover_cost_var = self.cp_model.NewIntVar(
            lb=0,
            ub=sum(
                int(max(self.problem.get_changeover_array()[i]))
                for i in range(self.problem.nb_items)
            )
            * horizon,
            name="total_changeover_cost",
        )
        if changeover_cost_terms:
            self.cp_model.Add(changeover_cost_var == sum(changeover_cost_terms))
        else:
            self.cp_model.Add(changeover_cost_var == 0)

        self.changeover_vars["changeover_cost"] = changeover_cost_var
