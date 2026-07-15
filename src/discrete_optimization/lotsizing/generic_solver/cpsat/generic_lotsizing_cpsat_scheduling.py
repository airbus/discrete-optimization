#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Generic CP-SAT scheduling solver for lot sizing problems.

This solver uses a scheduling-based formulation where each demand is modeled
as an event (interval variable) to be scheduled. This is fundamentally different
from the quantity-based formulation in generic_lotsizing_cpsat.py.

Key features:
- Interval variables for each demand occurrence
- NoOverlap constraints for capacity
- Circuit constraints for changeover sequencing
- Natural representation of inventory cost as (deadline - production_time)
- Efficient for problems with sparse demands or unit demands

The scheduling approach is particularly effective when:
- Demands are small relative to horizon (many idle periods)
- Strong changeover costs make sequencing important
- Backlog is allowed (flexible timing)
"""

import logging
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
    Item,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.lotsizing_solver_cpsat import (
    LotSizingCpSatSolver,
)
from discrete_optimization.lotsizing.production_solution import (
    ProductionBasedSolution,
    ProductionDecision,
)

logger = logging.getLogger(__name__)


class GenericLotSizingCpsatScheduling(LotSizingCpSatSolver[Item]):
    """Generic CP-SAT scheduling solver for lot sizing.

    This solver models lot sizing as a scheduling problem where each demand
    is an event to be scheduled. This contrasts with the quantity-based
    formulation in GenericLotSizingCpsat.

    Model structure:
    - For each demand occurrence (item, period, quantity), create demand events
    - Each event has a start time variable (when to produce)
    - Interval variables ensure no conflicts
    - Circuit constraint sequences productions for changeover costs
    - Inventory cost = (deadline - start) * holding_cost

    This formulation is most efficient for:
    - Unit demands or small demands
    - Sparse demand patterns (many idle periods)
    - Strong changeover cost structure
    """

    hyperparameters = [
        CategoricalHyperparameter(
            name="unit_demand_aggregation",
            choices=[True, False],
            default=True,
        ),
    ]

    problem: GenericLotSizingProblem[Item]
    variables: dict
    demand_events: list  # List of (item, period, quantity, event_id)
    event_deadlines: dict  # event_id -> deadline period

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the CP-SAT scheduling model.

        Args:
            **kwargs: Hyperparameters including unit_demand_aggregation
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        super().init_model(**kwargs)

        self.variables = {}
        self.demand_events = []
        self.event_deadlines = {}

        # Create demand events and interval variables
        self._create_demand_events(kwargs["unit_demand_aggregation"])
        self._create_interval_variables()
        self._create_capacity_constraints()
        self._create_objective(**kwargs)

    def _create_demand_events(self, unit_demand_aggregation: bool):
        """Create demand events from problem demands.

        Each demand can create one or more events depending on aggregation:
        - If unit_demand_aggregation=True: aggregate all units into one event
        - If unit_demand_aggregation=False: create one event per unit

        Args:
            unit_demand_aggregation: Whether to aggregate demand units
        """
        event_id = 0

        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                demand = int(self.problem.get_demand(item, t))

                if demand > 0:
                    if unit_demand_aggregation:
                        # One event for entire demand
                        self.demand_events.append((item, t, demand, event_id))
                        self.event_deadlines[event_id] = t
                        event_id += 1
                    else:
                        # One event per unit of demand
                        for unit in range(demand):
                            self.demand_events.append((item, t, 1, event_id))
                            self.event_deadlines[event_id] = t
                            event_id += 1

        logger.info(
            f"Created {len(self.demand_events)} demand events for {len(self.problem.items_list)} items"
        )

    def _create_interval_variables(self):
        """Create interval variables for each demand event.

        For each event, create:
        - start: when to produce this demand
        - interval: fixed-size interval (size=1 for one period production)
        """
        starts = {}
        intervals = {}

        # Track events by item for ordering constraints
        events_by_item = {}
        for item in self.problem.items_list:
            events_by_item[item] = []

        for item, deadline, quantity, event_id in self.demand_events:
            # Bounds on start time
            # Can't produce too early (based on item's previous demands)
            # Can't produce too late (deadline, or later if backlog allowed)
            lb = 0  # TODO: could be tighter based on previous demands
            ub = (
                deadline
                if not self.problem.is_backlog_allowed()
                else self.problem.horizon - 1
            )

            # Start variable
            starts[event_id] = self.cp_model.NewIntVar(
                lb=lb, ub=ub, name=f"start_{event_id}"
            )

            # Interval variable (fixed size = 1 period)
            intervals[event_id] = self.cp_model.NewFixedSizeIntervalVar(
                start=starts[event_id], size=1, name=f"interval_{event_id}"
            )

            events_by_item[item].append(event_id)

        # Ordering constraints within same item
        # Earlier demands must be produced before or at same time as later demands
        # The capacity constraint (NoOverlap or cumulative) will handle actual conflicts
        for item in self.problem.items_list:
            events = events_by_item[item]
            for i in range(len(events) - 1):
                self.cp_model.Add(starts[events[i]] <= starts[events[i + 1]])

        self.variables["starts"] = starts
        self.variables["intervals"] = intervals
        self.variables["events_by_item"] = events_by_item

    def _create_capacity_constraints(self):
        """Create capacity constraints.

        We always create explicit capacity constraints to properly account for:
        - Production time per event
        - Setup time when an item is produced in a period
        - Total capacity limit per period

        When parallel production is NOT allowed, we also add constraints to ensure
        at most one item TYPE is produced per period.

        Note: We do NOT use NoOverlap because:
        - Multiple events of the SAME item can be in the same period
        - NoOverlap would prevent this incorrectly
        - The explicit capacity and parallel production constraints handle everything correctly
        """
        # Always create explicit capacity constraints
        self._create_explicit_capacity_constraints()

    def _create_explicit_capacity_constraints(self):
        """Create explicit capacity constraints accounting for setup times.

        For each period t, create capacity constraint:
        sum(production_time + setup_time) <= capacity

        Setup times are charged conservatively: whenever an item is produced in a period,
        we charge the setup time. This is correct but may be conservative (overestimates
        setup costs when the same item is produced consecutively).
        """
        logger.info("Creating explicit capacity constraints for setup times")

        # Track which items are produced in each period (for parallel production constraint)
        item_produced_in_period = {}  # item_produced_in_period[item][t] = BoolVar

        # Capacity constraints per period
        for t in range(self.problem.horizon):
            capacity_terms = []

            for item in self.problem.items_list:
                # Production time for events of this item scheduled at time t
                events = self.variables["events_by_item"][item]

                # Collect all events of this item at time t
                events_at_t_for_item = []

                for event_id in events:
                    # Get event data
                    event_data = next(
                        (it, dl, q, eid)
                        for it, dl, q, eid in self.demand_events
                        if eid == event_id
                    )
                    item_ev, deadline, quantity, _ = event_data

                    # Binary: is this event scheduled at time t?
                    is_at_t = self.cp_model.NewBoolVar(name=f"event_{event_id}_at_{t}")
                    self.cp_model.Add(
                        self.variables["starts"][event_id] == t
                    ).OnlyEnforceIf(is_at_t)
                    self.cp_model.Add(
                        self.variables["starts"][event_id] != t
                    ).OnlyEnforceIf(is_at_t.Not())

                    events_at_t_for_item.append(is_at_t)

                    # Production time contribution
                    prod_time_per_unit = int(
                        self.problem.get_production_time_per_unit(item, t)
                    )
                    if prod_time_per_unit > 0:
                        capacity_terms.append(quantity * prod_time_per_unit * is_at_t)

                # Setup time: charged if ANY event of this item is at time t
                if events_at_t_for_item:
                    # item_at_t = 1 if any event of item is at time t
                    item_at_t = self.cp_model.NewBoolVar(name=f"item_{item}_at_{t}")
                    self.cp_model.AddMaxEquality(item_at_t, events_at_t_for_item)

                    # Track for parallel production constraint
                    if item not in item_produced_in_period:
                        item_produced_in_period[item] = {}
                    item_produced_in_period[item][t] = item_at_t

                    # Add setup time if item produced in period t
                    setup_time = int(self.problem.get_setup_time(item, t))
                    if setup_time > 0:
                        capacity_terms.append(setup_time * item_at_t)

            # Add capacity constraint for period t
            if capacity_terms:
                available = int(self.problem.get_available_production_time(t))
                self.cp_model.Add(sum(capacity_terms) <= available)
                logger.debug(
                    f"Period {t}: {len(capacity_terms)} capacity terms, limit={available}"
                )

        logger.info("Explicit capacity constraints created")

        # Parallel production constraint: at most one item TYPE per period
        if not self.problem.allows_parallel_production():
            for t in range(self.problem.horizon):
                # At most one item can be produced in period t
                items_at_t = []
                for item in self.problem.items_list:
                    if (
                        item in item_produced_in_period
                        and t in item_produced_in_period[item]
                    ):
                        items_at_t.append(item_produced_in_period[item][t])

                if len(items_at_t) > 1:
                    # At most 1 item type produced
                    self.cp_model.Add(sum(items_at_t) <= 1)

            logger.info(
                "Parallel production constraints added (at most one item type per period)"
            )

    def _create_changeover_model(self):
        """Create changeover cost model using circuit constraint.

        Models the production sequence as a Hamiltonian path through events:
        - Dummy start node connects to first event
        - Each event connects to next event
        - Last event connects to dummy end node
        - Arc costs represent changeover costs between items

        Returns:
            Linear expression for total changeover cost
        """
        if self.problem.allows_parallel_production():
            # Circuit doesn't make sense with parallel production
            return 0

        # Build nodes: dummy + all events
        nodes = [("dummy", -1)]  # Dummy node
        event_to_node = {}

        for idx, (item, deadline, quantity, event_id) in enumerate(self.demand_events):
            node_id = idx + 1  # +1 because dummy is 0
            nodes.append((item, event_id))
            event_to_node[event_id] = node_id

        n_nodes = len(nodes)

        # Create arc variables for circuit
        arcs = []
        arc_vars = {}

        # Maximum gap between consecutive productions
        total_demands = sum(
            int(self.problem.get_total_demand(item)) for item in self.problem.items_list
        )
        max_gap = max(1, self.problem.horizon - total_demands + 1)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue

                # Create arc variable
                arc_var = self.cp_model.NewBoolVar(name=f"arc_{i}_{j}")
                arcs.append((i, j, arc_var))
                arc_vars[(i, j)] = arc_var

                # Ordering constraints
                if i > 0 and j > 0:  # Not dummy nodes
                    item_i, event_i = nodes[i]
                    item_j, event_j = nodes[j]

                    # If arc is used, enforce start_j >= start_i
                    # Capacity constraints ensure no actual conflicts
                    self.cp_model.Add(
                        self.variables["starts"][event_j]
                        >= self.variables["starts"][event_i]
                    ).OnlyEnforceIf(arc_var)

                    # Limit gap
                    self.cp_model.Add(
                        self.variables["starts"][event_j]
                        <= self.variables["starts"][event_i] + max_gap
                    ).OnlyEnforceIf(arc_var)

                    # Same item ordering: enforce sequential demand order
                    if item_i == item_j:
                        # Check if these are sequential demands for the same item
                        events_i = [
                            ev_id
                            for it, dl, q, ev_id in self.demand_events
                            if it == item_i
                        ]
                        try:
                            idx_i = events_i.index(event_i)
                            idx_j = events_i.index(event_j)
                            if idx_j != idx_i + 1:
                                # Not sequential -> disallow arc
                                self.cp_model.Add(arc_var == 0)
                        except ValueError:
                            pass

        # Add circuit constraint
        self.cp_model.AddCircuit(arcs)

        # Changeover cost: sum over arcs between real productions
        changeover_cost_terms = []
        for (i, j), arc_var in arc_vars.items():
            if i > 0 and j > 0:  # Skip arcs involving dummy
                item_i, _ = nodes[i]
                item_j, _ = nodes[j]
                cost = int(self.problem.get_changeover_cost(item_i, item_j))
                if cost > 0:
                    changeover_cost_terms.append(cost * arc_var)

        self.variables["circuit_arcs"] = arc_vars
        self.variables["nodes"] = nodes

        if changeover_cost_terms:
            return sum(changeover_cost_terms)
        else:
            return 0

    def _create_objective(self, **kwargs: Any):
        """Create objective function.

        Objective components:
        1. Inventory cost: sum over events of (deadline - start) * holding_cost
        2. Backlog cost: sum over events of max(0, start - deadline) * backlog_cost
        3. Changeover cost: from circuit constraint
        4. Setup cost: not naturally modeled in scheduling (use penalty if needed)
        """
        objective_terms = []

        # Inventory holding cost
        # For each event, holding time = max(0, deadline - production_time)
        # Only charge inventory if we produce BEFORE the deadline
        inventory_cost_terms = []
        inventory_time_vars = {}

        for item, deadline, quantity, event_id in self.demand_events:
            holding_cost = int(self.problem.get_inventory_cost_per_unit(item, deadline))
            if holding_cost > 0:
                # Inventory time = max(0, deadline - start)
                inv_time = self.cp_model.NewIntVar(
                    lb=0,
                    ub=deadline,
                    name=f"inv_time_{event_id}",
                )
                inventory_time_vars[event_id] = inv_time

                # inv_time = max(0, deadline - start)
                self.cp_model.AddMaxEquality(
                    inv_time,
                    [deadline - self.variables["starts"][event_id], 0],
                )

                inventory_cost_terms.append(quantity * holding_cost * inv_time)

        if inventory_cost_terms:
            inventory_cost_expr = sum(inventory_cost_terms)
            objective_terms.append(inventory_cost_expr)
            self.variables["inventory_cost_expr"] = inventory_cost_expr
            self.variables["inventory_time_vars"] = inventory_time_vars

        # Backlog cost (if allowed)
        if self.problem.is_backlog_allowed():
            backlog_cost_terms = []
            delay_vars = {}

            for item, deadline, quantity, event_id in self.demand_events:
                backlog_cost = int(
                    self.problem.get_backlog_cost_per_unit(item, deadline)
                )
                if backlog_cost > 0:
                    # Delay = max(0, start - deadline)
                    delay_var = self.cp_model.NewIntVar(
                        lb=0,
                        ub=self.problem.horizon - deadline,
                        name=f"delay_{event_id}",
                    )
                    delay_vars[event_id] = delay_var

                    # delay = max(0, start - deadline)
                    self.cp_model.AddMaxEquality(
                        delay_var,
                        [self.variables["starts"][event_id] - deadline, 0],
                    )

                    backlog_cost_terms.append(quantity * backlog_cost * delay_var)

            if backlog_cost_terms:
                backlog_cost_expr = sum(backlog_cost_terms)
                objective_terms.append(backlog_cost_expr)
                self.variables["backlog_cost_expr"] = backlog_cost_expr
                self.variables["delay_vars"] = delay_vars

        # Changeover cost
        if not self.problem.allows_parallel_production():
            changeover_cost_expr = self._create_changeover_model()
            # Check if we got a non-zero expression (list of terms) or just 0
            if isinstance(changeover_cost_expr, int) and changeover_cost_expr == 0:
                # No changeover costs
                pass
            else:
                objective_terms.append(changeover_cost_expr)
                self.variables["changeover_cost_expr"] = changeover_cost_expr

        # Setup cost - not naturally modeled in scheduling
        # Could add as penalty for number of distinct production periods
        # For now, skip (typically zero in test problems or included in changeover)

        # Minimize total cost
        if objective_terms:
            self.cp_model.Minimize(sum(objective_terms))
            logger.info(f"Objective created with {len(objective_terms)} components")

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> ProductionBasedSolution:
        """Extract solution from CP-SAT solver.

        Args:
            cpsolvercb: CP-SAT solution callback

        Returns:
            ProductionBasedSolution with productions and deliveries
        """
        # Extract production times for each event
        production_times = {}
        for event_id, start_var in self.variables["starts"].items():
            production_times[event_id] = cpsolvercb.Value(start_var)

        # Group by (item, period) to aggregate production quantities
        prod_by_item_period = {}
        for item, deadline, quantity, event_id in self.demand_events:
            prod_time = production_times[event_id]
            key = (item, prod_time)
            if key not in prod_by_item_period:
                prod_by_item_period[key] = 0
            prod_by_item_period[key] += quantity

        # Create production decisions
        productions = []
        for (item, period), quantity in prod_by_item_period.items():
            productions.append(
                ProductionDecision(item=item, period=period, quantity=quantity)
            )

        # Let ProductionBasedSolution compute deliveries automatically
        # It will satisfy demands from production + inventory in the correct order
        solution = ProductionBasedSolution(
            problem=self.problem, productions=productions
        )

        # Log objective components
        self._log_objective_comparison(cpsolvercb, solution)

        return solution

    def _log_objective_comparison(
        self, cpsolvercb: CpSolverSolutionCallback, solution: ProductionBasedSolution
    ):
        """Log comparison between CP-SAT objective and actual evaluation.

        Args:
            cpsolvercb: CP-SAT solution callback
            solution: Extracted solution
        """
        # CP-SAT objective components
        cpsat_inventory = (
            cpsolvercb.Value(self.variables["inventory_cost_expr"])
            if "inventory_cost_expr" in self.variables
            else 0
        )
        cpsat_backlog = (
            cpsolvercb.Value(self.variables["backlog_cost_expr"])
            if "backlog_cost_expr" in self.variables
            else 0
        )
        cpsat_changeover = (
            cpsolvercb.Value(self.variables["changeover_cost_expr"])
            if "changeover_cost_expr" in self.variables
            else 0
        )
        cpsat_total = cpsat_inventory + cpsat_backlog + cpsat_changeover

        # Actual evaluation
        eval_dict = self.problem.evaluate(solution)
        actual_total = sum(eval_dict.values())

        logger.info("=" * 60)
        logger.info("CP-SAT Objective (scheduling formulation):")
        logger.info(f"  Inventory cost:  {cpsat_inventory:10.1f}")
        logger.info(f"  Backlog cost:    {cpsat_backlog:10.1f}")
        logger.info(f"  Changeover cost: {cpsat_changeover:10.1f}")
        logger.info(f"  Total:           {cpsat_total:10.1f}")
        logger.info("")
        logger.info("Actual Evaluation:")
        logger.info(f"  Inventory cost:  {eval_dict['inventory_cost']:10.1f}")
        logger.info(f"  Backlog cost:    {eval_dict['backlog_cost']:10.1f}")
        logger.info(f"  Changeover cost: {eval_dict['changeover_cost']:10.1f}")
        logger.info(f"  Total:           {actual_total:10.1f}")
        logger.info("=" * 60)

        # Check for discrepancies
        discrepancy = actual_total - cpsat_total
        if abs(discrepancy) > 0.1:
            logger.warning(f"DISCREPANCY: {discrepancy:+.1f} (actual - cpsat)")

    def get_production_quantity_var(self, item: Item, period: int) -> Any:
        """Not applicable for scheduling formulation."""
        raise NotImplementedError(
            "Scheduling formulation does not use quantity variables per period"
        )

    def get_production_binary_var(self, item: Item, period: int) -> Any:
        """Not applicable for scheduling formulation."""
        raise NotImplementedError(
            "Scheduling formulation does not use binary variables per period"
        )

    def get_inventory_var(self, item: Item, period: int) -> Any:
        """Not applicable for scheduling formulation."""
        raise NotImplementedError(
            "Scheduling formulation does not have explicit inventory variables"
        )

    def get_backlog_var(self, item: Item, period: int) -> Any:
        """Not applicable for scheduling formulation."""
        raise NotImplementedError(
            "Scheduling formulation uses event-based delay variables"
        )

    def get_delivery_var(self, item: Item, period: int) -> Any:
        """Not applicable for scheduling formulation."""
        raise NotImplementedError("Scheduling formulation uses event-based deliveries")
