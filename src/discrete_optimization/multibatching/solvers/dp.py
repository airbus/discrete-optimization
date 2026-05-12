#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections import defaultdict
from typing import Any, Dict, List

import didppy as dp

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
)
from discrete_optimization.multibatching.solvers import MultibatchingSolver
from discrete_optimization.multibatching.solvers.solver_utils import (
    precompute_valid_links,
)

logger = logging.getLogger(__name__)


class DpMultibatchingSolver(DpSolver, MultibatchingSolver):
    """
    Dynamic Programming solver for multibatching problem.

    Simplified approach:
    - State: Current supply/demand at each (product, location) pair
    - Transitions: Send a batch on a transport link with specific products/quantities
    - Base case: All supply/demand balanced (net = 0)
    - Cost: Transport cost for each batch

    Note: This models each product flow separately and aggregates in solution retrieval.
    """

    problem: MultibatchingProblem
    model: dp.Model
    transitions: dict

    def init_model(self, **kwargs: Any) -> None:
        """Initialize DP model for multibatching."""
        model = dp.Model(maximize=False, float_cost=False)

        # Options
        restrict_to_shortest_paths = kwargs.get("restrict_to_shortest_paths", False)
        shortest_path_tolerance = kwargs.get("shortest_path_tolerance", 0.0)

        products = self.problem.products
        locations = self.problem.locations
        transport_links = self.problem.transport_links

        nb_products = len(products)
        nb_locations = len(locations)
        nb_links = len(transport_links)

        # Precompute valid links if using shortest path heuristic
        valid_links_per_product = None
        if restrict_to_shortest_paths:
            logger.info(
                f"Computing valid links (Shortest Path Heuristic, tol={shortest_path_tolerance})..."
            )
            valid_links_per_product = precompute_valid_links(
                self.problem, tolerance=shortest_path_tolerance
            )
        # Create object types
        product_type = model.add_object_type(number=nb_products)
        link_type = model.add_object_type(number=nb_links)
        # State: net supply at each (product, location)
        # Positive = supply (source), Negative = demand (sink)
        # Initial value = current net_supply, Goal = 0 (balanced)
        net_supply = {}
        initial_supplies = {}
        for p_idx, p in enumerate(products):
            for l_idx, loc in enumerate(locations):
                supply_val = loc.net_supply.get(p, 0)
                # target = INITIAL value (current supply/demand)
                net_supply[(p_idx, l_idx)] = model.add_int_var(target=supply_val)
                initial_supplies[(p_idx, l_idx)] = supply_val
        # Create lookup tables
        sizes_list = [int(p.size) for p in products]
        product_sizes = model.add_int_table(sizes_list)
        link_costs = model.add_int_table(
            [int(link.transport_type.cost * link.distance) for link in transport_links]
        )
        # Build product-link compatibility matrix
        product_can_use_link = {}
        for p_idx, p in enumerate(products):
            for link_idx, link in enumerate(transport_links):
                can_use = (
                    link.transport_type in p.valid_transports
                    and p.size <= link.transport_type.capacity
                )
                product_can_use_link[(p_idx, link_idx)] = can_use
        self.transitions = {}
        current_batch = [model.add_int_var(target=0) for _ in range(nb_products)]
        capacities_list = [int(tl.transport_type.capacity) for tl in transport_links]
        capacities = model.add_int_table(
            [int(tl.transport_type.capacity) for tl in transport_links]
        )
        current_size = model.add_int_var(target=0)
        current_link = model.add_element_var(object_type=link_type, target=0)
        currently_open = model.add_int_var(target=0)
        # Create compatibility table: True if product can use link, False otherwise
        compatible = model.add_bool_table(product_can_use_link, default=False)

        for link_idx, link in enumerate(transport_links):
            source_idx = self.problem.locations_to_index[link.location_l1]
            dest_idx = self.problem.locations_to_index[link.location_l2]
            capacity = int(link.transport_type.capacity)

            # Check if this link should be considered based on shortest path heuristic
            if restrict_to_shortest_paths:
                # Check if this link is valid for at least one product
                is_valid_for_any_product = False
                for p_idx, p in enumerate(products):
                    if (
                        p.id in valid_links_per_product
                        and link_idx in valid_links_per_product[p.id]
                    ):
                        is_valid_for_any_product = True
                        break
                if not is_valid_for_any_product:
                    continue  # Skip this link

            # Build precondition: at least one product with supply>0 at source AND demand<0 at dest
            # This biases opening links that move us toward feasibility
            has_useful_product = False
            for p_idx in range(nb_products):
                condition = (net_supply[(p_idx, source_idx)] > 0) & (
                    net_supply[(p_idx, dest_idx)] < 0
                )
                if p_idx == 0:
                    has_useful_product = condition
                else:
                    has_useful_product = has_useful_product | condition

            self.transitions[("open", link_idx)] = dp.Transition(
                name=f"open_{link_idx}",
                cost=dp.IntExpr.state_cost(),
                effects=[
                    (currently_open, 1),
                    (current_size, 0),
                    (current_link, link_idx),
                ],
                preconditions=[currently_open == 0, has_useful_product],
            )
            model.add_transition(self.transitions[("open", link_idx)])
        cost_tl = [
            int(
                tl.transport_type.cost * tl.distance
                + tl.transport_type.emissions * tl.distance
            )
            for tl in transport_links
        ]
        cost_table = model.add_int_table(cost_tl)
        self.transitions["close"] = dp.Transition(
            name="close",
            cost=cost_table[current_link] + dp.IntExpr.state_cost(),
            effects=[(current_size, 0), (currently_open, 0)],
            preconditions=[currently_open == 1],
        )
        for link_idx, link in enumerate(transport_links):
            source_idx = self.problem.locations_to_index[link.location_l1]
            dest_idx = self.problem.locations_to_index[link.location_l2]
            for p_idx, p in enumerate(products):
                # Use the compatible table and check capacity
                is_compatible = compatible[p_idx, link_idx] & (
                    capacities[link_idx] >= p.size
                )
                self.transitions[("send_1", p_idx, link_idx)] = dp.Transition(
                    name=f"send_1_{p_idx}_{link_idx}",
                    cost=dp.IntExpr.state_cost(),
                    effects=[
                        (current_size, current_size + sizes_list[p_idx]),
                        (
                            net_supply[(p_idx, source_idx)],
                            net_supply[(p_idx, source_idx)] - 1,
                        ),
                        (
                            net_supply[(p_idx, dest_idx)],
                            net_supply[(p_idx, dest_idx)] + 1,
                        ),
                    ],
                    preconditions=[
                        currently_open == 1,
                        is_compatible,
                        link_idx == current_link,
                        current_size + sizes_list[p_idx] <= capacities_list[link_idx],
                        net_supply[(p_idx, source_idx)] >= 1,
                    ],
                )
                self.transitions[("send_all", p_idx, link_idx)] = dp.Transition(
                    name=f"send_all_{p_idx}_{link_idx}",
                    cost=dp.IntExpr.state_cost(),
                    effects=[
                        (
                            current_size,
                            current_size
                            + sizes_list[p_idx] * net_supply[(p_idx, source_idx)],
                        ),
                        (net_supply[(p_idx, source_idx)], 0),
                        (
                            net_supply[(p_idx, dest_idx)],
                            net_supply[(p_idx, dest_idx)]
                            + net_supply[(p_idx, source_idx)],
                        ),
                    ],
                    preconditions=[
                        currently_open == 1,
                        is_compatible,
                        link_idx == current_link,
                        current_size
                        + sizes_list[p_idx] * net_supply[(p_idx, source_idx)]
                        <= capacities_list[link_idx],
                        net_supply[(p_idx, source_idx)] >= 1,
                    ],
                )
                model.add_transition(self.transitions[("send_all", p_idx, link_idx)])
                model.add_transition(self.transitions[("send_1", p_idx, link_idx)])

        # Base case: GOAL state - all supply/demand balanced (net_supply = 0)
        base_conditions = []
        for (p_idx, l_idx), initial_val in initial_supplies.items():
            if initial_val != 0:
                # This location had supply/demand, so it must be 0 at goal
                base_conditions.append(net_supply[(p_idx, l_idx)] == 0)

        if base_conditions:
            model.add_base_case(base_conditions)
        else:
            # Trivial case: no supply/demand to route
            model.add_base_case([True])

        # Store for solution retrieval
        self.net_supply = net_supply
        self.initial_supplies = initial_supplies
        self.model = model

        logger.info(
            f"DP model initialized: "
            f"{nb_products} products, {nb_locations} locations, "
            f"{nb_links} links, {len(self.transitions)} transitions"
        )

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        """
        Extract MultibatchingSolution from DP solution.

        Parses transition sequence: open_{link} -> send_{product}_{link} -> close
        and aggregates into batches. Simulates state changes to determine quantities
        for send_all transitions.
        """
        # Debug: print all transitions
        logger.info(f"Transition sequence ({len(sol.transitions)} transitions):")
        for i, t in enumerate(sol.transitions):
            logger.info(f"  {i}: {t.name}")

        # Initialize shadow state with initial supplies
        current_state = {}
        for (p_idx, l_idx), initial_val in self.initial_supplies.items():
            current_state[(p_idx, l_idx)] = initial_val

        # Track batches: list of (link_idx, {p_idx: qty})
        batches = []
        current_link = None
        current_batch = defaultdict(int)
        batch_opened = False

        for i, transition in enumerate(sol.transitions):
            name = transition.name

            if name.startswith("open_"):
                # Start new batch
                link_idx = int(name.split("_")[1])
                current_link = link_idx
                current_batch = defaultdict(int)
                batch_opened = True

            elif name.startswith("send_1_"):
                # Format: send_1_{p_idx}_{link_idx}
                parts = name.split("_")
                p_idx = int(parts[2])
                link_idx = int(parts[3])
                if batch_opened and current_link == link_idx:
                    # Get source and destination for this link
                    link = self.problem.transport_links[link_idx]
                    source_idx = self.problem.locations_to_index[link.location_l1]
                    dest_idx = self.problem.locations_to_index[link.location_l2]

                    # Update shadow state
                    current_state[(p_idx, source_idx)] -= 1
                    current_state[(p_idx, dest_idx)] += 1

                    # Track in batch
                    current_batch[p_idx] += 1

            elif name.startswith("send_all_"):
                # Format: send_all_{p_idx}_{link_idx}
                # Determine quantity from current state before applying transition
                parts = name.split("_")
                p_idx = int(parts[2])
                link_idx = int(parts[3])

                if batch_opened and current_link == link_idx:
                    # Get source and destination for this link
                    link = self.problem.transport_links[link_idx]
                    source_idx = self.problem.locations_to_index[link.location_l1]
                    dest_idx = self.problem.locations_to_index[link.location_l2]

                    # The quantity sent is all remaining supply at source
                    qty_sent = current_state.get((p_idx, source_idx), 0)

                    if qty_sent > 0:
                        # Update shadow state
                        current_state[(p_idx, source_idx)] = 0
                        current_state[(p_idx, dest_idx)] += qty_sent

                        # Track in batch
                        current_batch[p_idx] += qty_sent

            elif name == "close":
                # Finalize current batch
                if batch_opened and current_link is not None and current_batch:
                    batches.append((current_link, dict(current_batch)))
                current_link = None
                current_batch = defaultdict(int)
                batch_opened = False

        # Finalize any remaining open batch (in case sequence ends without close)
        if batch_opened and current_link is not None and current_batch:
            batches.append((current_link, dict(current_batch)))
            logger.info(f"Finalizing unclosed batch on link {current_link}")

        # Aggregate batches by link
        link_batches: Dict[int, List[Dict[int, int]]] = defaultdict(list)
        for link_idx, batch in batches:
            link_batches[link_idx].append(batch)

        # Build solution
        list_flows = []
        for link_idx, batch_list in link_batches.items():
            link = self.problem.transport_links[link_idx]

            # For each link, create one PackingTransport per batch
            for batch in batch_list:
                packing = {}
                for p_idx, qty in batch.items():
                    product = self.problem.products[p_idx]
                    if qty > 0:
                        packing[product] = qty

                if packing:
                    list_flows.append(
                        PackingTransport(
                            transport_link=link,
                            product_packing=packing,
                            nb_packing=1,  # Each batch is 1 packing
                        )
                    )

        solution = MultibatchingSolution(problem=self.problem, list_flows=list_flows)

        logger.info(
            f"Retrieved solution with {len(list_flows)} flows from {len(batches)} batches "
            f"(satisfy={self.problem.satisfy(solution)})"
        )

        return solution
