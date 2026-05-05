#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Optional

import networkx as nx

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransportType:
    id: int = field(hash=True, repr=False)
    cost: int = field(hash=False)
    speed: int = field(hash=False)
    emissions: int = field(hash=False)
    capacity: int = field(hash=False, repr=True)
    name: str = field(hash=False, repr=True, default=None)


@dataclass
class Product:
    id: int = field(repr=False)
    name: str = field(repr=True, hash=False, default="")
    size: int = field(repr=True, default=0)
    value: int = field(repr=False, default=0)
    valid_transports: frozenset[TransportType] = field(
        default_factory=set, hash=False, repr=False
    )

    def __hash__(self):
        return hash(self.id)


@dataclass(frozen=True)
class Location:
    id: Hashable = field(repr=False, hash=True)
    name: str = field(repr=True, hash=False, default="")
    net_supply: dict[Product, int] = field(
        repr=False, hash=False, default_factory=lambda: {}
    )


@dataclass(frozen=True)
class TransportLink:
    """
    Represent an edge (location1, location2, transport_type)
    in the logistic graph
    """

    id: str
    location_l1: Location
    location_l2: Location
    distance: int
    transport_type: TransportType
    max_trips: Optional[int] = field(repr=True, hash=False, default=None)


@dataclass
class PackingTransport:
    """
    Represent a packing going through a transport link,
    containing a set of product.
    nb_packing is optional, it can represent a frequency of the same packing going through the edge.
    """

    transport_link: TransportLink
    product_packing: dict[Product, int]
    nb_packing: int


class MultibatchingSolution(Solution):
    def __init__(
        self, problem: "MultibatchingProblem", list_flows: list[PackingTransport]
    ):
        super().__init__(problem)
        self.list_flows = list_flows

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem

    def copy(self) -> "Solution":
        return MultibatchingSolution(
            problem=self.problem, list_flows=list(self.list_flows)
        )


class MultibatchingProblem(Problem):
    def __init__(
        self,
        transport_types: list[TransportType],
        products: list[Product],
        locations: list[Location],
        transport_links: list[TransportLink],
    ):
        self.transport_types = transport_types
        self.products = products
        self.locations = locations
        self.transport_links = transport_links
        self.nb_products = len(self.products)
        self.nb_locations = len(self.locations)
        self.nb_transport_types = len(self.transport_types)
        self.nb_transport_links = len(self.transport_links)
        self.product_to_index = {self.products[i]: i for i in range(self.nb_products)}
        self.product_dict = {p.id: p for p in self.products}
        self.transport_types_to_index = {
            self.transport_types[i]: i for i in range(self.nb_transport_types)
        }
        self.locations_to_index = {
            self.locations[i]: i for i in range(self.nb_locations)
        }
        self.transport_links_to_index = {
            self.transport_links[i]: i for i in range(self.nb_transport_links)
        }

        self.transport_type_id_to_transport_type = {
            tt.id: tt for tt in self.transport_types
        }
        self.location_id_to_location = {loc.id: loc for loc in self.locations}

        self.loc_and_transport_type_to_transport_link = {
            (tl.location_l1.id, tl.location_l2.id, tl.transport_type.id): tl
            for tl in self.transport_links
        }

        self.transport_links_id_to_object = {
            transport_link.id: transport_link for transport_link in transport_links
        }
        self.locations_pos_demand = {}
        self.locations_neg_demand = {}
        for p in self.products:
            self.locations_pos_demand[p] = []
            self.locations_neg_demand[p] = []
            for l in self.locations:
                if p in l.net_supply.keys():
                    if l.net_supply[p] < 0:
                        self.locations_pos_demand[p].append(l)
                    else:
                        self.locations_neg_demand[p].append(l)

    def precompute_valid_links(self, tolerance=0.1):
        """
        Robust Heuristic: Identifies which transport links are relevant for each product.

        It builds a specific graph G_p for each product P, containing only
        the TransportLinks compatible with P (i.e. link.transport_type in p.valid_transports).

        A link (u, v) is relevant for product P if there exists ANY pair of
        (Source s, Sink d) for P such that:
           dist_p(s, u) + link_len(u, v) + dist_p(v, d) <= (1 + tolerance) * dist_p(s, d)
        """

        valid_links_per_product = defaultdict(set)

        for p in self.products:
            # 1. Identify Producers and Consumers for this specific product
            sources = [l.id for l in self.locations if l.net_supply.get(p, 0) > 1e-6]
            sinks = [l.id for l in self.locations if l.net_supply.get(p, 0) < -1e-6]

            if not sources or not sinks:
                continue

            # 2. Build the Product-Specific Graph G_p
            # Only include links where the transport type is allowed for this product
            G_p = nx.DiGraph()

            # We also map (u, v) -> list of (link_index, weight)
            # because there might be multiple transport modes between u and v (e.g. Truck vs Train)
            # and we need to check them individually.
            edges_to_indices = defaultdict(list)

            for i, tl in enumerate(self.transport_links):
                # KEY CHECK: Is this transport type valid for this product?
                # Also check capacity constraints if relevant (size <= capacity)
                if (
                    tl.transport_type in p.valid_transports
                    and p.size <= tl.transport_type.capacity
                ):
                    u, v = tl.location_l1.id, tl.location_l2.id
                    w = 1  # tl.distance
                    # w = tl.distance
                    # For NetworkX, we usually want the *best* weight between u and v
                    # if we just want shortest path distance.
                    # But we must store all parallel edges to evaluate them later.
                    edges_to_indices[(u, v)].append((i, w))

                    # Update graph with the shortest edge between u and v found so far
                    if G_p.has_edge(u, v):
                        if w < G_p[u][v]["weight"]:
                            G_p[u][v]["weight"] = w
                    else:
                        G_p.add_edge(u, v, weight=w)

            # 3. Compute All-Pairs Shortest Paths on G_p
            try:
                # This gives the shortest path distance matrix for valid moves only
                path_lengths = dict(
                    nx.all_pairs_dijkstra_path_length(G_p, weight="weight")
                )
            except nx.NetworkXNoPath:
                continue  # Disconnected graph for this product

            # 4. Filter Links
            # We iterate over all physically existing links again, but only check valid ones
            for (u, v), link_list in edges_to_indices.items():
                # Pre-check: Can any source reach u?
                valid_sources = [
                    s for s in sources if s in path_lengths and u in path_lengths[s]
                ]
                if not valid_sources:
                    continue

                # Pre-check: Can v reach any sink?
                if v not in path_lengths:
                    continue
                valid_sinks = [d for d in sinks if d in path_lengths[v]]
                if not valid_sinks:
                    continue

                # Check each specific transport mode between u and v
                for index_tl, w in link_list:
                    is_useful = False

                    for s in valid_sources:
                        dist_s_u = path_lengths[s][u]

                        for d in valid_sinks:
                            if d not in path_lengths[s]:
                                continue  # Should not happen if graph is connected, but safety check

                            dist_s_d_opt = path_lengths[s][d]
                            dist_v_d = path_lengths[v][d]

                            # Check Path Quality
                            path_via_link = dist_s_u + w + dist_v_d

                            # Relaxed triangle inequality
                            if path_via_link <= dist_s_d_opt * (1.0 + tolerance) + 1e-5:
                                is_useful = True
                                break

                        if is_useful:
                            break

                    if is_useful:
                        valid_links_per_product[p.id].add(index_tl)
        return valid_links_per_product

    def evaluate(self, variable: "MultibatchingSolution") -> dict[str, float]:
        transport_cost = sum(
            [
                flow.transport_link.transport_type.cost
                * flow.transport_link.distance
                * flow.nb_packing
                for flow in variable.list_flows
            ]
        )
        emission_co2 = sum(
            [
                flow.transport_link.transport_type.emissions
                * flow.transport_link.distance
                * flow.nb_packing
                for flow in variable.list_flows
            ]
        )
        return {"transport": transport_cost, "emission": emission_co2}

    def get_max_nb_trips(self, solution: MultibatchingSolution) -> int:
        """Look for the most number of individual trip on a given solution"""
        nb_trips_per_link = defaultdict(lambda: 0)
        for flow in solution.list_flows:
            nb_trips_per_link[flow.transport_link.id] += flow.nb_packing
        return max(nb_trips_per_link.values())

    def check_feasibility_per_product(self, verbose=False) -> bool:
        """
        TODO : no need of max_flow in the current code (we put a big capacity)
        Checks if a feasible flow exists for each product independently.
        For each product, this method builds a flow network and solves a
        maximum flow problem to determine if the total supply can meet the
        total demand through the available transport links.
        Args:
            verbose (bool): If True, prints messages for each product's
                            feasibility check.
        Returns:
            bool: True if all products have a feasible flow, False otherwise.
        """
        for product in self.products:
            if product.size <= 0:
                continue  # or handle as an error

            # 1. Build the flow network for the current product
            G = nx.DiGraph()
            source = "source"
            sink = "sink"
            G.add_nodes_from([source, sink])

            total_supply = 0
            total_demand = 0

            # 2. Add nodes for each location and connect source/sink
            for loc in self.locations:
                G.add_node(loc.id)
                supply = loc.net_supply.get(product, 0)
                if supply > 0:
                    # Add edge from super-source to supplier locations
                    G.add_edge(source, loc.id, capacity=supply)
                    total_supply += supply
                elif supply < 0:
                    # Add edge from demander locations to super-sink
                    demand = -supply
                    G.add_edge(loc.id, sink, capacity=demand)
                    total_demand += demand

            # Check for overall balance
            if abs(total_supply - total_demand) > 1e-9:
                if verbose:
                    print(
                        f"Product {product.id}: Total supply ({total_supply}) "
                        f"does not equal total demand ({total_demand}). Infeasible."
                    )
                return False

            if total_supply == 0:
                if verbose:
                    print(f"Product {product.id}: No supply or demand. Skipping.")
                continue

            # 3. Add transport links as edges with capacities
            for link in self.transport_links:
                # A product can only be moved on a link if the transport type is valid for it.
                if link.transport_type in product.valid_transports:
                    # The capacity of the link for this product is the transport's total
                    # capacity divided by the product's size.
                    capacity = link.transport_type.capacity / product.size
                    G.add_edge(
                        link.location_l1.id, link.location_l2.id, capacity=10000000000
                    )
            # 4. Calculate maximum flow
            try:
                max_flow_value = nx.maximum_flow_value(G, source, sink)
            except nx.NetworkXUnbounded:
                # This can happen if there's a path from source to sink with infinite capacity edges.
                # In our case, capacities are finite, but it's good practice to handle it.
                max_flow_value = float("inf")
            # 5. Check if max flow is sufficient to meet demand
            if max_flow_value < total_demand - 1e-9:  # Use a tolerance
                if verbose:
                    print(
                        f"Product {product.id}: Infeasible. Max flow ({max_flow_value}) is less than total demand ({total_demand})."
                    )
                # return False
            else:
                if verbose:
                    print(
                        f"Product {product.id}: Feasible. Max flow ({max_flow_value}) can satisfy total demand ({total_demand})."
                    )

        return True

    def satisfy(self, variable: MultibatchingSolution) -> bool:
        """
        Checks if a given MultibatchingSolution is feasible.

        Feasibility conditions checked:
        1.  **Flow Conservation**: For each location and product, the total incoming flow
            plus local supply must equal total outgoing flow plus local demand.
        2.  **Packing Capacity**: For each PackingTransport, the total size of products
            packed must not exceed the capacity of its associated TransportType.
        3.  **Product-Transport Compatibility**: Each product in a PackingTransport must
            be valid for the TransportType of its TransportLink.
        4.  **Non-negative Frequencies**: nb_packing must be non-negative.
        5.  **Non-negative Packed Units**: Units in product_packing must be non-negative.
        6.  **Integer Packing Units**: Units in product_packing must be integers (as per problem definition).
        7.  **Positive Frequency for Non-empty Packing**: If nb_packing is > 0, then product_packing cannot be empty.
            (This is usually implicitly handled by solver, but explicit check for robustness).
        """
        epsilon = 1e-6  # Tolerance for floating-point comparisons

        # --- 1. Initialize net flows for each location and product ---
        # { (location_id, product_id): net_flow }
        # Start with initial net_supply/demand from the problem definition
        current_net_flows = defaultdict(float)
        for loc in self.locations:
            for product in self.products:
                current_net_flows[(loc.id, product.id)] = loc.net_supply.get(product, 0)

        # --- 2. Process all PackingTransport flows ---
        for flow_item in variable.list_flows:
            link = flow_item.transport_link
            # Check 4: Non-negative Frequencies
            if flow_item.nb_packing < -epsilon:
                logger.warning(
                    f"Validation Error: Negative nb_packing for link {link.location_l1.id}->{link.location_l2.id}: {flow_item.nb_packing}"
                )
                return False

            # Calculate packed volume for this specific packing
            current_packing_volume = 0.0
            # Check 7: If nb_packing > 0, packing must not be empty.
            if flow_item.nb_packing > epsilon and not flow_item.product_packing:
                logger.warning("Product packing : ", flow_item.product_packing)
                logger.warning(
                    f"Validation Error: Non-zero nb_packing with empty product_packing for link {link.location_l1.id}->{link.location_l2.id}"
                )
                return False

            for product, units in flow_item.product_packing.items():
                # Check 5: Non-negative Packed Units
                if units == 0:
                    continue
                if units < -epsilon:
                    logger.warning(
                        f"Validation Error: Negative packed units for P{product.id} in packing {flow_item.product_packing}: {units}"
                    )
                    return False

                # Check 6: Integer Packing Units
                # If units are not integers, the packing itself is invalid
                if abs(units - round(units)) > epsilon:
                    logger.info(
                        f"Validation Error: Non-integer packed units for P{product.id} in packing {flow_item.product_packing}: {units}"
                    )
                    return False

                # Check 3: Product-Transport Compatibility
                if link.transport_type not in product.valid_transports:
                    logger.warning(
                        f"Validation Error: Product P{product.id} not compatible with TR{link.transport_type.id} on link {link.location_l1.id}->{link.location_l2.id}"
                    )
                    return False
                current_packing_volume += product.size * units
                # Update net flows for flow conservation check
                # Outgoing flow from link.location_l1
                current_net_flows[(link.location_l1.id, product.id)] -= round(
                    units * flow_item.nb_packing
                )
                # Incoming flow to link.location_l2
                current_net_flows[(link.location_l2.id, product.id)] += round(
                    units * flow_item.nb_packing
                )

            # Check 2: Packing Capacity
            if current_packing_volume > link.transport_type.capacity + epsilon:
                logger.warning(
                    f"trying to load : {[(p.name, p.size) for p in flow_item.product_packing]}"
                )
                logger.warning(
                    f"Validation Error: Packing volume {current_packing_volume} exceeds capacity {link.transport_type.capacity} for link {link.location_l1.id}->{link.location_l2.id}"
                )
                return False

        # --- 3. Final Flow Conservation Check ---
        # After processing all flows, all current_net_flows should be close to zero
        any_imbalance = False
        for (loc_id, prod_id), net_flow in current_net_flows.items():
            if abs(net_flow) > epsilon:
                logger.warning(
                    f"Validation Error: Flow imbalance for Location {loc_id}, Product P{prod_id}: {net_flow}"
                )
                any_imbalance = True
                # return False
        if any_imbalance:
            return False
        return True  # All checks passed

    def check_flows(self, solution: MultibatchingSolution):
        current_net_flows = defaultdict(float)
        for loc in self.locations:
            for product in self.products:
                current_net_flows[(loc.id, product.id)] = loc.net_supply.get(product, 0)

        for flow_item in solution.list_flows:
            link = flow_item.transport_link
            for product, units in flow_item.product_packing.items():
                current_net_flows[(link.location_l1.id, product.id)] -= (
                    units * flow_item.nb_packing
                )
                # Incoming flow to link.location_l2
                current_net_flows[(link.location_l2.id, product.id)] += (
                    units * flow_item.nb_packing
                )
        epsilon = 1e-6  # Tolerance for floating-point comparisons
        for (loc_id, prod_id), net_flow in current_net_flows.items():
            if abs(net_flow) > epsilon:
                print(
                    f"Validation Error: Flow imbalance for Location {loc_id}, Product P{prod_id}: {net_flow}"
                )

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(dict_attribute_to_type={})

    def get_solution_type(self) -> type[Solution]:
        return MultibatchingSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                cost: ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1)
                for cost in ["transport", "emission"]
            },
        )

    def get_supplier_location(self, product: Product):
        """Returns the location where a given product is done (i.e net-supply>0)"""
        return [
            (l, l.net_supply.get(product, 0))
            for l in self.locations
            if l.net_supply.get(product, 0) > 0
        ]

    def get_consumer_location(self, product: Product):
        """Returns the location where a given product is consumed (i.e net-supply<0)"""
        return [
            (l, l.net_supply.get(product, 0))
            for l in self.locations
            if l.net_supply.get(product, 0) < 0
        ]

    def get_outgoing_tl_from_location(self, location: Location):
        """Returns all transport link that goes out of a given location"""
        return [tl for tl in self.transport_links if tl.location_l1 == location]

    def get_total_supply(self, product: Product):
        return sum([max(loc.net_supply.get(product, 0), 0) for loc in self.locations])

    def get_total_demand(self, product: Product):
        return -sum([min(loc.net_supply.get(product, 0), 0) for loc in self.locations])


def compute_graph_for_product(
    problem: MultibatchingProblem, product: Product
) -> nx.DiGraph:
    G = nx.DiGraph()
    # source = 'source'
    # sink = 'sink'
    # G.add_nodes_from([source, sink])

    total_supply = 0
    total_demand = 0
    # 2. Add nodes for each location and connect source/sink
    demand_per_location = {}
    for loc in problem.locations:
        supply = loc.net_supply.get(product, 0)
        G.add_node(loc.id, demand=0)
        demand_per_location[loc.name] = supply
        if supply > 0:
            # Add edge from super-source to supplier locations
            # G.add_edge(source, loc.id, capacity=supply)
            total_supply += supply
        elif supply < 0:
            # Add edge from demander locations to super-sink
            demand = -supply
            # G.add_edge(loc.id, sink, capacity=demand)
            total_demand += demand

    G.add_node("source", demand=-total_supply)
    G.add_node("sink", demand=total_demand)
    for loc in demand_per_location:
        if demand_per_location[loc] > 0:
            G.add_edge("source", loc, capacity=demand_per_location[loc], weight=0)
            print(f"adding link from source to " + loc)
        if demand_per_location[loc] < 0:
            G.add_edge(loc, "sink", capacity=-demand_per_location[loc], weight=0)
            print(f"adding link from {loc} to sink")

    for link in problem.transport_links:
        # A product can only be moved on a link if the transport type is valid for it.
        if link.transport_type in product.valid_transports:
            # The capacity of the link for this product is the transport's total
            # capacity divided by the product's size.
            # capacity = link.transport_type.capacity / product.size
            if link.transport_type.capacity == 0:
                continue
            coefficient = (
                product.size
                / link.transport_type.capacity
                * (link.transport_type.cost + link.transport_type.emissions)
                * link.distance
            )
            intermediary_node = (
                link.location_l1.name,
                link.location_l2.name,
                link.transport_type.name,
            )
            # G.add_edge(link.location_l1.id,
            #           link.location_l2.id,
            #           weight=int(100*coefficient),
            #           capacity=total_demand)
            G.add_node(intermediary_node, demand=0)
            G.add_edge(
                link.location_l1.name,
                intermediary_node,
                weight=int(100 * 1 / 2 * coefficient),
                capacity=total_supply,
            )
            G.add_edge(
                intermediary_node,
                link.location_l2.name,
                weight=int(100 * 1 / 2 * coefficient),
                capacity=total_supply,
            )
    dict_flow = nx.max_flow_min_cost(
        G, s="source", t="sink", capacity="capacity", weight="weight"
    )
    # dict_flow = nx.min_cost_flow(G,
    #                              demand="demand",
    #                              capacity="capacity",
    #                              weight="weight")
    cost_of_flow = nx.cost_of_flow(G, flowDict=dict_flow, weight="weight")
    logger.info(f"cost of flow {cost_of_flow}")
    return G, dict_flow


def analyse_solution(
    base_solution: MultibatchingSolution, new_solution: MultibatchingSolution
):
    """Compare 2 solutions, link per link, only looking at the number of trips"""
    nb_trips_per_link_base_solution = {
        x.transport_link: x.nb_packing for x in base_solution.list_flows
    }
    flow_per_link = {
        x.transport_link: {
            p: int(x.product_packing[p] * x.nb_packing) for p in x.product_packing
        }
        for x in base_solution.list_flows
    }
    nb_trips_per_link_new = defaultdict(lambda: 0)
    for flow in new_solution.list_flows:
        link = flow.transport_link
        nb_trips_per_link_new[link] += flow.nb_packing
    changes = []
    for link in nb_trips_per_link_new:
        if nb_trips_per_link_base_solution[link] != nb_trips_per_link_new[link]:
            changes.append(
                (
                    link,
                    flow_per_link[link],
                    nb_trips_per_link_new[link],
                    nb_trips_per_link_base_solution[link],
                )
            )
    return changes
