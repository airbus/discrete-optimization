#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Optional

import networkx as nx

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
    Product,
    TransportLink,
)

logger = logging.getLogger(__name__)


class NetxMultibatchingSolver(SolverDO):
    problem: MultibatchingProblem
    hyperparameters = [
        CategoricalHyperparameter(
            name="restrict_to_shortest_paths", choices=[True, False], default=False
        ),
        FloatHyperparameter(
            name="shortest_path_tolerance",
            depends_on=[("restrict_to_shortest_paths", [True])],
            default=1.2,
            low=0,
            high=5,
        ),
        FloatHyperparameter(name="weight_on_transport", default=1, low=0, high=1000),
        FloatHyperparameter(name="weight_on_emission", default=1, low=0, high=1000),
    ]

    def compute_optimal_flow_for_product(
        self, product: Product, weight_on_transport: float, weight_on_emission: float
    ):
        problem = self.problem
        G = nx.DiGraph()
        total_supply = 0
        total_demand = 0
        # 2. Add nodes for each location and connect source/sink
        demand_per_location = {}
        for loc in problem.locations:
            supply = loc.net_supply.get(product, 0)
            G.add_node(loc.id, demand=0, type="location")
            demand_per_location[loc.id] = supply
            if supply > 0:
                total_supply += supply
            elif supply < 0:
                demand = -supply
                total_demand += demand

        G.add_node("source", demand=-total_supply, type="source")
        G.add_node("sink", demand=total_demand, type="sink")
        for loc in demand_per_location:
            if demand_per_location[loc] > 0:
                G.add_edge("source", loc, capacity=demand_per_location[loc], weight=0)
            if demand_per_location[loc] < 0:
                G.add_edge(loc, "sink", capacity=-demand_per_location[loc], weight=0)
        for index_link in range(problem.nb_transport_links):
            if self.use_shortest_path:
                if index_link not in self.valid_links[product.id]:
                    continue
            link = problem.transport_links[index_link]
            if link.transport_type in product.valid_transports:
                if link.transport_type.capacity == 0:
                    continue
                if link.transport_type.capacity < product.size:
                    continue
                # coefficient = (max(1, product.size / link.transport_type.capacity) *
                #               (link.transport_type.cost + link.transport_type.emissions) * link.distance)
                coefficient = (
                    product.size
                    / link.transport_type.capacity
                    * (
                        weight_on_transport * link.transport_type.cost
                        + weight_on_emission * link.transport_type.emissions
                    )
                    * link.distance
                )
                intermediary_node = (
                    link.location_l1.id,
                    link.location_l2.id,
                    link.transport_type.id,
                )
                G.add_node(intermediary_node, demand=0, type="intermediary")
                G.add_edge(
                    link.location_l1.id,
                    intermediary_node,
                    weight=int(coefficient),
                    capacity=total_supply,
                )
                G.add_edge(
                    intermediary_node,
                    link.location_l2.id,
                    weight=0,
                    capacity=total_supply,
                )
        dict_flow = nx.min_cost_flow(
            G, demand="demand", capacity="capacity", weight="weight"
        )
        cost_of_flow = nx.cost_of_flow(G, flowDict=dict_flow, weight="weight")
        return G, dict_flow, cost_of_flow

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        use_shortest_path = kwargs["restrict_to_shortest_paths"]
        sp_tolerance = kwargs["shortest_path_tolerance"]
        self.use_shortest_path = use_shortest_path
        if use_shortest_path:
            self.valid_links = self.problem.precompute_valid_links(
                tolerance=sp_tolerance
            )
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }
        # demand_per_product = {p: self.problem.get_total_demand(p) for p in self.problem.products}
        # flow_per_product = defaultdict(lambda: set())
        packs: dict[TransportLink, PackingTransport] = {}
        total_cost_of_flow = 0
        for product in supply_per_product:
            if supply_per_product[product] == 0:
                continue
            graph, flow, cost_of_flow = self.compute_optimal_flow_for_product(
                product=product,
                weight_on_transport=kwargs["weight_on_transport"],
                weight_on_emission=kwargs["weight_on_emission"],
            )
            logger.info(f"{product.name}, cost of flow : , {cost_of_flow}")
            total_cost_of_flow += cost_of_flow
            for node_1 in flow:
                type_node_1 = graph.nodes[node_1]["type"]
                for node_2 in flow[node_1]:
                    type_node_2 = graph.nodes[node_2]["type"]
                    if flow[node_1][node_2] > 0:
                        value = flow[node_1][node_2]
                        if type_node_2 == "intermediary":
                            transport_link = (
                                self.problem.loc_and_transport_type_to_transport_link[
                                    node_2
                                ]
                            )
                            if transport_link in packs:
                                packs[transport_link].product_packing[product] = value
                            else:
                                packing = PackingTransport(
                                    transport_link=transport_link,
                                    product_packing={product: value},
                                    nb_packing=1,
                                )
                                packs[transport_link] = packing
        logger.info(f"{total_cost_of_flow} total cost flow")
        sol = MultibatchingSolution(
            problem=self.problem, list_flows=list(packs.values())
        )
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])
