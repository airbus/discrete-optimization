#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True

from discrete_optimization.binpack.solvers.cpsat import ModelingBinPack
from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    ParametersMilp,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
    TransportType,
)

logger = logging.getLogger(__name__)


class GurobiMultibatchingSolver(GurobiMilpSolver):
    """Gurobi solver for the Multibatching problem, based on a flow formulation.

    This solver supports two main modeling variants:
    1.  **Standard Flow Model**: Allows multiple different products to be packed
        into the same transport (multi-batching).
    2.  **Single Batching Flow Model**: A more constrained version where each transport
        can only carry a single type of product.

    Attributes:
        problem (MultibatchingProblem): The multibatching problem instance to solve.
        variables (Dict): A dictionary to store Gurobi variables created for the model.
        single_batching (bool): Flag indicating which model variant to use.
        scaling_factor (int): A factor to scale floating-point costs into integers for the model.
    """

    problem: MultibatchingProblem
    hyperparameters = [
        CategoricalHyperparameter(
            name="add_lb_constraint_nb_trips", choices=[True, False], default=False
        )
    ]

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables: Dict[str, Any] = {}
        self.single_batching: bool = False
        self.scaling_factor = kwargs.get("scaling_factor", 1)

    def init_model(self, single_batching: bool = False, **kwargs: Any) -> None:
        """
        Initializes the Gurobi model by dispatching to the appropriate
        model-building method based on the `single_batching` flag.
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.single_batching = single_batching
        if self.single_batching:
            self._init_model_flows_single_batching(**kwargs)
        else:
            self._init_model_flows_multi_batching(**kwargs)
        self.model.update()

    def precompute_valid_links(self, tolerance=0.1):
        """
        Robust Heuristic: Identifies which transport links are relevant for each product.

        It builds a specific graph G_p for each product P, containing only
        the TransportLinks compatible with P (i.e. link.transport_type in p.valid_transports).

        A link (u, v) is relevant for product P if there exists ANY pair of
        (Source s, Sink d) for P such that:
           dist_p(s, u) + link_len(u, v) + dist_p(v, d) <= (1 + tolerance) * dist_p(s, d)
        """
        import networkx as nx

        valid_links_per_product = defaultdict(set)

        for p in self.problem.products:
            # 1. Identify Producers and Consumers for this specific product
            sources = [
                l.id for l in self.problem.locations if l.net_supply.get(p, 0) > 1e-6
            ]
            sinks = [
                l.id for l in self.problem.locations if l.net_supply.get(p, 0) < -1e-6
            ]

            if not sources or not sinks:
                continue

            # 2. Build the Product-Specific Graph G_p
            # Only include links where the transport type is allowed for this product
            G_p = nx.DiGraph()

            # We also map (u, v) -> list of (link_index, weight)
            # because there might be multiple transport modes between u and v (e.g. Truck vs Train)
            # and we need to check them individually.
            edges_to_indices = defaultdict(list)

            for i, tl in enumerate(self.problem.transport_links):
                # KEY CHECK: Is this transport type valid for this product?
                # Also check capacity constraints if relevant (size <= capacity)
                if (
                    tl.transport_type in p.valid_transports
                    and p.size <= tl.transport_type.capacity
                ):
                    u, v = tl.location_l1.id, tl.location_l2.id
                    w = 1  # tl.distance

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

    def _init_model_flows_multi_batching(self, **kwargs: Any) -> None:
        """Builds the standard flow model where multiple products can share a transport."""
        use_shortest_path = kwargs.get("restrict_to_shortest_paths", False)
        sp_tolerance = kwargs.get(
            "shortest_path_tolerance", 0.0
        )  # 0.0 = Strict, 0.2 = +20% length allowed

        # Option B: No incoming flow at producers
        no_incoming_at_source = kwargs.get("prevent_incoming_at_source", False)

        valid_links_map = None
        if use_shortest_path:
            logger.info(
                f"Computing valid links (Shortest Path Heuristic, tol={sp_tolerance})..."
            )
            valid_links_map = self.precompute_valid_links(tolerance=sp_tolerance)

        logger.info("Initializing Gurobi model (Multi-Batching Flow)...")
        # env = gurobipy.Env()
        model = gurobipy.Model("multibatching-flow")  # , env=env)

        # --- Pre-computation ---
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }

        # --- Variables ---
        nb_trips = {
            i: model.addVar(vtype=gurobipy.GRB.INTEGER, name=f"nb_trips_{i}")
            for i in range(self.problem.nb_transport_links)
        }
        # for i in nb_trips:
        #    nb_trips[i].BranchPriority = 20

        flows = {
            i: {
                p_idx: model.addVar(
                    vtype=gurobipy.GRB.INTEGER,
                    ub=supply_per_product[p],
                    name=f"flow_{i}_{p_idx}",
                )
                for p_idx, p in enumerate(self.problem.products)
                if (
                    # 1. Transport Compatibility & Capacity
                    tl.transport_type in p.valid_transports
                    and tl.transport_type.capacity > p.size
                    # 2. Heuristic A: Restrict to Valid Paths (Shortest Path logic)
                    and (
                        valid_links_map is None or i in valid_links_map.get(p.id, set())
                    )
                    # 3. Heuristic B: Prevent Incoming Flow at Producers (No U-turns to factory)
                    and (
                        not no_incoming_at_source
                        or tl.location_l2.net_supply.get(p, 0) <= 1e-6
                    )
                )
            }
            for i, tl in enumerate(self.problem.transport_links)
        }
        # for i in flows:
        #    for p in flows[i]:
        #        flows[i][p].BranchPriority = 30

        # --- Constraints ---
        # Capacity Constraint
        for i, tl in enumerate(self.problem.transport_links):
            total_volume = gurobipy.quicksum(
                flows[i][p_idx] * p.size
                for p_idx, p in enumerate(self.problem.products)
                if p_idx in flows[i]
            )
            model.addLConstr(total_volume <= nb_trips[i] * tl.transport_type.capacity)

        # Flow Conservation
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = gurobipy.quicksum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l2 == loc and p_idx in flows[i]
                )
                out_flow = gurobipy.quicksum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l1 == loc and p_idx in flows[i]
                )
                model.addLConstr(in_flow + loc.net_supply.get(p, 0) == out_flow)
        if kwargs["add_lb_constraint_nb_trips"]:
            self.add_advanced_capacity_constraints(
                model=model,
                flows_variables=flows,
                nb_trips_per_link=nb_trips,
                solver_type="gurobi",
                max_k=10,
            )
            # self.add_global_flow_limit_constraints(model=model,
            #                                       flows_variables=flows,
            #                                       solver_type="gurobi",
            #                                       factor=2)
            # self.add_limit_active_links_constraints(model=model,
            #                                        flows_variables=flows,
            #                                        solver_type="gurobi",
            #                                        factor=3)
            model.update()
        # --- Objective ---
        self._set_objective(model, nb_trips, flows)

        self.model = model

        self.variables = {"nb_trips": nb_trips, "flows": flows}

    def _init_model_flows_single_batching(self, **kwargs: Any) -> None:
        """Builds the constrained flow model where only one product type is allowed per transport."""
        logger.info("Initializing Gurobi model (Single-Batching Flow)...")
        model = gurobipy.Model("multibatching-single-flow")

        # --- Pre-computation ---
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }

        # --- Variables ---
        nb_trips = {
            i: model.addVar(vtype=gurobipy.GRB.INTEGER, name=f"nb_trips_{i}")
            for i in range(self.problem.nb_transport_links)
        }
        flows = {
            i: {
                p_idx: model.addVar(
                    vtype=gurobipy.GRB.INTEGER,
                    ub=supply_per_product[p],
                    name=f"flow_{i}_{p_idx}",
                )
                for p_idx, p in enumerate(self.problem.products)
                if tl.transport_type in p.valid_transports
                and tl.transport_type.capacity > p.size
            }
            for i, tl in enumerate(self.problem.transport_links)
        }
        # Trips dedicated to a specific product on a specific link
        nb_trips_per_product = {
            (i, p_idx): model.addVar(
                vtype=gurobipy.GRB.INTEGER, name=f"nb_trips_{i}_{p_idx}"
            )
            for i in range(self.problem.nb_transport_links)
            for p_idx in flows[i]
        }

        # --- Constraints ---
        for i, tl in enumerate(self.problem.transport_links):
            # Total trips on a link is the sum of product-dedicated trips
            model.addLConstr(
                gurobipy.quicksum(
                    nb_trips_per_product[(i, p_idx)] for p_idx in flows[i]
                )
                == nb_trips[i]
            )

            for p_idx, p in enumerate(self.problem.products):
                if p_idx in flows[i]:
                    # Capacity constraint for single-product trips
                    products_per_trip = (
                        tl.transport_type.capacity // p.size if p.size > 0 else 0
                    )
                    if products_per_trip > 0:
                        model.addLConstr(
                            flows[i][p_idx]
                            <= nb_trips_per_product[(i, p_idx)] * products_per_trip
                        )
                    else:  # Cannot transport this product
                        model.addLConstr(flows[i][p_idx] == 0)

        # Flow conservation (same as multi-batching)
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = gurobipy.quicksum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l2 == loc and p_idx in flows[i]
                )
                out_flow = gurobipy.quicksum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l1 == loc and p_idx in flows[i]
                )
                model.addLConstr(in_flow + loc.net_supply.get(p, 0) == out_flow)

        # --- Objective ---
        self._set_objective(model, nb_trips, flows)

        self.model = model
        self.variables = {
            "nb_trips": nb_trips,
            "flows": flows,
            "nb_trips_per_product": nb_trips_per_product,
        }

    def add_advanced_capacity_constraints(
        self, model, flows_variables, nb_trips_per_link, solver_type="gurobi", max_k=30
    ):
        """
        Adds generalized lower bound constraints on the number of trips.
        For each k in [2..max_k], we consider items with size > Capacity/k.
        Since at most k-1 such items fit in one vehicle, we add the cut:
            (k-1) * NbTrips >= Sum(Flows of these items)
        """
        # Select summation function based on solver
        if solver_type == "gurobi":
            import gurobipy

            sum_func = gurobipy.quicksum
        else:
            sum_func = sum

        for index_tl in range(self.problem.nb_transport_links):
            if index_tl not in nb_trips_per_link:
                continue

            tl = self.problem.transport_links[index_tl]
            capacity = tl.transport_type.capacity
            if capacity <= 0:
                continue

            nb_trips_var = nb_trips_per_link[index_tl]
            current_flow_vars = flows_variables[index_tl]

            # Iterate k from 2 up to max_k (e.g., items > C/2, > C/3, ..., > C/10)
            for k in range(2, max_k + 1):
                relevant_flows = []

                for index_product, flow_var in current_flow_vars.items():
                    product = self.problem.products[index_product]
                    # Check condition: size > Capacity / k  <==>  size * k > Capacity
                    if product.size * k > capacity:
                        relevant_flows.append(flow_var)

                if relevant_flows:
                    # Constraint: (k-1) * NbTrips >= Sum(Relevant Flows)
                    lhs = (k - 1) * nb_trips_var
                    rhs = sum_func(relevant_flows)

                    if solver_type == "gurobi":
                        model.addLConstr(lhs >= rhs)
                    else:  # CP-SAT
                        model.Add(lhs >= rhs)

    def add_limit_active_links_constraints(
        self, model, flows_variables, solver_type="gurobi", factor=2
    ):
        """
        Limits the number of transport links where a given product flows.
        Limit = factor * (Number of nodes with non-zero supply/demand for that product).
        This helps pruning the search space by forcing sparser paths (e.g. tree-like).
        """
        if factor is None:
            return

        # Select summation function based on solver
        if solver_type == "gurobi":
            import gurobipy

            sum_func = gurobipy.quicksum
        else:
            sum_func = sum

        for index_product in range(self.problem.nb_products):
            product = self.problem.products[index_product]

            # 1. Count Active Nodes (Source or Sink for this product)
            # A node is active if it has non-zero net supply/demand
            nb_active_nodes = 0
            for loc in self.problem.locations:
                if abs(loc.net_supply.get(product, 0)) > 1e-6:
                    nb_active_nodes += 1

            # 2. Calculate Limit
            # A tree connecting N nodes has N-1 edges.
            # We use (N * factor) to allow some transshipment/cycles.
            # We ensure limit is at least nb_active_nodes to prevent infeasibility.
            limit = int(math.ceil(nb_active_nodes * factor))
            limit = max(limit, nb_active_nodes)  # Safety bound

            # 3. Create Indicators for Active Links
            indicators = []

            # Collect all flow vars for this product across all links
            # flows_variables structure: {link_idx: {prod_idx: var}}
            for index_tl in flows_variables:
                if index_product in flows_variables[index_tl]:
                    flow_var = flows_variables[index_tl][index_product]

                    if solver_type == "gurobi":
                        # z=1 if flow > 0
                        z = model.addVar(
                            vtype=gurobipy.GRB.BINARY,
                            name=f"used_{index_tl}_{index_product}",
                        )
                        # Constraint: flow <= Capacity * z
                        # Using total supply as Big-M for flow bound
                        M = self.problem.get_total_supply(product)
                        model.addLConstr(flow_var <= M * z)
                        indicators.append(z)
                    else:  # CP-SAT
                        z = model.NewBoolVar(f"used_{index_tl}_{index_product}")
                        # If z=0, then flow must be 0. (Equivalent to flow > 0 => z=1)
                        model.Add(flow_var == 0).OnlyEnforceIf(z.Not())
                        indicators.append(z)

            # 4. Add Cardinality Constraint
            if indicators:
                if solver_type == "gurobi":
                    model.addLConstr(
                        sum_func(indicators) <= limit,
                        name=f"limit_links_{index_product}",
                    )
                else:
                    model.Add(sum_func(indicators) <= limit)

    def add_global_flow_limit_constraints(
        self, model, flows_variables, solver_type="cpsat", factor=2
    ):
        """
        Limits the total volume of flow for each product across the entire network.
        Constraint: Sum(Flows_of_Product_P) <= factor * TotalDemand_of_Product_P

        - A factor of 1.0 implies direct shipments only (shortest path in terms of hops).
        - A factor of 2.0 implies that, on average, each unit can traverse 2 links (1 transshipment).
        - This prevents excessive detours without using binary variables.
        """
        if factor is None:
            return

        # Select summation function based on solver
        if solver_type == "gurobi":
            import gurobipy

            sum_func = gurobipy.quicksum
        else:
            sum_func = sum

        for index_product in range(self.problem.nb_products):
            product = self.problem.products[index_product]

            # 1. Compute Total Demand (sum of positive requirements) for this product
            total_demand = self.problem.get_total_demand(product)

            # If there is no demand, no flow should happen (handled by conservation, but safe to skip)
            if total_demand <= 0:
                continue

            # 2. Collect all flow variables for this product across all links
            product_flows = []
            for index_tl in flows_variables:
                if index_product in flows_variables[index_tl]:
                    product_flows.append(flows_variables[index_tl][index_product])

            if not product_flows:
                continue

            # 3. Add Constraint
            # We scale the demand by the factor (e.g., 1.5 * 100 = 150 max flow volume)
            limit = factor * total_demand

            if solver_type == "gurobi":
                model.addLConstr(
                    sum_func(product_flows) <= limit,
                    name=f"global_flow_limit_{index_product}",
                )
            else:
                # CP-SAT requires integer bounds
                model.Add(sum_func(product_flows) <= int(limit))

    def _set_objective(self, model, nb_trips, flows):
        """Helper function to build and set the objective expression."""
        transport_cost = gurobipy.quicksum(
            nb_trips[i]
            * int(tl.distance * tl.transport_type.cost * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        emission_cost = gurobipy.quicksum(
            nb_trips[i]
            * int(tl.distance * tl.transport_type.emissions * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        model.setObjective(transport_cost + emission_cost, sense=gurobipy.GRB.MINIMIZE)

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        """
        Converts a Gurobi solution into a `MultibatchingSolution`.
        If `single_batching` is used, it reconstructs the exact packings. Otherwise, it creates
        an "average" packing for multi-batch flows.
        """
        list_flows = []
        nb_trips_vals = {
            i: get_var_value_for_current_solution(var)
            for i, var in self.variables["nb_trips"].items()
        }

        for i, tl in enumerate(self.problem.transport_links):
            if nb_trips_vals.get(i, 0) > 1e-6:
                if self.single_batching:
                    # Reconstruct exact packings for single-batching
                    for p_idx, p in enumerate(self.problem.products):
                        if (i, p_idx) in self.variables["nb_trips_per_product"]:
                            flow_val = get_var_value_for_current_solution(
                                self.variables["flows"][i][p_idx]
                            )
                            if flow_val > 1e-6:
                                products_per_trip = tl.transport_type.capacity // p.size
                                num_full_trips = int(flow_val // products_per_trip)
                                remainder = flow_val % products_per_trip

                                if num_full_trips > 0:
                                    list_flows.append(
                                        PackingTransport(
                                            transport_link=tl,
                                            product_packing={p: products_per_trip},
                                            nb_packing=num_full_trips,
                                        )
                                    )
                                if remainder > 1e-6:
                                    list_flows.append(
                                        PackingTransport(
                                            transport_link=tl,
                                            product_packing={p: remainder},
                                            nb_packing=1,
                                        )
                                    )
                else:
                    # Create average packing for multi-batching
                    product_packing = {}
                    for p_idx, p in enumerate(self.problem.products):
                        if p_idx in self.variables["flows"][i]:
                            flow_val = get_var_value_for_current_solution(
                                self.variables["flows"][i][p_idx]
                            )
                            if flow_val > 1e-6:
                                product_packing[p] = flow_val / nb_trips_vals[i]
                    if product_packing:
                        list_flows.append(
                            PackingTransport(
                                transport_link=tl,
                                product_packing=product_packing,
                                nb_packing=int(round(nb_trips_vals[i])),
                            )
                        )
        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)

    def convert_to_variable_values(
        self, solution: MultibatchingSolution
    ) -> Dict["gurobipy.Var", float]:
        """Converts a `MultibatchingSolution` to a dictionary of Gurobi variable values for warm start."""
        variable_values = {}

        # Aggregate flows and trips from the solution
        agg_flows = defaultdict(lambda: defaultdict(float))
        agg_trips = defaultdict(float)
        agg_trips_per_product = defaultdict(float)

        for flow in solution.list_flows:
            link_idx = self.problem.transport_links_to_index[flow.transport_link]
            agg_trips[link_idx] += flow.nb_packing
            for product, units in flow.product_packing.items():
                prod_idx = self.problem.product_to_index[product]
                agg_flows[link_idx][prod_idx] += units * flow.nb_packing
                if self.single_batching:
                    agg_trips_per_product[(link_idx, prod_idx)] += flow.nb_packing

        # Map aggregated values to Gurobi variables
        for i, var in self.variables["nb_trips"].items():
            variable_values[var] = agg_trips.get(i, 0)
        for i, flow_map in self.variables["flows"].items():
            for p_idx, var in flow_map.items():
                variable_values[var] = agg_flows.get(i, {}).get(p_idx, 0)

        if self.single_batching:
            for (i, p_idx), var in self.variables["nb_trips_per_product"].items():
                variable_values[var] = agg_trips_per_product.get((i, p_idx), 0)

        return variable_values


# In your milp_flow.py file or a new one

# ... (keep existing imports and the GurobiMultibatchingSolver class) ...


class GurobiMultibatchingSolverUnitFlow(GurobiMilpSolver):
    """
    Gurobi solver for the Multibatching problem using a detailed "unit flow" formulation.

    This model creates variables for each potential trip on each transport link and
    decides the exact content of each trip. It is more precise than an aggregated
    flow model but can be more computationally expensive due to the large number
    of binary and integer variables.

    This formulation is analogous to the `UNIT_FLOW` model in the Cpsat solver.
    """

    hyperparameters = [
        CategoricalHyperparameter(
            "restrict_to_shortest_paths", choices=[True, False], default=False
        ),
        FloatHyperparameter(
            "shortest_path_tolerance",
            low=0,
            high=10,
            default=10,
            depends_on=[("restrict_to_shortest_paths", True)],
        ),
    ]
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables: Dict[str, Any] = {}
        self.scaling_factor = kwargs.get("scaling_factor", 1)

    def precompute_valid_links(self, tolerance=0.1):
        """
        Robust Heuristic: Identifies which transport links are relevant for each product.

        It builds a specific graph G_p for each product P, containing only
        the TransportLinks compatible with P (i.e. link.transport_type in p.valid_transports).

        A link (u, v) is relevant for product P if there exists ANY pair of
        (Source s, Sink d) for P such that:
           dist_p(s, u) + link_len(u, v) + dist_p(v, d) <= (1 + tolerance) * dist_p(s, d)
        """
        import networkx as nx

        valid_links_per_product = defaultdict(set)

        for p in self.problem.products:
            # 1. Identify Producers and Consumers for this specific product
            sources = [
                l.id for l in self.problem.locations if l.net_supply.get(p, 0) > 1e-6
            ]
            sinks = [
                l.id for l in self.problem.locations if l.net_supply.get(p, 0) < -1e-6
            ]

            if not sources or not sinks:
                continue

            # 2. Build the Product-Specific Graph G_p
            # Only include links where the transport type is allowed for this product
            G_p = nx.DiGraph()

            # We also map (u, v) -> list of (link_index, weight)
            # because there might be multiple transport modes between u and v (e.g. Truck vs Train)
            # and we need to check them individually.
            edges_to_indices = defaultdict(list)

            for i, tl in enumerate(self.problem.transport_links):
                # KEY CHECK: Is this transport type valid for this product?
                # Also check capacity constraints if relevant (size <= capacity)
                if (
                    tl.transport_type in p.valid_transports
                    and p.size <= tl.transport_type.capacity
                ):
                    u, v = tl.location_l1.id, tl.location_l2.id
                    w = 1  # tl.distance

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

    def init_model(self, max_trips_per_link: int = 15, **kwargs: Any) -> None:
        """
        Builds the Gurobi model based on a detailed, per-trip ("unit flow") formulation.

        Args:
            max_trips_per_link (int): A hard upper bound on the number of trips that can be
                                      used on any single transport link. This is necessary
                                      to define the variable space.
        """
        logger.info("Initializing Gurobi model (Unit Flow)...")
        logger.info(f"Max trips : {max_trips_per_link}")
        use_shortest_path = kwargs.get("restrict_to_shortest_paths", False)
        sp_tolerance = kwargs.get(
            "shortest_path_tolerance", 0.0
        )  # 0.0 = Strict, 0.2 = +20% length allowed
        # Option B: No incoming flow at producers
        no_incoming_at_source = kwargs.get("prevent_incoming_at_source", False)
        valid_links_map = None
        if use_shortest_path:
            logger.info(
                f"Computing valid links (Shortest Path Heuristic, tol={sp_tolerance})..."
            )
            valid_links_map = self.precompute_valid_links(tolerance=sp_tolerance)
        model = gurobipy.Model("multibatching-unit-flow")
        sol: MultibatchingSolution = kwargs.get("solution", None)
        delta: int = kwargs.get("delta_to_solution", 1)
        max_nb_trip_per_transport_link = {
            i: max_trips_per_link for i in range(self.problem.nb_transport_links)
        }
        if sol is not None:
            max_nb_trip_per_transport_link = defaultdict(lambda: delta)
            for pt in sol.list_flows:
                id_tl = self.problem.transport_links_to_index[pt.transport_link]
                max_nb_trip_per_transport_link[id_tl] += pt.nb_packing
        used_trips = {
            i: {
                j: model.addVar(vtype=gurobipy.GRB.BINARY, name=f"used_{i}_{j}")
                for j in range(max_nb_trip_per_transport_link[i])
            }
            for i in range(self.problem.nb_transport_links)
        }

        # `trip_contents[link_idx][trip_idx][prod_idx]` = quantity of a product on a specific trip
        trip_contents = {
            i: {
                j: {
                    p_idx: model.addVar(
                        vtype=gurobipy.GRB.INTEGER,
                        name=f"cont_{i}_{j}_{p_idx}",
                        ub=int(tl.transport_type.capacity / p.size)
                        if p.size > 0
                        else 0,
                    )
                    for p_idx, p in enumerate(self.problem.products)
                    if (
                        # 1. Transport Compatibility & Capacity
                        tl.transport_type in p.valid_transports
                        and tl.transport_type.capacity > p.size
                        # 2. Heuristic A: Restrict to Valid Paths (Shortest Path logic)
                        and (
                            valid_links_map is None
                            or i in valid_links_map.get(p.id, set())
                        )
                        # 3. Heuristic B: Prevent Incoming Flow at Producers (No U-turns to factory)
                        and (
                            not no_incoming_at_source
                            or tl.location_l2.net_supply.get(p, 0) <= 1e-6
                        )
                    )
                }
                for j in range(max_nb_trip_per_transport_link[i])
            }
            for i, tl in enumerate(self.problem.transport_links)
        }

        # --- Constraints ---
        for i, tl in enumerate(self.problem.transport_links):
            # Symmetry breaking: force trips to be used in order (trip j+1 can be used only if trip j is)
            # for j in range(max_trips_per_link - 1):
            #    model.addLConstr(used_trips[i][j+1] <= used_trips[i][j])
            for j in range(max_nb_trip_per_transport_link[i]):
                # Capacity constraint for each individual trip
                trip_volume = gurobipy.quicksum(
                    trip_contents[i][j][p_idx] * p.size
                    for p_idx, p in enumerate(self.problem.products)
                    if p_idx in trip_contents[i][j]
                )
                model.addConstr(trip_volume <= tl.transport_type.capacity)
                # Link `used_trips` to trip contents. A trip is used if it carries something.
                total_items_in_trip = gurobipy.quicksum(
                    trip_contents[i][j][p_idx] for p_idx in trip_contents[i][j]
                )
                # IF used_trips[i][j] is 1, THEN total_items > 0 (use >= 1 for integers)

                model.addGenConstrIndicator(
                    used_trips[i][j], True, total_items_in_trip >= 1
                )
                # IF used_trips[i][j] is 0, THEN total_items == 0
                model.addGenConstrIndicator(
                    used_trips[i][j], False, total_items_in_trip == 0
                )

        # Aggregate expressions for total flow, linking trip contents to flow conservation
        total_flows = {
            i: {
                p_idx: gurobipy.quicksum(
                    trip_contents[i][j][p_idx]
                    for j in range(max_nb_trip_per_transport_link[i])
                )
                for p_idx, p in enumerate(self.problem.products)
                if p_idx in trip_contents[i][0]
            }
            for i in range(self.problem.nb_transport_links)
        }

        # Flow conservation constraints (using the aggregated total_flows)
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = gurobipy.quicksum(
                    total_flows[i].get(p_idx, 0)
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l2 == loc
                )
                out_flow = gurobipy.quicksum(
                    total_flows[i].get(p_idx, 0)
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l1 == loc
                )
                model.addLConstr(in_flow + loc.net_supply.get(p, 0) == out_flow)

        # --- Objective Function ---
        nb_trips_per_link = {
            i: gurobipy.quicksum(
                used_trips[i][j] for j in range(max_nb_trip_per_transport_link[i])
            )
            for i in range(self.problem.nb_transport_links)
        }
        transport_cost = gurobipy.quicksum(
            nb_trips_per_link[i]
            * int(tl.distance * tl.transport_type.cost * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        emission_cost = gurobipy.quicksum(
            nb_trips_per_link[i]
            * int(tl.distance * tl.transport_type.emissions * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        model.setObjective(transport_cost + emission_cost, sense=gurobipy.GRB.MINIMIZE)

        self.model = model
        self.variables = {
            "used_trips": used_trips,
            "trip_contents": trip_contents,
        }
        self.model.update()

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        """
        Retrieves the solution by reconstructing packings from the detailed trip variables.
        It groups identical packings to create a concise solution.
        """
        list_flows = []
        for i, link in enumerate(self.problem.transport_links):
            packings_on_link = []
            for j in range(len(self.variables["used_trips"][i])):
                if (
                    get_var_value_for_current_solution(
                        self.variables["used_trips"][i][j]
                    )
                    > 0.5
                ):
                    current_packing = {
                        self.problem.products[p_idx]: round(
                            get_var_value_for_current_solution(content_var)
                        )
                        for p_idx, content_var in self.variables["trip_contents"][i][
                            j
                        ].items()
                        if get_var_value_for_current_solution(content_var) > 0.5
                    }
                    if current_packing:
                        # Use a frozenset of items for a hashable representation of the packing
                        packings_on_link.append(frozenset(current_packing.items()))

            if packings_on_link:
                # Count occurrences of each unique packing configuration
                for unique_packing_items, count in Counter(packings_on_link).items():
                    list_flows.append(
                        PackingTransport(
                            transport_link=link,
                            product_packing=dict(unique_packing_items),
                            nb_packing=count,
                        )
                    )
        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)

    def convert_to_variable_values(
        self, solution: MultibatchingSolution
    ) -> Dict["gurobipy.Var", float]:
        used_trips = self.variables["used_trips"]
        trip_contents = self.variables["trip_contents"]
        trips_per_link = defaultdict(list)
        dict_vars = {}
        for flow in solution.list_flows:
            index_tl = self.problem.transport_links_to_index[flow.transport_link]
            for _ in range(flow.nb_packing):
                trips_per_link[index_tl].append(flow.product_packing)
        for index_tl in trips_per_link:
            nb_trips = len(trips_per_link[index_tl])
            for i in range(nb_trips):
                dict_vars[used_trips[index_tl][i]] = 1
            for i in used_trips[index_tl]:
                if i >= nb_trips:
                    dict_vars[used_trips[index_tl][i]] = 0
            for i in range(nb_trips):
                pack = trips_per_link[index_tl][i]
                for p_index in trip_contents[index_tl][i]:
                    p = self.problem.products[p_index]
                    if p in pack:
                        dict_vars[trip_contents[index_tl][i][p_index]] = pack[p]
                    else:
                        dict_vars[trip_contents[index_tl][i][p_index]] = 0
            for i in trip_contents[index_tl]:
                if i >= nb_trips:
                    for p_index in trip_contents[index_tl][i]:
                        dict_vars[trip_contents[index_tl][i][p_index]] = 0
        for index_tl in used_trips:
            if index_tl not in trips_per_link:
                for i in used_trips[index_tl]:
                    dict_vars[used_trips[index_tl][i]] = 0
                for i in trip_contents[index_tl]:
                    for p_index in trip_contents[index_tl][i]:
                        dict_vars[trip_contents[index_tl][i][p_index]] = 0
        return dict_vars


class GurobiMultibatchingSolverWithLazyConstraint(GurobiMultibatchingSolver):
    """
    A Gurobi solver for the multi-batching problem that uses lazy constraints
    to iteratively refine the packing model.
    """

    def init_model(self, **kwargs: Any) -> None:
        """
        Initializes the standard multi-batching flow model and enables lazy constraints.
        """
        # We only apply this logic to the multi-batching case
        super().init_model(single_batching=False, **kwargs)
        # Tell Gurobi to enable lazy constraints
        self.model.setParam("LazyConstraints", 1)
        logger.info("Gurobi model with Lazy Constraints enabled.")

    def solve(
        self,
        callbacks: list[Callback] = None,
        parameters_milp: ParametersMilp = None,
        time_limit: float = 30,
        **kwargs: Any,
    ) -> ResultStorage:
        """
        Overrides the default solve method to use a custom callback for generating
        lazy constraints.
        """
        self.early_stopping_exception = None
        self._current_internal_objective_best_value = None
        self._current_internal_objective_best_bound = None

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        # Use our specialized callback for lazy constraints
        gurobi_lazy_callback = GurobiLazyCallback(
            do_solver=self, callback=callbacks_list
        )

        self.optimize_model(
            parameters_milp=parameters_milp,
            time_limit=time_limit,
            gurobi_callback=gurobi_lazy_callback,
            **kwargs,
        )

        if hasattr(self.model, "ObjVal"):
            self._current_internal_objective_best_value = self.model.ObjVal
        if hasattr(self.model, "ObjBound"):
            self._current_internal_objective_best_bound = self.model.ObjBound

        res = gurobi_lazy_callback.res
        callbacks_list.on_solve_end(res=res, solver=self)

        return res


class GurobiMultibatchingSolverWithLC(GurobiMultibatchingSolver):
    """
    A Gurobi solver for the multi-batching problem that uses lazy constraints
    to iteratively refine the packing model.
    """

    def init_model(self, **kwargs: Any) -> None:
        """
        Initializes the standard multi-batching flow model and enables lazy constraints.
        """
        # We only apply this logic to the multi-batching case
        super().init_model(single_batching=False, **kwargs)
        # self.add_one_hot_variables()
        # self.add_binary_expansion_variables()
        # Tell Gurobi to enable lazy constraints
        self.model.setParam("LazyConstraints", 1)
        logger.info("Gurobi model with Lazy Constraints enabled.")

    def add_one_hot_variables(self):
        flows = self.variables["flows"]
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }
        flows_one_hot = {
            i: {
                p_idx: {
                    k: self.model.addVar(
                        vtype=gurobipy.GRB.BINARY, name=f"flow_{i}_{p_idx}_{k}"
                    )
                    for k in range(supply_per_product[p] + 1)
                }
                for p_idx, p in enumerate(self.problem.products)
                if tl.transport_type in p.valid_transports
                and tl.transport_type.capacity >= p.size
            }
            for i, tl in enumerate(self.problem.transport_links)
        }

        for i in flows_one_hot:
            for p_idx in flows_one_hot[i]:
                for k in flows_one_hot[i][p_idx]:
                    flows_one_hot[i][p_idx][k].BranchPriority = 0
                self.add_linear_constraint(
                    gurobipy.quicksum(
                        [
                            flows_one_hot[i][p_idx][k] * k
                            for k in flows_one_hot[i][p_idx]
                        ]
                    )
                    == flows[i][p_idx],
                    name=f"one_hot_{i}_{p_idx}",
                )
                # for k in flows_one_hot[i][p_idx]:
                #    self.model: gurobipy.Model
                #    self.model.addGenConstrIndicator(flows_one_hot[i][p_idx][k], 1, flows[i][p_idx]==k)
                # self.model.addSOS(gurobipy.GRB.SOS_TYPE1, [flows_one_hot[i][p_idx][k]
                #                                           for k in flows_one_hot[i][p_idx]])
                self.add_linear_constraint(
                    gurobipy.quicksum(
                        [flows_one_hot[i][p_idx][k] for k in flows_one_hot[i][p_idx]]
                    )
                    <= 1,
                    name=f"one_hot_{i}_{p_idx}_",
                )
        self.variables["flows_one_hot"] = flows_one_hot
        # self.add_more_constraint()
        # self.limit_non_zeros()

    def add_more_constraint(self):
        demand_per_product = {
            p: self.problem.get_total_demand(p) for p in self.problem.products
        }
        nb_trips_min_for_product_flows = {}
        for index_product in range(self.problem.nb_products):
            product = self.problem.products[index_product]
            size_product = product.size
            for tl in range(self.problem.nb_transport_types):
                tt: TransportType = self.problem.transport_types[tl]
                capa_transport = tt.capacity
                nb_trips_min_for_product_flows[(index_product, tt)] = [0]
                current_greedy_bins = defaultdict(lambda: 0)
                current_bin = 0
                current_greedy_bins[0] = 0
                for i in range(demand_per_product[product]):
                    if (
                        current_greedy_bins[current_bin] + size_product
                        <= capa_transport
                    ):
                        current_greedy_bins[current_bin] += size_product
                    else:
                        current_bin += 1
                        current_greedy_bins[current_bin] = size_product
                    nb_trips_min_for_product_flows[(index_product, tt)].append(
                        current_bin + 1
                    )
        for index_tl in self.variables["flows"]:
            transport_link = self.problem.transport_links[index_tl]
            transport_type = transport_link.transport_type
            for index_product in self.variables["flows"][index_tl]:
                nb_trips_min = nb_trips_min_for_product_flows[
                    (index_product, transport_type)
                ]
                for j in range(len(nb_trips_min)):
                    if (
                        j + 1
                        in self.variables["flows_one_hot"][index_tl][index_product]
                    ):
                        self.model.addLConstr(
                            50
                            * (
                                1
                                - self.variables["flows_one_hot"][index_tl][
                                    index_product
                                ][j + 1]
                            )
                            + self.variables["nb_trips"][index_tl]
                            >= nb_trips_min[j]
                        )

    def limit_non_zeros(self):
        for p_index in range(self.problem.nb_products):
            links = [
                il
                for il in self.variables["flows"]
                if p_index in self.variables["flows"][il]
            ]
            len_ = len(links)
            zeros = [self.variables["flows_one_hot"][il][p_index][0] for il in links]
            self.model.addLConstr(gurobipy.quicksum(zeros) >= len_ - 4)

    def add_binary_expansion_variables(self):
        flows = self.variables["flows"]
        self.variables["flows_binary_bits"] = {}

        for i, tl in enumerate(self.problem.transport_links):
            self.variables["flows_binary_bits"][i] = {}
            for p_idx, p in enumerate(self.problem.products):
                if p_idx not in flows[i]:
                    continue

                # Number of bits needed to represent max supply
                max_val = self.problem.get_total_supply(p)
                n_bits = max_val.bit_length()

                # Create binary variables for the bits
                bits = [
                    self.model.addVar(
                        vtype=gurobipy.GRB.BINARY, name=f"bit_{i}_{p_idx}_{b}"
                    )
                    for b in range(n_bits)
                ]

                self.variables["flows_binary_bits"][i][p_idx] = bits
                for bit in bits:
                    bit.BranchPriority = 0

                # Link integer flow to binary expansion: flow = sum(2^b * bit_b)
                self.model.addLConstr(
                    gurobipy.quicksum((2**b) * bits[b] for b in range(n_bits))
                    == flows[i][p_idx],
                    name=f"link_binary_{i}_{p_idx}",
                )

    def solve(
        self,
        callbacks: list[Callback] = None,
        parameters_milp: ParametersMilp = None,
        time_limit: float = 30,
        **kwargs: Any,
    ) -> ResultStorage:
        """
        Overrides the default solve method to use a custom callback for generating
        lazy constraints.
        """
        self.early_stopping_exception = None
        self._current_internal_objective_best_value = None
        self._current_internal_objective_best_bound = None

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        # Use our specialized callback for lazy constraints
        # gurobi_lazy_callback = GurobiLC_BinaryExpansion(do_solver=self, callback=callbacks_list)
        gurobi_lazy_callback = GurobiLazyCallback(
            do_solver=self, callback=callbacks_list
        )
        # gurobi_lazy_callback = GurobiLC(do_solver=self, callback=callbacks_list)
        self.optimize_model(
            parameters_milp=parameters_milp,
            time_limit=time_limit,
            gurobi_callback=gurobi_lazy_callback,
            **kwargs,
        )

        if hasattr(self.model, "ObjVal"):
            self._current_internal_objective_best_value = self.model.ObjVal
        if hasattr(self.model, "ObjBound"):
            self._current_internal_objective_best_bound = self.model.ObjBound

        res = gurobi_lazy_callback.res
        callbacks_list.on_solve_end(res=res, solver=self)

        return res


class GurobiLazyCallback:
    """
    Custom Gurobi callback to generate valid, flow-dependent cuts for the multibatching problem.
    """

    def __init__(self, do_solver: GurobiMultibatchingSolver, callback: Callback):
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0
        self.nb_cuts_added = 0
        self.packing_solver = PackingViaBinPacking(problem=self.do_solver.problem)

    def __call__(self, model: "gurobipy.Model", where: int) -> None:
        if where == gurobipy.GRB.Callback.MIPSOL:
            try:
                # 1. Retrieve the candidate solution from the MILP
                candidate_solution = self.do_solver.retrieve_current_solution(
                    get_var_value_for_current_solution=model.cbGetSolution,
                    get_obj_value_for_current_solution=lambda: model.cbGet(
                        gurobipy.GRB.Callback.MIPSOL_OBJ
                    ),
                )

                # 2. Run the greedy packing subproblem to get the "true" number of trips
                # packer = GreedyPackingForMultibatching(problem=self.do_solver.problem)
                # packer.init_from_solution(solution=candidate_solution)
                # packing_result = packer.solve()
                self.packing_solver.init_from_solution(solution=candidate_solution)
                p = ParametersCp.default_cpsat()
                p.nb_process = 16
                packing_result = self.packing_solver.solve(
                    time_limit_per_link=3,
                    bin_packing_solver=SubBrick(
                        cls=CpSatBinPackSolver,
                        kwargs=dict(
                            modeling=ModelingBinPack.SCHEDULING, parameters_cp=p
                        ),
                    ),
                )
                print(
                    "fit packing", self.do_solver.aggreg_from_sol(packing_result[-1][0])
                )
                print(
                    "sat packing", self.do_solver.problem.satisfy(packing_result[-1][0])
                )
                if len(packing_result) == 0:
                    return

                packed_solution = packing_result.get_best_solution_fit()[0]

                # 3. Check for violations and add valid, flow-dependent lazy constraints
                milp_trips = defaultdict(int)
                for flow in candidate_solution.list_flows:
                    milp_trips[flow.transport_link] += flow.nb_packing

                greedy_trips = defaultdict(int)
                for flow in packed_solution.list_flows:
                    greedy_trips[flow.transport_link] += flow.nb_packing

                violation_found = False
                for link, milp_trip_count_float in milp_trips.items():
                    milp_trip_count = round(milp_trip_count_float)
                    greedy_trip_count = greedy_trips.get(link, 0)
                    if greedy_trip_count > milp_trip_count:
                        flows_product_in_this_link = {}
                        for flow in candidate_solution.list_flows:
                            if flow.transport_link == link:
                                for p in flow.product_packing:
                                    if p not in flows_product_in_this_link:
                                        flows_product_in_this_link[p] = 0
                                    flows_product_in_this_link[p] += round(
                                        flow.product_packing[p] * flow.nb_packing
                                    )
                        violation_found = True
                        self.nb_cuts_added += 1
                        link_idx = self.do_solver.problem.transport_links_to_index[link]
                        nb_trips_var = self.do_solver.variables["nb_trips"][link_idx]
                        # Calculate the total volume in the candidate solution for this link
                        candidate_volume = sum(
                            p.size
                            * round(flow.product_packing.get(p, 0) * flow.nb_packing)
                            for p in self.do_solver.problem.products
                            for flow in candidate_solution.list_flows
                            if flow.transport_link == link
                        )
                        if candidate_volume < 1e-6:
                            continue  # Avoid division by zero

                        # Define the total_volume_var as a Gurobi linear expression
                        total_volume_var = gurobipy.quicksum(
                            p.size * self.do_solver.variables["flows"][link_idx][p_idx]
                            for p_idx, p in enumerate(self.do_solver.problem.products)
                            if p_idx in self.do_solver.variables["flows"][link_idx]
                            and p in flows_product_in_this_link
                        )
                        # Add the valid, flow-dependent cut
                        # nb_trips >= (N_greedy / V_candidate) * total_volume_var
                        cut_coefficient = greedy_trip_count / candidate_volume
                        # model.cbLazy((total_volume_var > candidate_volume) >> nb_trips_var > greedy_trips)
                        model.cbLazy(nb_trips_var >= cut_coefficient * total_volume_var)
                        other_link = [
                            index_link
                            for index_link in range(
                                self.do_solver.problem.nb_transport_links
                            )
                            if self.do_solver.problem.transport_links[
                                index_link
                            ].transport_type
                            == link.transport_type
                        ]
                        for ot in other_link:
                            total_volume_var = gurobipy.quicksum(
                                p.size * self.do_solver.variables["flows"][ot][p_idx]
                                for p_idx, p in enumerate(
                                    self.do_solver.problem.products
                                )
                                if p_idx in self.do_solver.variables["flows"][ot]
                                and p in flows_product_in_this_link
                            )
                            nb_trips_var = self.do_solver.variables["nb_trips"][ot]
                            model.cbLazy(
                                nb_trips_var >= cut_coefficient * total_volume_var
                            )
                            # model.cbLazy((total_volume_var > candidate_volume) >> nb_trips_var > greedy_trips)

                        logger.info(
                            f"CUT #{self.nb_cuts_added} on link {link.id}: "
                            f"MILP trips={milp_trip_count}, Greedy trips={greedy_trip_count},"
                            f" Volume={candidate_volume:.2f}. "
                            f"Adding lazy cut."
                        )
                # 4. If no violation was found, the solution is valid from a packing perspective.
                if True:
                    fit = self.do_solver.aggreg_from_sol(packed_solution)
                    self.res.append((packed_solution, fit))
                    self.nb_solutions += 1
                    stopping = self.callback.on_step_end(
                        step=self.nb_solutions, res=self.res, solver=self.do_solver
                    )
                    if stopping:
                        model.terminate()
            except Exception as e:
                logger.error(f"Error in Gurobi lazy callback: {e}", exc_info=True)
                self.do_solver.early_stopping_exception = e
                model.terminate()


class GurobiLC:
    """
    Custom Gurobi callback to generate valid, flow-dependent cuts for the multibatching problem.
    """

    def __init__(self, do_solver: GurobiMultibatchingSolver, callback: Callback):
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0
        self.nb_cuts_added = 0
        self.packing_solver = PackingViaBinPacking(problem=self.do_solver.problem)

    def __call__(self, model: "gurobipy.Model", where: int) -> None:
        if where == gurobipy.GRB.Callback.MIPSOL:
            try:
                if model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ) >= 2.8 * 10**12:
                    return
                # 1. Retrieve the candidate solution from the MILP
                candidate_solution: MultibatchingSolution = (
                    self.do_solver.retrieve_current_solution(
                        get_var_value_for_current_solution=model.cbGetSolution,
                        get_obj_value_for_current_solution=lambda: model.cbGet(
                            gurobipy.GRB.Callback.MIPSOL_OBJ
                        ),
                    )
                )

                # 2. Run the greedy packing subproblem to get the "true" number of trips
                self.packing_solver.init_from_solution(solution=candidate_solution)
                packing_result = self.packing_solver.solve(
                    time_limit_per_link=2,
                    bin_packing_solver=SubBrick(
                        cls=CpSatBinPackSolver,
                        kwargs=dict(
                            modeling=ModelingBinPack.SCHEDULING,
                            parameters_cp=ParametersCp.default_cpsat(),
                        ),
                    ),
                )
                if len(packing_result) == 0:
                    return
                packed_solution: MultibatchingSolution = (
                    packing_result.get_best_solution_fit()[0]
                )
                print(
                    "Solution quality of the greedy : ",
                    packing_result.get_best_solution_fit()[1],
                )
                # 3. Check for violations and add valid, flow-dependent lazy constraints
                milp_trips = defaultdict(int)
                for flow in candidate_solution.list_flows:
                    milp_trips[flow.transport_link] += flow.nb_packing
                greedy_trips = defaultdict(int)
                for flow in packed_solution.list_flows:
                    greedy_trips[flow.transport_link] += flow.nb_packing
                violation_found = False
                for link, milp_trip_count_float in milp_trips.items():
                    milp_trip_count = round(milp_trip_count_float)
                    greedy_trip_count = greedy_trips.get(link, 0)
                    flows_product_in_this_link = {}
                    for flow in candidate_solution.list_flows:
                        if flow.transport_link == link:
                            for p in flow.product_packing:
                                if p not in flows_product_in_this_link:
                                    flows_product_in_this_link[p] = 0
                                flows_product_in_this_link[p] += round(
                                    flow.product_packing[p] * flow.nb_packing
                                )

                    if greedy_trip_count > milp_trip_count:
                        violation_found = True
                        self.nb_cuts_added += 1
                        link_idx = self.do_solver.problem.transport_links_to_index[link]
                        nb_trips_var = self.do_solver.variables["nb_trips"][link_idx]
                        flows_vars = self.do_solver.variables["flows_one_hot"][link_idx]

                        flows_vars_ = [
                            flows_vars[self.do_solver.problem.product_to_index[p]][
                                flows_product_in_this_link[p]
                            ]
                            for p in flows_product_in_this_link
                        ]  # values 1 for each product
                        # for p in flows_product_in_this_link:
                        #     index = self.do_solver.problem.product_to_index[p]
                        #     print("flows product in this link = ", flows_product_in_this_link[p])
                        #     for k in flows_vars[index]:
                        #         print(k, " : ", model.cbGetSolution(flows_vars[index][k]))
                        print([model.cbGetSolution(x) for x in flows_vars_])
                        # len_flows_vars = len(flows_vars_)
                        model.cbLazy(
                            100 * gurobipy.quicksum([1 - x for x in flows_vars_])
                            + nb_trips_var
                            >= greedy_trip_count
                        )
                        flows_vars_ = [
                            gurobipy.quicksum(
                                [
                                    flows_vars[
                                        self.do_solver.problem.product_to_index[p]
                                    ][quantity]
                                    for quantity in range(
                                        flows_product_in_this_link[p],
                                        max(
                                            flows_vars[
                                                self.do_solver.problem.product_to_index[
                                                    p
                                                ]
                                            ]
                                        )
                                        + 1,
                                    )
                                ]
                            )
                            for p in flows_product_in_this_link
                        ]  # values 1 for each product
                        flows_vars_ += [
                            gurobipy.quicksum(
                                [
                                    flows_vars[
                                        self.do_solver.problem.product_to_index[p]
                                    ][quantity]
                                    for quantity in range(
                                        max(
                                            flows_vars[
                                                self.do_solver.problem.product_to_index[
                                                    p
                                                ]
                                            ]
                                        )
                                        + 1
                                    )
                                ]
                            )
                            for p in self.do_solver.problem.products
                            if self.do_solver.problem.product_to_index[p] in flows_vars
                            and p not in flows_product_in_this_link
                        ]
                        # for p in flows_product_in_this_link:
                        #     index = self.do_solver.problem.product_to_index[p]
                        #     print("flows product in this link = ", flows_product_in_this_link[p])
                        #     for k in flows_vars[index]:
                        #         print(k, " : ", model.cbGetSolution(flows_vars[index][k]))
                        # len_flows_vars = len(flows_vars_)
                        model.cbLazy(
                            100 * gurobipy.quicksum([1 - x for x in flows_vars_])
                            + nb_trips_var
                            >= greedy_trip_count
                        )

                        # Generalize to other link with same transport type.
                        other_links = [
                            i_link
                            for i_link in range(
                                self.do_solver.problem.nb_transport_links
                            )
                            if self.do_solver.problem.transport_links[
                                i_link
                            ].transport_type
                            == link.transport_type
                        ]
                        if True:
                            for ol in other_links:
                                flows_vars_ol = self.do_solver.variables[
                                    "flows_one_hot"
                                ][ol]
                                flows_vars_ol_ = [
                                    flows_vars_ol[
                                        self.do_solver.problem.product_to_index[p]
                                    ][flows_product_in_this_link[p]]
                                    for p in flows_product_in_this_link
                                ]  # values 1 for each product
                                # len_flows_vars = len(flows_vars_)

                                model.cbLazy(
                                    100
                                    * gurobipy.quicksum([1 - x for x in flows_vars_ol_])
                                    + self.do_solver.variables["nb_trips"][ol]
                                    >= greedy_trip_count
                                )

                                flows_vars_ol_ = [
                                    gurobipy.quicksum(
                                        [
                                            flows_vars_ol[
                                                self.do_solver.problem.product_to_index[
                                                    p
                                                ]
                                            ][quantity]
                                            for quantity in range(
                                                flows_product_in_this_link[p],
                                                max(
                                                    flows_vars_ol[
                                                        self.do_solver.problem.product_to_index[
                                                            p
                                                        ]
                                                    ]
                                                )
                                                + 1,
                                            )
                                        ]
                                    )
                                    for p in flows_product_in_this_link
                                ]  # values 1 for each product
                                flows_vars_ol_ += [
                                    gurobipy.quicksum(
                                        [
                                            flows_vars_ol[
                                                self.do_solver.problem.product_to_index[
                                                    p
                                                ]
                                            ][quantity]
                                            for quantity in range(
                                                max(
                                                    flows_vars_ol[
                                                        self.do_solver.problem.product_to_index[
                                                            p
                                                        ]
                                                    ]
                                                )
                                                + 1
                                            )
                                        ]
                                    )
                                    for p in self.do_solver.problem.products
                                    if self.do_solver.problem.product_to_index[p]
                                    in flows_vars_ol
                                    and p not in flows_product_in_this_link
                                ]
                                model.cbLazy(
                                    100
                                    * gurobipy.quicksum([1 - x for x in flows_vars_ol_])
                                    + self.do_solver.variables["nb_trips"][ol]
                                    >= greedy_trip_count
                                )
                        model.cbSetSolution(nb_trips_var, greedy_trip_count)
                        logger.info(
                            f"CUT #{self.nb_cuts_added} on link {link.id}: "
                            + f"MILP trips={milp_trip_count}, Greedy trips={greedy_trip_count},"
                            + f"Adding lazy cut."
                        )
                    else:
                        link_idx = self.do_solver.problem.transport_links_to_index[link]
                        nb_trips_var = self.do_solver.variables["nb_trips"][link_idx]
                        model.cbSetSolution(nb_trips_var, milp_trip_count)
                    flows_vars = self.do_solver.variables["flows"][link_idx]
                    for p in flows_vars:
                        model.cbSetSolution(
                            flows_vars[p], model.cbGetSolution(flows_vars[p])
                        )
                    flows_vars_one_hot = self.do_solver.variables["flows_one_hot"][
                        link_idx
                    ]
                    for p in flows_vars_one_hot:
                        for k in flows_vars_one_hot[p]:
                            model.cbSetSolution(
                                flows_vars_one_hot[p][k],
                                model.cbGetSolution(flows_vars_one_hot[p][k]),
                            )
                for index_link in range(self.do_solver.problem.nb_transport_links):
                    if (
                        self.do_solver.problem.transport_links[index_link]
                        not in milp_trips
                    ):
                        nb_trips_var = self.do_solver.variables["nb_trips"][index_link]
                        model.cbSetSolution(nb_trips_var, 0)
                        flows_vars = self.do_solver.variables["flows"][index_link]
                        for p in flows_vars:
                            model.cbSetSolution(flows_vars[p], 0)
                        flows_vars_one_hot = self.do_solver.variables["flows_one_hot"][
                            index_link
                        ]
                        for p in flows_vars_one_hot:
                            for k in flows_vars_one_hot[p]:
                                if k == 0:
                                    model.cbSetSolution(flows_vars_one_hot[p][k], 1)
                                else:
                                    model.cbSetSolution(flows_vars_one_hot[p][k], 0)

                model.cbUseSolution()
                if not violation_found:
                    fit = self.do_solver.aggreg_from_sol(packed_solution)
                    self.res.append((packed_solution, fit))
                    self.nb_solutions += 1
                    stopping = self.callback.on_step_end(
                        step=self.nb_solutions, res=self.res, solver=self.do_solver
                    )
                    if stopping:
                        model.terminate()
            except Exception as e:
                logger.error(f"Error in Gurobi lazy callback: {e}", exc_info=True)
                self.do_solver.early_stopping_exception = e
                model.terminate()


from discrete_optimization.multibatching.solvers.packing_subproblem import (
    CpSatBinPackSolver,
    PackingViaBinPacking,
    SubBrick,
)


class GurobiLC_BinaryExpansion:
    def __init__(self, do_solver: GurobiMultibatchingSolver, callback: Callback):
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_cuts_added = 0
        self.packing_solver = PackingViaBinPacking(problem=self.do_solver.problem)

    def __call__(self, model: "gurobipy.Model", where: int) -> None:
        if where == gurobipy.GRB.Callback.MIPSOL:
            # 1. Retrieve candidate solution
            if model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ) >= 2.8 * 10**12:
                return
            candidate_sol: MultibatchingSolution = (
                self.do_solver.retrieve_current_solution(
                    get_var_value_for_current_solution=model.cbGetSolution,
                    get_obj_value_for_current_solution=lambda: model.cbGet(
                        gurobipy.GRB.Callback.MIPSOL_OBJ
                    ),
                )
            )
            # 2. Run packing subproblem
            self.packing_solver.init_from_solution(solution=candidate_sol)
            packing_result = self.packing_solver.solve(
                time_limit_per_link=1,
                bin_packing_solver=SubBrick(
                    cls=CpSatBinPackSolver,
                    kwargs=dict(
                        modeling=ModelingBinPack.SCHEDULING,
                        parameters_cp=ParametersCp.default_cpsat(),
                    ),
                ),
            )
            if not packing_result:
                return
            packed_sol: MultibatchingSolution = packing_result.get_best_solution_fit()[
                0
            ]
            # 3. Check and generate cuts
            violation_found = False
            for link_idx, tl in enumerate(self.do_solver.problem.transport_links):
                # Count trips from MILP vs Greedy
                milp_trips = sum(
                    f.nb_packing
                    for f in candidate_sol.list_flows
                    if f.transport_link == tl
                )
                greedy_trips = sum(
                    f.nb_packing
                    for f in packed_sol.list_flows
                    if f.transport_link == tl
                )

                if greedy_trips > round(milp_trips):
                    print(greedy_trips, " vs ", milp_trips)
                    violation_found = True
                    self.nb_cuts_added += 1
                    # Get bits for all products on this link
                    nb_trips_var = self.do_solver.variables["nb_trips"][link_idx]
                    bits_to_fix = []
                    for p_idx, p in enumerate(self.do_solver.problem.products):
                        if (
                            p_idx
                            not in self.do_solver.variables["flows_binary_bits"][
                                link_idx
                            ]
                        ):
                            continue

                        current_flow = round(
                            model.cbGetSolution(
                                self.do_solver.variables["flows"][link_idx][p_idx]
                            )
                        )
                        bits = self.do_solver.variables["flows_binary_bits"][link_idx][
                            p_idx
                        ]

                        # Identify which bits are 1 and which are 0 in the current solution
                        for b, bit_var in enumerate(bits):
                            is_one = (current_flow >> b) & 1
                            if is_one:
                                bits_to_fix.append(
                                    1 - bit_var
                                )  # Penalty if it flips to 0
                            else:
                                bits_to_fix.append(bit_var)  # Penalty if it flips to 1

                    # No-Good Cut: if the flow combination remains the same, nb_trips must increase
                    # M * sum(flipped_bits) + nb_trips >= greedy_trips
                    M = 100
                    model.cbLazy(
                        M * gurobipy.quicksum(bits_to_fix) + nb_trips_var
                        >= greedy_trips
                    )

                    other_link = [
                        i_link
                        for i_link in range(self.do_solver.problem.nb_transport_links)
                        if self.do_solver.problem.transport_links[i_link].transport_type
                        == tl.transport_type
                        and i_link != link_idx
                    ]
                    for index_link in other_link:
                        nb_trips_var = self.do_solver.variables["nb_trips"][index_link]
                        bits_to_fix = []
                        for p_idx, p in enumerate(self.do_solver.problem.products):
                            if (
                                p_idx
                                not in self.do_solver.variables["flows_binary_bits"][
                                    index_link
                                ]
                            ):
                                continue

                            current_flow = round(
                                model.cbGetSolution(
                                    self.do_solver.variables["flows"][link_idx][p_idx]
                                )
                            )
                            bits = self.do_solver.variables["flows_binary_bits"][
                                index_link
                            ][p_idx]

                            # Identify which bits are 1 and which are 0 in the current solution
                            for b, bit_var in enumerate(bits):
                                is_one = (current_flow >> b) & 1
                                if is_one:
                                    bits_to_fix.append(
                                        1 - bit_var
                                    )  # Penalty if it flips to 0
                                else:
                                    bits_to_fix.append(
                                        bit_var
                                    )  # Penalty if it flips to 1
                        # No-Good Cut: if the flow combination remains the same, nb_trips must increase
                        # M * sum(flipped_bits) + nb_trips >= greedy_trips
                        M = 100
                        model.cbLazy(
                            M * gurobipy.quicksum(bits_to_fix) + nb_trips_var
                            >= greedy_trips
                        )

            # 4. Success logic
            if not violation_found:
                fit = self.do_solver.aggreg_from_sol(packed_sol)
                self.res.append((packed_sol, fit))
                self.callback.on_step_end(
                    step=len(self.res), res=self.res, solver=self.do_solver
                )
