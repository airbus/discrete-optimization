#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections import Counter, defaultdict
from typing import Any, Callable, Dict

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
    MilpSolver,
    OrtoolsMathOptMilpSolver,
    ParametersMilp,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
)
from discrete_optimization.multibatching.solvers import MultibatchingSolver
from discrete_optimization.multibatching.solvers.solver_utils import (
    precompute_valid_links,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True

logger = logging.getLogger(__name__)


class _BaseLpMultibatchingSolver(MilpSolver, MultibatchingSolver):
    """Base class for Multibatching LP solvers.

    This solver supports two main modeling variants:
    1.  **Standard Flow Model**: Allows multiple different products to be packed
        into the same transport (multi-batching).
    2.  **Single Batching Flow Model**: A more constrained version where each transport
        can only carry a single type of product.

    Attributes:
        problem (MultibatchingProblem): The multibatching problem instance to solve.
        variables (Dict): A dictionary to store variables created for the model.
        single_batching (bool): Flag indicating which model variant to use.
        scaling_factor (int): A factor to scale floating-point costs into integers for the model.
    """

    problem: MultibatchingProblem
    hyperparameters = [
        CategoricalHyperparameter(
            name="add_lb_constraint_nb_trips", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="restrict_to_shortest_paths", choices=[True, False], default=False
        ),
        FloatHyperparameter(
            name="shortest_path_tolerance",
            low=0,
            high=30,
            default=0.2,
            depends_on=[("restrict_to_shortest_paths", True)],
        ),
    ]

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables: Dict[str, Any] = {}
        self.single_batching: bool = False
        self.scaling_factor = kwargs.get("scaling_factor", 1)

    def init_model(self, single_batching: bool = False, **kwargs: Any) -> None:
        """
        Initializes the model by dispatching to the appropriate
        model-building method based on the `single_batching` flag.
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.single_batching = single_batching
        if self.single_batching:
            self._init_model_flows_single_batching(**kwargs)
        else:
            self._init_model_flows_multi_batching(**kwargs)

    def _init_model_flows_multi_batching(self, **kwargs: Any) -> None:
        """Builds the standard flow model where multiple products can share a transport."""
        print(kwargs)
        use_shortest_path = kwargs.get("restrict_to_shortest_paths", False)
        sp_tolerance = kwargs.get(
            "shortest_path_tolerance", 0.2
        )  # 0.0 = Strict, 0.2 = +20% length allowed

        # Option B: No incoming flow at producers
        no_incoming_at_source = kwargs.get("prevent_incoming_at_source", False)

        valid_links_map = None
        if use_shortest_path:
            logger.info(
                f"Computing valid links (Shortest Path Heuristic, tol={sp_tolerance})..."
            )
            valid_links_map = precompute_valid_links(
                self.problem, tolerance=sp_tolerance
            )

        logger.info("Initializing model (Multi-Batching Flow)...")
        self.model = self.create_empty_model("multibatching-flow")

        # --- Pre-computation ---
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }

        # --- Variables ---
        nb_trips = {}
        for i in range(self.problem.nb_transport_links):
            nb_trips[i] = self.add_integer_variable(name=f"nb_trips_{i}")

        flows = {}
        for i, tl in enumerate(self.problem.transport_links):
            flows[i] = {}
            for p_idx, p in enumerate(self.problem.products):
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
                ):
                    flows[i][p_idx] = self.add_integer_variable(
                        name=f"flow_{i}_{p_idx}", ub=supply_per_product[p]
                    )

        # --- Constraints ---
        # Capacity Constraint
        for i, tl in enumerate(self.problem.transport_links):
            total_volume = self.construct_linear_sum(
                flows[i][p_idx] * p.size
                for p_idx, p in enumerate(self.problem.products)
                if p_idx in flows[i]
            )
            self.add_linear_constraint(
                total_volume <= nb_trips[i] * tl.transport_type.capacity
            )

        # Flow Conservation
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = self.construct_linear_sum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l2 == loc and p_idx in flows[i]
                )
                out_flow = self.construct_linear_sum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l1 == loc and p_idx in flows[i]
                )
                self.add_linear_constraint(
                    in_flow + loc.net_supply.get(p, 0) == out_flow
                )

        if kwargs["add_lb_constraint_nb_trips"]:
            self.add_advanced_capacity_constraints(
                flows_variables=flows,
                nb_trips_per_link=nb_trips,
                max_k=10,
            )

        # --- Objective ---
        self._set_objective(nb_trips, flows)

        self.variables = {"nb_trips": nb_trips, "flows": flows}

    def _init_model_flows_single_batching(self, **kwargs: Any) -> None:
        """Builds the constrained flow model where only one product type is allowed per transport."""
        logger.info("Initializing model (Single-Batching Flow)...")
        self.model = self.create_empty_model("multibatching-single-flow")

        # --- Pre-computation ---
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }

        # --- Variables ---
        nb_trips = {}
        for i in range(self.problem.nb_transport_links):
            nb_trips[i] = self.add_integer_variable(name=f"nb_trips_{i}")

        flows = {}
        for i, tl in enumerate(self.problem.transport_links):
            flows[i] = {}
            for p_idx, p in enumerate(self.problem.products):
                if (
                    tl.transport_type in p.valid_transports
                    and tl.transport_type.capacity > p.size
                ):
                    flows[i][p_idx] = self.add_integer_variable(
                        name=f"flow_{i}_{p_idx}", ub=supply_per_product[p]
                    )

        # Trips dedicated to a specific product on a specific link
        nb_trips_per_product = {}
        for i in range(self.problem.nb_transport_links):
            for p_idx in flows[i]:
                nb_trips_per_product[(i, p_idx)] = self.add_integer_variable(
                    name=f"nb_trips_{i}_{p_idx}"
                )

        # --- Constraints ---
        for i, tl in enumerate(self.problem.transport_links):
            # Total trips on a link is the sum of product-dedicated trips
            self.add_linear_constraint(
                self.construct_linear_sum(
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
                        self.add_linear_constraint(
                            flows[i][p_idx]
                            <= nb_trips_per_product[(i, p_idx)] * products_per_trip
                        )
                    else:  # Cannot transport this product
                        self.add_linear_constraint(flows[i][p_idx] == 0)

        # Flow conservation (same as multi-batching)
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = self.construct_linear_sum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l2 == loc and p_idx in flows[i]
                )
                out_flow = self.construct_linear_sum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l1 == loc and p_idx in flows[i]
                )
                self.add_linear_constraint(
                    in_flow + loc.net_supply.get(p, 0) == out_flow
                )

        # --- Objective ---
        self._set_objective(nb_trips, flows)

        self.variables = {
            "nb_trips": nb_trips,
            "flows": flows,
            "nb_trips_per_product": nb_trips_per_product,
        }

    def add_advanced_capacity_constraints(
        self, flows_variables, nb_trips_per_link, max_k=30
    ):
        """
        Adds generalized lower bound constraints on the number of trips.
        For each k in [2..max_k], we consider items with size > Capacity/k.
        Since at most k-1 such items fit in one vehicle, we add the cut:
            (k-1) * NbTrips >= Sum(Flows of these items)
        """
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
                    rhs = self.construct_linear_sum(relevant_flows)
                    self.add_linear_constraint(lhs >= rhs)

    def _set_objective(self, nb_trips, flows):
        """Helper function to build and set the objective expression."""
        transport_cost = self.construct_linear_sum(
            nb_trips[i]
            * int(tl.distance * tl.transport_type.cost * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        emission_cost = self.construct_linear_sum(
            nb_trips[i]
            * int(tl.distance * tl.transport_type.emissions * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        self.set_model_objective(transport_cost + emission_cost, minimize=True)

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        """
        Converts a solution into a `MultibatchingSolution`.
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
    ) -> Dict[Any, float]:
        """Converts a `MultibatchingSolution` to a dictionary of variable values for warm start."""
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

        # Map aggregated values to variables
        for i, var in self.variables["nb_trips"].items():
            variable_values[var] = agg_trips.get(i, 0)
        for i, flow_map in self.variables["flows"].items():
            for p_idx, var in flow_map.items():
                variable_values[var] = agg_flows.get(i, {}).get(p_idx, 0)

        if self.single_batching:
            for (i, p_idx), var in self.variables["nb_trips_per_product"].items():
                variable_values[var] = agg_trips_per_product.get((i, p_idx), 0)

        return variable_values


class GurobiMultibatchingSolver(GurobiMilpSolver, _BaseLpMultibatchingSolver):
    """Gurobi solver for the Multibatching problem, based on a flow formulation."""

    hyperparameters = _BaseLpMultibatchingSolver.hyperparameters

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict["gurobipy.Var", float]:
        return _BaseLpMultibatchingSolver.convert_to_variable_values(self, solution)

    def init_model(self, single_batching: bool = False, **kwargs: Any) -> None:
        _BaseLpMultibatchingSolver.init_model(self, single_batching, **kwargs)
        self.model.update()


class MathOptMultibatchingSolver(OrtoolsMathOptMilpSolver, _BaseLpMultibatchingSolver):
    """MathOpt solver for the Multibatching problem, based on a flow formulation."""

    hyperparameters = (
        OrtoolsMathOptMilpSolver.hyperparameters
        + _BaseLpMultibatchingSolver.hyperparameters
    )

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict["gurobipy.Var", float]:
        return _BaseLpMultibatchingSolver.convert_to_variable_values(self, solution)


# Unit Flow Solver
class _BaseLpMultibatchingSolverUnitFlow(MilpSolver, MultibatchingSolver):
    """Base class for Multibatching LP solvers using a detailed "unit flow" formulation.

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

    def init_model(self, max_trips_per_link: int = 15, **kwargs: Any) -> None:
        """
        Builds the Gurobi model based on a detailed, per-trip ("unit flow") formulation.

        Args:
            max_trips_per_link (int): A hard upper bound on the number of trips that can be
                                      used on any single transport link. This is necessary
                                      to define the variable space.
        """
        logger.info("Initializing model (Unit Flow)...")
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
            valid_links_map = precompute_valid_links(
                self.problem, tolerance=sp_tolerance
            )

        self.model = self.create_empty_model("multibatching-unit-flow")
        model = self.model
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
        # Create binary variables for whether a trip is used
        used_trips = {}
        for i in range(self.problem.nb_transport_links):
            used_trips[i] = {}
            for j in range(max_nb_trip_per_transport_link[i]):
                used_trips[i][j] = self.add_binary_variable(name=f"used_{i}_{j}")

        # `trip_contents[link_idx][trip_idx][prod_idx]` = quantity of a product on a specific trip
        trip_contents = {}
        for i, tl in enumerate(self.problem.transport_links):
            trip_contents[i] = {}
            for j in range(max_nb_trip_per_transport_link[i]):
                trip_contents[i][j] = {}
                for p_idx, p in enumerate(self.problem.products):
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
                    ):
                        ub_val = (
                            int(tl.transport_type.capacity / p.size)
                            if p.size > 0
                            else 0
                        )
                        trip_contents[i][j][p_idx] = self.add_integer_variable(
                            name=f"cont_{i}_{j}_{p_idx}", ub=ub_val
                        )

        # --- Constraints ---
        for i, tl in enumerate(self.problem.transport_links):
            for j in range(max_nb_trip_per_transport_link[i]):
                # Capacity constraint for each individual trip
                trip_volume = self.construct_linear_sum(
                    trip_contents[i][j][p_idx] * p.size
                    for p_idx, p in enumerate(self.problem.products)
                    if p_idx in trip_contents[i][j]
                )
                self.add_linear_constraint(trip_volume <= tl.transport_type.capacity)

                # Link `used_trips` to trip contents using big-M formulation
                # IF used_trips[i][j] is 1, THEN total_items >= 1
                # IF used_trips[i][j] is 0, THEN total_items == 0
                total_items_in_trip = self.construct_linear_sum(
                    trip_contents[i][j][p_idx] for p_idx in trip_contents[i][j]
                )

                # Big-M upper bound: compute maximum possible items in this trip
                M = sum(
                    int(tl.transport_type.capacity / p.size) if p.size > 0 else 0
                    for p_idx, p in enumerate(self.problem.products)
                    if p_idx in trip_contents[i][j]
                )
                if M == 0:
                    M = 1  # Safety for empty trips

                # Constraint 1: total_items >= used_trips (enforces items >= 1 when used=1)
                self.add_linear_constraint(total_items_in_trip >= used_trips[i][j])

                # Constraint 2: total_items <= M * used_trips (enforces items = 0 when used=0)
                self.add_linear_constraint(total_items_in_trip <= M * used_trips[i][j])

        # Aggregate expressions for total flow, linking trip contents to flow conservation
        total_flows = {}
        for i in range(self.problem.nb_transport_links):
            total_flows[i] = {}
            for p_idx, p in enumerate(self.problem.products):
                # Check if this product can be on any trip on this link
                if any(
                    p_idx in trip_contents[i][j]
                    for j in range(max_nb_trip_per_transport_link[i])
                ):
                    total_flows[i][p_idx] = self.construct_linear_sum(
                        trip_contents[i][j].get(p_idx, 0)
                        for j in range(max_nb_trip_per_transport_link[i])
                        if p_idx in trip_contents[i][j]
                    )

        # Flow conservation constraints (using the aggregated total_flows)
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = self.construct_linear_sum(
                    total_flows[i].get(p_idx, 0)
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l2 == loc and p_idx in total_flows[i]
                )
                out_flow = self.construct_linear_sum(
                    total_flows[i].get(p_idx, 0)
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l1 == loc and p_idx in total_flows[i]
                )
                self.add_linear_constraint(
                    in_flow + loc.net_supply.get(p, 0) == out_flow
                )

        # --- Objective Function ---
        nb_trips_per_link = {}
        for i in range(self.problem.nb_transport_links):
            nb_trips_per_link[i] = self.construct_linear_sum(
                used_trips[i][j] for j in range(max_nb_trip_per_transport_link[i])
            )

        transport_cost = self.construct_linear_sum(
            nb_trips_per_link[i]
            * int(tl.distance * tl.transport_type.cost * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        emission_cost = self.construct_linear_sum(
            nb_trips_per_link[i]
            * int(tl.distance * tl.transport_type.emissions * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        self.set_model_objective(transport_cost + emission_cost, minimize=True)

        self.variables = {
            "used_trips": used_trips,
            "trip_contents": trip_contents,
        }

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
    ) -> Dict[Any, float]:
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


class GurobiMultibatchingSolverUnitFlow(
    GurobiMilpSolver, _BaseLpMultibatchingSolverUnitFlow
):
    """Gurobi solver for the Multibatching problem using unit flow formulation."""

    def convert_to_variable_values(self, solution: MultibatchingSolution):
        return _BaseLpMultibatchingSolverUnitFlow.convert_to_variable_values(
            self, solution
        )

    def init_model(self, max_trips_per_link: int = 15, **kwargs: Any) -> None:
        _BaseLpMultibatchingSolverUnitFlow.init_model(
            self, max_trips_per_link, **kwargs
        )
        self.model.update()


class MathOptMultibatchingSolverUnitFlow(
    OrtoolsMathOptMilpSolver, _BaseLpMultibatchingSolverUnitFlow
):
    """MathOpt solver for the Multibatching problem using unit flow formulation."""

    def convert_to_variable_values(self, solution: MultibatchingSolution):
        return _BaseLpMultibatchingSolverUnitFlow.convert_to_variable_values(
            self, solution
        )


# Lazy constraint variants (Gurobi-specific)
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


# Import lazy callback helpers
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.multibatching.solvers.packing_subproblem import (
    PackingViaBinPacking,
    SubBrick,
)


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
                logger.debug(
                    f"fit packing {self.do_solver.aggreg_from_sol(packing_result[-1][0])}"
                )
                logger.debug(
                    f"sat packing {self.do_solver.problem.satisfy(packing_result[-1][0])}"
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
