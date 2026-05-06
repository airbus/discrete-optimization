#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import math
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Iterable

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import (
    WarmstartMixin,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import (
    OrtoolsCpSatSolver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
    Product,
    TransportType,
)
from discrete_optimization.multibatching.solvers.solver_utils import (
    precompute_valid_links,
)

logger = logging.getLogger(__name__)


class ModelingMultiBatch(Enum):
    FLOW = 0
    UNIT_FLOW = 1


class CpsatMultibatchingSolver(OrtoolsCpSatSolver, WarmstartMixin):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingMultiBatch, default=ModelingMultiBatch.FLOW
        ),
        CategoricalHyperparameter(
            name="detailed_capacity_constraint",
            choices=[True, False],
            default=False,
            depends_on=[("modeling", [ModelingMultiBatch.FLOW])],
        ),
        CategoricalHyperparameter(
            name="add_lb_constraint_nb_trips",
            choices=[True, False],
            default=False,
            depends_on=[("modeling", [ModelingMultiBatch.FLOW])],
        ),
        CategoricalHyperparameter(
            name="restrict_to_shortest_paths", choices=[True, False], default=False
        ),
        FloatHyperparameter(
            name="shortest_path_tolerance",
            low=0.0,
            high=30,
            depends_on=[("restrict_to_shortest_paths", True)],
        ),
        CategoricalHyperparameter(
            name="prevent_incoming_at_source", choices=[True, False]
        ),
    ]
    problem: MultibatchingProblem

    def __init__(
        self,
        problem: MultibatchingProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.modeling: ModelingMultiBatch = None
        self.single_batching: bool = None
        self.verbose_logs = kwargs.get("verbose", 0)
        self.scaling_factor = kwargs.get("scaling_factor", 1)

    def init_model(self, single_batching: bool = False, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        modeling = args["modeling"]
        if modeling == ModelingMultiBatch.FLOW:
            if single_batching:
                self.init_model_flows_single_batching(**args)
            else:
                self.init_model_flows(**args)
        if modeling == ModelingMultiBatch.UNIT_FLOW:
            self.init_model_detailed_trips(**args)
        self.modeling = modeling
        self.single_batching = single_batching

    def init_model_flows(self, **args: Any) -> None:
        use_shortest_path = args["restrict_to_shortest_paths"]
        sp_tolerance = args["shortest_path_tolerance"]
        no_incoming_at_source = args["prevent_incoming_at_source"]
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }
        demand_per_product = {
            p: self.problem.get_total_demand(p) for p in self.problem.products
        }
        total_size_per_product = {
            p: p.size * supply_per_product[p] for p in self.problem.products
        }
        total_size_in_problem = sum(total_size_per_product.values())
        max_nb_trip = max(
            1,
            int(
                math.ceil(
                    total_size_in_problem
                    / min(
                        [
                            tt.capacity
                            for tt in self.problem.transport_types
                            if tt.capacity > 0
                        ]
                    )
                )
            ),
        )
        transport_type_to_set_products: dict[TransportType, set[Product]] = {
            tt: set() for tt in self.problem.transport_types
        }
        for p in self.problem.products:
            for tt in p.valid_transports:
                transport_type_to_set_products[tt].add(p)
        max_nb_trips_per_transport_type = {
            tt: sum(
                [total_size_per_product[p] for p in transport_type_to_set_products[tt]]
            )
            / max(1, tt.capacity)
            for tt in transport_type_to_set_products
        }
        valid_links_map = None
        if use_shortest_path:
            logger.info(
                f"Computing valid links (Shortest Path Heuristic, tol={sp_tolerance})..."
            )
            valid_links_map = precompute_valid_links(
                self.problem, tolerance=sp_tolerance
            )

        model = cp_model.CpModel()
        transport_links = self.problem.transport_links
        flows_variables = {}
        nb_trips_per_link = {}
        keys_inflow_location_product = dict()
        keys_outflow_location_product = dict()
        for index_tl in range(self.problem.nb_transport_links):
            flows_variables[index_tl] = {}
            tl = transport_links[index_tl]
            loc1 = tl.location_l1
            loc2 = tl.location_l2
            if loc1.id not in keys_outflow_location_product:
                keys_outflow_location_product[loc1.id] = defaultdict(lambda: set())
            if loc2.id not in keys_inflow_location_product:
                keys_inflow_location_product[loc2.id] = defaultdict(lambda: set())
            tr: TransportType = tl.transport_type
            nb_trips_per_link[index_tl] = model.NewIntVar(
                lb=0, ub=max_nb_trip, name=f"num_trips_{index_tl}"
            )
            list_var_weight = []
            for index_product in range(self.problem.nb_products):
                product: Product = self.problem.products[index_product]
                # Heuristic A: Skip if link is not on a shortest path
                if use_shortest_path and valid_links_map:
                    if index_tl not in valid_links_map.get(product.id, set()):
                        continue
                # Heuristic B: Skip if destination is a Producer (NetSupply > 0)
                # "On node that produce product, you should not have incoming flows"
                if no_incoming_at_source:
                    if loc2.net_supply.get(product, 0) > 0:
                        continue
                max_flow_product = max(
                    supply_per_product[product], demand_per_product[product]
                )
                # Create flow of product when the current transport type is valid for this product.
                if tr in product.valid_transports:
                    if tr.capacity > product.size:
                        flows_variables[index_tl][index_product] = model.NewIntVar(
                            lb=0,
                            ub=max_flow_product,
                            name=f"flow_{index_tl}_{index_product}",
                        )
                        keys_outflow_location_product[loc1.id][product.id].add(
                            (index_tl, index_product)
                        )
                        keys_inflow_location_product[loc2.id][product.id].add(
                            (index_tl, index_product)
                        )
                        list_var_weight.append(
                            (
                                flows_variables[index_tl][index_product],
                                int(product.size),
                            )
                        )
            # Global capacity constraint
            model.Add(
                cp_model.LinearExpr.weighted_sum(
                    [x[0] for x in list_var_weight], [x[1] for x in list_var_weight]
                )
                <= nb_trips_per_link[index_tl] * tl.transport_type.capacity
            )
        if args["detailed_capacity_constraint"]:
            nb_trips_min_for_product_flows = {
                index_product: self.compute_nb_trips_min(
                    self.problem.products[index_product],
                    demand_of_product=demand_per_product[
                        self.problem.products[index_product]
                    ],
                )
                for index_product in range(self.problem.nb_products)
            }
            for index_tl in flows_variables:
                transport_link = self.problem.transport_links[index_tl]
                transport_type = transport_link.transport_type
                for index_product in flows_variables[index_tl]:
                    nb_trips_min = nb_trips_min_for_product_flows[index_product][
                        transport_type
                    ]
                    # binary_indicators = [model.NewBoolVar(f'flow_{index_tl}_{index_product}_equal_{k}')
                    #                      for k in range(len(nb_trips_min))]
                    # model.add_map_domain(flows_variables[index_tl][index_product],
                    #                     binary_indicators)
                    # for j in range(1, len(binary_indicators)):
                    #     (model.add(nb_trips_per_link[index_tl] >= nb_trips_min[j])
                    #     .OnlyEnforceIf(binary_indicators[j]))
                    # For a given number of product flow NB, we force nb_trips_per_link to be >= nb_trips_min[NB]
                    model.AddForbiddenAssignments(
                        [
                            flows_variables[index_tl][index_product],
                            nb_trips_per_link[index_tl],
                        ],
                        [
                            (j, nb_min)
                            for j in range(1, min(5, len(nb_trips_min)))
                            for nb_min in range(nb_trips_min[j])
                        ],
                    )

        if args["add_lb_constraint_nb_trips"]:
            self.add_advanced_capacity_constraints(
                model=model,
                flows_variables=flows_variables,
                nb_trips_per_link=nb_trips_per_link,
                solver_type="cpsat",
            )
            self.add_advanced_capacity_constraints2(
                model=model,
                flows_variables=flows_variables,
                nb_trips_per_link=nb_trips_per_link,
                max_k=5,
                solver_type="cpsat",
            )
            # self.add_global_flow_limit_constraints(model, flows_variables, "cpsat", 3)
        # Flows constraint
        for location in self.problem.locations:
            loc_id = location.id
            index_products = range(len(self.problem.products))
            for index_product in index_products:
                product = self.problem.products[index_product]
                net_supply = location.net_supply.get(product, 0)
                # Linear expressions (not constraint)
                flow_in = cp_model.LinearExpr.Sum(
                    [
                        flows_variables[x[0]][x[1]]
                        for x in keys_inflow_location_product.get(loc_id, {}).get(
                            product.id, set()
                        )
                    ]
                )
                # flow_in = sum([flows_variables[x[0]][x[1]]
                #               for x in keys_inflow_location_product.get(loc_id, {}).get(product.id, set())])
                flow_out = cp_model.LinearExpr.Sum(
                    [
                        flows_variables[x[0]][x[1]]
                        for x in keys_outflow_location_product.get(loc_id, {}).get(
                            product.id, set()
                        )
                    ]
                )
                model.Add(flow_in + net_supply - flow_out == 0)
        transport_cost = self.scaling_factor * sum(
            [
                nb_trips_per_link[index_tl]
                * int(
                    transport_links[index_tl].distance
                    * transport_links[index_tl].transport_type.cost
                )
                for index_tl in range(self.problem.nb_transport_links)
            ]
        )
        emission_cost = self.scaling_factor * sum(
            [
                nb_trips_per_link[index_tl]
                * int(
                    transport_links[index_tl].distance
                    * transport_links[index_tl].transport_type.emissions
                )
                for index_tl in range(self.problem.nb_transport_links)
            ]
        )
        self.variables["total_obj"] = transport_cost + emission_cost
        model.Minimize(self.variables["total_obj"])
        self.variables["objs"] = {
            "transport": transport_cost,
            "emission": emission_cost,
        }
        self.cp_model = model
        self.variables["flows"] = flows_variables
        self.variables["nb_trips"] = nb_trips_per_link

    def init_model_flows_single_batching(self, **args: Any) -> None:
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }
        demand_per_product = {
            p: self.problem.get_total_demand(p) for p in self.problem.products
        }
        total_size_per_product = {
            p: p.size * supply_per_product[p] for p in self.problem.products
        }
        total_size_in_problem = sum(total_size_per_product.values())
        max_nb_trip = max(
            1,
            int(
                total_size_in_problem
                / min(
                    [
                        tt.capacity
                        for tt in self.problem.transport_types
                        if tt.capacity > 0
                    ]
                )
            ),
        )
        model = cp_model.CpModel()
        self.cp_model = model
        transport_links = self.problem.transport_links
        flows_variables = {}
        # flows_nz_variables = {}
        nb_trips_per_link = {}
        nb_trips_per_link_per_product = {}
        keys_inflow_location_product = dict()
        keys_outflow_location_product = dict()

        for index_tl in range(self.problem.nb_transport_links):
            flows_variables[index_tl] = {}
            # flows_nz_variables[index_tl] = {}
            tl = transport_links[index_tl]
            loc1 = tl.location_l1
            loc2 = tl.location_l2
            if loc1.id not in keys_outflow_location_product:
                keys_outflow_location_product[loc1.id] = defaultdict(lambda: set())
            if loc2.id not in keys_inflow_location_product:
                keys_inflow_location_product[loc2.id] = defaultdict(lambda: set())
            tr: TransportType = tl.transport_type
            nb_trips_per_link[index_tl] = model.NewIntVar(
                lb=0, ub=max_nb_trip, name=f"num_trips_{index_tl}"
            )
            list_var_weight = []
            keys_nb_trips_per_link_per_product = []
            # for index_product in range(self.problem.nb_products):
            for index_product in range(self.problem.nb_products):
                product = self.problem.products[index_product]
                max_flow_product = max(
                    supply_per_product[product], demand_per_product[product]
                )
                if tr in product.valid_transports:
                    if tr.capacity > product.size:
                        flows_variables[index_tl][index_product] = model.NewIntVar(
                            lb=0,
                            ub=max_flow_product,
                            name=f"flow_{index_tl}_{index_product}",
                        )
                        # flows_nz_variables[index_tl][index_product] = model.NewBoolVar(name=f"nz_flow_{index_tl}_"
                        #                                                                     f"{index_product}")
                        nb_product_per_transport = int(tr.capacity // product.size)
                        capa_used_per_transport = (
                            nb_product_per_transport * product.size
                        )
                        nb_trips_max = int(
                            math.ceil(max_flow_product / nb_product_per_transport)
                        )

                        nb_trips_per_link_per_product[(index_tl, index_product)] = (
                            model.NewIntVar(
                                lb=0,
                                ub=nb_trips_max,
                                name=f"trips_{index_tl}_{index_product}",
                            )
                        )
                        keys_nb_trips_per_link_per_product.append(
                            (index_tl, index_product)
                        )
                        self.cp_model.add(
                            nb_trips_per_link_per_product[(index_tl, index_product)]
                            * nb_product_per_transport
                            >= flows_variables[index_tl][index_product]
                        )

                        keys_outflow_location_product[loc1.id][product.id].add(
                            (index_tl, index_product)
                        )
                        keys_inflow_location_product[loc2.id][product.id].add(
                            (index_tl, index_product)
                        )
                        list_var_weight.append(
                            (
                                flows_variables[index_tl][index_product],
                                int(product.size),
                            )
                        )
            model.Add(
                sum(
                    [
                        nb_trips_per_link_per_product[x]
                        for x in keys_nb_trips_per_link_per_product
                    ]
                )
                == nb_trips_per_link[index_tl]
            )
            # Global capacity constraint
            model.Add(
                cp_model.LinearExpr.weighted_sum(
                    [x[0] for x in list_var_weight], [x[1] for x in list_var_weight]
                )
                <= nb_trips_per_link[index_tl] * tl.transport_type.capacity
            )

        # Flows constraint
        for location in self.problem.locations:
            loc_id = location.id
            debug_mode = args.get("debug_mode", False)
            index_products = args.get("debug_settings", {}).get(
                "index_products", range(len(self.problem.products))
            )
            for index_product in index_products:
                product = self.problem.products[index_product]
                net_supply = location.net_supply.get(product, 0)
                flow_in = cp_model.LinearExpr.Sum(
                    [
                        flows_variables[x[0]][x[1]]
                        for x in keys_inflow_location_product.get(loc_id, {}).get(
                            product.id, set()
                        )
                    ]
                )
                flow_out = cp_model.LinearExpr.Sum(
                    [
                        flows_variables[x[0]][x[1]]
                        for x in keys_outflow_location_product.get(loc_id, {}).get(
                            product.id, set()
                        )
                    ]
                )
                model.Add(flow_in + net_supply - flow_out == 0)
        transport_cost = self.scaling_factor * sum(
            [
                nb_trips_per_link[index_tl]
                * int(
                    transport_links[index_tl].distance
                    * transport_links[index_tl].transport_type.cost
                )
                for index_tl in range(self.problem.nb_transport_links)
            ]
        )
        emission_cost = self.scaling_factor * sum(
            [
                nb_trips_per_link[index_tl]
                * int(
                    transport_links[index_tl].distance
                    * transport_links[index_tl].transport_type.emissions
                )
                for index_tl in range(self.problem.nb_transport_links)
            ]
        )
        estimated_compound = []
        for index_tl in range(self.problem.nb_transport_links):
            tl = self.problem.transport_links[index_tl]
            time = tl.distance / tl.transport_type.speed
            value = sum(
                flows_variables[index_tl][index_product]
                * int(self.problem.products[index_product].value)
                for index_product in flows_variables[index_tl]
            )
            estimated_compound.append(
                value
                * int(
                    self.scaling_factor
                    * ((1 + self.problem.capital_factor) ** time - 1)
                )
            )
        self.variables["total_obj"] = (
            transport_cost + emission_cost + sum(estimated_compound)
        )
        model.Minimize(self.variables["total_obj"])
        self.variables["objs"] = {
            "transport": transport_cost,
            "emission": emission_cost,
            "capital": sum(estimated_compound),
        }
        self.variables["flows"] = flows_variables
        self.variables["nb_trips"] = nb_trips_per_link
        self.variables["nb_trips_per_link_per_product"] = nb_trips_per_link_per_product

    def init_model_detailed_trips(self, **args: Any) -> None:
        sol: MultibatchingSolution = args.get("solution", None)
        delta: int = args.get("delta_to_solution", 1)
        max_nb_trips_per_link = args.get("max_trips", 50)
        logger.info(f"Max trips : {max_nb_trips_per_link}")
        max_nb_trip_per_transport_link = None
        if sol is not None:
            max_nb_trip_per_transport_link = defaultdict(lambda: delta)
            for pt in sol.list_flows:
                id_tl = self.problem.transport_links_to_index[pt.transport_link]
                max_nb_trip_per_transport_link[id_tl] += pt.nb_packing
        supply_per_product = {
            p: self.problem.get_total_supply(p) for p in self.problem.products
        }
        total_size_per_product = {
            p: p.size * supply_per_product[p] for p in self.problem.products
        }
        model = cp_model.CpModel()
        transport_links = self.problem.transport_links
        flows_variables = {}
        used_trips_variables = {}
        trips_variables = {}
        nb_trips_per_link = {}
        keys_inflow_location_product = dict()
        keys_outflow_location_product = dict()
        for index_tl in range(self.problem.nb_transport_links):
            flows_variables[index_tl] = {}
            trips_variables[index_tl] = {}
            used_trips_variables[index_tl] = {}
            tl = transport_links[index_tl]
            loc1 = tl.location_l1
            loc2 = tl.location_l2
            if loc1.id not in keys_outflow_location_product:
                keys_outflow_location_product[loc1.id] = defaultdict(lambda: set())
            if loc2.id not in keys_inflow_location_product:
                keys_inflow_location_product[loc2.id] = defaultdict(lambda: set())
            tr: TransportType = tl.transport_type
            max_trips_for_this_link = tl.max_trips
            if max_trips_for_this_link is None:
                max_trips_for_this_link = max_nb_trips_per_link
            if max_nb_trip_per_transport_link is not None:
                max_trips_for_this_link = max_nb_trip_per_transport_link[index_tl]
            nb_trips_per_link[index_tl] = model.NewIntVar(
                lb=0, ub=max_trips_for_this_link, name=f"num_trips_{index_tl}"
            )
            list_var_weight = []
            for nb_trip in range(max_trips_for_this_link):
                trips_variables[index_tl][nb_trip] = {}
                # means "nb_trip-th" vehicle is used.
                used_trips_variables[index_tl][nb_trip] = model.NewBoolVar(
                    name=f"used_{index_tl}_{nb_trip}"
                )
            # Symmetry breaking
            for i in range(1, max_trips_for_this_link):
                # ensure that if used_trip[i] == 0 then used_trip[i+1] == 0
                model.Add(
                    used_trips_variables[index_tl][i]
                    <= used_trips_variables[index_tl][i - 1]
                )
            for index_product in range(self.problem.nb_products):
                product = self.problem.products[index_product]
                if tr in product.valid_transports:
                    flows_variables[index_tl][index_product] = model.NewIntVar(
                        lb=0, ub=100, name=f"flow_{index_tl}_{index_product}"
                    )
                    keys_outflow_location_product[loc1.id][product.id].add(
                        (index_tl, index_product)
                    )
                    keys_inflow_location_product[loc2.id][product.id].add(
                        (index_tl, index_product)
                    )
                    list_var_weight.append(
                        (flows_variables[index_tl][index_product], int(product.size))
                    )
                    for nb_trip in trips_variables[index_tl]:
                        trips_variables[index_tl][nb_trip][index_product] = (
                            model.NewIntVar(
                                lb=0,
                                ub=math.floor(
                                    tl.transport_type.capacity / product.size
                                ),
                                name=f"flow_{index_tl}_{nb_trip}_{index_product}",
                            )
                        )
                    # Total flows of product in this edge ==
                    # the sum of transport product on individual trips on this edge.
                    model.Add(
                        sum(
                            [
                                trips_variables[index_tl][nb_trip][index_product]
                                for nb_trip in trips_variables[index_tl]
                            ]
                        )
                        == flows_variables[index_tl][index_product]
                    )
            for i in used_trips_variables[index_tl]:
                model.Add(
                    sum(
                        [
                            trips_variables[index_tl][i][p]
                            for p in trips_variables[index_tl][i]
                        ]
                    )
                    > 0
                ).OnlyEnforceIf(used_trips_variables[index_tl][i])
                model.Add(
                    sum(
                        [
                            trips_variables[index_tl][i][p]
                            for p in trips_variables[index_tl][i]
                        ]
                    )
                    == 0
                ).OnlyEnforceIf(used_trips_variables[index_tl][i].Not())
                (
                    model.Add(
                        sum(
                            [
                                trips_variables[index_tl][i][p]
                                * int(self.problem.products[p].size)
                                for p in trips_variables[index_tl][i]
                            ]
                        )
                        <= tl.transport_type.capacity
                    ).OnlyEnforceIf(used_trips_variables[index_tl][i])
                )
            # Global capacity constraint
            model.Add(
                cp_model.LinearExpr.weighted_sum(
                    [x[0] for x in list_var_weight], [x[1] for x in list_var_weight]
                )
                <= nb_trips_per_link[index_tl] * tl.transport_type.capacity
            )
            model.Add(
                nb_trips_per_link[index_tl]
                == sum(
                    used_trips_variables[index_tl][i]
                    for i in used_trips_variables[index_tl]
                )
            )

        # Flows constraint
        for location in self.problem.locations:
            loc_id = location.id
            for product in self.problem.products:
                net_supply = location.net_supply.get(product, 0)
                flow_in = cp_model.LinearExpr.Sum(
                    [
                        flows_variables[x[0]][x[1]]
                        for x in keys_inflow_location_product.get(loc_id, {}).get(
                            product.id, set()
                        )
                    ]
                )
                flow_out = cp_model.LinearExpr.Sum(
                    [
                        flows_variables[x[0]][x[1]]
                        for x in keys_outflow_location_product.get(loc_id, {}).get(
                            product.id, set()
                        )
                    ]
                )
                model.Add(flow_in + net_supply - flow_out == 0)

        transport_cost = self.scaling_factor * sum(
            [
                nb_trips_per_link[index_tl]
                * transport_links[index_tl].distance
                * transport_links[index_tl].transport_type.cost
                for index_tl in range(self.problem.nb_transport_links)
            ]
        )

        emission_cost = self.scaling_factor * sum(
            [
                nb_trips_per_link[index_tl]
                * transport_links[index_tl].distance
                * transport_links[index_tl].transport_type.emissions
                for index_tl in range(self.problem.nb_transport_links)
            ]
        )

        nb_trips_cost = sum(
            [
                used_trips_variables[index_tl][nb_trip]
                for index_tl in used_trips_variables
                for nb_trip in used_trips_variables[index_tl]
            ]
        )
        compound_cost = []
        for index_tl in range(self.problem.nb_transport_links):
            tl = transport_links[index_tl]
            time = tl.distance / tl.transport_type.speed
            for i in trips_variables[index_tl]:
                value = sum(
                    [
                        trips_variables[index_tl][i][index_p]
                        * self.problem.products[index_p].value
                        for index_p in trips_variables[index_tl][i]
                    ]
                )
                compound_cost.append(
                    value
                    * int(
                        self.scaling_factor
                        * ((1 + self.problem.capital_factor) ** time - 1)
                    )
                )
        self.variables["total_obj"] = (
            transport_cost + emission_cost + sum(compound_cost)
        )
        model.Minimize(self.variables["total_obj"])
        self.cp_model = model
        self.variables["flows"] = flows_variables
        self.variables["nb_trips"] = nb_trips_per_link
        self.variables["used_trips"] = used_trips_variables
        self.variables["trips_variables"] = trips_variables

    def compute_nb_trips_min(self, product: Product, demand_of_product: int):
        nb_trips_min_for_product_flows = {}
        size_product = product.size
        for tl in range(self.problem.nb_transport_types):
            tt: TransportType = self.problem.transport_types[tl]
            capa_transport = tt.capacity
            nb_trips_min_for_product_flows[tt] = [0]
            current_greedy_bins = defaultdict(lambda: 0)
            current_bin = 0
            current_greedy_bins[0] = 0
            for i in range(demand_of_product):
                if current_greedy_bins[current_bin] + size_product <= capa_transport:
                    current_greedy_bins[current_bin] += size_product
                else:
                    current_bin += 1
                    current_greedy_bins[current_bin] = size_product
                nb_trips_min_for_product_flows[tt].append(current_bin + 1)
        return nb_trips_min_for_product_flows

    def set_warm_start_from_prev_solve(self):
        if self.solver is not None:
            self.cp_model.ClearHints()
            response = self.solver.ResponseProto()  # Get the raw response
            for i in range(len(response.solution)):
                var = self.cp_model.GetIntVarFromProtoIndex(i)
                # print(f"Variable {var} = {response.solution[i]}")
                self.cp_model.AddHint(var, response.solution[i])

    def set_warm_start(self, solution: MultibatchingSolution) -> None:
        if self.modeling == ModelingMultiBatch.FLOW:
            self.set_warm_start_flow(solution=solution)
        if self.modeling == ModelingMultiBatch.UNIT_FLOW:
            self.set_warm_start_unit_flow(solution=solution)

    def set_warm_start_unit_flow(self, solution: MultibatchingSolution) -> None:
        """
        Provides a warm start for the UNIT_FLOW modeling by hinting individual trip contents.
        """
        self.cp_model.ClearHints()
        # Group all trips by transport link
        trips_per_link = defaultdict(list)
        for flow in solution.list_flows:
            # "Unroll" the aggregated solution: if nb_packing is 5, add the packing 5 times
            for _ in range(flow.nb_packing):
                trips_per_link[flow.transport_link].append(flow.product_packing)
        # Now, assign the actual trips from the solution to the model variables
        done_link_idx = set()
        for link, trips in trips_per_link.items():
            sum_per_product = defaultdict(lambda: 0)
            link_idx = self.problem.transport_links_to_index[link]
            max_trips_for_link = len(self.variables["used_trips"][link_idx])
            done_link_idx.add(link_idx)
            # Iterate through the unrolled trips and assign them one by one
            for trip_idx, packing_content in enumerate(trips):
                if trip_idx >= max_trips_for_link:
                    logger.warning(
                        f"Solution has more trips ({len(trips)}) on link {link.id} "
                        f"than available in the model ({max_trips_for_link}). "
                        f"Hint will be truncated."
                    )
                    break
                self.cp_model.AddHint(
                    self.variables["used_trips"][link_idx][trip_idx], 1
                )
                for product, units in packing_content.items():
                    product_idx = self.problem.product_to_index[product]
                    if (
                        product_idx
                        in self.variables["trips_variables"][link_idx][trip_idx]
                    ):
                        content_var = self.variables["trips_variables"][link_idx][
                            trip_idx
                        ][product_idx]
                        self.cp_model.AddHint(content_var, int(units))
                        sum_per_product[product_idx] += int(units)
                for p in self.variables["trips_variables"][link_idx][trip_idx]:
                    if self.problem.products[p] not in packing_content:
                        self.cp_model.AddHint(
                            self.variables["trips_variables"][link_idx][trip_idx][p], 0
                        )
            for product in self.variables["flows"][link_idx]:
                self.cp_model.AddHint(
                    self.variables["flows"][link_idx][product], sum_per_product[product]
                )
            self.cp_model.AddHint(self.variables["nb_trips"][link_idx], len(trips))
            for k in range(len(trips), max_trips_for_link):
                self.cp_model.AddHint(self.variables["used_trips"][link_idx][k], 0)
                for product_idx in self.variables["trips_variables"][link_idx][k]:
                    self.cp_model.AddHint(
                        self.variables["trips_variables"][link_idx][k][product_idx], 0
                    )
        for index_link in self.variables["used_trips"]:
            if index_link not in done_link_idx:
                max_trips_for_link = len(self.variables["used_trips"][index_link])
                self.cp_model.AddHint(self.variables["nb_trips"][index_link], 0)
                for prod in self.variables["flows"][index_link]:
                    self.cp_model.AddHint(self.variables["flows"][index_link][prod], 0)
                for k in range(max_trips_for_link):
                    self.cp_model.AddHint(
                        self.variables["used_trips"][index_link][k], 0
                    )
                    for product_idx in self.variables["trips_variables"][index_link][k]:
                        self.cp_model.AddHint(
                            self.variables["trips_variables"][index_link][k][
                                product_idx
                            ],
                            0,
                        )
        logger.info("Warm start hints added for UNIT_FLOW model.")

    def set_warm_start_flow(self, solution: MultibatchingSolution) -> None:
        aggreg_flows_solution = {}
        nb_trips = {}
        self.cp_model.ClearHints()
        for flow in solution.list_flows:
            ind = self.problem.transport_links_to_index[flow.transport_link]
            if ind not in aggreg_flows_solution:
                aggreg_flows_solution[ind] = defaultdict(lambda: 0)
                nb_trips[ind] = 0
            for p in flow.product_packing:
                aggreg_flows_solution[ind][self.problem.product_to_index[p]] += int(
                    flow.product_packing[p] * flow.nb_packing
                )
            nb_trips[ind] += flow.nb_packing
        for index_link in self.variables["flows"]:
            for index_product in self.variables["flows"][index_link]:
                self.cp_model.AddHint(
                    self.variables["flows"][index_link][index_product],
                    aggreg_flows_solution.get(index_link, {}).get(index_product, 0),
                )
            self.cp_model.AddHint(
                self.variables["nb_trips"][index_link], nb_trips.get(index_link, 0)
            )
        print("ws done")

    def logs_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        if "objs" in self.variables:
            for obj in self.variables["objs"]:
                logger.info(f"{obj}: {cpsolvercb.value(self.variables['objs'][obj])}")
        for index_tl in self.variables["nb_trips"]:
            tl = self.problem.transport_links[index_tl]
            nb_trips = cpsolvercb.value(self.variables["nb_trips"][index_tl])
            if nb_trips > 0:
                logger.info(
                    f"Nb trips on {tl.location_l1, tl.location_l2, tl.transport_type.id}, :"
                    f"{cpsolvercb.value(self.variables['nb_trips'][index_tl])}"
                )
                for index_product in self.variables["flows"][index_tl]:
                    val = cpsolvercb.value(
                        self.variables["flows"][index_tl][index_product]
                    )
                    if val > 0:
                        logger.info(
                            f"Nb product {self.problem.products[index_product].id} "
                            f"on {tl.location_l1, tl.location_l2, tl.transport_type.id}, :"
                            f"{cpsolvercb.value(self.variables['flows'][index_tl][index_product])}"
                        )
            if self.modeling == ModelingMultiBatch.UNIT_FLOW:
                for i in self.variables["trips_variables"][index_tl]:
                    used = self.variables["used_trips"][index_tl][i]
                    if cpsolvercb.value(used) == 1:
                        logger.info(
                            f"{i}-th trip used on route {tl.location_l1, tl.location_l2, tl.transport_type.id}"
                        )
                        for index_product in self.variables["trips_variables"][
                            index_tl
                        ][i]:
                            val = cpsolvercb.value(
                                self.variables["trips_variables"][index_tl][i][
                                    index_product
                                ]
                            )
                            if val > 0:
                                logger.info(
                                    f"Nb product {self.problem.products[index_product].id} "
                                    f"on {tl.location_l1, tl.location_l2, tl.transport_type.id}, :"
                                    f"{val}"
                                )
                        logger.info(f"------")
            if nb_trips > 0:
                logger.info(f"------End of transport link------")

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        if self.verbose_logs > 0:
            self.logs_solution(cpsolvercb)
        packing_solution = []
        products = self.problem.products
        for index_tl in self.variables["flows"]:
            nb = cpsolvercb.value(self.variables["nb_trips"][index_tl])
            if nb > 0:
                packings = []
                if self.modeling == ModelingMultiBatch.UNIT_FLOW:
                    for i in sorted(self.variables["trips_variables"][index_tl]):
                        used = self.variables["used_trips"][index_tl][i]
                        if cpsolvercb.value(used) > 0:
                            current_packing = set()
                            for index_product in self.variables["trips_variables"][
                                index_tl
                            ][i]:
                                val = cpsolvercb.value(
                                    self.variables["trips_variables"][index_tl][i][
                                        index_product
                                    ]
                                )
                                if val > 0:
                                    current_packing.add((index_product, int(val)))
                            cpack = frozenset(current_packing)
                            packings.append(cpack)
                    if len(packings) > 0:
                        c = Counter(packings)
                        for key, val in c.items():
                            packing_transport = PackingTransport(
                                transport_link=self.problem.transport_links[index_tl],
                                product_packing={products[x[0]]: x[1] for x in key},
                                nb_packing=val,
                            )
                            packing_solution.append(packing_transport)
                else:
                    cur_packing = {}
                    if not self.single_batching:
                        for index_product in self.variables["flows"][index_tl]:
                            val = cpsolvercb.value(
                                self.variables["flows"][index_tl][index_product]
                            )
                            if val > 0:
                                cur_packing[products[index_product]] = val / nb
                        if len(cur_packing) > 0:
                            packing_solution.append(
                                PackingTransport(
                                    transport_link=self.problem.transport_links[
                                        index_tl
                                    ],
                                    product_packing=cur_packing,
                                    nb_packing=nb,
                                )
                            )
                    else:
                        for index_product in self.variables["flows"][index_tl]:
                            val = cpsolvercb.value(
                                self.variables["flows"][index_tl][index_product]
                            )
                            if val > 0:
                                p = self.problem.products[index_product]
                                tl = self.problem.transport_links[index_tl]
                                tr = tl.transport_type
                                nb_trips_per_link_per_product = cpsolvercb.value(
                                    self.variables["nb_trips_per_link_per_product"][
                                        (index_tl, index_product)
                                    ]
                                )
                                nb_product_per_transport = (
                                    tr.capacity
                                    // self.problem.products[index_product].size
                                )
                                nb_trips_per_link_per_product = math.ceil(
                                    val / nb_product_per_transport
                                )
                                if nb_trips_per_link_per_product > 1:
                                    if val % nb_product_per_transport == 0:
                                        packings = [
                                            PackingTransport(
                                                transport_link=tl,
                                                product_packing={
                                                    p: nb_product_per_transport
                                                },
                                                nb_packing=nb_trips_per_link_per_product,
                                            )
                                        ]
                                    else:
                                        packings = [
                                            PackingTransport(
                                                transport_link=tl,
                                                product_packing={
                                                    p: nb_product_per_transport
                                                },
                                                nb_packing=nb_trips_per_link_per_product
                                                - 1,
                                            )
                                        ] + [
                                            PackingTransport(
                                                transport_link=tl,
                                                product_packing={
                                                    p: val % nb_product_per_transport
                                                },
                                                nb_packing=1,
                                            )
                                        ]
                                else:
                                    packings = [
                                        PackingTransport(
                                            tl, product_packing={p: val}, nb_packing=1
                                        )
                                    ]
                                if (
                                    sum(
                                        [
                                            pack.product_packing[p] * pack.nb_packing
                                            for pack in packings
                                        ]
                                    )
                                    != val
                                ):
                                    print(
                                        val,
                                        sum(
                                            [
                                                pack.product_packing[p]
                                                * pack.nb_packing
                                                for pack in packings
                                            ]
                                        ),
                                    )
                                    print(
                                        "Nb product per transport",
                                        nb_product_per_transport,
                                        val,
                                        nb_trips_per_link_per_product,
                                    )
                                    print(packings)
                                    print("Problem")
                                packing_solution.extend(packings)
        sol = MultibatchingSolution(problem=self.problem, list_flows=packing_solution)
        return sol

    def add_advanced_capacity_constraints(
        self, model, flows_variables, nb_trips_per_link, solver_type="cpsat"
    ):
        """
        Adds improved lower bounds on the number of trips based on bin packing dual feasible functions.
        Adapted from the DP bounds for SALBP/BinPacking.
        """
        # Import relevant library for summation based on solver type
        if solver_type == "gurobi":
            import gurobipy

            sum_func = gurobipy.quicksum
        else:
            # For CP-SAT, sum() works with LinearExpr
            sum_func = sum

        for index_tl in range(self.problem.nb_transport_links):
            # Skip if variables not defined for this link
            if index_tl not in nb_trips_per_link:
                continue

            tl = self.problem.transport_links[index_tl]
            capacity = tl.transport_type.capacity
            if capacity <= 0:
                continue

            nb_trips_var = nb_trips_per_link[index_tl]

            # --- Bound 2 (L2): Items related to C/2 ---
            # Logic: Items > C/2 take a full bin. Items = C/2 take half a bin.
            # Formula: 2 * NbTrips >= 2 * sum(Flow(>C/2)) + 1 * sum(Flow(=C/2))
            expr_2 = []

            # --- Bound 3 (L3): Items related to C/3 and 2C/3 ---
            # Logic:
            # > 2C/3 : weight 1 (takes full bin usually)
            # = 2C/3 : weight 2/3
            # > C/3  : weight 1/2
            # = C/3  : weight 1/3
            # Scaled by 6 to stay integer:
            # 6 * NbTrips >= 6*Flow(>2C/3) + 4*Flow(=2C/3) + 3*Flow(>C/3) + 2*Flow(=C/3)
            expr_3 = []

            # Iterate over products existing on this link
            current_flow_vars = flows_variables[index_tl]
            for index_product in current_flow_vars:
                product = self.problem.products[index_product]
                size = product.size
                flow_var = current_flow_vars[index_product]

                # Comparisons using multiplication to avoid float issues

                # --- For Bound 2 ---
                if 2 * size > capacity:  # size > C/2
                    expr_2.append(2 * flow_var)
                elif 2 * size == capacity:  # size == C/2
                    expr_2.append(flow_var)

                # --- For Bound 3 ---
                if 3 * size > 2 * capacity:  # size > 2C/3
                    expr_3.append(6 * flow_var)
                elif 3 * size == 2 * capacity:  # size == 2C/3
                    expr_3.append(4 * flow_var)
                elif 3 * size > capacity:  # size > C/3
                    expr_3.append(3 * flow_var)
                elif 3 * size == capacity:  # size == C/3
                    expr_3.append(2 * flow_var)

            # Add constraints if applicable
            if len(expr_2) > 0:
                if solver_type == "gurobi":
                    model.addLConstr(2 * nb_trips_var >= sum_func(expr_2))
                else:
                    model.Add(2 * nb_trips_var >= sum_func(expr_2))

            if len(expr_3) > 0:
                if solver_type == "gurobi":
                    model.addLConstr(6 * nb_trips_var >= sum_func(expr_3))
                else:
                    model.Add(6 * nb_trips_var >= sum_func(expr_3))

    def add_advanced_capacity_constraints2(
        self, model, flows_variables, nb_trips_per_link, solver_type="cpsat", max_k=30
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

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objs"].keys())

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.clear_objective()
        self.cp_model.minimize(self.variables["objs"][obj])

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        return [self.cp_model.add(self.variables["objs"][obj] <= value)]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        if len(res) > 0:
            sol: MultibatchingSolution = res[-1][0]
            if "_intern_obj" in sol.__dict__.keys():
                return sol._intern_obj[obj]
            else:
                kpis = self.problem.evaluate(sol)
                return int(self.scaling_factor * kpis[obj])
        else:
            return None
