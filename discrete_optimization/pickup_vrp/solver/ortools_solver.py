#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import print_function

import logging
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2,
    routing_parameters_pb2,
)
from ortools.util.optional_boolean_pb2 import BOOL_FALSE, BOOL_TRUE

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    fitness_class,
)
from discrete_optimization.pickup_vrp.gpdp import (
    GPDP,
    GPDPSolution,
    Node,
    build_matrix_distance,
    build_matrix_time,
)

logger = logging.getLogger(__name__)


class NodePosition(Enum):
    """Node position inside a trajectory.

    Useful e.g. to distinguish between the starting node and ending node
    when the trajectory is a loop.

    """

    START = "start"
    INTERMEDIATE = "intermediate"
    END = "end"


class ParametersCost:
    def __init__(
        self,
        dimension_name: str,
        global_span: bool = True,
        sum_over_vehicles: bool = False,
        coefficient_vehicles: Union[float, List[float]] = 100,
    ):
        self.dimension_name = dimension_name
        self.global_span = global_span
        self.sum_over_vehicles = sum_over_vehicles
        self.coefficient_vehicles = coefficient_vehicles
        self.different_coefficient = (
            isinstance(self.coefficient_vehicles, list)
            and len(set(self.coefficient_vehicles)) > 1
        )

    @staticmethod
    def default() -> "ParametersCost":
        return ParametersCost(
            dimension_name="Distance",
            global_span=True,
            sum_over_vehicles=False,
            coefficient_vehicles=100,
        )


def apply_cost(
    list_parameters_cost: List[ParametersCost], routing: pywrapcp.RoutingModel
) -> None:
    dimension_names = set([p.dimension_name for p in list_parameters_cost])
    dimension_dict = {d: routing.GetDimensionOrDie(d) for d in dimension_names}
    for p in list_parameters_cost:
        if p.global_span:
            dimension_dict[p.dimension_name].SetGlobalSpanCostCoefficient(
                p.coefficient_vehicles
            )
        else:
            if p.different_coefficient:
                for i in range(len(p.coefficient_vehicles)):  # type: ignore
                    dimension_dict[p.dimension_name].SetSpanCostCoefficientForVehicle(
                        p.coefficient_vehicles[i], i  # type: ignore
                    )
            else:
                dimension_dict[p.dimension_name].SetSpanCostCoefficientForAllVehicles(
                    p.coefficient_vehicles[0]  # type: ignore
                )


local_search_metaheuristic_enum = routing_enums_pb2.LocalSearchMetaheuristic
metaheuristic_names = [
    (k, getattr(local_search_metaheuristic_enum, k))
    for k in local_search_metaheuristic_enum.__dict__.keys()
    if isinstance(getattr(local_search_metaheuristic_enum, k), int)
]
name_metaheuristic_to_value = {x[0]: x[1] for x in metaheuristic_names}
value_metaheuristic_to_name = {x[1]: x[0] for x in metaheuristic_names}

first_solution_strategy_enum = routing_enums_pb2.FirstSolutionStrategy
first_solution_names = [
    (k, getattr(first_solution_strategy_enum, k))
    for k in first_solution_strategy_enum.__dict__.keys()
    if isinstance(getattr(first_solution_strategy_enum, k), int)
]
name_firstsolution_to_value = {x[0]: x[1] for x in first_solution_names}
value_firstsolution_to_name = {x[1]: x[0] for x in first_solution_names}


class MetaheuristicEnum(Enum):
    UNSET = "UNSET"
    AUTOMATIC = "AUTOMATIC"
    GREEDY_DESCENT = "GREEDY_DESCENT"
    GUIDED_LOCAL_SEARCH = "GUIDED_LOCAL_SEARCH"
    SIMULATED_ANNEALING = "SIMULATED_ANNEALING"
    TABU_SEARCH = "TABU_SEARCH"
    GENERIC_TABU_SEARCH = "GENERIC_TABU_SEARCH"


class FirstSolutionEnum(Enum):
    UNSET = "UNSET"
    AUTOMATIC = "AUTOMATIC"
    PATH_CHEAPEST_ARC = "PATH_CHEAPEST_ARC"
    PATH_MOST_CONSTRAINED_ARC = "PATH_MOST_CONSTRAINED_ARC"
    EVALUATOR_STRATEGY = "EVALUATOR_STRATEGY"
    SAVINGS = "SAVINGS"
    SWEEP = "SWEEP"
    CHRISTOFIDES = "CHRISTOFIDES"
    ALL_UNPERFORMED = "ALL_UNPERFORMED"
    BEST_INSERTION = "BEST_INSERTION"
    PARALLEL_CHEAPEST_INSERTION = "PARALLEL_CHEAPEST_INSERTION"
    SEQUENTIAL_CHEAPEST_INSERTION = "SEQUENTIAL_CHEAPEST_INSERTION"
    LOCAL_CHEAPEST_INSERTION = "LOCAL_CHEAPEST_INSERTION"
    GLOBAL_CHEAPEST_ARC = "GLOBAL_CHEAPEST_ARC"
    LOCAL_CHEAPEST_ARC = "LOCAL_CHEAPEST_ARC"
    FIRST_UNBOUND_MIN_VALUE = "FIRST_UNBOUND_MIN_VALUE"


# LocalSearchMetaheuristic = Enum([k[0] for k in metaheuristic_names])
# FirstSolutionStrategy = Enum([k[0] for k in first_solution_names])
# https://developers.google.com/optimization/routing/routing_options


status_description = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_FAIL",
    3: "ROUTING_FAIL_TIMEOUT",
    4: "ROUTING_INVALID",
}


class ORToolsGPDP(SolverDO):
    def __init__(
        self,
        problem: GPDP,
        factor_multiplier_distance: float = 1,
        factor_multiplier_time: float = 1,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        self.dimension_names: List[str] = []
        self.factor_multiplier_distance = factor_multiplier_distance  # 10**3
        self.factor_multiplier_time = factor_multiplier_time  # 10**3
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=problem,
            params_objective_function=params_objective_function,
        )

    def init_model(self, **kwargs: Any) -> None:
        include_time_windows = kwargs.get("include_time_windows", False)
        include_time_windows_cluster = kwargs.get("include_time_windows_cluster", False)
        include_cumulative = kwargs.get("include_cumulative", False)
        include_mandatory = kwargs.get("include_mandatory", True)
        include_demand = kwargs.get("include_demand", True)
        include_resource_dimension = kwargs.get("include_resource_dimension", False)
        include_time_dimension = kwargs.get("include_time_dimension", True)
        list_parameters_cost = kwargs.get("parameters_cost", [ParametersCost.default()])
        one_visit_per_cluster = kwargs.get("one_visit_per_cluster", False)
        include_pickup_and_delivery = kwargs.get("include_pickup_and_delivery", False)
        include_pickup_and_delivery_per_cluster = kwargs.get(
            "include_pickup_and_delivery_per_cluster", False
        )
        use_matrix = kwargs.get("use_matrix", True)
        include_equilibrate_charge = kwargs.get("include_equilibrate_charge", False)
        hard_equilibrate = kwargs.get("hard_equilibrate", False)
        use_constant_max_slack_time = kwargs.get("use_constant_max_slack_time", True)
        constant_max_slack_time = kwargs.get("max_slack_time", 1000)
        use_max_slack_time_per_node = kwargs.get("use_max_slack_time_per_node", False)
        max_slack_time_per_node = kwargs.get(
            "max_slack_time_per_node", self.problem.slack_time_bound_per_node
        )
        force_start_cumul_time_zero = kwargs.get("force_start_cumul_time_zero", True)
        max_time_per_vehicle = kwargs.get("max_time_per_vehicle", 1000000)
        max_distance_per_vehicle = kwargs.get("max_distance_per_vehicle", 1000000)
        set_transit_cost_by_default = kwargs.get("set_transit_cost_by_default", True)
        # use or not the distance callback function to evaluate the cost of each edge.
        # Whatever the cost function you want to optimize (via the list_parameters_cost argument),
        # these cost are still included in the objective function
        if use_matrix:
            matrix_distance_int = kwargs.get("matrix", None)
            if matrix_distance_int is None:
                matrix_distance = build_matrix_distance(self.problem)
                matrix_distance_int = np.array(
                    matrix_distance * self.factor_multiplier_distance, dtype=np.int_
                )
            if include_time_dimension:
                matrix_time = build_matrix_time(self.problem)
                matrix_time_int = np.array(
                    matrix_time * self.factor_multiplier_time, dtype=np.int_
                )
        capacities_dict = {
            r: [
                self.problem.capacities[v][r][1]
                for v in range(self.problem.number_vehicle)
            ]
            for r in self.problem.resources_set
        }
        # Neg Capacity version :
        neg_capacity_version = kwargs.get("neg_capacity_version", True)
        demands: Dict[str, List[float]]
        if neg_capacity_version:
            demands = {
                r: [
                    max(0.0, -self.problem.resources_flow_node[node].get(r, 0.0))
                    for node in self.problem.list_nodes
                ]
                for r in self.problem.resources_set
            }
        else:
            demands = {
                r: [
                    self.problem.resources_flow_node[node].get(r, 0.0)
                    for node in self.problem.list_nodes
                ]
                for r in self.problem.resources_set
            }
        self.demands = demands
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(self.problem.all_nodes_dict),
            self.problem.number_vehicle,
            [
                self.problem.index_nodes[self.problem.origin_vehicle[v]]
                for v in range(self.problem.number_vehicle)
            ],
            [
                self.problem.index_nodes[self.problem.target_vehicle[v]]
                for v in range(self.problem.number_vehicle)
            ],
        )
        routing = pywrapcp.RoutingModel(manager)
        consider_empty_route_cost = kwargs.get("consider_empty_route_cost", True)
        if consider_empty_route_cost:
            for v in range(self.problem.number_vehicle):
                try:
                    # Works for <ortools9.2
                    routing.ConsiderEmptyRouteCostsForVehicle(True, v)
                except:
                    # Works for >=ortools9.2
                    routing.SetVehicleUsedWhenEmpty(True, v)
        logger.info("routing init")
        if use_matrix:
            # Create and register a transit callback.
            def distance_callback(from_index: int, to_index: int) -> int:
                """Returns the distance between the two nodes."""
                # Convert from routing variable Index to distance matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return matrix_distance_int[from_node, to_node]

        else:
            # Create and register a transit callback.
            def distance_callback(from_index: int, to_index: int) -> int:
                """Returns the distance between the two nodes."""
                # Convert from routing variable Index to distance matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return int(
                    self.problem.distance_delta[
                        self.problem.nodes_to_index[from_node]
                    ].get(self.problem.nodes_to_index[to_node], 1000000000)
                )

        if include_equilibrate_charge:
            charge = routing.RegisterTransitCallback(lambda i, j: 1)
            charge_constraint = kwargs.get("charge_constraint", {})
            if all(charge_constraint[i][1] is not None for i in charge_constraint):
                max_charge = max([charge_constraint[i][1] for i in charge_constraint])
            else:
                max_charge = len(self.problem.list_nodes)
            routing.AddDimension(
                charge,
                0,
                max_charge,
                True,
                "charge",
            )
            charge_dimension = routing.GetDimensionOrDie("charge")
            self.dimension_names += ["charge"]
            for i in range(self.problem.number_vehicle):
                index = routing.End(i)
                constraint = charge_constraint[i]
                if constraint[0] is not None:
                    if not hard_equilibrate:
                        charge_dimension.SetCumulVarSoftLowerBound(
                            index, constraint[0], 100000000
                        )
                    else:
                        charge_dimension.CumulVar(index).SetMin(constraint[0])
                if constraint[1] is not None:
                    if not hard_equilibrate:
                        charge_dimension.SetCumulVarSoftUpperBound(
                            index, constraint[1], 100000000
                        )
                    else:
                        charge_dimension.CumulVar(index).SetMax(constraint[1])
        transit_distance_callback_index = routing.RegisterTransitCallback(
            distance_callback
        )
        # Define cost of each arc.
        if set_transit_cost_by_default:
            routing.SetArcCostEvaluatorOfAllVehicles(transit_distance_callback_index)
        logger.info("Distance callback done")
        if include_resource_dimension:

            def ressource_transition(
                from_index: int,
                to_index: int,
                ressource: str,
                problem: GPDP,
                vehicle: int,
            ) -> int:
                """Return the ressource consumption from one node to another."""
                # Convert from routing variable Index to demands NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                l = problem.resources_flow_edges.get(
                    (
                        self.problem.nodes_to_index[from_node],
                        self.problem.nodes_to_index[to_node],
                    ),
                    {ressource: 0},
                )[ressource] + problem.resources_flow_node.get(
                    to_node, {ressource: 0}
                ).get(
                    ressource, 0
                )
                return int(l)

            demand_callback_index_dict = {
                r: [
                    routing.RegisterTransitCallback(
                        partial(
                            ressource_transition,
                            ressource=r,
                            problem=self.problem,
                            vehicle=v,
                        )
                    )
                    for v in range(self.problem.number_vehicle)
                ]
                for r in demands
            }
            for r in demand_callback_index_dict:
                routing.AddDimensionWithVehicleTransitAndCapacity(
                    demand_callback_index_dict[r],
                    0,  # null capacity slack
                    capacities_dict[r],
                    # vehicle maximum capacities
                    True,  # start cumul to zero
                    "Ressource_" + str(r),
                )
                self.dimension_names += ["Ressource_" + str(r)]
        if include_demand:
            # Add Capacity constraint.
            def demand_callback(from_index: int, ressource: str) -> int:
                """Returns the demand of the node."""
                # Convert from routing variable Index to demands NodeIndex.
                from_node = manager.IndexToNode(from_index)
                return int(demands[ressource][from_node])

            demand_callback_index_dict = {
                r: routing.RegisterUnaryTransitCallback(
                    partial(demand_callback, ressource=r)
                )
                for r in demands
            }
            for r in demand_callback_index_dict:
                routing.AddDimensionWithVehicleCapacity(
                    demand_callback_index_dict[r],
                    0,  # null capacity slack
                    capacities_dict[r],  # vehicle maximum capacities
                    True,  # start cumul to zero
                    "Capacity_" + str(r),
                )
                self.dimension_names += ["Capacity_" + str(r)]
        if one_visit_per_cluster:
            for cluster in list(self.problem.clusters_to_node):
                nodes = [
                    manager.NodeToIndex(self.problem.index_nodes[n])
                    for n in self.problem.clusters_to_node[cluster]
                    if n
                    not in {
                        self.problem.origin_vehicle[v]
                        for v in self.problem.origin_vehicle
                    }
                    and n
                    not in {
                        self.problem.target_vehicle[v]
                        for v in self.problem.target_vehicle
                    }
                ]
                routing.AddDisjunction(nodes, 1000000000, 1)
                routing.solver().Add(
                    routing.solver().Sum([routing.ActiveVar(i) for i in nodes]) == 1
                )
        slacks = [
            int(self.problem.time_delta_node[t] * self.factor_multiplier_time)
            for t in self.problem.list_nodes
        ]
        max_slack = max(slacks)
        if use_constant_max_slack_time:
            max_slack = max(max_slack, constant_max_slack_time)
        elif use_max_slack_time_per_node:
            max_slack = max(
                max_slack,
                max([max_slack_time_per_node[n][1] for n in max_slack_time_per_node]),
            )
        if include_time_dimension:

            def time_callback(from_index: int, to_index: int) -> int:
                """Returns the travel time between the two nodes."""
                # Convert from routing variable Index to time matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return matrix_time_int[from_node, to_node]

            transit_time_callback_index = routing.RegisterTransitCallback(time_callback)
            time = "Time"

            routing.AddDimension(
                transit_time_callback_index,
                max_slack,  # allow waiting time
                max_time_per_vehicle,  # maximum time per vehicle
                force_start_cumul_time_zero,  # force or not  start cumul to zero.
                time,
            )
            self.dimension_names += [time]
            time_dimension = routing.GetDimensionOrDie("Time")
            for task in self.problem.all_nodes_dict:
                index = manager.NodeToIndex(self.problem.index_nodes[task])
                if index == -1:
                    continue
                routing.solver().Add(
                    time_dimension.SlackVar(index)
                    >= int(
                        self.problem.time_delta_node[task] * self.factor_multiplier_time
                    )
                )
        if include_pickup_and_delivery:
            distance = "Distance"
            routing.AddDimension(
                transit_distance_callback_index,
                0,  # allow waiting time on this dimension
                max_distance_per_vehicle,  # maximum distance per vehicle
                True,  # force distance cumul to zero.
                distance,
            )
            self.dimension_names += [distance]
            distance_dimension = routing.GetDimensionOrDie(distance)
            for pickup_deliver in self.problem.list_pickup_deliverable:
                pickup = pickup_deliver[0]
                deliver = pickup_deliver[1]
                for p in pickup:
                    for d in deliver:
                        pickup_index = manager.NodeToIndex(self.problem.index_nodes[p])
                        delivery_index = manager.NodeToIndex(
                            self.problem.index_nodes[d]
                        )
                        routing.AddPickupAndDelivery(pickup_index, delivery_index)
                        routing.solver().Add(
                            routing.VehicleVar(pickup_index)
                            == routing.VehicleVar(delivery_index)
                        )
                        routing.solver().Add(
                            distance_dimension.CumulVar(pickup_index)
                            <= distance_dimension.CumulVar(delivery_index)
                        )
        if include_pickup_and_delivery_per_cluster:
            distance = "Distance"
            routing.AddDimension(
                transit_distance_callback_index,
                0,  # allow waiting time
                max_distance_per_vehicle,  # maximum time per vehicle
                True,  # force start cumul to zero.
                distance,
            )
            self.dimension_names += [distance]
            distance_dimension = routing.GetDimensionOrDie(distance)
            if self.problem.list_pickup_deliverable_per_cluster is None:
                raise ValueError(
                    "When include_pickup_and_delivery_per_cluster is True, "
                    "self.problem.list_pickup_deliverable_per_cluster cannot be None."
                )
            for pickup_deliver in self.problem.list_pickup_deliverable_per_cluster:
                pickup = pickup_deliver[0]
                deliver = pickup_deliver[1]
                for p in pickup:
                    for d in deliver:
                        pp = [
                            manager.NodeToIndex(x)
                            for x in self.problem.clusters_to_node[p]
                        ]
                        dd = [
                            manager.NodeToIndex(x)
                            for x in self.problem.clusters_to_node[d]
                        ]
                        pickup = routing.AddDisjunction(pp)
                        deliver = routing.AddDisjunction(dd)
                        routing.AddPickupAndDeliverySets(pickup, deliver)
                        routing.solver().Add(
                            routing.solver().Sum(
                                [
                                    routing.ActiveVar(i)
                                    * distance_dimension.CumulVar(i)
                                    for i in pp
                                ]
                                + [
                                    -routing.ActiveVar(i)
                                    * distance_dimension.CumulVar(i)
                                    for i in dd
                                ]
                            )
                            < 0
                        )
        if include_cumulative:
            starts = {}
            intervals = {}
            time_dimension = routing.GetDimensionOrDie("Time")
            i = 0
            for (set_of_task, limit) in self.problem.cumulative_constraints:
                list_of_tasks = list(set_of_task)
                index_tasks = [
                    manager.NodeToIndex(self.problem.index_nodes[t])
                    for t in list_of_tasks
                ]
                time_delta = [
                    self.problem.time_delta_node[t] * self.factor_multiplier_time
                    for t in list_of_tasks
                ]
                for task, tdelta in zip(index_tasks, time_delta):
                    if task not in starts:
                        starts[task] = time_dimension.CumulVar(task)
                        intervals[task] = routing.solver().IntervalVar(
                            0,
                            10000000,
                            tdelta,
                            max_slack,
                            0,
                            10000000,
                            False,
                            f"interval_{task}",
                        )
                        routing.solver().Add(
                            intervals[task].StartExpr() == starts[task]
                        )
                        routing.solver().Add(
                            intervals[task].EndExpr()
                            == starts[task] + time_dimension.SlackVar(task)
                        )

                        routing.solver().Add(
                            intervals[task].DurationExpr()
                            == time_dimension.SlackVar(task)
                        )
                routing.solver().Add(
                    routing.solver().Cumulative(
                        [intervals[t] for t in index_tasks],
                        [1 for t in index_tasks],
                        limit,
                        "asset_" + str(i),
                    )
                )
                i += 1
        logger.info("cumulative done")
        if include_mandatory:
            mandatory_nodes = [
                manager.NodeToIndex(self.problem.index_nodes[n])
                for n in self.problem.mandatory_node_info
                if self.problem.mandatory_node_info[n]
                and n
                not in {
                    self.problem.origin_vehicle[v] for v in self.problem.origin_vehicle
                }
                and n
                not in {
                    self.problem.target_vehicle[v] for v in self.problem.target_vehicle
                }
            ]
            if len(mandatory_nodes) > 0:
                routing.AddDisjunction(
                    mandatory_nodes, 1000000000, len(mandatory_nodes)
                )
                routing.solver().Add(
                    routing.solver().Sum(
                        [routing.ActiveVar(i) for i in mandatory_nodes]
                    )
                    == len(mandatory_nodes)
                )
            other_nodes = [
                manager.NodeToIndex(self.problem.index_nodes[n])
                for n in self.problem.mandatory_node_info
                if not self.problem.mandatory_node_info[n]
                and n
                not in {
                    self.problem.origin_vehicle[v] for v in self.problem.origin_vehicle
                }
                and n
                not in {
                    self.problem.target_vehicle[v] for v in self.problem.target_vehicle
                }
            ]
            if len(other_nodes) > 0:
                routing.AddDisjunction(other_nodes, 0, len(other_nodes))
        include_node_vehicle = kwargs.get("include_node_vehicle", True)
        if include_node_vehicle:
            if self.problem.node_vehicle is not None:
                for n in self.problem.node_vehicle:
                    routing.SetAllowedVehiclesForIndex(
                        self.problem.node_vehicle[n],
                        manager.NodeToIndex(self.problem.index_nodes[n]),
                    )

        if "Time" in self.dimension_names:
            time_dimension = routing.GetDimensionOrDie("Time")
            for i in range(self.problem.number_vehicle):
                routing.AddVariableMinimizedByFinalizer(
                    time_dimension.CumulVar(routing.Start(i))
                )
                routing.AddVariableMinimizedByFinalizer(
                    time_dimension.CumulVar(routing.End(i))
                )
        origins = {self.problem.origin_vehicle[v] for v in self.problem.origin_vehicle}
        targets = {self.problem.target_vehicle[v] for v in self.problem.target_vehicle}
        if include_time_windows:
            time_dimension = routing.GetDimensionOrDie("Time")
            for node in self.problem.all_nodes_dict:
                if node in origins:
                    continue
                if node in targets:
                    continue
                index = manager.NodeToIndex(self.problem.index_nodes[node])
                if node in self.problem.time_windows_nodes:
                    mini, maxi = self.problem.time_windows_nodes[node]
                    if mini is not None and maxi is not None:
                        time_dimension.SetCumulVarSoftLowerBound(
                            index, self.factor_multiplier_time * mini, 10000
                        )
                        time_dimension.SetCumulVarSoftUpperBound(
                            index, self.factor_multiplier_time * maxi, 10000
                        )
                    elif mini is not None:
                        time_dimension.SetCumulVarSoftLowerBound(
                            index, self.factor_multiplier_time * mini, 10000
                        )
                    elif maxi is not None:
                        time_dimension.SetCumulVarSoftUpperBound(
                            index, self.factor_multiplier_time * maxi, 10000
                        )

            for vehicle in range(self.problem.number_vehicle):
                index = routing.Start(vehicle)
                if (
                    self.problem.origin_vehicle[vehicle]
                    in self.problem.time_windows_nodes
                ):
                    mini, maxi = self.problem.time_windows_nodes[
                        self.problem.origin_vehicle[vehicle]
                    ]
                    if mini is not None and maxi is not None:
                        time_dimension.CumulVar(index).SetRange(mini, maxi)
                    elif mini is not None:
                        time_dimension.CumulVar(index).SetMin(mini)
                    elif maxi is not None:
                        time_dimension.CumulVar(index).SetMax(maxi)
                index = routing.End(vehicle)
                if (
                    self.problem.target_vehicle[vehicle]
                    in self.problem.time_windows_nodes
                ):
                    mini, maxi = self.problem.time_windows_nodes[
                        self.problem.target_vehicle[vehicle]
                    ]
                    if mini is not None and maxi is not None:
                        time_dimension.CumulVar(index).SetRange(
                            self.factor_multiplier_time * mini,
                            self.factor_multiplier_time * maxi,
                        )
                    elif mini is not None:
                        time_dimension.CumulVar(index).SetMin(
                            self.factor_multiplier_time * mini
                        )
                    elif maxi is not None:
                        time_dimension.CumulVar(index).SetMax(
                            self.factor_multiplier_time * maxi
                        )
        if include_time_windows_cluster:
            time_dimension = routing.GetDimensionOrDie("Time")
            for cluster in list(self.problem.clusters_to_node):
                mini, maxi = self.problem.time_windows_cluster[cluster]
                if mini is not None or maxi is not None:
                    nodes = [
                        manager.NodeToIndex(self.problem.index_nodes[n])
                        for n in self.problem.clusters_to_node[cluster]
                        if n
                        not in {
                            self.problem.origin_vehicle[v]
                            for v in self.problem.origin_vehicle
                        }
                        and n
                        not in {
                            self.problem.target_vehicle[v]
                            for v in self.problem.target_vehicle
                        }
                    ]
                    if mini is not None:
                        routing.solver().Add(
                            routing.solver().Sum(
                                [
                                    routing.ActiveVar(i) * time_dimension.CumulVar(i)
                                    for i in nodes
                                ]
                            )
                            >= mini
                        )
                    if maxi is not None:
                        routing.solver().Add(
                            routing.solver().Sum(
                                [
                                    routing.ActiveVar(i) * time_dimension.CumulVar(i)
                                    for i in nodes
                                ]
                            )
                            <= mini
                        )
        if "Distance" not in self.dimension_names:
            dimension_name = "Distance"
            routing.AddDimension(
                transit_distance_callback_index,
                0,  # no slack
                max_distance_per_vehicle,  # vehicle maximum travel distance
                True,  # start cumul to zero
                dimension_name,
            )
            self.dimension_names += ["Distance"]
        apply_cost(list_parameters_cost=list_parameters_cost, routing=routing)
        self.manager = manager
        self.routing = routing
        self.search_parameters = self.build_search_parameters(**kwargs)
        logger.info("Routing problem initialized ")

    def build_search_parameters(
        self, **kwargs: Any
    ) -> routing_parameters_pb2.RoutingSearchParameters:
        first_solution_strategy = kwargs.get(
            "first_solution_strategy", first_solution_strategy_enum.SAVINGS
        )
        local_search_metaheuristic = kwargs.get(
            "local_search_metaheuristic",
            local_search_metaheuristic_enum.GUIDED_LOCAL_SEARCH,
        )
        one_visit_per_cluster = kwargs.get("one_visit_per_cluster", False)

        use_lns = kwargs.get("use_lns", True)
        use_cp = kwargs.get("use_cp", True)
        use_cp_sat = kwargs.get("use_cp_sat", False)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = first_solution_strategy
        search_parameters.local_search_metaheuristic = local_search_metaheuristic
        if use_lns:
            if one_visit_per_cluster:
                search_parameters.local_search_operators.use_inactive_lns = BOOL_TRUE
            search_parameters.local_search_operators.use_path_lns = BOOL_TRUE
            if one_visit_per_cluster:
                search_parameters.local_search_operators.use_extended_swap_active = (
                    BOOL_TRUE
                )
        search_parameters.use_cp = BOOL_TRUE if use_cp else BOOL_FALSE
        search_parameters.use_cp_sat = BOOL_TRUE if use_cp_sat else BOOL_FALSE
        search_parameters.number_of_solutions_to_collect = kwargs.get(
            "n_solutions", 100
        )
        search_parameters.time_limit.seconds = kwargs.get("time_limit", 100)
        return search_parameters

    def solve_intern(
        self,
        search_parameters: Optional[
            routing_parameters_pb2.RoutingSearchParameters
        ] = None,
        **kwargs: Any,
    ) -> Iterable[
        Tuple[
            Dict[int, List[int]],
            Dict[Tuple[int, int, NodePosition], Dict[str, Tuple[float, float, float]]],
            float,
            float,
            float,
        ]
    ]:
        if search_parameters is None:
            search_parameters = self.search_parameters
        sols = []  # useful for callback
        callback = make_routing_monitor(self)
        self.routing.AddAtSolutionCallback(callback)
        sols = self.routing.SolveWithParameters(
            search_parameters
        )  # useful for callback
        return callback.sols

    def solve(self, **kwargs: Any) -> ResultStorage:
        solutions_fit: List[Tuple[Solution, fitness_class]] = []
        sols = self.solve_intern()
        for sol in sols:
            gpdp_sol = convert_to_gpdpsolution(problem=self.problem, sol=sol)
            fit = self.aggreg_sol(gpdp_sol)
            solutions_fit.append((gpdp_sol, fit))
        return ResultStorage(
            list_solution_fits=solutions_fit,
            limit_store=False,
            mode_optim=self.params_objective_function.sense_function,
        )


class RoutingMonitor:
    def __init__(self, solver: ORToolsGPDP):
        self.model = solver.routing
        self.problem = solver.problem
        self.solver = solver
        self._counter = 0
        self._best_objective = np.inf
        self._counter_limit = 10000000
        self.nb_solutions = 0
        self.sols: List[
            Tuple[
                Dict[int, List[int]],
                Dict[
                    Tuple[int, int, NodePosition],
                    Dict[str, Tuple[float, float, float]],
                ],
                float,
                float,
                float,
            ]
        ] = []

    def __call__(self) -> None:
        logger.debug(
            f"New solution found : --Cur objective : {self.model.CostVar().Max()}"
        )
        logger.debug(status_description[self.model.status()])
        if self.nb_solutions % 100 == 0:
            self.retrieve_current_solution()
        if self.model.CostVar().Max() < self._best_objective:
            self._best_objective = self.model.CostVar().Max()
            self.retrieve_current_solution()
            self._counter = 0
        else:
            self._counter += 1
            if self._counter > self._counter_limit:
                self.model.solver().FinishCurrentSearch()
        self.nb_solutions += 1

    def retrieve_current_solution(self) -> None:
        postpro_sol: List[
            Tuple[
                Dict[int, List[int]],
                Dict[
                    Tuple[int, int, NodePosition],
                    Dict[str, Tuple[float, float, float]],
                ],
                float,
                float,
                float,
            ]
        ] = []
        vehicle_count = self.problem.number_vehicle
        vehicle_tours: Dict[int, List[int]] = {i: [] for i in range(vehicle_count)}
        dimension_output: Dict[
            Tuple[int, int, NodePosition], Dict[str, Tuple[float, float, float]]
        ] = {}
        dimensions_names = self.solver.dimension_names
        dimensions: Dict[str, pywrapcp.RoutingDimension] = {
            r: self.model.GetDimensionOrDie(r) for r in dimensions_names
        }
        objective = 0.0
        route_distance = 0.0
        for vehicle_id in range(vehicle_count):
            index = self.model.Start(vehicle_id)
            route_load: Dict[str, float] = {r: 0.0 for r in self.problem.resources_set}
            cnt = 0
            while not self.model.IsEnd(index) or cnt > 10000:
                node_index = self.solver.manager.IndexToNode(index)
                if cnt == 0:
                    node_position = NodePosition.START
                else:
                    node_position = NodePosition.INTERMEDIATE
                vehicle_tours[vehicle_id] += [node_index]
                try:
                    dimension_output[(vehicle_id, node_index, node_position)] = {
                        r: (
                            dimensions[r].CumulVar(index).Min(),
                            dimensions[r].CumulVar(index).Max(),
                            dimensions[r].SlackVar(index).Value(),
                        )
                        for r in dimensions
                    }
                except Exception as e:
                    logger.warning(("1,", e))
                    break
                cnt += 1
                for r in route_load:
                    route_load[r] += self.solver.demands[r][node_index]
                previous_index = index
                try:
                    index = self.model.NextVar(index).Value()
                except Exception as e:
                    logger.warning(("3", e))
                    break
                route_distance += self.model.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
                if self.model.IsEnd(index):
                    vehicle_tours[vehicle_id] += [
                        self.solver.manager.IndexToNode(index)
                    ]
                    try:
                        dimension_output[
                            (
                                vehicle_id,
                                self.solver.manager.IndexToNode(index),
                                NodePosition.END,
                            )
                        ] = {
                            r: (
                                dimensions[r].CumulVar(index).Min(),
                                dimensions[r].CumulVar(index).Max(),
                                0,
                            )
                            for r in dimensions
                        }
                    except Exception as e:
                        logger.warning(("2,", e))
                        break
        postpro_sol += [
            (
                vehicle_tours,
                dimension_output,
                route_distance,
                objective,
                self.model.CostVar().Max(),
            )
        ]
        self.sols += postpro_sol


def make_routing_monitor(solver: ORToolsGPDP) -> RoutingMonitor:

    return RoutingMonitor(solver)


def convert_to_gpdpsolution(
    problem: GPDP,
    sol: Tuple[
        Dict[int, List[int]],
        Dict[Tuple[int, int, NodePosition], Dict[str, Tuple[float, float, float]]],
        float,
        float,
        float,
    ],
) -> GPDPSolution:
    (
        vehicle_tours,
        dimension_output,
        route_distance,
        objective,
        cost,
    ) = sol
    times: Dict[Node, float] = {}
    trajectories = {
        v: [problem.list_nodes[nodeindex] for nodeindex in traj]
        for v, traj in vehicle_tours.items()
    }
    for (v, nodeindex, node_position), output in dimension_output.items():
        node = problem.list_nodes[nodeindex]
        if (node not in times) or (node_position != NodePosition.START):
            if "Time" in output:
                times[node] = output["Time"][1]
    resource_evolution: Dict[Node, Dict[Node, List[int]]] = {}
    return GPDPSolution(
        problem=problem,
        trajectories=trajectories,
        times=times,
        resource_evolution=resource_evolution,
    )


def plot_ortools_solution(
    result: Tuple[
        Dict[int, List[int]],
        Dict[Tuple[int, int, NodePosition], Dict[str, Tuple[float, float, float]]],
        float,
        float,
        float,
    ],
    problem: GPDP,
) -> Tuple[Figure, Axes]:
    if problem.coordinates_2d is None:
        raise ValueError(
            "problem.coordinates_2d cannot be None when calling plot_ortools_solution."
        )
    (
        vehicle_tours,
        dimension_output,
        route_distance,
        objective,
        cost,
    ) = result
    fig, ax = plt.subplots(1)
    nb_colors = problem.number_vehicle
    nb_colors_clusters = len(problem.clusters_set)
    colors_nodes = plt.cm.get_cmap("hsv", nb_colors_clusters)
    ax.scatter(
        [problem.coordinates_2d[node][0] for node in problem.clusters_dict],
        [problem.coordinates_2d[node][1] for node in problem.clusters_dict],
        s=1,
        color=[
            colors_nodes(problem.clusters_dict[node]) for node in problem.clusters_dict
        ],
    )
    for v in range(len(vehicle_tours)):
        ax.plot(
            [
                problem.coordinates_2d[problem.list_nodes[n]][0]
                for n in vehicle_tours[v]
            ],
            [
                problem.coordinates_2d[problem.list_nodes[n]][1]
                for n in vehicle_tours[v]
            ],
            label="vehicle n°" + str(v),
        )
        ax.scatter(
            [
                problem.coordinates_2d[problem.list_nodes[n]][0]
                for n in vehicle_tours[v]
            ],
            [
                problem.coordinates_2d[problem.list_nodes[n]][1]
                for n in vehicle_tours[v]
            ],
            s=10,
            color=[
                colors_nodes(problem.clusters_dict[problem.list_nodes[n]])
                for n in vehicle_tours[v]
            ],
        )
    ax.legend()
    return fig, ax
