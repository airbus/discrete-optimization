#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from VRP to GPDP (General Pickup and Delivery Problem)."""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    subset_transformation,
)
from discrete_optimization.gpdp.problem import GpdpProblem, GpdpSolution, Node
from discrete_optimization.vrp.problem import Customer2DVrpProblem, VrpSolution


class VrpToGpdpTransformation(
    ProblemTransformation[Customer2DVrpProblem, VrpSolution, GpdpProblem, GpdpSolution]
):
    """Transform VRP to GPDP.

    Mapping:
    - VRP vehicles → GPDP vehicles
    - VRP customers → GPDP transportation nodes
    - VRP start/end depots → GPDP origin/target nodes (virtual)
    - VRP capacity → GPDP resource flow (demand)
    - VRP distances → GPDP distance matrix

    VRP is a special case of GPDP where:
    - No pickup-delivery pairs
    - Single resource (demand/capacity)
    - No time windows (use VRPTW → GPDP for time windows)
    """

    def __init__(self, compute_graph: bool = False):
        """Initialize transformation.

        Args:
            compute_graph: Whether to compute the GPDP graph (default: False)

        """
        self.compute_graph = compute_graph

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward transformation (VRP → GPDP)."""
        return subset_transformation(
            use_cases=[
                "Use GPDP solvers for VRP problems",
                "VRP is a special case of GPDP (no pickup-delivery pairs)",
                "Benchmark GPDP algorithms on VRP instances",
            ],
            assumptions=[
                "No pickup-delivery constraints",
                "Single resource dimension (demand/capacity)",
                "No time windows",
            ],
        )

    def transform_problem(self, source_problem: Customer2DVrpProblem) -> GpdpProblem:
        """Transform VRP to GPDP.

        Args:
            source_problem: VRP problem instance

        Returns:
            Equivalent GPDP problem

        """
        nb_vehicle = source_problem.vehicle_count
        nb_customers = len(source_problem.customers)

        # Identify start and end depot indices
        all_start_index = set(source_problem.start_indexes)
        all_end_index = set(source_problem.end_indexes)

        # Real customers (exclude depot nodes)
        clients_node = [
            (i, source_problem.customers[i])
            for i in range(nb_customers)
            if i not in all_start_index and i not in all_end_index
        ]

        # Create virtual nodes for vehicle origins and targets
        # Virtual origin nodes: one per vehicle
        virtual_origin_nodes = [len(clients_node) + k for k in range(nb_vehicle)]

        # Virtual target nodes: one per vehicle
        virtual_target_nodes = [
            len(clients_node) + nb_vehicle + k for k in range(nb_vehicle)
        ]

        # Mapping: virtual node → original depot index
        virtual_to_depot = {
            virtual_origin_nodes[i]: source_problem.start_indexes[i]
            for i in range(nb_vehicle)
        }
        virtual_to_depot.update(
            {
                virtual_target_nodes[i]: source_problem.end_indexes[i]
                for i in range(nb_vehicle)
            }
        )

        # Mapping: GPDP customer node → VRP customer index
        gpdp_to_vrp = {j: clients_node[j][0] for j in range(len(clients_node))}

        # Vehicle origin and target assignments
        origin_vehicle: dict[int, Node] = {
            i: virtual_origin_nodes[i] for i in range(nb_vehicle)
        }
        target_vehicle: dict[int, Node] = {
            i: virtual_target_nodes[i] for i in range(nb_vehicle)
        }

        # All GPDP nodes
        all_gpdp_nodes = (
            list(range(len(clients_node))) + virtual_origin_nodes + virtual_target_nodes
        )

        # Build distance and time matrices
        distance_delta: dict[Node, dict[Node, float]] = {}
        time_delta: dict[Node, dict[Node, float]] = {}
        resources_flow_node: dict[Node, dict[str, float]] = {}
        coordinates: dict[Node, tuple[float, float]] = {}

        for node1 in all_gpdp_nodes:
            distance_delta[node1] = {}
            time_delta[node1] = {}

            for node2 in all_gpdp_nodes:
                # Map GPDP nodes to VRP indices
                vrp_idx1 = (
                    gpdp_to_vrp[node1]
                    if node1 in gpdp_to_vrp
                    else virtual_to_depot[node1]
                )
                vrp_idx2 = (
                    gpdp_to_vrp[node2]
                    if node2 in gpdp_to_vrp
                    else virtual_to_depot[node2]
                )

                dist = source_problem.evaluate_function_indexes(vrp_idx1, vrp_idx2)
                distance_delta[node1][node2] = dist
                time_delta[node1][node2] = dist  # Assume time = distance

        # Coordinates
        for node in all_gpdp_nodes:
            vrp_idx = (
                gpdp_to_vrp[node] if node in gpdp_to_vrp else virtual_to_depot[node]
            )
            customer = source_problem.customers[vrp_idx]
            coordinates[node] = (customer.x, customer.y)

            # Resource flow: negative demand for customers, positive capacity for origins
            resources_flow_node[node] = {"demand": -customer.demand}

        # Vehicle origins have positive capacity
        for v in range(nb_vehicle):
            resources_flow_node[origin_vehicle[v]]["demand"] = (
                source_problem.vehicle_capacities[v]
            )

        # Resource and capacity definitions
        resources_set = {"demand"}
        capacities = {
            i: {"demand": (0.0, source_problem.vehicle_capacities[i])}
            for i in range(nb_vehicle)
        }

        # Edge resource flow (all zero for VRP)
        resources_flow_edges = {
            (x, y): {"demand": 0.0} for x in distance_delta for y in distance_delta[x]
        }

        # No pickup-delivery pairs in VRP
        list_pickup_deliverable = []

        # Nodes sets
        nodes_transportation = set(gpdp_to_vrp.keys())
        nodes_origin = set(virtual_to_depot.keys()) & set(virtual_origin_nodes)
        nodes_target = set(virtual_to_depot.keys()) & set(virtual_target_nodes)

        return GpdpProblem(
            number_vehicle=nb_vehicle,
            nodes_transportation=nodes_transportation,
            nodes_origin=nodes_origin,
            nodes_target=nodes_target,
            list_pickup_deliverable=list_pickup_deliverable,
            origin_vehicle=origin_vehicle,
            target_vehicle=target_vehicle,
            resources_set=resources_set,
            capacities=capacities,
            resources_flow_node=resources_flow_node,
            resources_flow_edges=resources_flow_edges,
            distance_delta=distance_delta,
            time_delta=time_delta,
            coordinates_2d=coordinates,
            compute_graph=self.compute_graph,
        )

    def back_transform_solution(
        self, solution: GpdpSolution, source_problem: Customer2DVrpProblem
    ) -> VrpSolution:
        """Transform GPDP solution back to VRP solution.

        Args:
            solution: GPDP solution
            source_problem: Original VRP problem

        Returns:
            Equivalent VRP solution

        """
        # Extract routes from GPDP trajectories
        # GPDP trajectories: dict[vehicle_id, list[Node]]
        # VRP list_paths: list[list[int]] (customer indices)

        # Build reverse mapping: GPDP node → VRP customer index
        # This needs to be reconstructed from the transformation
        # For now, simplified approach: assume GPDP customer nodes map directly
        nb_customers = len(source_problem.customers)
        all_start_index = set(source_problem.start_indexes)
        all_end_index = set(source_problem.end_indexes)

        clients_node = [
            (i, source_problem.customers[i])
            for i in range(nb_customers)
            if i not in all_start_index and i not in all_end_index
        ]
        gpdp_to_vrp = {j: clients_node[j][0] for j in range(len(clients_node))}

        # Extract paths for each vehicle
        list_paths = []
        for v in range(source_problem.vehicle_count):
            if v in solution.trajectories:
                trajectory = solution.trajectories[v]
                # Filter out origin/target nodes, keep only customer nodes
                vrp_path = [
                    gpdp_to_vrp[node] for node in trajectory if node in gpdp_to_vrp
                ]
                list_paths.append(vrp_path)
            else:
                list_paths.append([])

        return VrpSolution(
            problem=source_problem,
            list_start_index=list(source_problem.start_indexes),
            list_end_index=list(source_problem.end_indexes),
            list_paths=list_paths,
        )

    def forward_transform_solution(
        self, solution: VrpSolution, target_problem: GpdpProblem
    ) -> Optional[GpdpSolution]:
        """Transform VRP solution to GPDP solution (for warmstart).

        Args:
            solution: VRP solution
            target_problem: Target GPDP problem

        Returns:
            Equivalent GPDP solution

        """
        # Build GPDP trajectories from VRP paths
        # Need to map VRP customer indices to GPDP nodes
        nb_customers = len(solution.problem.customers)
        all_start_index = set(solution.problem.start_indexes)
        all_end_index = set(solution.problem.end_indexes)

        clients_node = [
            (i, solution.problem.customers[i])
            for i in range(nb_customers)
            if i not in all_start_index and i not in all_end_index
        ]
        vrp_to_gpdp = {clients_node[j][0]: j for j in range(len(clients_node))}

        trajectories = {}
        for v in range(solution.problem.vehicle_count):
            # Convert VRP path to GPDP trajectory
            gpdp_path = [target_problem.origin_vehicle[v]]  # Start with origin node
            gpdp_path.extend(
                [vrp_to_gpdp[vrp_idx] for vrp_idx in solution.list_paths[v]]
            )
            gpdp_path.append(target_problem.target_vehicle[v])  # End with target node
            trajectories[v] = gpdp_path

        # Compute times (simplified: cumulative distance)
        times = {}
        for v, traj in trajectories.items():
            current_time = 0.0
            for i, node in enumerate(traj):
                times[node] = current_time
                if i < len(traj) - 1:
                    next_node = traj[i + 1]
                    current_time += target_problem.distance_delta[node][next_node]

        # Resource evolution (simplified: empty for now)
        resource_evolution = {}

        return GpdpSolution(
            problem=target_problem,
            trajectories=trajectories,
            times=times,
            resource_evolution=resource_evolution,
        )
