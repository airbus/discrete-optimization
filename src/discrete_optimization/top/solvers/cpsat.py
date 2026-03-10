#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import math
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.top.problem import TeamOrienteeringProblem, VrpSolution
from discrete_optimization.vrp.utils import compute_length_matrix

logger = logging.getLogger(__name__)


class CpsatTopSolver(OrtoolsCpSatSolver, WarmstartMixin):
    problem: TeamOrienteeringProblem

    def __init__(
        self,
        problem: TeamOrienteeringProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        _, self.distance = compute_length_matrix(self.problem)
        self.distance[self.problem.start_indexes[0], self.problem.end_indexes[0]] = 0
        self.distance[self.problem.end_indexes[0], self.problem.start_indexes[0]] = 0
        self.variables = {}

    def init_model(self, scaling: float = 1, **kwargs: Any) -> None:
        # Inspired by the CVRP model
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        super().init_model(**kwargs)
        distance_matrix = (scaling * self.distance).astype(int)
        model = self.cp_model
        num_nodes = self.problem.customer_count
        all_nodes = range(num_nodes)
        # Create the circuit constraint.
        visited_per_vehicle = {}
        arc_literals_per_vehicles = {}
        dist_path_per_vehicle = {}
        ingoing_arc_per_node = {}
        non_dummy_nodes = set(
            n
            for n in all_nodes
            if n not in self.problem.start_indexes and n not in self.problem.end_indexes
        )
        for vehicle in range(self.problem.vehicle_count):
            obj_vars = []
            obj_coeffs = []
            arcs = []
            arc_non_loop = []
            arc_literals_per_vehicles[vehicle] = {}
            for i in all_nodes:
                for j in all_nodes:
                    lit = model.new_bool_var(
                        "vehicle %i : %i follows %i" % (vehicle, j, i)
                    )
                    arcs.append((i, j, lit))
                    arc_literals_per_vehicles[vehicle][i, j] = lit
                    if i != j:
                        if j in non_dummy_nodes:
                            if j not in ingoing_arc_per_node:
                                ingoing_arc_per_node[j] = []
                            ingoing_arc_per_node[j].append(lit)
                        if not (
                            (i, j)
                            in {
                                (
                                    self.problem.end_indexes[vehicle],
                                    self.problem.start_indexes[vehicle],
                                ),
                                (
                                    self.problem.start_indexes[vehicle],
                                    self.problem.end_indexes[vehicle],
                                ),
                            }
                        ):
                            arc_non_loop.append(lit)
                        obj_vars.append(lit)
                        obj_coeffs.append(distance_matrix[i, j])
            model.add_circuit(arcs)
            visited = model.NewBoolVar(f"non_nul_trip_{vehicle}")
            visited_per_vehicle[vehicle] = visited
            for lit in arc_non_loop:
                model.Add(visited >= lit)
            model.Add(
                arc_literals_per_vehicles[vehicle][
                    self.problem.start_indexes[vehicle],
                    self.problem.end_indexes[vehicle],
                ]
                == visited.Not()
            )
            if self.problem.end_indexes[vehicle] != self.problem.start_indexes[vehicle]:
                model.Add(
                    arc_literals_per_vehicles[vehicle][
                        self.problem.end_indexes[vehicle],
                        self.problem.start_indexes[vehicle],
                    ]
                    == 1
                )
            dist_path_per_vehicle[vehicle] = sum(
                obj_vars[i] * obj_coeffs[i] for i in range(len(obj_vars))
            )
            # Maximum distance.
            model.add(
                dist_path_per_vehicle[vehicle]
                <= int(math.floor(self.problem.max_length_tours * scaling))
            )
        for j in ingoing_arc_per_node:
            model.AddAtMostOne(ingoing_arc_per_node[j])

        reward = sum(
            [
                lit * self.problem.customers[j].reward
                for j in ingoing_arc_per_node
                for lit in ingoing_arc_per_node[j]
            ]
        )
        self.variables["reward"] = reward
        self.variables["arc_literals_per_vehicles"] = arc_literals_per_vehicles
        self.variables["ingoing_arc_per_node"] = ingoing_arc_per_node
        self.variables["visited"] = visited_per_vehicle
        self.variables["nb_nodes"] = sum(
            [x for j in ingoing_arc_per_node for x in ingoing_arc_per_node[j]]
        )
        self.cp_model.Maximize(self.variables["reward"])

    def set_warm_start(self, solution: VrpSolution, debug_mode: bool = False) -> None:
        self.cp_model.clear_hints()
        arc_literals_per_vehicles = self.variables["arc_literals_per_vehicles"]
        all_hints = {v: {} for v in arc_literals_per_vehicles}
        if debug_mode:
            self.variables["to_hint"] = {v: {} for v in arc_literals_per_vehicles}
        for vehicle, arc_literals_vehicle in arc_literals_per_vehicles.items():
            hints = {}
            path = solution.list_paths[vehicle]
            start_index = self.problem.start_indexes[vehicle]
            end_index = self.problem.end_indexes[vehicle]
            for i, j in arc_literals_vehicle:
                if i == j:
                    hints[i, j] = 1
                else:
                    hints[i, j] = 0
            if len(path) > 0:
                hints[start_index, start_index] = 0
            current_node = start_index
            for next_node in path:
                hints[current_node, next_node] = 1
                hints[next_node, next_node] = 0
                current_node = next_node
            if len(path) > 0:
                # end the loop
                last_node = path[-1]
                if end_index not in path:
                    # end node not in last 2 nodes of permutation
                    hints[last_node, end_index] = 1
                    last_node = end_index
                    if end_index not in path and last_node != start_index:
                        # close the cycle
                        hints[last_node, start_index] = 1
                self.cp_model.AddHint(self.variables["visited"][vehicle], 1)
            else:
                self.cp_model.AddHint(self.variables["visited"][vehicle], 0)
            for (i, j), lit in arc_literals_vehicle.items():
                self.cp_model.AddHint(lit, hints[i, j])
                if debug_mode:
                    self.variables["to_hint"][vehicle][i, j] = self.cp_model.NewBoolVar(
                        f"{vehicle}, {i, j}"
                    )
                    self.cp_model.Add(lit == hints[i, j]).OnlyEnforceIf(
                        self.variables["to_hint"][vehicle][i, j]
                    )
                    self.cp_model.Add(lit == (not hints[i, j])).OnlyEnforceIf(
                        self.variables["to_hint"][vehicle][i, j].Not()
                    )
            all_hints[vehicle] = hints
        return all_hints

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> VrpSolution:
        logger.info(
            f"obj value ={cpsolvercb.ObjectiveValue()} bound {cpsolvercb.BestObjectiveBound()}"
        )
        list_paths = []
        for vehicle in range(self.problem.vehicle_count):
            current_node = self.problem.start_indexes[vehicle]
            route_is_finished = False
            path = []
            route_distance = 0
            for arc in self.variables["arc_literals_per_vehicles"][vehicle]:
                if cpsolvercb.boolean_value(
                    self.variables["arc_literals_per_vehicles"][vehicle][arc]
                ):
                    logger.debug(f"Vehicle {vehicle} from {arc[0]} to {arc[1]}")
            while not route_is_finished:
                for i in range(self.problem.customer_count):
                    if i == current_node:
                        continue
                    if (current_node, i) not in self.variables[
                        "arc_literals_per_vehicles"
                    ][vehicle]:
                        continue
                    if cpsolvercb.boolean_value(
                        self.variables["arc_literals_per_vehicles"][vehicle][
                            current_node, i
                        ]
                    ):
                        route_distance += self.distance[current_node, i]
                        current_node = i
                        if current_node == self.problem.end_indexes[vehicle]:
                            route_is_finished = True
                        break
                if current_node == self.problem.end_indexes[vehicle]:
                    break
                if not route_is_finished:
                    path.append(current_node)
            list_paths.append(path)
        sol = VrpSolution(
            problem=self.problem,
            list_start_index=self.problem.start_indexes,
            list_end_index=self.problem.end_indexes,
            list_paths=list_paths,
        )
        evaluation = self.problem.evaluate(sol)
        logger.info(f"Recomputed Sol = {evaluation}")
        return sol
