#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Optional

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver
from discrete_optimization.vrp.solver.vrp_solver import SolverVrp
from discrete_optimization.vrp.vrp_model import VrpProblem, VrpSolution
from discrete_optimization.vrp.vrp_toolbox import compute_length_matrix

logger = logging.getLogger(__name__)


class CpSatVrpSolver(OrtoolsCPSatSolver, SolverVrp, WarmstartMixin):
    problem: VrpProblem
    hyperparameters = [
        CategoricalHyperparameter(
            name="cut_transition", choices=[True, False], default=False
        ),
        IntegerHyperparameter(name="nb_cut", low=1, high=100, default=None),
        CategoricalHyperparameter(
            name="optional_node", choices=[True, False], default=False
        ),
    ]

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.closest, self.distance_matrix = compute_length_matrix(self.problem)
        for k in range(self.problem.vehicle_count):
            self.distance_matrix[
                self.problem.end_indexes[k], self.problem.start_indexes[k]
            ] = 0

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

            for (i, j) in arc_literals_vehicle:
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
                        f"{vehicle}, {i,j}"
                    )
                    self.cp_model.Add(lit == hints[i, j]).OnlyEnforceIf(
                        self.variables["to_hint"][vehicle][i, j]
                    )
                    self.cp_model.Add(lit == (not hints[i, j])).OnlyEnforceIf(
                        self.variables["to_hint"][vehicle][i, j].Not()
                    )
            all_hints[vehicle] = hints
        if debug_mode:
            self.cp_model.Maximize(
                sum(
                    self.variables["to_hint"][x][y]
                    for x in self.variables["to_hint"]
                    for y in self.variables["to_hint"][x]
                )
            )
            self.all_hints = all_hints
        return all_hints

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        logger.info(
            f"obj value ={cpsolvercb.ObjectiveValue()} bound {cpsolvercb.BestObjectiveBound()}"
        )
        if "to_hint" in self.variables:
            for v in self.variables["to_hint"]:
                for x in self.variables["to_hint"][v]:
                    if not cpsolvercb.boolean_value(self.variables["to_hint"][v][x]):
                        logger.debug(f"{v, x} not put to expected value")
                        logger.debug(f"should be {self.all_hints[v][x]}")
        if self.variables["optional_node"]:
            logger.info(
                f"Nb nodes visited = {cpsolvercb.Value(self.variables['nb_nodes'])}"
            )
        try:
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
                            route_distance += self.distance_matrix[current_node, i]
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
        except Exception as e:
            logger.error(f"Problem {e}")
        return sol

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        model = CpModel()
        """Entry point of the program."""
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
            demand_coeffs = []
            arcs = []
            arc_non_loop = []
            arc_literals_per_vehicles[vehicle] = {}
            for i in all_nodes:
                for j in all_nodes:
                    if args["cut_transition"]:
                        if j not in list(self.closest[i, : args["nb_cut"]]):
                            continue
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
                    obj_coeffs.append(int(self.distance_matrix[i, j]))
                    if i != j:
                        demand_coeffs.append(int(self.problem.customers[j].demand))
                    else:
                        demand_coeffs.append(0)
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
                    == True
                )
            dist_path_per_vehicle[vehicle] = sum(
                obj_vars[i] * obj_coeffs[i] for i in range(len(obj_vars))
            )
            model.Add(
                sum(obj_vars[i] * demand_coeffs[i] for i in range(len(obj_vars)))
                <= self.problem.vehicle_capacities[vehicle]
            )
        for j in ingoing_arc_per_node:
            if args["optional_node"]:
                model.AddAtMostOne(ingoing_arc_per_node[j])
            else:
                model.AddExactlyOne(ingoing_arc_per_node[j])
        self.variables["arc_literals_per_vehicles"] = arc_literals_per_vehicles
        self.variables["ingoing_arc_per_node"] = ingoing_arc_per_node
        self.variables["visited"] = visited_per_vehicle
        self.cp_model = model
        self.variables["optional_node"] = args["optional_node"]
        if args["optional_node"]:
            self.variables["nb_nodes"] = sum(
                [x for j in ingoing_arc_per_node for x in ingoing_arc_per_node[j]]
            )
            self.cp_model.Minimize(
                -10000 * self.variables["nb_nodes"]
                + sum(
                    dist_path_per_vehicle[vehicle] for vehicle in dist_path_per_vehicle
                )
            )

        else:
            self.cp_model.Minimize(
                sum(dist_path_per_vehicle[vehicle] for vehicle in dist_path_per_vehicle)
            )
