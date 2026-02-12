#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Dict, Optional

try:
    import optalcp as cp
except ImportError:
    cp = None

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.vrptw.problem import (
    Task,
    VRPTWProblem,
    VRPTWSolution,
)

logger = logging.getLogger(__name__)


class OptalVRPTWSolver(SchedulingOptalSolver[Task]):
    """
    Optal solver for the Vehicle Routing Problem with Time Windows (VRPTW).

    """

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["intervals_per_node"][task]

    problem: VRPTWProblem

    def __init__(
        self,
        problem: VRPTWProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables: Dict[str, Any] = {}
        self.scaling_factor = None  # Scale distances and times to use integers
        self.distance_matrix = self.problem.distance_matrix

    def init_model(self, scaling: int, cost_per_vehicle: int, **kwargs: Any) -> None:
        """Initialise the CP-SAT model."""
        self.cp_model = cp.Model()
        self.scaling_factor = scaling
        self.distance_matrix = [
            [
                int(self.scaling_factor * self.distance_matrix[i, j])
                for j in range(self.distance_matrix.shape[1])
            ]
            for i in range(self.distance_matrix.shape[0])
        ]

        k = self.problem.nb_vehicles
        depot = self.problem.depot_node
        intervals_per_vehicle = {}
        intervals_distance_per_vehicle = {}
        intervals_distance_come_back = {}
        intervals_come_back = {}
        intervals_per_node = {}
        used_vehicle = {}
        for vehicle in range(k):
            intervals_per_vehicle[vehicle] = {}
            intervals_distance_per_vehicle[vehicle] = {}
            for node in self.problem.customers:
                lb = max(
                    int(self.scaling_factor * self.problem.time_windows[node][0]),
                    self.distance_matrix[depot][node],
                )
                ub = max(
                    int(self.scaling_factor * self.problem.time_windows[node][1]),
                    self.distance_matrix[depot][node],
                )
                service_time = int(
                    self.scaling_factor * self.problem.service_times[node]
                )
                intervals_per_vehicle[vehicle][node] = self.cp_model.interval_var(
                    start=(lb, ub),
                    end=(lb + service_time, ub + service_time),
                    length=service_time,
                    optional=True,
                    name=f"visit_{node}_{vehicle}",
                )
                intervals_distance_per_vehicle[vehicle][node] = (
                    self.cp_model.interval_var(
                        start=(self.distance_matrix[depot][node], None),
                        end=(self.distance_matrix[depot][node], None),
                        length=0,
                        optional=True,
                        name=f"visitd_{node}_{vehicle}",
                    )
                )
                self.cp_model.enforce(
                    self.cp_model.presence(intervals_per_vehicle[vehicle][node])
                    == self.cp_model.presence(
                        intervals_distance_per_vehicle[vehicle][node]
                    )
                )
            lb = int(self.scaling_factor * self.problem.time_windows[depot][0])
            ub = int(self.scaling_factor * self.problem.time_windows[depot][1])
            intervals_come_back[vehicle] = self.cp_model.interval_var(
                start=(lb, ub), end=(lb, ub), length=0, optional=True
            )
            intervals_distance_come_back[vehicle] = self.cp_model.interval_var(
                start=(0, None), end=(0, None), length=0, optional=True
            )
            used_vehicle[vehicle] = self.cp_model.max(
                [
                    self.cp_model.presence(intervals_per_vehicle[vehicle][node])
                    for node in intervals_per_vehicle[vehicle]
                ]
            )
            self.cp_model.enforce(
                self.cp_model.presence(intervals_come_back[vehicle])
                == used_vehicle[vehicle]
            )
            self.cp_model.enforce(
                self.cp_model.presence(intervals_distance_come_back[vehicle])
                == used_vehicle[vehicle]
            )
            for node in intervals_per_vehicle[vehicle]:
                self.cp_model.end_before_start(
                    intervals_per_vehicle[vehicle][node], intervals_come_back[vehicle]
                )
                self.cp_model.end_before_start(
                    intervals_distance_per_vehicle[vehicle][node],
                    intervals_distance_come_back[vehicle],
                )

        for node in self.problem.customers:
            lb = int(self.scaling_factor * self.problem.time_windows[node][0])
            ub = int(self.scaling_factor * self.problem.time_windows[node][1])
            service_time = int(self.scaling_factor * self.problem.service_times[node])
            intervals_per_node[node] = self.cp_model.interval_var(
                start=(lb, ub),
                end=(lb, ub),
                length=service_time,
                optional=False,
                name=f"visit_{node}",
            )
            self.cp_model.alternative(
                intervals_per_node[node],
                [intervals_per_vehicle[v][node] for v in intervals_per_vehicle],
            )

        for v in range(self.problem.nb_vehicles):
            seq = self.cp_model.sequence_var(
                [self.cp_model.interval_var(start=0, end=0, length=0, optional=False)]
                + [intervals_per_vehicle[v][node] for node in self.problem.customers]
                + [intervals_come_back[v]],
                types=[depot] + self.problem.customers + [depot],
            )
            seq_2 = self.cp_model.sequence_var(
                [self.cp_model.interval_var(start=0, end=0, length=0, optional=False)]
                + [
                    intervals_distance_per_vehicle[v][node]
                    for node in self.problem.customers
                ]
                + [intervals_distance_come_back[v]],
                types=[depot] + self.problem.customers + [depot],
            )
            self.cp_model._same_sequence(seq, seq_2)
            self.cp_model.no_overlap(seq, self.distance_matrix)
            self.cp_model.no_overlap(seq_2, self.distance_matrix)
            loads = self.cp_model.sum(
                [
                    int(self.problem.demands[n])
                    * self.cp_model.presence(intervals_per_vehicle[v][n])
                    for n in self.problem.customers
                ]
            )
            self.cp_model.enforce(loads <= int(self.problem.vehicle_capacity))

        self.variables["nb_vehicles"] = self.cp_model.sum(
            [used_vehicle[v] for v in used_vehicle]
        )
        self.variables["total_distance"] = self.cp_model.sum(
            [
                self.cp_model.guard(
                    self.cp_model.end(intervals_distance_come_back[v]), 0
                )
                for v in used_vehicle
            ]
        )
        self.variables["intervals_per_vehicle"] = intervals_per_vehicle
        self.variables["intervals_per_node"] = intervals_per_node
        self.cp_model.minimize(
            self.variables["nb_vehicles"] * cost_per_vehicle
            + self.variables["total_distance"]
        )

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        routes = []
        for v in range(self.problem.nb_vehicles):
            all_intervals = []
            for n in self.problem.customers:
                if result.solution.is_present(
                    self.variables["intervals_per_vehicle"][v][n]
                ):
                    st, end = result.solution.get_value(
                        self.variables["intervals_per_vehicle"][v][n]
                    )
                    all_intervals.append((st, end, n))
            sorted_intervals = sorted(all_intervals, key=lambda x: x[0])
            routes.append([x[2] for x in sorted_intervals])
            cumul_time = 0
            current_node = self.problem.depot_node
            for n in routes[-1]:
                cumul_time += self.distance_matrix[current_node][n]
                current_node = n
        sol = VRPTWSolution(problem=self.problem, routes=routes)
        # print(result.solution.get_value(self.variables["nb_vehicles"]))
        # print(result.solution.get_value(self.variables["total_distance"]))
        print(self.problem.evaluate(sol))
        return sol
