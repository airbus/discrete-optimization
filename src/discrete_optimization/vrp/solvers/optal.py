#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Optional

try:
    import optalcp as cp
except ImportError:
    cp = None

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)
from discrete_optimization.vrp.problem import VrpProblem, VrpSolution
from discrete_optimization.vrp.utils import compute_length_matrix

logger = logging.getLogger(__name__)


class OptalVrpSolver(OptalCpSolver):
    problem: VrpProblem

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

    def init_model(self, scale: int = 10, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        self.cp_model = cp.Model()
        self.distance_matrix = [
            [
                int(scale * self.distance_matrix[i, j])
                for j in range(self.distance_matrix.shape[1])
            ]
            for i in range(self.distance_matrix.shape[0])
        ]
        num_nodes = self.problem.customer_count
        all_nodes = range(num_nodes)
        intervals_global = dict()
        intervals_per_vehicle = dict()
        intervals_leaving_per_vehicle = dict()
        intervals_back_per_vehicle = dict()
        for v in range(self.problem.vehicle_count):
            intervals_per_vehicle[v] = {}
            st = self.problem.start_indexes[v]
            end = self.problem.end_indexes[v]
            for n in range(self.problem.customer_count):
                if n not in {st, end}:
                    intervals_per_vehicle[v][n] = self.cp_model.interval_var(
                        start=(0, None),
                        end=(end, None),
                        length=0,
                        optional=True,
                        name=f"visit_{n}_{v}",
                    )
            intervals_leaving_per_vehicle[v] = self.cp_model.interval_var(
                start=(0, None),
                end=(end, None),
                length=0,
                optional=True,
                name=f"leaving_{v}",
            )
            intervals_back_per_vehicle[v] = self.cp_model.interval_var(
                start=(0, None),
                end=(end, None),
                length=0,
                optional=True,
                name=f"back_{v}",
            )
            for n in intervals_per_vehicle[v]:
                self.cp_model.end_before_start(
                    intervals_leaving_per_vehicle[v], intervals_per_vehicle[v][n]
                )
                self.cp_model.end_before_start(
                    intervals_per_vehicle[v][n], intervals_back_per_vehicle[v]
                )
                self.cp_model.enforce(
                    self.cp_model.presence(intervals_leaving_per_vehicle[v])
                    >= self.cp_model.presence(intervals_per_vehicle[v][n])
                )
            self.cp_model.enforce(
                self.cp_model.presence(intervals_leaving_per_vehicle[v])
                == self.cp_model.presence(intervals_back_per_vehicle[v])
            )

        for n in range(self.problem.customer_count):
            if (
                n not in self.problem.start_indexes
                and n not in self.problem.end_indexes
            ):
                intervals_global[n] = self.cp_model.interval_var(
                    start=(0, None),
                    end=(0, None),
                    length=0,
                    optional=False,
                    name=f"global_{n}",
                )
                self.cp_model.alternative(
                    intervals_global[n],
                    [
                        intervals_per_vehicle[v][n]
                        for v in intervals_per_vehicle
                        if n in intervals_per_vehicle[v]
                    ],
                )
        for v in range(self.problem.vehicle_count):
            seq = self.cp_model.sequence_var(
                [intervals_leaving_per_vehicle[v]]
                + [intervals_per_vehicle[v][n] for n in intervals_per_vehicle[v]]
                + [intervals_back_per_vehicle[v]],
                types=[self.problem.start_indexes[v]]
                + [n for n in intervals_per_vehicle[v]]
                + [self.problem.end_indexes[v]],
            )
            self.cp_model.no_overlap(seq, self.distance_matrix)

            # Load :
            capacity = self.problem.vehicle_capacities[v]
            self.cp_model.enforce(
                self.cp_model.sum(
                    [
                        self.problem.customers[n].demand
                        * self.cp_model.presence(intervals_per_vehicle[v][n])
                        for n in intervals_per_vehicle[v]
                    ]
                )
                <= capacity
            )
        obj_expr = []
        obj_name = []
        obj_weight = []
        mode_optim = self.params_objective_function.sense_function

        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "nb_vehicles":
                nb_used_vehicle = self.cp_model.sum(
                    [
                        self.cp_model.presence(intervals_leaving_per_vehicle[v])
                        for v in intervals_leaving_per_vehicle
                    ]
                )
                obj_expr.append(nb_used_vehicle)
            if obj == "max_length":
                max_distance = self.cp_model.max(
                    [
                        self.cp_model.end(intervals_back_per_vehicle[v])
                        for v in intervals_back_per_vehicle
                    ]
                )
                obj_expr.append(max_distance)
            if obj == "length":
                sum_distance = self.cp_model.sum(
                    [
                        self.cp_model.end(intervals_back_per_vehicle[v])
                        for v in intervals_back_per_vehicle
                    ]
                )
                obj_expr.append(sum_distance)
            obj_name.append(obj)
            if mode_optim == ModeOptim.MAXIMIZATION:
                obj_weight.append(-weight)
            else:
                obj_weight.append(weight)
        self.variables["intervals"] = intervals_per_vehicle
        self.cp_model.minimize(
            self.cp_model.sum([w * expr for w, expr in zip(obj_weight, obj_expr)])
        )

    def retrieve_solution(self, result: cp.SolveResult) -> VrpSolution:
        paths = []
        for v in range(self.problem.vehicle_count):
            path = []
            for n in self.variables["intervals"][v]:
                if result.solution.is_present(self.variables["intervals"][v][n]):
                    st, end = result.solution.get_value(
                        self.variables["intervals"][v][n]
                    )
                    path.append((st, end, n))
            path = sorted(path, key=lambda x: x[0])
            paths.append([p[2] for p in path])
        sol = VrpSolution(
            problem=self.problem,
            list_start_index=self.problem.start_indexes,
            list_end_index=self.problem.end_indexes,
            list_paths=paths,
        )
        return sol
