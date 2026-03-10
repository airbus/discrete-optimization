#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import math
from typing import Any

import numpy as np

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
    optalcp_available = False
else:
    optalcp_available = True
from discrete_optimization.top.problem import TeamOrienteeringProblem, VrpSolution
from discrete_optimization.top.solvers import TopSolver


class OptalTopSolver(TopSolver, OptalCpSolver):
    def __init__(
        self,
        problem: TeamOrienteeringProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, scaling: float, **args: Any) -> None:
        distance = (np.ceil(scaling * self.distance)).astype(int).tolist()
        max_length = int(math.floor(scaling * self.problem.max_length_tours))
        self.cp_model = cp.Model()
        intervals_leaving_per_vehicle = {}
        intervals_back_per_vehicle = {}
        interval_per_customer = {}
        intervals_per_vehicle = {}
        for v in range(self.problem.vehicle_count):
            intervals_per_vehicle[v] = {}
            st = self.problem.start_indexes[v]
            end = self.problem.end_indexes[v]
            for n in range(self.problem.customer_count):
                if n not in {st, end}:
                    intervals_per_vehicle[v][n] = self.cp_model.interval_var(
                        start=(0, max_length),
                        end=(None, max_length),
                        length=0,
                        optional=True,
                        name=f"visit_{n}_{v}",
                    )
            intervals_leaving_per_vehicle[v] = self.cp_model.interval_var(
                start=(0, max_length),
                end=(None, max_length),
                length=0,
                optional=True,
                name=f"leaving_{v}",
            )
            intervals_back_per_vehicle[v] = self.cp_model.interval_var(
                start=(0, max_length),
                end=(None, max_length),
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
        for v in range(self.problem.vehicle_count):
            seq = self.cp_model.sequence_var(
                [intervals_leaving_per_vehicle[v]]
                + [intervals_per_vehicle[v][n] for n in intervals_per_vehicle[v]]
                + [intervals_back_per_vehicle[v]],
                types=[self.problem.start_indexes[v]]
                + [n for n in intervals_per_vehicle[v]]
                + [self.problem.end_indexes[v]],
            )
            self.cp_model.no_overlap(seq, distance)
        for n in range(self.problem.customer_count):
            if (
                n not in self.problem.start_indexes
                and n not in self.problem.end_indexes
            ):
                interval_per_customer[n] = self.cp_model.interval_var(
                    end=(0, max_length), length=0, optional=True, name=f"visit_{n}"
                )
                self.cp_model.alternative(
                    interval_per_customer[n],
                    [
                        intervals_per_vehicle[v][n]
                        for v in intervals_per_vehicle
                        if n in intervals_per_vehicle[v]
                    ],
                )
        reward = self.cp_model.sum(
            [
                self.cp_model.presence(interval_per_customer[n])
                * int(self.problem.customers[n].reward)
                for n in interval_per_customer
            ]
        )
        sum_end = self.cp_model.sum(
            [
                self.cp_model.end(intervals_back_per_vehicle[v])
                for v in intervals_back_per_vehicle
            ]
        )
        self.variables["objs"] = {"reward": -reward, "sum_length": sum_end}
        self.cp_model.minimize(self.variables["objs"]["reward"])
        self.variables["intervals"] = intervals_per_vehicle

    def retrieve_solution(self, result: "cp.SolveResult") -> VrpSolution:
        # Same as the vrp one.
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
