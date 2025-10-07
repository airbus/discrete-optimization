#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback, LinearExpr

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)


class CpsatWTSolver(OrtoolsCpSatSolver):
    problem: WeightedTardinessProblem
    variables: dict

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        schedule = []
        for i in range(self.problem.num_jobs):
            st = cpsolvercb.value(self.variables["starts"][i])
            end = st + self.problem.processing_times[i]
            schedule.append((st, end))
        return WTSolution(problem=self.problem, schedule=schedule)

    def init_model(self, **args: Any) -> None:
        self.variables = {"starts": [], "intervals": [], "lateness": []}
        super().init_model(**args)
        max_time = sum(self.problem.processing_times)
        for i in range(self.problem.num_jobs):
            start = self.cp_model.NewIntVar(lb=0, ub=max_time, name=f"start_{i}")
            interval = self.cp_model.NewFixedSizeIntervalVar(
                start=start, size=self.problem.processing_times[i], name=f"interval_{i}"
            )
            self.variables["starts"].append(start)
            self.variables["intervals"].append(interval)
            lateness = self.cp_model.NewIntVar(
                lb=0,
                ub=max(max_time - self.problem.due_dates[i], 0),
                name=f"lateness_{i}",
            )
            self.cp_model.AddMaxEquality(
                lateness,
                [
                    start
                    + self.problem.processing_times[i]
                    - self.problem.due_dates[i],
                    0,
                ],
            )
            self.variables["lateness"].append(lateness)
        self.cp_model.AddNoOverlap(self.variables["intervals"])
        self.cp_model.Minimize(
            LinearExpr.weighted_sum(self.variables["lateness"], self.problem.weights)
        )
